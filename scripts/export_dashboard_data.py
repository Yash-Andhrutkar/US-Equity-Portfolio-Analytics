"""
Serializes the approved analytics layer into a single JSON payload consumed
by the interactive dashboard in docs/.

DESIGN RULE - READ BEFORE EDITING
=================================
This script contains NO financial formulas. Every metric, series, signal and
simulation figure is obtained by calling the already-approved modules
(src.metrics, src.portfolio, src.signals, src.simulation) exactly as main.py
does. The only arithmetic performed here is:

  * slicing a return series to a date window before handing it to those
    modules (windowing, not a formula),
  * rounding for payload size,
  * classical multidimensional scaling of the correlation DISTANCE matrix
    produced by metrics.calculate_correlation_matrix(), used purely as a
    2-D LAYOUT for the diversification map (documented below), and
  * histogram binning / quantile extraction of the simulated terminal
    returns produced by src.simulation, for chart rendering.

The dashboard must never compute a financial metric in JavaScript. If a
number needs to appear on screen, it is added here.

TIME WINDOWS
============
The dashboard exposes six ranges (1M, 6M, YTD, 1Y, 3Y, 5Y). For each range
the metric set is recomputed here by re-calling the same approved functions
on the windowed return series, so the browser only ever selects a
precomputed set.

Annualizing a sub-annual window extrapolates a short sample to a yearly
figure, which is inherently noisy. Windows shorter than one trading year
are therefore flagged `short_window: true` so the dashboard can display a
caveat rather than presenting, say, a 21-day annualized return as if it
carried the same weight as the 5-year figure.

MONTE CARLO SCOPE
=================
The Monte Carlo block is parameterized from the FULL sample only, matching
main.py and output/tables/monte_carlo_summary.csv. It is not recomputed per
window: doing so would produce simulation figures that no committed
analytics table corroborates, and validate_dashboard_data.py cross-checks
this block against that table. See src/simulation.py for the methodology
and its limitations - it is historical scenario analysis, not a forecast.

CORRELATION LAYOUT (diversification map)
========================================
The `universe` block carries a two-dimensional layout of the holdings plus
the per-holding quantities the dashboard's diversification map encodes:

  x / z               classical MDS of correlation distance
                      d_ij = sqrt(2 * (1 - rho_ij)), so proximity in the
                      layout means genuine historical co-movement
  radius_weight       capital weight (the map scales mark area by its root)
  height_volatility   annualized volatility
  halo_...            percentage contribution to portfolio risk
  color_sharpe        Sharpe ratio
  edges               pairwise correlation

The MDS orientation is pinned to a deterministic convention and each window
is Procrustes-aligned to the full-sample layout, so the arrangement does not
rotate or mirror between weekly refreshes. Field names are retained for
schema stability; the dashboard presents this as a correlation-distance
diversification map, a standard way to read correlation structure.

Usage:
    python scripts/export_dashboard_data.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    BENCHMARK,
    MA_LONG,
    MA_SHORT,
    MONTE_CARLO_DAYS,
    MONTE_CARLO_SIMULATIONS,
    PORTFOLIO_WEIGHTS,
    RANDOM_SEED,
    RISK_FREE_RATE,
    TICKERS,
    TRADING_DAYS,
)
from src import data_loader, metrics, portfolio, signals, simulation  # noqa: E402

OUTPUT_PATH = PROJECT_ROOT / "docs" / "data" / "dashboard_data.json"

SCHEMA_VERSION = 2

BENCHMARK_LABEL = "S&P 500"

# Sample paths shipped for the Monte Carlo fan animation. Selected at evenly
# spaced column indices (deterministic, not random) from the simulated paths.
MC_SAMPLE_PATHS = 60
MC_HISTOGRAM_BINS = 60
MC_BAND_QUANTILES = {"p5": 0.05, "p25": 0.25, "p50": 0.50, "p75": 0.75, "p95": 0.95}

# Every pair is shipped as an edge; the renderer scales opacity and width by
# |correlation| so weak links stay visible but recede. Pairs at or above this
# absolute correlation are additionally emphasized (brighter, labelled) - this
# is a styling threshold only, it never removes data.
UNIVERSE_EDGE_EMPHASIS_ABS_CORR = 0.30

# Rounding: prices in dollars need 4 dp; returns and ratios are unitless and
# 6 dp is far finer than any displayed precision.
PRICE_DP = 4
RATIO_DP = 6

WINDOW_DEFINITIONS = [
    {"key": "1M", "label": "1 Month", "trading_days": 21},
    {"key": "6M", "label": "6 Months", "trading_days": 126},
    {"key": "YTD", "label": "Year to Date", "trading_days": None},
    {"key": "1Y", "label": "1 Year", "trading_days": 252},
    {"key": "3Y", "label": "3 Years", "trading_days": 756},
    {"key": "5Y", "label": "Full Sample", "trading_days": None},
]

DEFAULT_WINDOW = "5Y"

METRIC_COLUMNS = [
    "Annualized Return",
    "Annualized Volatility",
    "Sharpe Ratio",
    "Max Drawdown",
    "Beta",
]

METRIC_KEYS = {
    "Annualized Return": "ann_return",
    "Annualized Volatility": "ann_volatility",
    "Sharpe Ratio": "sharpe",
    "Max Drawdown": "max_drawdown",
    "Beta": "beta",
}


# ---------------------------------------------------------------------------
# JSON-safety helpers
# ---------------------------------------------------------------------------


def _num(value, dp: int = RATIO_DP):
    """Round a scalar to `dp` places, mapping non-finite values to None."""
    if value is None:
        return None
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    return round(as_float, dp)


def _series(values, dp: int = RATIO_DP) -> list:
    """Round an iterable of scalars, preserving NaN as null."""
    return [_num(v, dp) for v in values]


def _date(value):
    """Render a timestamp as an ISO date string, or None when missing."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date().isoformat()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Resolve the commit this export was produced from."""
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def resolve_windows(returns: pd.DataFrame) -> list[dict]:
    """Resolve each configured range to a concrete slice of the return index.

    A window is dropped when the available history cannot support it (fewer
    observations than the window asks for, or fewer than the 2 observations
    the analytics modules require). The full sample is always retained.
    """
    total = len(returns)
    latest = returns.index.max()
    resolved = []

    for definition in WINDOW_DEFINITIONS:
        key = definition["key"]

        if key == "5Y":
            start_index = 0
        elif key == "YTD":
            year_start = pd.Timestamp(year=latest.year, month=1, day=1)
            positions = np.flatnonzero(returns.index >= year_start)
            if positions.size == 0:
                continue
            start_index = int(positions[0])
        else:
            requested = definition["trading_days"]
            if total < requested:
                continue
            start_index = total - requested

        observations = total - start_index
        if observations < 2:
            continue

        resolved.append(
            {
                "key": key,
                "label": definition["label"],
                "start_index": start_index,
                "observations": observations,
                "start_date": _date(returns.index[start_index]),
                "end_date": _date(returns.index[-1]),
                # Annualizing fewer than TRADING_DAYS observations extrapolates
                # a partial year; the dashboard shows a caveat for these.
                "short_window": observations < TRADING_DAYS,
            }
        )

    return resolved


# ---------------------------------------------------------------------------
# Metric blocks (all values delegated to the approved modules)
# ---------------------------------------------------------------------------


def _metrics_row_to_dict(row: pd.Series) -> dict:
    return {METRIC_KEYS[column]: _num(row[column]) for column in METRIC_COLUMNS}


def build_window_metrics(returns: pd.DataFrame, window: dict) -> dict:
    """Compute the full metric set for one window via the approved modules."""
    window_returns = returns.iloc[window["start_index"] :]

    portfolio_summary = portfolio.create_portfolio_summary(window_returns, PORTFOLIO_WEIGHTS)
    comparison = portfolio.create_portfolio_comparison(window_returns, PORTFOLIO_WEIGHTS)
    asset_metrics = metrics.create_metrics_summary(window_returns)
    risk_contribution = portfolio.calculate_risk_contribution(window_returns, PORTFOLIO_WEIGHTS)
    correlation = metrics.calculate_correlation_matrix(window_returns)

    portfolio_returns = portfolio.calculate_portfolio_returns(window_returns, PORTFOLIO_WEIGHTS)
    cumulative = portfolio.calculate_portfolio_cumulative_return(portfolio_returns)
    benchmark_cumulative = metrics.calculate_cumulative_returns(window_returns[[BENCHMARK]])[BENCHMARK]

    return {
        "key": window["key"],
        "label": window["label"],
        "start_date": window["start_date"],
        "end_date": window["end_date"],
        "observations": window["observations"],
        "short_window": window["short_window"],
        "start_index": window["start_index"],
        "portfolio": _metrics_row_to_dict(portfolio_summary.loc["Portfolio"]),
        "benchmark": _metrics_row_to_dict(comparison.loc[BENCHMARK_LABEL]),
        "total_return": {
            "portfolio": _num(cumulative.iloc[-1]),
            "benchmark": _num(benchmark_cumulative.iloc[-1]),
        },
        "assets": {
            ticker: {
                **_metrics_row_to_dict(asset_metrics.loc[ticker]),
                "weight": _num(risk_contribution.loc[ticker, "Weight"]),
                "risk_contribution_pct": _num(risk_contribution.loc[ticker, "Risk Contribution %"]),
                "risk_contribution": _num(risk_contribution.loc[ticker, "Risk Contribution"]),
                "marginal_risk_contribution": _num(
                    risk_contribution.loc[ticker, "Marginal Risk Contribution"]
                ),
            }
            for ticker in TICKERS
        },
        "correlation": {
            row: {column: _num(correlation.loc[row, column]) for column in correlation.columns}
            for row in correlation.index
        },
        "universe": build_universe(asset_metrics, risk_contribution, correlation),
    }


# ---------------------------------------------------------------------------
# Correlation layout for the diversification map
# ---------------------------------------------------------------------------


def correlation_distance_matrix(correlation: pd.DataFrame) -> np.ndarray:
    """Convert a correlation matrix into the standard correlation distance.

    d_ij = sqrt(2 * (1 - rho_ij)), the usual metric embedding of correlation:
    identical series are distance 0 apart, uncorrelated series sqrt(2), and
    perfectly inverse series 2.
    """
    rho = correlation.loc[TICKERS, TICKERS].to_numpy(dtype=float)
    rho = np.clip(rho, -1.0, 1.0)
    distance = np.sqrt(np.maximum(2.0 * (1.0 - rho), 0.0))
    np.fill_diagonal(distance, 0.0)
    return distance


def classical_mds(distance: np.ndarray) -> np.ndarray:
    """Classical (Torgerson) MDS to two dimensions.

    Pure linear algebra on the supplied distance matrix - this is a layout
    projection, not a financial calculation.
    """
    n = distance.shape[0]
    squared = distance**2
    centering = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centering @ squared @ centering

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    top_values = np.maximum(eigenvalues[order], 0.0)
    coordinates = eigenvectors[:, order] * np.sqrt(top_values)
    return coordinates


def _canonical_orientation(coordinates: np.ndarray) -> np.ndarray:
    """Pin an MDS layout to a deterministic rotation and reflection.

    eigh returns eigenvectors with arbitrary sign, so an unpinned layout can
    rotate or mirror between refreshes even when the underlying correlations
    barely move. The convention: translate the configuration to its centroid,
    rotate so the first ticker sits on the positive x-axis, then reflect so
    the second ticker sits in the upper half-plane.
    """
    centered = coordinates - coordinates.mean(axis=0)

    anchor = centered[0]
    norm = np.hypot(anchor[0], anchor[1])
    if norm > 1e-12:
        cos_a, sin_a = anchor[0] / norm, anchor[1] / norm
        rotation = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
        centered = centered @ rotation.T

    if centered.shape[0] > 1 and centered[1, 1] < 0:
        centered[:, 1] *= -1.0

    return centered


def _procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate/reflect `source` onto `target` without rescaling it.

    Keeps the per-window layouts registered against the full-sample layout so
    changing the time range nudges nodes rather than reshuffling the field.
    """
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = u @ vt
    return source_centered @ rotation


def _normalize_layout(coordinates: np.ndarray) -> np.ndarray:
    """Scale a layout into a unit-radius disc for predictable rendering."""
    radii = np.linalg.norm(coordinates, axis=1)
    largest = float(radii.max()) if radii.size else 0.0
    if largest <= 1e-12:
        return coordinates
    return coordinates / largest


def build_universe(
    asset_metrics: pd.DataFrame,
    risk_contribution: pd.DataFrame,
    correlation: pd.DataFrame,
    reference_layout: np.ndarray | None = None,
) -> dict:
    """Build the node/edge layout behind the diversification map.

    Positions come from MDS of the correlation distance matrix; every other
    property is read straight off the analytics tables. See the module
    docstring for the full mapping.
    """
    distance = correlation_distance_matrix(correlation)
    layout = _canonical_orientation(classical_mds(distance))
    if reference_layout is not None:
        layout = _procrustes_align(layout, reference_layout)
    normalized = _normalize_layout(layout)

    weights = np.array([float(risk_contribution.loc[t, "Weight"]) for t in TICKERS])
    centroid = weights @ normalized

    nodes = []
    for index, ticker in enumerate(TICKERS):
        nodes.append(
            {
                "ticker": ticker,
                # Layout position (unitless, unit disc) from correlation MDS
                "x": _num(normalized[index, 0]),
                "z": _num(normalized[index, 1]),
                # Every property below is a real quantity from the analytics layer
                "height_volatility": _num(asset_metrics.loc[ticker, "Annualized Volatility"]),
                "radius_weight": _num(risk_contribution.loc[ticker, "Weight"]),
                "halo_risk_contribution_pct": _num(risk_contribution.loc[ticker, "Risk Contribution %"]),
                "color_sharpe": _num(asset_metrics.loc[ticker, "Sharpe Ratio"]),
                "beta": _num(asset_metrics.loc[ticker, "Beta"]),
            }
        )

    edges = []
    for i, left in enumerate(TICKERS):
        for j, right in enumerate(TICKERS):
            if j <= i:
                continue
            rho = float(correlation.loc[left, right])
            edges.append(
                {
                    "source": left,
                    "target": right,
                    "correlation": _num(rho),
                    "emphasis": bool(abs(rho) >= UNIVERSE_EDGE_EMPHASIS_ABS_CORR),
                }
            )

    return {
        "nodes": nodes,
        "edges": edges,
        "centroid": {"x": _num(centroid[0]), "z": _num(centroid[1])},
        "layout_raw": [[_num(v) for v in row] for row in layout],
        "edge_emphasis_abs_correlation": UNIVERSE_EDGE_EMPHASIS_ABS_CORR,
    }


# ---------------------------------------------------------------------------
# Series blocks
# ---------------------------------------------------------------------------


def build_series(prices: pd.DataFrame, returns: pd.DataFrame) -> dict:
    """Build the full-sample chart series.

    Charts pan across the full sample rather than being recomputed per
    window. The dashboard rebases a cumulative series to a window start with
    the identity (1 + c_t) / (1 + c_start) - 1, which is exact and is
    asserted against the windowed Python computation in the test suite.
    """
    cumulative = metrics.calculate_cumulative_returns(returns)
    portfolio_returns = portfolio.calculate_portfolio_returns(returns, PORTFOLIO_WEIGHTS)
    portfolio_cumulative = portfolio.calculate_portfolio_cumulative_return(portfolio_returns)
    portfolio_drawdown = portfolio.calculate_portfolio_drawdown(portfolio_returns)
    benchmark_drawdown = metrics.calculate_drawdown(returns[[BENCHMARK]])[BENCHMARK]

    moving_averages = signals.calculate_moving_averages(prices, MA_SHORT, MA_LONG)
    trend = signals.calculate_trend_state(moving_averages)

    cumulative_block = {ticker: _series(cumulative[ticker]) for ticker in TICKERS}
    cumulative_block["PORTFOLIO"] = _series(portfolio_cumulative)
    cumulative_block["BENCHMARK"] = _series(cumulative[BENCHMARK])

    moving_average_block = {}
    for ticker in TICKERS:
        moving_average_block[ticker] = {
            "price": _series(moving_averages[(ticker, "Price")], PRICE_DP),
            "ma_short": _series(moving_averages[(ticker, "MA_Short")], PRICE_DP),
            "ma_long": _series(moving_averages[(ticker, "MA_Long")], PRICE_DP),
            "trend": [str(state) for state in trend[ticker]],
        }

    backtest_block = {}
    for ticker in TICKERS:
        backtest = signals.backtest_moving_average_signal(prices, ticker, MA_SHORT, MA_LONG)
        backtest_block[ticker] = {
            "buy_hold_wealth": _series(backtest["Buy & Hold Wealth"]),
            "strategy_wealth": _series(backtest["Strategy Wealth"]),
            "position": [int(p) for p in backtest["Position"]],
        }

    return {
        "dates": [d.date().isoformat() for d in prices.index],
        # Return series start one observation after the price series.
        "return_dates": [d.date().isoformat() for d in returns.index],
        "cumulative": cumulative_block,
        "drawdown": {
            "PORTFOLIO": _series(portfolio_drawdown),
            "BENCHMARK": _series(benchmark_drawdown),
        },
        "moving_averages": moving_average_block,
        "backtest": backtest_block,
    }


def build_signals_block(prices: pd.DataFrame) -> dict:
    """Build the signal summary, crossover events and backtest comparison."""
    signal_summary = signals.create_signal_summary(prices, MA_SHORT, MA_LONG)
    crossovers = signals.detect_crossovers(
        signals.calculate_moving_averages(prices, MA_SHORT, MA_LONG)
    )
    backtest_summary = signals.create_backtest_summary(prices, MA_SHORT, MA_LONG)

    short_label = f"MA{MA_SHORT}"
    long_label = f"MA{MA_LONG}"

    summary_block = {}
    for ticker in TICKERS:
        row = signal_summary.loc[ticker]
        summary_block[ticker] = {
            "latest_price": _num(row["Latest Price"], PRICE_DP),
            "ma_short": _num(row[short_label], PRICE_DP),
            "ma_long": _num(row[long_label], PRICE_DP),
            "trend": str(row["Trend"]),
            "last_crossover": str(row["Last Crossover"]),
            "last_crossover_date": _date(row["Last Crossover Date"]),
        }

    events = [
        {
            "date": _date(event["Date"]),
            "ticker": str(event["Ticker"]),
            "signal": str(event["Signal"]),
            "price": _num(event["Price"], PRICE_DP),
            "ma_short": _num(event["Short MA"], PRICE_DP),
            "ma_long": _num(event["Long MA"], PRICE_DP),
        }
        for _, event in crossovers.iterrows()
    ]

    backtest_block = {}
    for ticker in TICKERS:
        row = backtest_summary.loc[ticker]
        backtest_block[ticker] = {
            "buy_hold": {
                "total_return": _num(row["Buy & Hold Total Return"]),
                "ann_return": _num(row["Buy & Hold Annualized Return"]),
                "ann_volatility": _num(row["Buy & Hold Annualized Volatility"]),
                "max_drawdown": _num(row["Buy & Hold Max Drawdown"]),
            },
            "strategy": {
                "total_return": _num(row["Strategy Total Return"]),
                "ann_return": _num(row["Strategy Annualized Return"]),
                "ann_volatility": _num(row["Strategy Annualized Volatility"]),
                "max_drawdown": _num(row["Strategy Max Drawdown"]),
            },
        }

    return {"summary": summary_block, "events": events, "backtest": backtest_block}


def build_monte_carlo_block(returns: pd.DataFrame) -> dict:
    """Run the approved Monte Carlo and reduce it to a renderable payload.

    The raw simulation is 10,000 x 252 values, far too large to ship. It is
    reduced to daily percentile bands, a deterministic sample of individual
    paths for the drawing animation, and a histogram of terminal returns.
    All figures originate from src.simulation.
    """
    result = simulation.run_monte_carlo_simulation(
        returns,
        simulations=MONTE_CARLO_SIMULATIONS,
        days=MONTE_CARLO_DAYS,
        seed=RANDOM_SEED,
        weights=PORTFOLIO_WEIGHTS,
    )

    wealth_paths = result["wealth_paths"]
    terminal_returns = result["terminal_returns"]

    bands = {
        name: _series(wealth_paths.quantile(q, axis=1))
        for name, q in MC_BAND_QUANTILES.items()
    }

    total_paths = wealth_paths.shape[1]
    sample_count = min(MC_SAMPLE_PATHS, total_paths)
    # Evenly spaced column indices - deterministic, so the animated sample
    # is identical for a given seed rather than a fresh random draw.
    sample_indices = np.linspace(0, total_paths - 1, sample_count, dtype=int)
    sample_paths = [_series(wealth_paths.iloc[:, int(i)]) for i in sample_indices]

    counts, edges = np.histogram(terminal_returns.to_numpy(), bins=MC_HISTOGRAM_BINS)

    summary = result["summary"]
    downside = result["downside_probabilities"]

    return {
        "config": {
            "simulations": MONTE_CARLO_SIMULATIONS,
            "days": MONTE_CARLO_DAYS,
            "seed": RANDOM_SEED,
            "distribution": "normal",
            "parameterized_from": "full sample historical daily portfolio mean and volatility",
        },
        "bands": bands,
        "sample_paths": sample_paths,
        "terminal_histogram": {
            "bin_edges": _series(edges),
            "counts": [int(c) for c in counts],
        },
        "summary": {
            "mean_terminal_return": _num(summary["Mean Terminal Return"]),
            "median_terminal_return": _num(summary["Median Terminal Return"]),
            "p5_terminal_return": _num(summary["5th Percentile Return"]),
            "p95_terminal_return": _num(summary["95th Percentile Return"]),
            "probability_of_loss": _num(summary["Probability of Loss"]),
            "expected_terminal_wealth": _num(summary["Expected Terminal Wealth"]),
            "p5_terminal_wealth": _num(summary["5th Percentile Wealth"]),
            "p95_terminal_wealth": _num(summary["95th Percentile Wealth"]),
        },
        "downside_probabilities": {
            "loss": _num(downside["Probability of Loss"]),
            "below_minus_10": _num(downside["Probability Return < -10%"]),
            "below_minus_20": _num(downside["Probability Return < -20%"]),
            "below_minus_30": _num(downside["Probability Return < -30%"]),
        },
        "var": _num(result["var"]),
        "cvar": _num(result["cvar"]),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_payload() -> dict:
    """Assemble the complete dashboard payload."""
    prices = data_loader.load_price_data()
    returns = metrics.calculate_daily_returns(prices)

    windows = resolve_windows(returns)
    if not windows:
        raise ValueError("No time window could be resolved from the available history.")

    window_blocks = {}
    reference_layout = None

    # Compute the full sample first so every shorter window can be
    # Procrustes-aligned to it.
    ordered_keys = [DEFAULT_WINDOW] + [w["key"] for w in windows if w["key"] != DEFAULT_WINDOW]
    for key in ordered_keys:
        window = next(w for w in windows if w["key"] == key)
        block = build_window_metrics(returns, window)

        if reference_layout is None:
            reference_layout = np.array(block["universe"]["layout_raw"], dtype=float)
        else:
            window_returns = returns.iloc[window["start_index"] :]
            block["universe"] = build_universe(
                metrics.create_metrics_summary(window_returns),
                portfolio.calculate_risk_contribution(window_returns, PORTFOLIO_WEIGHTS),
                metrics.calculate_correlation_matrix(window_returns),
                reference_layout=reference_layout,
            )

        window_blocks[key] = block

    latest_date = prices.index.max()
    first_date = prices.index.min()

    payload = {
        "meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "first_market_date": _date(first_date),
            "latest_market_date": _date(latest_date),
            "trading_days": int(len(prices)),
            "analysis_period_label": f"{_date(first_date)} to {_date(latest_date)}",
            "git_sha": _git_sha(),
            "pipeline_status": "ok",
            "tickers": list(TICKERS),
            "benchmark": {"symbol": BENCHMARK, "label": BENCHMARK_LABEL},
            "weights": {ticker: _num(weight) for ticker, weight in PORTFOLIO_WEIGHTS.items()},
            "assumptions": {
                "risk_free_rate": _num(RISK_FREE_RATE),
                "trading_days_per_year": int(TRADING_DAYS),
                "ma_short": int(MA_SHORT),
                "ma_long": int(MA_LONG),
                "transaction_costs_modelled": False,
            },
            "default_window": DEFAULT_WINDOW,
            "windows": [
                {
                    "key": w["key"],
                    "label": w["label"],
                    "observations": w["observations"],
                    "start_date": w["start_date"],
                    "end_date": w["end_date"],
                    "short_window": w["short_window"],
                }
                for w in windows
            ],
        },
        "windows": window_blocks,
        "series": build_series(prices, returns),
        "signals": build_signals_block(prices),
        "monte_carlo": build_monte_carlo_block(returns),
    }

    return payload


def write_payload(payload: dict, output_path: Path = OUTPUT_PATH) -> Path:
    """Write the payload as compact JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    """Export the dashboard payload and print a short provenance summary."""
    header = "Dashboard data export"
    print(header)
    print("-" * len(header))

    payload = build_payload()
    path = write_payload(payload)

    size_kb = path.stat().st_size / 1024
    meta = payload["meta"]

    print(f"Latest market date:  {meta['latest_market_date']}")
    print(f"Analysis period:     {meta['analysis_period_label']}")
    print(f"Trading days:        {meta['trading_days']}")
    print(f"Windows exported:    {', '.join(payload['windows'].keys())}")
    print(f"Crossover events:    {len(payload['signals']['events'])}")
    print(f"Commit:              {meta['git_sha']}")
    print(f"Written:             {path.relative_to(PROJECT_ROOT)} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
