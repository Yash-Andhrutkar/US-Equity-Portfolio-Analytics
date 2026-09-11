"""
Independent validation gate for docs/data/dashboard_data.json.

Runs in CI after scripts/export_dashboard_data.py and before anything is
committed or deployed. It deliberately does NOT import the exporter: it
re-reads the committed analytics tables under output/tables/ and the
processed price data, and asserts that the JSON the dashboard will load
agrees with them. A silently wrong export therefore fails the build instead
of being published.

Checks performed:
  1. Schema      - required keys exist at every level, schema version matches
  2. Finiteness  - no NaN/Infinity anywhere; nulls only where allowed
  3. Agreement   - portfolio, benchmark, per-asset, risk-contribution, signal
                   and Monte Carlo figures match output/tables/*.csv
  4. Identities  - weights sum to 1, risk contribution percentages sum to 1
  5. Provenance  - latest market date matches the processed price data
  6. Coherence   - series lengths align with the price index
  7. Budget      - payload size stays within the documented transfer budget

Exit code 0 = valid, 1 = one or more failures (all failures are reported,
not just the first).

Usage:
    python scripts/validate_dashboard_data.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "docs" / "data" / "dashboard_data.json"
PRICES_PATH = PROJECT_ROOT / "data" / "processed" / "adjusted_close.csv"
TABLES_DIR = PROJECT_ROOT / "output" / "tables"

EXPECTED_SCHEMA_VERSION = 2

# The dashboard is a static GitHub Pages site; this is the raw JSON ceiling
# (it compresses to roughly a third of this over HTTP).
MAX_PAYLOAD_KB = 1600

# Tolerance for float agreement between the JSON (rounded to 6 dp) and the
# full-precision CSV values.
TOLERANCE = 1e-5

EXPECTED_TOP_LEVEL_KEYS = ["meta", "windows", "series", "signals", "monte_carlo"]

EXPECTED_META_KEYS = [
    "schema_version",
    "generated_at_utc",
    "first_market_date",
    "latest_market_date",
    "trading_days",
    "analysis_period_label",
    "git_sha",
    "pipeline_status",
    "tickers",
    "benchmark",
    "weights",
    "assumptions",
    "default_window",
    "windows",
]

EXPECTED_METRIC_KEYS = ["ann_return", "ann_volatility", "sharpe", "max_drawdown", "beta"]

EXPECTED_UNIVERSE_NODE_KEYS = [
    "ticker",
    "x",
    "z",
    "height_volatility",
    "radius_weight",
    "halo_risk_contribution_pct",
    "color_sharpe",
    "beta",
]

FULL_SAMPLE_WINDOW = "5Y"


class Validator:
    """Collects failures so a single run reports every problem found."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> bool:
        self.checks += 1
        if not condition:
            self.failures.append(message)
        return bool(condition)

    def close(self, actual, expected, message: str, tolerance: float = TOLERANCE) -> bool:
        self.checks += 1
        if actual is None or expected is None:
            self.failures.append(f"{message}: got {actual!r}, expected {expected!r}")
            return False
        if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
            self.failures.append(
                f"{message}: got {float(actual):.10f}, expected {float(expected):.10f}"
            )
            return False
        return True


def _walk_for_non_finite(node, path: str, found: list[str]) -> None:
    """Recursively assert no NaN/Infinity survived serialization."""
    if isinstance(node, dict):
        for key, value in node.items():
            _walk_for_non_finite(value, f"{path}.{key}", found)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            _walk_for_non_finite(value, f"{path}[{index}]", found)
    elif isinstance(node, float) and not math.isfinite(node):
        found.append(path)


def validate() -> Validator:
    v = Validator()

    if not DATA_PATH.exists():
        v.check(False, f"{DATA_PATH.relative_to(PROJECT_ROOT)} does not exist")
        return v

    raw = DATA_PATH.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        v.check(False, f"dashboard_data.json is not valid JSON: {exc}")
        return v

    # --- 1. schema ---------------------------------------------------------
    for key in EXPECTED_TOP_LEVEL_KEYS:
        v.check(key in payload, f"missing top-level key '{key}'")
    if v.failures:
        return v

    meta = payload["meta"]
    for key in EXPECTED_META_KEYS:
        v.check(key in meta, f"missing meta key '{key}'")

    v.check(
        meta.get("schema_version") == EXPECTED_SCHEMA_VERSION,
        f"schema_version is {meta.get('schema_version')}, expected {EXPECTED_SCHEMA_VERSION}",
    )
    v.check(meta.get("pipeline_status") == "ok", "pipeline_status is not 'ok'")

    tickers = meta.get("tickers", [])
    v.check(len(tickers) > 0, "meta.tickers is empty")

    windows = payload["windows"]
    v.check(FULL_SAMPLE_WINDOW in windows, f"missing '{FULL_SAMPLE_WINDOW}' window")
    v.check(
        meta.get("default_window") in windows,
        f"default_window '{meta.get('default_window')}' is not an exported window",
    )

    for key, window in windows.items():
        for metric_key in EXPECTED_METRIC_KEYS:
            v.check(metric_key in window.get("portfolio", {}), f"{key}.portfolio missing {metric_key}")
            v.check(metric_key in window.get("benchmark", {}), f"{key}.benchmark missing {metric_key}")
        for ticker in tickers:
            v.check(ticker in window.get("assets", {}), f"{key}.assets missing {ticker}")

        nodes = window.get("universe", {}).get("nodes", [])
        v.check(len(nodes) == len(tickers), f"{key}.universe has {len(nodes)} nodes, expected {len(tickers)}")
        for node in nodes:
            for node_key in EXPECTED_UNIVERSE_NODE_KEYS:
                v.check(node_key in node, f"{key}.universe node {node.get('ticker')} missing {node_key}")

    # --- 2. finiteness -----------------------------------------------------
    non_finite: list[str] = []
    _walk_for_non_finite(payload, "payload", non_finite)
    v.check(not non_finite, f"non-finite numbers at: {non_finite[:5]}")

    # Nulls are legitimate only in the moving-average lead-in period and in
    # an absent crossover date. Nowhere else.
    for window_key, window in windows.items():
        for metric_key, value in window.get("portfolio", {}).items():
            v.check(value is not None, f"{window_key}.portfolio.{metric_key} is null")
        for ticker, asset in window.get("assets", {}).items():
            for metric_key, value in asset.items():
                v.check(value is not None, f"{window_key}.assets.{ticker}.{metric_key} is null")

    # --- 3. agreement with the committed analytics tables ------------------
    prices = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)

    portfolio_summary = pd.read_csv(TABLES_DIR / "portfolio_summary.csv", index_col=0)
    comparison = pd.read_csv(TABLES_DIR / "portfolio_vs_sp500.csv", index_col=0)
    asset_metrics = pd.read_csv(TABLES_DIR / "asset_metrics.csv", index_col=0)
    risk_contribution = pd.read_csv(TABLES_DIR / "risk_contribution.csv", index_col=0)
    signal_summary = pd.read_csv(TABLES_DIR / "signal_summary.csv", index_col=0)
    backtest_summary = pd.read_csv(TABLES_DIR / "backtest_summary.csv", index_col=0)
    monte_carlo_summary = pd.read_csv(TABLES_DIR / "monte_carlo_summary.csv", index_col=0)

    csv_column_for = {
        "ann_return": "Annualized Return",
        "ann_volatility": "Annualized Volatility",
        "sharpe": "Sharpe Ratio",
        "max_drawdown": "Max Drawdown",
        "beta": "Beta",
    }

    full = windows[FULL_SAMPLE_WINDOW]

    for metric_key, column in csv_column_for.items():
        v.close(
            full["portfolio"][metric_key],
            portfolio_summary.loc["Portfolio", column],
            f"5Y portfolio {metric_key} disagrees with portfolio_summary.csv",
        )
        v.close(
            full["benchmark"][metric_key],
            comparison.loc["S&P 500", column],
            f"5Y benchmark {metric_key} disagrees with portfolio_vs_sp500.csv",
        )

    for ticker in tickers:
        asset = full["assets"][ticker]
        for metric_key, column in csv_column_for.items():
            v.close(
                asset[metric_key],
                asset_metrics.loc[ticker, column],
                f"5Y {ticker} {metric_key} disagrees with asset_metrics.csv",
            )
        v.close(
            asset["weight"],
            risk_contribution.loc[ticker, "Weight"],
            f"5Y {ticker} weight disagrees with risk_contribution.csv",
        )
        v.close(
            asset["risk_contribution_pct"],
            risk_contribution.loc[ticker, "Risk Contribution %"],
            f"5Y {ticker} risk_contribution_pct disagrees with risk_contribution.csv",
        )
        v.close(
            asset["marginal_risk_contribution"],
            risk_contribution.loc[ticker, "Marginal Risk Contribution"],
            f"5Y {ticker} marginal_risk_contribution disagrees with risk_contribution.csv",
        )

    # Signals
    signal_block = payload["signals"]["summary"]
    for ticker in tickers:
        v.check(ticker in signal_block, f"signals.summary missing {ticker}")
        if ticker not in signal_block:
            continue
        v.close(
            signal_block[ticker]["latest_price"],
            signal_summary.loc[ticker, "Latest Price"],
            f"{ticker} latest_price disagrees with signal_summary.csv",
            tolerance=1e-3,
        )
        v.check(
            signal_block[ticker]["trend"] == signal_summary.loc[ticker, "Trend"],
            f"{ticker} trend disagrees with signal_summary.csv",
        )
        v.check(
            signal_block[ticker]["last_crossover"] == signal_summary.loc[ticker, "Last Crossover"],
            f"{ticker} last_crossover disagrees with signal_summary.csv",
        )

    # Backtest
    backtest_block = payload["signals"]["backtest"]
    for ticker in tickers:
        v.close(
            backtest_block[ticker]["buy_hold"]["total_return"],
            backtest_summary.loc[ticker, "Buy & Hold Total Return"],
            f"{ticker} buy & hold total return disagrees with backtest_summary.csv",
        )
        v.close(
            backtest_block[ticker]["strategy"]["total_return"],
            backtest_summary.loc[ticker, "Strategy Total Return"],
            f"{ticker} strategy total return disagrees with backtest_summary.csv",
        )
        v.close(
            backtest_block[ticker]["strategy"]["max_drawdown"],
            backtest_summary.loc[ticker, "Strategy Max Drawdown"],
            f"{ticker} strategy max drawdown disagrees with backtest_summary.csv",
        )

    # Monte Carlo
    mc = payload["monte_carlo"]
    mc_csv = monte_carlo_summary["Value"]
    mc_pairs = {
        "mean_terminal_return": "Mean Terminal Return",
        "median_terminal_return": "Median Terminal Return",
        "p5_terminal_return": "5th Percentile Return",
        "p95_terminal_return": "95th Percentile Return",
        "probability_of_loss": "Probability of Loss",
        "expected_terminal_wealth": "Expected Terminal Wealth",
        "p5_terminal_wealth": "5th Percentile Wealth",
        "p95_terminal_wealth": "95th Percentile Wealth",
    }
    for json_key, csv_key in mc_pairs.items():
        v.close(
            mc["summary"][json_key],
            mc_csv.loc[csv_key],
            f"monte carlo {json_key} disagrees with monte_carlo_summary.csv",
        )

    # VaR is defined as the positive magnitude of the 5th-percentile loss, so
    # it must mirror the 5th-percentile terminal return when that is negative.
    p5 = mc["summary"]["p5_terminal_return"]
    v.close(mc["var"], max(0.0, -p5), "monte carlo VaR is inconsistent with the 5th percentile return")
    v.check(
        mc["cvar"] >= mc["var"] - TOLERANCE,
        f"CVaR ({mc['cvar']}) should not be below VaR ({mc['var']})",
    )

    # --- 4. identities -----------------------------------------------------
    v.close(sum(meta["weights"].values()), 1.0, "portfolio weights do not sum to 1")

    for window_key, window in windows.items():
        rc_total = sum(a["risk_contribution_pct"] for a in window["assets"].values())
        v.close(rc_total, 1.0, f"{window_key} risk contribution percentages do not sum to 1")

        weight_total = sum(a["weight"] for a in window["assets"].values())
        v.close(weight_total, 1.0, f"{window_key} asset weights do not sum to 1")

        for node in window["universe"]["nodes"]:
            v.check(
                node["height_volatility"] is not None and node["height_volatility"] >= 0,
                f"{window_key} {node['ticker']} volatility is negative",
            )
            v.check(
                0.0 <= node["radius_weight"] <= 1.0,
                f"{window_key} {node['ticker']} weight outside [0,1]",
            )

        corr = window["correlation"]
        for row in corr:
            v.close(corr[row][row], 1.0, f"{window_key} correlation diagonal for {row} is not 1")
            for column in corr[row]:
                v.close(
                    corr[row][column],
                    corr[column][row],
                    f"{window_key} correlation matrix is not symmetric at {row}/{column}",
                )
                v.check(
                    -1.0 - TOLERANCE <= corr[row][column] <= 1.0 + TOLERANCE,
                    f"{window_key} correlation {row}/{column} outside [-1,1]",
                )

    # --- 5. provenance -----------------------------------------------------
    v.check(
        meta["latest_market_date"] == prices.index.max().date().isoformat(),
        f"latest_market_date {meta['latest_market_date']} does not match the processed "
        f"price data ({prices.index.max().date().isoformat()})",
    )
    v.check(
        meta["first_market_date"] == prices.index.min().date().isoformat(),
        "first_market_date does not match the processed price data",
    )
    v.check(
        meta["trading_days"] == len(prices),
        f"trading_days {meta['trading_days']} does not match the price data ({len(prices)})",
    )

    # --- 6. coherence ------------------------------------------------------
    series = payload["series"]
    v.check(
        len(series["dates"]) == len(prices),
        f"series.dates has {len(series['dates'])} entries, price data has {len(prices)}",
    )
    v.check(
        len(series["return_dates"]) == len(prices) - 1,
        "series.return_dates should be one shorter than series.dates",
    )

    for name, values in series["cumulative"].items():
        v.check(
            len(values) == len(series["return_dates"]),
            f"series.cumulative.{name} length does not match return_dates",
        )
    for name, values in series["drawdown"].items():
        v.check(
            len(values) == len(series["return_dates"]),
            f"series.drawdown.{name} length does not match return_dates",
        )
    for ticker, block in series["moving_averages"].items():
        for field, values in block.items():
            v.check(
                len(values) == len(series["dates"]),
                f"series.moving_averages.{ticker}.{field} length does not match dates",
            )
    for ticker, block in series["backtest"].items():
        for field, values in block.items():
            v.check(
                len(values) == len(series["dates"]),
                f"series.backtest.{ticker}.{field} length does not match dates",
            )

    band_length = mc["config"]["days"]
    for name, values in mc["bands"].items():
        v.check(len(values) == band_length, f"monte_carlo.bands.{name} is not {band_length} long")
    for index, path in enumerate(mc["sample_paths"]):
        v.check(len(path) == band_length, f"monte_carlo.sample_paths[{index}] is not {band_length} long")

    histogram = mc["terminal_histogram"]
    v.check(
        len(histogram["bin_edges"]) == len(histogram["counts"]) + 1,
        "monte carlo histogram bin_edges should be one longer than counts",
    )
    v.close(
        sum(histogram["counts"]),
        mc["config"]["simulations"],
        "monte carlo histogram counts do not sum to the simulation count",
    )

    # Percentile bands must be ordered at every horizon.
    for day in range(band_length):
        ordered = (
            mc["bands"]["p5"][day]
            <= mc["bands"]["p25"][day]
            <= mc["bands"]["p50"][day]
            <= mc["bands"]["p75"][day]
            <= mc["bands"]["p95"][day]
        )
        if not ordered:
            v.check(False, f"monte carlo percentile bands are out of order at day {day + 1}")
            break
    v.checks += 1

    # Crossover events must be chronological and reference known tickers.
    events = payload["signals"]["events"]
    dates = [e["date"] for e in events]
    v.check(dates == sorted(dates), "crossover events are not in chronological order")
    v.check(
        all(e["ticker"] in tickers for e in events),
        "a crossover event references an unknown ticker",
    )
    v.check(
        all(e["signal"] in ("Golden Cross", "Death Cross") for e in events),
        "a crossover event has an unrecognized signal type",
    )

    # --- 7. budget ---------------------------------------------------------
    size_kb = len(raw.encode("utf-8")) / 1024
    v.check(
        size_kb <= MAX_PAYLOAD_KB,
        f"payload is {size_kb:.0f} KB, over the {MAX_PAYLOAD_KB} KB budget",
    )

    return v


def main() -> None:
    header = "Dashboard data validation"
    print(header)
    print("-" * len(header))

    validator = validate()

    if validator.failures:
        print(f"FAILED - {len(validator.failures)} problem(s) across {validator.checks} checks:\n")
        for failure in validator.failures:
            print(f"  ::error::{failure}")
        raise SystemExit(1)

    size_kb = len(DATA_PATH.read_text(encoding="utf-8").encode("utf-8")) / 1024
    print(f"PASSED - {validator.checks} checks")
    print(f"Payload size: {size_kb:.0f} KB (budget {MAX_PAYLOAD_KB} KB)")
    print("dashboard_data.json agrees with output/tables/*.csv and data/processed/adjusted_close.csv")


if __name__ == "__main__":
    main()
