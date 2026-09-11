"""
Tests for scripts/export_dashboard_data.py.

These tests guard the contract between the Python analytics layer and the
dashboard front end. The central concern is that the exported JSON is not
merely well-formed but numerically identical to what the approved modules
(src.metrics, src.portfolio, src.signals, src.simulation) produce, so the
dashboard can never drift away from the analytics.

The full payload is built once per session because the Monte Carlo run and
the six per-window metric sets are the expensive part.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BENCHMARK,
    MA_LONG,
    MA_SHORT,
    MONTE_CARLO_DAYS,
    MONTE_CARLO_SIMULATIONS,
    PORTFOLIO_WEIGHTS,
    RISK_FREE_RATE,
    TICKERS,
    TRADING_DAYS,
)
from scripts import export_dashboard_data as exporter
from src import data_loader, metrics, portfolio, signals

TOLERANCE = 1e-5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prices():
    return data_loader.load_price_data()


@pytest.fixture(scope="module")
def returns(prices):
    return metrics.calculate_daily_returns(prices)


@pytest.fixture(scope="module")
def payload():
    return exporter.build_payload()


# ---------------------------------------------------------------------------
# Scalar / series helpers
# ---------------------------------------------------------------------------


def test_num_rounds_and_maps_non_finite():
    assert exporter._num(0.1234567) == 0.123457
    assert exporter._num(1.5, 1) == 1.5
    assert exporter._num(float("nan")) is None
    assert exporter._num(float("inf")) is None
    assert exporter._num(None) is None


def test_series_preserves_length_and_nulls():
    values = [1.0, float("nan"), 3.0]
    assert exporter._series(values) == [1.0, None, 3.0]


def test_date_handles_missing_values():
    assert exporter._date(pd.Timestamp("2026-01-02")) == "2026-01-02"
    assert exporter._date(pd.NaT) is None
    assert exporter._date(None) is None


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def test_resolve_windows_includes_full_sample_and_flags_short_windows(returns):
    windows = exporter.resolve_windows(returns)
    keys = [w["key"] for w in windows]

    assert "5Y" in keys
    assert len(keys) == len(set(keys)), "duplicate window keys"

    by_key = {w["key"]: w for w in windows}

    full = by_key["5Y"]
    assert full["start_index"] == 0
    assert full["observations"] == len(returns)
    assert full["short_window"] is False

    # Sub-annual windows must be flagged so the dashboard can caveat the
    # annualized figures they produce.
    for key in ("1M", "6M"):
        if key in by_key:
            assert by_key[key]["short_window"] is True
    if "1Y" in by_key:
        assert by_key["1Y"]["short_window"] is False


def test_resolve_windows_respects_requested_trading_days(returns):
    by_key = {w["key"]: w for w in exporter.resolve_windows(returns)}
    expected = {"1M": 21, "6M": 126, "1Y": 252, "3Y": 756}
    for key, trading_days in expected.items():
        if key in by_key:
            assert by_key[key]["observations"] == trading_days


def test_resolve_windows_ytd_starts_in_the_latest_calendar_year(returns):
    by_key = {w["key"]: w for w in exporter.resolve_windows(returns)}
    if "YTD" not in by_key:
        pytest.skip("YTD window not resolvable for this sample")

    ytd = by_key["YTD"]
    latest_year = returns.index.max().year
    assert pd.Timestamp(ytd["start_date"]).year == latest_year


def test_resolve_windows_drops_windows_longer_than_the_sample(returns):
    short_sample = returns.iloc[-30:]
    keys = [w["key"] for w in exporter.resolve_windows(short_sample)]

    assert "5Y" in keys
    assert "1Y" not in keys
    assert "3Y" not in keys


def test_resolve_windows_requires_two_observations(returns):
    keys = [w["key"] for w in exporter.resolve_windows(returns.iloc[-1:])]
    assert keys == []


# ---------------------------------------------------------------------------
# Correlation distance and MDS layout
# ---------------------------------------------------------------------------


def test_correlation_distance_matrix_matches_definition(returns):
    correlation = metrics.calculate_correlation_matrix(returns)
    distance = exporter.correlation_distance_matrix(correlation)

    assert distance.shape == (len(TICKERS), len(TICKERS))
    assert np.allclose(np.diag(distance), 0.0)
    assert np.allclose(distance, distance.T)

    for i, left in enumerate(TICKERS):
        for j, right in enumerate(TICKERS):
            rho = correlation.loc[left, right]
            assert distance[i, j] == pytest.approx(math.sqrt(2 * (1 - rho)), abs=1e-12)


def test_classical_mds_recovers_a_known_configuration():
    # Four points on a unit square; MDS should reproduce the pairwise
    # distances up to rotation/reflection.
    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    distance = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    embedded = exporter.classical_mds(distance)
    embedded_distance = np.linalg.norm(
        embedded[:, None, :] - embedded[None, :, :], axis=-1
    )

    assert np.allclose(embedded_distance, distance, atol=1e-9)


def test_canonical_orientation_is_deterministic_under_rotation():
    points = np.array([[1.0, 0.2], [0.1, 0.9], [-0.8, -0.3], [0.4, -0.7]])

    angle = 0.7
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rotated = points @ rotation.T
    reflected = points * np.array([1.0, -1.0])

    base = exporter._canonical_orientation(points)
    from_rotated = exporter._canonical_orientation(rotated)
    from_reflected = exporter._canonical_orientation(reflected)

    assert np.allclose(base, from_rotated, atol=1e-9)
    assert np.allclose(base, from_reflected, atol=1e-9)


def test_canonical_orientation_pins_the_anchor_to_the_positive_x_axis():
    points = np.array([[0.3, 0.9], [-0.5, 0.4], [0.2, -0.8]])
    oriented = exporter._canonical_orientation(points)

    assert oriented[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert oriented[0, 0] > 0
    assert oriented[1, 1] >= 0


def test_procrustes_alignment_preserves_pairwise_distances():
    source = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    target = source @ np.array([[0.0, -1.0], [1.0, 0.0]])

    aligned = exporter._procrustes_align(source, target)

    def pairwise(points):
        return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)

    assert np.allclose(pairwise(aligned), pairwise(source), atol=1e-9)
    assert np.allclose(aligned, target - target.mean(axis=0), atol=1e-9)


def test_universe_layout_is_reproducible(returns):
    asset_metrics = metrics.create_metrics_summary(returns)
    risk_contribution = portfolio.calculate_risk_contribution(returns, PORTFOLIO_WEIGHTS)
    correlation = metrics.calculate_correlation_matrix(returns)

    first = exporter.build_universe(asset_metrics, risk_contribution, correlation)
    second = exporter.build_universe(asset_metrics, risk_contribution, correlation)

    assert first == second, "the spatial layout must be deterministic across runs"


def test_universe_layout_is_stable_across_repeated_exports(payload):
    # Re-exporting from identical inputs must not move the field, otherwise
    # the weekly refresh would visually reshuffle for no analytical reason.
    rebuilt = exporter.build_payload()
    assert rebuilt["windows"]["5Y"]["universe"] == payload["windows"]["5Y"]["universe"]


def test_universe_properties_match_the_analytics_layer(payload, returns):
    asset_metrics = metrics.create_metrics_summary(returns)
    risk_contribution = portfolio.calculate_risk_contribution(returns, PORTFOLIO_WEIGHTS)

    nodes = {n["ticker"]: n for n in payload["windows"]["5Y"]["universe"]["nodes"]}
    assert set(nodes) == set(TICKERS)

    for ticker, node in nodes.items():
        assert node["height_volatility"] == pytest.approx(
            asset_metrics.loc[ticker, "Annualized Volatility"], abs=TOLERANCE
        )
        assert node["color_sharpe"] == pytest.approx(
            asset_metrics.loc[ticker, "Sharpe Ratio"], abs=TOLERANCE
        )
        assert node["beta"] == pytest.approx(asset_metrics.loc[ticker, "Beta"], abs=TOLERANCE)
        assert node["radius_weight"] == pytest.approx(PORTFOLIO_WEIGHTS[ticker], abs=TOLERANCE)
        assert node["halo_risk_contribution_pct"] == pytest.approx(
            risk_contribution.loc[ticker, "Risk Contribution %"], abs=TOLERANCE
        )


def test_universe_positions_reflect_correlation_proximity(payload):
    """The most-correlated pair must not be the most distant pair in space.

    This is the property that makes the spatial layout analytically honest:
    position encodes co-movement, so it has to actually track correlation.
    """
    universe = payload["windows"]["5Y"]["universe"]
    nodes = {n["ticker"]: n for n in universe["nodes"]}

    def spatial_distance(a, b):
        return math.hypot(nodes[a]["x"] - nodes[b]["x"], nodes[a]["z"] - nodes[b]["z"])

    edges = universe["edges"]
    assert len(edges) == len(TICKERS) * (len(TICKERS) - 1) // 2

    most_correlated = max(edges, key=lambda e: e["correlation"])
    least_correlated = min(edges, key=lambda e: e["correlation"])

    assert spatial_distance(
        most_correlated["source"], most_correlated["target"]
    ) < spatial_distance(least_correlated["source"], least_correlated["target"])


def test_universe_centroid_is_the_weighted_mean_of_node_positions(payload):
    universe = payload["windows"]["5Y"]["universe"]
    nodes = universe["nodes"]

    expected_x = sum(n["x"] * n["radius_weight"] for n in nodes)
    expected_z = sum(n["z"] * n["radius_weight"] for n in nodes)

    assert universe["centroid"]["x"] == pytest.approx(expected_x, abs=1e-4)
    assert universe["centroid"]["z"] == pytest.approx(expected_z, abs=1e-4)


def test_universe_edges_cover_every_pair_without_duplication(payload):
    for key, window in payload["windows"].items():
        pairs = {
            frozenset((edge["source"], edge["target"]))
            for edge in window["universe"]["edges"]
        }
        assert len(pairs) == len(window["universe"]["edges"]), f"{key} has duplicate edges"
        for edge in window["universe"]["edges"]:
            assert edge["source"] != edge["target"]
            assert -1.0 <= edge["correlation"] <= 1.0


# ---------------------------------------------------------------------------
# Window metrics agree with the approved modules
# ---------------------------------------------------------------------------


def test_full_sample_portfolio_metrics_match_portfolio_module(payload, returns):
    summary = portfolio.create_portfolio_summary(returns, PORTFOLIO_WEIGHTS).loc["Portfolio"]
    exported = payload["windows"]["5Y"]["portfolio"]

    assert exported["ann_return"] == pytest.approx(summary["Annualized Return"], abs=TOLERANCE)
    assert exported["ann_volatility"] == pytest.approx(
        summary["Annualized Volatility"], abs=TOLERANCE
    )
    assert exported["sharpe"] == pytest.approx(summary["Sharpe Ratio"], abs=TOLERANCE)
    assert exported["max_drawdown"] == pytest.approx(summary["Max Drawdown"], abs=TOLERANCE)
    assert exported["beta"] == pytest.approx(summary["Beta"], abs=TOLERANCE)


def test_full_sample_benchmark_metrics_match_comparison_table(payload, returns):
    comparison = portfolio.create_portfolio_comparison(returns, PORTFOLIO_WEIGHTS)
    benchmark = comparison.loc["S&P 500"]
    exported = payload["windows"]["5Y"]["benchmark"]

    assert exported["ann_return"] == pytest.approx(benchmark["Annualized Return"], abs=TOLERANCE)
    assert exported["sharpe"] == pytest.approx(benchmark["Sharpe Ratio"], abs=TOLERANCE)
    assert exported["beta"] == pytest.approx(1.0, abs=TOLERANCE)


def test_every_window_recomputes_metrics_from_its_own_slice(payload, returns):
    """Each range must be a genuine recomputation, not a copy of the 5Y set."""
    for key, window in payload["windows"].items():
        window_returns = returns.iloc[window["start_index"] :]
        expected = portfolio.create_portfolio_summary(window_returns, PORTFOLIO_WEIGHTS).loc[
            "Portfolio"
        ]

        assert window["portfolio"]["ann_return"] == pytest.approx(
            expected["Annualized Return"], abs=TOLERANCE
        ), f"{key} annualized return was not recomputed from its own window"
        assert window["portfolio"]["sharpe"] == pytest.approx(
            expected["Sharpe Ratio"], abs=TOLERANCE
        ), f"{key} Sharpe ratio was not recomputed from its own window"
        assert window["observations"] == len(window_returns)


def test_windows_are_not_all_identical(payload):
    sharpes = {key: w["portfolio"]["sharpe"] for key, w in payload["windows"].items()}
    assert len(set(sharpes.values())) > 1, "time ranges are not producing distinct metrics"


def test_every_window_asset_metrics_match_metrics_module(payload, returns):
    for key, window in payload["windows"].items():
        window_returns = returns.iloc[window["start_index"] :]
        expected = metrics.create_metrics_summary(window_returns)

        for ticker in TICKERS:
            assert window["assets"][ticker]["ann_volatility"] == pytest.approx(
                expected.loc[ticker, "Annualized Volatility"], abs=TOLERANCE
            ), f"{key}/{ticker} volatility disagrees with metrics module"
            assert window["assets"][ticker]["sharpe"] == pytest.approx(
                expected.loc[ticker, "Sharpe Ratio"], abs=TOLERANCE
            ), f"{key}/{ticker} Sharpe disagrees with metrics module"


def test_every_window_risk_contribution_sums_to_one(payload):
    for key, window in payload["windows"].items():
        total = sum(a["risk_contribution_pct"] for a in window["assets"].values())
        assert total == pytest.approx(1.0, abs=1e-4), f"{key} risk contributions sum to {total}"


def test_weights_sum_to_one_and_match_config(payload):
    assert payload["meta"]["weights"] == {
        t: pytest.approx(w) for t, w in PORTFOLIO_WEIGHTS.items()
    }
    assert sum(payload["meta"]["weights"].values()) == pytest.approx(1.0)


def test_window_total_returns_match_cumulative_series(payload, returns):
    for key, window in payload["windows"].items():
        window_returns = returns.iloc[window["start_index"] :]
        portfolio_returns = portfolio.calculate_portfolio_returns(
            window_returns, PORTFOLIO_WEIGHTS
        )
        expected = portfolio.calculate_portfolio_cumulative_return(portfolio_returns).iloc[-1]
        assert window["total_return"]["portfolio"] == pytest.approx(
            expected, abs=TOLERANCE
        ), f"{key} total return disagrees with the cumulative series"


# ---------------------------------------------------------------------------
# The rebasing identity the front end relies on
# ---------------------------------------------------------------------------


def test_front_end_rebasing_identity_matches_windowed_python_computation(payload, returns):
    """The dashboard rebases with (1 + c_t) / (1 + c_start) - 1.

    That must be exactly equal to recomputing cumulative returns on the
    windowed slice in Python, otherwise panning the chart would silently
    change the numbers being displayed.
    """
    cumulative = payload["series"]["cumulative"]["PORTFOLIO"]

    for key, window in payload["windows"].items():
        start_index = window["start_index"]
        window_returns = returns.iloc[start_index:]
        expected = portfolio.calculate_portfolio_cumulative_return(
            portfolio.calculate_portfolio_returns(window_returns, PORTFOLIO_WEIGHTS)
        )

        if start_index == 0:
            rebased = cumulative
        else:
            base = 1.0 + cumulative[start_index - 1]
            rebased = [(1.0 + c) / base - 1.0 for c in cumulative[start_index - 1 :]]
            # The windowed computation starts at the first in-window return,
            # whose cumulative value is that return itself.
            rebased = rebased[1:]

        assert len(rebased) == len(expected), f"{key} rebased length mismatch"
        assert np.allclose(rebased, expected.to_numpy(), atol=1e-4), (
            f"{key} rebasing identity does not reproduce the windowed computation"
        )


# ---------------------------------------------------------------------------
# Series blocks
# ---------------------------------------------------------------------------


def test_series_lengths_are_coherent(payload, prices):
    series = payload["series"]

    assert len(series["dates"]) == len(prices)
    assert len(series["return_dates"]) == len(prices) - 1

    for name, values in series["cumulative"].items():
        assert len(values) == len(prices) - 1, f"cumulative.{name} length"
    for name, values in series["drawdown"].items():
        assert len(values) == len(prices) - 1, f"drawdown.{name} length"
    for ticker, block in series["moving_averages"].items():
        for field, values in block.items():
            assert len(values) == len(prices), f"moving_averages.{ticker}.{field} length"
    for ticker, block in series["backtest"].items():
        for field, values in block.items():
            assert len(values) == len(prices), f"backtest.{ticker}.{field} length"


def test_moving_average_series_match_signals_module(payload, prices):
    moving_averages = signals.calculate_moving_averages(prices, MA_SHORT, MA_LONG)

    for ticker in TICKERS:
        exported = payload["series"]["moving_averages"][ticker]
        expected_price = moving_averages[(ticker, "Price")].to_numpy()
        assert np.allclose(exported["price"], expected_price, atol=1e-3)

        expected_short = moving_averages[(ticker, "MA_Short")].to_numpy()
        for index, value in enumerate(exported["ma_short"]):
            if value is None:
                assert np.isnan(expected_short[index])
            else:
                assert value == pytest.approx(expected_short[index], abs=1e-3)


def test_moving_average_lead_in_is_null_not_backfilled(payload):
    for ticker in TICKERS:
        block = payload["series"]["moving_averages"][ticker]
        assert block["ma_short"][0] is None
        assert block["ma_long"][0] is None
        assert block["ma_long"][MA_LONG - 2] is None
        assert block["ma_long"][MA_LONG - 1] is not None
        assert block["trend"][0] == "Unavailable"


def test_cumulative_series_match_metrics_module(payload, returns):
    cumulative = metrics.calculate_cumulative_returns(returns)

    for ticker in TICKERS:
        assert np.allclose(
            payload["series"]["cumulative"][ticker], cumulative[ticker].to_numpy(), atol=1e-5
        )
    assert np.allclose(
        payload["series"]["cumulative"]["BENCHMARK"],
        cumulative[BENCHMARK].to_numpy(),
        atol=1e-5,
    )


def test_drawdown_series_are_non_positive_and_match_module(payload, returns):
    portfolio_returns = portfolio.calculate_portfolio_returns(returns, PORTFOLIO_WEIGHTS)
    expected = portfolio.calculate_portfolio_drawdown(portfolio_returns)

    exported = payload["series"]["drawdown"]["PORTFOLIO"]
    assert np.allclose(exported, expected.to_numpy(), atol=1e-5)
    assert max(exported) <= 1e-9, "a drawdown series must never be positive"


def test_max_drawdown_metric_matches_the_drawdown_series(payload):
    series_min = min(payload["series"]["drawdown"]["PORTFOLIO"])
    assert payload["windows"]["5Y"]["portfolio"]["max_drawdown"] == pytest.approx(
        series_min, abs=1e-4
    )


def test_backtest_wealth_series_match_signals_module(payload, prices):
    for ticker in TICKERS:
        expected = signals.backtest_moving_average_signal(prices, ticker, MA_SHORT, MA_LONG)
        exported = payload["series"]["backtest"][ticker]

        assert np.allclose(
            exported["buy_hold_wealth"], expected["Buy & Hold Wealth"].to_numpy(), atol=1e-5
        )
        assert np.allclose(
            exported["strategy_wealth"], expected["Strategy Wealth"].to_numpy(), atol=1e-5
        )
        assert set(exported["position"]) <= {0, 1}


# ---------------------------------------------------------------------------
# Signals block
# ---------------------------------------------------------------------------


def test_signal_summary_matches_signals_module(payload, prices):
    expected = signals.create_signal_summary(prices, MA_SHORT, MA_LONG)

    for ticker in TICKERS:
        exported = payload["signals"]["summary"][ticker]
        assert exported["trend"] == expected.loc[ticker, "Trend"]
        assert exported["last_crossover"] == expected.loc[ticker, "Last Crossover"]
        assert exported["latest_price"] == pytest.approx(
            expected.loc[ticker, "Latest Price"], abs=1e-3
        )


def test_crossover_events_match_signals_module(payload, prices):
    expected = signals.detect_crossovers(
        signals.calculate_moving_averages(prices, MA_SHORT, MA_LONG)
    )
    events = payload["signals"]["events"]

    assert len(events) == len(expected)

    dates = [e["date"] for e in events]
    assert dates == sorted(dates), "events must be chronological"

    for event in events:
        assert event["ticker"] in TICKERS
        assert event["signal"] in ("Golden Cross", "Death Cross")
        assert event["price"] is not None


def test_backtest_summary_matches_signals_module(payload, prices):
    expected = signals.create_backtest_summary(prices, MA_SHORT, MA_LONG)

    for ticker in TICKERS:
        exported = payload["signals"]["backtest"][ticker]
        assert exported["buy_hold"]["total_return"] == pytest.approx(
            expected.loc[ticker, "Buy & Hold Total Return"], abs=TOLERANCE
        )
        assert exported["strategy"]["ann_return"] == pytest.approx(
            expected.loc[ticker, "Strategy Annualized Return"], abs=TOLERANCE
        )


# ---------------------------------------------------------------------------
# Monte Carlo block
# ---------------------------------------------------------------------------


def test_monte_carlo_config_matches_project_configuration(payload):
    config = payload["monte_carlo"]["config"]
    assert config["simulations"] == MONTE_CARLO_SIMULATIONS
    assert config["days"] == MONTE_CARLO_DAYS
    assert config["distribution"] == "normal"


def test_monte_carlo_bands_are_ordered_at_every_horizon(payload):
    bands = payload["monte_carlo"]["bands"]
    for day in range(payload["monte_carlo"]["config"]["days"]):
        assert (
            bands["p5"][day]
            <= bands["p25"][day]
            <= bands["p50"][day]
            <= bands["p75"][day]
            <= bands["p95"][day]
        ), f"percentile bands out of order at day {day + 1}"


def test_monte_carlo_bands_start_near_the_initial_value(payload):
    # Day one of a wealth path is 1 + a single daily return, so every
    # percentile must still be close to the unit starting value.
    for name, values in payload["monte_carlo"]["bands"].items():
        assert values[0] == pytest.approx(1.0, abs=0.1), f"band {name} does not start near 1.0"


def test_monte_carlo_histogram_accounts_for_every_simulation(payload):
    histogram = payload["monte_carlo"]["terminal_histogram"]
    assert len(histogram["bin_edges"]) == len(histogram["counts"]) + 1
    assert sum(histogram["counts"]) == payload["monte_carlo"]["config"]["simulations"]
    assert histogram["bin_edges"] == sorted(histogram["bin_edges"])


def test_monte_carlo_sample_paths_are_deterministic_and_shaped(payload):
    paths = payload["monte_carlo"]["sample_paths"]
    assert len(paths) == exporter.MC_SAMPLE_PATHS
    for path in paths:
        assert len(path) == payload["monte_carlo"]["config"]["days"]
        assert all(value is not None for value in path)


def test_var_and_cvar_follow_the_positive_loss_convention(payload):
    mc = payload["monte_carlo"]
    p5 = mc["summary"]["p5_terminal_return"]

    assert mc["var"] == pytest.approx(max(0.0, -p5), abs=TOLERANCE)
    assert mc["var"] >= 0.0
    assert mc["cvar"] >= mc["var"] - TOLERANCE, "CVaR must be at least as large as VaR"


def test_downside_probabilities_are_monotonically_decreasing(payload):
    downside = payload["monte_carlo"]["downside_probabilities"]
    assert (
        downside["loss"]
        >= downside["below_minus_10"]
        >= downside["below_minus_20"]
        >= downside["below_minus_30"]
    )
    assert 0.0 <= downside["loss"] <= 1.0


# ---------------------------------------------------------------------------
# Metadata and provenance
# ---------------------------------------------------------------------------


def test_meta_records_provenance_and_assumptions(payload, prices):
    meta = payload["meta"]

    assert meta["schema_version"] == exporter.SCHEMA_VERSION
    assert meta["pipeline_status"] == "ok"
    assert meta["latest_market_date"] == prices.index.max().date().isoformat()
    assert meta["first_market_date"] == prices.index.min().date().isoformat()
    assert meta["trading_days"] == len(prices)
    assert meta["tickers"] == list(TICKERS)
    assert meta["benchmark"]["symbol"] == BENCHMARK
    assert meta["default_window"] in payload["windows"]
    assert meta["git_sha"]

    assumptions = meta["assumptions"]
    assert assumptions["risk_free_rate"] == pytest.approx(RISK_FREE_RATE)
    assert assumptions["trading_days_per_year"] == TRADING_DAYS
    assert assumptions["ma_short"] == MA_SHORT
    assert assumptions["ma_long"] == MA_LONG
    assert assumptions["transaction_costs_modelled"] is False


def test_generated_at_is_an_iso_utc_timestamp(payload):
    stamp = payload["meta"]["generated_at_utc"]
    assert stamp.endswith("Z")
    pd.Timestamp(stamp)


def test_meta_window_list_matches_exported_windows(payload):
    listed = {w["key"] for w in payload["meta"]["windows"]}
    assert listed == set(payload["windows"])


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def test_payload_serializes_without_nan_or_infinity(payload):
    # allow_nan=False raises rather than emitting the invalid JSON literals
    # NaN / Infinity, which would break JSON.parse in the browser.
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text

    reloaded = json.loads(text)
    assert reloaded["meta"]["latest_market_date"] == payload["meta"]["latest_market_date"]


def test_write_payload_round_trips(tmp_path, payload):
    destination = tmp_path / "nested" / "dashboard_data.json"
    written = exporter.write_payload(payload, destination)

    assert written.exists()
    reloaded = json.loads(written.read_text(encoding="utf-8"))
    assert reloaded["meta"]["schema_version"] == exporter.SCHEMA_VERSION
    assert set(reloaded["windows"]) == set(payload["windows"])


def test_payload_stays_within_the_transfer_budget(tmp_path, payload):
    destination = tmp_path / "dashboard_data.json"
    exporter.write_payload(payload, destination)

    size_kb = destination.stat().st_size / 1024
    assert size_kb < 1600, f"payload grew to {size_kb:.0f} KB; revisit the reduction strategy"


def test_no_null_metrics_in_any_window(payload):
    for key, window in payload["windows"].items():
        for metric, value in window["portfolio"].items():
            assert value is not None, f"{key}.portfolio.{metric} is null"
        for ticker, asset in window["assets"].items():
            for metric, value in asset.items():
                assert value is not None, f"{key}.assets.{ticker}.{metric} is null"
