"""
Unit tests for src/visualizations.py. Uses synthetic deterministic price
data and pytest's tmp_path fixture for output files; no live Yahoo
Finance data is used. Tests check functional behavior (files are
created, non-empty, figures closed, paths returned correctly), not
subjective chart aesthetics.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import pandas as pd
import pytest

from config import BENCHMARK, TICKERS
from src import metrics, simulation, visualizations


def _synthetic_prices(n_days: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    columns = list(TICKERS) + [BENCHMARK]
    data = {}
    for column in columns:
        daily_returns = rng.normal(loc=0.0004, scale=0.015, size=n_days)
        data[column] = 100 * np.cumprod(1 + daily_returns)
    return pd.DataFrame(data, index=dates)[columns]


PRICES = _synthetic_prices()
RETURNS = metrics.calculate_daily_returns(PRICES)
SIMULATION_RESULT = simulation.run_monte_carlo_simulation(RETURNS, simulations=50, days=30, seed=1)


def _open_figure_count() -> int:
    return len(plt.get_fignums())


# --- 1. Portfolio-vs-benchmark chart file created ---------------------------------

def test_plot_portfolio_vs_benchmark_creates_file(tmp_path):
    output_path = tmp_path / "portfolio_vs_sp500.png"
    result = visualizations.plot_portfolio_vs_benchmark(RETURNS, output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- 2. Correlation heatmap file created --------------------------------------------

def test_plot_correlation_heatmap_creates_file(tmp_path):
    output_path = tmp_path / "correlation_heatmap.png"
    result = visualizations.plot_correlation_heatmap(RETURNS, output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- 3. Risk-return scatter file created ---------------------------------------------

def test_plot_risk_return_scatter_creates_file(tmp_path):
    output_path = tmp_path / "risk_return_scatter.png"
    result = visualizations.plot_risk_return_scatter(RETURNS, output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- 4. Drawdown chart file created -----------------------------------------------------

def test_plot_portfolio_drawdown_creates_file(tmp_path):
    output_path = tmp_path / "portfolio_drawdown.png"
    result = visualizations.plot_portfolio_drawdown(RETURNS, output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- 5. Risk-contribution chart file created ----------------------------------------------

def test_plot_risk_contribution_creates_file(tmp_path):
    output_path = tmp_path / "risk_contribution.png"
    result = visualizations.plot_risk_contribution(RETURNS, output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- 6. Moving-average chart file created --------------------------------------------------

def test_plot_moving_average_signals_creates_file(tmp_path):
    output_path = tmp_path / "nvda_ma_signals.png"
    result = visualizations.plot_moving_average_signals(PRICES, "NVDA", output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- 7. Monte Carlo path chart file created -----------------------------------------------

def test_plot_monte_carlo_paths_creates_file(tmp_path):
    output_path = tmp_path / "monte_carlo_paths.png"
    result = visualizations.plot_monte_carlo_paths(SIMULATION_RESULT, output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- 8. Terminal-distribution chart file created -------------------------------------------

def test_plot_terminal_return_distribution_creates_file(tmp_path):
    output_path = tmp_path / "terminal_return_distribution.png"
    result = visualizations.plot_terminal_return_distribution(SIMULATION_RESULT, output_path=output_path)

    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


# --- 9. generate_all_charts returns expected mapping ----------------------------------------

def test_generate_all_charts_returns_expected_mapping(tmp_path):
    result = visualizations.generate_all_charts(
        PRICES, RETURNS, SIMULATION_RESULT, output_dir=tmp_path
    )

    assert set(result.keys()) == {
        "portfolio_vs_sp500",
        "correlation_heatmap",
        "risk_return_scatter",
        "portfolio_drawdown",
        "risk_contribution",
        "moving_average_signals",
        "monte_carlo_paths",
        "terminal_return_distribution",
    }


# --- 10. All returned paths exist -------------------------------------------------------------

def test_generate_all_charts_all_paths_exist(tmp_path):
    result = visualizations.generate_all_charts(
        PRICES, RETURNS, SIMULATION_RESULT, output_dir=tmp_path
    )

    for key, path in result.items():
        assert path.exists(), f"missing chart file for {key}"
        assert path.stat().st_size > 0


# --- 11. Invalid ticker raises -------------------------------------------------------------------

def test_plot_moving_average_signals_invalid_ticker_raises(tmp_path):
    with pytest.raises(ValueError):
        visualizations.plot_moving_average_signals(
            PRICES, "NOT_A_TICKER", output_path=tmp_path / "bad.png"
        )


# --- 12. Figures are closed after export -----------------------------------------------------------

def test_figures_closed_after_export(tmp_path):
    baseline = _open_figure_count()

    visualizations.plot_portfolio_vs_benchmark(RETURNS, output_path=tmp_path / "a.png")
    assert _open_figure_count() == baseline

    visualizations.plot_correlation_heatmap(RETURNS, output_path=tmp_path / "b.png")
    assert _open_figure_count() == baseline

    visualizations.plot_monte_carlo_paths(SIMULATION_RESULT, output_path=tmp_path / "c.png")
    assert _open_figure_count() == baseline

    visualizations.generate_all_charts(PRICES, RETURNS, SIMULATION_RESULT, output_dir=tmp_path / "all")
    assert _open_figure_count() == baseline


# --- 13. Output directory is created when missing ------------------------------------------------------

def test_output_directory_created_when_missing(tmp_path):
    missing_dir = tmp_path / "does" / "not" / "exist" / "yet"
    assert not missing_dir.exists()

    result = visualizations.generate_all_charts(PRICES, RETURNS, SIMULATION_RESULT, output_dir=missing_dir)

    assert missing_dir.exists()
    for path in result.values():
        assert path.exists()
