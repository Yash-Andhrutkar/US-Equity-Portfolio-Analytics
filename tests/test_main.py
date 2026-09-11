"""
Unit tests for main.py. Uses synthetic deterministic price data and
monkeypatches the data-loading step (src.data_loader.load_price_data /
run_data_pipeline) so no live Yahoo Finance data or network access is
used. Table, report, and chart outputs are all redirected under pytest's
tmp_path fixture so these tests never touch the project's real output/
directory.
"""

import numpy as np
import pandas as pd
import pytest

import main
from config import BENCHMARK, TICKERS


def _synthetic_prices(n_days: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    columns = list(TICKERS) + [BENCHMARK]
    data = {}
    for column in columns:
        daily_returns = rng.normal(loc=0.0004, scale=0.015, size=n_days)
        data[column] = 100 * np.cumprod(1 + daily_returns)
    return pd.DataFrame(data, index=dates)[columns]


PRICES = _synthetic_prices()

EXPECTED_TABLE_FILENAMES = {
    "asset_metrics.csv",
    "portfolio_summary.csv",
    "portfolio_vs_sp500.csv",
    "risk_contribution.csv",
    "signal_summary.csv",
    "backtest_summary.csv",
    "monte_carlo_summary.csv",
}

EXPECTED_CHART_KEYS = {
    "portfolio_vs_sp500",
    "correlation_heatmap",
    "risk_return_scatter",
    "portfolio_drawdown",
    "risk_contribution",
    "moving_average_signals",
    "monte_carlo_paths",
    "terminal_return_distribution",
}

EXPECTED_RESULT_KEYS = {
    "prices",
    "returns",
    "data_source",
    "asset_metrics",
    "portfolio_summary",
    "portfolio_comparison",
    "risk_contribution",
    "signal_summary",
    "backtest_summary",
    "simulation_result",
    "chart_paths",
    "table_paths",
    "report_path",
}

EXPECTED_SIMULATION_RESULT_KEYS = {
    "return_paths",
    "wealth_paths",
    "terminal_returns",
    "summary",
    "downside_probabilities",
    "var",
    "cvar",
}


def _patch_output_dirs(monkeypatch, tmp_path):
    """Redirect main's table/report/chart output locations under tmp_path."""
    tables_dir = tmp_path / "tables"
    reports_dir = tmp_path / "reports"
    charts_dir = tmp_path / "charts"
    monkeypatch.setattr(main, "TABLES_DIR", tables_dir)
    monkeypatch.setattr(main, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(main, "CHARTS_DIR", charts_dir)
    return tables_dir, reports_dir, charts_dir


def _patch_cached_data(monkeypatch, prices=PRICES):
    """Simulate a cache hit: load_price_data succeeds, run_data_pipeline must not be called."""
    monkeypatch.setattr(main.data_loader, "load_price_data", lambda: prices)

    def _unexpected_pipeline_call():
        raise AssertionError(
            "run_data_pipeline() should not be called when cached data is available"
        )

    monkeypatch.setattr(main.data_loader, "run_data_pipeline", _unexpected_pipeline_call)


def _patch_missing_cache(monkeypatch, prices=PRICES):
    """Simulate a cache miss: load_price_data raises FileNotFoundError, pipeline provides data."""

    def _missing_cache():
        raise FileNotFoundError("no cached data for test")

    monkeypatch.setattr(main.data_loader, "load_price_data", _missing_cache)
    monkeypatch.setattr(main.data_loader, "run_data_pipeline", lambda: prices)


# --- 1. Output directories are created --------------------------------------------------

def test_output_directories_created(tmp_path, monkeypatch):
    tables_dir, reports_dir, charts_dir = _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    main.run_analysis()

    assert tables_dir.is_dir()
    assert reports_dir.is_dir()
    assert charts_dir.is_dir()


# --- 2. Expected CSV files are exported --------------------------------------------------

def test_expected_csv_files_exported(tmp_path, monkeypatch):
    tables_dir, _, _ = _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    main.run_analysis()

    actual = {p.name for p in tables_dir.glob("*.csv")}
    assert actual == EXPECTED_TABLE_FILENAMES


# --- 3. Report file is created -------------------------------------------------------------

def test_report_file_created(tmp_path, monkeypatch):
    _, reports_dir, _ = _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    result = main.run_analysis()

    report_path = reports_dir / "analysis_summary.txt"
    assert report_path.exists()
    assert result["report_path"] == report_path


# --- 4. Chart mapping is returned -----------------------------------------------------------

def test_chart_mapping_returned(tmp_path, monkeypatch):
    _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    result = main.run_analysis()

    assert set(result["chart_paths"].keys()) == EXPECTED_CHART_KEYS
    for path in result["chart_paths"].values():
        assert path.exists()
        assert path.stat().st_size > 0


# --- 5. Expected keys exist in run_analysis() result --------------------------------------------

def test_run_analysis_result_keys(tmp_path, monkeypatch):
    _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    result = main.run_analysis()

    assert set(result.keys()) == EXPECTED_RESULT_KEYS


# --- 6. Cached-data path is used when available --------------------------------------------------

def test_cached_data_path_used_when_available(tmp_path, monkeypatch):
    _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    result = main.run_analysis()

    assert result["data_source"] == "cache"
    pd.testing.assert_frame_equal(result["prices"], PRICES)


# --- 7. Fallback data pipeline is used when cache is missing --------------------------------------

def test_fallback_pipeline_used_when_cache_missing(tmp_path, monkeypatch):
    _patch_output_dirs(monkeypatch, tmp_path)
    _patch_missing_cache(monkeypatch)

    result = main.run_analysis()

    assert result["data_source"] == "downloaded"
    pd.testing.assert_frame_equal(result["prices"], PRICES)


# --- 8. Monte Carlo result is included -----------------------------------------------------------

def test_monte_carlo_result_included(tmp_path, monkeypatch):
    _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    result = main.run_analysis()

    simulation_result = result["simulation_result"]
    assert set(simulation_result.keys()) == EXPECTED_SIMULATION_RESULT_KEYS
    assert isinstance(simulation_result["var"], float)
    assert isinstance(simulation_result["cvar"], float)


# --- 9. Report contains "not a forecast" -----------------------------------------------------------

def test_report_contains_not_a_forecast_disclaimer(tmp_path, monkeypatch):
    _, reports_dir, _ = _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    main.run_analysis()

    report_text = (reports_dir / "analysis_summary.txt").read_text(encoding="utf-8").lower()
    assert "not a forecast" in report_text


# --- 10. Exported CSV files are non-empty ------------------------------------------------------------

def test_exported_csv_files_are_nonempty(tmp_path, monkeypatch):
    tables_dir, _, _ = _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    main.run_analysis()

    csv_files = list(tables_dir.glob("*.csv"))
    assert len(csv_files) == len(EXPECTED_TABLE_FILENAMES)
    for csv_path in csv_files:
        assert csv_path.stat().st_size > 0


# --- 11. main() prints a completion summary and exits cleanly on success --------------------------

def test_main_runs_cleanly_and_prints_summary(tmp_path, monkeypatch, capsys):
    _patch_output_dirs(monkeypatch, tmp_path)
    _patch_cached_data(monkeypatch)

    main.main()

    captured = capsys.readouterr().out
    assert "US Equity Portfolio Analytics" in captured
    assert "Analysis complete." in captured


# --- 12. main() exits non-zero and does not swallow a pipeline failure ----------------------------

def test_main_exits_nonzero_on_pipeline_failure(tmp_path, monkeypatch):
    _patch_output_dirs(monkeypatch, tmp_path)

    def _broken_load_price_data():
        raise FileNotFoundError("cache missing")

    def _broken_pipeline():
        raise RuntimeError("simulated download failure")

    monkeypatch.setattr(main.data_loader, "load_price_data", _broken_load_price_data)
    monkeypatch.setattr(main.data_loader, "run_data_pipeline", _broken_pipeline)

    with pytest.raises(SystemExit) as excinfo:
        main.main()

    assert excinfo.value.code == 1
