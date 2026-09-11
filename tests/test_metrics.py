"""
Unit tests for src/metrics.py. All datasets are small and deterministic;
no live Yahoo Finance data is used. Expected values are computed
independently from the documented formulas, not by re-calling the
functions under test.
"""

import math

import numpy as np
import pandas as pd
import pytest

from config import BENCHMARK, RISK_FREE_RATE, TICKERS, TRADING_DAYS
from src import metrics


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


# --- 1. Daily returns ---------------------------------------------------

def test_calculate_daily_returns():
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0], "B": [50.0, 45.0, 40.5]},
        index=_dates(3),
    )
    returns = metrics.calculate_daily_returns(prices)

    assert len(returns) == 2
    assert returns.index[0] == prices.index[1]
    assert returns["A"].tolist() == pytest.approx([0.10, 0.10])
    assert returns["B"].tolist() == pytest.approx([-0.10, -0.10])
    assert not returns.isna().any().any()


def test_calculate_daily_returns_rejects_empty_input():
    with pytest.raises(ValueError):
        metrics.calculate_daily_returns(pd.DataFrame())


# --- 2. Cumulative compounding ------------------------------------------

def test_calculate_cumulative_returns():
    returns = pd.DataFrame({"A": [0.10, 0.10]}, index=_dates(2))
    cumulative = metrics.calculate_cumulative_returns(returns)

    expected_day1 = 1.10 - 1
    expected_day2 = (1.10 * 1.10) - 1
    assert cumulative["A"].tolist() == pytest.approx([expected_day1, expected_day2])


# --- 3. Annualized return -------------------------------------------------

def test_calculate_annualized_return():
    values = [0.01, 0.02, -0.01, 0.03]
    returns = pd.DataFrame({"A": values}, index=_dates(len(values)))

    total_return = np.prod([1 + r for r in values]) - 1
    expected = (1 + total_return) ** (TRADING_DAYS / len(values)) - 1

    result = metrics.calculate_annualized_return(returns)
    assert result["A"] == pytest.approx(expected)


# --- 4. Annualized volatility ---------------------------------------------

def test_calculate_annualized_volatility():
    values = [0.01, 0.02, -0.01, 0.03]
    returns = pd.DataFrame({"A": values}, index=_dates(len(values)))

    expected = np.std(values, ddof=1) * math.sqrt(TRADING_DAYS)

    result = metrics.calculate_annualized_volatility(returns)
    assert result["A"] == pytest.approx(expected)


# --- 5. Sharpe ratio -------------------------------------------------------

def test_calculate_sharpe_ratio():
    values = [0.01, 0.02, -0.01, 0.03]
    returns = pd.DataFrame({"A": values}, index=_dates(len(values)))

    total_return = np.prod([1 + r for r in values]) - 1
    expected_return = (1 + total_return) ** (TRADING_DAYS / len(values)) - 1
    expected_vol = np.std(values, ddof=1) * math.sqrt(TRADING_DAYS)
    expected_sharpe = (expected_return - RISK_FREE_RATE) / expected_vol

    result = metrics.calculate_sharpe_ratio(returns)
    assert result["A"] == pytest.approx(expected_sharpe)


# --- 6. Drawdown series -----------------------------------------------------

def test_calculate_drawdown():
    returns = pd.DataFrame({"A": [0.10, -0.20, 0.05]}, index=_dates(3))
    drawdown = metrics.calculate_drawdown(returns)

    assert drawdown["A"].tolist() == pytest.approx([0.0, -0.20, -0.16])


# --- 7. Maximum drawdown -----------------------------------------------------

def test_calculate_max_drawdown():
    returns = pd.DataFrame({"A": [0.10, -0.20, 0.05]}, index=_dates(3))
    max_dd = metrics.calculate_max_drawdown(returns)

    assert max_dd["A"] == pytest.approx(-0.20)
    assert max_dd["A"] < 0


# --- 8. Beta against a known benchmark series --------------------------------

def test_calculate_beta_against_known_benchmark():
    bench = [0.01, -0.02, 0.03, 0.00, 0.02]
    returns = pd.DataFrame(
        {
            BENCHMARK: bench,
            "DOUBLE": [2 * r for r in bench],
            "HALF": [0.5 * r for r in bench],
        },
        index=_dates(len(bench)),
    )

    beta = metrics.calculate_beta(returns, benchmark=BENCHMARK)

    assert beta["DOUBLE"] == pytest.approx(2.0)
    assert beta["HALF"] == pytest.approx(0.5)
    assert BENCHMARK not in beta.index


# --- 9. Correlation matrix ----------------------------------------------------

def test_calculate_correlation_matrix():
    bench = [0.01, -0.02, 0.03, 0.00, 0.02, -0.01]
    returns = pd.DataFrame(
        {
            BENCHMARK: bench,
            "PERFECT_POS": bench,
            "PERFECT_NEG": [-r for r in bench],
        },
        index=_dates(len(bench)),
    )

    corr = metrics.calculate_correlation_matrix(returns)

    assert corr.loc[BENCHMARK, "PERFECT_POS"] == pytest.approx(1.0)
    assert corr.loc[BENCHMARK, "PERFECT_NEG"] == pytest.approx(-1.0)
    for asset in corr.columns:
        assert corr.loc[asset, asset] == pytest.approx(1.0)


# --- 10. Summary table structure ----------------------------------------------

def test_create_metrics_summary_structure():
    n_days = 60
    rng = np.random.default_rng(42)
    columns = list(TICKERS) + [BENCHMARK]
    daily_moves = rng.normal(loc=0.0005, scale=0.01, size=(n_days, len(columns)))
    prices = 100 * (1 + pd.DataFrame(daily_moves, columns=columns, index=_dates(n_days))).cumprod()

    returns = metrics.calculate_daily_returns(prices)
    summary = metrics.create_metrics_summary(returns)

    assert list(summary.index) == list(TICKERS)
    assert BENCHMARK not in summary.index
    assert list(summary.columns) == [
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Beta",
    ]
    assert summary.shape == (len(TICKERS), 5)
    assert all(pd.api.types.is_numeric_dtype(summary[c]) for c in summary.columns)


# --- 11. Invalid / missing benchmark behaviour --------------------------------

def test_calculate_beta_missing_benchmark_raises():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]}, index=_dates(3))
    with pytest.raises(ValueError):
        metrics.calculate_beta(returns, benchmark=BENCHMARK)


def test_calculate_beta_zero_variance_benchmark_raises():
    returns = pd.DataFrame(
        {BENCHMARK: [0.01, 0.01, 0.01, 0.01], "A": [0.02, -0.01, 0.03, 0.00]},
        index=_dates(4),
    )
    with pytest.raises(ValueError):
        metrics.calculate_beta(returns, benchmark=BENCHMARK)


def test_create_metrics_summary_missing_benchmark_raises():
    returns = pd.DataFrame(
        {t: [0.01, 0.02, -0.01] for t in TICKERS}, index=_dates(3)
    )
    with pytest.raises(ValueError):
        metrics.create_metrics_summary(returns)


# --- 12. Zero-volatility behaviour ---------------------------------------------

def test_zero_volatility_produces_nan_sharpe():
    returns = pd.DataFrame({"FLAT": [0.0, 0.0, 0.0, 0.0]}, index=_dates(4))

    volatility = metrics.calculate_annualized_volatility(returns)
    assert volatility["FLAT"] == pytest.approx(0.0)

    sharpe = metrics.calculate_sharpe_ratio(returns)
    assert math.isnan(sharpe["FLAT"])
