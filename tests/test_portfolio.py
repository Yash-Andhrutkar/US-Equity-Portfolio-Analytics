"""
Unit tests for src/portfolio.py. All datasets are small and deterministic;
no live Yahoo Finance data is used. Expected values are computed
independently from the documented formulas, not by re-calling the
functions under test.
"""

import math

import numpy as np
import pandas as pd
import pytest

from config import BENCHMARK, PORTFOLIO_WEIGHTS, RISK_FREE_RATE, TICKERS, TRADING_DAYS
from src import portfolio


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


# --- 1. Weight validation ------------------------------------------------

def test_validate_portfolio_weights_accepts_valid_weights():
    portfolio.validate_portfolio_weights(PORTFOLIO_WEIGHTS, TICKERS)
    portfolio.validate_portfolio_weights({"A": 0.5, "B": 0.5}, ["A", "B"])


def test_validate_portfolio_weights_rejects_missing_ticker():
    with pytest.raises(ValueError):
        portfolio.validate_portfolio_weights({"A": 1.0}, ["A", "B"])


def test_validate_portfolio_weights_rejects_unexpected_ticker():
    with pytest.raises(ValueError):
        portfolio.validate_portfolio_weights({"A": 0.5, "B": 0.3, "C": 0.2}, ["A", "B"])


def test_validate_portfolio_weights_rejects_negative_weight():
    with pytest.raises(ValueError):
        portfolio.validate_portfolio_weights({"A": 1.2, "B": -0.2}, ["A", "B"])


def test_validate_portfolio_weights_rejects_non_numeric_weight():
    with pytest.raises(ValueError):
        portfolio.validate_portfolio_weights({"A": "0.5", "B": 0.5}, ["A", "B"])


# --- 2. Weights not summing to 1 raises -----------------------------------

def test_validate_portfolio_weights_rejects_wrong_sum():
    with pytest.raises(ValueError):
        portfolio.validate_portfolio_weights({"A": 0.5, "B": 0.6}, ["A", "B"])


# --- 3. Missing stock column raises ----------------------------------------

def test_calculate_portfolio_returns_missing_column_raises():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]}, index=_dates(3))
    with pytest.raises(ValueError):
        portfolio.calculate_portfolio_returns(returns, {"A": 0.5, "B": 0.5})


# --- 4. Weighted daily portfolio return calculation ------------------------

def test_calculate_portfolio_returns():
    returns = pd.DataFrame(
        {"A": [0.10, -0.05], "B": [0.02, 0.03]},
        index=_dates(2),
    )
    weights = {"A": 0.6, "B": 0.4}

    portfolio_returns = portfolio.calculate_portfolio_returns(returns, weights)

    expected = [0.6 * 0.10 + 0.4 * 0.02, 0.6 * -0.05 + 0.4 * 0.03]
    assert portfolio_returns.name == "Portfolio"
    assert portfolio_returns.tolist() == pytest.approx(expected)
    assert list(portfolio_returns.index) == list(returns.index)


# --- 5. Cumulative portfolio return -----------------------------------------

def test_calculate_portfolio_cumulative_return():
    portfolio_returns = pd.Series([0.10, 0.10], index=_dates(2), name="Portfolio")
    cumulative = portfolio.calculate_portfolio_cumulative_return(portfolio_returns)

    assert cumulative.tolist() == pytest.approx([0.10, 0.21])


# --- 6. Geometric annualised portfolio return -------------------------------

def test_calculate_portfolio_annualized_return():
    values = [0.01, 0.02, -0.01, 0.03]
    portfolio_returns = pd.Series(values, index=_dates(len(values)), name="Portfolio")

    total_return = np.prod([1 + r for r in values]) - 1
    expected = (1 + total_return) ** (TRADING_DAYS / len(values)) - 1

    result = portfolio.calculate_portfolio_annualized_return(portfolio_returns)
    assert result == pytest.approx(expected)


# --- 7. Annualised portfolio volatility -------------------------------------

def test_calculate_portfolio_annualized_volatility():
    values = [0.01, 0.02, -0.01, 0.03]
    portfolio_returns = pd.Series(values, index=_dates(len(values)), name="Portfolio")

    expected = np.std(values, ddof=1) * math.sqrt(TRADING_DAYS)

    result = portfolio.calculate_portfolio_annualized_volatility(portfolio_returns)
    assert result == pytest.approx(expected)


# --- 8. Portfolio Sharpe ratio -----------------------------------------------

def test_calculate_portfolio_sharpe_ratio():
    values = [0.01, 0.02, -0.01, 0.03]
    portfolio_returns = pd.Series(values, index=_dates(len(values)), name="Portfolio")

    total_return = np.prod([1 + r for r in values]) - 1
    expected_return = (1 + total_return) ** (TRADING_DAYS / len(values)) - 1
    expected_vol = np.std(values, ddof=1) * math.sqrt(TRADING_DAYS)
    expected_sharpe = (expected_return - RISK_FREE_RATE) / expected_vol

    result = portfolio.calculate_portfolio_sharpe_ratio(portfolio_returns)
    assert result == pytest.approx(expected_sharpe)


# --- 9. Portfolio drawdown ----------------------------------------------------

def test_calculate_portfolio_drawdown():
    portfolio_returns = pd.Series([0.10, -0.20, 0.05], index=_dates(3), name="Portfolio")
    drawdown = portfolio.calculate_portfolio_drawdown(portfolio_returns)

    assert drawdown.tolist() == pytest.approx([0.0, -0.20, -0.16])


# --- 10. Portfolio maximum drawdown --------------------------------------------

def test_calculate_portfolio_max_drawdown():
    portfolio_returns = pd.Series([0.10, -0.20, 0.05], index=_dates(3), name="Portfolio")
    max_dd = portfolio.calculate_portfolio_max_drawdown(portfolio_returns)

    assert max_dd == pytest.approx(-0.20)
    assert max_dd < 0


# --- 11. Portfolio beta against a known benchmark -------------------------------

def test_calculate_portfolio_beta_against_known_benchmark():
    bench_values = [0.01, -0.02, 0.03, 0.00, 0.02]
    benchmark_returns = pd.Series(bench_values, index=_dates(5), name=BENCHMARK)
    portfolio_returns = pd.Series([2 * r for r in bench_values], index=_dates(5), name="Portfolio")

    beta = portfolio.calculate_portfolio_beta(portfolio_returns, benchmark_returns)
    assert beta == pytest.approx(2.0)


def test_calculate_portfolio_beta_empty_overlap_raises():
    portfolio_returns = pd.Series([0.01, 0.02], index=_dates(2), name="Portfolio")
    benchmark_returns = pd.Series(
        [0.01, 0.02], index=pd.bdate_range("2030-01-01", periods=2), name=BENCHMARK
    )
    with pytest.raises(ValueError):
        portfolio.calculate_portfolio_beta(portfolio_returns, benchmark_returns)


# --- 12. Portfolio comparison structure -------------------------------------------

def test_create_portfolio_comparison_structure():
    n_days = 60
    rng = np.random.default_rng(7)
    columns = list(TICKERS) + [BENCHMARK]
    daily_moves = rng.normal(loc=0.0005, scale=0.01, size=(n_days, len(columns)))
    prices = 100 * (1 + pd.DataFrame(daily_moves, columns=columns, index=_dates(n_days))).cumprod()
    returns = prices.pct_change(fill_method=None).iloc[1:]

    comparison = portfolio.create_portfolio_comparison(returns)

    assert list(comparison.index) == ["Portfolio", "S&P 500"]
    assert list(comparison.columns) == [
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Beta",
    ]
    assert comparison.loc["S&P 500", "Beta"] == pytest.approx(1.0)


# --- 13 & 14. Risk contribution calculation, and percentages sum to ~100% -------

def test_calculate_risk_contribution():
    values_a = [0.01, -0.01, 0.01, -0.01, 0.02, -0.02]
    values_b = [0.02, 0.02, -0.02, -0.02, 0.01, -0.01]
    returns = pd.DataFrame({"A": values_a, "B": values_b}, index=_dates(len(values_a)))
    weights = {"A": 0.5, "B": 0.5}

    expected_cov = np.cov(np.array([values_a, values_b]), ddof=1) * TRADING_DAYS
    w = np.array([0.5, 0.5])
    expected_portfolio_var = w @ expected_cov @ w
    expected_portfolio_vol = math.sqrt(expected_portfolio_var)
    expected_mcr = (expected_cov @ w) / expected_portfolio_vol
    expected_ccr = w * expected_mcr
    expected_pcr = expected_ccr / expected_portfolio_vol

    result = portfolio.calculate_risk_contribution(returns, weights)

    assert list(result.index) == ["A", "B"]
    assert list(result.columns) == [
        "Weight",
        "Marginal Risk Contribution",
        "Risk Contribution",
        "Risk Contribution %",
    ]
    assert result["Weight"].tolist() == pytest.approx([0.5, 0.5])
    assert result["Marginal Risk Contribution"].tolist() == pytest.approx(expected_mcr.tolist())
    assert result["Risk Contribution"].tolist() == pytest.approx(expected_ccr.tolist())
    assert result["Risk Contribution %"].tolist() == pytest.approx(expected_pcr.tolist())
    assert result["Risk Contribution %"].sum() == pytest.approx(1.0)


# --- 15. Portfolio summary structure -----------------------------------------------

def test_create_portfolio_summary_structure():
    n_days = 60
    rng = np.random.default_rng(11)
    columns = list(TICKERS) + [BENCHMARK]
    daily_moves = rng.normal(loc=0.0005, scale=0.01, size=(n_days, len(columns)))
    prices = 100 * (1 + pd.DataFrame(daily_moves, columns=columns, index=_dates(n_days))).cumprod()
    returns = prices.pct_change(fill_method=None).iloc[1:]

    summary = portfolio.create_portfolio_summary(returns)

    assert list(summary.index) == ["Portfolio"]
    assert list(summary.columns) == [
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Max Drawdown",
        "Beta",
    ]
    assert summary.shape == (1, 5)


# --- 16. Zero-volatility handling ------------------------------------------------

def test_zero_volatility_portfolio_produces_nan_sharpe():
    portfolio_returns = pd.Series([0.0, 0.0, 0.0, 0.0], index=_dates(4), name="Portfolio")

    volatility = portfolio.calculate_portfolio_annualized_volatility(portfolio_returns)
    assert volatility == pytest.approx(0.0)

    sharpe = portfolio.calculate_portfolio_sharpe_ratio(portfolio_returns)
    assert math.isnan(sharpe)


# --- 17. Zero benchmark variance handling ----------------------------------------

def test_calculate_portfolio_beta_zero_benchmark_variance_raises():
    portfolio_returns = pd.Series([0.01, 0.02, -0.01, 0.00], index=_dates(4), name="Portfolio")
    benchmark_returns = pd.Series([0.01, 0.01, 0.01, 0.01], index=_dates(4), name=BENCHMARK)

    with pytest.raises(ValueError):
        portfolio.calculate_portfolio_beta(portfolio_returns, benchmark_returns)
