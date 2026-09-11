"""
Aggregates individual asset data into a weighted multi-asset portfolio and
computes portfolio-level performance and risk statistics relative to the
S&P 500 benchmark. Reuses src.metrics for the underlying formulas rather
than duplicating them; this module does not download or cache market data.
"""

import math

import numpy as np
import pandas as pd

from config import BENCHMARK, PORTFOLIO_WEIGHTS, TICKERS, TRADING_DAYS

from . import metrics

WEIGHT_SUM_TOLERANCE = 1e-6


def validate_portfolio_weights(
    weights: dict = PORTFOLIO_WEIGHTS, required_tickers: list = TICKERS
) -> None:
    """Validate a portfolio weight mapping against the required tickers."""
    if not isinstance(weights, dict):
        raise TypeError("weights must be a dict mapping ticker to weight.")

    required = set(required_tickers)
    provided = set(weights.keys())

    missing = required - provided
    if missing:
        raise ValueError(f"Missing weight(s) for required ticker(s): {sorted(missing)}")

    unexpected = provided - required
    if unexpected:
        raise ValueError(f"Unexpected ticker(s) in weights: {sorted(unexpected)}")

    for ticker, weight in weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float, np.floating, np.integer)):
            raise ValueError(f"Weight for '{ticker}' must be numeric, got {type(weight).__name__}.")
        if not math.isfinite(weight):
            raise ValueError(f"Weight for '{ticker}' must be finite, got {weight}.")
        if weight < 0:
            raise ValueError(f"Weight for '{ticker}' must be non-negative, got {weight}.")

    total = sum(weights.values())
    if not math.isclose(total, 1.0, rel_tol=WEIGHT_SUM_TOLERANCE, abs_tol=WEIGHT_SUM_TOLERANCE):
        raise ValueError(f"Portfolio weights must sum to 1.0, got {total:.6f}.")


def _validate_return_columns(returns: pd.DataFrame, required_columns: list) -> None:
    """Validate that a return DataFrame contains clean, numeric data for the required columns."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")

    missing = [c for c in required_columns if c not in returns.columns]
    if missing:
        raise ValueError(f"Returns data is missing required columns: {missing}")

    subset = returns[required_columns]
    non_numeric = [c for c in required_columns if not pd.api.types.is_numeric_dtype(subset[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric return columns: {non_numeric}")

    if subset.isna().any().any():
        raise ValueError("Returns data contains NaN values in required columns.")

    if np.isinf(subset.to_numpy()).any():
        raise ValueError("Returns data contains infinite values in required columns.")

    if len(subset) < 2:
        raise ValueError(f"At least 2 return observations are required, got {len(subset)}.")


def calculate_portfolio_returns(returns: pd.DataFrame, weights: dict = PORTFOLIO_WEIGHTS) -> pd.Series:
    """Calculate weighted daily portfolio returns from individual stock returns."""
    tickers = list(weights.keys())
    validate_portfolio_weights(weights, tickers)
    _validate_return_columns(returns, tickers)

    weight_vector = np.array([weights[t] for t in tickers])
    portfolio_values = returns[tickers].to_numpy() @ weight_vector

    return pd.Series(portfolio_values, index=returns.index, name="Portfolio")


def _as_portfolio_frame(portfolio_returns: pd.Series) -> pd.DataFrame:
    """Wrap a portfolio return Series as a one-column DataFrame for reuse with src.metrics."""
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series.")
    return portfolio_returns.rename("Portfolio").to_frame()


def calculate_portfolio_cumulative_return(portfolio_returns: pd.Series) -> pd.Series:
    """Calculate the cumulative compounded return series for the portfolio."""
    frame = _as_portfolio_frame(portfolio_returns)
    return metrics.calculate_cumulative_returns(frame)["Portfolio"]


def calculate_portfolio_annualized_return(portfolio_returns: pd.Series) -> float:
    """Calculate the geometric annualized return of the portfolio."""
    frame = _as_portfolio_frame(portfolio_returns)
    return float(metrics.calculate_annualized_return(frame)["Portfolio"])


def calculate_portfolio_annualized_volatility(portfolio_returns: pd.Series) -> float:
    """Calculate the annualized volatility of the portfolio."""
    frame = _as_portfolio_frame(portfolio_returns)
    return float(metrics.calculate_annualized_volatility(frame)["Portfolio"])


def calculate_portfolio_sharpe_ratio(portfolio_returns: pd.Series) -> float:
    """Calculate the annualized Sharpe ratio of the portfolio."""
    frame = _as_portfolio_frame(portfolio_returns)
    return float(metrics.calculate_sharpe_ratio(frame)["Portfolio"])


def calculate_portfolio_drawdown(portfolio_returns: pd.Series) -> pd.Series:
    """Calculate the full portfolio drawdown series."""
    frame = _as_portfolio_frame(portfolio_returns)
    return metrics.calculate_drawdown(frame)["Portfolio"]


def calculate_portfolio_max_drawdown(portfolio_returns: pd.Series) -> float:
    """Calculate the maximum (most negative) historical portfolio drawdown."""
    frame = _as_portfolio_frame(portfolio_returns)
    return float(metrics.calculate_max_drawdown(frame)["Portfolio"])


def calculate_portfolio_beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate portfolio beta against a benchmark return series, aligned by date."""
    if not isinstance(portfolio_returns, pd.Series):
        raise TypeError("portfolio_returns must be a pandas Series.")
    if not isinstance(benchmark_returns, pd.Series):
        raise TypeError("benchmark_returns must be a pandas Series.")

    benchmark_name = benchmark_returns.name if benchmark_returns.name else BENCHMARK
    combined = pd.concat(
        [portfolio_returns.rename("Portfolio"), benchmark_returns.rename(benchmark_name)],
        axis=1,
        join="inner",
    ).dropna()

    if combined.empty:
        raise ValueError(
            "No overlapping, non-missing dates between portfolio and benchmark returns."
        )

    beta = metrics.calculate_beta(combined, benchmark=benchmark_name)
    return float(beta["Portfolio"])


def create_portfolio_comparison(returns: pd.DataFrame, weights: dict = PORTFOLIO_WEIGHTS) -> pd.DataFrame:
    """Build a Portfolio vs. S&P 500 comparison table of return and risk metrics."""
    required_columns = list(weights.keys()) + [BENCHMARK]
    _validate_return_columns(returns, required_columns)

    portfolio_returns = calculate_portfolio_returns(returns, weights)
    benchmark_returns = returns[BENCHMARK]
    benchmark_frame = returns[[BENCHMARK]]

    comparison = pd.DataFrame(
        {
            "Annualized Return": [
                calculate_portfolio_annualized_return(portfolio_returns),
                metrics.calculate_annualized_return(benchmark_frame)[BENCHMARK],
            ],
            "Annualized Volatility": [
                calculate_portfolio_annualized_volatility(portfolio_returns),
                metrics.calculate_annualized_volatility(benchmark_frame)[BENCHMARK],
            ],
            "Sharpe Ratio": [
                calculate_portfolio_sharpe_ratio(portfolio_returns),
                metrics.calculate_sharpe_ratio(benchmark_frame)[BENCHMARK],
            ],
            "Max Drawdown": [
                calculate_portfolio_max_drawdown(portfolio_returns),
                metrics.calculate_max_drawdown(benchmark_frame)[BENCHMARK],
            ],
            "Beta": [
                calculate_portfolio_beta(portfolio_returns, benchmark_returns),
                1.0,
            ],
        },
        index=["Portfolio", "S&P 500"],
    )
    return comparison


def calculate_risk_contribution(returns: pd.DataFrame, weights: dict = PORTFOLIO_WEIGHTS) -> pd.DataFrame:
    """Calculate each asset's marginal, component, and percentage contribution to portfolio risk."""
    tickers = list(weights.keys())
    validate_portfolio_weights(weights, tickers)
    _validate_return_columns(returns, tickers)

    weight_vector = np.array([weights[t] for t in tickers])
    annualized_covariance = returns[tickers].cov().to_numpy() * TRADING_DAYS

    portfolio_variance = weight_vector @ annualized_covariance @ weight_vector
    if portfolio_variance <= 0:
        raise ValueError("Portfolio variance is zero or negative; risk contribution is undefined.")
    portfolio_volatility = math.sqrt(portfolio_variance)

    marginal_contribution = (annualized_covariance @ weight_vector) / portfolio_volatility
    component_contribution = weight_vector * marginal_contribution
    percent_contribution = component_contribution / portfolio_volatility

    total_percent = float(percent_contribution.sum())
    if not math.isclose(total_percent, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Risk contribution percentages sum to {total_percent:.6f}, expected ~1.0."
        )

    return pd.DataFrame(
        {
            "Weight": weight_vector,
            "Marginal Risk Contribution": marginal_contribution,
            "Risk Contribution": component_contribution,
            "Risk Contribution %": percent_contribution,
        },
        index=tickers,
    )


def create_portfolio_summary(returns: pd.DataFrame, weights: dict = PORTFOLIO_WEIGHTS) -> pd.DataFrame:
    """Build a one-row summary of portfolio-level return and risk metrics."""
    required_columns = list(weights.keys()) + [BENCHMARK]
    _validate_return_columns(returns, required_columns)

    portfolio_returns = calculate_portfolio_returns(returns, weights)
    benchmark_returns = returns[BENCHMARK]

    summary = pd.DataFrame(
        {
            "Annualized Return": [calculate_portfolio_annualized_return(portfolio_returns)],
            "Annualized Volatility": [calculate_portfolio_annualized_volatility(portfolio_returns)],
            "Sharpe Ratio": [calculate_portfolio_sharpe_ratio(portfolio_returns)],
            "Max Drawdown": [calculate_portfolio_max_drawdown(portfolio_returns)],
            "Beta": [calculate_portfolio_beta(portfolio_returns, benchmark_returns)],
        },
        index=["Portfolio"],
    )
    return summary
