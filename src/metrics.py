"""
Computes return and risk metrics for individual assets, including daily and
cumulative returns, volatility, Sharpe ratio, beta, drawdowns, and
correlation analysis. Operates purely on pandas Series/DataFrames supplied
by the caller; this module does not download or cache market data.
"""

import numpy as np
import pandas as pd

from config import BENCHMARK, RISK_FREE_RATE, TICKERS, TRADING_DAYS

MIN_OBSERVATIONS = 2


def _validate_prices(prices: pd.DataFrame) -> None:
    """Validate a raw price DataFrame before computing returns."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty:
        raise ValueError("Price DataFrame is empty.")

    non_numeric = [c for c in prices.columns if not pd.api.types.is_numeric_dtype(prices[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric price columns: {non_numeric}")

    if len(prices) < MIN_OBSERVATIONS:
        raise ValueError(
            f"At least {MIN_OBSERVATIONS} price observations are required, "
            f"got {len(prices)}."
        )


def _validate_returns(returns: pd.DataFrame) -> None:
    """Validate a return DataFrame before computing downstream metrics."""
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")
    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")

    non_numeric = [c for c in returns.columns if not pd.api.types.is_numeric_dtype(returns[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric return columns: {non_numeric}")

    if returns.isna().any().any():
        raise ValueError("Returns DataFrame contains NaN values.")

    if len(returns) < MIN_OBSERVATIONS:
        raise ValueError(
            f"At least {MIN_OBSERVATIONS} return observations are required, "
            f"got {len(returns)}."
        )


def calculate_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate simple daily percentage returns from a price DataFrame."""
    _validate_prices(prices)
    returns = prices.pct_change(fill_method=None)
    return returns.iloc[1:]


def calculate_cumulative_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative compounded percentage returns from daily returns."""
    _validate_returns(returns)
    return (1 + returns).cumprod() - 1


def calculate_annualized_return(returns: pd.DataFrame) -> pd.Series:
    """Calculate geometric (CAGR-style) annualized return per asset."""
    _validate_returns(returns)
    n_observations = len(returns)
    total_return = (1 + returns).prod() - 1
    annualized = (1 + total_return) ** (TRADING_DAYS / n_observations) - 1
    annualized.name = "Annualized Return"
    return annualized


def calculate_annualized_volatility(returns: pd.DataFrame) -> pd.Series:
    """Calculate annualized volatility from the sample standard deviation of daily returns."""
    _validate_returns(returns)
    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    volatility.name = "Annualized Volatility"
    return volatility


def calculate_sharpe_ratio(returns: pd.DataFrame) -> pd.Series:
    """Calculate the annualized Sharpe ratio per asset, using the configured risk-free rate."""
    _validate_returns(returns)
    annualized_return = calculate_annualized_return(returns)
    annualized_volatility = calculate_annualized_volatility(returns)

    safe_volatility = annualized_volatility.replace(0, np.nan)
    sharpe = (annualized_return - RISK_FREE_RATE) / safe_volatility
    sharpe.name = "Sharpe Ratio"
    return sharpe


def calculate_drawdown(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the full drawdown series (wealth relative to running peak) per asset."""
    _validate_returns(returns)
    wealth = (1 + returns).cumprod()
    running_peak = wealth.cummax()
    return wealth / running_peak - 1


def calculate_max_drawdown(returns: pd.DataFrame) -> pd.Series:
    """Calculate the maximum (most negative) historical drawdown per asset."""
    drawdown = calculate_drawdown(returns)
    max_dd = drawdown.min()
    max_dd.name = "Max Drawdown"
    return max_dd


def calculate_beta(returns: pd.DataFrame, benchmark: str = BENCHMARK) -> pd.Series:
    """Calculate each asset's beta against the benchmark using sample covariance/variance."""
    _validate_returns(returns)
    if benchmark not in returns.columns:
        raise ValueError(f"Benchmark column '{benchmark}' not found in returns data.")

    benchmark_returns = returns[benchmark]
    benchmark_variance = benchmark_returns.var()
    if benchmark_variance == 0:
        raise ValueError(f"Benchmark '{benchmark}' has zero variance; beta is undefined.")

    asset_columns = [c for c in returns.columns if c != benchmark]
    covariances = returns[asset_columns].apply(lambda col: col.cov(benchmark_returns))
    beta = covariances / benchmark_variance
    beta.name = "Beta"
    return beta


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate the Pearson correlation matrix across all return columns supplied."""
    _validate_returns(returns)
    return returns.corr()


def create_metrics_summary(returns: pd.DataFrame) -> pd.DataFrame:
    """Build a per-stock summary table of return and risk metrics, excluding the benchmark."""
    _validate_returns(returns)

    if BENCHMARK not in returns.columns:
        raise ValueError(f"Benchmark column '{BENCHMARK}' not found in returns data.")

    missing_tickers = [t for t in TICKERS if t not in returns.columns]
    if missing_tickers:
        raise ValueError(f"Missing expected tickers in returns data: {missing_tickers}")

    asset_returns = returns[TICKERS]

    summary = pd.DataFrame(
        {
            "Annualized Return": calculate_annualized_return(asset_returns),
            "Annualized Volatility": calculate_annualized_volatility(asset_returns),
            "Sharpe Ratio": calculate_sharpe_ratio(asset_returns),
            "Max Drawdown": calculate_max_drawdown(asset_returns),
            "Beta": calculate_beta(returns, benchmark=BENCHMARK)[TICKERS],
        }
    )
    return summary.loc[TICKERS]
