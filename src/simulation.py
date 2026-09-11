"""
Runs Monte Carlo simulations of future portfolio value paths based on
historical daily return and volatility characteristics of the weighted
portfolio.

IMPORTANT INTERPRETATION - read before using any output from this module:

- This is scenario analysis derived from HISTORICAL mean and volatility,
  not a price prediction and not investment advice.
- Daily portfolio returns are assumed to be independently and identically
  normally distributed, parameterized by the sample mean and sample
  standard deviation of historical daily portfolio returns.
- Real markets exhibit fat tails, volatility clustering, regime changes,
  and structural breaks that a normal-distribution random walk does not
  capture. Extreme scenarios are therefore likely under-represented.
- All outputs (terminal returns, VaR, CVaR, downside probabilities, etc.)
  describe the simulated distribution under these assumptions, not a
  forecast of what the portfolio will actually do.
"""

import math

import numpy as np
import pandas as pd

from config import MONTE_CARLO_DAYS, MONTE_CARLO_SIMULATIONS, PORTFOLIO_WEIGHTS, RANDOM_SEED

from . import portfolio


def _validate_positive_int(name: str, value) -> None:
    """Validate that a value is a positive (non-bool) integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")


def _validate_seed(seed) -> None:
    """Validate that a random seed is an integer."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}.")


def _validate_initial_value(initial_value) -> None:
    """Validate that an initial portfolio value is a positive, finite number."""
    if isinstance(initial_value, bool) or not isinstance(
        initial_value, (int, float, np.floating, np.integer)
    ):
        raise ValueError(f"initial_value must be numeric, got {type(initial_value).__name__}.")
    if not math.isfinite(initial_value):
        raise ValueError(f"initial_value must be finite, got {initial_value}.")
    if initial_value <= 0:
        raise ValueError(f"initial_value must be positive, got {initial_value}.")


def _validate_confidence(confidence) -> None:
    """Validate that a confidence level is a number strictly between 0 and 1."""
    if isinstance(confidence, bool) or not isinstance(
        confidence, (int, float, np.floating, np.integer)
    ):
        raise ValueError(f"confidence must be numeric, got {type(confidence).__name__}.")
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be strictly between 0 and 1, got {confidence}.")


def _validate_simulation_frame(frame: pd.DataFrame, label: str) -> None:
    """Validate a simulated return/wealth path DataFrame."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} must be a pandas DataFrame.")
    if frame.empty:
        raise ValueError(f"{label} is empty.")

    non_numeric = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise ValueError(f"{label} contains non-numeric columns: {non_numeric}")

    if frame.isna().any().any():
        raise ValueError(f"{label} contains NaN values.")

    if np.isinf(frame.to_numpy()).any():
        raise ValueError(f"{label} contains infinite values.")


def _validate_terminal_returns(terminal_returns: pd.Series) -> None:
    """Validate a Series of per-simulation terminal returns."""
    if not isinstance(terminal_returns, pd.Series):
        raise TypeError("terminal_returns must be a pandas Series.")
    if terminal_returns.empty:
        raise ValueError("terminal_returns is empty.")
    if not pd.api.types.is_numeric_dtype(terminal_returns):
        raise ValueError("terminal_returns must be numeric.")
    if terminal_returns.isna().any():
        raise ValueError("terminal_returns contains NaN values.")
    if np.isinf(terminal_returns.to_numpy()).any():
        raise ValueError("terminal_returns contains infinite values.")


def estimate_portfolio_parameters(returns: pd.DataFrame, weights: dict = PORTFOLIO_WEIGHTS) -> pd.Series:
    """Estimate historical DAILY portfolio mean return and volatility (not annualized).

    Uses the existing weighted portfolio return methodology
    (src.portfolio.calculate_portfolio_returns), which also validates the
    input return data (required columns, numeric, no NaN/inf, sufficient
    observations).
    """
    portfolio_returns = portfolio.calculate_portfolio_returns(returns, weights)

    mean_daily_return = float(portfolio_returns.mean())
    daily_volatility = float(portfolio_returns.std(ddof=1))

    if not math.isfinite(mean_daily_return) or not math.isfinite(daily_volatility):
        raise ValueError("Estimated portfolio parameters are not finite.")

    return pd.Series({"Mean Daily Return": mean_daily_return, "Daily Volatility": daily_volatility})


def simulate_portfolio_paths(
    returns: pd.DataFrame,
    simulations: int = MONTE_CARLO_SIMULATIONS,
    days: int = MONTE_CARLO_DAYS,
    seed: int = RANDOM_SEED,
    weights: dict = PORTFOLIO_WEIGHTS,
) -> pd.DataFrame:
    """Simulate daily portfolio return paths from a normal distribution.

    The distribution is parameterized directly by the historical weighted
    PORTFOLIO's daily mean return and volatility (estimated via
    estimate_portfolio_parameters) - individual stocks are not simulated
    separately and combined. Uses numpy.random.default_rng(seed) for a
    reproducible, locally-scoped random generator (no global random-state
    mutation), so the same seed always reproduces the same paths.

    If the historical daily volatility is exactly 0 (a degenerate/constant
    historical return series), every simulated path collapses to the
    deterministic mean daily return - this is expected behavior for that
    edge case, not an error.

    Returns a DataFrame of shape (days, simulations): rows are simulation
    days, columns are simulation numbers.
    """
    _validate_positive_int("simulations", simulations)
    _validate_positive_int("days", days)
    _validate_seed(seed)

    params = estimate_portfolio_parameters(returns, weights)
    mean_daily_return = params["Mean Daily Return"]
    daily_volatility = params["Daily Volatility"]

    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(loc=mean_daily_return, scale=daily_volatility, size=(days, simulations))

    columns = [f"Simulation_{i + 1}" for i in range(simulations)]
    index = pd.RangeIndex(1, days + 1, name="Day")
    return pd.DataFrame(simulated_returns, index=index, columns=columns)


def calculate_simulated_wealth(return_paths: pd.DataFrame, initial_value: float = 1.0) -> pd.DataFrame:
    """Compound simulated daily return paths into wealth paths."""
    _validate_simulation_frame(return_paths, "return_paths")
    _validate_initial_value(initial_value)

    return initial_value * (1 + return_paths).cumprod()


def calculate_terminal_returns(wealth_paths: pd.DataFrame, initial_value: float = 1.0) -> pd.Series:
    """Calculate the terminal (final-day) return for every simulation."""
    _validate_simulation_frame(wealth_paths, "wealth_paths")
    _validate_initial_value(initial_value)

    terminal_returns = wealth_paths.iloc[-1] / initial_value - 1
    terminal_returns.name = "Terminal Return"
    return terminal_returns


def create_simulation_summary(terminal_returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Summarize the simulated terminal-return and terminal-wealth distribution."""
    _validate_terminal_returns(terminal_returns)
    _validate_initial_value(initial_value)

    terminal_wealth = initial_value * (1 + terminal_returns)

    return pd.Series(
        {
            "Mean Terminal Return": float(terminal_returns.mean()),
            "Median Terminal Return": float(terminal_returns.median()),
            "5th Percentile Return": float(terminal_returns.quantile(0.05)),
            "95th Percentile Return": float(terminal_returns.quantile(0.95)),
            "Probability of Loss": float((terminal_returns < 0).mean()),
            "Expected Terminal Wealth": float(terminal_wealth.mean()),
            "5th Percentile Wealth": float(terminal_wealth.quantile(0.05)),
            "95th Percentile Wealth": float(terminal_wealth.quantile(0.95)),
        }
    )


def calculate_var(terminal_returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Value at Risk as a positive loss magnitude.

    VaR = max(0, -quantile) where quantile is the (1 - confidence) quantile
    of terminal returns. E.g. a 5th-percentile terminal return of -0.20 at
    95% confidence gives VaR = 0.20. A positive percentile (no loss at that
    confidence level) gives VaR = 0.0 rather than a misleading negative
    number.
    """
    _validate_terminal_returns(terminal_returns)
    _validate_confidence(confidence)

    quantile_return = terminal_returns.quantile(1 - confidence)
    return float(max(0.0, -quantile_return))


def calculate_cvar(terminal_returns: pd.Series, confidence: float = 0.95) -> float:
    """Calculate Conditional VaR (Expected Shortfall) as a positive loss magnitude.

    CVaR is the average terminal return among observations at or below the
    (1 - confidence) quantile, reported as max(0, -mean_tail_return) - the
    same positive-loss convention as calculate_var. E.g. a mean tail return
    of -0.28 gives CVaR = 0.28.
    """
    _validate_terminal_returns(terminal_returns)
    _validate_confidence(confidence)

    threshold = terminal_returns.quantile(1 - confidence)
    tail = terminal_returns[terminal_returns <= threshold]

    if tail.empty:
        # Degenerate case (can only occur with a very small/unusual sample):
        # no observations at or below the threshold. Fall back to VaR.
        return calculate_var(terminal_returns, confidence)

    mean_tail_return = float(tail.mean())
    return float(max(0.0, -mean_tail_return))


def calculate_downside_probabilities(terminal_returns: pd.Series) -> pd.Series:
    """Calculate scenario frequencies of loss at several thresholds (not forecasts)."""
    _validate_terminal_returns(terminal_returns)

    return pd.Series(
        {
            "Probability of Loss": float((terminal_returns < 0).mean()),
            "Probability Return < -10%": float((terminal_returns < -0.10).mean()),
            "Probability Return < -20%": float((terminal_returns < -0.20).mean()),
            "Probability Return < -30%": float((terminal_returns < -0.30).mean()),
        }
    )


def run_monte_carlo_simulation(
    returns: pd.DataFrame,
    simulations: int = MONTE_CARLO_SIMULATIONS,
    days: int = MONTE_CARLO_DAYS,
    seed: int = RANDOM_SEED,
    weights: dict = PORTFOLIO_WEIGHTS,
    initial_value: float = 1.0,
    confidence: float = 0.95,
) -> dict:
    """Orchestrate the full Monte Carlo scenario analysis end to end.

    Returns a plain dict (deliberately not a class) containing:
    return_paths, wealth_paths, terminal_returns, summary,
    downside_probabilities, var, cvar.

    See the module docstring for the methodology and its limitations -
    this is historical scenario analysis, not a forecast.
    """
    return_paths = simulate_portfolio_paths(returns, simulations, days, seed, weights)
    wealth_paths = calculate_simulated_wealth(return_paths, initial_value)
    terminal_returns = calculate_terminal_returns(wealth_paths, initial_value)
    summary = create_simulation_summary(terminal_returns, initial_value)
    downside_probabilities = calculate_downside_probabilities(terminal_returns)
    var = calculate_var(terminal_returns, confidence)
    cvar = calculate_cvar(terminal_returns, confidence)

    return {
        "return_paths": return_paths,
        "wealth_paths": wealth_paths,
        "terminal_returns": terminal_returns,
        "summary": summary,
        "downside_probabilities": downside_probabilities,
        "var": var,
        "cvar": cvar,
    }
