"""
Unit tests for src/simulation.py. All data is small and deterministic;
no live Yahoo Finance data is used. Expected values are computed
independently from the documented formulas, not by re-calling the
functions under test (VaR/CVaR/downside-probability expectations below
were independently verified with a standalone pandas.Series.quantile()
check before being hard-coded).
"""

import math

import numpy as np
import pandas as pd
import pytest

from src import simulation


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


# --- 1. Parameter estimation --------------------------------------------------

def test_estimate_portfolio_parameters():
    values = [0.01, -0.01, 0.02, -0.02, 0.005]
    returns = pd.DataFrame({"A": values, "B": values}, index=_dates(len(values)))
    weights = {"A": 0.5, "B": 0.5}

    params = simulation.estimate_portfolio_parameters(returns, weights)

    expected_mean = np.mean(values)
    expected_std = np.std(values, ddof=1)
    assert params["Mean Daily Return"] == pytest.approx(expected_mean)
    assert params["Daily Volatility"] == pytest.approx(expected_std)


# --- 2. Simulation output shape --------------------------------------------------

def test_simulate_portfolio_paths_shape():
    values = [0.01, -0.01, 0.02, -0.02, 0.005, 0.01, -0.005]
    returns = pd.DataFrame({"A": values, "B": values}, index=_dates(len(values)))
    weights = {"A": 0.5, "B": 0.5}

    paths = simulation.simulate_portfolio_paths(
        returns, simulations=7, days=4, seed=1, weights=weights
    )

    assert paths.shape == (4, 7)
    assert list(paths.columns) == [f"Simulation_{i + 1}" for i in range(7)]


# --- 3. Same seed produces identical paths --------------------------------------

def test_same_seed_reproducible():
    values = [0.01, -0.01, 0.02, -0.02, 0.005, 0.01, -0.005]
    returns = pd.DataFrame({"A": values, "B": values}, index=_dates(len(values)))
    weights = {"A": 0.5, "B": 0.5}

    paths1 = simulation.simulate_portfolio_paths(returns, simulations=5, days=10, seed=42, weights=weights)
    paths2 = simulation.simulate_portfolio_paths(returns, simulations=5, days=10, seed=42, weights=weights)

    pd.testing.assert_frame_equal(paths1, paths2)


# --- 4. Different seed changes paths ---------------------------------------------

def test_different_seed_changes_paths():
    values = [0.01, -0.01, 0.02, -0.02, 0.005, 0.01, -0.005]
    returns = pd.DataFrame({"A": values, "B": values}, index=_dates(len(values)))
    weights = {"A": 0.5, "B": 0.5}

    paths1 = simulation.simulate_portfolio_paths(returns, simulations=5, days=10, seed=1, weights=weights)
    paths2 = simulation.simulate_portfolio_paths(returns, simulations=5, days=10, seed=2, weights=weights)

    assert not paths1.equals(paths2)


# --- 5. Wealth compounding --------------------------------------------------------

def test_calculate_simulated_wealth():
    return_paths = pd.DataFrame(
        {"Simulation_1": [0.10, -0.05, 0.02], "Simulation_2": [-0.10, 0.05, 0.00]},
        index=pd.RangeIndex(1, 4, name="Day"),
    )

    wealth = simulation.calculate_simulated_wealth(return_paths)

    expected_sim1 = [1.10, 1.10 * 0.95, 1.10 * 0.95 * 1.02]
    expected_sim2 = [0.90, 0.90 * 1.05, 0.90 * 1.05 * 1.00]
    assert wealth["Simulation_1"].tolist() == pytest.approx(expected_sim1)
    assert wealth["Simulation_2"].tolist() == pytest.approx(expected_sim2)


# --- 6. Custom initial value ------------------------------------------------------

def test_calculate_simulated_wealth_custom_initial_value():
    return_paths = pd.DataFrame(
        {"Simulation_1": [0.10, -0.05, 0.02]}, index=pd.RangeIndex(1, 4, name="Day")
    )

    wealth = simulation.calculate_simulated_wealth(return_paths, initial_value=100.0)

    expected = [100 * 1.10, 100 * 1.10 * 0.95, 100 * 1.10 * 0.95 * 1.02]
    assert wealth["Simulation_1"].tolist() == pytest.approx(expected)


# --- 7. Terminal return calculation -----------------------------------------------

def test_calculate_terminal_returns():
    return_paths = pd.DataFrame(
        {"Simulation_1": [0.10, -0.05, 0.02], "Simulation_2": [-0.10, 0.05, 0.00]},
        index=pd.RangeIndex(1, 4, name="Day"),
    )
    wealth = simulation.calculate_simulated_wealth(return_paths)

    terminal = simulation.calculate_terminal_returns(wealth)

    assert terminal["Simulation_1"] == pytest.approx(1.10 * 0.95 * 1.02 - 1)
    assert terminal["Simulation_2"] == pytest.approx(0.90 * 1.05 * 1.00 - 1)


# --- 8. Simulation summary structure ----------------------------------------------

def test_create_simulation_summary_structure():
    terminal_returns = pd.Series([-0.30 + 0.03 * i for i in range(20)])

    summary = simulation.create_simulation_summary(terminal_returns)

    assert list(summary.index) == [
        "Mean Terminal Return",
        "Median Terminal Return",
        "5th Percentile Return",
        "95th Percentile Return",
        "Probability of Loss",
        "Expected Terminal Wealth",
        "5th Percentile Wealth",
        "95th Percentile Wealth",
    ]
    assert summary["Probability of Loss"] == pytest.approx(0.5)


# --- 9. Probability of loss --------------------------------------------------------

def test_probability_of_loss():
    terminal_returns = pd.Series([-0.30 + 0.03 * i for i in range(20)])

    probs = simulation.calculate_downside_probabilities(terminal_returns)

    assert probs["Probability of Loss"] == pytest.approx(0.5)


# --- 10. VaR calculation with known terminal returns -------------------------------

def test_calculate_var_known_values():
    terminal_returns = pd.Series([-0.30 + 0.03 * i for i in range(20)])

    var = simulation.calculate_var(terminal_returns, confidence=0.95)

    assert var == pytest.approx(0.2715, abs=1e-6)


# --- 11. CVaR calculation with known terminal returns -------------------------------

def test_calculate_cvar_known_values():
    terminal_returns = pd.Series([-0.30 + 0.03 * i for i in range(20)])

    cvar = simulation.calculate_cvar(terminal_returns, confidence=0.95)

    assert cvar == pytest.approx(0.3, abs=1e-6)


# --- 12. VaR positive-loss convention -----------------------------------------------

def test_var_positive_loss_convention():
    all_positive_returns = pd.Series([0.01 + 0.01 * i for i in range(20)])

    var = simulation.calculate_var(all_positive_returns, confidence=0.95)

    assert var == pytest.approx(0.0, abs=1e-9)
    assert var >= 0.0


# --- 13. Downside probability calculations ------------------------------------------

def test_downside_probabilities_known_values():
    terminal_returns = pd.Series([-0.30 + 0.03 * i for i in range(20)])

    probs = simulation.calculate_downside_probabilities(terminal_returns)

    assert probs["Probability of Loss"] == pytest.approx(0.5)
    assert probs["Probability Return < -10%"] == pytest.approx(0.35)
    assert probs["Probability Return < -20%"] == pytest.approx(0.2)
    assert probs["Probability Return < -30%"] == pytest.approx(0.0)


# --- 14. Orchestration dictionary structure -----------------------------------------

def test_run_monte_carlo_simulation_structure():
    values = [0.01, -0.01, 0.02, -0.02, 0.005, 0.01, -0.005]
    returns = pd.DataFrame({"A": values, "B": values}, index=_dates(len(values)))
    weights = {"A": 0.5, "B": 0.5}

    result = simulation.run_monte_carlo_simulation(
        returns, simulations=10, days=5, seed=7, weights=weights
    )

    assert set(result.keys()) == {
        "return_paths",
        "wealth_paths",
        "terminal_returns",
        "summary",
        "downside_probabilities",
        "var",
        "cvar",
    }
    assert result["return_paths"].shape == (5, 10)
    assert result["wealth_paths"].shape == (5, 10)
    assert len(result["terminal_returns"]) == 10
    assert isinstance(result["summary"], pd.Series)
    assert isinstance(result["var"], float)
    assert isinstance(result["cvar"], float)


# --- 15. Invalid confidence raises ----------------------------------------------------

def test_invalid_confidence_raises():
    terminal_returns = pd.Series([-0.30 + 0.03 * i for i in range(20)])

    with pytest.raises(ValueError):
        simulation.calculate_var(terminal_returns, confidence=0.0)
    with pytest.raises(ValueError):
        simulation.calculate_var(terminal_returns, confidence=1.0)
    with pytest.raises(ValueError):
        simulation.calculate_var(terminal_returns, confidence=-0.1)
    with pytest.raises(ValueError):
        simulation.calculate_cvar(terminal_returns, confidence=1.5)


# --- 16. Invalid simulations/days raises ------------------------------------------------

def test_invalid_simulations_and_days_raise():
    values = [0.01, -0.01, 0.02, -0.02, 0.005]
    returns = pd.DataFrame({"A": values, "B": values}, index=_dates(len(values)))
    weights = {"A": 0.5, "B": 0.5}

    with pytest.raises(ValueError):
        simulation.simulate_portfolio_paths(returns, simulations=0, days=5, weights=weights)
    with pytest.raises(ValueError):
        simulation.simulate_portfolio_paths(returns, simulations=-1, days=5, weights=weights)
    with pytest.raises(ValueError):
        simulation.simulate_portfolio_paths(returns, simulations=5, days=0, weights=weights)
    with pytest.raises(ValueError):
        simulation.simulate_portfolio_paths(returns, simulations=5, days=5, seed=1.5, weights=weights)


# --- 17. Invalid initial value raises ---------------------------------------------------

def test_invalid_initial_value_raises():
    return_paths = pd.DataFrame({"Simulation_1": [0.01, 0.02]}, index=pd.RangeIndex(1, 3, name="Day"))

    with pytest.raises(ValueError):
        simulation.calculate_simulated_wealth(return_paths, initial_value=0.0)
    with pytest.raises(ValueError):
        simulation.calculate_simulated_wealth(return_paths, initial_value=-5.0)


# --- 18. Zero volatility handling ---------------------------------------------------------

def test_zero_volatility_produces_deterministic_paths():
    constant_values = [0.01] * 10
    returns = pd.DataFrame({"A": constant_values, "B": constant_values}, index=_dates(10))
    weights = {"A": 0.5, "B": 0.5}

    params = simulation.estimate_portfolio_parameters(returns, weights)
    assert params["Daily Volatility"] == pytest.approx(0.0, abs=1e-12)

    paths = simulation.simulate_portfolio_paths(returns, simulations=4, days=3, seed=1, weights=weights)
    assert np.allclose(paths.to_numpy(), 0.01)


# --- 19. Malformed return data raises -------------------------------------------------------

def test_malformed_return_data_raises():
    values = [0.01, -0.01, 0.02, -0.02, 0.005]
    incomplete_returns = pd.DataFrame({"A": values}, index=_dates(len(values)))

    with pytest.raises(ValueError):
        simulation.estimate_portfolio_parameters(incomplete_returns, {"A": 0.5, "B": 0.5})

    nan_returns = pd.DataFrame({"A": values, "B": values}, index=_dates(len(values)))
    nan_returns.iloc[0, 0] = np.nan
    with pytest.raises(ValueError):
        simulation.estimate_portfolio_parameters(nan_returns, {"A": 0.5, "B": 0.5})
