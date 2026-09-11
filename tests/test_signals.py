"""
Unit tests for src/signals.py. All price data is small and deterministic;
no live Yahoo Finance data is used. Expected values are computed
independently from the documented formulas, not by re-calling the
functions under test.
"""

import math

import numpy as np
import pandas as pd
import pytest

from config import TICKERS
from src import signals

# A single crafted NVDA price path with exactly one Death Cross (index 10)
# and one Golden Cross (index 15) under short_window=3, long_window=5,
# verified independently with pandas .rolling().mean() before being
# hard-coded here. Other tickers are held flat so they stay Neutral/
# crossover-free and don't interfere with NVDA-specific assertions.
NVDA_PATH = [100, 100, 100, 100, 105, 110, 115, 120, 115, 108, 100, 95, 90, 95, 100, 105, 110]
SHORT_WINDOW = 3
LONG_WINDOW = 5
DEATH_CROSS_IDX = 10
GOLDEN_CROSS_IDX = 15


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n)


def _prices_for(primary_values, primary_ticker: str = "NVDA", filler: float = 100.0) -> pd.DataFrame:
    n = len(primary_values)
    data = {t: [filler] * n for t in TICKERS}
    data[primary_ticker] = list(primary_values)
    return pd.DataFrame(data, index=_dates(n))[TICKERS]


PRICES = _prices_for(NVDA_PATH)


# --- 1. Moving-average calculation ------------------------------------------

def test_calculate_moving_averages():
    result = signals.calculate_moving_averages(PRICES, SHORT_WINDOW, LONG_WINDOW)

    expected_short = pd.Series(NVDA_PATH).rolling(SHORT_WINDOW).mean()
    expected_long = pd.Series(NVDA_PATH).rolling(LONG_WINDOW).mean()

    assert isinstance(result.columns, pd.MultiIndex)
    got_short = result[("NVDA", "MA_Short")].reset_index(drop=True)
    got_long = result[("NVDA", "MA_Long")].reset_index(drop=True)
    assert got_short.equals(expected_short)
    assert got_long.equals(expected_long)
    assert result[("NVDA", "Price")].tolist() == NVDA_PATH


# --- 2. Initial long-MA NaNs remain NaN --------------------------------------

def test_initial_long_ma_nans_not_backfilled():
    result = signals.calculate_moving_averages(PRICES, SHORT_WINDOW, LONG_WINDOW)
    long_ma = result[("NVDA", "MA_Long")]

    assert long_ma.iloc[: LONG_WINDOW - 1].isna().all()
    assert long_ma.iloc[LONG_WINDOW - 1 :].notna().all()


# --- 3. Invalid windows raise --------------------------------------------------

def test_invalid_windows_raise():
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(PRICES, short_window=5, long_window=5)
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(PRICES, short_window=0, long_window=5)
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(PRICES, short_window=-1, long_window=5)
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(PRICES, short_window=10, long_window=5)


# --- 4. Insufficient history raises --------------------------------------------

def test_insufficient_history_raises():
    short_prices = PRICES.iloc[: LONG_WINDOW - 1]
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(short_prices, SHORT_WINDOW, LONG_WINDOW)


# --- 5, 6, 7. Trend state: Bullish, Bearish, Unavailable -----------------------

def test_trend_state_bullish_bearish_and_unavailable():
    moving_averages = signals.calculate_moving_averages(PRICES, SHORT_WINDOW, LONG_WINDOW)
    trend = signals.calculate_trend_state(moving_averages)

    # Before the long MA exists (indices 0..LONG_WINDOW-2): Unavailable.
    assert (trend["NVDA"].iloc[: LONG_WINDOW - 1] == "Unavailable").all()

    # Index 8: short=116.67 > long=113.0 -> Bullish.
    assert trend["NVDA"].iloc[8] == "Bullish"

    # Index 10 (the Death Cross day itself): short=107.67 < long=111.6 -> Bearish.
    assert trend["NVDA"].iloc[DEATH_CROSS_IDX] == "Bearish"


# --- 8. Golden Cross detection --------------------------------------------------

def test_golden_cross_detection():
    moving_averages = signals.calculate_moving_averages(PRICES, SHORT_WINDOW, LONG_WINDOW)
    events = signals.detect_crossovers(moving_averages)

    golden_events = events[(events["Ticker"] == "NVDA") & (events["Signal"] == "Golden Cross")]
    assert len(golden_events) == 1
    assert golden_events.iloc[0]["Date"] == PRICES.index[GOLDEN_CROSS_IDX]
    assert golden_events.iloc[0]["Price"] == NVDA_PATH[GOLDEN_CROSS_IDX]


# --- 9. Death Cross detection ---------------------------------------------------

def test_death_cross_detection():
    moving_averages = signals.calculate_moving_averages(PRICES, SHORT_WINDOW, LONG_WINDOW)
    events = signals.detect_crossovers(moving_averages)

    death_events = events[(events["Ticker"] == "NVDA") & (events["Signal"] == "Death Cross")]
    assert len(death_events) == 1
    assert death_events.iloc[0]["Date"] == PRICES.index[DEATH_CROSS_IDX]
    assert death_events.iloc[0]["Price"] == NVDA_PATH[DEATH_CROSS_IDX]


# --- 10. No false repeated crossover events -------------------------------------

def test_no_repeated_crossover_events_while_trend_persists():
    moving_averages = signals.calculate_moving_averages(PRICES, SHORT_WINDOW, LONG_WINDOW)
    events = signals.detect_crossovers(moving_averages)

    nvda_events = events[events["Ticker"] == "NVDA"]
    # Exactly one Death Cross and one Golden Cross, not one per bullish/bearish day.
    assert len(nvda_events) == 2
    assert sorted(nvda_events["Signal"].tolist()) == ["Death Cross", "Golden Cross"]

    # Flat filler tickers never cross (short == long whenever both exist).
    flat_events = events[events["Ticker"] == "JPM"]
    assert flat_events.empty


# --- 11. Signal summary structure -----------------------------------------------

def test_create_signal_summary_structure():
    summary = signals.create_signal_summary(PRICES, SHORT_WINDOW, LONG_WINDOW)

    assert list(summary.index) == list(TICKERS)
    assert list(summary.columns) == [
        "Latest Price",
        f"MA{SHORT_WINDOW}",
        f"MA{LONG_WINDOW}",
        "Trend",
        "Last Crossover",
        "Last Crossover Date",
    ]
    assert summary.loc["NVDA", "Last Crossover"] == "Golden Cross"
    assert summary.loc["NVDA", "Last Crossover Date"] == PRICES.index[GOLDEN_CROSS_IDX]
    assert summary.loc["JPM", "Last Crossover"] == "No crossover in sample"
    assert pd.isna(summary.loc["JPM", "Last Crossover Date"])


# --- 12. Backtest position logic ------------------------------------------------

def test_backtest_position_logic():
    backtest = signals.backtest_moving_average_signal(PRICES, "NVDA", SHORT_WINDOW, LONG_WINDOW)

    assert backtest["Position"].iloc[: LONG_WINDOW - 1].tolist() == [0.0] * (LONG_WINDOW - 1)
    assert backtest["Position"].iloc[8] == 1.0  # Bullish day
    assert backtest["Position"].iloc[DEATH_CROSS_IDX] == 0.0  # Bearish day
    assert set(backtest["Position"].unique().tolist()) <= {0.0, 1.0}


# --- 13. One-day shift prevents look-ahead bias ---------------------------------

def test_backtest_shift_prevents_lookahead():
    backtest = signals.backtest_moving_average_signal(PRICES, "NVDA", SHORT_WINDOW, LONG_WINDOW)

    # Day 10 itself is Bearish (position=0), but its strategy return must use
    # day 9's position (Bullish, =1) rather than day 10's own position.
    price_day9 = NVDA_PATH[9]
    price_day10 = NVDA_PATH[10]
    expected_day10_strategy_return = 1.0 * ((price_day10 - price_day9) / price_day9)

    assert backtest["Strategy Return"].iloc[DEATH_CROSS_IDX] == pytest.approx(
        expected_day10_strategy_return
    )
    # Confirms this is NOT simply position[t] * return[t] (which would give 0).
    assert backtest["Strategy Return"].iloc[DEATH_CROSS_IDX] != 0.0


# --- 14. Strategy return calculation --------------------------------------------

def test_strategy_return_matches_shifted_position_times_return():
    backtest = signals.backtest_moving_average_signal(PRICES, "NVDA", SHORT_WINDOW, LONG_WINDOW)

    expected = backtest["Position"].shift(1).fillna(0.0) * backtest["Buy & Hold Return"]
    pd.testing.assert_series_equal(
        backtest["Strategy Return"], expected, check_names=False
    )


# --- 15. Wealth compounding -----------------------------------------------------

def test_wealth_compounding():
    backtest = signals.backtest_moving_average_signal(PRICES, "NVDA", SHORT_WINDOW, LONG_WINDOW)

    expected_buy_hold_wealth = (1 + backtest["Buy & Hold Return"].fillna(0.0)).cumprod()
    expected_strategy_wealth = (1 + backtest["Strategy Return"].fillna(0.0)).cumprod()

    assert backtest["Buy & Hold Wealth"].tolist() == pytest.approx(expected_buy_hold_wealth.tolist())
    assert backtest["Strategy Wealth"].tolist() == pytest.approx(expected_strategy_wealth.tolist())
    assert backtest["Buy & Hold Wealth"].iloc[0] == pytest.approx(1.0)


# --- 16. Backtest summary structure ----------------------------------------------

def test_create_backtest_summary_structure():
    summary = signals.create_backtest_summary(PRICES, SHORT_WINDOW, LONG_WINDOW)

    assert list(summary.index) == list(TICKERS)
    assert list(summary.columns) == [
        "Buy & Hold Total Return",
        "Strategy Total Return",
        "Buy & Hold Annualized Return",
        "Strategy Annualized Return",
        "Buy & Hold Annualized Volatility",
        "Strategy Annualized Volatility",
        "Buy & Hold Max Drawdown",
        "Strategy Max Drawdown",
    ]
    assert all(pd.api.types.is_numeric_dtype(summary[c]) for c in summary.columns)


# --- 17. Missing ticker raises ----------------------------------------------------

def test_missing_ticker_column_raises():
    incomplete_prices = PRICES.drop(columns=["JPM"])
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(incomplete_prices, SHORT_WINDOW, LONG_WINDOW)


def test_invalid_ticker_in_backtest_raises():
    with pytest.raises(ValueError):
        signals.backtest_moving_average_signal(PRICES, "NOT_A_TICKER", SHORT_WINDOW, LONG_WINDOW)


# --- 18. Malformed price data raises ----------------------------------------------

def test_non_positive_price_raises():
    bad_prices = PRICES.copy()
    bad_prices.iloc[0, bad_prices.columns.get_loc("NVDA")] = -5.0
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(bad_prices, SHORT_WINDOW, LONG_WINDOW)


def test_unsorted_dates_raise():
    shuffled = PRICES.iloc[::-1]
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(shuffled, SHORT_WINDOW, LONG_WINDOW)


def test_nan_price_raises():
    bad_prices = PRICES.copy()
    bad_prices.iloc[3, bad_prices.columns.get_loc("NVDA")] = np.nan
    with pytest.raises(ValueError):
        signals.calculate_moving_averages(bad_prices, SHORT_WINDOW, LONG_WINDOW)
