"""
Generates moving-average based technical indicators and crossover trading
signals from historical price data for the individual portfolio stocks.

This module performs HISTORICAL signal analysis only. Nothing here is a
claim that moving-average crossovers predict future returns, and the
optional backtest in this module is a simplified educational exercise,
not an investment recommendation.

Column convention: calculate_moving_averages() returns a DataFrame with a
(Ticker, Field) MultiIndex on the columns, where Field is one of "Price",
"MA_Short", "MA_Long" (generic names, since the window lengths are
parameters rather than fixed at 50/200). create_signal_summary() renders
the short/long moving averages under labels that reflect the actual
window sizes used (e.g. "MA50", "MA200" under the default configuration).
"""

import numpy as np
import pandas as pd

from config import MA_LONG, MA_SHORT, TICKERS

from . import metrics


def _validate_windows(short_window: int, long_window: int) -> None:
    """Validate moving-average window lengths."""
    for name, window in (("short_window", short_window), ("long_window", long_window)):
        if isinstance(window, bool) or not isinstance(window, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer, got {window!r}.")
        if window <= 0:
            raise ValueError(f"{name} must be positive, got {window}.")

    if short_window >= long_window:
        raise ValueError(
            f"short_window ({short_window}) must be less than long_window ({long_window})."
        )


def _validate_prices(prices: pd.DataFrame, required_columns: list) -> None:
    """Validate a raw price DataFrame before computing signals."""
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame.")
    if prices.empty:
        raise ValueError("Price DataFrame is empty.")

    missing = [c for c in required_columns if c not in prices.columns]
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")

    subset = prices[required_columns]
    non_numeric = [c for c in required_columns if not pd.api.types.is_numeric_dtype(subset[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric price columns: {non_numeric}")

    if subset.isna().any().any():
        raise ValueError("Price data contains NaN values in required columns.")

    if (subset <= 0).any().any():
        raise ValueError("Price data contains non-positive prices.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price DataFrame index must be a DatetimeIndex.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price DataFrame index is not sorted chronologically.")

    if prices.index.duplicated().any():
        raise ValueError("Price DataFrame index contains duplicate dates.")


def _validate_moving_averages(moving_averages: pd.DataFrame) -> None:
    """Validate that a DataFrame has the expected (Ticker, Field) MultiIndex columns."""
    if not isinstance(moving_averages, pd.DataFrame):
        raise TypeError("moving_averages must be a pandas DataFrame.")
    if not isinstance(moving_averages.columns, pd.MultiIndex):
        raise ValueError("moving_averages must have a (Ticker, Field) MultiIndex column structure.")

    fields = set(moving_averages.columns.get_level_values(1))
    required_fields = {"Price", "MA_Short", "MA_Long"}
    missing_fields = required_fields - fields
    if missing_fields:
        raise ValueError(f"moving_averages is missing expected fields: {sorted(missing_fields)}")


def calculate_moving_averages(
    prices: pd.DataFrame, short_window: int = MA_SHORT, long_window: int = MA_LONG
) -> pd.DataFrame:
    """Calculate simple short/long moving averages per portfolio stock.

    Returns a DataFrame with a (Ticker, Field) MultiIndex on the columns,
    Field in {"Price", "MA_Short", "MA_Long"}. The first long_window - 1
    observations of MA_Long (and first short_window - 1 of MA_Short)
    remain NaN by construction; they are never backfilled.
    """
    _validate_windows(short_window, long_window)
    _validate_prices(prices, TICKERS)

    if len(prices) < long_window:
        raise ValueError(
            f"Insufficient price history: got {len(prices)} observations, "
            f"need at least {long_window} for a {long_window}-day moving average."
        )

    columns = {}
    for ticker in TICKERS:
        series = prices[ticker]
        columns[(ticker, "Price")] = series
        columns[(ticker, "MA_Short")] = series.rolling(window=short_window).mean()
        columns[(ticker, "MA_Long")] = series.rolling(window=long_window).mean()

    result = pd.DataFrame(columns, index=prices.index)
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["Ticker", "Field"])
    return result.sort_index(axis=1, level=0)


def calculate_trend_state(moving_averages: pd.DataFrame) -> pd.DataFrame:
    """Classify each date's trend per stock as Bullish, Bearish, Neutral, or Unavailable.

    "Unavailable" marks dates before both moving averages exist (i.e. the
    initial long_window - 1 NaN period) rather than assigning a false
    trend label.
    """
    _validate_moving_averages(moving_averages)
    tickers = list(moving_averages.columns.get_level_values("Ticker").unique())

    trend = pd.DataFrame(index=moving_averages.index, columns=tickers, dtype=object)

    for ticker in tickers:
        short_ma = moving_averages[(ticker, "MA_Short")]
        long_ma = moving_averages[(ticker, "MA_Long")]
        available = short_ma.notna() & long_ma.notna()

        state = pd.Series("Unavailable", index=moving_averages.index, dtype=object)
        state[available & (short_ma > long_ma)] = "Bullish"
        state[available & (short_ma < long_ma)] = "Bearish"
        state[available & (short_ma == long_ma)] = "Neutral"
        trend[ticker] = state

    return trend


def detect_crossovers(moving_averages: pd.DataFrame) -> pd.DataFrame:
    """Detect genuine Golden Cross / Death Cross transitions per stock.

    A Golden Cross requires the previous day's short MA <= long MA and the
    current day's short MA > long MA (and the reverse for a Death Cross).
    Both the current and previous day's moving averages must be available,
    so no events are generated from the initial NaN period. Events are
    returned sorted chronologically.
    """
    _validate_moving_averages(moving_averages)
    tickers = list(moving_averages.columns.get_level_values("Ticker").unique())

    records = []
    for ticker in tickers:
        short_ma = moving_averages[(ticker, "MA_Short")]
        long_ma = moving_averages[(ticker, "MA_Long")]
        price = moving_averages[(ticker, "Price")]

        prev_short = short_ma.shift(1)
        prev_long = long_ma.shift(1)
        valid = short_ma.notna() & long_ma.notna() & prev_short.notna() & prev_long.notna()

        golden_mask = valid & (prev_short <= prev_long) & (short_ma > long_ma)
        death_mask = valid & (prev_short >= prev_long) & (short_ma < long_ma)

        for mask, signal in ((golden_mask, "Golden Cross"), (death_mask, "Death Cross")):
            event_dates = moving_averages.index[mask]
            if len(event_dates) == 0:
                continue
            records.append(
                pd.DataFrame(
                    {
                        "Date": event_dates,
                        "Ticker": ticker,
                        "Signal": signal,
                        "Price": price.loc[event_dates].to_numpy(),
                        "Short MA": short_ma.loc[event_dates].to_numpy(),
                        "Long MA": long_ma.loc[event_dates].to_numpy(),
                    }
                )
            )

    columns = ["Date", "Ticker", "Signal", "Price", "Short MA", "Long MA"]
    if not records:
        return pd.DataFrame(columns=columns)

    events = pd.concat(records, ignore_index=True)
    return events.sort_values(["Date", "Ticker"]).reset_index(drop=True)[columns]


def create_signal_summary(
    prices: pd.DataFrame, short_window: int = MA_SHORT, long_window: int = MA_LONG
) -> pd.DataFrame:
    """Build a one-row-per-stock summary of the latest trend and crossover signal."""
    moving_averages = calculate_moving_averages(prices, short_window, long_window)
    trend = calculate_trend_state(moving_averages)
    crossovers = detect_crossovers(moving_averages)

    short_label = f"MA{short_window}"
    long_label = f"MA{long_window}"

    rows = []
    for ticker in TICKERS:
        ticker_crossovers = crossovers[crossovers["Ticker"] == ticker]
        if ticker_crossovers.empty:
            last_signal = "No crossover in sample"
            last_signal_date = pd.NaT
        else:
            last_event = ticker_crossovers.iloc[-1]
            last_signal = last_event["Signal"]
            last_signal_date = last_event["Date"]

        rows.append(
            {
                "Ticker": ticker,
                "Latest Price": moving_averages[(ticker, "Price")].iloc[-1],
                short_label: moving_averages[(ticker, "MA_Short")].iloc[-1],
                long_label: moving_averages[(ticker, "MA_Long")].iloc[-1],
                "Trend": trend[ticker].iloc[-1],
                "Last Crossover": last_signal,
                "Last Crossover Date": last_signal_date,
            }
        )

    return pd.DataFrame(rows).set_index("Ticker")


def backtest_moving_average_signal(
    prices: pd.DataFrame, ticker: str, short_window: int = MA_SHORT, long_window: int = MA_LONG
) -> pd.DataFrame:
    """Run a simplified moving-average regime backtest for one stock.

    Position is 1 when short MA > long MA and 0 otherwise, SHIFTED by one
    trading day before being applied to returns, so the strategy return on
    day t uses only the position known at the close of day t-1 (no
    look-ahead bias). No transaction costs are modelled. This is a
    simplified historical backtest for educational purposes, not an
    investment recommendation, and past regime performance is not a
    prediction of future performance.
    """
    if ticker not in TICKERS:
        raise ValueError(f"'{ticker}' is not a recognized portfolio ticker: {TICKERS}")

    moving_averages = calculate_moving_averages(prices, short_window, long_window)
    price = moving_averages[(ticker, "Price")]
    short_ma = moving_averages[(ticker, "MA_Short")]
    long_ma = moving_averages[(ticker, "MA_Long")]

    available = short_ma.notna() & long_ma.notna()
    position = pd.Series(0.0, index=price.index)
    position[available & (short_ma > long_ma)] = 1.0

    buy_hold_return = price.pct_change(fill_method=None)
    shifted_position = position.shift(1).fillna(0.0)
    strategy_return = shifted_position * buy_hold_return

    # Wealth indices start at 1.0; the undefined first-day return is treated
    # as 0% for compounding purposes only (Buy & Hold/Strategy Return above
    # retain NaN for that day, matching pct_change's convention).
    buy_hold_wealth = (1 + buy_hold_return.fillna(0.0)).cumprod()
    strategy_wealth = (1 + strategy_return.fillna(0.0)).cumprod()

    return pd.DataFrame(
        {
            "Price": price,
            "Short MA": short_ma,
            "Long MA": long_ma,
            "Position": position,
            "Buy & Hold Return": buy_hold_return,
            "Strategy Return": strategy_return,
            "Buy & Hold Wealth": buy_hold_wealth,
            "Strategy Wealth": strategy_wealth,
        },
        index=price.index,
    )


def create_backtest_summary(
    prices: pd.DataFrame, short_window: int = MA_SHORT, long_window: int = MA_LONG
) -> pd.DataFrame:
    """Summarize buy & hold vs. moving-average strategy performance per stock.

    Historical outperformance of the strategy in this summary is not a
    prediction of future performance.
    """
    rows = []
    for ticker in TICKERS:
        backtest = backtest_moving_average_signal(prices, ticker, short_window, long_window)
        returns = backtest[["Buy & Hold Return", "Strategy Return"]].iloc[1:]
        returns = returns.rename(
            columns={"Buy & Hold Return": "BuyHold", "Strategy Return": "Strategy"}
        )

        total_return = metrics.calculate_cumulative_returns(returns).iloc[-1]
        annualized_return = metrics.calculate_annualized_return(returns)
        annualized_volatility = metrics.calculate_annualized_volatility(returns)
        max_drawdown = metrics.calculate_max_drawdown(returns)

        rows.append(
            {
                "Ticker": ticker,
                "Buy & Hold Total Return": total_return["BuyHold"],
                "Strategy Total Return": total_return["Strategy"],
                "Buy & Hold Annualized Return": annualized_return["BuyHold"],
                "Strategy Annualized Return": annualized_return["Strategy"],
                "Buy & Hold Annualized Volatility": annualized_volatility["BuyHold"],
                "Strategy Annualized Volatility": annualized_volatility["Strategy"],
                "Buy & Hold Max Drawdown": max_drawdown["BuyHold"],
                "Strategy Max Drawdown": max_drawdown["Strategy"],
            }
        )

    return pd.DataFrame(rows).set_index("Ticker")
