"""
Handles retrieval, validation, and caching of historical daily market data
for the portfolio tickers and benchmark via yfinance, including persistence
to the data/raw and data/processed directories.

Intended usage: run as `python -m src.data_loader` from the project root so
that both the `src` package and the root-level `config` module resolve
correctly on the import path.
"""

import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import BENCHMARK, END_DATE, MA_LONG, START_DATE, TICKERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "market_data_raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "adjusted_close.csv"

REQUIRED_COLUMNS = list(TICKERS) + [BENCHMARK]


def download_market_data() -> pd.DataFrame:
    """Download raw daily OHLCV data for all portfolio tickers and the benchmark."""
    tickers = list(TICKERS) + [BENCHMARK]
    raw_data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=True,
    )

    if raw_data is None or raw_data.empty:
        raise ValueError(
            f"yfinance returned no data for {tickers} between "
            f"{START_DATE} and {END_DATE}."
        )

    raw_data.index = pd.to_datetime(raw_data.index)
    if raw_data.index.tz is not None:
        raw_data.index = raw_data.index.tz_localize(None)

    raw_data = raw_data.sort_index()
    raw_data = raw_data[~raw_data.index.duplicated(keep="first")]

    return raw_data


def extract_adjusted_close(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Extract adjusted-close prices for all required tickers from raw yfinance data."""
    if not isinstance(raw_data.columns, pd.MultiIndex):
        raise ValueError(
            "Expected a MultiIndex column structure from a multi-ticker "
            "yfinance download."
        )

    level_0 = raw_data.columns.get_level_values(0)
    level_1 = raw_data.columns.get_level_values(1)

    if "Adj Close" in level_0:
        adj_close = raw_data["Adj Close"]
    elif "Adj Close" in level_1:
        adj_close = raw_data.xs("Adj Close", axis=1, level=1)
    else:
        raise ValueError("'Adj Close' column not found in downloaded data.")

    missing_tickers = [t for t in REQUIRED_COLUMNS if t not in adj_close.columns]
    if missing_tickers:
        raise ValueError(f"Missing expected tickers in downloaded data: {missing_tickers}")

    adj_close = adj_close[REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    adj_close = adj_close.sort_index()
    adj_close = adj_close[~adj_close.index.duplicated(keep="first")]

    missing_mask = adj_close.isna()
    affected_rows = int(missing_mask.any(axis=1).sum())
    if affected_rows > 0:
        per_column = missing_mask.sum()
        per_column = per_column[per_column > 0]
        warnings.warn(
            f"Removing {affected_rows} row(s) with missing adjusted-close "
            f"values. Missing counts by column: {per_column.to_dict()}",
            stacklevel=2,
        )
        adj_close = adj_close.dropna(axis=0, how="any")

    return adj_close


def validate_price_data(prices: pd.DataFrame) -> None:
    """Validate a processed adjusted-close price DataFrame, raising on failure."""
    if prices.empty:
        raise ValueError("Price DataFrame is empty.")

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in prices.columns]
    if missing_columns:
        raise ValueError(f"Price DataFrame is missing expected columns: {missing_columns}")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Price DataFrame index must be a DatetimeIndex.")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("Price DataFrame index is not sorted chronologically.")

    if prices.index.duplicated().any():
        raise ValueError("Price DataFrame contains duplicate dates.")

    non_numeric = [c for c in REQUIRED_COLUMNS if not pd.api.types.is_numeric_dtype(prices[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric price columns found: {non_numeric}")

    if (prices[REQUIRED_COLUMNS] <= 0).any().any():
        raise ValueError("Price DataFrame contains non-positive prices.")

    if prices[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Price DataFrame contains missing values.")

    min_observations = MA_LONG + 1
    if len(prices) < min_observations:
        raise ValueError(
            f"Insufficient observations for analysis: got {len(prices)}, "
            f"need at least {min_observations}."
        )


def save_market_data(raw_data: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Persist raw and processed market data to the data/ directory."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    raw_data.to_csv(RAW_DATA_PATH)
    prices.to_csv(PROCESSED_DATA_PATH)


def load_price_data() -> pd.DataFrame:
    """Load and validate the cached processed adjusted-close price data."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{PROCESSED_DATA_PATH} not found. Run run_data_pipeline() first."
        )

    prices = pd.read_csv(PROCESSED_DATA_PATH, index_col=0, parse_dates=True)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    validate_price_data(prices)
    return prices


def run_data_pipeline() -> pd.DataFrame:
    """Download, extract, validate, and persist adjusted-close price data."""
    raw_data = download_market_data()
    prices = extract_adjusted_close(raw_data)
    validate_price_data(prices)
    save_market_data(raw_data, prices)
    return prices


if __name__ == "__main__":
    prices = run_data_pipeline()
    print(prices.tail())
    print(f"\nRows: {len(prices)}")
    print(f"Date range: {prices.index.min().date()} to "
          f"{prices.index.max().date()}")
