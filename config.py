"""
Central configuration for the US Equity Portfolio Analytics & Risk
Automation project. All modules and scripts should import their
parameters from here rather than hard-coding values.
"""

from datetime import datetime, timedelta

TICKERS = ["NVDA", "JPM", "AMZN", "LLY", "XOM"]

BENCHMARK = "^GSPC"

PORTFOLIO_WEIGHTS = {
    "NVDA": 0.25,
    "JPM": 0.20,
    "AMZN": 0.20,
    "LLY": 0.20,
    "XOM": 0.15,
}

if round(sum(PORTFOLIO_WEIGHTS.values()), 6) != 1.0:
    raise ValueError(
        f"PORTFOLIO_WEIGHTS must sum to 1.00, got "
        f"{sum(PORTFOLIO_WEIGHTS.values()):.6f}"
    )

HISTORY_YEARS = 5

END_DATE = datetime.today().strftime("%Y-%m-%d")
START_DATE = (datetime.today() - timedelta(days=365 * HISTORY_YEARS)).strftime("%Y-%m-%d")

TRADING_DAYS = 252

# Documented assumption: approximate yield on 3-month US Treasury bills as
# of project setup (2026). Update periodically to reflect prevailing rates
# rather than treating this as a live market feed.
RISK_FREE_RATE = 0.04

MA_SHORT = 50
MA_LONG = 200

MONTE_CARLO_SIMULATIONS = 10000
MONTE_CARLO_DAYS = 252
RANDOM_SEED = 42
