# US Equity Portfolio Analytics & Risk Automation

An automated Python analytics pipeline for a diversified US equity portfolio: data acquisition, return and risk metrics, benchmark comparison, correlation and drawdown analysis, moving-average signals, and Monte Carlo simulation.

## Status

Project foundation stage. Data acquisition, analytics, and reporting modules are being built incrementally. This README will be expanded with methodology, setup instructions, and sample results as each component is completed.

## Portfolio

| Ticker | Company | Weight |
|--------|---------|--------|
| NVDA | NVIDIA | 0.25 |
| JPM | JPMorgan Chase | 0.20 |
| AMZN | Amazon | 0.20 |
| LLY | Eli Lilly | 0.20 |
| XOM | Exxon Mobil | 0.15 |

Benchmark: S&P 500 (`^GSPC`)

## Project Structure

See `config.py` for portfolio and analysis parameters, and `main.py` for the pipeline entry point (to be implemented).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
