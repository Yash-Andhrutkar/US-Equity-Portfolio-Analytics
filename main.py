"""
Master entry point for the US Equity Portfolio Analytics & Risk Automation
pipeline.

Coordinates the already-approved modules (src.data_loader, src.metrics,
src.portfolio, src.signals, src.simulation, src.visualizations) to run the
full analysis end to end: load or build the price data, compute return and
risk metrics, run the Monte Carlo scenario simulation, generate the
standard chart set, export summary tables, and write a plain-text
recruiter-readable report. This module does not duplicate any financial
formulas or download data outside of the data-loading step below.

Usage:
    python main.py
"""

from pathlib import Path

import pandas as pd

from config import (
    BENCHMARK,
    MA_LONG,
    MA_SHORT,
    MONTE_CARLO_DAYS,
    MONTE_CARLO_SIMULATIONS,
    PORTFOLIO_WEIGHTS,
    RANDOM_SEED,
    TICKERS,
)
from src import data_loader, metrics, portfolio, signals, simulation, visualizations

PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / "output" / "tables"
REPORTS_DIR = PROJECT_ROOT / "output" / "reports"
CHARTS_DIR = PROJECT_ROOT / "output" / "charts"

REPORT_FILENAME = "analysis_summary.txt"
MA_CHART_TICKER = "NVDA"

TABLE_FILENAMES = {
    "asset_metrics": "asset_metrics.csv",
    "portfolio_summary": "portfolio_summary.csv",
    "portfolio_vs_sp500": "portfolio_vs_sp500.csv",
    "risk_contribution": "risk_contribution.csv",
    "signal_summary": "signal_summary.csv",
    "backtest_summary": "backtest_summary.csv",
    "monte_carlo_summary": "monte_carlo_summary.csv",
}


def _load_prices() -> tuple[pd.DataFrame, str]:
    """Load the cached processed price data, falling back to the data pipeline.

    Only a missing cache (FileNotFoundError, raised by
    data_loader.load_price_data()) triggers the fallback download/build via
    data_loader.run_data_pipeline(); any other error (e.g. a corrupt cache
    failing validation) is left to propagate rather than silently retried.
    """
    try:
        prices = data_loader.load_price_data()
        return prices, "cache"
    except FileNotFoundError:
        prices = data_loader.run_data_pipeline()
        return prices, "downloaded"


def _largest_risk_contributor(risk_contribution: pd.DataFrame) -> tuple[str, float]:
    """Identify the ticker with the largest percentage contribution to portfolio risk."""
    ticker = risk_contribution["Risk Contribution %"].idxmax()
    value = float(risk_contribution.loc[ticker, "Risk Contribution %"])
    return ticker, value


def _export_tables(
    asset_metrics: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    portfolio_comparison: pd.DataFrame,
    risk_contribution: pd.DataFrame,
    signal_summary: pd.DataFrame,
    backtest_summary: pd.DataFrame,
    simulation_result: dict,
) -> dict:
    """Export the standard set of analysis tables as CSV files under TABLES_DIR."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    table_paths = {key: TABLES_DIR / filename for key, filename in TABLE_FILENAMES.items()}

    asset_metrics.to_csv(table_paths["asset_metrics"])
    portfolio_summary.to_csv(table_paths["portfolio_summary"])
    portfolio_comparison.to_csv(table_paths["portfolio_vs_sp500"])
    risk_contribution.to_csv(table_paths["risk_contribution"])
    signal_summary.to_csv(table_paths["signal_summary"])
    backtest_summary.to_csv(table_paths["backtest_summary"])
    simulation_result["summary"].rename_axis("Metric").reset_index(name="Value").to_csv(
        table_paths["monte_carlo_summary"], index=False
    )

    return table_paths


def _format_signal_lines(signal_summary: pd.DataFrame) -> str:
    """Render one line per ticker summarizing its current trend and latest crossover."""
    lines = []
    for ticker, row in signal_summary.iterrows():
        crossover_date = row["Last Crossover Date"]
        date_str = "n/a" if pd.isna(crossover_date) else pd.Timestamp(crossover_date).date().isoformat()
        lines.append(f"  {ticker}: Trend={row['Trend']}, Last Signal={row['Last Crossover']} ({date_str})")
    return "\n".join(lines)


def _build_report_text(
    prices: pd.DataFrame,
    portfolio_summary: pd.DataFrame,
    portfolio_comparison: pd.DataFrame,
    risk_contribution: pd.DataFrame,
    signal_summary: pd.DataFrame,
    simulation_result: dict,
) -> str:
    """Build the plain-text recruiter-readable analysis summary report."""
    portfolio_row = portfolio_summary.loc["Portfolio"]
    comparison_portfolio = portfolio_comparison.loc["Portfolio"]
    comparison_benchmark = portfolio_comparison.loc["S&P 500"]
    mc_summary = simulation_result["summary"]
    downside = simulation_result["downside_probabilities"]
    top_ticker, top_pct = _largest_risk_contributor(risk_contribution)

    weight_lines = "\n".join(f"  {ticker}: {weight:.1%}" for ticker, weight in PORTFOLIO_WEIGHTS.items())

    lines = [
        "US EQUITY PORTFOLIO ANALYTICS - SUMMARY REPORT",
        "=" * 48,
        "",
        "DATA PERIOD",
        f"  {prices.index.min().date()} to {prices.index.max().date()} ({len(prices)} trading days)",
        "",
        "TICKERS",
        f"  {', '.join(TICKERS)} (Benchmark: {BENCHMARK})",
        "",
        "PORTFOLIO WEIGHTS",
        weight_lines,
        "",
        "PORTFOLIO PERFORMANCE",
        f"  Annualized Return:      {portfolio_row['Annualized Return']:.2%}",
        f"  Annualized Volatility:  {portfolio_row['Annualized Volatility']:.2%}",
        f"  Sharpe Ratio:           {portfolio_row['Sharpe Ratio']:.2f}",
        f"  Max Drawdown:           {portfolio_row['Max Drawdown']:.2%}",
        f"  Beta (vs {BENCHMARK}):        {portfolio_row['Beta']:.2f}",
        "",
        "PORTFOLIO VS S&P 500",
        f"  Portfolio:  Return {comparison_portfolio['Annualized Return']:.2%}, "
        f"Volatility {comparison_portfolio['Annualized Volatility']:.2%}, "
        f"Sharpe {comparison_portfolio['Sharpe Ratio']:.2f}, "
        f"Max Drawdown {comparison_portfolio['Max Drawdown']:.2%}",
        f"  S&P 500:    Return {comparison_benchmark['Annualized Return']:.2%}, "
        f"Volatility {comparison_benchmark['Annualized Volatility']:.2%}, "
        f"Sharpe {comparison_benchmark['Sharpe Ratio']:.2f}, "
        f"Max Drawdown {comparison_benchmark['Max Drawdown']:.2%}",
        "",
        "RISK CONTRIBUTION",
        f"  Largest contributor to portfolio risk: {top_ticker} "
        f"({top_pct:.1%} of total portfolio risk)",
        "",
        f"MOVING-AVERAGE SIGNALS (MA{MA_SHORT} / MA{MA_LONG})",
        _format_signal_lines(signal_summary),
        "",
        f"MONTE CARLO SIMULATION ({MONTE_CARLO_SIMULATIONS:,} scenarios, {MONTE_CARLO_DAYS}-day horizon)",
        "  These results are historical-parameter scenario simulations based on",
        "  the portfolio's historical daily mean return and volatility under a",
        "  normal-distribution assumption. They are not a forecast of future",
        "  performance and should not be interpreted as predicting what the",
        "  portfolio will actually do.",
        "",
        f"  Median Terminal Return:         {mc_summary['Median Terminal Return']:.2%}",
        f"  5th Percentile Terminal Return: {mc_summary['5th Percentile Return']:.2%}",
        f"  Simulated Frequency of Loss:    {downside['Probability of Loss']:.1%} of simulated "
        f"scenarios ended below the starting value (a scenario frequency, not a "
        f"real-world forecast probability)",
        f"  95% VaR:                        {simulation_result['var']:.2%}",
        f"  95% CVaR:                       {simulation_result['cvar']:.2%}",
        "",
        "=" * 48,
    ]
    return "\n".join(lines) + "\n"


def _export_report(report_text: str) -> Path:
    """Write the analysis summary report to REPORTS_DIR and return its path."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / REPORT_FILENAME
    report_path.write_text(report_text, encoding="utf-8")
    return report_path


def run_analysis() -> dict:
    """Run the full analytics pipeline end to end and return the key results.

    Loads or builds the price data, computes returns and all approved
    analytics (asset metrics, portfolio summary, portfolio-vs-benchmark
    comparison, risk contribution, moving-average signals and backtest),
    runs the Monte Carlo simulation, generates the full chart set, exports
    the summary tables and text report, and returns a dictionary of the
    major results (DataFrames, the Monte Carlo result dict, exported table
    paths, the report path, and chart paths) for reuse by callers or tests.
    """
    prices, data_source = _load_prices()
    returns = metrics.calculate_daily_returns(prices)

    asset_metrics = metrics.create_metrics_summary(returns)
    portfolio_summary = portfolio.create_portfolio_summary(returns, PORTFOLIO_WEIGHTS)
    portfolio_comparison = portfolio.create_portfolio_comparison(returns, PORTFOLIO_WEIGHTS)
    risk_contribution = portfolio.calculate_risk_contribution(returns, PORTFOLIO_WEIGHTS)

    signal_summary = signals.create_signal_summary(prices, MA_SHORT, MA_LONG)
    backtest_summary = signals.create_backtest_summary(prices, MA_SHORT, MA_LONG)

    simulation_result = simulation.run_monte_carlo_simulation(
        returns,
        simulations=MONTE_CARLO_SIMULATIONS,
        days=MONTE_CARLO_DAYS,
        seed=RANDOM_SEED,
        weights=PORTFOLIO_WEIGHTS,
    )

    chart_paths = visualizations.generate_all_charts(
        prices,
        returns,
        simulation_result,
        weights=PORTFOLIO_WEIGHTS,
        ma_chart_ticker=MA_CHART_TICKER,
        output_dir=CHARTS_DIR,
    )

    table_paths = _export_tables(
        asset_metrics,
        portfolio_summary,
        portfolio_comparison,
        risk_contribution,
        signal_summary,
        backtest_summary,
        simulation_result,
    )

    report_text = _build_report_text(
        prices,
        portfolio_summary,
        portfolio_comparison,
        risk_contribution,
        signal_summary,
        simulation_result,
    )
    report_path = _export_report(report_text)

    return {
        "prices": prices,
        "returns": returns,
        "data_source": data_source,
        "asset_metrics": asset_metrics,
        "portfolio_summary": portfolio_summary,
        "portfolio_comparison": portfolio_comparison,
        "risk_contribution": risk_contribution,
        "signal_summary": signal_summary,
        "backtest_summary": backtest_summary,
        "simulation_result": simulation_result,
        "chart_paths": chart_paths,
        "table_paths": table_paths,
        "report_path": report_path,
    }


def main() -> None:
    """Run the full pipeline and print a concise completion summary."""
    header = "US Equity Portfolio Analytics"
    print(header)
    print("-" * len(header))

    try:
        results = run_analysis()
    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as exc:
        print(f"\nAnalysis failed: {exc}")
        raise SystemExit(1) from exc

    print(f"Data loaded successfully ({results['data_source']})")
    print("Analytics completed")
    print("Monte Carlo simulation completed")
    print(f"{len(results['chart_paths'])} charts generated")
    print(f"{len(results['table_paths'])} tables exported")
    print("Report generated")
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
