"""
Generates and exports charts summarizing portfolio performance, risk
metrics, signals, and Monte Carlo simulation results to output/charts.

All calculations are delegated to the approved analytics modules
(src.metrics, src.portfolio, src.signals) or consume an already-computed
Monte Carlo result dict (from src.simulation.run_monte_carlo_simulation);
this module does not download market data or duplicate financial
formulas. Every chart is rendered on the non-interactive "Agg" backend
and saved directly to disk (this module never calls plt.show()).

Monte Carlo charts visualize scenario analysis based on HISTORICAL mean
and volatility under a normal-return assumption - they are not price
forecasts. See src/simulation.py for the full methodology and its
limitations.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from config import BENCHMARK, MA_LONG, MA_SHORT, PORTFOLIO_WEIGHTS, RANDOM_SEED, TICKERS  # noqa: E402

from . import metrics, portfolio, signals  # noqa: E402

PathLike = Union[str, Path]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "charts"

DPI = 150
FIGSIZE_STANDARD = (10, 6)
FIGSIZE_WIDE = (12, 6.5)
FIGSIZE_SQUARE = (8, 7)

COLOR_PORTFOLIO = "#1f4e79"
COLOR_BENCHMARK = "#a6373f"
COLOR_MEDIAN = "#1f4e79"
COLOR_P05 = "#a6373f"
COLOR_P95 = "#2e7d5b"
COLOR_SAMPLE_PATH = "#9fb8cf"

FILENAMES = {
    "portfolio_vs_sp500": "portfolio_vs_sp500.png",
    "correlation_heatmap": "correlation_heatmap.png",
    "risk_return_scatter": "risk_return_scatter.png",
    "portfolio_drawdown": "portfolio_drawdown.png",
    "risk_contribution": "risk_contribution.png",
    "monte_carlo_paths": "monte_carlo_paths.png",
    "terminal_return_distribution": "terminal_return_distribution.png",
}


def _apply_style() -> None:
    """Apply the shared professional chart theme. Centralizes all styling."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _resolve_output_path(output_path: Optional[PathLike], key: str) -> Path:
    """Resolve a chart's output path, defaulting to the standard project location."""
    if output_path is None:
        return OUTPUT_DIR / FILENAMES[key]
    return Path(output_path)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Save a figure as PNG (min. 150 DPI, tight bbox), then close it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _percent_formatter() -> mticker.PercentFormatter:
    return mticker.PercentFormatter(xmax=1.0)


def plot_portfolio_vs_benchmark(
    returns: pd.DataFrame,
    weights: dict = PORTFOLIO_WEIGHTS,
    initial_investment: float = 10_000.0,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Plot cumulative growth of the portfolio vs. the S&P 500, normalized to a fixed investment."""
    _apply_style()
    resolved_path = _resolve_output_path(output_path, "portfolio_vs_sp500")

    portfolio_returns = portfolio.calculate_portfolio_returns(returns, weights)
    portfolio_cum = portfolio.calculate_portfolio_cumulative_return(portfolio_returns)
    benchmark_cum = metrics.calculate_cumulative_returns(returns[[BENCHMARK]])[BENCHMARK]

    portfolio_wealth = initial_investment * (1 + portfolio_cum)
    benchmark_wealth = initial_investment * (1 + benchmark_cum)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.plot(portfolio_wealth.index, portfolio_wealth, color=COLOR_PORTFOLIO, linewidth=2, label="Portfolio")
    ax.plot(benchmark_wealth.index, benchmark_wealth, color=COLOR_BENCHMARK, linewidth=2, label="S&P 500")

    for series, color in ((portfolio_wealth, COLOR_PORTFOLIO), (benchmark_wealth, COLOR_BENCHMARK)):
        ax.annotate(
            f"${series.iloc[-1]:,.0f}",
            xy=(series.index[-1], series.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            color=color,
            fontweight="bold",
            va="center",
        )

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_title("Portfolio Growth vs S&P 500")
    ax.set_ylabel(f"Value of ${initial_investment:,.0f} Invested")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()

    return _save_figure(fig, resolved_path)


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Plot the Pearson return correlation matrix for all portfolio stocks and the benchmark."""
    _apply_style()
    resolved_path = _resolve_output_path(output_path, "correlation_heatmap")

    columns = list(TICKERS) + [BENCHMARK]
    correlation_matrix = metrics.calculate_correlation_matrix(returns[columns])

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Correlation"},
        ax=ax,
    )
    ax.set_title("Daily Return Correlation Matrix")

    return _save_figure(fig, resolved_path)


def plot_risk_return_scatter(
    returns: pd.DataFrame,
    weights: dict = PORTFOLIO_WEIGHTS,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Plot annualized risk vs. return for each stock, the portfolio, and the benchmark."""
    _apply_style()
    resolved_path = _resolve_output_path(output_path, "risk_return_scatter")

    stock_summary = metrics.create_metrics_summary(returns)[["Annualized Return", "Annualized Volatility"]]
    comparison = portfolio.create_portfolio_comparison(returns, weights)[
        ["Annualized Return", "Annualized Volatility"]
    ]

    fig, ax = plt.subplots(figsize=(9, 7))

    for name, row in stock_summary.iterrows():
        ax.scatter(row["Annualized Volatility"], row["Annualized Return"], s=90, color=COLOR_PORTFOLIO, alpha=0.75, zorder=3)
        ax.annotate(name, (row["Annualized Volatility"], row["Annualized Return"]), textcoords="offset points", xytext=(6, 4), fontsize=9)

    portfolio_row = comparison.loc["Portfolio"]
    benchmark_row = comparison.loc["S&P 500"]
    ax.scatter(portfolio_row["Annualized Volatility"], portfolio_row["Annualized Return"], s=260, marker="*", color="#c98a1f", edgecolor="black", linewidth=0.6, zorder=4, label="Portfolio")
    ax.scatter(benchmark_row["Annualized Volatility"], benchmark_row["Annualized Return"], s=140, marker="D", color=COLOR_BENCHMARK, edgecolor="black", linewidth=0.6, zorder=4, label="S&P 500")
    ax.annotate("Portfolio", (portfolio_row["Annualized Volatility"], portfolio_row["Annualized Return"]), textcoords="offset points", xytext=(8, 6), fontsize=10, fontweight="bold")
    ax.annotate("S&P 500", (benchmark_row["Annualized Volatility"], benchmark_row["Annualized Return"]), textcoords="offset points", xytext=(8, -12), fontsize=10, fontweight="bold")

    ax.margins(x=0.18, y=0.18)
    ax.xaxis.set_major_formatter(_percent_formatter())
    ax.yaxis.set_major_formatter(_percent_formatter())
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    ax.set_title("Annualized Risk vs Return")
    ax.legend(loc="best")

    return _save_figure(fig, resolved_path)


def plot_portfolio_drawdown(
    returns: pd.DataFrame,
    weights: dict = PORTFOLIO_WEIGHTS,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Plot the weighted portfolio's historical drawdown series, annotating the maximum drawdown."""
    _apply_style()
    resolved_path = _resolve_output_path(output_path, "portfolio_drawdown")

    portfolio_returns = portfolio.calculate_portfolio_returns(returns, weights)
    drawdown = portfolio.calculate_portfolio_drawdown(portfolio_returns)
    max_drawdown_date = drawdown.idxmin()
    max_drawdown_value = drawdown.min()

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    ax.fill_between(drawdown.index, drawdown, 0, color=COLOR_BENCHMARK, alpha=0.35)
    ax.plot(drawdown.index, drawdown, color=COLOR_BENCHMARK, linewidth=1.2)

    ax.annotate(
        f"Max Drawdown: {max_drawdown_value:.1%}",
        xy=(max_drawdown_date, max_drawdown_value),
        xytext=(0.5, 0.92),
        textcoords="axes fraction",
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="top",
        arrowprops={"arrowstyle": "->", "color": "#333333"},
    )

    ax.yaxis.set_major_formatter(_percent_formatter())
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown from Prior Peak")
    fig.autofmt_xdate()

    return _save_figure(fig, resolved_path)


def plot_risk_contribution(
    returns: pd.DataFrame,
    weights: dict = PORTFOLIO_WEIGHTS,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Plot each stock's percentage contribution to total portfolio risk."""
    _apply_style()
    resolved_path = _resolve_output_path(output_path, "risk_contribution")

    risk_contribution = portfolio.calculate_risk_contribution(returns, weights)
    ordered = risk_contribution["Risk Contribution %"].sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(ordered.index, ordered.values, color=COLOR_PORTFOLIO)

    for bar, value in zip(bars, ordered.values):
        ax.annotate(
            f"{value:.1%}",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
        )

    ax.xaxis.set_major_formatter(_percent_formatter())
    ax.set_xlabel("Risk Contribution %")
    ax.set_title("Contribution to Portfolio Risk")

    return _save_figure(fig, resolved_path)


def plot_moving_average_signals(
    prices: pd.DataFrame,
    ticker: str = "NVDA",
    short_window: int = MA_SHORT,
    long_window: int = MA_LONG,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Plot a stock's price, short/long moving averages, and Golden/Death Cross markers.

    Crossover events are taken directly from src.signals.detect_crossovers -
    this function does not recalculate crossover logic independently.
    """
    if ticker not in TICKERS:
        raise ValueError(f"'{ticker}' is not a recognized portfolio ticker: {TICKERS}")

    _apply_style()
    if output_path is None:
        resolved_path = OUTPUT_DIR / f"{ticker.lower()}_moving_average_signals.png"
    else:
        resolved_path = Path(output_path)

    moving_averages = signals.calculate_moving_averages(prices, short_window, long_window)
    events = signals.detect_crossovers(moving_averages)
    ticker_events = events[events["Ticker"] == ticker]

    price = moving_averages[(ticker, "Price")]
    short_ma = moving_averages[(ticker, "MA_Short")]
    long_ma = moving_averages[(ticker, "MA_Long")]
    golden_events = ticker_events[ticker_events["Signal"] == "Golden Cross"]
    death_events = ticker_events[ticker_events["Signal"] == "Death Cross"]

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot(price.index, price, color="#555555", linewidth=1.0, label="Price", alpha=0.8)
    ax.plot(short_ma.index, short_ma, color=COLOR_PORTFOLIO, linewidth=1.6, label=f"MA{short_window}")
    ax.plot(long_ma.index, long_ma, color=COLOR_BENCHMARK, linewidth=1.6, label=f"MA{long_window}")

    if not golden_events.empty:
        ax.scatter(golden_events["Date"], golden_events["Price"], marker="^", s=130, color="#2e7d5b", edgecolor="black", linewidth=0.6, zorder=5, label="Golden Cross")
    if not death_events.empty:
        ax.scatter(death_events["Date"], death_events["Price"], marker="v", s=130, color="#a6373f", edgecolor="black", linewidth=0.6, zorder=5, label="Death Cross")

    ax.set_title(f"{ticker}: Price & Moving Average Signals")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="best")
    fig.autofmt_xdate()

    return _save_figure(fig, resolved_path)


def plot_monte_carlo_paths(
    simulation_result: dict,
    sample_size: int = 50,
    seed: int = RANDOM_SEED,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Plot a readable sample of simulated wealth paths plus median/5th/95th percentile paths.

    Consumes an already-computed simulation_result dict (as returned by
    src.simulation.run_monte_carlo_simulation) rather than running a new
    simulation. This is historical-parameter scenario analysis, not a
    forecast.
    """
    _apply_style()
    resolved_path = _resolve_output_path(output_path, "monte_carlo_paths")

    wealth_paths = simulation_result["wealth_paths"]
    n_simulations = wealth_paths.shape[1]
    sample_n = min(sample_size, n_simulations)

    rng = np.random.default_rng(seed)
    sample_columns = rng.choice(wealth_paths.columns, size=sample_n, replace=False)

    median_path = wealth_paths.median(axis=1)
    p05_path = wealth_paths.quantile(0.05, axis=1)
    p95_path = wealth_paths.quantile(0.95, axis=1)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for column in sample_columns:
        ax.plot(wealth_paths.index, wealth_paths[column], color=COLOR_SAMPLE_PATH, linewidth=0.6, alpha=0.5, zorder=1)

    ax.plot(median_path.index, median_path, color=COLOR_MEDIAN, linewidth=2.2, label="Median Path", zorder=3)
    ax.plot(p05_path.index, p05_path, color=COLOR_P05, linewidth=1.8, linestyle="--", label="5th Percentile Path", zorder=3)
    ax.plot(p95_path.index, p95_path, color=COLOR_P95, linewidth=1.8, linestyle="--", label="95th Percentile Path", zorder=3)

    ax.set_title("One-Year Monte Carlo Portfolio Scenarios")
    ax.set_xlabel("Simulation Day")
    ax.set_ylabel("Simulated Portfolio Value (x Initial Investment)")
    ax.legend(loc="best")
    fig.text(
        0.5, -0.02,
        f"{n_simulations:,} Scenarios Simulated - {sample_n} Paths Displayed - Not a Forecast",
        ha="center", fontsize=9, style="italic", color="#555555",
    )

    return _save_figure(fig, resolved_path)


def plot_terminal_return_distribution(
    simulation_result: dict,
    confidence: float = 0.95,
    output_path: Optional[PathLike] = None,
) -> Path:
    """Plot the distribution of simulated terminal returns with median/VaR/CVaR markers."""
    _apply_style()
    resolved_path = _resolve_output_path(output_path, "terminal_return_distribution")

    terminal_returns = simulation_result["terminal_returns"]
    median_return = terminal_returns.median()
    var = simulation_result.get("var")
    cvar = simulation_result.get("cvar")
    var_threshold_return = terminal_returns.quantile(1 - confidence)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)
    sns.histplot(terminal_returns, bins=50, color=COLOR_PORTFOLIO, alpha=0.7, ax=ax)

    ax.axvline(median_return, color="#2e7d5b", linewidth=2, label=f"Median: {median_return:.1%}")
    var_percentile = (1 - confidence) * 100
    ax.axvline(
        var_threshold_return,
        color=COLOR_BENCHMARK,
        linewidth=2,
        linestyle="--",
        label=f"{var_percentile:.0f}th Percentile / {int(confidence * 100)}% VaR: {var_threshold_return:.1%}",
    )
    if cvar is not None and cvar > 0:
        ax.axvline(-cvar, color="#7a2020", linewidth=2, linestyle=":", label=f"CVaR (avg. tail loss): -{cvar:.1%}")

    ax.xaxis.set_major_formatter(_percent_formatter())
    ax.set_title("Monte Carlo Terminal Return Distribution")
    ax.set_xlabel("Terminal Return (1-Year)")
    ax.set_ylabel("Number of Simulations")
    ax.legend(loc="best", fontsize=9)
    fig.text(
        0.5, -0.02,
        f"{len(terminal_returns):,} historical-parameter scenarios; not a forecast.",
        ha="center", fontsize=9, style="italic", color="#555555",
    )

    return _save_figure(fig, resolved_path)


def generate_all_charts(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    simulation_result: dict,
    weights: dict = PORTFOLIO_WEIGHTS,
    ma_chart_ticker: str = "NVDA",
    output_dir: Optional[PathLike] = None,
) -> dict:
    """Generate all standard project charts into output_dir and return their paths.

    Creates output_dir if it does not exist. Only the project's own
    standard chart filenames within output_dir are written/overwritten;
    no other files are touched.
    """
    directory = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    directory.mkdir(parents=True, exist_ok=True)

    ma_chart_path = directory / f"{ma_chart_ticker.lower()}_moving_average_signals.png"

    return {
        "portfolio_vs_sp500": plot_portfolio_vs_benchmark(
            returns, weights, output_path=directory / FILENAMES["portfolio_vs_sp500"]
        ),
        "correlation_heatmap": plot_correlation_heatmap(
            returns, output_path=directory / FILENAMES["correlation_heatmap"]
        ),
        "risk_return_scatter": plot_risk_return_scatter(
            returns, weights, output_path=directory / FILENAMES["risk_return_scatter"]
        ),
        "portfolio_drawdown": plot_portfolio_drawdown(
            returns, weights, output_path=directory / FILENAMES["portfolio_drawdown"]
        ),
        "risk_contribution": plot_risk_contribution(
            returns, weights, output_path=directory / FILENAMES["risk_contribution"]
        ),
        "moving_average_signals": plot_moving_average_signals(
            prices, ma_chart_ticker, output_path=ma_chart_path
        ),
        "monte_carlo_paths": plot_monte_carlo_paths(
            simulation_result, output_path=directory / FILENAMES["monte_carlo_paths"]
        ),
        "terminal_return_distribution": plot_terminal_return_distribution(
            simulation_result, output_path=directory / FILENAMES["terminal_return_distribution"]
        ),
    }
