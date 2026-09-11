# US Equity Portfolio Analytics & Risk Automation

An automated Python analytics pipeline for a diversified US equity portfolio — data acquisition, return and risk metrics, benchmark comparison, correlation and drawdown analysis, moving-average signals and Monte Carlo scenario analysis — published every week as an interactive analytics dashboard.

**Live dashboard:** https://yash-andhrutkar.github.io/US-Equity-Portfolio-Analytics/

```
GitHub Actions (Sat 06:00 UTC)
        ↓
Fresh Yahoo Finance data
        ↓
Python analytics  (src/)
        ↓
Tests + validation gates
        ↓
Analytics outputs  (output/tables, output/charts, output/reports)
        ↓
dashboard_data.json  (scripts/export_dashboard_data.py)
        ↓
Interactive dashboard  (docs/)
        ↓
GitHub Pages deployment
```

No number on the dashboard is ever edited by hand. If any stage fails, nothing is committed and the previous successful deployment stays live.

## Portfolio

| Ticker | Company | Weight |
|--------|---------|--------|
| NVDA | NVIDIA | 0.25 |
| JPM | JPMorgan Chase | 0.20 |
| AMZN | Amazon | 0.20 |
| LLY | Eli Lilly | 0.20 |
| XOM | Exxon Mobil | 0.15 |

Benchmark: S&P 500 (`^GSPC`). History: 5 years of daily adjusted closes.

## Project structure

```
config.py                         Portfolio and analysis parameters — single source of truth
main.py                           Pipeline entry point: analytics → tables, charts, report
src/
  data_loader.py                  Download, validate and cache adjusted-close prices
  metrics.py                      Returns, volatility, Sharpe, beta, drawdown, correlation
  portfolio.py                    Weighted portfolio metrics and risk contribution
  signals.py                      Moving averages, crossovers, lag-shifted backtest
  simulation.py                   Monte Carlo scenario analysis, VaR and CVaR
  visualizations.py               Static matplotlib chart set
scripts/
  export_dashboard_data.py        Serializes the analytics into docs/data/dashboard_data.json
  validate_dashboard_data.py      Independently cross-checks that payload against output/tables
  check_no_hardcoded_numbers.py   Fails the build if the front end contains a financial literal
docs/                             The published dashboard (GitHub Pages source)
  index.html                      Eight numbered sections, sticky section navigation
  assets/css/                     Design tokens, layout, components
  assets/js/                      Store, data loader, formatting, motion, charts, table
  assets/fonts/                   Inter variable, self-hosted — no CDN dependency
  data/dashboard_data.json        Generated payload; the dashboard's only data source
tests/                            Unit tests for every analytics module and the exporter
notebooks/                        Exploratory analysis
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the full analytics pipeline:

```bash
python main.py
```

Build and check the dashboard payload:

```bash
python scripts/export_dashboard_data.py
python scripts/validate_dashboard_data.py
python scripts/check_no_hardcoded_numbers.py
```

Preview the dashboard locally — it fetches JSON, so it needs a server rather than `file://`:

```bash
python -m http.server 8000 --directory docs
# then open http://localhost:8000
```

Run the tests:

```bash
python -m pytest tests -v
```

## The dashboard

A dark institutional analytics interface: near-black ground with a navy cast, a
single azure accent, crisp off-white type, hairline rules and generous
whitespace. There are no cards — hierarchy comes from typography, rules and
space. Green appears only on a positive financial value and red only on a
negative or risk value; azure is structural and never means "good".

### One data source

The page reads exactly one file, `docs/data/dashboard_data.json`, produced by
`scripts/export_dashboard_data.py`. That script contains **no financial
formulas** — it calls the same approved functions in `src/` that `main.py` does
and serializes the result. The browser never computes a financial quantity.

Three gates enforce this in CI:

- **`validate_dashboard_data.py`** re-reads `output/tables/*.csv` and
  `data/processed/adjusted_close.csv` independently and asserts the payload
  agrees with them value for value — roughly 1,400 assertions covering schema,
  finiteness, agreement, identities (weights and risk contributions each
  summing to 1), provenance, series coherence and payload size.
- **`check_no_hardcoded_numbers.py`** renders every figure in
  `output/tables/*.csv` at several display precisions and searches the
  front-end source for them, rejects numeric array literals longer than 16
  entries (how a price series would be smuggled in), and requires the payload
  to be loaded from exactly one place.
- **`tests/test_export_dashboard_data.py`** asserts the exported values equal
  what `src.metrics`, `src.portfolio`, `src.signals` and `src.simulation`
  return directly.

### Sections

| # | Section | Contents |
|---|---|---|
| — | Overview | Title, analysis period, latest market date, last refresh, holdings, global range selector |
| — | KPI rail | Annualized return, volatility, Sharpe, max drawdown, beta, total return — with a benchmark row and a difference row on the same column grid |
| 01 | Performance | Portfolio vs benchmark cumulative return or drawdown, crosshair inspection, series toggles |
| 02 | Allocation & risk | Weight vs risk contribution, risk/return scatter, correlation matrix, diversification map, portfolio drawdown |
| 03 | Asset explorer | Per-holding metrics, trend state and latest crossover |
| 04 | Technical signals | Price with MA50/MA200, crossover markers, crossover timeline |
| 05 | Strategy backtest | Buy & hold vs the moving-average rule, growth of one dollar, comparison table |
| 06 | Monte Carlo | Percentile fan with horizon inspection, terminal distribution, VaR/CVaR, simulated loss frequency |
| 07 | Detailed analytics | Sortable per-holding table |
| 08 | Methodology | Expandable assumptions and provenance |

### Six time ranges

1M, 6M, YTD, 1Y, 3Y and the full sample. Each range's metric set is
**recomputed in Python** by re-calling the same approved functions on the
windowed return series, so changing the range genuinely changes the KPIs with
no formula duplicated in JavaScript.

Sub-annual ranges annualize a partial year, which is inherently noisy, so they
are flagged and the dashboard shows a caveat rather than presenting an
extrapolated figure as if it carried five years of weight.

### Diversification map

Holdings are positioned by classical multidimensional scaling of the
correlation distance matrix `d_ij = √(2(1 − ρ_ij))`, computed in Python, so
distance between two marks approximates how independently they have behaved.
Mark area is proportional to capital weight and edge brightness encodes
pairwise correlation. It is a standard way to read correlation structure at a
glance, alongside the matrix itself.

The layout is orientation-stabilized — pinned to a deterministic rotation and
reflection, with each range Procrustes-aligned to the full-sample layout — so
the field does not rotate or mirror between weekly refreshes.

### Interaction

Selecting a holding anywhere — a ticker tab, a risk bar, a scatter mark, a
diversification-map mark, a matrix diagonal, a table row, a crossover event —
updates every related panel through one shared store, and is mirrored into the
URL hash so a view is shareable (`#nvda`, `#nvda/1y`). Also: the global range
selector, return/drawdown switching, portfolio and benchmark toggles,
MA50/MA200 toggles, crossover inspection, buy-and-hold versus strategy
switching, Monte Carlo percentile inspection at any horizon, sortable table
columns, and keyboard navigation throughout — arrow keys traverse the range
selector, the ticker tabs, every chart's crosshair and the simulation horizon.

### Motion

Motion is event-driven and then settles completely: there is no ambient drift,
no parallax and no idle movement anywhere. Section reveals run once at 500ms
with a 60ms stagger, KPIs count up over 700ms, chart paths draw once on first
view, range changes crossfade at 320ms, and selection changes settle at 180ms.
`prefers-reduced-motion` reduces every duration to 1ms, snaps count-ups,
renders paths complete and disables smooth scrolling.

### Performance and accessibility

- Inter variable is self-hosted (48KB woff2, latin subset, preloaded). No CDN,
  no third-party requests, no JavaScript charting library.
- Charts are hand-built SVG; the Monte Carlo fan and histogram are Canvas 2D,
  where thousands of segments belong. Device pixel ratio capped at 1.75.
- Chart reveals are deferred until first view via `IntersectionObserver`;
  resize handlers are debounced.
- Every chart has an `aria-label`, a focusable inspection surface and keyboard
  stepping. Every interactive mark is a real button or has a `role` and label.
  The analytics table maintains `aria-sort`, and below 720px it becomes a
  stacked per-holding layout rather than a horizontal scroll.
- Payload: ~640 KB raw, ~210 KB gzipped over the wire.
## Automation

`.github/workflows/weekly_refresh.yml` runs every Saturday at 06:00 UTC (11:30 IST) — safely after Friday's US close year-round, without a DST-dependent cron — and on manual dispatch.

The `refresh` job runs the tests, deletes the cached CSVs to force a real download, runs `main.py`, then checks that every expected output exists and is non-empty, that the data is no more than four days old, that the exported payload agrees with the analytics tables, and that the front end contains no hardcoded figures. Only then does it commit.

The `deploy` job declares `needs: refresh`, so **any** failing gate means no deployment happens at all and the last good dashboard remains the live one. Nothing half-built can reach the public URL.

The dashboard always shows the latest market-data date and a pipeline status pill in the navigation bar, and the analysis period, latest market date, last successful refresh and session count in the overview. The pill turns amber client-side if the latest observation is more than ten days old, so a stalled pipeline is visible on the page itself rather than only in the Actions tab.

### One-time GitHub Pages setup

Pages deploys from the workflow, so this must be set once:

1. Push this repository to `main`.
2. Open **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **GitHub Actions** (not "Deploy from a branch").
4. Run the workflow: **Actions → Weekly Portfolio Analytics Refresh → Run workflow**.
5. The site appears at `https://yash-andhrutkar.github.io/US-Equity-Portfolio-Analytics/`.

No other configuration, secrets or tokens are required — `GITHUB_TOKEN` is provided automatically and the workflow requests only `contents: write` for the refresh commit plus `pages: write` and `id-token: write` for the deployment.

## Methodology and assumptions

Everything that would change a number, stated explicitly:

- **Prices** are end-of-day adjusted closes from Yahoo Finance via `yfinance`. Rows with any missing value are dropped with a warning rather than filled.
- **Returns** are simple daily percentage changes. Annualized returns are geometric (CAGR-style) over 252 trading days.
- **Risk-free rate** is 4.00%, a documented standing assumption in `config.py` approximating the 3-month US Treasury bill yield at project setup. It is not a live feed and should be revisited periodically.
- **Beta** uses sample covariance over sample variance against `^GSPC`.
- **Risk contribution** is the standard Euler decomposition: component contribution `wᵢ · (Σw)ᵢ / σ_p`, which sums to 1 by construction — asserted in both the analytics and the validator.
- **Moving averages** are 50-day and 200-day simple averages. The lead-in period is left undefined rather than backfilled, and no trend is asserted before both averages exist.
- **Crossovers** require a genuine transition: the previous day on one side and the current day strictly on the other, with both days' averages available.
- **Backtest** holds a long position when the short average is above the long one, with the position **shifted one trading day** before being applied to returns, so there is no look-ahead. No transaction costs, slippage or taxes are modelled. It is a simplified educational exercise, not a strategy recommendation.
- **Monte Carlo** draws daily portfolio returns from a normal distribution parameterized by the historical daily mean and standard deviation of the weighted portfolio, over 252 days with 10,000 scenarios and a fixed seed. Real markets exhibit fat tails, volatility clustering and regime changes that this does not capture, so extreme outcomes are under-represented. Outputs describe a simulated distribution under these assumptions — they are frequencies, not real-world probabilities, and not a forecast.
- **VaR and CVaR** are reported as positive loss magnitudes at 95% confidence.

## Disclaimer

This is an analytics and engineering project, not investment advice. Historical performance is not a prediction of future performance. Nothing here is a recommendation to buy or sell any security.
