/**
 * Dashboard bootstrap and panel wiring.
 *
 * Loads the generated payload, builds every panel, and connects them through
 * the shared selection store so that choosing a holding anywhere updates every
 * related panel at once, and changing the range re-reads every precomputed
 * metric set.
 *
 * No financial quantity is computed here. Values come from the payload's
 * per-window metric sets and are formatted for display.
 */

import { DataError, loadDashboardData, pipelineHealth } from "./data.js";
import { store } from "./store.js";
import {
  EMPTY,
  assetColor,
  count,
  cssVar,
  dateLong,
  daysSince,
  money,
  multiple,
  pct,
  pctSigned,
  ratio,
  signClass,
  timestamp,
} from "./format.js";
import { countUp, motion, observeReveals, observeSections } from "./motion.js";
import {
  createDrawdownChart,
  createPerformanceChart,
} from "./charts/performance.js";
import {
  createCorrelationMatrix,
  createDiversificationMap,
  createRiskReturnScatter,
  createWeightRisk,
} from "./charts/risk.js";
import {
  createBacktestChart,
  createSignalChart,
  createSignalTimeline,
} from "./charts/signals.js";
import {
  createMonteCarloFan,
  createTerminalHistogram,
} from "./charts/montecarlo.js";
import { createAnalyticsTable } from "./table.js";

const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];

boot();

async function boot() {
  const app = $("[data-app]");

  let payload;
  try {
    payload = await loadDashboardData();
  } catch (error) {
    renderFailure(error);
    return;
  }

  $("[data-loading]")?.remove();
  app.hidden = false;

  try {
    initDashboard(payload);
  } catch (error) {
    console.error(error);
    renderFailure(error);
  }
}

function renderFailure(error) {
  const message =
    error instanceof DataError
      ? error.message
      : "The dashboard could not be initialised.";

  const app = $("[data-app]");
  if (app) app.hidden = true;

  const node = $("[data-loading]") ?? document.createElement("div");
  node.className = "state state--error";
  node.innerHTML =
    `<p><strong>Dashboard unavailable</strong>${escapeHtml(message)}</p>` +
    `<p class="muted">The analytics pipeline republishes this dashboard on every ` +
    `successful weekly refresh. If a refresh failed, the previously published ` +
    `dashboard remains the live one.</p>`;
  if (!node.isConnected) document.body.appendChild(node);
}

function escapeHtml(value) {
  const div = document.createElement("div");
  div.textContent = String(value);
  return div.innerHTML;
}

/* -------------------------------------------------------------------------- */

function initDashboard(payload) {
  const meta = payload.meta;
  const tickers = meta.tickers;
  const windowKeys = meta.windows.map((w) => w.key);

  store.set({ window: meta.default_window });
  store.readHash(tickers, windowKeys);

  renderStatus(payload);
  renderStaticCopy(payload);
  renderMethodology(payload);
  renderMonteCarloStats(payload);

  const select = (ticker) => store.set({ ticker, eventIndex: null });

  const panels = [
    buildRangeSelector(payload),
    buildKpiRail(payload),
    buildTickerTabs(payload),
    buildAssetPanel(payload),
    buildBacktestMetrics(payload),
    buildPerformanceControls(),
    buildMaControls(),
    buildBacktestControls(),
    buildNavSpy(),

    wrap(createPerformanceChart($("[data-chart-performance]"), payload), (s) => ({
      mode: s.performanceMode,
      windowKey: s.window,
      ticker: s.ticker,
      showPortfolio: s.showPortfolio,
      showBenchmark: s.showBenchmark,
    })),
    wrap(createDrawdownChart($("[data-chart-drawdown]"), payload), (s) => ({
      windowKey: s.window,
      ticker: s.ticker,
    })),
    createWeightRisk($("[data-wrisk]"), payload, { onSelect: select }),
    createRiskReturnScatter($("[data-scatter]"), payload, { onSelect: select }),
    createDiversificationMap($("[data-distance]"), payload, { onSelect: select }),
    createCorrelationMatrix($("[data-heatmap]"), payload, { onSelect: select }),
    wrap(createSignalChart($("[data-chart-signals]"), payload), (s) => ({
      ticker: s.ticker,
      showMaShort: s.showMaShort,
      showMaLong: s.showMaLong,
      windowKey: s.window,
      eventIndex: s.eventIndex,
    })),
    createSignalTimeline($("[data-timeline]"), payload, {
      onSelect: (ticker, eventIndex) => store.set({ ticker, eventIndex }),
    }),
    wrap(createBacktestChart($("[data-chart-backtest]"), payload), (s) => ({
      ticker: s.ticker,
      backtestMode: s.backtestMode,
    })),
    createAnalyticsTable($("[data-analytics-table]"), payload, {
      onSelect: select,
      store,
    }),
  ];

  createMonteCarloFan($("[data-chart-mc]"), payload);
  createTerminalHistogram($("[data-chart-histogram]"), payload);

  // Panels receive one normalized view object. `windowKey` aliases `window` so
  // panel code never shadows the global.
  const apply = (state) => {
    const view = { ...state, windowKey: state.window };
    for (const panel of panels) {
      try {
        panel.update(view);
      } catch (error) {
        console.error("panel update failed", error);
      }
    }
  };

  store.subscribe(apply);
  apply(store.get());

  observeReveals();

  window.addEventListener("hashchange", () => {
    store.readHash(tickers, windowKeys);
  });
}

/** Adapt a chart module's update signature to the shared view object. */
function wrap(instance, project) {
  return { update: (view) => instance.update(project(view)) };
}

/* -------------------------------------------------------------------------- */
/* Status and static copy                                                    */
/* -------------------------------------------------------------------------- */

function renderStatus(payload) {
  const meta = payload.meta;
  const age = daysSince(meta.latest_market_date);
  const health = pipelineHealth(meta, age);

  $("[data-nav-date]").textContent = dateLong(meta.latest_market_date);

  const pill = $("[data-status-pill]");
  pill.className = `pill pill--${health.level}`;
  pill.innerHTML = `<span class="pill__dot"></span>${escapeHtml(health.label)}`;
  pill.title =
    age === null
      ? "Pipeline status reported by the analytics export."
      : `Latest market observation is ${age} day${age === 1 ? "" : "s"} old. ` +
        `The pipeline refreshes weekly after the Friday US close.`;

  $("[data-meta-period]").textContent =
    `${dateLong(meta.first_market_date)} – ${dateLong(meta.latest_market_date)} · ` +
    `${count(meta.trading_days)} sessions`;
  $("[data-meta-market]").textContent = dateLong(meta.latest_market_date);
  $("[data-meta-refresh]").textContent = timestamp(meta.generated_at_utc);
  $("[data-meta-weights]").textContent = meta.tickers
    .map((t) => `${t} ${pct(meta.weights[t], 0)}`)
    .join(" · ");

  $("[data-commit]").textContent = meta.git_sha;
  $("[data-schema]").textContent = meta.schema_version;
}

function renderStaticCopy(payload) {
  const meta = payload.meta;
  $$("[data-benchmark-label]").forEach((n) => {
    n.textContent = meta.benchmark.label;
  });
  $$("[data-ma-short]").forEach((n) => {
    n.textContent = String(meta.assumptions.ma_short);
  });
  $$("[data-ma-long]").forEach((n) => {
    n.textContent = String(meta.assumptions.ma_long);
  });
  $$("[data-sim-count]").forEach((n) => {
    n.textContent = count(payload.monte_carlo.config.simulations);
  });
  $$("[data-sim-days]").forEach((n) => {
    n.textContent = String(payload.monte_carlo.config.days);
  });
  $("[data-event-count]").textContent = `${payload.signals.events.length} events`;
}

/* -------------------------------------------------------------------------- */
/* Controls                                                                   */
/* -------------------------------------------------------------------------- */

function buildRangeSelector(payload) {
  const host = $("[data-range]");
  host.innerHTML = payload.meta.windows
    .map(
      (w) =>
        `<button type="button" class="switch__btn" role="tab" data-window="${w.key}" ` +
        `aria-selected="false" title="${w.observations} sessions from ${dateLong(
          w.start_date
        )}">${w.key}</button>`
    )
    .join("");

  const buttons = $$("[data-window]", host);
  buttons.forEach((button, index) => {
    button.addEventListener("click", () => store.set({ window: button.dataset.window }));
    button.addEventListener("keydown", (event) => {
      let next = null;
      if (event.key === "ArrowRight") next = buttons[(index + 1) % buttons.length];
      else if (event.key === "ArrowLeft")
        next = buttons[(index - 1 + buttons.length) % buttons.length];
      if (!next) return;
      event.preventDefault();
      next.focus();
      store.set({ window: next.dataset.window });
    });
  });

  return {
    update(state) {
      buttons.forEach((button) => {
        const on = button.dataset.window === state.window;
        button.setAttribute("aria-selected", String(on));
        button.tabIndex = on ? 0 : -1;
      });
    },
  };
}

function buildTickerTabs(payload) {
  const host = $("[data-tickers]");
  host.innerHTML =
    `<button type="button" class="ticker-tab" role="tab" data-ticker="" aria-selected="false" ` +
    `style="--tint:${cssVar("--s-portfolio")}">` +
    `<span class="ticker-tab__dot"></span>Portfolio</button>` +
    payload.meta.tickers
      .map(
        (t) =>
          `<button type="button" class="ticker-tab" role="tab" data-ticker="${t}" ` +
          `aria-selected="false" style="--tint:${assetColor(t)}">` +
          `<span class="ticker-tab__dot"></span>${t}</button>`
      )
      .join("");

  const tabs = $$("[data-ticker]", host);
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () =>
      store.set({ ticker: tab.dataset.ticker || null, eventIndex: null })
    );
    tab.addEventListener("keydown", (event) => {
      let next = null;
      if (event.key === "ArrowRight") next = tabs[(index + 1) % tabs.length];
      else if (event.key === "ArrowLeft")
        next = tabs[(index - 1 + tabs.length) % tabs.length];
      if (!next) return;
      event.preventDefault();
      next.focus();
      store.set({ ticker: next.dataset.ticker || null, eventIndex: null });
    });
  });

  return {
    update(state) {
      tabs.forEach((tab) => {
        const on = (tab.dataset.ticker || null) === state.ticker;
        tab.setAttribute("aria-selected", String(on));
        tab.tabIndex = on ? 0 : -1;
      });
    },
  };
}

function buildPerformanceControls() {
  const modeHost = $("[data-performance-mode]");
  $$("button", modeHost).forEach((button) => {
    button.addEventListener("click", () =>
      store.set({ performanceMode: button.dataset.mode })
    );
  });

  const portfolioToggle = $("[data-toggle-portfolio]");
  const benchmarkToggle = $("[data-toggle-benchmark]");
  portfolioToggle.addEventListener("click", () => store.toggle("showPortfolio"));
  benchmarkToggle.addEventListener("click", () => store.toggle("showBenchmark"));

  return {
    update(state) {
      $$("button", modeHost).forEach((button) => {
        button.setAttribute(
          "aria-selected",
          String(button.dataset.mode === state.performanceMode)
        );
      });
      portfolioToggle.setAttribute("aria-pressed", String(state.showPortfolio));
      benchmarkToggle.setAttribute("aria-pressed", String(state.showBenchmark));
    },
  };
}

function buildMaControls() {
  const shortToggle = $("[data-toggle-ma-short]");
  const longToggle = $("[data-toggle-ma-long]");
  shortToggle.addEventListener("click", () => store.toggle("showMaShort"));
  longToggle.addEventListener("click", () => store.toggle("showMaLong"));

  return {
    update(state) {
      shortToggle.setAttribute("aria-pressed", String(state.showMaShort));
      longToggle.setAttribute("aria-pressed", String(state.showMaLong));
    },
  };
}

function buildBacktestControls() {
  const host = $("[data-backtest-mode]");
  $$("button", host).forEach((button) => {
    button.addEventListener("click", () =>
      store.set({ backtestMode: button.dataset.mode })
    );
  });

  return {
    update(state) {
      $$("button", host).forEach((button) => {
        button.setAttribute(
          "aria-selected",
          String(button.dataset.mode === state.backtestMode)
        );
      });
    },
  };
}

/** Scroll-spy driving the navigation underline. */
function buildNavSpy() {
  const links = $$("[data-navlink]");
  const sections = links
    .map((link) => document.getElementById(link.dataset.navlink))
    .filter(Boolean);

  observeSections(sections, (id) => store.set({ activeSection: id }));

  return {
    update(state) {
      links.forEach((link) => {
        link.setAttribute(
          "aria-current",
          String(link.dataset.navlink === state.activeSection)
        );
      });
    },
  };
}

/* -------------------------------------------------------------------------- */
/* KPI rail                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * `invert` marks metrics where a LARGER value is unfavorable, so the colour of
 * the difference row has to flip. Volatility qualifies: more of it is worse.
 *
 * Maximum drawdown deliberately does NOT, even though it is a "bad" quantity:
 * it is reported as a negative number, so a positive difference already means
 * the portfolio's drawdown was shallower than the benchmark's, which is
 * favorable. Inverting it would paint a deeper decline green.
 */
const KPI_DEFS = [
  { key: "ann_return", label: "Ann. return", format: (v) => pct(v, 2), tone: "sign" },
  { key: "ann_volatility", label: "Ann. volatility", format: (v) => pct(v, 2), invert: true },
  { key: "sharpe", label: "Sharpe ratio", format: (v) => ratio(v), tone: "sign" },
  { key: "max_drawdown", label: "Max drawdown", format: (v) => pct(v, 2), tone: "neg" },
  { key: "beta", label: "Beta", format: (v) => ratio(v) },
];

function buildKpiRail(payload) {
  const host = $("[data-kpis]");
  const benchmarkLabel = payload.meta.benchmark.label;

  const cells = (content) =>
    KPI_DEFS.map((def) => content(def)).join("") + content({ key: "total_return" });

  // The benchmark and difference rows carry a label only in the first column,
  // but every cell renders the label element so all six values in a row share
  // one baseline. An empty label in the other columns reserves the same space.
  const rowLabel = (text, first) =>
    first
      ? `<span class="kpi__rowlabel">${escapeHtml(text)}</span>`
      : `<span class="kpi__rowlabel" aria-hidden="true">&nbsp;</span>`;

  const cellsIndexed = (content) =>
    KPI_DEFS.map((def, i) => content(def, i === 0)).join("") +
    content({ key: "total_return" }, false);

  host.innerHTML =
    `<div class="kpis__row" role="row">` +
    cells(
      (def) =>
        `<div class="kpi" data-kpi="${def.key}" role="cell">` +
        `<span class="kpi__label">${def.label ?? "Total return in range"}</span>` +
        `<span class="kpi__value num" data-value>${EMPTY}</span></div>`
    ) +
    `</div>` +
    `<div class="kpis__row kpis__row--bench" role="row">` +
    cellsIndexed(
      (def, first) =>
        `<div class="kpi" role="cell">` +
        rowLabel(benchmarkLabel, first) +
        `<span class="kpi__bench num" data-bench="${def.key}">${EMPTY}</span></div>`
    ) +
    `</div>` +
    `<div class="kpis__row kpis__row--delta" role="row">` +
    cellsIndexed(
      (def, first) =>
        `<div class="kpi" role="cell">` +
        rowLabel("Difference", first) +
        `<span class="kpi__delta num" data-delta="${def.key}">${EMPTY}</span></div>`
    ) +
    `</div>`;

  const compact = $("[data-kpis-compact]");

  /** Difference between portfolio and benchmark, with its favorable sign. */
  const difference = (def, value, bench) => {
    if (def.key === "beta") return { text: "—", tone: "muted" };
    const delta = value - bench;
    const favorable = def.invert ? -delta : delta;
    const magnitude =
      def.key === "sharpe" ? ratio(Math.abs(delta)) : pct(Math.abs(delta), 2);
    return {
      text: (delta >= 0 ? "+" : "−") + magnitude,
      tone: signClass(favorable),
    };
  };

  return {
    update(state) {
      const block = payload.windows[state.window];

      for (const def of KPI_DEFS) {
        const value = block.portfolio[def.key];
        const bench = block.benchmark[def.key];

        const valueNode = $(`[data-kpi="${def.key}"] [data-value]`, host);
        valueNode.className = "kpi__value num";
        if (def.tone === "sign") {
          valueNode.classList.add(value >= 0 ? "kpi__value--pos" : "kpi__value--neg");
        } else if (def.tone === "neg") {
          valueNode.classList.add("kpi__value--neg");
        }
        countUp(valueNode, value, def.format, { duration: motion.reduced ? 0 : 700 });

        $(`[data-bench="${def.key}"]`, host).textContent = def.format(bench);

        const diff = difference(def, value, bench);
        const deltaNode = $(`[data-delta="${def.key}"]`, host);
        deltaNode.className = `kpi__delta num ${diff.tone}`;
        deltaNode.textContent = diff.text;
      }

      const totalNode = $('[data-kpi="total_return"] [data-value]', host);
      totalNode.className = `kpi__value num ${
        block.total_return.portfolio >= 0 ? "kpi__value--pos" : "kpi__value--neg"
      }`;
      countUp(totalNode, block.total_return.portfolio, (v) => pctSigned(v, 1), {
        duration: motion.reduced ? 0 : 700,
      });
      $('[data-bench="total_return"]', host).textContent = pctSigned(
        block.total_return.benchmark,
        1
      );
      const totalDelta =
        block.total_return.portfolio - block.total_return.benchmark;
      const totalDeltaNode = $('[data-delta="total_return"]', host);
      totalDeltaNode.className = `kpi__delta num ${signClass(totalDelta)}`;
      totalDeltaNode.textContent =
        (totalDelta >= 0 ? "+" : "−") + pct(Math.abs(totalDelta), 1);

      // Compact layout for narrow screens: one block per metric, each keeping
      // its value, the benchmark figure and the difference together.
      const blocks = [
        ...KPI_DEFS.map((def) => ({
          label: def.label,
          value: def.format(block.portfolio[def.key]),
          tone:
            def.tone === "sign"
              ? block.portfolio[def.key] >= 0
                ? "pos"
                : "neg"
              : def.tone === "neg"
                ? "neg"
                : "",
          bench: def.format(block.benchmark[def.key]),
          diff: difference(def, block.portfolio[def.key], block.benchmark[def.key]),
        })),
        {
          label: "Total return in range",
          value: pctSigned(block.total_return.portfolio, 1),
          tone: block.total_return.portfolio >= 0 ? "pos" : "neg",
          bench: pctSigned(block.total_return.benchmark, 1),
          diff: {
            text:
              (block.total_return.portfolio >= block.total_return.benchmark ? "+" : "−") +
              pct(Math.abs(block.total_return.portfolio - block.total_return.benchmark), 1),
            tone: signClass(block.total_return.portfolio - block.total_return.benchmark),
          },
        },
      ];

      compact.innerHTML = blocks
        .map(
          (b) =>
            `<div class="kpi-block">` +
            `<span class="kpi-block__label">${b.label}</span>` +
            `<span class="kpi-block__value num${
              b.tone ? ` kpi-block__value--${b.tone}` : ""
            }">${b.value}</span>` +
            `<span class="kpi-block__compare">` +
            `<span>${escapeHtml(benchmarkLabel)}<b class="num">${b.bench}</b></span>` +
            `<span>Difference<b class="num ${b.diff.tone}">${b.diff.text}</b></span>` +
            `</span></div>`
        )
        .join("");

      // Sub-annual ranges annualize a partial year; say so rather than
      // presenting an extrapolated figure as equivalent to the full sample.
      const caveat = $("[data-window-caveat]");
      if (block.short_window) {
        caveat.hidden = false;
        caveat.textContent =
          `${block.label} covers ${block.observations} trading sessions. ` +
          `Annualized figures extrapolate that sample to a full year and are volatile; ` +
          `total return in range is unaffected.`;
      } else {
        caveat.hidden = true;
      }
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Asset explorer                                                             */
/* -------------------------------------------------------------------------- */

function buildAssetPanel(payload) {
  const head = $("[data-asset-head]");
  const metrics = $("[data-asset-metrics]");

  return {
    update(state) {
      const block = payload.windows[state.window];
      const ticker = state.ticker;

      if (!ticker) {
        const ranked = payload.meta.tickers
          .map((t) => ({ t, value: block.assets[t].risk_contribution_pct }))
          .sort((a, b) => b.value - a.value);
        const top = ranked[0];

        head.innerHTML =
          `<h3 class="asset__name">` +
          `<span class="asset__swatch" style="background:${cssVar("--s-portfolio")}"></span>` +
          `Portfolio</h3>` +
          `<p class="asset__state">Weighted across ${payload.meta.tickers.length} holdings` +
          ` · select a holding above to inspect it</p>` +
          `<p class="note">${top.t} carries ${pct(top.value, 1)} of total portfolio ` +
          `risk on ${pct(block.assets[top.t].weight, 0)} of capital — the largest ` +
          `single concentration in the portfolio.</p>`;

        metrics.innerHTML = metricGrid([
          ["Annualized return", pct(block.portfolio.ann_return), signClass(block.portfolio.ann_return)],
          ["Annualized volatility", pct(block.portfolio.ann_volatility)],
          ["Sharpe ratio", ratio(block.portfolio.sharpe), signClass(block.portfolio.sharpe)],
          ["Beta", ratio(block.portfolio.beta)],
          ["Maximum drawdown", pct(block.portfolio.max_drawdown), "neg"],
          ["Total return in range", pctSigned(block.total_return.portfolio, 1), signClass(block.total_return.portfolio)],
        ]);
        return;
      }

      const asset = block.assets[ticker];
      const signal = payload.signals.summary[ticker];
      const color = assetColor(ticker);
      const gap = asset.risk_contribution_pct - asset.weight;

      head.innerHTML =
        `<h3 class="asset__name">` +
        `<span class="asset__swatch" style="background:${color}"></span>${ticker}</h3>` +
        `<p class="asset__state">` +
        `<span class="tag tag--${
          signal.trend === "Bullish" ? "bull" : signal.trend === "Bearish" ? "bear" : ""
        }">${escapeHtml(signal.trend)}</span>` +
        `<span>Close ${money(signal.latest_price)}</span>` +
        `<span>${escapeHtml(signal.last_crossover)}${
          signal.last_crossover_date ? ` · ${dateLong(signal.last_crossover_date)}` : ""
        }</span></p>` +
        `<p class="note">${
          gap >= 0
            ? `Contributes ${pct(gap, 1)} more of portfolio risk than of capital.`
            : `Contributes ${pct(Math.abs(gap), 1)} less of portfolio risk than of capital.`
        } Metrics shown over ${block.label}.</p>`;

      metrics.innerHTML = metricGrid([
        ["Annualized return", pct(asset.ann_return), signClass(asset.ann_return)],
        ["Annualized volatility", pct(asset.ann_volatility)],
        ["Sharpe ratio", ratio(asset.sharpe), signClass(asset.sharpe)],
        ["Beta", ratio(asset.beta)],
        ["Maximum drawdown", pct(asset.max_drawdown), "neg"],
        ["Portfolio weight", pct(asset.weight, 1)],
        ["Risk contribution", pct(asset.risk_contribution_pct, 1), signClass(-gap)],
        [`MA${payload.meta.assumptions.ma_short} / MA${payload.meta.assumptions.ma_long}`,
          `${money(signal.ma_short)} / ${money(signal.ma_long)}`],
      ]);
    },
  };
}

function metricGrid(rows) {
  return rows
    .map(
      ([label, value, tone = ""]) =>
        `<div><dt>${label}</dt><dd class="${tone}">${value}</dd></div>`
    )
    .join("");
}

/* -------------------------------------------------------------------------- */
/* Backtest comparison                                                        */
/* -------------------------------------------------------------------------- */

function buildBacktestMetrics(payload) {
  const host = $("[data-backtest-metrics]");
  const title = $("[data-backtest-title]");

  return {
    update(state) {
      const key = state.ticker ?? payload.meta.tickers[0];
      const summary = payload.signals.backtest[key];
      title.textContent = `${key} · buy & hold vs strategy`;

      const row = (label, a, b, tone = "") =>
        `<div><dt>${label}</dt><dd class="${tone}">${a}` +
        `<span class="muted" style="font-weight:400"> / ${b}</span></dd></div>`;

      host.innerHTML =
        row(
          "Total return",
          multiple(1 + summary.buy_hold.total_return),
          multiple(1 + summary.strategy.total_return)
        ) +
        row(
          "Annualized return",
          pct(summary.buy_hold.ann_return),
          pct(summary.strategy.ann_return)
        ) +
        row(
          "Annualized volatility",
          pct(summary.buy_hold.ann_volatility),
          pct(summary.strategy.ann_volatility)
        ) +
        row(
          "Maximum drawdown",
          pct(summary.buy_hold.max_drawdown),
          pct(summary.strategy.max_drawdown)
        ) +
        `<div><dt>Reading</dt><dd style="font-weight:400;font-size:12.5px;text-align:right;color:var(--ink-secondary)">` +
        `${
          summary.strategy.max_drawdown > summary.buy_hold.max_drawdown
            ? "The rule reduced drawdown"
            : "The rule deepened drawdown"
        } and ${
          summary.strategy.ann_return > summary.buy_hold.ann_return
            ? "raised"
            : "lowered"
        } annualized return on this holding.</dd></div>`;
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Monte Carlo summary                                                        */
/* -------------------------------------------------------------------------- */

function renderMonteCarloStats(payload) {
  const mc = payload.monte_carlo;

  const tiles = [
    ["Median terminal return", pctSigned(mc.summary.median_terminal_return), signClass(mc.summary.median_terminal_return)],
    ["5th percentile", pctSigned(mc.summary.p5_terminal_return), signClass(mc.summary.p5_terminal_return)],
    ["95th percentile", pctSigned(mc.summary.p95_terminal_return), signClass(mc.summary.p95_terminal_return)],
    ["95% VaR", pct(mc.var), "neg"],
    ["95% CVaR", pct(mc.cvar), "neg"],
    ["Expected terminal wealth", multiple(mc.summary.expected_terminal_wealth), ""],
  ];

  $("[data-mc-stats]").innerHTML = tiles
    .map(
      ([label, value, tone]) =>
        `<div class="statrail__item"><span class="statrail__label">${label}</span>` +
        `<span class="statrail__value ${tone}">${value}</span></div>`
    )
    .join("");

  const d = mc.downside_probabilities;
  $("[data-mc-downside]").innerHTML = [
    ["Ended below the starting value", d.loss],
    ["Ended more than 10% below", d.below_minus_10],
    ["Ended more than 20% below", d.below_minus_20],
    ["Ended more than 30% below", d.below_minus_30],
  ]
    .map(
      ([label, value]) =>
        `<div class="downside__row"><span>${label}</span><b>${pct(value, 2)}</b></div>`
    )
    .join("");

  // The required framing, stated in the sanctioned wording.
  $("[data-mc-framing]").textContent =
    `${pct(d.loss, 1)} of simulated one-year scenarios ended below the starting ` +
    `portfolio value. These are frequencies within a modelled distribution under ` +
    `the stated assumptions, not probabilities about the future. Value at Risk is ` +
    `reported as a positive loss magnitude at the 95% confidence level; ` +
    `Conditional VaR is the mean loss across the scenarios at or beyond it.`;
}

/* -------------------------------------------------------------------------- */
/* Methodology                                                                */
/* -------------------------------------------------------------------------- */

function renderMethodology(payload) {
  const meta = payload.meta;
  const a = meta.assumptions;
  const mc = payload.monte_carlo.config;

  const entries = [
    [
      "Data source",
      `<p>End-of-day adjusted closing prices for ${meta.tickers.join(", ")} and ` +
        `${escapeHtml(meta.benchmark.label)} (<code>${escapeHtml(meta.benchmark.symbol)}</code>), ` +
        `retrieved from Yahoo Finance via <code>yfinance</code>. Rows with any missing ` +
        `value are dropped with a warning rather than filled. This is a periodically ` +
        `refreshed end-of-day dataset, not streaming market data.</p>`,
    ],
    [
      "Analysis period",
      `<p>${dateLong(meta.first_market_date)} to ${dateLong(meta.latest_market_date)}, ` +
        `${count(meta.trading_days)} trading sessions. The six range selectors each ` +
        `have their own metric set, recomputed in Python on the windowed return ` +
        `series — the browser selects a precomputed set and never recalculates.</p>`,
    ],
    [
      "Portfolio weights",
      `<p>${meta.tickers.map((t) => `${t} ${pct(meta.weights[t], 0)}`).join(", ")}. ` +
        `Fixed weights, validated to sum to 1.00. No rebalancing is modelled.</p>`,
    ],
    [
      "Risk-free rate",
      `<p>${pct(a.risk_free_rate, 2)}, a documented standing assumption in ` +
        `<code>config.py</code> approximating the 3-month US Treasury bill yield at ` +
        `project setup. It is not a live feed and should be revisited periodically. ` +
        `It affects the Sharpe ratio only.</p>`,
    ],
    [
      "Annualization",
      `<p>${a.trading_days_per_year} trading days per year. Returns are geometric ` +
        `(CAGR-style); volatility is the sample standard deviation of daily returns ` +
        `scaled by the square root of ${a.trading_days_per_year}. Beta uses sample ` +
        `covariance over sample variance against the benchmark.</p>` +
        `<p>Annualizing a sub-annual range extrapolates a partial year and is ` +
        `inherently noisy, so those ranges are flagged on the page.</p>`,
    ],
    [
      `MA${a.ma_short} / MA${a.ma_long} methodology`,
      `<p>Simple moving averages over ${a.ma_short} and ${a.ma_long} sessions. The ` +
        `lead-in period is left undefined rather than backfilled, and no trend is ` +
        `asserted before both averages exist. A crossover requires a genuine ` +
        `transition: the previous session on one side and the current session ` +
        `strictly on the other, with both sessions' averages available.</p>` +
        `<p>Averages are always computed on the full price history, because a ` +
        `${a.ma_long}-session average requires ${a.ma_long} prior observations to ` +
        `exist at all.</p>`,
    ],
    [
      "No-lookahead backtest treatment",
      `<p>The strategy holds a long position when the short average is above the ` +
        `long one, with the position <strong>shifted one trading day</strong> before ` +
        `being applied to returns. The return on day <em>t</em> therefore uses only ` +
        `the position known at the close of day <em>t−1</em>, so no future ` +
        `information enters the result.</p>` +
        `<p>${
          a.transaction_costs_modelled ? "Transaction costs are modelled." : "No transaction costs, slippage or taxes are modelled."
        } This is a simplified educational exercise, not a strategy recommendation, ` +
        `and the rule is not superior in general — it reduces drawdown on some ` +
        `holdings and gives up return on others.</p>`,
    ],
    [
      "Monte Carlo assumptions",
      `<p>${count(mc.simulations)} scenarios over ${mc.days} trading days. Daily ` +
        `portfolio returns are drawn from a normal distribution parameterized by the ` +
        `weighted portfolio's historical daily mean and standard deviation over the ` +
        `full sample, then compounded. Random seed ${mc.seed}, so the result is ` +
        `reproducible.</p>` +
        `<p>Returns are assumed independent and identically distributed. Real markets ` +
        `exhibit fat tails, volatility clustering, regime changes and structural ` +
        `breaks that a normal random walk does not capture, so extreme outcomes are ` +
        `under-represented.</p>`,
    ],
    [
      "Limitations",
      `<p>Fixed weights with no rebalancing, no dividends beyond what adjusted ` +
        `closes already incorporate, no transaction costs, no taxes, and a single ` +
        `static risk-free rate. Five holdings is a concentrated portfolio, and the ` +
        `sample period covers one particular market regime.</p>` +
        `<p>Simulated figures are frequencies within a modelled distribution, not ` +
        `real-world probabilities. Historical performance is not a prediction of ` +
        `future performance.</p>`,
    ],
    [
      "Provenance",
      `<p>Every figure on this page is computed in Python by the modules under ` +
        `<code>src/</code>, serialized by ` +
        `<code>scripts/export_dashboard_data.py</code>, and independently ` +
        `cross-checked against the committed analytics tables by ` +
        `<code>scripts/validate_dashboard_data.py</code> before deployment. A ` +
        `separate guard rejects any hardcoded financial value in the front end.</p>` +
        `<p>Published from commit <code>${escapeHtml(meta.git_sha)}</code> at ` +
        `${timestamp(meta.generated_at_utc)}. If a weekly refresh fails any gate, ` +
        `nothing is deployed and the previously published dashboard stays live.</p>`,
    ],
  ];

  $("[data-methodology]").innerHTML = entries
    .map(
      ([title, body], index) =>
        `<details${index === 0 ? " open" : ""}>` +
        `<summary>${title}</summary>` +
        `<div class="method__body">${body}</div></details>`
    )
    .join("");
}
