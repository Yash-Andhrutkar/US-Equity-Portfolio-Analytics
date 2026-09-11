/**
 * Performance panel: cumulative return or drawdown, portfolio against the
 * benchmark, with the selected holding overlaid.
 *
 * The time range pans this chart. Cumulative series are rebased to the window
 * start with the identity in format.rebaseSeries; drawdown is shown as the
 * analytics layer computed it over the full sample, because a drawdown
 * re-anchored to an arbitrary window start would understate a decline that
 * began before that date.
 */

import {
  ChartFrame,
  Tooltip,
  animateDraw,
  areaPath,
  extent,
  linePath,
  niceTicks,
  padDomain,
  scaleIndex,
  scaleLinear,
  svgEl,
  tipHead,
  tipNote,
  tipRow,
  trackPointer,
} from "./base.js";
import {
  assetColor,
  cssVar,
  dateLong,
  dateShort,
  pct,
  pctSigned,
  rebaseSeries,
} from "../format.js";
import { debounce, motion, onFirstVisible } from "../motion.js";

export function createPerformanceChart(container, payload) {
  const frame = new ChartFrame(container, {
    margin: { top: 16, right: 20, bottom: 32, left: 62 },
    aspect: 0.36,
    minHeight: 260,
    maxHeight: 560,
    label: "Portfolio cumulative performance against the benchmark",
  });
  const tooltip = new Tooltip(container);

  let state = null;
  let drawn = false;
  let visible = false;

  onFirstVisible(container, () => {
    visible = true;
    if (state) render();
  });

  const render = () => {
    if (!state) return;
    const { mode, windowKey, ticker, showPortfolio, showBenchmark } = state;

    const block = payload.windows[windowKey];
    const startIndex = block.start_index;
    const dates = payload.series.return_dates.slice(startIndex);
    const isDrawdown = mode === "drawdown";

    const series = [];
    if (isDrawdown) {
      if (showPortfolio) {
        series.push({
          key: "portfolio",
          label: "Portfolio",
          values: payload.series.drawdown.PORTFOLIO.slice(startIndex),
          color: cssVar("--s-portfolio"),
          area: true,
        });
      }
      if (showBenchmark) {
        series.push({
          key: "benchmark",
          label: payload.meta.benchmark.label,
          values: payload.series.drawdown.BENCHMARK.slice(startIndex),
          color: cssVar("--s-benchmark"),
          dashed: true,
        });
      }
    } else {
      if (showPortfolio) {
        series.push({
          key: "portfolio",
          label: "Portfolio",
          values: rebaseSeries(payload.series.cumulative.PORTFOLIO, startIndex),
          color: cssVar("--s-portfolio"),
          area: true,
        });
      }
      if (showBenchmark) {
        series.push({
          key: "benchmark",
          label: payload.meta.benchmark.label,
          values: rebaseSeries(payload.series.cumulative.BENCHMARK, startIndex),
          color: cssVar("--s-benchmark"),
          dashed: true,
        });
      }
      if (ticker) {
        series.push({
          key: ticker,
          label: ticker,
          values: rebaseSeries(payload.series.cumulative[ticker], startIndex),
          color: assetColor(ticker),
          thin: true,
        });
      }
    }

    frame.reset();

    if (!series.length) {
      const empty = svgEl(
        "text",
        {
          x: frame.plotWidth / 2,
          y: frame.plotHeight / 2,
          "text-anchor": "middle",
          fill: cssVar("--ink-tertiary"),
          "font-size": 13,
        },
        frame.plot
      );
      empty.textContent = "Enable a series to plot";
      renderLegend(container, series, isDrawdown, payload);
      return;
    }

    const xScale = scaleIndex(dates.length, [0, frame.plotWidth]);
    const [rawMin, rawMax] = extent(...series.map((s) => s.values));
    const domain = isDrawdown
      ? [Math.min(rawMin, 0) * 1.08, 0]
      : padDomain([Math.min(rawMin, 0), rawMax], 0.08);
    const yScale = scaleLinear(domain, [frame.plotHeight, 0]);

    frame.yAxis(yScale, {
      format: (value) => pct(value, 0),
      ticks: niceTicks(domain[0], domain[1], 5),
      parityAt: 0,
    });

    frame.xAxis(
      frame.dateTicks(dates, xScale, 7).map(({ x, iso }) => ({
        x,
        label: dateShort(iso),
      }))
    );

    for (const entry of series) {
      if (!entry.area) continue;
      const fill = frame.gradient(
        `grad-perf-${entry.key}`,
        entry.color,
        isDrawdown ? { from: 0.02, to: 0.24 } : { from: 0.2, to: 0.01 }
      );
      svgEl(
        "path",
        { d: areaPath(entry.values, xScale, yScale, 0), fill, stroke: "none" },
        frame.plot
      );
    }

    for (const entry of series) {
      const classes = ["series"];
      if (entry.dashed) classes.push("series--benchmark");
      if (entry.thin) classes.push("series--thin");
      const node = svgEl(
        "path",
        {
          class: classes.join(" "),
          d: linePath(entry.values, xScale, yScale),
          stroke: entry.color,
        },
        frame.plot
      );
      if (!drawn && visible && !motion.reduced) animateDraw(node);
    }
    if (visible) drawn = true;

    // Crosshair inspection.
    const overlay = svgEl("g", { style: "pointer-events:none" }, frame.plot);
    const crosshair = svgEl(
      "line",
      { class: "crosshair", y1: 0, y2: frame.plotHeight, opacity: 0 },
      overlay
    );
    const markers = series.map((entry) =>
      svgEl("circle", { class: "marker", r: 3.6, fill: entry.color, opacity: 0 }, overlay)
    );

    trackPointer(frame, dates.length, {
      onMove: (index, position) => {
        const x = xScale(index);
        crosshair.setAttribute("x1", x.toFixed(2));
        crosshair.setAttribute("x2", x.toFixed(2));
        crosshair.setAttribute("opacity", 1);

        const rows = series
          .map((entry, i) => {
            const value = entry.values[index];
            const marker = markers[i];
            if (value === null || value === undefined || Number.isNaN(value)) {
              marker.setAttribute("opacity", 0);
              return "";
            }
            marker.setAttribute("cx", x.toFixed(2));
            marker.setAttribute("cy", yScale(value).toFixed(2));
            marker.setAttribute("opacity", 1);
            return tipRow(
              entry.label,
              isDrawdown ? pct(value) : pctSigned(value),
              entry.color
            );
          })
          .join("");

        tooltip.show(
          tipHead(dateLong(dates[index])) +
            rows +
            tipNote(
              isDrawdown
                ? "Decline from the running peak of the full sample."
                : `Cumulative return rebased to ${dateLong(dates[0])}.`
            ),
          position.x,
          position.y
        );
      },
      onLeave: () => {
        tooltip.hide();
        crosshair.setAttribute("opacity", 0);
        markers.forEach((marker) => marker.setAttribute("opacity", 0));
      },
    });

    renderLegend(container, series, isDrawdown, payload);
  };

  const onResize = debounce(() => {
    drawn = true;
    render();
  }, 180);
  window.addEventListener("resize", onResize);

  return {
    update(next) {
      const modeChanged = state && state.mode !== next.mode;
      state = next;
      if (modeChanged) drawn = false;
      render();
    },
    destroy() {
      window.removeEventListener("resize", onResize);
      tooltip.destroy();
    },
  };
}

function renderLegend(container, series, isDrawdown, payload) {
  const existing = container.querySelector(".chart__legend");
  if (existing) existing.remove();
  if (!series.length) return;

  const legend = document.createElement("div");
  legend.className = "chart__legend";

  legend.innerHTML = series
    .map((entry) => {
      const total =
        !isDrawdown && entry.values.length
          ? ` ${pctSigned(entry.values[entry.values.length - 1], 1)}`
          : "";
      return (
        `<span class="chart__legend-item" style="color:${entry.color}">` +
        `<span class="chart__legend-key${entry.dashed ? " chart__legend-key--dash" : ""}"></span>` +
        `<span style="color:var(--ink-secondary)">${entry.label}` +
        `<span class="num" style="color:var(--ink-tertiary)">${total}</span></span></span>`
      );
    })
    .join("");

  if (isDrawdown) {
    const note = document.createElement("span");
    note.className = "chart__legend-item";
    note.style.color = "var(--ink-quaternary)";
    note.textContent = "Measured against the full-sample peak";
    legend.appendChild(note);
  }

  container.appendChild(legend);
}

/* -------------------------------------------------------------------------- */
/* Standalone drawdown chart for the risk section                             */
/* -------------------------------------------------------------------------- */

export function createDrawdownChart(container, payload) {
  const frame = new ChartFrame(container, {
    margin: { top: 14, right: 20, bottom: 30, left: 58 },
    aspect: 0.24,
    minHeight: 200,
    maxHeight: 300,
    label: "Portfolio drawdown against the benchmark over the full sample",
  });
  const tooltip = new Tooltip(container);

  let state = null;

  const render = () => {
    const dates = payload.series.return_dates;
    const portfolio = payload.series.drawdown.PORTFOLIO;
    const benchmark = payload.series.drawdown.BENCHMARK;

    frame.reset();

    const xScale = scaleIndex(dates.length, [0, frame.plotWidth]);
    const [min] = extent(portfolio, benchmark);
    const yScale = scaleLinear([min * 1.08, 0], [frame.plotHeight, 0]);

    frame.yAxis(yScale, { format: (v) => pct(v, 0), parityAt: 0 });
    frame.xAxis(
      frame.dateTicks(dates, xScale, 7).map(({ x, iso }) => ({ x, label: dateShort(iso) }))
    );

    const fill = frame.gradient("grad-dd", cssVar("--negative"), { from: 0.03, to: 0.26 });
    svgEl(
      "path",
      { d: areaPath(portfolio, xScale, yScale, 0), fill, stroke: "none" },
      frame.plot
    );

    svgEl(
      "path",
      {
        class: "series series--benchmark",
        d: linePath(benchmark, xScale, yScale),
        stroke: cssVar("--s-benchmark"),
      },
      frame.plot
    );
    svgEl(
      "path",
      {
        class: "series",
        d: linePath(portfolio, xScale, yScale),
        stroke: cssVar("--s-portfolio"),
      },
      frame.plot
    );

    // Mark the trough — the single most relevant point on a drawdown chart.
    let troughIndex = 0;
    for (let i = 1; i < portfolio.length; i += 1) {
      if (portfolio[i] < portfolio[troughIndex]) troughIndex = i;
    }
    svgEl(
      "circle",
      {
        class: "marker",
        cx: xScale(troughIndex).toFixed(2),
        cy: yScale(portfolio[troughIndex]).toFixed(2),
        r: 4,
        fill: cssVar("--negative"),
      },
      frame.plot
    );
    const troughLabel = svgEl(
      "text",
      {
        x: xScale(troughIndex).toFixed(2),
        y: (yScale(portfolio[troughIndex]) + 18).toFixed(2),
        "text-anchor": "middle",
        fill: cssVar("--negative"),
        "font-size": 11,
        "font-weight": 600,
      },
      frame.plot
    );
    troughLabel.textContent = `${pct(portfolio[troughIndex])} · ${dateShort(dates[troughIndex])}`;

    const overlay = svgEl("g", { style: "pointer-events:none" }, frame.plot);
    const crosshair = svgEl(
      "line",
      { class: "crosshair", y1: 0, y2: frame.plotHeight, opacity: 0 },
      overlay
    );

    trackPointer(frame, dates.length, {
      onMove: (index, position) => {
        const x = xScale(index);
        crosshair.setAttribute("x1", x.toFixed(2));
        crosshair.setAttribute("x2", x.toFixed(2));
        crosshair.setAttribute("opacity", 1);
        tooltip.show(
          tipHead(dateLong(dates[index])) +
            tipRow("Portfolio", pct(portfolio[index]), cssVar("--s-portfolio")) +
            tipRow(
              payload.meta.benchmark.label,
              pct(benchmark[index]),
              cssVar("--s-benchmark")
            ) +
            tipNote("Decline from the running peak of the full sample."),
          position.x,
          position.y
        );
      },
      onLeave: () => {
        tooltip.hide();
        crosshair.setAttribute("opacity", 0);
      },
    });
  };

  const onResize = debounce(render, 180);
  window.addEventListener("resize", onResize);

  return {
    update(next) {
      // Drawdown is a full-sample view; it does not depend on the range, but
      // it is re-rendered on selection so the panel stays in step.
      if (state === null) render();
      state = next;
    },
    destroy() {
      window.removeEventListener("resize", onResize);
      tooltip.destroy();
    },
  };
}
