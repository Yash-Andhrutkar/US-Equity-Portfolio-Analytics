/**
 * Technical signals: price with its moving averages and crossover markers,
 * the crossover timeline, and the buy-and-hold versus strategy backtest.
 *
 * The moving-average chart shows the full price history rather than
 * truncating to the selected range. A 200-day average needs 200 prior
 * observations, so a short window would either have no long average or would
 * recompute it on a shorter lookback — a different indicator wearing the same
 * name. The window is shaded in place instead.
 */

import {
  ChartFrame,
  Tooltip,
  animateDraw,
  extent,
  linePath,
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
  money,
  multiple,
} from "../format.js";
import { debounce, motion, onFirstVisible } from "../motion.js";

/* -------------------------------------------------------------------------- */
/* Price and moving averages                                                  */
/* -------------------------------------------------------------------------- */

export function createSignalChart(container, payload) {
  const frame = new ChartFrame(container, {
    margin: { top: 16, right: 20, bottom: 32, left: 62 },
    aspect: 0.34,
    minHeight: 260,
    maxHeight: 520,
    label: "Price with moving averages and crossover markers",
  });
  const tooltip = new Tooltip(container);
  const assumptions = payload.meta.assumptions;

  let state = null;
  let drawn = false;
  let visible = false;

  onFirstVisible(container, () => {
    visible = true;
    if (state) render();
  });

  const render = () => {
    if (!state) return;
    const { ticker, showMaShort, showMaLong, windowKey, eventIndex } = state;
    const key = ticker ?? payload.meta.tickers[0];

    const block = payload.series.moving_averages[key];
    const dates = payload.series.dates;
    const color = assetColor(key);

    frame.reset();

    const xScale = scaleIndex(dates.length, [0, frame.plotWidth]);
    const active = [block.price];
    if (showMaShort) active.push(block.ma_short);
    if (showMaLong) active.push(block.ma_long);
    const yScale = scaleLinear(padDomain(extent(...active), 0.07), [frame.plotHeight, 0]);

    frame.yAxis(yScale, { format: (value) => money(value).replace(/\.00$/, "") });
    frame.xAxis(
      frame.dateTicks(dates, xScale, 7).map(({ x, iso }) => ({ x, label: dateShort(iso) }))
    );

    // Shade the history outside the selected range, so the global range
    // control still means something on a chart that must show everything.
    const windowStart = payload.windows[windowKey].start_index + 1;
    if (windowStart > 1) {
      svgEl(
        "rect",
        {
          x: 0,
          y: 0,
          width: Math.max(0, xScale(windowStart)).toFixed(2),
          height: frame.plotHeight,
          fill: "rgba(5,7,12,0.58)",
        },
        frame.plot
      );
      svgEl(
        "line",
        {
          x1: xScale(windowStart).toFixed(2),
          x2: xScale(windowStart).toFixed(2),
          y1: 0,
          y2: frame.plotHeight,
          stroke: cssVar("--rule-strong"),
          "stroke-dasharray": "3 3",
        },
        frame.plot
      );
    }

    const lines = [{ values: block.price, color, width: 1.9, label: `${key} close` }];
    if (showMaShort) {
      lines.push({
        values: block.ma_short,
        color: cssVar("--azure-bright"),
        width: 1.5,
        label: `MA${assumptions.ma_short}`,
      });
    }
    if (showMaLong) {
      lines.push({
        values: block.ma_long,
        color: cssVar("--s-benchmark"),
        width: 1.5,
        label: `MA${assumptions.ma_long}`,
      });
    }

    for (const line of lines) {
      const node = svgEl(
        "path",
        {
          class: "series",
          d: linePath(line.values, xScale, yScale),
          stroke: line.color,
          "stroke-width": line.width,
        },
        frame.plot
      );
      if (!drawn && visible && !motion.reduced) animateDraw(node);
    }
    if (visible) drawn = true;

    // Crossover markers for this holding.
    const dateIndex = new Map(dates.map((iso, index) => [iso, index]));
    const events = payload.signals.events
      .map((event, index) => ({ ...event, globalIndex: index }))
      .filter((event) => event.ticker === key);

    for (const event of events) {
      const index = dateIndex.get(event.date);
      if (index === undefined) continue;
      const isGolden = event.signal === "Golden Cross";
      const active_ = eventIndex === event.globalIndex;

      const marker = svgEl(
        "circle",
        {
          cx: xScale(index).toFixed(2),
          cy: yScale(event.price).toFixed(2),
          r: active_ ? 7 : 4.4,
          fill: isGolden ? cssVar("--positive") : cssVar("--negative"),
          "fill-opacity": 0.85,
          stroke: cssVar("--bg-base"),
          "stroke-width": 1.6,
          style: "cursor:pointer",
          tabindex: 0,
          role: "button",
          "aria-label": `${event.signal} on ${key}, ${dateLong(event.date)}`,
        },
        frame.plot
      );

      const describe = (clientX, clientY) => {
        const bounds = container.getBoundingClientRect();
        tooltip.show(
          tipHead(dateLong(event.date)) +
            tipRow(
              event.signal,
              key,
              isGolden ? cssVar("--positive") : cssVar("--negative")
            ) +
            tipRow("Close", money(event.price)) +
            tipRow(`MA${assumptions.ma_short}`, money(event.ma_short)) +
            tipRow(`MA${assumptions.ma_long}`, money(event.ma_long)) +
            tipNote(
              `The ${assumptions.ma_short}-day average crossed ` +
                `${isGolden ? "above" : "below"} the ${assumptions.ma_long}-day average ` +
                `on this date. A crossover describes past price behaviour, not a forecast.`
            ),
          clientX - bounds.left,
          clientY - bounds.top
        );
      };

      marker.addEventListener("pointerenter", (e) => describe(e.clientX, e.clientY));
      marker.addEventListener("focus", () => {
        const r = marker.getBoundingClientRect();
        describe(r.left + r.width / 2, r.top);
      });
      marker.addEventListener("pointerleave", () => tooltip.hide());
      marker.addEventListener("blur", () => tooltip.hide());
    }

    const overlay = svgEl("g", { style: "pointer-events:none" }, frame.plot);
    const crosshair = svgEl(
      "line",
      { class: "crosshair", y1: 0, y2: frame.plotHeight, opacity: 0 },
      overlay
    );
    const markers = lines.map((line) =>
      svgEl("circle", { class: "marker", r: 3.4, fill: line.color, opacity: 0 }, overlay)
    );

    trackPointer(frame, dates.length, {
      onMove: (index, position) => {
        const x = xScale(index);
        crosshair.setAttribute("x1", x.toFixed(2));
        crosshair.setAttribute("x2", x.toFixed(2));
        crosshair.setAttribute("opacity", 1);

        const rows = lines
          .map((line, i) => {
            const value = line.values[index];
            const marker = markers[i];
            if (value === null || value === undefined) {
              marker.setAttribute("opacity", 0);
              return tipRow(line.label, "not yet available");
            }
            marker.setAttribute("cx", x.toFixed(2));
            marker.setAttribute("cy", yScale(value).toFixed(2));
            marker.setAttribute("opacity", 1);
            return tipRow(line.label, money(value), line.color);
          })
          .join("");

        const trend = block.trend[index];

        tooltip.show(
          tipHead(dateLong(dates[index])) +
            rows +
            tipRow("Trend", trend) +
            (trend === "Unavailable"
              ? tipNote(
                  `Before the ${assumptions.ma_long}-day average has enough history, no trend is asserted.`
                )
              : ""),
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

    renderLegend(container, key, lines, payload);
  };

  const onResize = debounce(() => {
    drawn = true;
    render();
  }, 180);
  window.addEventListener("resize", onResize);

  return {
    update(next) {
      if (state && state.ticker !== next.ticker) drawn = false;
      state = next;
      render();
    },
    destroy() {
      window.removeEventListener("resize", onResize);
      tooltip.destroy();
    },
  };
}

function renderLegend(container, key, lines, payload) {
  const existing = container.querySelector(".chart__legend");
  if (existing) existing.remove();

  const signal = payload.signals.summary[key];
  const legend = document.createElement("div");
  legend.className = "chart__legend";

  legend.innerHTML =
    lines
      .map(
        (line) =>
          `<span class="chart__legend-item" style="color:${line.color}">` +
          `<span class="chart__legend-key"></span>` +
          `<span style="color:var(--ink-secondary)">${line.label}</span></span>`
      )
      .join("") +
    `<span class="chart__legend-item" style="color:var(--positive)">` +
    `<span class="chart__legend-key" style="height:7px;width:7px;border-radius:50%"></span>` +
    `<span style="color:var(--ink-secondary)">Golden cross</span></span>` +
    `<span class="chart__legend-item" style="color:var(--negative)">` +
    `<span class="chart__legend-key" style="height:7px;width:7px;border-radius:50%"></span>` +
    `<span style="color:var(--ink-secondary)">Death cross</span></span>` +
    `<span class="chart__legend-item" style="color:var(--ink-quaternary);margin-left:auto">` +
    `Current trend: <strong style="color:var(--ink-primary)">${signal.trend}</strong>` +
    (signal.last_crossover_date
      ? ` · last ${signal.last_crossover.toLowerCase()} ${dateLong(signal.last_crossover_date)}`
      : "") +
    `</span>`;

  container.appendChild(legend);
}

/* -------------------------------------------------------------------------- */
/* Crossover timeline                                                         */
/* -------------------------------------------------------------------------- */

export function createSignalTimeline(container, payload, { onSelect }) {
  const tooltip = new Tooltip(container);
  const dates = payload.series.dates;
  const first = new Date(`${dates[0]}T00:00:00Z`).getTime();
  const last = new Date(`${dates[dates.length - 1]}T00:00:00Z`).getTime();
  const span = Math.max(1, last - first);

  container.innerHTML =
    '<div class="timeline__track"></div>' +
    `<div class="timeline__scale"><span>${dateShort(dates[0])}</span>` +
    `<span>${dateShort(dates[dates.length - 1])}</span></div>`;

  const track = container.querySelector(".timeline__track");

  return {
    update({ ticker, eventIndex }) {
      track.innerHTML = "";

      payload.signals.events.forEach((event, index) => {
        const when = new Date(`${event.date}T00:00:00Z`).getTime();
        const isGolden = event.signal === "Golden Cross";
        const dimmed = ticker && event.ticker !== ticker;

        const dot = document.createElement("button");
        dot.type = "button";
        dot.className = `timeline__event timeline__event--${isGolden ? "golden" : "death"}`;
        dot.style.left = `${((when - first) / span) * 100}%`;
        dot.style.opacity = dimmed ? "0.24" : "1";
        if (index === eventIndex) {
          dot.style.transform = "scale(1.6)";
          dot.style.zIndex = "3";
        }
        dot.setAttribute(
          "aria-label",
          `${event.signal} on ${event.ticker}, ${dateLong(event.date)}`
        );

        const describe = (clientX, clientY) => {
          const bounds = container.getBoundingClientRect();
          tooltip.show(
            tipHead(dateLong(event.date)) +
              tipRow(
                event.signal,
                event.ticker,
                isGolden ? cssVar("--positive") : cssVar("--negative")
              ) +
              tipRow("Close", money(event.price)),
            clientX - bounds.left,
            clientY - bounds.top
          );
        };

        dot.addEventListener("pointerenter", (e) => describe(e.clientX, e.clientY));
        dot.addEventListener("pointerleave", () => tooltip.hide());
        dot.addEventListener("focus", () => {
          const r = dot.getBoundingClientRect();
          describe(r.left + r.width / 2, r.top);
        });
        dot.addEventListener("blur", () => tooltip.hide());
        dot.addEventListener("click", () => onSelect(event.ticker, index));

        track.appendChild(dot);
      });
    },
    destroy() {
      tooltip.destroy();
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Buy and hold vs moving-average strategy                                    */
/* -------------------------------------------------------------------------- */

export function createBacktestChart(container, payload) {
  const frame = new ChartFrame(container, {
    margin: { top: 14, right: 20, bottom: 30, left: 54 },
    aspect: 0.46,
    minHeight: 240,
    maxHeight: 360,
    label: "Growth of one dollar: buy and hold against the moving-average strategy",
  });
  const tooltip = new Tooltip(container);

  let state = null;

  const render = () => {
    if (!state) return;
    const { ticker, backtestMode } = state;
    const key = ticker ?? payload.meta.tickers[0];
    const block = payload.series.backtest[key];
    const dates = payload.series.dates;

    const lines = [];
    if (backtestMode !== "strategy") {
      lines.push({
        label: "Buy & hold",
        values: block.buy_hold_wealth,
        color: assetColor(key),
        dashed: false,
      });
    }
    if (backtestMode !== "buy_hold") {
      lines.push({
        label: "MA strategy",
        values: block.strategy_wealth,
        color: cssVar("--azure-bright"),
        dashed: true,
      });
    }

    frame.reset();

    const xScale = scaleIndex(dates.length, [0, frame.plotWidth]);
    const yScale = scaleLinear(
      padDomain(extent(...lines.map((l) => l.values), [1]), 0.08),
      [frame.plotHeight, 0]
    );

    frame.yAxis(yScale, { format: (value) => `${value.toFixed(1)}×`, parityAt: 1 });
    frame.xAxis(
      frame.dateTicks(dates, xScale, 6).map(({ x, iso }) => ({ x, label: dateShort(iso) }))
    );

    // Shade the periods the strategy held cash — the mechanism behind its
    // shallower drawdown, made visible.
    if (backtestMode !== "buy_hold") {
      let runStart = null;
      for (let i = 0; i <= block.position.length; i += 1) {
        const flat = block.position[i] === 0;
        if (flat && runStart === null) runStart = i;
        if ((!flat || i === block.position.length) && runStart !== null) {
          svgEl(
            "rect",
            {
              x: xScale(runStart).toFixed(2),
              y: 0,
              width: Math.max(0.6, xScale(i) - xScale(runStart)).toFixed(2),
              height: frame.plotHeight,
              fill: "rgba(255,255,255,0.035)",
            },
            frame.plot
          );
          runStart = null;
        }
      }
    }

    for (const line of lines) {
      svgEl(
        "path",
        {
          class: `series${line.dashed ? " series--benchmark" : ""}`,
          d: linePath(line.values, xScale, yScale),
          stroke: line.color,
        },
        frame.plot
      );
    }

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
            lines
              .map((line) => tipRow(line.label, multiple(line.values[index]), line.color))
              .join("") +
            tipRow("Strategy position", block.position[index] === 1 ? "Invested" : "In cash") +
            tipNote(
              "Growth of one dollar. Positions are lagged one trading day and no transaction costs are modelled."
            ),
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
      state = next;
      render();
    },
    destroy() {
      window.removeEventListener("resize", onResize);
      tooltip.destroy();
    },
  };
}
