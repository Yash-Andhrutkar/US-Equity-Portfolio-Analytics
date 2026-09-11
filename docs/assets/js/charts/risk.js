/**
 * Allocation and risk: capital weight against risk contribution, the
 * risk/return scatter, the correlation matrix, and the diversification map.
 *
 * The weight-versus-risk pairing is the analytically interesting view in this
 * portfolio, so it gets the most direct encoding: two bars on a shared scale
 * where the gap between them is the point.
 *
 * The diversification map plots holdings by multidimensional scaling of the
 * correlation distance matrix d_ij = sqrt(2(1 - rho_ij)), computed in Python.
 * It is a standard way to read correlation structure: distance between two
 * marks approximates how independently they have behaved.
 */

import {
  ChartFrame,
  Tooltip,
  extent,
  niceTicks,
  padDomain,
  scaleLinear,
  svgEl,
  tipHead,
  tipNote,
  tipRow,
} from "./base.js";
import {
  assetColor,
  correlationColor,
  cssVar,
  pct,
  ratio,
  signClass,
} from "../format.js";
import { debounce } from "../motion.js";

/* -------------------------------------------------------------------------- */
/* Capital weight vs risk contribution                                        */
/* -------------------------------------------------------------------------- */

export function createWeightRisk(container, payload, { onSelect }) {
  const tickers = payload.meta.tickers;

  container.innerHTML =
    `<div class="wrisk">` +
    tickers
      .map(
        (ticker) => `
        <button type="button" class="wrisk__row" data-wrisk-ticker="${ticker}"
                aria-selected="false"
                aria-label="${ticker}: select to inspect capital weight and risk contribution">
          <span class="wrisk__ticker">
            <span class="wrisk__dot" style="background:${assetColor(ticker)}"></span>${ticker}
          </span>
          <span class="wrisk__bars">
            <span class="wrisk__track"><span class="wrisk__fill wrisk__fill--weight" data-fill="weight"></span></span>
            <span class="wrisk__track"><span class="wrisk__fill wrisk__fill--risk" data-fill="risk"></span></span>
          </span>
          <span class="wrisk__readout" data-readout></span>
        </button>`
      )
      .join("") +
    `</div>`;

  container.querySelectorAll("[data-wrisk-ticker]").forEach((row) => {
    row.addEventListener("click", () => onSelect(row.dataset.wriskTicker));
  });

  return {
    update({ windowKey, ticker }) {
      const assets = payload.windows[windowKey].assets;

      // Both bars share one scale so the quantities are directly comparable.
      const largest = Math.max(
        ...tickers.flatMap((t) => [assets[t].weight, assets[t].risk_contribution_pct])
      );

      container.querySelectorAll("[data-wrisk-ticker]").forEach((row) => {
        const key = row.dataset.wriskTicker;
        const asset = assets[key];
        const scale = (value) => `${((value / largest) * 100).toFixed(1)}%`;

        row.querySelector('[data-fill="weight"]').style.width = scale(asset.weight);
        row.querySelector('[data-fill="risk"]').style.width = scale(
          asset.risk_contribution_pct
        );

        const gap = asset.risk_contribution_pct - asset.weight;
        row.querySelector("[data-readout]").innerHTML =
          `${pct(asset.weight, 1)} <span class="muted">capital</span> · ` +
          `${pct(asset.risk_contribution_pct, 1)} <span class="muted">risk</span> ` +
          `<span class="wrisk__gap ${signClass(-gap)}">${gap >= 0 ? "+" : "−"}${pct(
            Math.abs(gap),
            1
          )}</span>`;

        row.setAttribute("aria-selected", String(key === ticker));
      });

      const scaleLabel = container.parentElement?.querySelector("[data-wrisk-scale]");
      if (scaleLabel) scaleLabel.textContent = `Scaled to ${pct(largest, 0)}`;

      // Name the concentration explicitly rather than leaving it to be inferred.
      const noteNode = container.parentElement?.querySelector("[data-wrisk-note]");
      if (noteNode) {
        const top = tickers
          .map((t) => ({ t, gap: assets[t].risk_contribution_pct - assets[t].weight }))
          .sort((a, b) => b.gap - a.gap)[0];
        noteNode.innerHTML =
          `Upper bar is capital weight, lower bar is share of portfolio risk. ` +
          `<strong style="color:var(--ink-primary)">${top.t}</strong> carries ` +
          `${pct(assets[top.t].risk_contribution_pct, 1)} of portfolio risk on ` +
          `${pct(assets[top.t].weight, 0)} of capital.`;
      }
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Risk / return scatter                                                      */
/* -------------------------------------------------------------------------- */

export function createRiskReturnScatter(container, payload, { onSelect }) {
  const frame = new ChartFrame(container, {
    margin: { top: 18, right: 26, bottom: 44, left: 56 },
    aspect: 0.72,
    minHeight: 280,
    maxHeight: 380,
    label: "Annualized return against annualized volatility",
  });
  const tooltip = new Tooltip(container);

  let state = null;

  const render = () => {
    if (!state) return;
    const { windowKey, ticker } = state;
    const block = payload.windows[windowKey];

    const points = payload.meta.tickers.map((key) => ({
      key,
      label: key,
      x: block.assets[key].ann_volatility,
      y: block.assets[key].ann_return,
      weight: block.assets[key].weight,
      sharpe: block.assets[key].sharpe,
      color: assetColor(key),
      kind: "asset",
    }));

    points.push({
      key: "PORTFOLIO",
      label: "Portfolio",
      x: block.portfolio.ann_volatility,
      y: block.portfolio.ann_return,
      sharpe: block.portfolio.sharpe,
      color: cssVar("--s-portfolio"),
      kind: "portfolio",
    });
    points.push({
      key: "BENCHMARK",
      label: payload.meta.benchmark.label,
      x: block.benchmark.ann_volatility,
      y: block.benchmark.ann_return,
      sharpe: block.benchmark.sharpe,
      color: cssVar("--s-benchmark"),
      kind: "benchmark",
    });

    frame.reset();

    const xDomain = padDomain(extent(points.map((p) => p.x)), 0.18);
    const yDomain = padDomain(extent([...points.map((p) => p.y), 0]), 0.16);
    const xScale = scaleLinear([Math.max(0, xDomain[0]), xDomain[1]], [0, frame.plotWidth]);
    const yScale = scaleLinear(yDomain, [frame.plotHeight, 0]);

    frame.yAxis(yScale, { format: (v) => pct(v, 0), parityAt: 0 });
    frame.xAxis(
      niceTicks(xScale.domain[0], xScale.domain[1], 5).map((value) => ({
        x: xScale(value),
        label: pct(value, 0),
      }))
    );

    axisTitle(frame, "Annualized volatility", "Annualized return");

    for (const point of points) {
      const cx = xScale(point.x);
      const cy = yScale(point.y);
      const selected = point.key === ticker;
      const dimmed = ticker && point.kind === "asset" && !selected;
      const radius = point.kind === "asset" ? 6 + Math.sqrt(point.weight) * 15 : 7;

      const group = svgEl(
        "g",
        {
          style: point.kind === "asset" ? "cursor:pointer" : "",
          tabindex: point.kind === "asset" ? 0 : null,
          role: point.kind === "asset" ? "button" : null,
          "aria-label": `${point.label}: return ${pct(point.y)}, volatility ${pct(point.x)}`,
          opacity: dimmed ? 0.42 : 1,
        },
        frame.plot
      );

      svgEl(
        "circle",
        {
          cx: cx.toFixed(2),
          cy: cy.toFixed(2),
          r: radius.toFixed(1),
          fill: point.kind === "asset" ? point.color : "none",
          "fill-opacity": selected ? 0.55 : 0.26,
          stroke: point.color,
          "stroke-width": selected ? 2.2 : 1.5,
          "stroke-dasharray": point.kind === "benchmark" ? "3 2.5" : null,
        },
        group
      );

      const text = svgEl(
        "text",
        {
          x: cx.toFixed(2),
          y: (cy - radius - 7).toFixed(2),
          "text-anchor": "middle",
          fill: selected ? cssVar("--ink-primary") : cssVar("--ink-tertiary"),
          "font-size": 10.5,
          "font-weight": selected ? 700 : 500,
        },
        group
      );
      text.textContent = point.label;

      group.addEventListener("pointerenter", (event) => {
        const bounds = container.getBoundingClientRect();
        tooltip.show(
          tipHead(point.label) +
            tipRow("Annualized return", pct(point.y), point.color) +
            tipRow("Annualized volatility", pct(point.x)) +
            tipRow("Sharpe ratio", ratio(point.sharpe)) +
            (point.kind === "asset" ? tipRow("Capital weight", pct(point.weight, 1)) : "") +
            (point.kind === "asset"
              ? tipNote("Mark area is proportional to capital weight.")
              : ""),
          event.clientX - bounds.left,
          event.clientY - bounds.top
        );
      });
      group.addEventListener("pointerleave", () => tooltip.hide());

      if (point.kind === "asset") {
        group.addEventListener("click", () => onSelect(point.key));
        group.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            onSelect(point.key);
          }
        });
      }
    }
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

/* -------------------------------------------------------------------------- */
/* Diversification map — correlation distance MDS                             */
/* -------------------------------------------------------------------------- */

export function createDiversificationMap(container, payload, { onSelect }) {
  const frame = new ChartFrame(container, {
    margin: { top: 22, right: 30, bottom: 30, left: 30 },
    aspect: 0.74,
    minHeight: 280,
    maxHeight: 380,
    label:
      "Diversification map: holdings positioned by correlation distance, so holdings that moved together sit together",
  });
  const tooltip = new Tooltip(container);

  let state = null;

  const render = () => {
    if (!state) return;
    const { windowKey, ticker } = state;
    const block = payload.windows[windowKey];
    const universe = block.universe;
    const correlation = block.correlation;

    frame.reset();

    // The layout is a unit disc; keep it square and centred so distances read
    // proportionally in both directions.
    const side = Math.min(frame.plotWidth, frame.plotHeight);
    const cx0 = frame.plotWidth / 2;
    const cy0 = frame.plotHeight / 2;
    const radius = (side / 2) * 0.82;
    const px = (x) => cx0 + x * radius;
    const py = (z) => cy0 + z * radius;

    // Reference rings give the eye a distance scale without implying a
    // measured axis.
    for (const ring of [0.4, 0.72, 1]) {
      svgEl(
        "circle",
        {
          cx: cx0.toFixed(2),
          cy: cy0.toFixed(2),
          r: (radius * ring).toFixed(2),
          fill: "none",
          stroke: cssVar("--rule"),
          "stroke-dasharray": ring === 1 ? null : "2 4",
        },
        frame.plot
      );
    }

    // Edges: brightness encodes |correlation|, so the strongest pairing is the
    // most visible link.
    for (const edge of universe.edges) {
      const from = universe.nodes.find((n) => n.ticker === edge.source);
      const to = universe.nodes.find((n) => n.ticker === edge.target);
      if (!from || !to) continue;
      const magnitude = Math.min(Math.abs(edge.correlation), 1);
      svgEl(
        "line",
        {
          x1: px(from.x).toFixed(2),
          y1: py(from.z).toFixed(2),
          x2: px(to.x).toFixed(2),
          y2: py(to.z).toFixed(2),
          stroke: edge.correlation >= 0 ? cssVar("--azure") : cssVar("--negative"),
          "stroke-opacity": (0.07 + magnitude * 0.42).toFixed(3),
          "stroke-width": (0.7 + magnitude * 1.7).toFixed(2),
        },
        frame.plot
      );
    }

    // Weighted centroid of the holdings.
    svgEl(
      "circle",
      {
        cx: px(universe.centroid.x).toFixed(2),
        cy: py(universe.centroid.z).toFixed(2),
        r: 3,
        fill: "none",
        stroke: cssVar("--ink-quaternary"),
        "stroke-width": 1,
      },
      frame.plot
    );

    // Label placement, de-collided.
    //
    // Two holdings that moved together sit close together by design, so their
    // labels can overlap. The marks stay exactly where the data puts them —
    // only the labels are nudged, in a deterministic top-down pass, so the
    // field stays readable without misstating a position.
    const LABEL_H = 13;
    const labelPlacement = new Map();
    const placed = [];

    for (const node of [...universe.nodes].sort((a, b) => a.z - b.z)) {
      const r = 5 + Math.sqrt(node.radius_weight) * 13;
      const x = px(node.x);
      let y = py(node.z) - r - 7;

      while (
        placed.some(
          (p) => Math.abs(p.y - y) < LABEL_H && Math.abs(p.x - x) < 34
        )
      ) {
        y -= LABEL_H;
      }

      placed.push({ x, y });
      labelPlacement.set(node.ticker, { x, y });
    }

    for (const node of universe.nodes) {
      const selected = node.ticker === ticker;
      const dimmed = ticker && !selected;
      const r = 5 + Math.sqrt(node.radius_weight) * 13;
      const color = assetColor(node.ticker);
      const label = labelPlacement.get(node.ticker);

      const group = svgEl(
        "g",
        {
          style: "cursor:pointer",
          tabindex: 0,
          role: "button",
          opacity: dimmed ? 0.42 : 1,
          "aria-label":
            `${node.ticker}: capital weight ${pct(node.radius_weight, 1)}, ` +
            `volatility ${pct(node.height_volatility)}, Sharpe ${ratio(node.color_sharpe)}`,
        },
        frame.plot
      );

      svgEl(
        "circle",
        {
          cx: px(node.x).toFixed(2),
          cy: py(node.z).toFixed(2),
          r: r.toFixed(1),
          fill: color,
          "fill-opacity": selected ? 0.62 : 0.3,
          stroke: color,
          "stroke-width": selected ? 2.2 : 1.4,
        },
        group
      );

      // A leader line whenever the label had to move clear of its mark.
      if (Math.abs(label.y - (py(node.z) - r - 7)) > 2) {
        svgEl(
          "line",
          {
            x1: label.x.toFixed(2),
            y1: (label.y + 3).toFixed(2),
            x2: px(node.x).toFixed(2),
            y2: (py(node.z) - r - 2).toFixed(2),
            stroke: cssVar("--rule-strong"),
            "stroke-width": 1,
          },
          group
        );
      }

      const text = svgEl(
        "text",
        {
          x: label.x.toFixed(2),
          y: label.y.toFixed(2),
          "text-anchor": "middle",
          fill: selected ? cssVar("--ink-primary") : cssVar("--ink-tertiary"),
          "font-size": 10.5,
          "font-weight": selected ? 700 : 500,
        },
        group
      );
      text.textContent = node.ticker;

      group.addEventListener("pointerenter", (event) => {
        const bounds = container.getBoundingClientRect();
        // Report the holding this one is most and least correlated with: the
        // plain-language reading of its position in the field.
        const others = payload.meta.tickers.filter((t) => t !== node.ticker);
        const sorted = others
          .map((t) => ({ t, rho: correlation[node.ticker][t] }))
          .sort((a, b) => b.rho - a.rho);

        tooltip.show(
          tipHead(node.ticker) +
            tipRow("Capital weight", pct(node.radius_weight, 1), color) +
            tipRow("Annualized volatility", pct(node.height_volatility)) +
            tipRow("Risk contribution", pct(node.halo_risk_contribution_pct, 1)) +
            tipRow(
              "Most correlated",
              `${sorted[0].t} ${sorted[0].rho.toFixed(2)}`
            ) +
            tipRow(
              "Least correlated",
              `${sorted[sorted.length - 1].t} ${sorted[sorted.length - 1].rho.toFixed(2)}`
            ) +
            tipNote(
              "Position comes from multidimensional scaling of correlation distance. Nearby holdings have moved together."
            ),
          event.clientX - bounds.left,
          event.clientY - bounds.top
        );
      });
      group.addEventListener("pointerleave", () => tooltip.hide());
      group.addEventListener("click", () => onSelect(node.ticker));
      group.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(node.ticker);
        }
      });
    }

    const caption = svgEl(
      "text",
      {
        x: frame.plotWidth / 2,
        y: frame.plotHeight + 20,
        "text-anchor": "middle",
        fill: cssVar("--ink-quaternary"),
        "font-size": 10.5,
      },
      frame.plot
    );
    caption.textContent = "Further apart = more independent · mark area = capital weight";
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

/* -------------------------------------------------------------------------- */
/* Correlation matrix                                                         */
/* -------------------------------------------------------------------------- */

export function createCorrelationMatrix(container, payload, { onSelect }) {
  const tooltip = new Tooltip(container);
  const benchmarkSymbol = payload.meta.benchmark.symbol;
  const short = (key) => (key === benchmarkSymbol ? "S&P" : key);
  const full = (key) => (key === benchmarkSymbol ? payload.meta.benchmark.label : key);

  return {
    update({ windowKey, ticker }) {
      const correlation = payload.windows[windowKey].correlation;
      const labels = Object.keys(correlation);

      container.innerHTML = "";

      const grid = document.createElement("div");
      grid.className = "heat";
      grid.style.gridTemplateColumns = `minmax(38px, auto) repeat(${labels.length}, minmax(0, 1fr))`;
      grid.setAttribute("role", "table");
      grid.setAttribute("aria-label", "Correlation of daily returns");

      grid.appendChild(document.createElement("span"));
      for (const key of labels) {
        const head = document.createElement("span");
        head.className = "heat__head";
        head.textContent = short(key);
        grid.appendChild(head);
      }

      for (const rowKey of labels) {
        const rowHead = document.createElement("span");
        rowHead.className = "heat__head heat__head--row";
        rowHead.textContent = short(rowKey);
        grid.appendChild(rowHead);

        for (const colKey of labels) {
          const value = correlation[rowKey][colKey];
          const cell = document.createElement("span");
          const isDiag = rowKey === colKey;

          cell.className = `heat__cell${isDiag ? " heat__cell--diag" : ""}`;
          cell.textContent = value.toFixed(2);

          if (!isDiag) {
            cell.style.background = correlationColor(value);
            cell.style.color =
              Math.abs(value) > 0.5 ? "var(--ink-primary)" : "var(--ink-secondary)";
          }

          if (ticker && (rowKey === ticker || colKey === ticker)) {
            cell.classList.add("heat__cell--linked");
          }

          cell.addEventListener("pointerenter", (event) => {
            const bounds = container.getBoundingClientRect();
            tooltip.show(
              tipHead("Correlation of daily returns") +
                tipRow(`${full(rowKey)} · ${full(colKey)}`, value.toFixed(3)) +
                tipNote(
                  `Pearson correlation over ${payload.windows[windowKey].label}. ` +
                    `Lower values between holdings mean more diversification.`
                ),
              event.clientX - bounds.left,
              event.clientY - bounds.top
            );
          });
          cell.addEventListener("pointerleave", () => tooltip.hide());

          if (isDiag && rowKey !== benchmarkSymbol) {
            cell.style.cursor = "pointer";
            cell.addEventListener("click", () => onSelect(rowKey));
          }

          grid.appendChild(cell);
        }
      }

      container.appendChild(grid);

      const scale = document.createElement("div");
      scale.className = "heat__scale";
      scale.innerHTML =
        `<span>−1.0</span><span class="heat__ramp"></span><span>+1.0</span>`;
      container.appendChild(scale);
    },
    destroy() {
      tooltip.destroy();
    },
  };
}

/* -------------------------------------------------------------------------- */

function axisTitle(frame, xLabel, yLabel) {
  const x = svgEl(
    "text",
    {
      x: frame.plotWidth / 2,
      y: frame.plotHeight + 38,
      "text-anchor": "middle",
      fill: cssVar("--ink-quaternary"),
      "font-size": 10,
      "letter-spacing": "0.1em",
    },
    frame.plot
  );
  x.textContent = xLabel.toUpperCase();

  const y = svgEl(
    "text",
    {
      transform: `translate(-44,${frame.plotHeight / 2}) rotate(-90)`,
      "text-anchor": "middle",
      fill: cssVar("--ink-quaternary"),
      "font-size": 10,
      "letter-spacing": "0.1em",
    },
    frame.plot
  );
  y.textContent = yLabel.toUpperCase();
}
