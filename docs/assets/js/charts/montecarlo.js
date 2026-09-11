/**
 * Monte Carlo scenario analysis: the percentile fan with an inspectable
 * horizon, and the distribution of terminal returns.
 *
 * Both are drawn on canvas because the fan renders sixty individual paths plus
 * five percentile bands — roughly twenty thousand line segments, past the
 * point where SVG nodes are sensible. The reveal is animated by advancing how
 * far along the horizon is drawn; the completed frame is cached so hover
 * inspection never repaints the paths.
 *
 * Everything here is scenario analysis parameterized from historical mean and
 * volatility under a normal-return assumption. It is not a forecast, and the
 * wording in every tooltip and label says so: figures are frequencies within
 * a simulated distribution, never real-world probabilities.
 */

import { Tooltip, sizeCanvas, tipHead, tipNote, tipRow } from "./base.js";
import { count, cssVar, multiple, pct, pctSigned } from "../format.js";
import { debounce, motion, onFirstVisible, tween } from "../motion.js";

const BAND_KEYS = ["p5", "p25", "p50", "p75", "p95"];

export function createMonteCarloFan(container, payload) {
  const mc = payload.monte_carlo;
  const days = mc.config.days;

  const canvas = document.createElement("canvas");
  canvas.className = "chart__canvas";
  canvas.setAttribute("role", "img");
  canvas.setAttribute("tabindex", "0");
  canvas.setAttribute(
    "aria-label",
    `Simulated portfolio value across ${count(mc.config.simulations)} scenarios over ` +
      `${days} trading days, shown as percentile bands. Use arrow keys to inspect a horizon.`
  );
  container.classList.add("chart");
  container.appendChild(canvas);

  const tooltip = new Tooltip(container);
  const context = canvas.getContext("2d");

  const margin = { top: 16, right: 20, bottom: 32, left: 62 };
  let width = 0;
  let height = 0;
  let plotWidth = 0;
  let plotHeight = 0;
  let progress = motion.reduced ? 1 : 0;
  let cached = null;
  let hoverDay = null;

  const lowest = Math.min(1, ...mc.bands.p5);
  const highest = Math.max(...mc.bands.p95);
  const domain = [lowest * 0.98, highest * 1.02];

  const xAt = (day) => margin.left + (day / Math.max(1, days - 1)) * plotWidth;
  const yAt = (value) =>
    margin.top + plotHeight - ((value - domain[0]) / (domain[1] - domain[0])) * plotHeight;

  function measure() {
    width = Math.max(240, container.clientWidth || 640);
    height = Math.round(Math.min(480, Math.max(280, width * 0.34)));
    plotWidth = Math.max(10, width - margin.left - margin.right);
    plotHeight = Math.max(10, height - margin.top - margin.bottom);
    sizeCanvas(canvas, context, width, height);
    cached = null;
  }

  function drawChrome() {
    context.clearRect(0, 0, width, height);
    context.font = "10.5px Inter Variable, system-ui, sans-serif";
    context.lineWidth = 1;

    const step = (domain[1] - domain[0]) / 4;
    for (let i = 0; i <= 4; i += 1) {
      const value = domain[0] + step * i;
      const y = Math.round(yAt(value)) + 0.5;
      context.beginPath();
      context.moveTo(margin.left, y);
      context.lineTo(margin.left + plotWidth, y);
      context.strokeStyle = cssVar("--rule");
      context.stroke();

      context.fillStyle = cssVar("--ink-quaternary");
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(`${value.toFixed(2)}×`, margin.left - 10, y);
    }

    // Starting value: the reference every percentile is read against.
    const baseY = Math.round(yAt(1)) + 0.5;
    context.beginPath();
    context.moveTo(margin.left, baseY);
    context.lineTo(margin.left + plotWidth, baseY);
    context.strokeStyle = cssVar("--rule-strong");
    context.setLineDash([2, 3]);
    context.stroke();
    context.setLineDash([]);

    context.fillStyle = cssVar("--ink-quaternary");
    context.textAlign = "center";
    context.textBaseline = "top";
    for (let month = 0; month <= 12; month += 3) {
      const day = Math.min(days - 1, Math.round((month / 12) * (days - 1)));
      context.fillText(
        month === 0 ? "start" : `+${month}m`,
        xAt(day),
        margin.top + plotHeight + 10
      );
    }
  }

  function bandShape(upper, lower, upTo) {
    context.beginPath();
    for (let i = 0; i <= upTo; i += 1) {
      const x = xAt(i);
      const y = yAt(upper[i]);
      if (i === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    for (let i = upTo; i >= 0; i -= 1) context.lineTo(xAt(i), yAt(lower[i]));
    context.closePath();
  }

  function drawFan(fraction) {
    const upTo = Math.max(1, Math.round((days - 1) * fraction));
    drawChrome();

    // Individual paths sit furthest back and very faint: they convey
    // dispersion without competing with the percentile structure.
    context.strokeStyle = "rgba(10, 132, 255, 0.06)";
    context.lineWidth = 1;
    for (const path of mc.sample_paths) {
      context.beginPath();
      for (let i = 0; i <= upTo; i += 1) {
        const x = xAt(i);
        const y = yAt(path[i]);
        if (i === 0) context.moveTo(x, y);
        else context.lineTo(x, y);
      }
      context.stroke();
    }

    bandShape(mc.bands.p95, mc.bands.p5, upTo);
    context.fillStyle = "rgba(10, 132, 255, 0.11)";
    context.fill();

    bandShape(mc.bands.p75, mc.bands.p25, upTo);
    context.fillStyle = "rgba(10, 132, 255, 0.2)";
    context.fill();

    context.beginPath();
    for (let i = 0; i <= upTo; i += 1) {
      const x = xAt(i);
      const y = yAt(mc.bands.p50[i]);
      if (i === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.strokeStyle = cssVar("--s-portfolio");
    context.lineWidth = 2;
    context.stroke();
  }

  function paintHover() {
    if (!cached) return;
    context.putImageData(cached, 0, 0);
    if (hoverDay === null) return;

    const x = xAt(hoverDay);
    context.beginPath();
    context.moveTo(x, margin.top);
    context.lineTo(x, margin.top + plotHeight);
    context.strokeStyle = cssVar("--rule-strong");
    context.lineWidth = 1;
    context.stroke();

    for (const key of BAND_KEYS) {
      const y = yAt(mc.bands[key][hoverDay]);
      context.beginPath();
      context.arc(x, y, key === "p50" ? 4 : 3, 0, Math.PI * 2);
      context.fillStyle =
        key === "p50" ? cssVar("--s-portfolio") : "rgba(10, 132, 255, 0.8)";
      context.fill();
      context.strokeStyle = cssVar("--bg-base");
      context.lineWidth = 1.5;
      context.stroke();
    }
  }

  function render() {
    measure();
    drawFan(progress);
    if (progress >= 1) {
      cached = context.getImageData(0, 0, canvas.width, canvas.height);
      paintHover();
    }
  }

  onFirstVisible(container, () => {
    measure();
    if (motion.reduced) {
      progress = 1;
      render();
      return;
    }
    tween({
      from: 0,
      to: 1,
      duration: 1400,
      onUpdate: (value) => {
        progress = value;
        drawFan(value);
      },
      onDone: () => {
        progress = 1;
        cached = context.getImageData(0, 0, canvas.width, canvas.height);
        paintHover();
      },
    });
  });

  const describe = (day, position) => {
    const b = mc.bands;
    tooltip.show(
      tipHead(`Simulated horizon · day ${day + 1} of ${days}`) +
        tipRow("95th percentile", multiple(b.p95[day]), "rgba(10,132,255,0.8)") +
        tipRow("75th percentile", multiple(b.p75[day]), "rgba(10,132,255,0.8)") +
        tipRow("Median", multiple(b.p50[day]), cssVar("--s-portfolio")) +
        tipRow("25th percentile", multiple(b.p25[day]), "rgba(10,132,255,0.8)") +
        tipRow("5th percentile", multiple(b.p5[day]), "rgba(10,132,255,0.8)") +
        tipNote(
          `Percentiles of ${count(mc.config.simulations)} simulated paths. ` +
            `Historical-parameter scenario simulation — not a forecast.`
        ),
      position.x,
      position.y
    );
  };

  const dayFromEvent = (event) => {
    const bounds = canvas.getBoundingClientRect();
    const scale = width / bounds.width;
    const x = (event.clientX - bounds.left) * scale - margin.left;
    return Math.round(Math.max(0, Math.min(1, x / plotWidth)) * (days - 1));
  };

  canvas.addEventListener("pointermove", (event) => {
    if (progress < 1) return;
    hoverDay = dayFromEvent(event);
    const bounds = canvas.getBoundingClientRect();
    paintHover();
    describe(hoverDay, {
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    });
  });

  canvas.addEventListener("pointerleave", () => {
    hoverDay = null;
    tooltip.hide();
    paintHover();
  });

  canvas.addEventListener("keydown", (event) => {
    if (progress < 1) return;
    const stride = event.shiftKey ? 21 : 1;
    let next = hoverDay ?? Math.floor(days / 2);
    if (event.key === "ArrowRight") next = Math.min(days - 1, next + stride);
    else if (event.key === "ArrowLeft") next = Math.max(0, next - stride);
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = days - 1;
    else if (event.key === "Escape") {
      hoverDay = null;
      tooltip.hide();
      paintHover();
      return;
    } else return;

    event.preventDefault();
    hoverDay = next;
    paintHover();
    describe(next, { x: xAt(next), y: height / 2 });
  });

  canvas.addEventListener("blur", () => {
    hoverDay = null;
    tooltip.hide();
    paintHover();
  });

  const onResize = debounce(() => {
    progress = 1;
    render();
  }, 200);
  window.addEventListener("resize", onResize);

  return {
    destroy() {
      window.removeEventListener("resize", onResize);
      tooltip.destroy();
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Terminal return distribution                                               */
/* -------------------------------------------------------------------------- */

export function createTerminalHistogram(container, payload) {
  const mc = payload.monte_carlo;
  const histogram = mc.terminal_histogram;

  const canvas = document.createElement("canvas");
  canvas.className = "chart__canvas";
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    `Distribution of simulated terminal returns across ${count(mc.config.simulations)} scenarios`
  );
  container.classList.add("chart");
  container.appendChild(canvas);

  const tooltip = new Tooltip(container);
  const context = canvas.getContext("2d");
  const margin = { top: 14, right: 16, bottom: 32, left: 48 };

  let width = 0;
  let height = 0;
  let plotWidth = 0;
  let plotHeight = 0;
  let bars = [];
  let hovered = null;

  const maxCount = Math.max(...histogram.counts);
  const edges = histogram.bin_edges;
  const lowEdge = edges[0];
  const highEdge = edges[edges.length - 1];

  const xAt = (value) =>
    margin.left + ((value - lowEdge) / (highEdge - lowEdge)) * plotWidth;

  function measure() {
    width = Math.max(240, container.clientWidth || 520);
    height = Math.round(Math.min(320, Math.max(220, width * 0.52)));
    plotWidth = Math.max(10, width - margin.left - margin.right);
    plotHeight = Math.max(10, height - margin.top - margin.bottom);
    sizeCanvas(canvas, context, width, height);
  }

  function render(fraction = 1) {
    context.clearRect(0, 0, width, height);
    context.font = "10.5px Inter Variable, system-ui, sans-serif";
    bars = [];

    context.strokeStyle = cssVar("--rule");
    context.lineWidth = 1;
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (let i = 0; i <= 2; i += 1) {
      const value = (maxCount / 2) * i;
      const y = Math.round(margin.top + plotHeight - (value / maxCount) * plotHeight) + 0.5;
      context.beginPath();
      context.moveTo(margin.left, y);
      context.lineTo(margin.left + plotWidth, y);
      context.stroke();
      context.fillStyle = cssVar("--ink-quaternary");
      context.fillText(count(Math.round(value)), margin.left - 8, y);
    }

    context.textAlign = "center";
    context.textBaseline = "top";
    for (let value = Math.ceil(lowEdge * 4) / 4; value < highEdge; value += 0.25) {
      context.fillStyle = cssVar("--ink-quaternary");
      context.fillText(pct(value, 0), xAt(value), margin.top + plotHeight + 10);
    }

    for (let i = 0; i < histogram.counts.length; i += 1) {
      const x0 = xAt(edges[i]);
      const x1 = xAt(edges[i + 1]);
      const barHeight = (histogram.counts[i] / maxCount) * plotHeight * fraction;
      const y = margin.top + plotHeight - barHeight;
      const barWidth = Math.max(1, x1 - x0 - 1);

      // Bins entirely below the starting value are losses: the one place a
      // colour split carries real meaning in this chart.
      const isLoss = edges[i + 1] <= 0;
      context.fillStyle =
        hovered === i
          ? isLoss
            ? "rgba(229,72,77,0.95)"
            : "rgba(10,132,255,0.95)"
          : isLoss
            ? "rgba(229,72,77,0.55)"
            : "rgba(10,132,255,0.48)";
      context.fillRect(x0, y, barWidth, barHeight);

      bars.push({ index: i, x0, x1, from: edges[i], to: edges[i + 1] });
    }

    for (const marker of [
      { value: mc.summary.median_terminal_return, label: "Median", color: cssVar("--s-portfolio") },
      { value: mc.summary.p5_terminal_return, label: "5th pct", color: cssVar("--caution") },
    ]) {
      const x = xAt(marker.value);
      context.beginPath();
      context.moveTo(x, margin.top);
      context.lineTo(x, margin.top + plotHeight);
      context.strokeStyle = marker.color;
      context.lineWidth = 1.5;
      context.setLineDash([4, 3]);
      context.stroke();
      context.setLineDash([]);

      context.fillStyle = marker.color;
      context.textAlign = "left";
      context.textBaseline = "top";
      context.fillText(marker.label, x + 5, margin.top + 2);
    }

    const zeroX = xAt(0);
    if (zeroX > margin.left && zeroX < margin.left + plotWidth) {
      context.beginPath();
      context.moveTo(zeroX, margin.top);
      context.lineTo(zeroX, margin.top + plotHeight);
      context.strokeStyle = cssVar("--rule-strong");
      context.lineWidth = 1;
      context.stroke();
    }
  }

  onFirstVisible(container, () => {
    measure();
    if (motion.reduced) {
      render(1);
      return;
    }
    tween({ from: 0, to: 1, duration: 800, onUpdate: (value) => render(value) });
  });

  canvas.addEventListener("pointermove", (event) => {
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * (width / bounds.width);
    const bar = bars.find((b) => x >= b.x0 && x <= b.x1);

    if (!bar) {
      if (hovered !== null) {
        hovered = null;
        render(1);
        tooltip.hide();
      }
      return;
    }

    if (hovered !== bar.index) {
      hovered = bar.index;
      render(1);
    }

    const scenarios = histogram.counts[bar.index];
    tooltip.show(
      tipHead("Terminal return bucket") +
        tipRow("Range", `${pctSigned(bar.from, 1)} to ${pctSigned(bar.to, 1)}`) +
        tipRow("Scenarios", count(scenarios)) +
        tipRow("Share of scenarios", pct(scenarios / mc.config.simulations, 2)) +
        tipNote(
          "Frequency within the simulated distribution, not a real-world probability."
        ),
      event.clientX - bounds.left,
      event.clientY - bounds.top
    );
  });

  canvas.addEventListener("pointerleave", () => {
    hovered = null;
    render(1);
    tooltip.hide();
  });

  const onResize = debounce(() => {
    measure();
    render(1);
  }, 200);
  window.addEventListener("resize", onResize);

  return {
    destroy() {
      window.removeEventListener("resize", onResize);
      tooltip.destroy();
    },
  };
}
