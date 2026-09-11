/**
 * Shared chart primitives: scales, ticks, SVG construction, path building,
 * a crosshair-tracking tooltip and responsive sizing.
 *
 * The charts are hand-built SVG rather than a charting library so that path
 * reveals can be animated, every mark carries its own accessible label, and
 * the axis treatment stays identical across every panel. No module here
 * computes a financial quantity — they receive prepared series and draw them.
 *
 * Chart chrome follows one rule: horizontal gridlines only, no axis spine,
 * no vertical grid, no frame. The plot is defined by its type and its data.
 */

const SVG_NS = "http://www.w3.org/2000/svg";

export function svgEl(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    node.setAttribute(key, String(value));
  }
  if (parent) parent.appendChild(node);
  return node;
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Linear scale from a numeric domain to a pixel range. */
export function scaleLinear(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  const factor = span === 0 ? 0 : (r1 - r0) / span;

  const scale = (value) => r0 + (value - d0) * factor;
  scale.invert = (pixel) => (factor === 0 ? d0 : d0 + (pixel - r0) / factor);
  scale.domain = domain;
  scale.range = range;
  return scale;
}

/** Index scale: array position to pixel range. */
export function scaleIndex(length, range) {
  return scaleLinear([0, Math.max(1, length - 1)], range);
}

/**
 * Round tick values covering a domain. Steps are constrained to 1, 2, 2.5 and
 * 5 times a power of ten so labels stay legible at any magnitude.
 */
export function niceTicks(min, max, target = 5) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];

  const rawStep = (max - min) / Math.max(1, target);
  const magnitude = Math.pow(10, Math.floor(Math.log10(Math.abs(rawStep))));
  const normalized = rawStep / magnitude;

  let step;
  if (normalized <= 1) step = magnitude;
  else if (normalized <= 2) step = 2 * magnitude;
  else if (normalized <= 2.5) step = 2.5 * magnitude;
  else if (normalized <= 5) step = 5 * magnitude;
  else step = 10 * magnitude;

  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let value = start; value <= max + step * 1e-6; value += step) {
    ticks.push(Number(value.toFixed(10)));
  }
  return ticks;
}

/** Pad a domain so series never touch the plot edges. */
export function padDomain([min, max], fraction = 0.06) {
  if (min === max) {
    const nudge = Math.abs(min) * 0.1 || 1;
    return [min - nudge, max + nudge];
  }
  const pad = (max - min) * fraction;
  return [min - pad, max + pad];
}

/** Extent across one or more series, ignoring nulls. */
export function extent(...series) {
  let min = Infinity;
  let max = -Infinity;
  for (const values of series) {
    if (!values) continue;
    for (const value of values) {
      if (value === null || value === undefined || Number.isNaN(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
  }
  if (min === Infinity) return [0, 1];
  return [min, max];
}

/**
 * Build an SVG path from a series, breaking the line wherever the data is
 * null. Moving averages have a genuine lead-in with no value, and bridging
 * that gap would draw a number that does not exist.
 */
export function linePath(values, xScale, yScale) {
  let path = "";
  let penDown = false;

  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (value === null || value === undefined || Number.isNaN(value)) {
      penDown = false;
      continue;
    }
    const x = xScale(i).toFixed(2);
    const y = yScale(value).toFixed(2);
    path += penDown ? `L${x},${y}` : `M${x},${y}`;
    penDown = true;
  }

  return path;
}

/** Area path between a series and a baseline value. */
export function areaPath(values, xScale, yScale, baseline) {
  const base = yScale(baseline).toFixed(2);
  const segments = [];
  let current = null;

  const flush = () => {
    if (current && current.points.length > 1) {
      const first = current.points[0];
      const last = current.points[current.points.length - 1];
      segments.push(
        `M${first.x},${base}L` +
          current.points.map((p) => `${p.x},${p.y}`).join("L") +
          `L${last.x},${base}Z`
      );
    }
    current = null;
  };

  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (value === null || value === undefined || Number.isNaN(value)) {
      flush();
      continue;
    }
    if (!current) current = { points: [] };
    current.points.push({
      x: xScale(i).toFixed(2),
      y: yScale(value).toFixed(2),
    });
  }
  flush();

  return segments.join("");
}

/** Start the one-shot stroke reveal on a freshly built path. */
export function animateDraw(pathNode) {
  let length = 0;
  try {
    length = pathNode.getTotalLength();
  } catch {
    length = 0;
  }
  if (!length || !Number.isFinite(length)) return;
  pathNode.style.setProperty("--len", String(Math.ceil(length)));
  pathNode.classList.add("series--draw");
}

/**
 * Responsive SVG chart frame.
 *
 * Height is driven by the container width and a target aspect ratio, with a
 * clamp so a very wide screen does not produce an absurdly tall plot.
 */
export class ChartFrame {
  constructor(container, options = {}) {
    this.container = container;
    this.margin = { top: 12, right: 16, bottom: 26, left: 52, ...options.margin };
    this.aspect = options.aspect ?? 0.4;
    this.minHeight = options.minHeight ?? 200;
    this.maxHeight = options.maxHeight ?? 420;

    this.container.classList.add("chart");
    this.svg = svgEl("svg", {
      class: "chart__svg",
      preserveAspectRatio: "xMidYMid meet",
      role: "img",
    });
    if (options.label) this.svg.setAttribute("aria-label", options.label);
    this.container.appendChild(this.svg);

    this.measure();
  }

  measure() {
    const width = Math.max(220, this.container.clientWidth || 640);
    const height = Math.round(
      Math.min(this.maxHeight, Math.max(this.minHeight, width * this.aspect))
    );

    this.width = width;
    this.height = height;
    this.plotWidth = Math.max(10, width - this.margin.left - this.margin.right);
    this.plotHeight = Math.max(10, height - this.margin.top - this.margin.bottom);

    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.svg.setAttribute("width", width);
    this.svg.setAttribute("height", height);
    return this;
  }

  reset() {
    clear(this.svg);
    this.measure();
    this.defs = svgEl("defs", {}, this.svg);
    this.plot = svgEl(
      "g",
      { transform: `translate(${this.margin.left},${this.margin.top})` },
      this.svg
    );
    return this;
  }

  /** Vertical gradient for area fills under a line. */
  gradient(id, color, { from = 0.22, to = 0 } = {}) {
    const gradient = svgEl(
      "linearGradient",
      { id, x1: 0, y1: 0, x2: 0, y2: 1 },
      this.defs
    );
    svgEl("stop", { offset: "0%", "stop-color": color, "stop-opacity": from }, gradient);
    svgEl("stop", { offset: "100%", "stop-color": color, "stop-opacity": to }, gradient);
    return `url(#${id})`;
  }

  /** Horizontal gridlines plus left-hand value labels. */
  yAxis(scale, { format, ticks, parityAt = null } = {}) {
    const group = svgEl("g", { class: "axis" }, this.plot);
    const values = ticks ?? niceTicks(scale.domain[0], scale.domain[1], 5);

    for (const value of values) {
      const y = scale(value);
      if (!Number.isFinite(y)) continue;
      const isParity = parityAt !== null && Math.abs(value - parityAt) < 1e-12;
      svgEl(
        "line",
        {
          class: isParity ? "gridline gridline--parity" : "gridline",
          x1: 0,
          x2: this.plotWidth,
          y1: y.toFixed(2),
          y2: y.toFixed(2),
        },
        group
      );
      const label = svgEl(
        "text",
        { x: -10, y: y.toFixed(2), "text-anchor": "end", "dominant-baseline": "middle" },
        group
      );
      label.textContent = format ? format(value) : String(value);
    }
    return group;
  }

  /** Bottom axis labels, anchored inward at the extremes so nothing clips. */
  xAxis(pairs) {
    const group = svgEl("g", { class: "axis" }, this.plot);
    const edge = 26;

    for (const { x, label } of pairs) {
      let anchor = "middle";
      if (x < edge) anchor = "start";
      else if (x > this.plotWidth - edge) anchor = "end";

      const text = svgEl(
        "text",
        { x: x.toFixed(2), y: this.plotHeight + 18, "text-anchor": anchor },
        group
      );
      text.textContent = label;
    }
    return group;
  }

  /**
   * Evenly spaced date labels. The final observation is always labelled, but
   * replaces the previous tick when it would collide with it.
   */
  dateTicks(dates, xScale, count = 6) {
    if (!dates.length) return [];

    const step = Math.max(1, Math.floor((dates.length - 1) / Math.max(1, count - 1)));
    const pairs = [];
    for (let i = 0; i < dates.length; i += step) {
      pairs.push({ index: i, x: xScale(i), iso: dates[i] });
    }

    const last = dates.length - 1;
    const previous = pairs[pairs.length - 1];

    if (previous.index !== last) {
      if (last - previous.index < step * 0.55 && pairs.length > 1) pairs.pop();
      pairs.push({ index: last, x: xScale(last), iso: dates[last] });
    }

    return pairs;
  }
}

/**
 * Floating tooltip anchored to the chart container.
 *
 * The only elevated surface in the interface. Flips to the other side of the
 * cursor near the right edge so it never leaves the panel, and is
 * pointer-events: none so it can never swallow a click meant for a mark.
 */
export class Tooltip {
  constructor(container) {
    this.container = container;
    if (getComputedStyle(container).position === "static") {
      container.style.position = "relative";
    }
    this.node = document.createElement("div");
    this.node.className = "tt";
    this.node.setAttribute("role", "status");
    this.node.setAttribute("aria-live", "polite");
    this.node.dataset.open = "false";
    container.appendChild(this.node);
  }

  show(html, x, y) {
    this.node.innerHTML = html;
    this.node.dataset.open = "true";

    const bounds = this.container.getBoundingClientRect();
    const size = this.node.getBoundingClientRect();

    let left = x + 16;
    if (left + size.width > bounds.width - 4) left = x - size.width - 16;
    if (left < 4) left = 4;

    let top = y - size.height / 2;
    top = Math.max(4, Math.min(top, bounds.height - size.height - 4));

    this.node.style.left = `${left}px`;
    this.node.style.top = `${top}px`;
  }

  hide() {
    this.node.dataset.open = "false";
  }

  destroy() {
    this.node.remove();
  }
}

/** Tooltip header. */
export function tipHead(text) {
  return `<div class="tt__head">${text}</div>`;
}

/** Tooltip row. */
export function tipRow(label, value, color = null) {
  const swatch = color
    ? `<span class="tt__swatch" style="background:${color}"></span>`
    : "";
  return (
    `<div class="tt__row"><span class="tt__key">${swatch}${label}</span>` +
    `<span class="tt__val num">${value}</span></div>`
  );
}

/** Tooltip footnote. */
export function tipNote(text) {
  return `<div class="tt__note">${text}</div>`;
}

/**
 * Attach crosshair tracking that reports the nearest series index.
 * Supports pointer and keyboard so data is inspectable without a mouse.
 */
export function trackPointer(frame, length, { onMove, onLeave }) {
  const { margin } = frame;
  const surface = svgEl(
    "rect",
    {
      x: 0,
      y: 0,
      width: frame.plotWidth,
      height: frame.plotHeight,
      fill: "transparent",
      style: "cursor:crosshair",
      tabindex: 0,
      role: "application",
      "aria-label":
        "Chart inspection area. Use the left and right arrow keys to step through dates.",
    },
    frame.plot
  );

  let keyIndex = Math.floor(length / 2);

  const indexFromEvent = (event) => {
    const bounds = frame.svg.getBoundingClientRect();
    const scaleFactor = frame.width / bounds.width;
    const x = (event.clientX - bounds.left) * scaleFactor - margin.left;
    const ratio = frame.plotWidth === 0 ? 0 : x / frame.plotWidth;
    return Math.max(0, Math.min(length - 1, Math.round(ratio * (length - 1))));
  };

  const handleMove = (event) => {
    const index = indexFromEvent(event);
    keyIndex = index;
    const bounds = frame.svg.getBoundingClientRect();
    onMove(index, {
      x: event.clientX - bounds.left,
      y: event.clientY - bounds.top,
    });
  };

  surface.addEventListener("pointermove", handleMove);
  surface.addEventListener("pointerdown", handleMove);
  surface.addEventListener("pointerleave", () => onLeave?.());

  surface.addEventListener("keydown", (event) => {
    const stride = event.shiftKey ? 20 : 1;
    if (event.key === "ArrowRight") keyIndex = Math.min(length - 1, keyIndex + stride);
    else if (event.key === "ArrowLeft") keyIndex = Math.max(0, keyIndex - stride);
    else if (event.key === "Home") keyIndex = 0;
    else if (event.key === "End") keyIndex = length - 1;
    else if (event.key === "Escape") {
      onLeave?.();
      return;
    } else return;

    event.preventDefault();
    const bounds = frame.svg.getBoundingClientRect();
    onMove(keyIndex, {
      x: margin.left + (keyIndex / Math.max(1, length - 1)) * frame.plotWidth,
      y: bounds.height / 2,
    });
  });

  surface.addEventListener("blur", () => onLeave?.());

  return surface;
}

/** Reduce a canvas to a crisp device-pixel-ratio-aware drawing surface. */
export function sizeCanvas(canvas, context, width, height, maxDpr = 1.75) {
  const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width = "100%";
  canvas.style.height = `${height}px`;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return dpr;
}
