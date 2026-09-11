/**
 * Number, date and colour formatting.
 *
 * Formatting only. No module in the front end derives a financial quantity:
 * every value rendered here arrives already computed by the Python analytics
 * layer. The single transformation the dashboard performs on a series is
 * rebaseSeries() below, which re-expresses an existing cumulative-return
 * series against a later starting point; that identity is asserted against
 * the windowed Python computation in tests/test_export_dashboard_data.py.
 */

const PCT = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const PCT_1 = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

const PCT_0 = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 0,
  maximumFractionDigits: 0,
});

const RATIO = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const MONEY = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const DATE_LONG = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

const DATE_SHORT = new Intl.DateTimeFormat("en-US", {
  month: "short",
  year: "2-digit",
  timeZone: "UTC",
});

const EM_DASH = "—";

export function pct(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  if (digits === 0) return PCT_0.format(value);
  if (digits === 1) return PCT_1.format(value);
  return PCT.format(value);
}

export function pctSigned(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  const sign = value > 0 ? "+" : "";
  return sign + pct(value, digits);
}

export function ratio(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return RATIO.format(value);
}

export function money(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return MONEY.format(value);
}

export function multiple(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return `${RATIO.format(value)}×`;
}

export function count(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return EM_DASH;
  return value.toLocaleString("en-US");
}

export function dateLong(iso) {
  if (!iso) return EM_DASH;
  return DATE_LONG.format(new Date(`${iso}T00:00:00Z`));
}

export function dateShort(iso) {
  if (!iso) return EM_DASH;
  return DATE_SHORT.format(new Date(`${iso}T00:00:00Z`));
}

/** Render a pipeline timestamp in both UTC and the viewer's local zone. */
export function timestamp(iso) {
  if (!iso) return EM_DASH;
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return iso;

  const utc = new Intl.DateTimeFormat("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
    hour12: false,
  }).format(when);

  const local = new Intl.DateTimeFormat("en-GB", {
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
    hour12: false,
  }).format(when);

  return `${utc} UTC · ${local} local`;
}

/** Whole days between an ISO date and today, in UTC. */
export function daysSince(iso) {
  if (!iso) return null;
  const then = new Date(`${iso}T00:00:00Z`).getTime();
  if (Number.isNaN(then)) return null;
  const today = new Date();
  const todayUTC = Date.UTC(
    today.getUTCFullYear(),
    today.getUTCMonth(),
    today.getUTCDate()
  );
  return Math.round((todayUTC - then) / 86400000);
}

export const EMPTY = EM_DASH;

export function signClass(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "";
  return value >= 0 ? "pos" : "neg";
}

/**
 * Re-express a cumulative-return series against a later start index.
 *
 * Given c_t computed from the first observation, the cumulative return from
 * observation i0 onward is (1 + c_t) / (1 + c_i0) - 1. This is an algebraic
 * identity on values the analytics layer already produced, not a new
 * calculation: with i0 = 0 the series is returned unchanged.
 */
export function rebaseSeries(series, startIndex) {
  if (!startIndex) return series;
  const base = 1 + series[startIndex - 1];
  if (!Number.isFinite(base) || base === 0) return series.slice(startIndex);
  return series.slice(startIndex).map((value) => (1 + value) / base - 1);
}

/** Per-asset identity colours, read from the stylesheet so CSS stays the source. */
export function assetColor(ticker) {
  const styles = getComputedStyle(document.documentElement);
  const value = styles.getPropertyValue(`--s-${ticker.toLowerCase()}`).trim();
  return value || styles.getPropertyValue("--ink-tertiary").trim();
}

export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Diverging ramp for correlation cells: negative through neutral to positive.
 * Correlation is bounded on [-1, 1] so the midpoint is fixed at zero, which is
 * what makes a diverging ramp honest here — the neutral colour always sits at
 * a real zero rather than at the middle of whatever the data happens to span.
 */
export function correlationColor(value) {
  const magnitude = Math.min(Math.abs(value), 1);
  const alpha = 0.07 + magnitude * 0.6;
  return value >= 0
    ? `rgba(10, 132, 255, ${alpha.toFixed(3)})`
    : `rgba(229, 72, 77, ${alpha.toFixed(3)})`;
}
