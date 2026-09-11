/**
 * Shared selection state.
 *
 * One store, one set of subscribers. Every panel that can be driven by a
 * selection reads from here, which is what makes selecting a holding anywhere
 * — a ticker tab, a risk bar, a scatter mark, a diversification-map mark, a
 * matrix diagonal, a table row, a crossover event — update the signal chart,
 * the backtest, the asset metrics and every highlighted mark at once.
 *
 * The store holds view state only: which holding, which range, which toggles,
 * which sort. It never holds a derived financial value.
 */

const DEFAULTS = {
  /** Selected time window key, e.g. "5Y". */
  window: "5Y",
  /** Selected ticker, or null for the whole portfolio. */
  ticker: null,
  /** "cumulative" | "drawdown" */
  performanceMode: "cumulative",
  showPortfolio: true,
  showBenchmark: true,
  showMaShort: true,
  showMaLong: true,
  /** "both" | "buy_hold" | "strategy" */
  backtestMode: "both",
  /** Crossover event index under inspection, or null. */
  eventIndex: null,
  /** Analytics table sort. */
  sortKey: "risk_contribution_pct",
  sortDir: "desc",
  /** Section currently in the reading position, for the nav underline. */
  activeSection: "overview",
};

/** Only these keys are mirrored into the URL. */
const HASH_KEYS = ["window", "ticker"];

class Store {
  constructor(initial) {
    this.state = { ...initial };
    this.listeners = new Set();
    this.suppressHash = false;
  }

  get(key) {
    return key ? this.state[key] : this.state;
  }

  /**
   * Apply a patch and notify subscribers once with the set of changed keys.
   * Unchanged values are dropped so subscribers can cheaply skip work.
   */
  set(patch) {
    const changed = [];
    for (const [key, value] of Object.entries(patch)) {
      if (this.state[key] !== value) {
        this.state[key] = value;
        changed.push(key);
      }
    }
    if (!changed.length) return;
    this.notify(changed);
    if (!this.suppressHash && changed.some((key) => HASH_KEYS.includes(key))) {
      this.writeHash();
    }
  }

  toggle(key) {
    this.set({ [key]: !this.state[key] });
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify(changed) {
    for (const listener of this.listeners) {
      try {
        listener(this.state, changed);
      } catch (error) {
        // A failing panel must not take the rest of the dashboard down.
        console.error("store subscriber failed", error);
      }
    }
  }

  /** Mirror the shareable part of the state into the URL hash. */
  writeHash() {
    const parts = [];
    if (this.state.ticker) parts.push(this.state.ticker.toLowerCase());
    if (this.state.window && this.state.window !== DEFAULTS.window) {
      parts.push(this.state.window.toLowerCase());
    }
    const hash = parts.length ? `#${parts.join("/")}` : "";
    if (hash !== window.location.hash) {
      window.history.replaceState(null, "", hash || window.location.pathname);
    }
  }

  /**
   * Read the URL hash, validating against what the payload actually contains
   * so a stale or hand-edited link cannot put the dashboard into a state with
   * no data behind it. Section anchors are left to the browser.
   */
  readHash(validTickers, validWindows) {
    const raw = window.location.hash.replace(/^#/, "");
    if (!raw) return;

    const patch = {};
    for (const token of raw.split("/")) {
      const upper = token.toUpperCase();
      if (validTickers.includes(upper)) patch.ticker = upper;
      else if (validWindows.includes(upper)) patch.window = upper;
    }

    if (Object.keys(patch).length) {
      this.suppressHash = true;
      this.set(patch);
      this.suppressHash = false;
    }
  }
}

export const store = new Store(DEFAULTS);
export { DEFAULTS };
