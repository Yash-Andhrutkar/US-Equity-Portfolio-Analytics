/**
 * Motion primitives.
 *
 * Motion here is event-driven and then settles completely. A count-up shows a
 * figure arriving, a drawn line shows a series advancing through time, a
 * reveal shows a section taking its place. Nothing animates continuously:
 * there is no ambient drift, no parallax and no idle movement anywhere in the
 * interface, so once the page has settled it is perfectly still.
 *
 * Everything here is a no-op under prefers-reduced-motion, so callers never
 * need to branch.
 */

const reduceQuery = window.matchMedia("(prefers-reduced-motion: reduce)");

export const motion = {
  get reduced() {
    return reduceQuery.matches;
  },
};

const syncFlag = () => {
  document.documentElement.dataset.reducedMotion = String(reduceQuery.matches);
};
reduceQuery.addEventListener("change", syncFlag);
syncFlag();

export const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

/**
 * Animate a numeric value. Jumps straight to the final value under reduced
 * motion.
 */
export function tween({ from, to, duration = 700, ease = easeOutCubic, onUpdate, onDone }) {
  if (motion.reduced || duration <= 0) {
    onUpdate(to, 1);
    onDone?.();
    return () => {};
  }

  let raf = 0;
  let cancelled = false;
  const start = performance.now();

  const step = (now) => {
    if (cancelled) return;
    const t = Math.min(1, (now - start) / duration);
    onUpdate(from + (to - from) * ease(t), t);
    if (t < 1) raf = requestAnimationFrame(step);
    else onDone?.();
  };

  raf = requestAnimationFrame(step);
  return () => {
    cancelled = true;
    cancelAnimationFrame(raf);
  };
}

/**
 * Count a value up to its target. The element stores its last value so a range
 * change animates from where it was rather than snapping back to zero.
 */
export function countUp(element, value, formatter, { duration = 700 } = {}) {
  const previous = Number(element.dataset.value);
  const from = Number.isFinite(previous) ? previous : 0;
  element.dataset.value = String(value);

  if (value === null || value === undefined || Number.isNaN(value)) {
    element.textContent = formatter(value);
    return () => {};
  }

  return tween({
    from,
    to: value,
    duration,
    onUpdate: (current) => {
      element.textContent = formatter(current);
    },
  });
}

/**
 * Reveal elements once as they enter the viewport, with a small stagger so a
 * section assembles rather than appearing all at once. Fires once per element
 * and never replays.
 */
export function observeReveals(root = document) {
  const targets = root.querySelectorAll(".reveal:not([data-shown])");
  if (!targets.length) return;

  if (motion.reduced || !("IntersectionObserver" in window)) {
    targets.forEach((element) => {
      element.dataset.shown = "true";
    });
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry, i) => {
        if (!entry.isIntersecting) return;
        window.setTimeout(() => {
          entry.target.dataset.shown = "true";
        }, i * 60);
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -10% 0px", threshold: 0.06 }
  );

  targets.forEach((element) => observer.observe(element));
}

/** Run a callback the first time an element is scrolled into view. */
export function onFirstVisible(element, callback, { threshold = 0.12 } = {}) {
  if (!("IntersectionObserver" in window)) {
    callback();
    return () => {};
  }

  const observer = new IntersectionObserver(
    (entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        observer.disconnect();
        callback();
      }
    },
    { threshold }
  );

  observer.observe(element);
  return () => observer.disconnect();
}

/**
 * Scroll-spy: reports which section currently occupies the reading position.
 * Uses a band near the top of the viewport rather than element visibility, so
 * the active nav item changes when a section takes over the view rather than
 * when it merely appears at the bottom.
 */
export function observeSections(sections, onChange) {
  if (!("IntersectionObserver" in window)) return () => {};

  const visible = new Map();

  const observer = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        visible.set(entry.target.id, entry.isIntersecting ? entry.intersectionRatio : 0);
      }
      let bestId = null;
      let bestRatio = 0;
      for (const [id, ratio] of visible) {
        if (ratio > bestRatio) {
          bestRatio = ratio;
          bestId = id;
        }
      }
      if (bestId) onChange(bestId);
    },
    {
      rootMargin: "-56px 0px -55% 0px",
      threshold: [0, 0.1, 0.25, 0.5, 0.75, 1],
    }
  );

  sections.forEach((section) => observer.observe(section));
  return () => observer.disconnect();
}

/** Trailing-edge debounce, for resize handlers. */
export function debounce(fn, wait = 160) {
  let timer = 0;
  return (...args) => {
    window.clearTimeout(timer);
    timer = window.setTimeout(() => fn(...args), wait);
  };
}
