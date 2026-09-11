/**
 * The detailed analytics table.
 *
 * Sortable on every column, selectable by row, and mirrored into a stacked
 * layout below 720px so the numbers stay readable on a phone instead of
 * requiring a horizontal scroll through nine columns.
 *
 * Sorting is a view operation on values the analytics layer already computed.
 * No column is derived here.
 */

import { assetColor, pct, ratio, signClass } from "./format.js";

const COLUMNS = [
  { key: "ticker", label: "Ticker", type: "text", align: "left" },
  { key: "weight", label: "Weight", type: "pct1" },
  { key: "ann_return", label: "Ann. return", type: "pct", tone: true },
  { key: "ann_volatility", label: "Volatility", type: "pct" },
  { key: "sharpe", label: "Sharpe", type: "ratio", tone: true },
  { key: "beta", label: "Beta", type: "ratio" },
  { key: "max_drawdown", label: "Max drawdown", type: "pct", forceNeg: true },
  { key: "risk_contribution_pct", label: "Risk contrib.", type: "pct1" },
  { key: "trend", label: "Trend", type: "trend", align: "left" },
];

function formatCell(column, row) {
  switch (column.type) {
    case "pct":
      return pct(row[column.key]);
    case "pct1":
      return pct(row[column.key], 1);
    case "ratio":
      return ratio(row[column.key]);
    case "trend":
      return `<span class="tag tag--${
        row.trend === "Bullish" ? "bull" : row.trend === "Bearish" ? "bear" : ""
      }">${row.trend}</span>`;
    default:
      return row[column.key];
  }
}

function cellClass(column, row) {
  const classes = [];
  if (column.type !== "text" && column.type !== "trend") classes.push("num");
  if (column.forceNeg) classes.push("neg");
  else if (column.tone) classes.push(signClass(row[column.key]));
  return classes.join(" ");
}

export function createAnalyticsTable(container, payload, { onSelect, store }) {
  const tickers = payload.meta.tickers;

  container.innerHTML =
    `<div class="table-wrap">` +
    `<table class="data">` +
    `<caption data-caption></caption>` +
    `<thead><tr>` +
    COLUMNS.map(
      (c, i) =>
        `<th scope="col" class="sortable" data-col="${c.key}" aria-sort="none" ` +
        `tabindex="0" role="columnheader"` +
        `${c.align === "left" || i === 0 ? "" : ""}>${c.label}</th>`
    ).join("") +
    `</tr></thead>` +
    `<tbody data-body></tbody>` +
    `</table></div>` +
    `<div class="data-cards" data-cards></div>`;

  const body = container.querySelector("[data-body]");
  const cards = container.querySelector("[data-cards]");
  const caption = container.querySelector("[data-caption]");

  container.querySelectorAll("[data-col]").forEach((th) => {
    const activate = () => {
      const key = th.dataset.col;
      const current = store.get("sortKey");
      const direction =
        current === key ? (store.get("sortDir") === "asc" ? "desc" : "asc") : "desc";
      store.set({ sortKey: key, sortDir: direction });
    };
    th.addEventListener("click", activate);
    th.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        activate();
      }
    });
  });

  return {
    update({ windowKey, ticker, sortKey, sortDir }) {
      const block = payload.windows[windowKey];

      const rows = tickers.map((key) => ({
        ticker: key,
        ...block.assets[key],
        trend: payload.signals.summary[key].trend,
        last_crossover: payload.signals.summary[key].last_crossover,
      }));

      const column = COLUMNS.find((c) => c.key === sortKey) ?? COLUMNS[0];
      rows.sort((a, b) => {
        const av = a[column.key];
        const bv = b[column.key];
        const cmp =
          typeof av === "string" ? av.localeCompare(bv) : (av ?? 0) - (bv ?? 0);
        return sortDir === "asc" ? cmp : -cmp;
      });

      caption.textContent =
        `Per-holding metrics over ${block.label} ` +
        `(${block.start_date} to ${block.end_date}, ${block.observations} sessions). ` +
        `Trend and crossover state are always as of the latest session.`;

      container.querySelectorAll("[data-col]").forEach((th) => {
        th.setAttribute(
          "aria-sort",
          th.dataset.col === sortKey
            ? sortDir === "asc"
              ? "ascending"
              : "descending"
            : "none"
        );
      });

      body.innerHTML = rows
        .map(
          (row) => `
          <tr data-row="${row.ticker}" tabindex="0" aria-selected="${row.ticker === ticker}">
            ${COLUMNS.map((c) =>
              c.key === "ticker"
                ? `<td><span class="cell-ticker"><span class="cell-ticker__dot" style="background:${assetColor(
                    row.ticker
                  )}"></span>${row.ticker}</span></td>`
                : `<td class="${cellClass(c, row)}">${formatCell(c, row)}</td>`
            ).join("")}
          </tr>`
        )
        .join("");

      cards.innerHTML = rows
        .map(
          (row) => `
          <button type="button" class="data-card" data-row="${row.ticker}"
                  aria-selected="${row.ticker === ticker}">
            <span class="data-card__head">
              <span class="cell-ticker"><span class="cell-ticker__dot" style="background:${assetColor(
                row.ticker
              )}"></span>${row.ticker}</span>
              <span class="tag tag--${
                row.trend === "Bullish" ? "bull" : row.trend === "Bearish" ? "bear" : ""
              }">${row.trend}</span>
            </span>
            <span class="data-card__grid">
              <span class="data-card__item">Weight<b>${pct(row.weight, 1)}</b></span>
              <span class="data-card__item">Ann. return<b class="${signClass(
                row.ann_return
              )}">${pct(row.ann_return)}</b></span>
              <span class="data-card__item">Volatility<b>${pct(row.ann_volatility)}</b></span>
              <span class="data-card__item">Sharpe<b class="${signClass(
                row.sharpe
              )}">${ratio(row.sharpe)}</b></span>
              <span class="data-card__item">Beta<b>${ratio(row.beta)}</b></span>
              <span class="data-card__item">Max DD<b class="neg">${pct(
                row.max_drawdown
              )}</b></span>
              <span class="data-card__item">Risk contrib.<b>${pct(
                row.risk_contribution_pct,
                1
              )}</b></span>
            </span>
          </button>`
        )
        .join("");

      container.querySelectorAll("[data-row]").forEach((node) => {
        const select = () => onSelect(node.dataset.row);
        node.addEventListener("click", select);
        node.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            select();
          }
        });
      });
    },
  };
}
