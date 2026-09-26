// Same origin when served by FastAPI; change if frontend is hosted separately
const API = "/api";

const COLORS = ["#ff9900", "#232f3e", "#146eb4", "#16a34a", "#dc2626", "#8b5cf6", "#0891b2", "#ca8a04", "#db2777", "#64748b"];
const charts = {};
const loaded = {};

// ---------------- Helpers ----------------
const $ = (sel) => document.querySelector(sel);
const inr = (v) => "₹" + Number(v).toLocaleString("en-IN", { maximumFractionDigits: 0 });
const num = (v, d = 0) => Number(v).toLocaleString("en-IN", { maximumFractionDigits: d });
const compact = (v) => "₹" + Intl.NumberFormat("en-IN", { notation: "compact", maximumFractionDigits: 1 }).format(v);

async function api(path, options) {
  $("#loader").classList.add("show");
  try {
    const res = await fetch(API + path, options);
    if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
    return await res.json();
  } finally {
    $("#loader").classList.remove("show");
  }
}

function drawChart(id, type, labels, datasets, extra = {}) {
  charts[id]?.destroy();
  charts[id] = new Chart(document.getElementById(id), {
    type,
    data: { labels, datasets },
    options: {
      responsive: true,
      plugins: { legend: { display: datasets.length > 1 || type === "doughnut" } },
      ...extra,
    },
  });
}

function kpis(el, items) {
  el.innerHTML = items.map(([label, value]) => `<div class="kpi"><span>${label}</span><b>${value}</b></div>`).join("");
}

// columns: [key, header, formatter?, isNumeric?]
function table(el, rows, columns) {
  const head = columns.map(([, h, , n]) => `<th class="${n ? "num" : ""}">${h}</th>`).join("");
  const body = rows.map((r) =>
    "<tr>" + columns.map(([k, , f, n]) =>
      `<td class="${n ? "num" : ""}">${r[k] == null ? "-" : f ? f(r[k]) : r[k]}</td>`).join("") + "</tr>"
  ).join("");
  el.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

// ---------------- Overview ----------------
async function loadOverview() {
  const [s, eda] = await Promise.all([api("/summary"), api("/eda")]);

  kpis($("#kpis"), [
    ["Products", num(s.products)],
    ["Brands", num(s.brands)],
    ["Units Sold", num(s.total_units)],
    ["Net Revenue", compact(s.net_revenue)],
    ["Total Profit", compact(s.total_profit)],
    ["Avg Rating", s.avg_rating + " ★"],
    ["Avg Return Rate", s.avg_return_rate + "%"],
  ]);

  const hbar = { indexAxis: "y" };
  drawChart("catChart", "bar", eda.categories.map((c) => c.name),
    [{ label: "Profit", data: eda.categories.map((c) => c.profit), backgroundColor: COLORS[0] }], hbar);
  drawChart("brandChart", "bar", eda.brands.map((b) => b.name),
    [{ label: "Profit", data: eda.brands.map((b) => b.profit), backgroundColor: COLORS[2] }], hbar);
  drawChart("regionChart", "bar", eda.regions.map((r) => r.name),
    [{ label: "Profit", data: eda.regions.map((r) => r.profit), backgroundColor: COLORS[1] }]);
  drawChart("fulfilChart", "doughnut", eda.fulfilment.map((f) => f.name),
    [{ data: eda.fulfilment.map((f) => f.revenue), backgroundColor: COLORS }]);
  drawChart("ratingChart", "bar", eda.distributions.rating.labels,
    [{ label: "Products", data: eda.distributions.rating.values, backgroundColor: COLORS[3] }]);
  drawChart("discountChart", "bar", eda.distributions.discount.labels,
    [{ label: "Products", data: eda.distributions.discount.values, backgroundColor: COLORS[5] }]);

  table($("#topProducts"), eda.top_profit, [
    ["product_name", "Product"],
    ["brand", "Brand"],
    ["main_category", "Category"],
    ["selling_price_inr", "Price", inr, true],
    ["total_orders", "Orders", num, true],
    ["net_revenue_inr", "Net Revenue", inr, true],
    ["total_profit_inr", "Profit", inr, true],
  ]);
}

// ---------------- Predict ----------------
async function loadPredict() {
  const opts = await api("/options");
  const defaults = { warehouse_region: "North", fulfilment_type: "FBA", demand_tier: "High", main_category: "Electronics" };

  for (const [name, values] of Object.entries(opts)) {
    if (name === "brand") {
      $("#brandList").innerHTML = values.map((v) => `<option value="${v}">`).join("");
      continue;
    }
    const sel = document.querySelector(`select[name="${name}"]`);
    sel.innerHTML = values.map((v) => `<option ${v === defaults[name] ? "selected" : ""}>${v}</option>`).join("");
  }
  updateDerived();
}

// Discount % and margin are derived from price inputs
function updateDerived() {
  const f = $("#predictForm");
  const sp = +f.selling_price_inr.value, mrp = +f.mrp_inr.value, cost = +f.cost_price_inr.value;
  if (mrp > 0) f.discount_percentage.value = Math.max(0, ((mrp - sp) / mrp) * 100).toFixed(1);
  if (sp > 0) f.profit_margin_pct.value = (((sp - cost) / sp) * 100).toFixed(1);
}

async function submitPredict(e) {
  e.preventDefault();
  const f = e.target;
  const data = Object.fromEntries(new FormData(f));
  const text = ["warehouse_region", "fulfilment_type", "demand_tier", "main_category", "brand"];

  const payload = {};
  for (const [k, v] of Object.entries(data)) {
    if (text.includes(k)) payload[k] = v;
    else if (k === "is_seasonal_product") payload[k] = v === "true";
    else payload[k] = Number(v);
  }
  payload.discount_percentage /= 100; // model expects 0-1

  const out = $("#predictResult");
  try {
    const r = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    out.innerHTML = `
      <h3>Prediction</h3>
      <p class="muted">Predicted total units sold</p>
      <div class="big">${num(r.predicted_units)}</div>
      <dl>
        <dt>Estimated Revenue</dt><dd>${inr(r.estimated_revenue)}</dd>
        <dt>Estimated Profit</dt><dd>${inr(r.estimated_profit)}</dd>
        <dt>Model CV MAE</dt><dd>± ${num(r.model_cv_mae, 1)} units</dd>
        <dt>Model CV R²</dt><dd>${r.model_cv_r2}</dd>
      </dl>
      <p class="muted" style="margin-top:14px">Tuned Random Forest trained on historical product data.</p>`;
  } catch (err) {
    out.innerHTML = `<h3>Prediction</h3><p class="error">${err.message}</p>`;
  }
}

// ---------------- Forecast ----------------
async function loadForecast() {
  const months = $("#fcMonths").value;
  const d = await api(`/forecast?months=${months}`);
  const h = d.history, f = d.forecast;
  const labels = [...h.labels, ...f.labels];
  const pad = Array(h.labels.length - 1).fill(null);
  const last = h.values[h.values.length - 1];

  drawChart("fcChart", "line", labels, [
    { label: "Historical", data: h.values, borderColor: COLORS[1], backgroundColor: COLORS[1], tension: .3, pointRadius: 2 },
    { label: "Forecast", data: [...pad, last, ...f.mean], borderColor: COLORS[0], backgroundColor: COLORS[0], borderWidth: 3, tension: .3 },
    { label: "95% Upper", data: [...pad, last, ...f.upper], borderColor: "rgba(255,153,0,.3)", borderDash: [5, 5], pointRadius: 0, fill: "+1", backgroundColor: "rgba(255,153,0,.12)" },
    { label: "95% Lower", data: [...pad, last, ...f.lower], borderColor: "rgba(255,153,0,.3)", borderDash: [5, 5], pointRadius: 0 },
  ], { interaction: { mode: "index", intersect: false } });

  const rows = f.labels.map((m, i) => ({ month: m, mean: f.mean[i], lower: f.lower[i], upper: f.upper[i] }));
  table($("#fcTable"), rows, [
    ["month", "Month"],
    ["mean", "Forecast Units", num, true],
    ["lower", "Lower 95%", num, true],
    ["upper", "Upper 95%", num, true],
  ]);
}

// ---------------- Insights ----------------
async function loadInsights() {
  const d = await api("/insights");
  const r = d.restock, hr = d.high_returns;

  kpis($("#restockKpis"), [
    ["Demand Growth Factor", r.growth_factor + "x"],
    ["Urgent Restock", num(r.status_counts["URGENT restock"] || 0)],
    ["Restock", num(r.status_counts["Restock"] || 0)],
    ["High-Return Products", `${hr.flagged} / ${hr.total}`],
    ["Revenue Lost to Returns", compact(hr.revenue_lost)],
  ]);

  const status = Object.keys(r.status_counts);
  drawChart("restockChart", "doughnut", status,
    [{ data: Object.values(r.status_counts), backgroundColor: status.map((s) =>
      ({ "URGENT restock": "#dc2626", Restock: "#f59e0b", Healthy: "#16a34a", Overstock: "#146eb4" })[s]) }]);
  drawChart("warehouseChart", "bar", Object.keys(r.by_warehouse),
    [{ label: "Units to restock", data: Object.values(r.by_warehouse), backgroundColor: COLORS[0] }]);

  table($("#discountTable"), d.discount_bands, [
    ["main_category", "Category"],
    ["optimal_band", "Optimal Discount"],
    ["n", "Products", num, true],
    ["profit_per_product", "Median Profit / Product", inr, true],
    ["margin", "Median Margin %", (v) => v + "%", true],
  ]);

  const badge = (v) => `<span class="badge ${v.startsWith("URGENT") ? "urgent" : "restock"}">${v}</span>`;
  table($("#restockTable"), r.rows, [
    ["product_name", "Product"],
    ["main_category", "Category"],
    ["warehouse_region", "Warehouse"],
    ["stock_quantity", "Stock", num, true],
    ["monthly_demand", "Monthly Demand", (v) => num(v, 1), true],
    ["months_of_cover", "Months Cover", (v) => num(v, 1), true],
    ["restock_qty", "Restock Qty", num, true],
    ["restock_action", "Action", badge],
  ]);

  $("#returnsTitle").textContent = `High-Return Products (top 10% within category)`;
  table($("#returnsTable"), hr.rows, [
    ["product_name", "Product"],
    ["brand", "Brand"],
    ["main_category", "Category"],
    ["return_rate_pct", "Return %", (v) => v + "%", true],
    ["cat_ret_mean", "Category Avg %", (v) => v + "%", true],
    ["rating", "Rating", null, true],
    ["fulfilment_type", "Fulfilment"],
    ["return_loss_inr", "Revenue Lost", inr, true],
  ]);
}

// ---------------- Tabs ----------------
const loaders = { overview: loadOverview, predict: loadPredict, forecast: loadForecast, insights: loadInsights };

async function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === name));
  if (!loaded[name]) {
    try {
      await loaders[name]();
      loaded[name] = true;
    } catch (err) {
      document.getElementById(name).insertAdjacentHTML("afterbegin",
        `<div class="card error">Failed to load data: ${err.message}</div>`);
    }
  }
}

document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => showTab(t.dataset.tab)));
$("#predictForm").addEventListener("submit", submitPredict);
["selling_price_inr", "mrp_inr", "cost_price_inr"].forEach((n) =>
  document.querySelector(`[name="${n}"]`).addEventListener("input", updateDerived));
$("#fcMonths").addEventListener("change", loadForecast);

showTab("overview");
