// Shared page shell and API access. No provider keys exist in the frontend.
const API_BASE = (window.FLOOD_API_BASE || "http://localhost:8000").replace(/\/$/, "");
const PAGES = [
  ["index.html", "Dashboard"], ["rivers.html", "All Rivers"], ["forecast.html", "Water Level"], ["rainfall.html", "Rainfall"],
  ["flood-risk.html", "Flood Risk"], ["alerts.html", "Early Warnings"], ["rainfall-impact.html", "Rainfall Impact"],
  ["model-performance.html", "Model Performance"],
];
const RISK_COLORS = { NORMAL: "#2e9e5b", WARNING: "#e4b400", DANGER: "#e8772e", CRITICAL: "#d13438", UNAVAILABLE: "#8a949e" };

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmt(value, digits = 2, unit = "") {
  return value === null || value === undefined ? "–" : `${Number(value).toFixed(digits)}${unit}`;
}

function riskBadge(category) {
  const name = category === "UNAVAILABLE" ? "Unavailable" : category[0] + category.slice(1).toLowerCase();
  return `<span class="badge-risk ${escapeHtml(category)}">${name}</span>`;
}

async function apiGet(path, params = {}) {
  const url = new URL(`${API_BASE}${path}`);
  Object.entries(params).forEach(([k, v]) => { if (v) url.searchParams.set(k, v); });
  let response;
  try {
    response = await fetch(url);
  } catch {
    showOffline("The forecast service can't be reached right now. Try again in a minute.");
    throw new Error("offline");
  }
  const body = await response.json();
  if (!response.ok) {
    showOffline(body.error?.message || "The forecast service returned an error. Try again shortly.");
    throw new Error(body.error?.code || "error");
  }
  return body;
}

function showOffline(message) {
  document.getElementById("offline-banner")?.remove();
  const banner = document.createElement("div");
  banner.id = "offline-banner";
  banner.className = "offline";
  banner.textContent = message;
  document.body.prepend(banner);
}

function renderShell() {
  const current = location.pathname.split("/").pop() || "index.html";
  const links = PAGES.map(([href, label]) =>
    `<li class="nav-item"><a class="nav-link ${href === current ? "active" : ""}" href="${href}">${label}</a></li>`).join("");
  document.body.insertAdjacentHTML("afterbegin", `
    <nav class="navbar navbar-expand-xl"><div class="container-fluid">
      <a class="navbar-brand" href="index.html">River Flood Forecast</a>
      <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#nav" aria-label="Open navigation"><span class="navbar-toggler-icon"></span></button>
      <div class="collapse navbar-collapse" id="nav"><ul class="navbar-nav ms-auto">${links}</ul></div>
    </div></nav>`);
  document.body.insertAdjacentHTML("beforeend", `<div class="container"><footer>
    AI-generated decision-support information, not an official government emergency notification.
    Forecasts are uncertain, and confidence falls with lead time.</footer></div>`);
}

// Fills a searchable station box and calls onChange with the chosen station id.
async function stationPicker(inputId, onChange) {
  const input = document.getElementById(inputId);
  const { stations } = await apiGet("/api/stations");
  const usable = stations.filter((s) => s.eligible).length ? stations.filter((s) => s.eligible) : stations;
  const labelFor = (s) => `${s.station} – ${s.river} (${s.state})`;
  const byLabel = new Map(usable.map((s) => [labelFor(s), s]));
  const list = document.createElement("datalist");
  list.id = `${inputId}-list`;
  list.innerHTML = [...byLabel.keys()].map((l) => `<option value="${escapeHtml(l)}">`).join("");
  input.after(list);
  input.setAttribute("list", list.id);
  input.addEventListener("change", () => { const s = byLabel.get(input.value); if (s) onChange(s.station_id); });
  const fromUrl = new URLSearchParams(location.search).get("station");
  const first = usable.find((s) => s.station_id === fromUrl) || usable.find((s) => s.thresholds) || usable[0];
  if (first) { input.value = labelFor(first); onChange(first.station_id); }
}

renderShell();
