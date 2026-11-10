// Rainfall impact page.
let impactCharts = [];

function heatColor(v) {
  if (v === null) return "#f1f3f5";
  const strength = Math.min(Math.abs(v) * 2, 1);
  return v >= 0 ? `rgba(11,110,153,${strength})` : `rgba(209,52,56,${strength})`;
}

function renderHeatmap(heat) {
  const head = `<tr><th>Rain window</th>${heat.lags.map((l) => `<th>${l}d</th>`).join("")}</tr>`;
  const body = heat.windows.map((name, i) => `<tr><th>${name}</th>${heat.values[i].map((v) =>
    `<td style="background:${heatColor(v)};color:${v !== null && Math.abs(v) > 0.25 ? "#fff" : "inherit"}">${v === null ? "–" : v.toFixed(2)}</td>`).join("")}</tr>`).join("");
  document.getElementById("heatmap").innerHTML = head + body;
}

stationPicker("station-box", async (stationId) => {
  const [{ analysis }, rain, water] = await Promise.all([
    apiGet("/api/rainfall-impact", { station_id: stationId }),
    apiGet("/api/forecast/rainfall", { station_id: stationId }),
    apiGet("/api/forecast/water-level", { station_id: stationId }),
  ]);
  impactCharts.forEach((c) => c.destroy());
  impactCharts = [];
  if (!analysis) {
    document.getElementById("note").textContent = "No rainfall analysis is available for this station.";
    return;
  }
  document.getElementById("note").textContent = analysis.note;
  const heavy = analysis.heavy_rain_response;
  document.getElementById("heavy").textContent = heavy.events
    ? `${heavy.events} heavy rain days (64.5 mm or more). In ${heavy.measured} with level data, the median rise within 7 days was ${fmt(heavy.median_rise_m, 2, " m")}.`
    : "No heavy rain days in the record for this station.";
  renderHeatmap(analysis.heatmap);
  const past = rain.stations[0].history;
  const levels = water.stations[0].history;
  const scales = { x: { ticks: { maxTicksLimit: 8 } }, y: { title: { display: true, text: "Rain (mm)" } }, y1: { position: "right", grid: { display: false }, title: { display: true, text: "Level (m)" } } };
  impactCharts.push(new Chart(document.getElementById("compare-chart"), { data: { labels: past.dates, datasets: [
    { type: "bar", label: "Daily rain", data: past.rain.map((v, i) => (past.rain_observed[i] ? v : null)), backgroundColor: "#0b6e99" },
    { type: "line", label: "River level", data: levels.level, yAxisID: "y1", borderColor: "#17212b", pointRadius: 0, borderWidth: 1.5 },
  ] }, options: { responsive: true, maintainAspectRatio: false, scales, plugins: { legend: { position: "bottom" } } } }));
  const rolling = (n) => past.rain.map((_, i) => (i < n - 1 ? null : past.rain.slice(i - n + 1, i + 1).reduce((a, b) => a + b, 0)));
  impactCharts.push(new Chart(document.getElementById("rolling-chart"), { type: "line", data: { labels: past.dates, datasets: [
    { label: "3-day total", data: rolling(3), borderColor: "#2e9e5b", pointRadius: 0, borderWidth: 1.5 },
    { label: "7-day total", data: rolling(7), borderColor: "#e8772e", pointRadius: 0, borderWidth: 1.5 },
    { label: "14-day total", data: rolling(14), borderColor: "#0b6e99", pointRadius: 0, borderWidth: 1.5 },
  ] }, options: { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { maxTicksLimit: 8 } } }, plugins: { legend: { position: "bottom" } } } }));
  const lags = analysis.lag_correlation;
  impactCharts.push(new Chart(document.getElementById("lag-chart"), { type: "bar", data: { labels: lags.all_year.map((l) => `${l.lag}d`), datasets: [
    { label: "All year", data: lags.all_year.map((l) => l.correlation), backgroundColor: "#8a949e" },
    { label: "June to September", data: lags.monsoon.map((l) => l.correlation), backgroundColor: "#0b6e99" },
  ] }, options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } } }));
});
