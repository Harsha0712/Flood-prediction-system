// Water level and rainfall forecast pages.
const charts = {};

function drawChart(id, config) {
  charts[id]?.destroy();
  charts[id] = new Chart(document.getElementById(id), { ...config, options: {
    responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
    scales: { x: { ticks: { maxTicksLimit: 10 } }, ...(config.options?.scales || {}) }, plugins: { legend: { position: "bottom" } },
  } });
}

async function showWaterLevel(stationId) {
  const { stations, note } = await apiGet("/api/forecast/water-level", { station_id: stationId });
  const s = stations[0];
  const past = s.history;
  const keep = -180;
  const dates = [...past.dates.slice(keep), ...s.forecast.map((d) => d.date)];
  const pad = Array(past.dates.slice(keep).length).fill(null);
  const fill = Array(s.forecast.length).fill(null);
  const lines = [{ label: "Observed level", data: [...past.level.slice(keep), ...fill], borderColor: "#17212b", pointRadius: 0, borderWidth: 1.5 },
    { label: "Forecast", data: [...pad, ...s.forecast.map((d) => d.p50)], borderColor: "#0b6e99", pointRadius: 0, borderWidth: 2 },
    { label: "P90", data: [...pad, ...s.forecast.map((d) => d.p90)], borderColor: "#9cc7da", pointRadius: 0, borderWidth: 1, fill: false },
    { label: "P10", data: [...pad, ...s.forecast.map((d) => d.p10)], borderColor: "#9cc7da", pointRadius: 0, borderWidth: 1, fill: "-1", backgroundColor: "rgba(11,110,153,.15)" }];
  const t = s.thresholds || {};
  [["Warning", t.warning, RISK_COLORS.WARNING], ["Danger", t.danger, RISK_COLORS.DANGER], ["HFL", t.critical ?? t.hfl, RISK_COLORS.CRITICAL]]
    .filter(([, v]) => v !== null && v !== undefined)
    .forEach(([label, value, color]) => lines.push({ label, data: dates.map(() => value), borderColor: color, borderDash: [6, 4], pointRadius: 0, borderWidth: 1 }));
  drawChart("level-chart", { type: "line", data: { labels: dates, datasets: lines }, options: { scales: { y: { title: { display: true, text: "Metres" } } } } });
  document.getElementById("summary").innerHTML = `
    <div class="col-6 col-lg-3"><div class="stat"><div class="value">${fmt(s.latest_level, 2)}</div><div class="label">Latest level (m), ${escapeHtml(s.latest_date)}</div></div></div>
    <div class="col-6 col-lg-3"><div class="stat"><div class="value">${fmt(s.predicted_max, 2)}</div><div class="label">Predicted peak (m), ${escapeHtml(s.predicted_max_date ?? "–")}</div></div></div>
    <div class="col-6 col-lg-3"><div class="stat"><div class="value text-capitalize">${escapeHtml(s.trend ?? "–")}</div><div class="label">7-day trend</div></div></div>
    <div class="col-6 col-lg-3"><div class="stat ${s.category}"><div class="value">${riskBadge(s.category)}</div><div class="label">${escapeHtml(s.river)}, ${escapeHtml(s.basin)}, ${escapeHtml(s.state)}</div></div></div>`;
  document.getElementById("note").textContent = s.forecast.length ? `${note} Model used: ${s.model}.` : "No forecast for this station: it doesn't have enough recent data.";
  document.getElementById("rows").innerHTML = s.forecast.map((d) => `<tr><td>${d.date}</td><td>${fmt(d.p50)}</td><td>${fmt(d.p10)} – ${fmt(d.p90)}</td><td>${fmt(d.p50_max)}</td><td>${d.confidence}</td></tr>`).join("");
}

async function showRainfall(stationId) {
  const { stations, note } = await apiGet("/api/forecast/rainfall", { station_id: stationId });
  const s = stations[0];
  const days = 120;
  const past = { dates: s.history.dates.slice(-days), rain: s.history.rain.slice(-days), observed: s.history.rain_observed.slice(-days) };
  const isApi = (d) => d.source !== "gru";
  const labels = [...past.dates, ...s.forecast.map((d) => d.date)];
  const series = (values) => [...values];
  const blanks = (n) => Array(n).fill(null);
  drawChart("rain-chart", { type: "bar", data: { labels, datasets: [
    { label: "Observed", data: series([...past.rain.map((v, i) => (past.observed[i] ? v : null)), ...blanks(s.forecast.length)]), backgroundColor: "#17212b" },
    { label: "Seasonal average (no observation)", data: series([...past.rain.map((v, i) => (past.observed[i] ? null : v)), ...blanks(s.forecast.length)]), backgroundColor: "#c5ccd3" },
    { label: "Weather API forecast", data: series([...blanks(past.dates.length), ...s.forecast.map((d) => (isApi(d) ? d.rain_mm : null))]), backgroundColor: "#e8772e" },
    { label: "GRU model forecast", data: series([...blanks(past.dates.length), ...s.forecast.map((d) => (isApi(d) ? null : d.rain_mm))]), backgroundColor: "#0b6e99" },
  ] }, options: { scales: { x: { stacked: true, ticks: { maxTicksLimit: 10 } }, y: { stacked: true, title: { display: true, text: "Rain (mm per day)" } } } } });
  document.getElementById("note").textContent = note;
  const label = { gru: "GRU model", openweather: "OpenWeather", weatherapi: "WeatherAPI", blended: "OpenWeather and WeatherAPI" };
  document.getElementById("rows").innerHTML = s.forecast.map((d) => `<tr><td>${d.date}</td><td>${fmt(d.rain_mm, 1)}</td><td>${label[d.source] || escapeHtml(d.source)}</td></tr>`).join("");
}
