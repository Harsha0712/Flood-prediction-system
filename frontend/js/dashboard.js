// Home page statistics.
(async () => {
  const data = await apiGet("/api/dashboard");
  const stat = (value, label, cls = "") => `<div class="col-6 col-md-4 col-xl-2"><div class="stat ${cls}"><div class="value">${value}</div><div class="label">${label}</div></div></div>`;
  const c = data.categories;
  document.getElementById("stats").innerHTML = [
    stat(data.total_rivers, "Rivers"), stat(data.total_stations, "Monitoring stations"),
    stat(c.NORMAL, "Normal", "NORMAL"), stat(c.WARNING, "Warning", "WARNING"),
    stat(c.DANGER, "Danger", "DANGER"), stat(c.CRITICAL, "Critical", "CRITICAL"),
  ].join("");
  document.getElementById("coverage").textContent =
    `${data.forecast_stations} of ${data.total_stations} stations have forecasts. ` +
    `${data.stations_with_thresholds} have official thresholds; ${c.UNAVAILABLE} stations show Unavailable and are never counted as Normal. ` +
    `Forecasts generated ${data.run.generated_at}, latest observation ${data.run.latest_observation}.`;
  document.getElementById("latest").innerHTML = data.latest_observations.map((s) => `<tr>
    <td>${escapeHtml(s.river)}</td><td>${escapeHtml(s.station)}</td><td>${escapeHtml(s.state)}</td>
    <td>${fmt(s.latest_level, 2, " m")}</td><td>${escapeHtml(s.latest_date)}</td><td>${riskBadge(s.category)}</td></tr>`).join("");
})().catch(() => {});
