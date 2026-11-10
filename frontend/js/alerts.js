// Early warning and flood risk pages.
async function renderAlerts() {
  const { alerts, disclaimer, rollup } = await apiGet("/api/early-warnings");
  document.getElementById("disclaimer").textContent = disclaimer;
  const stateText = { already_crossed: "Already crossed", predicted_crossing: "Predicted crossing", possible_crossing: "Possible crossing" };
  document.getElementById("alerts").innerHTML = alerts.map((a) => `<div class="panel alert-card ${a.category}">
    <div class="d-flex justify-content-between flex-wrap gap-2">
      <strong>${escapeHtml(a.station)}, ${escapeHtml(a.river)} (${escapeHtml(a.state)})</strong>${riskBadge(a.category)}</div>
    <p class="my-2">${escapeHtml(a.message)}</p>
    <div class="row small">
      <div class="col-6 col-md-3">Predicted level<br><strong>${fmt(a.predicted_level, 2, " m")}</strong> on ${escapeHtml(a.date)}</div>
      <div class="col-6 col-md-3">Official ${escapeHtml(a.threshold_name)} level<br><strong>${fmt(a.threshold_m, 2, " m")}</strong></div>
      <div class="col-6 col-md-3">${stateText[a.crossing_state]}<br><strong>${a.crossing_date ? `${escapeHtml(a.crossing_date)} (${a.days_remaining} days)` : "Now"}</strong></div>
      <div class="col-6 col-md-3">Confidence<br><strong class="text-capitalize">${escapeHtml(a.confidence)}</strong></div>
    </div>
    <p class="mt-2 mb-1 small"><strong>Reason:</strong> ${escapeHtml(a.reason)}</p>
    <p class="mb-0 small"><strong>Preparedness:</strong> ${escapeHtml(a.action)}</p></div>`).join("")
    || `<div class="panel empty">No stations with official thresholds are near a flood level in this forecast.</div>`;
  const areas = Object.entries(rollup.states).map(([k, v]) => `${escapeHtml(k)} (${v})`).join(", ");
  document.getElementById("rollup").textContent = areas ? `States needing attention: ${areas}.` : "";
}

async function renderRisk() {
  const { categories } = await apiGet("/api/flood-risk");
  const order = ["NORMAL", "WARNING", "DANGER", "CRITICAL", "UNAVAILABLE"];
  document.getElementById("stats").innerHTML = order.map((c) => `<div class="col"><div class="stat ${c}"><div class="value">${categories[c]}</div><div class="label">${riskBadge(c)}</div></div></div>`).join("");
  stationPicker("station-box", async (stationId) => {
    const { stations } = await apiGet("/api/flood-risk", { station_id: stationId });
    const s = stations[0];
    const t = s.thresholds;
    document.getElementById("thresholds").textContent = t
      ? `Warning ${fmt(t.warning, 2, " m")}, danger ${fmt(t.danger, 2, " m")}, HFL or critical ${fmt(t.critical ?? t.hfl, 2, " m")}. Source date ${t.source_date}.`
      : "No official thresholds are on file for this station, so its risk is Unavailable.";
    document.getElementById("calendar").innerHTML = s.calendar.map((d) => `<div class="day ${d.category}" title="${escapeHtml(d.category)}"><div>${d.date.slice(5)}</div><strong>${fmt(d.predicted_level)}</strong></div>`).join("")
      || `<div class="empty">No forecast for this station.</div>`;
  });
}
