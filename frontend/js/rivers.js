// Searchable table of all stations.
(async () => {
  const { stations } = await apiGet("/api/stations");
  const search = document.getElementById("search");
  const filters = { state: document.getElementById("state"), basin: document.getElementById("basin"), category: document.getElementById("category") };
  ["state", "basin", "category"].forEach((key) => {
    const values = [...new Set(stations.map((s) => s[key]).filter(Boolean))].sort();
    filters[key].insertAdjacentHTML("beforeend", values.map((v) => `<option>${escapeHtml(v)}</option>`).join(""));
  });
  const body = document.getElementById("rows");
  const count = document.getElementById("count");

  function render() {
    const term = search.value.toLowerCase();
    const shown = stations.filter((s) => `${s.river} ${s.station}`.toLowerCase().includes(term)
      && ["state", "basin", "category"].every((k) => !filters[k].value || s[k] === filters[k].value));
    count.textContent = `${shown.length} of ${stations.length} stations`;
    body.innerHTML = shown.slice(0, 500).map((s) => `<tr>
      <td>${escapeHtml(s.river)}</td><td><a href="forecast.html?station=${encodeURIComponent(s.station_id)}">${escapeHtml(s.station)}</a></td>
      <td>${escapeHtml(s.state)}</td><td>${escapeHtml(s.basin)}</td><td>${fmt(s.latest_level, 2, " m")}</td>
      <td>${fmt(s.predicted_max, 2, " m")}</td><td>${escapeHtml(s.trend ?? "–")}</td><td>${riskBadge(s.category)}</td>
      <td>${s.eligible ? "Forecast" : escapeHtml((s.eligibility_reason || "").replaceAll("_", " "))}</td></tr>`).join("")
      || `<tr><td colspan="9" class="empty">No stations match. Clear a filter to see more.</td></tr>`;
  }
  [search, ...Object.values(filters)].forEach((el) => el.addEventListener("input", render));
  render();
})().catch(() => {});
