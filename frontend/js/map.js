// India-wide station map.
(async () => {
  const map = L.map("map", { preferCanvas: true }).setView([22.5, 80], 5);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 12, attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);
  const { stations } = await apiGet("/api/stations");
  stations.filter((s) => s.latitude && s.longitude).forEach((s) => {
    const crossing = s.crossing ? `${escapeHtml(s.crossing.date)} (${s.crossing.threshold})` : "None forecast";
    L.circleMarker([s.latitude, s.longitude], {
      radius: 5, color: "#fff", weight: 1, fillColor: RISK_COLORS[s.category], fillOpacity: 0.95,
    }).addTo(map).bindPopup(`<strong>${escapeHtml(s.station)}</strong><br>
      River: ${escapeHtml(s.river)}<br>State: ${escapeHtml(s.state)}<br>
      Latest level: ${fmt(s.latest_level, 2, " m")}<br>Predicted peak: ${fmt(s.predicted_max, 2, " m")}<br>
      Risk: ${riskBadge(s.category)}<br>Expected crossing: ${crossing}<br>
      <a href="forecast.html?station=${encodeURIComponent(s.station_id)}">Open forecast</a>`);
  });
})().catch(() => {});
