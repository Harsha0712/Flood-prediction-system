// Model evaluation page.
const MODEL_NAMES = { lstm: "LSTM", cnn_lstm: "CNN-LSTM", gru: "GRU", persistence: "Persistence baseline", climatology: "Seasonal average baseline" };

function metricRows(table) {
  return Object.entries(table).flatMap(([model, horizons]) => Object.entries(horizons).map(([h, m]) =>
    `<tr><td>${MODEL_NAMES[model] || escapeHtml(model)}</td><td>${h} day</td><td>${fmt(m.mae, 3)}</td><td>${fmt(m.rmse, 3)}</td><td>${fmt(m.r2, 3)}</td><td>${m.n}</td></tr>`)).join("");
}

(async () => {
  const data = await apiGet("/api/model-performance");
  document.getElementById("selected").textContent = `Selected on validation error: ${MODEL_NAMES[data.selection.overall]}. Backtests used ${data.windows.test} forecasts issued monthly on held-out dates, with no future observations as input.`;
  document.getElementById("water").innerHTML = metricRows(data.water_level);
  document.getElementById("rain").innerHTML = data.rainfall ? metricRows(data.rainfall) : `<tr><td colspan="6" class="empty">No rainfall backtest available.</td></tr>`;
  const rows = Object.entries(data.classification).flatMap(([model, horizons]) => Object.entries(horizons).map(([h, m]) =>
    `<tr><td>${MODEL_NAMES[model]}</td><td>${h} day</td><td>${fmt(m.accuracy, 3)}</td><td>${fmt(m.precision, 3)}</td><td>${fmt(m.recall, 3)}</td><td>${fmt(m.f1, 3)}</td><td>${m.support}</td><td>${m.flood_days}</td></tr>`)).join("");
  document.getElementById("class").innerHTML = rows;
  const best = data.classification[data.selection.overall]?.["7"];
  if (best) {
    const head = `<tr><th>Actual \\ Predicted</th>${best.labels.map((l) => `<th>${l}</th>`).join("")}</tr>`;
    document.getElementById("confusion").innerHTML = head + best.labels.map((l, i) => `<tr><th>${l}</th>${best.confusion[i].map((v) => `<td>${v}</td>`).join("")}</tr>`).join("");
  }
})().catch(() => {});
