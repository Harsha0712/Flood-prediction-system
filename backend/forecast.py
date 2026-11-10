"""Step 2: backtest the models, make 30-day forecasts and write results.json. Run: python forecast.py"""
import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

from api_services.weather import forecast as weather_forecast
from flood_analysis import DISCLAIMER, classify, find_crossings, load_thresholds, make_alert, name_key, rainfall_impact, rising_fast, worst
from models.networks import CNNLSTMModel, GRUModel, LSTMModel
from preprocessing.data_preprocessing import calendar, make_windows, scaled_inputs
from settings import DATA_DIR, EVAL_HORIZONS, HORIZON, LOOKBACK, MODEL_DIR, RESULTS, THRESHOLDS
from train import N_LEVEL_INPUTS, N_RAIN_INPUTS, level_future, predict, rain_future

LABELS = ["NORMAL", "WARNING", "DANGER", "CRITICAL"]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    keep = ~(np.isnan(y) | np.isnan(p))
    y, p = y[keep], p[keep]
    if not len(y):
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    total = ((y - y.mean()) ** 2).sum()
    return {"mae": float(np.abs(y - p).mean()), "rmse": float(np.sqrt(((y - p) ** 2).mean())),
            "r2": float(1 - ((y - p) ** 2).sum() / total) if total else None, "n": int(len(y))}


def by_horizon(y: np.ndarray, p: np.ndarray) -> dict:
    return {str(h): metrics(y[:, h - 1], p[:, h - 1]) for h in EVAL_HORIZONS}


def class_metrics(y: list[str], p: list[str]) -> dict:
    confusion = [[sum(a == t and b == q for a, b in zip(y, p)) for q in LABELS] for t in LABELS]
    flood_y, flood_p = np.array([a != "NORMAL" for a in y], bool), np.array([b != "NORMAL" for b in p], bool)
    hits = int((flood_y & flood_p).sum())
    precision = hits / flood_p.sum() if flood_p.sum() else None
    recall = hits / flood_y.sum() if flood_y.sum() else None
    return {"accuracy": sum(a == b for a, b in zip(y, p)) / len(y) if y else None, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision and recall else None,
            "confusion": confusion, "labels": LABELS, "support": len(y), "flood_days": int(flood_y.sum())}


def evaluate(w: dict, models: dict, future: np.ndarray, station_t: list) -> tuple[dict, dict]:
    """Test on the final 15% of each station's dates. Returns scores and validation error bands."""
    val, test = w["split"] == 1, w["split"] == 2
    to_level = lambda delta, part: w["last"][part, None, None] + delta * w["spread"][part, None, None]
    actual = lambda part: w["last"][part, None, None] + w["y"][part] * w["spread"][part, None, None]
    water, classes, val_rmse, bands = {}, {}, {}, {}
    for name in ("lstm", "cnn_lstm"):
        val_delta = predict(models[name], w["x"][val], future[val], w["station"][val])
        val_rmse[name] = metrics(actual(val)[..., 0].ravel(), to_level(val_delta, val)[..., 0].ravel())["rmse"]
        residual = w["y"][val] - val_delta
        bands[name] = {k: np.nanpercentile(residual[..., c], q, axis=0).tolist() for k, c, q in (("p10", 0, 10), ("p90", 0, 90), ("p90_max", 1, 90))}
        pred = to_level(predict(models[name], w["x"][test], future[test], w["station"][test]), test)
        water[name] = by_horizon(actual(test)[..., 0], pred[..., 0])
        classes[name] = {}
        for h in EVAL_HORIZONS:
            t_list = [t for t, keep in zip(station_t, test) if keep]
            pairs = [(classify(float(a), t), classify(float(b), t)) for a, b, t in zip(actual(test)[:, h - 1, 1], pred[:, h - 1, 1], t_list) if t and not np.isnan(a)]
            pairs = [p for p in pairs if "UNAVAILABLE" not in p]
            classes[name][str(h)] = class_metrics([a for a, _ in pairs], [b for _, b in pairs])
    water["persistence"] = by_horizon(actual(test)[..., 0], np.repeat(w["last"][test, None], HORIZON, 1))
    rain_y = np.where(w["rain_ok"][test] > 0, w["rain"][test], np.nan)
    rain_pred = predict(models["gru"], w["x"][test][:, :, 6:], rain_future(w)[test].astype(np.float32), w["station"][test])
    scores = {"water_level": water, "classification": classes, "windows": {"test": int(test.sum())},
              "rainfall": {"gru": by_horizon(rain_y, rain_pred), "climatology": by_horizon(rain_y, w["clim"][test])},
              "selection": {"overall": min(val_rmse, key=val_rmse.get), "validation_rmse": val_rmse}}
    return scores, bands


def station_forecast(group, sid_index, centre, spread, gru, model, band, api_days, latest) -> dict:
    """Forecast one station from data up to the latest date only."""
    group = group.sort_values("date").tail(LOOKBACK)
    dates = pd.date_range(latest + pd.Timedelta(days=1), periods=HORIZON)
    cal = torch.from_numpy(calendar(dates)).unsqueeze(0)
    past, sid = torch.from_numpy(scaled_inputs(group, centre, spread)).unsqueeze(0), torch.tensor([sid_index])
    clim = group.set_index(group["date"].dt.dayofyear)["rain_clim"]
    climate = [float(clim[~clim.index.duplicated()].get(d.dayofyear, clim.mean())) for d in dates]
    days = torch.cat([torch.log1p(torch.tensor(climate, dtype=torch.float32))[None, :, None], cal], -1)
    with torch.no_grad():
        gru_mm = gru(past[:, :, 6:], days, sid)[0].numpy()
    api = {d["date"]: d for d in api_days}
    rain = []
    for date, mm in zip(dates, gru_mm):
        key = date.date().isoformat()
        rain.append({"date": key, "rain_mm": api[key]["rain_mm"], "source": "+".join(api[key]["sources"])} if key in api
                    else {"date": key, "rain_mm": round(float(mm), 2), "source": "gru"})
    rain_values = torch.tensor([r["rain_mm"] for r in rain], dtype=torch.float32)
    with torch.no_grad():
        delta = model(past, torch.cat([torch.log1p(rain_values)[None, :, None], cal], -1), sid)[0].numpy()
    last = float(group["level_mean"].dropna().iloc[-1])
    water = []
    for i, date in enumerate(dates):
        p50, top = last + float(delta[i, 0]) * spread, last + float(delta[i, 1]) * spread
        water.append({"date": date.date().isoformat(), "lead": i + 1, "p50": round(p50, 3),
                      "p10": round(p50 + min(band["p10"][i], 0) * spread, 3), "p90": round(p50 + max(band["p90"][i], 0) * spread, 3),
                      "p50_max": round(max(top, p50), 3), "p90_max": round(max(top, p50) + max(band["p90_max"][i], 0) * spread, 3),
                      "confidence": "high" if i < 7 else "medium" if i < 15 else "low"})
    return {"water": water, "rain": rain}


def main() -> None:
    meta = pickle.loads((MODEL_DIR / "meta.pkl").read_bytes())
    table, stations = pd.read_parquet(DATA_DIR / "table.parquet"), pd.read_parquet(DATA_DIR / "stations.parquet")
    index, scales, status, latest = meta["index"], meta["scales"], meta["status"], meta["latest"]
    n = len(index)
    saved = torch.load(MODEL_DIR / "models.pt", weights_only=True)
    models = {"gru": GRUModel(N_RAIN_INPUTS, n), "lstm": LSTMModel(N_LEVEL_INPUTS, n), "cnn_lstm": CNNLSTMModel(N_LEVEL_INPUTS, n)}
    for name, model in models.items():
        model.load_state_dict(saved[name])
        model.eval()
    thresholds = load_thresholds(THRESHOLDS)
    names = dict(zip(stations["station_id"], stations["station"]))
    t_for = {sid: thresholds.get(name_key(names[sid])) for sid in index}
    w = make_windows(table[table["station_id"].isin(index)], scales, index)
    rain_inputs = (w["x"][:, :, 6:], rain_future(w).astype(np.float32), w["station"])
    future = level_future(w, predict(models["gru"], *rain_inputs))
    ordered = sorted(index, key=index.get)
    scores, bands = evaluate(w, models, future, [t_for[ordered[i]] for i in w["station"]])
    best = scores["selection"]["overall"]
    print(json.dumps(scores["water_level"], indent=1))

    rows, forecasts, history, risk, alerts, impact = [], {}, {}, {}, [], {}
    groups = dict(tuple(table.groupby("station_id")))
    for s in stations.to_dict("records"):
        sid = s["station_id"]
        group, t = groups[sid].sort_values("date"), thresholds.get(name_key(s["station"]))
        seen = group.dropna(subset=["level_mean"])
        row = {**s, "latitude": s.pop("lat"), "longitude": s.pop("lon"), "eligible": status[sid] == "eligible", "eligibility_reason": status[sid],
               "thresholds": t, "latest_level": float(seen["level_mean"].iloc[-1]), "latest_date": seen["date"].iloc[-1].date().isoformat(),
               "current_category": classify(float(seen["level_max"].iloc[-1]), t), "category": "UNAVAILABLE",
               "predicted_max": None, "predicted_max_date": None, "trend": None, "crossing": None}
        tail = group.tail(365)
        history[sid] = {"dates": tail["date"].dt.strftime("%Y-%m-%d").tolist(), "level": tail["level_mean"].round(3).tolist(),
                        "level_max": tail["level_max"].round(3).tolist(), "rain": tail["rain_mm"].round(2).tolist(), "rain_observed": tail["rain_observed"].tolist()}
        impact[sid] = rainfall_impact(group)
        if row["eligible"]:
            api_days = weather_forecast(row["latitude"], row["longitude"])["days"]
            result = station_forecast(seen, index[sid], *scales[sid], models["gru"], models[best], bands[best], api_days, latest)
            forecasts[sid] = {**result, "model": best}
            risk[sid] = [{"date": d["date"], "predicted_level": d["p50_max"], "category": classify(d["p50_max"], t), "confidence": d["confidence"]} for d in result["water"]]
            peak = max(result["water"], key=lambda d: d["p50_max"])
            crossings = find_crossings(row["latest_level"], result["water"], t)
            change = result["water"][6]["p50"] - row["latest_level"]
            row.update(category=worst(risk[sid]), predicted_max=peak["p50_max"], predicted_max_date=peak["date"],
                       trend="steady" if abs(change) < 0.01 else "rising" if change > 0 else "falling", crossing=next((c for c in crossings if c["date"]), None))
            alert = make_alert(row, result["water"], crossings, result["rain"], rising_fast(tail["level_mean"].to_numpy()))
            alerts += [alert] if alert else []
        rows.append(row)
    alerts.sort(key=lambda a: (-LABELS.index(a["category"]), a["days_remaining"]))
    state_counts = pd.Series([a["state"] for a in alerts], dtype=str).value_counts().to_dict()
    results = {"stations": rows, "forecasts": forecasts, "history": history, "risk": risk, "rainfall_impact": impact, "model_performance": scores,
               "alerts": {"alerts": alerts, "rollup": {"states": state_counts}, "disclaimer": DISCLAIMER},
               "coverage": {"reasons": pd.Series(list(status.values())).value_counts().to_dict(), "stations_with_thresholds": sum(1 for r in rows if r["thresholds"])},
               "run_metadata": {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "latest_observation": latest.date().isoformat(),
                                "uncertainty_note": "Bands come from validation errors. Confidence falls with lead time, and beyond about a week forecasts mainly reflect seasonal patterns."}}
    text = json.dumps(results, separators=(",", ":")).replace("NaN", "null")
    RESULTS.write_text(text, encoding="utf-8")
    print(f"{len(rows)} stations, {len(forecasts)} forecasts, {len(alerts)} alerts, {len(text) // 1024} KB")


if __name__ == "__main__":
    main()
