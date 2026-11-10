"""Risk classification, early warnings and rainfall impact."""
import csv
import re

import numpy as np
import pandas as pd

ORDER = {"UNAVAILABLE": -1, "NORMAL": 0, "WARNING": 1, "DANGER": 2, "CRITICAL": 3}
DISCLAIMER = "AI-generated decision-support information, not an official government emergency notification."
HEAVY_RAIN_MM = 64.5
MESSAGES = {
    "WARNING": "Warning: River water levels are approaching the warning threshold. Authorities should monitor rainfall and river conditions.",
    "DANGER": "Danger Alert: Predicted water levels have reached the official danger threshold. Prepare flood response measures.",
    "CRITICAL": "Critical Flood Alert: Predicted water levels have reached or exceeded the highest flood threshold. Immediate emergency preparedness is required.",
}
ACTIONS = {"WARNING": "Monitor river and rainfall updates.", "DANGER": "Prepare response teams and review evacuation routes.",
           "CRITICAL": "Activate emergency plans and follow official instructions."}


def name_key(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def load_thresholds(path) -> dict:
    """Official thresholds by station name. Rows out of order are skipped."""
    found = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = {k: float(row[k]) if row[k] else None for k in ("warning", "danger", "hfl")}
            t.update(critical=t.pop("hfl"), source=row["source"], source_date=row["source_date"])
            known = [v for v in (t["warning"], t["danger"], t["critical"]) if v is not None]
            if known and known == sorted(set(known)):
                found[name_key(row["station"])] = t
    return found


def classify(level, t) -> str:
    if t is None or level is None:
        return "UNAVAILABLE"
    if t.get("critical") is not None and level >= t["critical"]:
        return "CRITICAL"
    if t.get("danger") is not None and level >= t["danger"]:
        return "DANGER"
    if t.get("warning") is not None:
        return "WARNING" if level >= t["warning"] else "NORMAL"
    return "UNAVAILABLE"  # missing thresholds are never treated as Normal


def worst(calendar: list[dict]) -> str:
    return max((d["category"] for d in calendar), key=ORDER.get, default="UNAVAILABLE")


def find_crossings(latest: float, days: list[dict], t) -> list[dict]:
    crossings = []
    for name in ("warning", "danger", "critical") if t else ():
        value = t.get(name)
        if value is None:
            continue
        info = {"threshold": name, "category": name.upper(), "value": value}
        if latest >= value:
            crossings.append({**info, "state": "already_crossed", "date": None, "days_remaining": 0})
            continue
        for key, state in (("p50_max", "predicted_crossing"), ("p90_max", "possible_crossing")):
            hit = next((d for d in days if d[key] >= value), None)
            if hit:
                crossings.append({**info, "state": state, "date": hit["date"], "days_remaining": hit["lead"]})
                break
    return crossings


def make_alert(station: dict, days: list[dict], crossings: list[dict], rain: list[dict], rising_fast: bool) -> dict | None:
    if not crossings:
        return None
    top = max(crossings, key=lambda c: ORDER[c["category"]])
    peak = max(days, key=lambda d: d["p50_max"])
    words = {"already_crossed": "The latest level is already above", "predicted_crossing": "The forecast reaches",
             "possible_crossing": "The upper forecast range reaches"}
    reason = f"{words[top['state']]} the {top['threshold']} level."
    heavy = next((d for d in rain if d["rain_mm"] >= HEAVY_RAIN_MM), None)
    reason += " The river rose unusually fast in the last day." if rising_fast else ""
    reason += f" Heavy rain is forecast for {heavy['date']}." if heavy else ""
    return {**{k: station[k] for k in ("station_id", "station", "river", "state", "basin")}, "date": peak["date"],
            "predicted_level": peak["p50_max"], "category": top["category"], "threshold_name": top["threshold"],
            "threshold_m": top["value"], "crossing_state": top["state"], "crossing_date": top["date"],
            "days_remaining": top["days_remaining"], "reason": reason, "message": MESSAGES[top["category"]],
            "action": ACTIONS[top["category"]], "confidence": days[max(top["days_remaining"] - 1, 0)]["confidence"], "disclaimer": DISCLAIMER}


def rising_fast(levels: np.ndarray) -> bool:
    """Last one-day rise beats the station's own 99th percentile."""
    change = np.diff(levels)
    change = change[~np.isnan(change)]
    return len(change) > 30 and bool(change[-1] > max(np.percentile(change[:-1], 99), 0))


def spearman(a: pd.Series, b: pd.Series):
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 60 or pair.nunique().min() < 2:
        return None
    return round(float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")), 3)


def rainfall_impact(df: pd.DataFrame) -> dict:
    rain = df["rain_mm"].where(df["rain_observed"]).reset_index(drop=True)
    level = df["level_mean"].reset_index(drop=True)
    change, monsoon = level.diff(), pd.Series(df["date"].dt.month.between(6, 9).to_numpy())
    windows = {f"{w}d": rain.rolling(w, min_periods=w).sum() for w in (1, 3, 7, 14)}
    lags = range(15)
    rises = [level.iloc[i : i + 8].max() - level.iloc[i] for i in np.where(rain >= HEAVY_RAIN_MM)[0] if pd.notna(level.iloc[i])]
    return {"note": "Correlation shows association between rainfall and river levels. It does not prove causation.",
            "heatmap": {"windows": list(windows), "lags": list(lags), "values": [[spearman(v.shift(k), change) for k in lags] for v in windows.values()]},
            "lag_correlation": {"all_year": [{"lag": k, "correlation": spearman(rain.shift(k), change)} for k in lags],
                                "monsoon": [{"lag": k, "correlation": spearman(rain.shift(k).where(monsoon), change)} for k in lags]},
            "heavy_rain_response": {"events": int((rain >= HEAVY_RAIN_MM).sum()), "measured": len(rises),
                                    "median_rise_m": round(float(np.median(rises)), 3) if rises else None}}
