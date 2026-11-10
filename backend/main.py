"""FastAPI backend. Serves results.json and live weather. Run: uvicorn main:app"""
import json
import re
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api_services import weather
from settings import ALLOWED_ORIGINS, RESULTS

app = FastAPI(title="River Flood Forecast API")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET"], allow_headers=["*"])
Days = Annotated[int, Query(ge=1, le=30)]
CATEGORIES = ("NORMAL", "WARNING", "DANGER", "CRITICAL", "UNAVAILABLE")
HISTORY_FIELDS = ("dates", "level", "level_max", "rain", "rain_observed")
_data = {"stamp": None, "value": None}


@app.exception_handler(HTTPException)
def http_error(_, error: HTTPException) -> JSONResponse:
    return JSONResponse({"error": {"message": str(error.detail)}}, error.status_code)


def data() -> dict:
    """Latest results, reloaded when the file changes."""
    if not RESULTS.exists():
        raise HTTPException(503, "Forecasts aren't ready yet. Run train.py and then forecast.py.")
    stamp = RESULTS.stat().st_mtime
    if stamp != _data["stamp"]:
        _data.update(stamp=stamp, value=json.loads(RESULTS.read_text(encoding="utf-8")))
    return _data["value"]


def key(text: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def select(river=None, state=None, basin=None, station_id=None) -> list[dict]:
    rows = data()["stations"]
    if station_id:
        rows = [s for s in rows if s["station_id"] == station_id]
        if not rows:
            raise HTTPException(404, f"Unknown station: {station_id}")
    for field, wanted in (("river", river), ("state", state), ("basin", basin)):
        rows = [s for s in rows if not wanted or key(s[field]) == key(wanted)]
    return rows


def counts(rows: list[dict]) -> dict:
    return {c: sum(s["category"] == c for s in rows) for c in CATEGORIES}


def with_forecast(field: str, days: int, river, state, basin, station_id) -> dict:
    stored = data()
    items = []
    for row in select(river, state, basin, station_id):
        entry = stored["forecasts"].get(row["station_id"], {})
        item = {**row, "model": entry.get("model"), "forecast": entry.get(field, [])[:days]}
        if station_id:
            item["history"] = {k: stored["history"][station_id][k] for k in HISTORY_FIELDS}
        items.append(item)
    return {"count": len(items), "days": days, "stations": items, "note": stored["run_metadata"]["uncertainty_note"]}


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "results_ready": RESULTS.exists()}


@app.get("/api/rivers")
def rivers(state: Optional[str] = None, basin: Optional[str] = None) -> dict:
    groups: dict = {}
    for row in select(state=state, basin=basin):
        groups.setdefault(row["river"], []).append(row)
    return {"count": len(groups), "rivers": [{"river": r, "stations": len(v), "categories": counts(v)} for r, v in sorted(groups.items())]}


@app.get("/api/stations")
def stations(river: Optional[str] = None, state: Optional[str] = None, basin: Optional[str] = None, station_id: Optional[str] = None) -> dict:
    rows = select(river, state, basin, station_id)
    return {"count": len(rows), "stations": rows}


@app.get("/api/dashboard")
def dashboard() -> dict:
    stored, rows = data(), select()
    recent = sorted(rows, key=lambda s: s["latest_date"], reverse=True)[:10]
    return {"total_rivers": len({s["river"] for s in rows}), "total_stations": len(rows), "categories": counts(rows),
            "forecast_stations": len(stored["forecasts"]), "stations_with_thresholds": stored["coverage"]["stations_with_thresholds"],
            "exclusions": stored["coverage"]["reasons"], "latest_observations": recent, "run": stored["run_metadata"]}


@app.get("/api/forecast/water-level")
def water_level(river: Optional[str] = None, state: Optional[str] = None, basin: Optional[str] = None, station_id: Optional[str] = None, days: Days = 30) -> dict:
    return with_forecast("water", days, river, state, basin, station_id)


@app.get("/api/forecast/rainfall")
def rainfall(river: Optional[str] = None, state: Optional[str] = None, basin: Optional[str] = None, station_id: Optional[str] = None, days: Days = 30) -> dict:
    answer = with_forecast("rain", days, river, state, basin, station_id)
    return {**answer, "note": "Weather API rainfall covers only the short-term days labelled with it. All other days are GRU model forecasts."}


@app.get("/api/flood-risk")
def flood_risk(river: Optional[str] = None, state: Optional[str] = None, basin: Optional[str] = None, station_id: Optional[str] = None, days: Days = 30) -> dict:
    rows = select(river, state, basin, station_id)
    return {"count": len(rows), "categories": counts(rows), "stations": [{**s, "calendar": data()["risk"].get(s["station_id"], [])[:days]} for s in rows]}


@app.get("/api/early-warnings")
def early_warnings(river: Optional[str] = None, state: Optional[str] = None, basin: Optional[str] = None, station_id: Optional[str] = None) -> dict:
    allowed = {s["station_id"] for s in select(river, state, basin, station_id)}
    stored = data()["alerts"]
    alerts = [a for a in stored["alerts"] if a["station_id"] in allowed]
    return {"count": len(alerts), "alerts": alerts, "rollup": stored["rollup"], "disclaimer": stored["disclaimer"]}


@app.get("/api/rainfall-impact")
def rainfall_impact(station_id: str) -> dict:
    select(station_id=station_id)
    return {"analysis": data()["rainfall_impact"].get(station_id)}


@app.get("/api/model-performance")
def model_performance() -> dict:
    return data()["model_performance"]


def point(lat, lon, station_id) -> tuple[float, float]:
    if station_id:
        row = select(station_id=station_id)[0]
        return row["latitude"], row["longitude"]
    if lat is None or lon is None:
        raise HTTPException(422, "Provide station_id, or lat and lon.")
    return lat, lon


@app.get("/api/weather/current")
def weather_current(lat: Optional[float] = None, lon: Optional[float] = None, station_id: Optional[str] = None) -> dict:
    return weather.current(*point(lat, lon, station_id))


@app.get("/api/weather/forecast")
def weather_forecast(lat: Optional[float] = None, lon: Optional[float] = None, station_id: Optional[str] = None) -> dict:
    return weather.forecast(*point(lat, lon, station_id))
