"""WeatherAPI and OpenWeather rainfall, with cache and fallback. Keys stay on the server."""
import logging
import time
from datetime import datetime, timedelta, timezone

import httpx

from settings import OPENWEATHER_API_KEY, WEATHERAPI_KEY

logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx would log URLs containing keys
IST = timezone(timedelta(hours=5, minutes=30))
CACHE_SECONDS = 3 * 3600
STALE_SECONDS = 24 * 3600
OW_URL = "https://api.openweathermap.org/data/2.5"
WA_URL = "https://api.weatherapi.com/v1"
_cache: dict = {}


class WeatherError(Exception):
    """Holds no URL or key, so it is safe to log."""


def fetch(url: str, params: dict, client=httpx) -> dict:
    try:
        response = client.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, ValueError):
        raise WeatherError(url.split("/")[2]) from None


def openweather_days(lat: float, lon: float, client=httpx) -> dict[str, float]:
    """Whole days only. The free plan gives 5 days in 3-hour slots."""
    slots = fetch(f"{OW_URL}/forecast", {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}, client)["list"]
    totals: dict[str, list] = {}
    for slot in slots:
        day = datetime.fromtimestamp(slot["dt"], IST).date().isoformat()
        totals.setdefault(day, []).append(slot.get("rain", {}).get("3h", 0.0))
    return {day: round(sum(v), 2) for day, v in totals.items() if len(v) == 8}


def weatherapi_days(lat: float, lon: float, client=httpx) -> dict[str, float]:
    data = fetch(f"{WA_URL}/forecast.json", {"q": f"{lat},{lon}", "days": 3, "key": WEATHERAPI_KEY}, client)
    return {d["date"]: float(d["day"]["totalprecip_mm"]) for d in data["forecast"]["forecastday"]}


def providers() -> list:
    return [p for p in (("weatherapi", WEATHERAPI_KEY, weatherapi_days), ("openweather", OPENWEATHER_API_KEY, openweather_days)) if p[1]]


def combine(results: dict[str, dict[str, float]]) -> list[dict]:
    """One record per date. Providers covering the same date are averaged."""
    dates = sorted({d for days in results.values() for d in days})
    rows = []
    for date in dates:
        names = [n for n, days in results.items() if date in days]
        rows.append({"date": date, "rain_mm": round(sum(results[n][date] for n in names) / len(names), 2), "sources": names})
    return rows


def forecast(lat: float, lon: float) -> dict:
    """Live rainfall, or the other provider, then recent cache, then the model."""
    key = (round(lat, 2), round(lon, 2))
    saved = _cache.get(key)
    if saved and time.time() - saved[0] < CACHE_SECONDS:
        return saved[1]
    results, failed = {}, 0
    for name, _, get_days in providers():
        try:
            results[name] = get_days(lat, lon)
        except WeatherError:
            failed += 1
    if results:
        answer = {"days": combine(results), "fallback": "other_provider" if failed else "none"}
        _cache[key] = (time.time(), answer)
        return answer
    if saved and time.time() - saved[0] < STALE_SECONDS:
        return {**saved[1], "fallback": "cache"}
    return {"days": [], "fallback": "model"}


def current(lat: float, lon: float) -> dict:
    if OPENWEATHER_API_KEY:
        try:
            data = fetch(f"{OW_URL}/weather", {"lat": lat, "lon": lon, "units": "metric", "appid": OPENWEATHER_API_KEY})
            return {"provider": "openweather", "temperature_c": data["main"]["temp"], "rain_mm_last_hour": data.get("rain", {}).get("1h", 0.0), "fallback": "none"}
        except WeatherError:
            pass
    return {"provider": None, "fallback": "model", "message": "Live weather is unavailable"}
