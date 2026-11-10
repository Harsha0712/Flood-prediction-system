"""Tests for the rules that matter most. Sample values here exist only for testing."""
import json

import httpx
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import main
from api_services import weather
from flood_analysis import classify, find_crossings

T = {"warning": 8, "danger": 10, "critical": 12}


def test_classification_uses_thresholds():
    assert [classify(v, T) for v in (7.9, 8, 10, 12)] == ["NORMAL", "WARNING", "DANGER", "CRITICAL"]


def test_missing_thresholds_are_never_normal():
    assert classify(5, None) == "UNAVAILABLE"
    assert classify(9, {"danger": 10}) == "UNAVAILABLE"


def test_crossing_states():
    days = [{"date": "d1", "lead": 1, "p50_max": 7, "p90_max": 8.5}, {"date": "d2", "lead": 2, "p50_max": 7.5, "p90_max": 9}]
    assert find_crossings(7, days, {"warning": 8})[0]["state"] == "possible_crossing"
    assert find_crossings(9, days, {"warning": 8})[0]["state"] == "already_crossed"
    assert find_crossings(7, days, {"warning": 7.2})[0]["days_remaining"] == 2


def test_dates_are_day_first():
    pytest.importorskip("imdlib")
    from data.data_loader import parse_times
    assert parse_times(pd.Series(["01-02-2021 05:30"]))[0].month == 2


def test_providers_combine_per_date():
    rows = weather.combine({"a": {"2026-01-01": 10.0}, "b": {"2026-01-01": 20.0}})
    assert rows == [{"date": "2026-01-01", "rain_mm": 15.0, "sources": ["a", "b"]}]


def test_errors_do_not_expose_key():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(weather.WeatherError) as caught:
        weather.fetch("https://t.test/x", {"appid": "secret-key"}, client)
    assert "secret-key" not in str(caught.value) and caught.value.__cause__ is None


def test_no_weather_falls_back_to_model(monkeypatch):
    monkeypatch.setattr(weather, "providers", list)
    monkeypatch.setattr(weather, "_cache", {})
    assert weather.forecast(1, 2) == {"days": [], "fallback": "model"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    row = {"station_id": "s1", "station": "Test", "river": "Godavari", "state": "Uttar Pradesh", "basin": "Godavari", "category": "NORMAL",
           "latest_date": "2026-11-01", "latitude": 1.0, "longitude": 2.0}
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"stations": [row], "forecasts": {}, "run_metadata": {"uncertainty_note": ""}, "risk": {}}))
    monkeypatch.setattr(main, "RESULTS", path)
    return TestClient(main.app)


@pytest.mark.parametrize("days", ["0", "31", "abc"])
def test_days_are_validated(client, days):
    assert client.get("/api/forecast/water-level", params={"days": days}).status_code == 422


def test_filters_and_unknown_station(client):
    assert client.get("/api/stations", params={"state": "UTTAR  PRADESH"}).json()["count"] == 1
    assert client.get("/api/stations", params={"river": "None"}).json()["stations"] == []
    assert client.get("/api/stations", params={"station_id": "x"}).status_code == 404


def test_missing_results_give_503(client, tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RESULTS", tmp_path / "none.json")
    assert client.get("/api/stations").status_code == 503
