"""Download CWC water levels and IMD rainfall, and load them into simple tables."""
import re

import httpx
import imdlib
import pandas as pd

from settings import BASIN_CODE, DATA_DIR, START_YEAR

CWC_PAGE = "https://nwdp.nwic.gov.in/en/dataset/river-water-level-telemetry-hourly-central-water-commission-cwc"
COLUMNS = {"Station": "station", "State": "state", "River": "river", "Basin": "basin", "Latitude": "lat", "Longitude": "lon",
           "RL_of_zeroGauge": "datum", "Data Acquisition Time": "time"}


def download_water() -> None:
    """Fetch this basin's CSV files. Files already on disk are skipped."""
    DATA_DIR.mkdir(exist_ok=True)
    page = httpx.get(CWC_PAGE, timeout=60, follow_redirects=True).text
    links = set(re.findall(rf'href="([^"]+_cwc_{BASIN_CODE}_[^"]+\.csv)"', page))
    for link in links:
        target = DATA_DIR / link.rsplit("/", 1)[-1]
        if target.exists():
            continue
        url = link if link.startswith("http") else f"https://nwdp.nwic.gov.in{link}"
        with httpx.stream("GET", url, timeout=300, follow_redirects=True) as response:
            response.raise_for_status()
            target.write_bytes(response.read())
        print("downloaded", target.name)


def parse_times(values: pd.Series) -> pd.Series:
    """CWC dates are day-first: 01-02-2021 means 1 February."""
    return pd.to_datetime(values, format="%d-%m-%Y %H:%M", errors="coerce")


def load_water() -> pd.DataFrame:
    frames = []
    for path in sorted(DATA_DIR.glob("*.csv")):
        df = pd.read_csv(path, low_memory=False)
        level = next(c for c in df.columns if "Water Level" in c)
        df = df.rename(columns={**COLUMNS, level: "level"})[[*COLUMNS.values(), "level"]]
        df["time"] = parse_times(df["time"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_rain(stations: pd.DataFrame) -> pd.DataFrame:
    """Mean IMD grid rainfall within about 50 km of each station, one row per station and day."""
    folder = DATA_DIR / "imd"
    folder.mkdir(exist_ok=True)
    frames = []
    now = pd.Timestamp.now()
    for year in range(START_YEAR, now.year + 1):
        try:
            grid = imdlib.get_data("rain", year, year, fn_format="yearwise", file_dir=str(folder)).get_xarray()
        except Exception:  # the current year isn't in the yearly archive yet, so use realtime data
            try:
                grid = imdlib.load("rain", f"{year}-01-01", (now - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                                   source="realtime", cache_dir=str(folder)).get_xarray()
            except Exception as error:
                print("no IMD rain for", year, type(error).__name__)
                continue
        for row in stations.itertuples():
            box = grid["rain"].sel(lat=slice(row.lat - 0.5, row.lat + 0.5), lon=slice(row.lon - 0.5, row.lon + 0.5))
            series = box.where(box >= 0).mean(["lat", "lon"]).to_series().dropna()
            frames.append(pd.DataFrame({"station_id": row.station_id, "date": series.index, "rain_mm": series.values}))
    return pd.concat(frames, ignore_index=True)
