"""Clean readings, build daily features and make time-ordered training windows."""
import re

import numpy as np
import pandas as pd

from settings import HORIZON, LOOKBACK, START_YEAR

MAX_SWING_M = 15
LEVEL_COLS = ["level_mean", "level_max", "level_prev", "roll7", "roll14"]
RAIN_COLS = ["rain_mm", "rain_3d", "rain_7d", "rain_14d"]
MIN_DAYS = 365
MAX_STALE_DAYS = 14


def station_id(river: str, station: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", f"{river}-{station}".lower()).strip("-")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, placeholders and one-reading spikes. Nothing is invented."""
    df = df.dropna(subset=["station", "time", "level"]).drop_duplicates(["station", "time"]).sort_values(["station", "time"])
    df = df[df["level"] > 0].copy()
    df["level"] = df["level"].where(df["level"] >= 0.5 * df["datum"], df["level"] + df["datum"])  # gauge height to RL
    df = df[(df["level"] >= df["datum"].fillna(0)) & (df["time"].dt.year >= START_YEAR)]
    middle = df.groupby("station")["level"].transform("median")
    df = df[(df["level"] - middle).abs() <= MAX_SWING_M]  # impossible sensor values
    before, after = df.groupby("station")["level"].shift(1), df.groupby("station")["level"].shift(-1)
    spike = ((df["level"] - before).abs() > 2) & ((df["level"] - after).abs() > 2) & ((before - after).abs() < 1)
    return df[~spike]


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily mean and max per station. Short gaps are interpolated, long gaps stay empty."""
    frames = []
    for sid, group in df.groupby("station_id"):
        daily = group.groupby(group["time"].dt.normalize())["level"].agg(level_mean="mean", level_max="max").asfreq("D")
        daily = daily.interpolate(limit=3, limit_area="inside")
        frames.append(daily.rename_axis("date").reset_index().assign(station_id=sid))
    return pd.concat(frames, ignore_index=True)


def add_features(water: pd.DataFrame, rain: pd.DataFrame) -> pd.DataFrame:
    """Backward-looking features only. Missing rain uses the seasonal average and is flagged."""
    df = water.merge(rain, on=["station_id", "date"], how="left").sort_values(["station_id", "date"])
    df["doy"] = df["date"].dt.dayofyear
    df["rain_observed"] = df["rain_mm"].notna()
    df["rain_clim"] = df.groupby(["station_id", "doy"])["rain_mm"].transform("mean")
    df["rain_clim"] = df.groupby("station_id")["rain_clim"].transform(lambda s: s.rolling(15, center=True, min_periods=1).mean()).fillna(0)
    df["rain_mm"] = df["rain_mm"].fillna(df["rain_clim"])
    level, rainfall = df.groupby("station_id")["level_mean"], df.groupby("station_id")["rain_mm"]
    df["level_prev"], df["level_change"] = level.shift(1), level.diff()
    df["roll7"] = level.transform(lambda s: s.rolling(7, min_periods=3).mean())
    df["roll14"] = level.transform(lambda s: s.rolling(14, min_periods=3).mean())
    for days in (3, 7, 14):
        df[f"rain_{days}d"] = rainfall.transform(lambda s: s.rolling(days, min_periods=1).sum())
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    return df


def eligibility(group: pd.DataFrame, latest: pd.Timestamp) -> str:
    valid = group.dropna(subset=["level_mean"])
    if len(valid) < MIN_DAYS:
        return "too_few_days"
    return "eligible" if valid["date"].max() >= latest - pd.Timedelta(days=MAX_STALE_DAYS) else "stale"


def get_scale(group: pd.DataFrame) -> tuple[float, float]:
    """Centre and spread of a station's level, from its first 70% of days only."""
    train = group["level_mean"].iloc[: int(len(group) * 0.7)].dropna()
    return float(train.median()), max(float(train.quantile(0.75) - train.quantile(0.25)), 0.05)


def scaled_inputs(group: pd.DataFrame, centre: float, spread: float) -> np.ndarray:
    levels = (group[LEVEL_COLS].to_numpy(np.float32) - centre) / spread
    change = group[["level_change"]].to_numpy(np.float32) / spread
    rain = np.log1p(group[RAIN_COLS].to_numpy(np.float32))
    table = np.hstack([levels, change, rain, group[["month_sin", "month_cos"]].to_numpy(np.float32)])
    return pd.DataFrame(table).ffill().fillna(0).to_numpy(np.float32)


def calendar(dates: pd.DatetimeIndex) -> np.ndarray:
    angle = 2 * np.pi * dates.dayofyear / 366
    return np.column_stack([np.sin(angle), np.cos(angle)]).astype(np.float32)


def make_windows(table: pd.DataFrame, scales: dict, station_index: dict, stride: int = 3) -> dict:
    """Windows ending at each origin. Splits follow time: train, then validation, then test."""
    rows = {k: [] for k in ("x", "cal", "clim", "y", "rain", "rain_ok", "station", "last", "spread", "split")}
    for sid, group in table.groupby("station_id"):
        group = group.sort_values("date")
        inputs, cal = scaled_inputs(group, *scales[sid]), calendar(pd.DatetimeIndex(group["date"]))
        levels = group[["level_mean", "level_max"]].to_numpy(np.float32)
        rain, clim, seen = (group[c].to_numpy() for c in ("rain_mm", "rain_clim", "rain_observed"))
        ok, n = ~np.isnan(levels[:, 0]), len(group)
        for t in range(LOOKBACK - 1, n - HORIZON, stride):
            end = t + HORIZON
            split = 0 if end < n * 0.7 else 1 if t >= n * 0.7 and end < n * 0.85 else 2 if t >= n * 0.85 else -1
            if split < 0 or not ok[t] or ok[t - LOOKBACK + 1 : t + 1].mean() < 0.8:
                continue
            future = slice(t + 1, end + 1)
            values = (inputs[t - LOOKBACK + 1 : t + 1], cal[future], clim[future], (levels[future] - levels[t, 0]) / scales[sid][1],
                      rain[future], seen[future], station_index[sid], levels[t, 0], scales[sid][1], split)
            for key, value in zip(rows, values):
                rows[key].append(value)
    return {k: np.array(v, np.float32 if k != "split" and k != "station" else np.int64) for k, v in rows.items()}
