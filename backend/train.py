"""Step 1: build the dataset and train GRU, LSTM and CNN-LSTM. Run: python train.py"""
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data.data_loader import download_water, load_rain, load_water
from models.networks import CNNLSTMModel, GRUModel, LSTMModel
from preprocessing.data_preprocessing import add_features, clean, eligibility, get_scale, make_windows, station_id, to_daily
from settings import DATA_DIR, MODEL_DIR

EPOCHS, BATCH, PATIENCE = 30, 128, 5
N_RAIN_INPUTS, N_LEVEL_INPUTS = 6, 12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = clean(load_water())
    raw["station_id"] = [station_id(r, s) for r, s in zip(raw["river"], raw["station"])]
    stations = raw.groupby("station_id", as_index=False).last()[["station_id", "station", "river", "basin", "state", "lat", "lon"]]
    table = add_features(to_daily(raw), load_rain(stations.dropna(subset=["lat", "lon"])))
    table.to_parquet(DATA_DIR / "table.parquet")
    stations.to_parquet(DATA_DIR / "stations.parquet")
    return table, stations


def batches(arrays: tuple, shuffle: bool = False) -> DataLoader:
    return DataLoader(TensorDataset(*map(torch.as_tensor, arrays)), BATCH, shuffle=shuffle)


@torch.no_grad()
def predict(model: nn.Module, past: np.ndarray, future: np.ndarray, station: np.ndarray) -> np.ndarray:
    model.to(DEVICE).eval()
    parts = [model(p.to(DEVICE), f.to(DEVICE), s.to(DEVICE)).cpu().numpy() for p, f, s in batches((past, future, station))]
    return np.concatenate(parts) if parts else np.zeros((0, 30))


def fit(model: nn.Module, inputs: tuple, target: np.ndarray, mask: np.ndarray, split: np.ndarray, log: bool = False) -> nn.Module:
    """Train with Huber loss, keeping the weights with the best validation loss."""
    optimiser = torch.optim.Adam(model.to(DEVICE).parameters(), 1e-3)
    target = np.nan_to_num(np.log1p(target) if log else target)

    def loss_of(past, future, station, y, ok):
        pred = model(past.to(DEVICE), future.to(DEVICE), station.to(DEVICE))
        pred = torch.log1p(pred) if log else pred
        loss = nn.functional.huber_loss(pred, y.to(DEVICE), reduction="none") * ok.to(DEVICE)
        return loss.sum() / ok.to(DEVICE).sum().clamp(min=1)

    def part(k):
        return tuple(a[split == k] for a in (*inputs, target.astype(np.float32), mask.astype(np.float32)))

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        for batch in batches(part(0), shuffle=True):
            optimiser.zero_grad()
            loss_of(*batch).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val = np.mean([float(loss_of(*b)) for b in batches(part(1))])
        print(f"{type(model).__name__} epoch {epoch} val {val:.4f}")
        if val < best:
            best, bad = val, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif (bad := bad + 1) >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model.cpu()


def rain_future(w: dict) -> np.ndarray:
    return np.concatenate([np.log1p(w["clim"])[..., None], w["cal"]], axis=-1)


def level_future(w: dict, rain_mm: np.ndarray) -> np.ndarray:
    """Model rain forecasts, never observed future rain, feed the water level models."""
    return np.concatenate([np.log1p(rain_mm)[..., None], w["cal"]], axis=-1).astype(np.float32)


def main() -> None:
    download_water()
    table, _ = build_table()
    latest = table.dropna(subset=["level_mean"])["date"].max()
    status = {sid: eligibility(g, latest) for sid, g in table.groupby("station_id")}
    usable = table[table["station_id"].map(status) != "too_few_days"]
    index = {sid: i for i, sid in enumerate(sorted(usable["station_id"].unique()))}
    scales = {sid: get_scale(g.sort_values("date")) for sid, g in usable.groupby("station_id")}
    w = make_windows(usable, scales, index)
    print("windows by split", np.bincount(w["split"]), "stations", len(index), "device", DEVICE)

    gru_inputs = (w["x"][:, :, 6:], rain_future(w).astype(np.float32), w["station"])
    gru = fit(GRUModel(N_RAIN_INPUTS, len(index)), gru_inputs, w["rain"], w["rain_ok"], w["split"], log=True)
    future = level_future(w, predict(gru, *gru_inputs))
    mask = ~np.isnan(w["y"])
    models = {"gru": gru}
    for name, cls in (("lstm", LSTMModel), ("cnn_lstm", CNNLSTMModel)):
        models[name] = fit(cls(N_LEVEL_INPUTS, len(index)), (w["x"], future, w["station"]), w["y"], mask, w["split"])
    MODEL_DIR.mkdir(exist_ok=True)
    torch.save({k: m.state_dict() for k, m in models.items()}, MODEL_DIR / "models.pt")
    (MODEL_DIR / "meta.pkl").write_bytes(pickle.dumps({"index": index, "scales": scales, "status": status, "latest": latest}))


if __name__ == "__main__":
    main()
