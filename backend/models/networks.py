"""The three PyTorch forecasters. Each reads 90 past days and outputs 30 future days."""
import torch
from torch import nn


class Base(nn.Module):
    def __init__(self, n_inputs: int, n_stations: int, n_outputs: int, hidden: int = 64) -> None:
        super().__init__()
        self.embed = nn.Embedding(n_stations, 8)
        self.decoder = nn.GRU(3 + hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_outputs)
        self.n_inputs, self.hidden = n_inputs + 8, hidden

    def with_station(self, past: torch.Tensor, station: torch.Tensor) -> torch.Tensor:
        return torch.cat([past, self.embed(station).unsqueeze(1).expand(-1, past.size(1), -1)], dim=-1)

    def decode(self, context: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        repeated = context.unsqueeze(1).expand(-1, future.size(1), -1)
        return self.out(self.decoder(torch.cat([future, repeated], dim=-1))[0])


class LSTMModel(Base):
    """Water level: 30 days of change in daily mean and daily max."""

    def __init__(self, n_inputs: int, n_stations: int) -> None:
        super().__init__(n_inputs, n_stations, 2)
        self.lstm = nn.LSTM(self.n_inputs, self.hidden, 2, batch_first=True, dropout=0.2)

    def forward(self, past, future, station):
        return self.decode(self.lstm(self.with_station(past, station))[1][0][-1], future)


class CNNLSTMModel(LSTMModel):
    """Water level, with convolutions finding short patterns before the LSTM."""

    def __init__(self, n_inputs: int, n_stations: int) -> None:
        super().__init__(n_inputs, n_stations)
        self.conv = nn.Sequential(nn.Conv1d(self.n_inputs, self.n_inputs, 5, padding=2), nn.ReLU())

    def forward(self, past, future, station):
        patterns = self.conv(self.with_station(past, station).transpose(1, 2)).transpose(1, 2)
        return self.decode(self.lstm(patterns)[1][0][-1], future)


class GRUModel(Base):
    """Rainfall in mm per day, never negative."""

    def __init__(self, n_inputs: int, n_stations: int) -> None:
        super().__init__(n_inputs, n_stations, 1)
        self.gru = nn.GRU(self.n_inputs, self.hidden, 2, batch_first=True, dropout=0.2)

    def forward(self, past, future, station):
        return nn.functional.softplus(self.decode(self.gru(self.with_station(past, station))[1][-1], future)).squeeze(-1)
