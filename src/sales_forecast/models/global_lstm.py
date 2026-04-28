"""Global multi-state PyTorch LSTM with state embeddings.

Trained jointly across all states, this model shares a single LSTM and a small
state-embedding lookup. Useful as a fallback for low-history states (transfer
learning) and as a benchmark to compare against per-state models.

Persisted alongside the per-state registry under the version's `global/` folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.io import ensure_dir, load_json, save_json
from ..utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class GlobalLSTMArtifacts:
    state_to_id: dict[str, int]
    sequence_length: int
    horizon: int
    hidden_size: int
    num_layers: int
    embed_dim: int
    target_mean: float
    target_std: float


class GlobalLSTMForecaster:
    """One LSTM, K state embeddings, one shared decoder head."""

    def __init__(
        self,
        sequence_length: int = 26,
        horizon: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        embed_dim: int = 8,
        dropout: float = 0.2,
        batch_size: int = 64,
        epochs: int = 60,
        learning_rate: float = 1e-3,
        patience: int = 8,
        random_state: int = 42,
    ):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state

        self.artifacts: GlobalLSTMArtifacts | None = None
        self.model = None
        self._device: str = "cpu"

    # --- Helpers ---------------------------------------------------------- #

    @staticmethod
    def _stack_per_state(series: dict[str, pd.Series]) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
        """Concatenate states along an index axis. Returns y_array, state_to_id, lengths."""
        states = sorted(series.keys())
        state_to_id = {s: i for i, s in enumerate(states)}
        max_len = max(len(s) for s in series.values())
        # Right-aligned padding with NaN; we'll skip NaNs in supervised pair generation.
        arr = np.full((len(states), max_len), np.nan, dtype=np.float32)
        lengths = np.zeros(len(states), dtype=np.int32)
        for s, sid in state_to_id.items():
            v = series[s].values.astype(np.float32)
            arr[sid, -len(v) :] = v
            lengths[sid] = len(v)
        return arr, state_to_id, lengths

    def _make_supervised(
        self, padded: np.ndarray, lengths: np.ndarray, mean: float, std: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        seq, hor = self.sequence_length, self.horizon
        X, Y, S = [], [], []
        for sid in range(padded.shape[0]):
            length = int(lengths[sid])
            series = padded[sid, -length:]
            series_s = (series - mean) / (std if std > 0 else 1.0)
            for i in range(length - seq - hor + 1):
                X.append(series_s[i : i + seq])
                Y.append(series_s[i + seq : i + seq + hor])
                S.append(sid)
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        S = np.asarray(S, dtype=np.int64)
        return X, Y, S

    # --- Fit -------------------------------------------------------------- #

    def fit(self, series_per_state: dict[str, pd.Series]) -> GlobalLSTMForecaster:
        import torch
        from torch import nn

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        padded, state_to_id, lengths = self._stack_per_state(series_per_state)
        flat = padded[~np.isnan(padded)]
        mean = float(np.mean(flat))
        std = float(np.std(flat, ddof=1))
        X, Y, S = self._make_supervised(padded, lengths, mean, std)
        if len(X) < 16:
            raise ValueError("Not enough samples to train global LSTM.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class Net(nn.Module):
            def __init__(self, n_states, embed_dim, hidden_size, num_layers, dropout, horizon):
                super().__init__()
                self.embed = nn.Embedding(n_states, embed_dim)
                self.lstm = nn.LSTM(
                    input_size=1 + embed_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, horizon),
                )

            def forward(self, x_seq, state_ids):
                emb = self.embed(state_ids).unsqueeze(1).expand(-1, x_seq.shape[1], -1)
                inp = torch.cat([x_seq, emb], dim=-1)
                out, _ = self.lstm(inp)
                last = out[:, -1, :]
                return self.head(last)

        net = Net(
            n_states=len(state_to_id),
            embed_dim=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()

        cut = max(1, int(len(X) * 0.85))
        Xt = torch.from_numpy(X).float().unsqueeze(-1).to(device)
        Yt = torch.from_numpy(Y).float().to(device)
        St = torch.from_numpy(S).long().to(device)
        Xt_tr, Xt_va = Xt[:cut], Xt[cut:]
        Yt_tr, Yt_va = Yt[:cut], Yt[cut:]
        St_tr, St_va = St[:cut], St[cut:]

        best_val = float("inf")
        no_improve = 0
        best_state = None
        for _epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xt_tr))
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt_tr[idx], St_tr[idx]), Yt_tr[idx])
                loss.backward()
                opt.step()
            if len(Xt_va) > 0:
                net.eval()
                with torch.no_grad():
                    val = float(loss_fn(net(Xt_va, St_va), Yt_va).item())
                if val < best_val - 1e-6:
                    best_val = val
                    no_improve = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.model = net
        self._device = str(device)
        self.artifacts = GlobalLSTMArtifacts(
            state_to_id=state_to_id,
            sequence_length=self.sequence_length,
            horizon=self.horizon,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            embed_dim=self.embed_dim,
            target_mean=mean,
            target_std=std,
        )
        log.info(
            "Global LSTM trained: %d states, best val MSE=%.6f, n_samples=%d",
            len(state_to_id),
            best_val,
            len(X),
        )
        return self

    # --- Forecast --------------------------------------------------------- #

    def forecast(self, history: pd.Series, state: str, horizon: int | None = None) -> pd.Series:
        import torch

        if self.model is None or self.artifacts is None:
            raise RuntimeError("Call fit() first.")
        sid = self.artifacts.state_to_id.get(state)
        if sid is None:
            raise KeyError(f"Unknown state for global LSTM: {state!r}")
        seq = self.sequence_length
        h_target = horizon or self.horizon
        m, s = self.artifacts.target_mean, self.artifacts.target_std or 1.0
        y = history.astype(float).values
        x_in = (y[-seq:] - m) / s
        x_t = torch.from_numpy(x_in.astype(np.float32).reshape(1, seq, 1)).to(self._device)
        s_t = torch.tensor([sid], dtype=torch.long, device=self._device)
        self.model.eval()
        with torch.no_grad():
            yhat_s = self.model(x_t, s_t).cpu().numpy().flatten()
        if h_target > self.horizon:
            preds = list(yhat_s)
            while len(preds) < h_target:
                window = np.concatenate([(y - m) / s, np.array(preds, dtype=np.float32)])[-seq:]
                xt = torch.from_numpy(window.astype(np.float32).reshape(1, seq, 1)).to(self._device)
                with torch.no_grad():
                    extra = self.model(xt, s_t).cpu().numpy().flatten()
                preds.extend(extra.tolist())
            yhat_s = np.array(preds[:h_target], dtype=np.float32)
        else:
            yhat_s = yhat_s[:h_target]
        yhat = yhat_s * s + m
        # Future weekly Sunday index
        freq = pd.infer_freq(history.index) or history.index.freqstr or "W-SUN"
        future = pd.date_range(
            start=history.index.max() + pd.tseries.frequencies.to_offset(freq),
            periods=h_target,
            freq=freq,
        )
        return pd.Series(yhat, index=future, name="global_lstm")

    # --- Persistence ------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        import torch

        if self.model is None or self.artifacts is None:
            raise RuntimeError("Nothing to save.")
        path = Path(path)
        ensure_dir(path)
        torch.save(self.model.state_dict(), path / "global_lstm.pt")
        save_json(
            {
                "state_to_id": self.artifacts.state_to_id,
                "sequence_length": self.artifacts.sequence_length,
                "horizon": self.artifacts.horizon,
                "hidden_size": self.artifacts.hidden_size,
                "num_layers": self.artifacts.num_layers,
                "embed_dim": self.artifacts.embed_dim,
                "target_mean": self.artifacts.target_mean,
                "target_std": self.artifacts.target_std,
                "dropout": self.dropout,
            },
            path / "config.json",
        )

    @classmethod
    def load(cls, path: str | Path) -> GlobalLSTMForecaster:
        import torch
        from torch import nn

        path = Path(path)
        cfg = load_json(path / "config.json")
        m = cls(
            sequence_length=cfg["sequence_length"],
            horizon=cfg["horizon"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            embed_dim=cfg["embed_dim"],
            dropout=cfg.get("dropout", 0.2),
        )
        m.artifacts = GlobalLSTMArtifacts(
            state_to_id=cfg["state_to_id"],
            sequence_length=cfg["sequence_length"],
            horizon=cfg["horizon"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            embed_dim=cfg["embed_dim"],
            target_mean=cfg["target_mean"],
            target_std=cfg["target_std"],
        )

        class Net(nn.Module):
            def __init__(self, n_states, embed_dim, hidden_size, num_layers, dropout, horizon):
                super().__init__()
                self.embed = nn.Embedding(n_states, embed_dim)
                self.lstm = nn.LSTM(
                    input_size=1 + embed_dim,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, horizon),
                )

            def forward(self, x_seq, state_ids):
                emb = self.embed(state_ids).unsqueeze(1).expand(-1, x_seq.shape[1], -1)
                inp = torch.cat([x_seq, emb], dim=-1)
                out, _ = self.lstm(inp)
                last = out[:, -1, :]
                return self.head(last)

        net = Net(
            n_states=len(cfg["state_to_id"]),
            embed_dim=cfg["embed_dim"],
            hidden_size=cfg["hidden_size"],
            num_layers=cfg["num_layers"],
            dropout=cfg.get("dropout", 0.2),
            horizon=cfg["horizon"],
        )
        net.load_state_dict(torch.load(path / "global_lstm.pt", map_location="cpu"))
        net.eval()
        m.model = net
        m._device = "cpu"
        return m


def build_global_lstm(cfg: Config) -> GlobalLSTMForecaster:
    c = cfg.models.lstm
    return GlobalLSTMForecaster(
        sequence_length=c.sequence_length,
        horizon=c.horizon,
        hidden_size=c.hidden_size,
        num_layers=c.num_layers,
        embed_dim=8,
        dropout=c.dropout,
        batch_size=c.batch_size,
        epochs=c.epochs,
        learning_rate=c.learning_rate,
        patience=c.patience,
        random_state=cfg.project.random_seed,
    )
