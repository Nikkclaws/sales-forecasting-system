"""PyTorch LSTM forecaster with multi-step horizon output and proper scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base import BaseForecaster, ForecastResult


class _LSTMNet:
    """Container so we can lazy-import torch only when needed (keeps base imports light)."""

    def __init__(self):
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(
                self, input_size: int, hidden_size: int, num_layers: int, dropout: float, horizon: int
            ):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
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

            def forward(self, x):  # noqa: D401
                out, _ = self.lstm(x)
                last = out[:, -1, :]
                return self.head(last)

        self.Net = Net
        self.torch = torch


class LSTMForecaster(BaseForecaster):
    name = "lstm"

    def __init__(
        self,
        sequence_length: int = 26,
        horizon: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_size: int = 32,
        epochs: int = 80,
        learning_rate: float = 1e-3,
        patience: int = 10,
        random_state: int = 42,
    ):
        self.sequence_length = sequence_length
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.random_state = random_state

        self.model = None
        self._scaler: StandardScaler | None = None
        self._history: pd.Series | None = None
        self._freq: str = "W-SUN"
        self.residual_std: float = 0.0
        self._best_state = None

    # --- Helpers ---------------------------------------------------------- #

    def _make_supervised(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        seq, hor = self.sequence_length, self.horizon
        X, y = [], []
        for i in range(len(series) - seq - hor + 1):
            X.append(series[i : i + seq])
            y.append(series[i + seq : i + seq + hor])
        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

    # --- Fit -------------------------------------------------------------- #

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> LSTMForecaster:
        env = _LSTMNet()
        torch = env.torch

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        y = history.astype(float).values.reshape(-1, 1)
        if len(y) < self.sequence_length + self.horizon + 5:
            # Not enough data; fall back to a tiny config.
            self.sequence_length = max(8, len(y) // 3)
            if len(y) < self.sequence_length + self.horizon + 5:
                raise ValueError("Series too short for LSTM training.")

        self._scaler = StandardScaler().fit(y)
        y_s = self._scaler.transform(y).flatten()
        self._history = history.copy()
        self._freq = pd.infer_freq(history.index) or history.index.freqstr or "W-SUN"

        X, Y = self._make_supervised(y_s)
        if len(X) < 8:
            raise ValueError("Not enough supervised samples for LSTM training.")

        # Reserve the final 20% as in-training validation for early stopping.
        cut = max(1, int(len(X) * 0.8))
        X_tr, X_va = X[:cut], X[cut:]
        Y_tr, Y_va = Y[:cut], Y[cut:]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = env.Net(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            horizon=self.horizon,
        ).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.MSELoss()

        def _to_tensor(a: np.ndarray) -> torch.Tensor:  # noqa: F821
            return (
                torch.from_numpy(a).float().unsqueeze(-1).to(device)
                if a.ndim == 2
                else torch.from_numpy(a).float().to(device)
            )

        Xt_tr = torch.from_numpy(X_tr).float().unsqueeze(-1).to(device)
        Yt_tr = torch.from_numpy(Y_tr).float().to(device)
        Xt_va = torch.from_numpy(X_va).float().unsqueeze(-1).to(device) if len(X_va) else None
        Yt_va = torch.from_numpy(Y_va).float().to(device) if len(Y_va) else None

        best_val = float("inf")
        no_improve = 0
        for _epoch in range(self.epochs):
            net.train()
            # Mini-batch shuffle
            perm = np.random.permutation(len(X_tr))
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = Xt_tr[idx]
                yb = Yt_tr[idx]
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

            if Xt_va is not None and len(Xt_va) > 0:
                net.eval()
                with torch.no_grad():
                    val_pred = net(Xt_va)
                    val_loss = float(loss_fn(val_pred, Yt_va).item())
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    no_improve = 0
                    self._best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        break

        if self._best_state is not None:
            net.load_state_dict(self._best_state)
        self.model = net
        self._device = str(device)

        # In-sample residual std for CI estimation
        net.eval()
        with torch.no_grad():
            pred_all = net(torch.from_numpy(X).float().unsqueeze(-1).to(device)).cpu().numpy()
        residuals = (Y - pred_all).flatten()
        # Inverse-scale residuals for real-units uncertainty
        scale = self._scaler.scale_[0] if self._scaler else 1.0
        self.residual_std = float(np.std(residuals, ddof=1) * scale) if residuals.size else 0.0
        return self

    # --- Forecast --------------------------------------------------------- #

    def forecast(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
        ci_alpha: float = 0.1,
    ) -> ForecastResult:
        from scipy.stats import norm

        if self.model is None or self._scaler is None or self._history is None:
            raise RuntimeError("Call fit() first.")
        env = _LSTMNet()
        torch = env.torch
        device = torch.device(self._device)

        y_full = self._scaler.transform(self._history.values.reshape(-1, 1)).flatten()
        if len(y_full) < self.sequence_length:
            raise RuntimeError("Insufficient history for LSTM forecast.")
        x_in = y_full[-self.sequence_length :].astype(np.float32).reshape(1, self.sequence_length, 1)
        x_t = torch.from_numpy(x_in).to(device)

        self.model.eval()
        with torch.no_grad():
            yhat_scaled = self.model(x_t).cpu().numpy().flatten()  # shape (model_horizon,)

        # If requested horizon != trained horizon, recurse or truncate.
        model_h = self.horizon
        if horizon <= model_h:
            yhat_scaled = yhat_scaled[:horizon]
        else:
            preds = list(yhat_scaled)
            while len(preds) < horizon:
                window = np.concatenate([y_full, np.array(preds, dtype=np.float32)])[-self.sequence_length :]
                xt = torch.from_numpy(window.astype(np.float32).reshape(1, self.sequence_length, 1)).to(
                    device
                )
                with torch.no_grad():
                    extra = self.model(xt).cpu().numpy().flatten()
                preds.extend(extra.tolist())
            yhat_scaled = np.array(preds[:horizon], dtype=np.float32)

        yhat = self._scaler.inverse_transform(yhat_scaled.reshape(-1, 1)).flatten()
        future_idx = pd.date_range(
            start=self._history.index.max() + pd.tseries.frequencies.to_offset(self._freq),
            periods=horizon,
            freq=self._freq,
        )
        mean = pd.Series(yhat, index=future_idx)
        z = float(norm.ppf(1 - ci_alpha / 2))
        sigmas = self.residual_std * np.sqrt(np.arange(1, horizon + 1))
        lower = pd.Series(yhat - z * sigmas, index=future_idx)
        upper = pd.Series(yhat + z * sigmas, index=future_idx)
        return ForecastResult(
            mean=mean,
            lower=lower,
            upper=upper,
            metadata={"residual_std": self.residual_std, "trained_horizon": self.horizon},
        )

    # ---- Pickling -------------------------------------------------------- #
    def __getstate__(self):
        state = self.__dict__.copy()
        # Replace torch modules with cpu state_dict for portability.
        if self.model is not None:
            state["model"] = {k: v.detach().cpu().numpy() for k, v in self.model.state_dict().items()}
            state["_model_arch"] = {
                "input_size": 1,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "horizon": self.horizon,
            }
        return state

    def __setstate__(self, state):
        arch = state.pop("_model_arch", None)
        weights = state.pop("model", None)
        self.__dict__.update(state)
        if arch is not None and weights is not None:
            env = _LSTMNet()
            torch = env.torch
            net = env.Net(**arch)
            sd = {k: torch.from_numpy(v) for k, v in weights.items()}
            net.load_state_dict(sd)
            net.eval()
            self.model = net
            self._device = "cpu"
