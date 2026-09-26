"""LSTM next-close model with early stopping. CPU compatible."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from app.core.config import settings
from app.core.exceptions import ModelTrainingError, PredictionError
from app.core.logging_config import get_logger
from app.core.seeding import set_global_seed
from app.forecasting.base import BaseForecastModel
from app.forecasting.dataset import SequenceDataset
from app.forecasting.evaluator import evaluate_forecast
from app.forecasting.lstm_network import LSTMNetwork

logger = get_logger(__name__)


class LSTMForecastModel(BaseForecastModel):
    """Sequence model predicting next close from lagged Gold features."""

    model_name = "lstm"

    def __init__(
        self,
        sequence_length: int | None = None,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        epochs: int | None = None,
        patience: int | None = None,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length or settings.lstm_sequence_length
        self.hidden_size = hidden_size or settings.lstm_hidden_size
        self.num_layers = num_layers or settings.lstm_num_layers
        self.dropout = dropout if dropout is not None else settings.lstm_dropout
        self.learning_rate = learning_rate or settings.lstm_learning_rate
        self.batch_size = batch_size or settings.lstm_batch_size
        self.epochs = epochs or settings.lstm_epochs
        self.patience = patience or settings.lstm_patience
        self.device = torch.device("cpu")
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.network: LSTMNetwork | None = None
        self.train_loss_history: list[float] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "LSTMForecastModel":
        set_global_seed(settings.random_seed)
        features = pd.DataFrame(X)
        targets = np.asarray(y, dtype=float).reshape(-1, 1)
        if len(features) <= self.sequence_length + 5:
            raise ModelTrainingError(
                f"LSTM needs more than {self.sequence_length + 5} rows after warmup."
            )
        self.feature_names = list(features.columns)
        scaled_x = self.feature_scaler.fit_transform(features)
        scaled_y = self.target_scaler.fit_transform(targets).ravel()
        dataset = SequenceDataset(scaled_x, scaled_y, self.sequence_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        if X_val is not None and y_val is not None and len(X_val) > self.sequence_length:
            scaled_val_x = self.feature_scaler.transform(pd.DataFrame(X_val, columns=self.feature_names))
            scaled_val_y = self.target_scaler.transform(np.asarray(y_val, dtype=float).reshape(-1, 1)).ravel()
            val_loader = DataLoader(
                SequenceDataset(scaled_val_x, scaled_val_y, self.sequence_length),
                batch_size=self.batch_size,
                shuffle=False,
            )

        self.network = LSTMNetwork(
            input_size=len(self.feature_names),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        best_state = None
        best_val = float("inf")
        wait = 0
        self.network.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            batches = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                output = self.network(batch_x)
                loss = loss_fn(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batches += 1
            train_loss = epoch_loss / max(batches, 1)
            self.train_loss_history.append(train_loss)
            val_loss = train_loss
            if val_loader is not None:
                val_loss = self._average_loss(val_loader, loss_fn)
            logger.info("LSTM epoch %s/%s train_loss=%.6f val_loss=%.6f", epoch + 1, self.epochs, train_loss, val_loss)
            if val_loss + 1e-6 < best_val:
                best_val = val_loss
                wait = 0
                best_state = {k: v.detach().cpu().clone() for k, v in self.network.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    logger.info("LSTM early stopping at epoch %s", epoch + 1)
                    break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.is_fitted = True
        self.metadata = {
            "sequence_length": self.sequence_length,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs_ran": len(self.train_loss_history),
            "patience": self.patience,
        }
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.network is None:
            raise PredictionError("LSTM model is not trained.")
        features = pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
        scaled_x = self.feature_scaler.transform(features)
        if len(scaled_x) < self.sequence_length:
            raise PredictionError("Not enough rows to build an LSTM sequence.")
        self.network.eval()
        preds: list[float] = []
        with torch.no_grad():
            for index in range(self.sequence_length, len(scaled_x) + 1):
                window = scaled_x[index - self.sequence_length : index]
                tensor = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(self.device)
                scaled_pred = self.network(tensor).cpu().numpy()
                preds.append(float(scaled_pred[0][0]))
        predicted = self.target_scaler.inverse_transform(np.asarray(preds).reshape(-1, 1)).ravel()
        return predicted

    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> dict[str, float]:
        frame = pd.DataFrame(X)
        frame = frame.loc[:, ~frame.columns.duplicated()]
        y_true = np.asarray(y, dtype=float).reshape(-1)[self.sequence_length - 1 :]
        y_pred = self.predict(frame)
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]
        previous = None
        if "close" in frame.columns:
            close = np.asarray(frame["close"], dtype=float).reshape(len(frame), -1)[:, 0]
            previous = close[self.sequence_length - 1 : self.sequence_length - 1 + n]
        return evaluate_forecast(y_true, y_pred, previous)

    def save(self, directory: Path) -> Path:
        if self.network is None:
            raise ModelTrainingError("Cannot save an unfitted LSTM model.")
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.network.state_dict(), directory / "lstm.pt")
        joblib_payload = {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
        }
        import joblib

        joblib.dump(joblib_payload, directory / "scalers.joblib")
        (directory / "metadata.json").write_text(
            json.dumps(
                {
                    "feature_names": self.feature_names,
                    "metadata": self.metadata,
                    "sequence_length": self.sequence_length,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return directory

    @classmethod
    def load(cls, directory: Path) -> "LSTMForecastModel":
        import joblib

        payload = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        instance = cls(
            sequence_length=payload["sequence_length"],
            hidden_size=payload["hidden_size"],
            num_layers=payload["num_layers"],
            dropout=payload["dropout"],
        )
        scalers = joblib.load(directory / "scalers.joblib")
        instance.feature_scaler = scalers["feature_scaler"]
        instance.target_scaler = scalers["target_scaler"]
        instance.feature_names = payload.get("feature_names", [])
        instance.network = LSTMNetwork(
            input_size=len(instance.feature_names),
            hidden_size=instance.hidden_size,
            num_layers=instance.num_layers,
            dropout=instance.dropout,
        )
        instance.network.load_state_dict(torch.load(directory / "lstm.pt", map_location="cpu"))
        instance.network.eval()
        instance.metadata = payload.get("metadata", {})
        instance.is_fitted = True
        return instance

    def _average_loss(self, loader: DataLoader, loss_fn: nn.Module) -> float:
        assert self.network is not None
        self.network.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch_x, batch_y in loader:
                output = self.network(batch_x.to(self.device))
                total += float(loss_fn(output, batch_y.to(self.device)).item())
                count += 1
        self.network.train()
        return total / max(count, 1)
