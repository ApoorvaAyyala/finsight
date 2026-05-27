"""
FinSight — Phase 2a: Deep Learning — LSTM Volatility Forecaster
===============================================================
• Loads merged_features.parquet (stock + sentiment)
• Trains a 2-layer LSTM with attention on multi-variate sequences
• Outputs next-period volatility forecasts per ticker
• Saves model weights + scaler + volatility_forecasts.csv
"""
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, MODELS_DIR,
    SEQUENCE_LENGTH, HIDDEN_SIZE, NUM_LAYERS,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "close", "volume", "daily_return",
    "ma_7d", "ma_30d", "vol_ma_7d",
    "price_momentum", "avg_sentiment", "sentiment_confidence",
]
TARGET_COL = "rolling_volatility_20d"


# ──────────────────────────────────────────────────────────────────────────────
# Model Architecture
# ──────────────────────────────────────────────────────────────────────────────

class FinSightLSTM(nn.Module):
    """
    2-layer LSTM + temporal attention → Sharpe-weighted volatility forecast.

    Architecture:
        Input  →  LSTM(hidden=64, layers=2, dropout=0.2)
               →  Attention(linear scoring)
               →  FC(64→32) + ReLU + Dropout
               →  FC(32→1)
    """

    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.2,
        )
        self.attn   = nn.Linear(HIDDEN_SIZE, 1)
        self.fc1    = nn.Linear(HIDDEN_SIZE, 32)
        self.fc2    = nn.Linear(32, 1)
        self.relu   = nn.ReLU()
        self.drop   = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        h0 = torch.zeros(NUM_LAYERS, B, HIDDEN_SIZE, device=x.device)
        c0 = torch.zeros(NUM_LAYERS, B, HIDDEN_SIZE, device=x.device)

        out, _ = self.lstm(x, (h0, c0))                    # (B, T, H)
        scores  = torch.softmax(self.attn(out), dim=1)     # (B, T, 1)
        context = (out * scores).sum(dim=1)                 # (B, H)

        return self.fc2(self.drop(self.relu(self.fc1(context))))


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame):
    """Create sliding-window (X, y) sequences across all tickers."""
    feat_scaler   = MinMaxScaler()
    target_scaler = MinMaxScaler()

    all_X, all_y = [], []

    for ticker in df["ticker"].unique():
        sub = (
            df[df["ticker"] == ticker]
            .sort_values("date")
            .dropna(subset=FEATURE_COLS + [TARGET_COL])
            .copy()
        )
        if len(sub) < SEQUENCE_LENGTH + 1:
            continue

        scaled_feat   = feat_scaler.fit_transform(sub[FEATURE_COLS])
        scaled_target = target_scaler.fit_transform(sub[[TARGET_COL]])

        for i in range(len(sub) - SEQUENCE_LENGTH):
            all_X.append(scaled_feat[i : i + SEQUENCE_LENGTH])
            all_y.append(scaled_target[i + SEQUENCE_LENGTH, 0])

    return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32), feat_scaler, target_scaler


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame):
    # Use only columns that actually exist in this dataset
    available = [c for c in FEATURE_COLS if c in df.columns]
    log.info(f"Feature columns used: {available}")

    X, y, feat_sc, tgt_sc = build_sequences(df)
    log.info(f"Dataset — X: {X.shape}  y: {y.shape}")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    X_tr_t  = torch.tensor(X_tr)
    y_tr_t  = torch.tensor(y_tr).unsqueeze(1)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val).unsqueeze(1)

    model     = FinSightLSTM(input_size=len(available))
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=2, factor=0.5)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, len(X_tr_t), BATCH_SIZE):
            bx = X_tr_t[i : i + BATCH_SIZE]
            by = y_tr_t[i : i + BATCH_SIZE]
            optimiser.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            epoch_loss += loss.item()
            n_batches  += 1

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        avg_tr = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_tr)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        log.info(
            f"  Epoch {epoch}/{EPOCHS}  "
            f"train_loss={avg_tr:.6f}  val_loss={val_loss:.6f}"
        )

    return model, feat_sc, tgt_sc, history, available


# ──────────────────────────────────────────────────────────────────────────────
# Forecast generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_forecasts(model, df: pd.DataFrame, feat_sc, available_cols: list) -> pd.DataFrame:
    model.eval()
    rows = []

    for ticker in df["ticker"].unique():
        sub = (
            df[df["ticker"] == ticker]
            .sort_values("date")
            .dropna(subset=available_cols)
        )
        if len(sub) < SEQUENCE_LENGTH:
            log.warning(f"  {ticker}: insufficient data — using default volatility 0.25")
            rows.append({"ticker": ticker, "predicted_volatility": 0.25})
            continue

        last_seq = sub[available_cols].values[-SEQUENCE_LENGTH:]
        scaled   = feat_sc.transform(last_seq)

        with torch.no_grad():
            pred = model(torch.tensor(scaled[None], dtype=torch.float32)).item()

        rows.append({"ticker": ticker, "predicted_volatility": abs(pred)})

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_deep_learning():
    log.info("═══ PHASE 2a ─ Deep Learning (LSTM) ═══")

    df = pd.read_parquet(DATA_DIR / "merged_features.parquet")
    model, feat_sc, tgt_sc, history, available = train_model(df)

    forecasts = generate_forecasts(model, df, feat_sc, available)

    # Persist artefacts
    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pth")
    with open(MODELS_DIR / "feat_scaler.pkl", "wb") as f:
        pickle.dump(feat_sc, f)
    with open(MODELS_DIR / "tgt_scaler.pkl", "wb") as f:
        pickle.dump(tgt_sc, f)

    forecasts.to_csv(DATA_DIR / "volatility_forecasts.csv", index=False)

    # Also persist training history for dashboard
    import json
    history_path = DATA_DIR / "training_history.json"
    history_path.write_text(json.dumps(history))

    print("\n╔══ Volatility Forecasts ════════════════╗")
    print(forecasts.sort_values("predicted_volatility").to_string(index=False))
    print("╚═══════════════════════════════════════╝\n")

    log.info("═══ Phase 2a complete ✓ ═══\n")
    return model, forecasts, history


if __name__ == "__main__":
    run_deep_learning()
