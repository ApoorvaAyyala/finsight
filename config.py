"""
FinSight Platform — Central Configuration
All parameters, paths, and constants live here.
"""
import os
from pathlib import Path

# ─── Directory Layout ──────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MOCKS_DIR  = BASE_DIR / "mocks"

for d in [DATA_DIR, MODELS_DIR, MOCKS_DIR]:
    d.mkdir(exist_ok=True)

# ─── Portfolio Universe ────────────────────────────────────────────────────────
TICKERS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM"]
START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"

# ─── API Keys ──────────────────────────────────────────────────────────────────
# Set GEMINI_API_KEY in your environment; mock data is used when absent.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─── LSTM Hyper-parameters ─────────────────────────────────────────────────────
SEQUENCE_LENGTH = 20      # Look-back window (trading days)
HIDDEN_SIZE     = 64
NUM_LAYERS      = 2
EPOCHS          = 5
BATCH_SIZE      = 32
LEARNING_RATE   = 0.001

# ─── Genetic Algorithm Parameters ──────────────────────────────────────────────
POPULATION_SIZE = 100
GENERATIONS     = 50
CROSSOVER_PROB  = 0.70
MUTATION_PROB   = 0.20
RISK_FREE_RATE  = 0.05    # 5 % annualised (proxy for 10-yr treasury)
