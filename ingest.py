"""
FinSight — Phase 1a: Data Ingestion & Big-Data Analytics
=========================================================
• Downloads OHLCV data via yfinance
• Loads into DuckDB and materialises feature views
• Exports stock_features.parquet for downstream stages
"""
import duckdb
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from config import TICKERS, START_DATE, END_DATE, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Download
# ──────────────────────────────────────────────────────────────────────────────

def download_stock_data() -> pd.DataFrame:
    """Pull historical OHLCV from Yahoo Finance for every ticker."""
    log.info(f"Downloading data for: {TICKERS}")
    frames = []
    for ticker in TICKERS:
        log.info(f"  ↳ {ticker} …")
        df = yf.Ticker(ticker).history(start=START_DATE, end=END_DATE)
        df["Ticker"] = ticker
        df.reset_index(inplace=True)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    # Ensure date column is timezone-naive string
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")
    
    # Ensure directory exists before saving
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(DATA_DIR / "raw_stocks.csv", index=False)
    log.info(f"  ✓ Saved {len(combined):,} rows → raw_stocks.csv")
    return combined


# ──────────────────────────────────────────────────────────────────────────────
# 2. DuckDB Analytics Engine
# ──────────────────────────────────────────────────────────────────────────────

def build_feature_store(df):
    """
    Phase 1a: Big Data Feature Store Generation.
    Explicitly instantiates a local, out-of-core in-memory connection 
    to calculate predictive volatility indicators and technical features
    sequentially over the dataset without triggering nesting conflicts.
    """
    con = duckdb.connect(database=':memory:')
    
    # Execute the updated sequential view layout to resolve structural column gaps
    con.execute("""
        CREATE OR REPLACE VIEW stock_features AS
        WITH base_lags AS (
            SELECT 
                ticker,
                date,
                close,
                volume,
                -- Step 1: Compute historical trailing values cleanly
                LAG(close, 1) OVER (PARTITION BY ticker ORDER BY date) as prev_close,
                LAG(close, 10) OVER (PARTITION BY ticker ORDER BY date) as close_10d_ago
            FROM df
        ),
        base_returns AS (
            SELECT 
                ticker,
                date,
                close,
                volume,
                -- Step 2: Extract normalized mathematical returns and momentum changes
                (close - prev_close) / NULLIF(prev_close, 0) as daily_return,
                LN(close / NULLIF(prev_close, 0)) as log_return,
                close / NULLIF(close_10d_ago, 0) as price_momentum
            FROM base_lags
        )
        SELECT 
            ticker,
            date,
            close,
            volume,
            daily_return,
            log_return,
            price_momentum,
            -- Step 3: Extract structural technical indicators over scalar targets
            AVG(close) OVER (
                PARTITION BY ticker ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as ma_7d,
            AVG(close) OVER (
                PARTITION BY ticker ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
            ) as ma_30d,
            AVG(volume) OVER (
                PARTITION BY ticker ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as vol_ma_7d,
            STDDEV(daily_return) OVER (
                PARTITION BY ticker 
                ORDER BY date 
                ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
            ) as rolling_volatility_20d
        FROM base_returns
        ORDER BY ticker, date
    """)
    
    # Formulate portfolio reporting diagnostics metrics
    summary = con.execute("""
        SELECT
            ticker,
            COUNT(*)                                         AS trading_days,
            ROUND(MIN(close), 2)                             AS min_price,
            ROUND(MAX(close), 2)                             AS max_price,
            ROUND(AVG(close), 2)                             AS avg_price,
            ROUND(AVG(daily_return) * 252 * 100, 2)         AS ann_return_pct,
            ROUND(AVG(rolling_volatility_20d) * 100, 2)     AS avg_vol_pct
        FROM stock_features
        WHERE daily_return IS NOT NULL
        GROUP BY ticker
        ORDER BY ann_return_pct DESC
    """).fetchdf()

    print("\n╔══ Portfolio Analytics Summary ════════════════════════════╗")
    print(summary.to_string(index=False))
    print("╚═══════════════════════════════════════════════════════════╝\n")

    # Filter out initialization lag records and collect features metrics matrix
    features_df = con.execute(
        "SELECT * FROM stock_features WHERE daily_return IS NOT NULL"
    ).fetchdf()
    
    # Ensure directory exists before saving Parquet file distributes
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "stock_features.parquet"
    features_df.to_parquet(out, index=False)
    log.info(f"  ✓ Feature store: {len(features_df):,} rows → {out}")

    con.close()
    return features_df


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_ingestion() -> pd.DataFrame:
    log.info("═══ PHASE 1a ─ Data Ingestion & Big-Data Analytics ═══")
    df       = download_stock_data()
    features = build_feature_store(df)
    log.info("═══ Phase 1a complete ✓ ═══\n")
    return features


if __name__ == "__main__":
    run_ingestion()