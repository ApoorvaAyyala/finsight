"""
FinSight — Phase 1b: NLP Sentiment Pipeline
============================================
• Generates financial headlines (mock or via Gemini API)
• Extracts daily sentiment vectors per ticker
• Merges with stock_features.parquet → merged_features.parquet

Set the GEMINI_API_KEY env variable to enable real API calls.
Without it, statistically realistic mock scores are generated.
"""
import json
import time
import random
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from config import TICKERS, START_DATE, END_DATE, GEMINI_API_KEY, DATA_DIR, MOCKS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_HEADLINE_TEMPLATES = [
    "{t} beats Q{q} earnings estimates by wide margin",
    "{t} shares rally after analyst upgrade to 'strong buy'",
    "{t} faces antitrust probe; shares slide on regulatory concerns",
    "{t} announces strategic AI partnership, boosting investor sentiment",
    "{t} misses revenue targets; management cites macro headwinds",
    "Institutional investors increase stake in {t} amid sector rotation",
    "{t} CEO outlines five-year growth roadmap at investor day",
    "Market volatility weighs on {t} as broader indices retreat",
    "{t} cutting workforce by 8% in cost-efficiency restructuring",
    "{t} secures major government contract worth billions",
]


# ──────────────────────────────────────────────────────────────────────────────
# Headline generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_headlines(tickers: list, start: str, end: str) -> list[dict]:
    """Create realistic-looking mock headlines for each ticker × business day."""
    records = []
    bdays = pd.bdate_range(start, end)[:120]          # ~6 months of trading days
    for date in bdays:
        for ticker in tickers[:6]:                     # 6 tickers to keep volume sensible
            tmpl = random.choice(_HEADLINE_TEMPLATES)
            records.append({
                "ticker":   ticker,
                "date":     str(date.date()),
                "headline": tmpl.format(t=ticker, q=random.randint(1, 4)),
            })
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Gemini sentiment extraction
# ──────────────────────────────────────────────────────────────────────────────

def _call_gemini(model, batch: list[dict]) -> list[dict] | None:
    """Send one batch of headlines to Gemini; return parsed JSON or None."""
    prompt = (
        "Analyse the financial sentiment of these news headlines.\n"
        "Return ONLY a valid JSON array — no markdown, no preamble.\n"
        "Schema per element: {ticker, date, sentiment_score [-1,1], confidence [0,1]}\n\n"
        + json.dumps(batch, indent=2)
    )
    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip().lstrip("```json").rstrip("```").strip()
        return json.loads(text)
    except Exception as exc:
        log.warning(f"Gemini call failed ({exc}); falling back to mock for this batch.")
        return None


def get_sentiments(headlines: list[dict], use_mock: bool = False) -> list[dict]:
    """Return sentiment scores, using cache → Gemini → mock (in priority order)."""
    cache = MOCKS_DIR / "sentiments_cache.json"

    if cache.exists():
        log.info("Loading cached sentiment data …")
        return json.loads(cache.read_text())

    results: list[dict] = []

    if GEMINI_API_KEY and not use_mock:
        log.info("Calling Gemini API for sentiment analysis …")
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gmodel = genai.GenerativeModel("gemini-1.5-flash")

        BATCH = 10
        for i in range(0, len(headlines), BATCH):
            batch = headlines[i : i + BATCH]
            parsed = _call_gemini(gmodel, batch)
            if parsed:
                results.extend(parsed)
            else:
                results.extend(_mock_sentiments(batch))
            time.sleep(0.8)
            log.info(f"  {min(i + BATCH, len(headlines))}/{len(headlines)} headlines processed")
    else:
        log.info("No GEMINI_API_KEY found — generating mock sentiment vectors …")
        results = _mock_sentiments(headlines)

    # Persist cache
    cache.write_text(json.dumps(results, indent=2))
    log.info(f"  ✓ {len(results)} sentiment scores cached → {cache}")
    return results


def _mock_sentiments(records: list[dict]) -> list[dict]:
    """Generate statistically realistic mock sentiment scores."""
    out = []
    for r in records:
        score = float(np.clip(np.random.normal(0.08, 0.38), -1, 1))
        out.append({
            "ticker":          r["ticker"],
            "date":            r["date"],
            "headline":        r.get("headline", ""),
            "sentiment_score": round(score, 4),
            "confidence":      round(float(np.random.uniform(0.55, 0.95)), 4),
        })
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Merge with structural stock data
# ──────────────────────────────────────────────────────────────────────────────

def merge_sentiment_with_stocks(sentiments: list[dict]) -> pd.DataFrame:
    stocks = pd.read_parquet(DATA_DIR / "stock_features.parquet")
    stocks["date"] = stocks["date"].astype(str)

    sent_df = pd.DataFrame(sentiments)
    sent_df["date"] = sent_df["date"].astype(str)

    daily_agg = (
        sent_df.groupby(["ticker", "date"])
        .agg(
            avg_sentiment       = ("sentiment_score", "mean"),
            sentiment_std       = ("sentiment_score", "std"),
            sentiment_confidence= ("confidence", "mean"),
            headline_count      = ("ticker", "count"),
        )
        .reset_index()
    )

    merged = stocks.merge(daily_agg, on=["ticker", "date"], how="left")
    merged["avg_sentiment"]        = merged["avg_sentiment"].fillna(0.0)
    merged["sentiment_std"]        = merged["sentiment_std"].fillna(0.0)
    merged["sentiment_confidence"] = merged["sentiment_confidence"].fillna(0.5)
    merged["headline_count"]       = merged["headline_count"].fillna(0).astype(int)

    out = DATA_DIR / "merged_features.parquet"
    merged.to_parquet(out, index=False)
    log.info(f"  ✓ Merged dataset: {len(merged):,} rows → {out}")
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_nlp_pipeline() -> pd.DataFrame:
    log.info("═══ PHASE 1b ─ NLP Sentiment Pipeline ═══")
    headlines  = generate_headlines(TICKERS, START_DATE, END_DATE)
    log.info(f"  Generated {len(headlines):,} headlines")
    sentiments = get_sentiments(headlines)
    merged     = merge_sentiment_with_stocks(sentiments)
    log.info("═══ Phase 1b complete ✓ ═══\n")
    return merged


if __name__ == "__main__":
    run_nlp_pipeline()
