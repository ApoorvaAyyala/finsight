"""
FinSight — Master Pipeline Orchestrator
========================================
Runs all four phases in sequence with checkpointing.
Each phase writes its artefacts to data/ so subsequent
phases can skip re-computation if already done.

Usage:
    python run_all.py            # full pipeline
    python run_all.py --phase 1  # phase 1 only (ingest + NLP)
    python run_all.py --phase 2  # phase 2 only (LSTM + GA)
    python run_all.py --phase 3  # launch Streamlit dashboard
    python run_all.py --reset    # wipe data/ and rerun everything
"""
import sys
import shutil
import argparse
import logging
import time

from config import DATA_DIR, MODELS_DIR, MOCKS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BANNER = r"""
╔═══════════════════════════════════════════════════════╗
║           F I N S I G H T   P L A T F O R M          ║
║     AI-Powered Portfolio Risk Intelligence System     ║
╚═══════════════════════════════════════════════════════╝
"""


def phase_1():
    """Data Ingestion + NLP Sentiment"""
    print("\n" + "─" * 55)
    print("  PHASE 1  ·  Big Data Analytics + NLP")
    print("─" * 55)

    # 1a — ingest
    if not (DATA_DIR / "stock_features.parquet").exists():
        from ingest import run_ingestion
        run_ingestion()
    else:
        log.info("stock_features.parquet already exists — skipping ingest.")

    # 1b — sentiment
    if not (DATA_DIR / "merged_features.parquet").exists():
        from sentiment import run_nlp_pipeline
        run_nlp_pipeline()
    else:
        log.info("merged_features.parquet already exists — skipping NLP.")


def phase_2():
    """LSTM training + GA optimisation"""
    print("\n" + "─" * 55)
    print("  PHASE 2  ·  Deep Learning + Soft Computing")
    print("─" * 55)

    if not (DATA_DIR / "merged_features.parquet").exists():
        log.error("merged_features.parquet missing — run phase 1 first.")
        sys.exit(1)

    # 2a — LSTM
    if not (DATA_DIR / "volatility_forecasts.csv").exists():
        from model import run_deep_learning
        run_deep_learning()
    else:
        log.info("volatility_forecasts.csv already exists — skipping LSTM training.")

    # 2b — GA
    if not (DATA_DIR / "optimization_results.json").exists():
        from optimize import run_optimization
        run_optimization()
    else:
        log.info("optimization_results.json already exists — skipping GA.")


def phase_3():
    """Launch Streamlit dashboard"""
    print("\n" + "─" * 55)
    print("  PHASE 3  ·  Business Analytics Dashboard")
    print("─" * 55)
    import subprocess
    log.info("Starting Streamlit dashboard on http://localhost:8501 …")
    subprocess.run(["streamlit", "run", "app.py"], check=True)


def reset():
    """Wipe all generated artefacts and start fresh."""
    for d in [DATA_DIR, MODELS_DIR, MOCKS_DIR]:
        if d.exists():
            shutil.rmtree(d)
            d.mkdir()
    log.info("All artefacts cleared — ready for a fresh pipeline run.")


# ─── Entry-point ──────────────────────────────────────────────────────────────

def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="FinSight pipeline runner")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run a specific phase only")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all cached artefacts before running")
    args = parser.parse_args()

    if args.reset:
        reset()

    t0 = time.time()

    if args.phase == 1:
        phase_1()
    elif args.phase == 2:
        phase_2()
    elif args.phase == 3:
        phase_3()
    else:
        phase_1()
        phase_2()
        phase_3()

    elapsed = time.time() - t0
    if args.phase != 3:
        print(f"\n✅  Pipeline finished in {elapsed:.1f}s")
        print("   Run  →  streamlit run app.py  to open the dashboard.\n")


if __name__ == "__main__":
    main()
