"""
FinSight — Phase 2b: Soft Computing — Genetic Algorithm Portfolio Optimiser
===========================================================================
Fitness function:  Sharpe Ratio  = (Rp − Rf) / σp

  Rp  = expected portfolio return  (dot product of weights × annualised returns)
  Rf  = risk-free rate (config.RISK_FREE_RATE)
  σp  = portfolio volatility (LSTM-predicted; from volatility_forecasts.csv)

Outputs:
  data/optimization_results.json  — optimal allocation + evolution log
"""
import json
import logging
import random
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

from config import (
    DATA_DIR,
    POPULATION_SIZE, GENERATIONS, CROSSOVER_PROB, MUTATION_PROB,
    RISK_FREE_RATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_portfolio_inputs() -> pd.DataFrame:
    """Merge LSTM volatility forecasts with historical annualised returns."""
    forecasts = pd.read_csv(DATA_DIR / "volatility_forecasts.csv")
    features  = pd.read_parquet(DATA_DIR / "merged_features.parquet")

    ann_returns = (
        features.groupby("ticker")
        .apply(lambda g: g.sort_values("date")["daily_return"].dropna().mean() * 252)
        .reset_index()
    )
    ann_returns.columns = ["ticker", "annualised_return"]

    portfolio = forecasts.merge(ann_returns, on="ticker")
    portfolio["annualised_return"] = portfolio["annualised_return"].clip(-0.50, 1.50)

    # Floor tiny volatility values to avoid division-by-zero
    portfolio["predicted_volatility"] = portfolio["predicted_volatility"].clip(lower=0.01)

    log.info("Portfolio inputs:")
    log.info(portfolio.to_string(index=False))
    return portfolio


# ──────────────────────────────────────────────────────────────────────────────
# Fitness function
# ──────────────────────────────────────────────────────────────────────────────

def sharpe_fitness(weights_raw, returns: np.ndarray, volatilities: np.ndarray):
    """
    Evaluate one individual.  Weights are soft-normalised (L1) so the GA
    explores the full positive simplex freely.
    """
    w = np.abs(np.array(weights_raw))
    w /= (w.sum() + 1e-9)

    rp = float(np.dot(w, returns))
    sp = float(np.sqrt(np.dot(w ** 2, volatilities ** 2)))

    sharpe = (rp - RISK_FREE_RATE) / (sp + 1e-9)
    return (sharpe,)


# ──────────────────────────────────────────────────────────────────────────────
# DEAP setup
# ──────────────────────────────────────────────────────────────────────────────

def build_toolbox(n_assets: int, returns: np.ndarray, volatilities: np.ndarray):
    # Purge stale class definitions across repeated runs (e.g. Jupyter)
    for name in ("FitnessMax", "Individual"):
        if hasattr(creator, name):
            delattr(creator, name)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    tb = base.Toolbox()
    tb.register("gene",       random.uniform, 0.01, 1.0)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.gene, n=n_assets)
    tb.register("population", tools.initRepeat, list, tb.individual)

    tb.register("evaluate", sharpe_fitness, returns=returns, volatilities=volatilities)
    tb.register("mate",     tools.cxBlend,         alpha=0.5)
    tb.register("mutate",   tools.mutGaussian,      mu=0, sigma=0.15, indpb=0.25)
    tb.register("select",   tools.selTournament,    tournsize=5)
    return tb


# ──────────────────────────────────────────────────────────────────────────────
# GA execution
# ──────────────────────────────────────────────────────────────────────────────

def run_ga(portfolio: pd.DataFrame) -> dict:
    tickers     = portfolio["ticker"].tolist()
    returns     = portfolio["annualised_return"].values
    volatilities= portfolio["predicted_volatility"].values
    n           = len(tickers)

    tb = build_toolbox(n, returns, volatilities)

    pop = tb.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max",  np.max)
    stats.register("mean", np.mean)
    stats.register("std",  np.std)

    pop, logbook = algorithms.eaSimple(
        pop, tb,
        cxpb=CROSSOVER_PROB,
        mutpb=MUTATION_PROB,
        ngen=GENERATIONS,
        stats=stats,
        halloffame=hof,
        verbose=False,
    )

    log.info(f"  Evolution complete — {GENERATIONS} generations × {POPULATION_SIZE} individuals")

    # Extract best solution
    best_raw = np.array(hof[0])
    best_w   = np.abs(best_raw) / (np.abs(best_raw).sum() + 1e-9)

    allocation = {t: round(float(w), 6) for t, w in zip(tickers, best_w)}

    rp = float(np.dot(best_w, returns))
    sp = float(np.sqrt(np.dot(best_w ** 2, volatilities ** 2)))
    sr = (rp - RISK_FREE_RATE) / (sp + 1e-9)

    # Collect evolution log
    evo_log = [
        {
            "generation": r["gen"],
            "max_sharpe":  round(float(r["max"]),  4),
            "mean_sharpe": round(float(r["mean"]), 4),
            "std_sharpe":  round(float(r["std"]),  4),
        }
        for r in logbook
    ]

    results = {
        "optimal_allocation": allocation,
        "metrics": {
            "expected_annual_return_pct": round(rp * 100, 2),
            "portfolio_volatility_pct":   round(sp * 100, 2),
            "sharpe_ratio":               round(sr, 4),
            "risk_free_rate_pct":         round(RISK_FREE_RATE * 100, 2),
        },
        "evolution_log": evo_log,
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Entry-point
# ──────────────────────────────────────────────────────────────────────────────

def run_optimization() -> dict:
    log.info("═══ PHASE 2b ─ Genetic Algorithm Portfolio Optimisation ═══")

    portfolio = load_portfolio_inputs()
    results   = run_ga(portfolio)

    out = DATA_DIR / "optimization_results.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n╔══ Optimal Portfolio Allocation ════════════════════════════╗")
    for t, w in sorted(results["optimal_allocation"].items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        print(f"  {t:6s}  {w*100:5.1f}%  {bar}")
    print("╠════════════════════════════════════════════════════════════╣")
    m = results["metrics"]
    print(f"  Expected Annual Return : {m['expected_annual_return_pct']}%")
    print(f"  Portfolio Volatility   : {m['portfolio_volatility_pct']}%")
    print(f"  Sharpe Ratio           : {m['sharpe_ratio']}")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    log.info(f"  ✓ Results saved → {out}")
    log.info("═══ Phase 2b complete ✓ ═══\n")
    return results


if __name__ == "__main__":
    run_optimization()
