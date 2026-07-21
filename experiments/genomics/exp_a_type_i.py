"""
Experiment A: Type-I error control with real-world null distributions.

Uses the genome-wide codon-usage profiles of five organisms (see `cutg_data.py`) as null
distributions, each treated as a single-null Type-I control problem: for a range of sample sizes n,
histograms are drawn under each organism's own null and the empirical Type-I error (rate of
rejecting all candidates) is measured for MNSquared and three Holm-corrected baselines.

Usage
-----
  python -m experiments.genomics.exp_a_type_i

Expected runtime: approximately 10-20 minutes on a modern laptop.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from experiments.genomics.cutg_data import ORGANISMS
from experiments.genomics.mn2_batch import mn2_decide_batch
from experiments.baselines.multiple_testing import multinull_decisions_holm_batch
from experiments.registry import BASELINE_SINGLE
from experiments.io_utils import RESULTS_DIR
from experiments.settings import IntArray, FloatArray


ALPHA: float = 0.05
N_REPS: int = 2_000
N_MC: int = 5_000
SEED: int = 42
N_GRID: IntArray = np.unique(
    np.round(np.logspace(np.log10(50), np.log10(3_000), 30)).astype(int)
)


def run_experiment_a() -> pd.DataFrame:
    """
    Run the Type-I error simulation for all five organisms and return a results DataFrame.

    Returns
    -------
    DataFrame with columns: organism, n, method, type_i_error.
    """
    rng = np.random.default_rng(seed=SEED)
    rows: list[dict] = []

    for org_idx, org in enumerate(ORGANISMS):
        p: FloatArray = org["probs"]
        null_probs = p[np.newaxis, :]  # single null for a pure Type-I evaluation
        print(f"\nOrganism {org_idx + 1}/{len(ORGANISMS)}: {org['name']} ({len(N_GRID)} sample sizes)")

        for n_idx, n in enumerate(N_GRID):
            t0 = time.time()
            histograms: IntArray = rng.multinomial(int(n), p, size=N_REPS)

            dec_mn2 = mn2_decide_batch(histograms, null_probs, alpha=ALPHA, n_mc=N_MC, seed=SEED)
            type1: dict[str, float] = {"MNSquared": float(np.mean(dec_mn2 == -1))}

            for method_name, pval_fn in BASELINE_SINGLE.items():
                if method_name.startswith("MMD-Gaussian+") or method_name.startswith("MMD-Laplacian+"):
                    continue  # permutation-based kernel baselines are not used in this experiment
                dec = multinull_decisions_holm_batch(
                    histograms=histograms,
                    null_probabilities=null_probs,
                    alpha_global=ALPHA,
                    single_null_pvalue_fn=pval_fn,
                    show_progress=False,
                )
                type1[method_name] = float(np.mean(dec == -1))

            elapsed = time.time() - t0
            print(f"  n={n:4d} ({n_idx + 1:2d}/{len(N_GRID)})  " + "  ".join(
                f"{name}={value:.3f}" for name, value in type1.items()
            ) + f"  [{elapsed:.1f}s]")

            for method_name, value in type1.items():
                rows.append({"organism": org["name"], "n": int(n), "method": method_name, "type_i_error": value})

    return pd.DataFrame(data=rows)


if __name__ == "__main__":
    print("=" * 60)
    print("Genomics Experiment A: Type-I error control")
    print(f"Organisms : {len(ORGANISMS)}")
    print("k         : 64 codons")
    print(f"n grid    : {N_GRID[0]}-{N_GRID[-1]} ({len(N_GRID)} points)")
    print(f"N_reps    : {N_REPS}")
    print(f"N_MC      : {N_MC}")
    print(f"alpha     : {ALPHA}")
    print("=" * 60)

    t_start = time.time()
    df_results = run_experiment_a()
    df_results.to_csv(RESULTS_DIR / "genomics_a_type_i.csv", index=False)

    print(f"\nTotal runtime: {(time.time() - t_start) / 60:.1f} minutes")
    print(f"Results written to {RESULTS_DIR / 'genomics_a_type_i.csv'}")
