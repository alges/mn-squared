"""
Experiment C: Multi-null FWER control and attribution accuracy.

C.1 sweeps the number of candidate null distributions L, using the five real codon-usage profiles
(see `cutg_data.py`) as the first five nulls and additional synthetic nulls drawn from
Dirichlet(0.3 * 1_64) for L > 5, at a fixed sample size and alphabet size. This evaluates whether
per-null Type-I control degrades as the candidate set grows.

C.2 fixes L = 5 (the five real organisms) and sweeps the sample size n, recording the rate of
correct attribution, misclassification, and rejection for each method, as a measure of
gene-attribution accuracy in the multi-null setting.

Usage
-----
  python -m experiments.genomics.exp_c_fwer_accuracy

Expected runtime: approximately 20-30 minutes on a modern laptop.
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


# ---------------------------------------------------------------------------
# C.1 -- FWER under many candidate nulls
# ---------------------------------------------------------------------------

C1_L_VALUES: list[int] = [3, 5, 8, 12, 16, 20, 30, 40, 50]
C1_K: int = 64
C1_N: int = 300
C1_ALPHA: float = 0.05
C1_N_REPS: int = 500
C1_N_MC: int = 3_000
C1_SEED: int = 42

_extra_probs_rng = np.random.default_rng(seed=0)
_EXTRA_PROBS: list[FloatArray] = [_extra_probs_rng.dirichlet(0.3 * np.ones(C1_K)) for _ in range(45)]


def _build_null_probs(num_nulls: int) -> FloatArray:
    """Return an (L, 64) array: the first five rows are real organisms, the rest synthetic."""
    n_real = min(num_nulls, 5)
    probs = [ORGANISMS[i]["probs"] for i in range(n_real)]
    if num_nulls > 5:
        probs += [_EXTRA_PROBS[i] for i in range(num_nulls - 5)]
    return np.stack(probs, axis=0)


def run_c1() -> pd.DataFrame:
    """
    Run the FWER-vs-L simulation and return per-L mean empirical per-null Type-I error.

    Returns
    -------
    DataFrame with columns: L, method, mean_type_i_error.
    """
    rng = np.random.default_rng(seed=C1_SEED)
    rows: list[dict] = []

    for num_nulls in C1_L_VALUES:
        t0 = time.time()
        null_probs = _build_null_probs(num_nulls)
        alphas_mn2 = np.full(num_nulls, C1_ALPHA, dtype=np.float64)
        n_real = min(num_nulls, 5)

        errors_by_method: dict[str, list[float]] = {"MNSquared": []}
        for method_name in BASELINE_SINGLE:
            if method_name.startswith("MMD-Gaussian+") or method_name.startswith("MMD-Laplacian+"):
                continue
            errors_by_method[method_name] = []

        for ell in range(n_real):
            p_null = null_probs[ell]
            histograms: IntArray = rng.multinomial(C1_N, p_null, size=C1_N_REPS)
            correct_null = ell + 1  # 1-based

            dec_mn2 = mn2_decide_batch(histograms, null_probs, alpha=alphas_mn2, n_mc=C1_N_MC, seed=C1_SEED)
            errors_by_method["MNSquared"].append(float(np.mean(dec_mn2 != correct_null)))

            for method_name in list(errors_by_method):
                if method_name == "MNSquared":
                    continue
                dec = multinull_decisions_holm_batch(
                    histograms=histograms,
                    null_probabilities=null_probs,
                    alpha_global=C1_ALPHA,
                    single_null_pvalue_fn=BASELINE_SINGLE[method_name],
                    show_progress=False,
                )
                errors_by_method[method_name].append(float(np.mean(dec != correct_null)))

        elapsed = time.time() - t0
        summary = {name: float(np.mean(values)) for name, values in errors_by_method.items()}
        print(f"  C.1 | L={num_nulls:3d}  " + "  ".join(f"{n}={v:.3f}" for n, v in summary.items()) + f"  [{elapsed:.1f}s]")

        for method_name, value in summary.items():
            rows.append({"L": num_nulls, "method": method_name, "mean_type_i_error": value})

    return pd.DataFrame(data=rows)


# ---------------------------------------------------------------------------
# C.2 -- Attribution accuracy as a function of n
# ---------------------------------------------------------------------------

C2_L: int = 5
C2_ALPHA: float = 0.05
C2_N_REPS: int = 1_000
C2_N_MC: int = 3_000
C2_SEED: int = 42
C2_N_GRID: IntArray = np.unique(np.round(np.logspace(np.log10(50), np.log10(3_000), 25)).astype(int))


def run_c2() -> pd.DataFrame:
    """
    Run the attribution-accuracy-vs-n simulation for the five real organisms.

    Returns
    -------
    DataFrame with columns: organism, n, method, outcome, rate, where `outcome` is one of
    "correct", "misclassified", or "rejected" for MNSquared, and "correct" only for the baselines.
    """
    rng = np.random.default_rng(seed=C2_SEED)
    null_probs = np.stack([ORGANISMS[i]["probs"] for i in range(C2_L)], axis=0)
    alphas = np.full(C2_L, C2_ALPHA, dtype=np.float64)

    rows: list[dict] = []

    for org_idx in range(C2_L):
        org = ORGANISMS[org_idx]
        correct_null = org_idx + 1  # 1-based
        print(f"\n  C.2 | Organism {org_idx + 1}/{C2_L}: {org['name']}")

        for n_idx, n in enumerate(C2_N_GRID):
            t0 = time.time()
            histograms: IntArray = rng.multinomial(int(n), org["probs"], size=C2_N_REPS)

            dec_mn2 = mn2_decide_batch(histograms, null_probs, alpha=alphas, n_mc=C2_N_MC, seed=C2_SEED)
            mn2_correct = float(np.mean(dec_mn2 == correct_null))
            mn2_reject = float(np.mean(dec_mn2 == -1))
            mn2_misclass = float(np.mean((dec_mn2 != correct_null) & (dec_mn2 != -1)))

            baseline_correct: dict[str, float] = {}
            for method_name, pval_fn in BASELINE_SINGLE.items():
                if method_name.startswith("MMD-Gaussian+") or method_name.startswith("MMD-Laplacian+"):
                    continue
                dec = multinull_decisions_holm_batch(
                    histograms=histograms,
                    null_probabilities=null_probs,
                    alpha_global=C2_ALPHA,
                    single_null_pvalue_fn=pval_fn,
                    show_progress=False,
                )
                baseline_correct[method_name] = float(np.mean(dec == correct_null))

            elapsed = time.time() - t0
            print(
                f"    n={n:4d} ({n_idx + 1:2d}/{len(C2_N_GRID)})  "
                f"MNSquared_correct={mn2_correct:.3f}  misclass={mn2_misclass:.3f}  reject={mn2_reject:.3f}  "
                + "  ".join(f"{name}_correct={value:.3f}" for name, value in baseline_correct.items())
                + f"  [{elapsed:.1f}s]"
            )

            rows.append({"organism": org["name"], "n": int(n), "method": "MNSquared", "outcome": "correct", "rate": mn2_correct})
            rows.append({"organism": org["name"], "n": int(n), "method": "MNSquared", "outcome": "misclassified", "rate": mn2_misclass})
            rows.append({"organism": org["name"], "n": int(n), "method": "MNSquared", "outcome": "rejected", "rate": mn2_reject})
            for method_name, value in baseline_correct.items():
                rows.append({"organism": org["name"], "n": int(n), "method": method_name, "outcome": "correct", "rate": value})

    return pd.DataFrame(data=rows)


if __name__ == "__main__":
    print("=" * 68)
    print("Genomics Experiment C: FWER scaling and attribution accuracy")
    print("  C.1: mean per-null Type-I error vs number of candidate nulls L")
    print("  C.2: attribution accuracy vs sample size n")
    print(f"Real organisms : {len(ORGANISMS)} ({', '.join(o['name'] for o in ORGANISMS)})")
    print(f"C.1 -- n={C1_N}, alpha={C1_ALPHA}, N_reps={C1_N_REPS}, N_MC={C1_N_MC}, L values={C1_L_VALUES}")
    print(f"C.2 -- L={C2_L}, alpha={C2_ALPHA}, N_reps={C2_N_REPS}, N_MC={C2_N_MC}, "
          f"n grid={C2_N_GRID[0]}-{C2_N_GRID[-1]} ({len(C2_N_GRID)} points)")
    print("=" * 68)

    t_start = time.time()

    print("\nRunning C.1...")
    df_c1 = run_c1()
    df_c1.to_csv(RESULTS_DIR / "genomics_c1_fwer_vs_l.csv", index=False)

    print("\nRunning C.2...")
    df_c2 = run_c2()
    df_c2.to_csv(RESULTS_DIR / "genomics_c2_attribution_vs_n.csv", index=False)

    print(f"\nTotal runtime: {(time.time() - t_start) / 60:.1f} minutes")
    print(f"Results written to {RESULTS_DIR / 'genomics_c1_fwer_vs_l.csv'}")
    print(f"Results written to {RESULTS_DIR / 'genomics_c2_attribution_vs_n.csv'}")
