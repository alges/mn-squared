"""
Experiment B: Type-I error control across alphabet sizes.

Using the same five codon-usage null distributions as Experiment A (see `cutg_data.py`), adjacent
codons are merged into k equipartitioned groups for k < 64, and each codon is split uniformly into
k / 64 sub-categories for k > 64. This spans the range from the dense regime (k = 8) to the extreme
sparse regime (k = 512) at a fixed sample size n, covering the range of alphabet granularities
relevant to codon-usage analysis.

Usage
-----
  python -m experiments.genomics.exp_b_alphabet

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
from experiments.settings import FloatArray


ALPHA: float = 0.05
SEED: int = 42
K_TARGETS: list[int] = [8, 16, 32, 64, 128, 256, 512]
N: int = 300
N_REPS: int = 1_000
N_MC: int = 5_000


def _coarsen_probs(p64: FloatArray, k_target: int) -> FloatArray:
    """Merge adjacent codons to produce a distribution over `k_target` bins."""
    assert k_target <= 64 and 64 % k_target == 0, f"k_target={k_target} must divide 64 evenly"
    group = 64 // k_target
    return p64.reshape(k_target, group).sum(axis=1)


def _expand_probs(p64: FloatArray, k_target: int) -> FloatArray:
    """Split each codon into (k_target // 64) equal sub-categories."""
    assert k_target > 64 and k_target % 64 == 0, f"k_target={k_target} must be a multiple of 64"
    sub = k_target // 64
    return np.repeat(p64, sub) / sub


def _make_probs_k(p64: FloatArray, k_target: int) -> FloatArray:
    """Return a probability vector of length `k_target` derived from a 64-codon base profile."""
    p = _coarsen_probs(p64, k_target) if k_target <= 64 else _expand_probs(p64, k_target)
    return p / p.sum()


def run_experiment_b() -> pd.DataFrame:
    """
    Run the Type-I error simulation across alphabet sizes for all five organisms.

    Returns
    -------
    DataFrame with columns: organism, k, method, type_i_error.
    """
    rng = np.random.default_rng(seed=SEED)
    rows: list[dict] = []

    for org_idx, org in enumerate(ORGANISMS):
        p64: FloatArray = org["probs"]
        print(f"\nOrganism {org_idx + 1}/{len(ORGANISMS)}: {org['name']} (n={N}, {len(K_TARGETS)} alphabet sizes)")

        for k_target in K_TARGETS:
            t0 = time.time()
            p_k = _make_probs_k(p64, k_target)
            null_probs = p_k[np.newaxis, :]

            histograms = rng.multinomial(N, p_k, size=N_REPS)

            dec_mn2 = mn2_decide_batch(histograms, null_probs, alpha=ALPHA, n_mc=N_MC, seed=SEED)
            type1: dict[str, float] = {"MNSquared": float(np.mean(dec_mn2 == -1))}

            for method_name, pval_fn in BASELINE_SINGLE.items():
                if method_name.startswith("MMD-Gaussian+") or method_name.startswith("MMD-Laplacian+"):
                    continue
                dec = multinull_decisions_holm_batch(
                    histograms=histograms,
                    null_probabilities=null_probs,
                    alpha_global=ALPHA,
                    single_null_pvalue_fn=pval_fn,
                    show_progress=False,
                )
                type1[method_name] = float(np.mean(dec == -1))

            elapsed = time.time() - t0
            print(f"  k={k_target:4d}  " + "  ".join(
                f"{name}={value:.3f}" for name, value in type1.items()
            ) + f"  [{elapsed:.1f}s]")

            for method_name, value in type1.items():
                rows.append({"organism": org["name"], "k": k_target, "method": method_name, "type_i_error": value})

    return pd.DataFrame(data=rows)


if __name__ == "__main__":
    print("=" * 60)
    print("Genomics Experiment B: Type-I error across alphabet sizes")
    print(f"Organisms : {len(ORGANISMS)}")
    print(f"n         : {N}")
    print(f"k values  : {K_TARGETS}")
    print(f"N_reps    : {N_REPS}")
    print(f"N_MC      : {N_MC}")
    print(f"alpha     : {ALPHA}")
    print("=" * 60)

    t_start = time.time()
    df_results = run_experiment_b()
    df_results.to_csv(RESULTS_DIR / "genomics_b_alphabet.csv", index=False)

    print(f"\nTotal runtime: {(time.time() - t_start) / 60:.1f} minutes")
    print(f"Results written to {RESULTS_DIR / 'genomics_b_alphabet.csv'}")
