"""
Experiment D: gene-origin attribution on real coding-sequence (CDS) data.

Tests whether MNSquared can correctly identify the organism of origin for a real gene, given only
its codon histogram, by choosing among the five CUTG null distributions used in Experiments A-C
(see `cutg_data.py`). Unlike Experiments A-C, the test observations here are genuine gene sequences
downloaded from NCBI RefSeq rather than exact multinomial draws, so gene-specific codon bias may
cause individual genes to deviate from the genome-wide null.

Requires
--------
  experiments/genomics/data/real_genes.npz
  (produced by running `download_real_genes.py` first)

Usage
-----
  python -m experiments.genomics.exp_d_real_cds

Expected runtime: 10-30 minutes, depending on the number of distinct gene lengths sampled.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.genomics.cutg_data import ORGANISMS, ORGANISM_KEYS, NULL_PROBABILITIES
from experiments.genomics.mn2_batch import mn2_decide_batch
from experiments.baselines.multiple_testing import multinull_decisions_holm_batch
from experiments.registry import BASELINE_SINGLE
from experiments.io_utils import RESULTS_DIR
from experiments.settings import IntArray


N_GENES: int = 500   # genes sampled per organism
N_MC: int = 1_000    # Monte Carlo samples for the MNSquared and MMD-paper CDF approximations
ALPHA: float = 0.05  # per-null significance level (uniform)
SEED: int = 0

DATA_PATH: Path = Path(__file__).parent / "data" / "real_genes.npz"

L: int = len(ORGANISMS)


def run_attribution(hists: IntArray, alpha: float) -> dict[str, IntArray]:
    """Apply MNSquared and the Holm-corrected baselines to a batch of real gene histograms."""
    alphas = np.full(L, alpha, dtype=np.float64)

    print("    Running MNSquared ... ", end="", flush=True)
    t0 = time.time()
    dec_mn2 = mn2_decide_batch(hists, NULL_PROBABILITIES, alpha=alphas, n_mc=N_MC, seed=SEED)
    print(f"{time.time() - t0:.1f}s")

    decisions: dict[str, IntArray] = {"MNSquared": dec_mn2}
    for method_name, pval_fn in BASELINE_SINGLE.items():
        if method_name.startswith("MMD-Gaussian+") or method_name.startswith("MMD-Laplacian+"):
            continue
        print(f"    Running {method_name} ... ", end="", flush=True)
        t0 = time.time()
        decisions[method_name] = multinull_decisions_holm_batch(
            histograms=hists,
            null_probabilities=NULL_PROBABILITIES,
            alpha_global=alpha,
            single_null_pvalue_fn=pval_fn,
            show_progress=False,
        )
        print(f"{time.time() - t0:.1f}s")

    return decisions


def compute_accuracy(decisions: IntArray, correct_label: int) -> float:
    """Fraction of decisions equal to `correct_label` (1-based)."""
    return float(np.mean(decisions == correct_label))


if __name__ == "__main__":
    if not DATA_PATH.exists():
        print(f"ERROR: data file not found at {DATA_PATH}")
        print("Run `python -m experiments.genomics.download_real_genes` first.")
        raise SystemExit(1)

    print("=" * 60)
    print("Genomics Experiment D: gene-origin attribution on real CDS sequences")
    print(f"Data       : {DATA_PATH}")
    print(f"N_genes    : {N_GENES} per organism")
    print(f"N_MC       : {N_MC}")
    print(f"alpha      : {ALPHA}")
    print("=" * 60)

    data = np.load(str(DATA_PATH))
    rng = np.random.default_rng(seed=SEED)

    methods: list[str] = ["MNSquared"] + [
        name for name in BASELINE_SINGLE
        if not (name.startswith("MMD-Gaussian+") or name.startswith("MMD-Laplacian+"))
    ]
    acc_rows: list[dict] = []
    conf_mat = np.zeros((L, L + 1), dtype=int)  # last column: rejected (-1)

    t_total = time.time()

    for ell, (org, org_key) in enumerate(zip(ORGANISMS, ORGANISM_KEYS)):
        key = f"{org_key}_hists"
        if key not in data:
            print(f"  WARNING: {key} not in data file -- skipping {org['name']}")
            continue

        all_hists = data[key]
        n_sample = min(N_GENES, len(all_hists))
        idx = rng.choice(len(all_hists), n_sample, replace=False)
        hists = all_hists[idx]

        n_codons = hists.sum(axis=1)
        print(
            f"\n[{org['name']}]  {n_sample} genes  "
            f"(n_codons: median={np.median(n_codons):.0f}, range=[{n_codons.min()}, {n_codons.max()}])"
        )

        decisions = run_attribution(hists, ALPHA)
        correct_label = ell + 1  # 1-based

        for method_name in methods:
            acc = compute_accuracy(decisions[method_name], correct_label)
            acc_rows.append({"organism": org["name"], "method": method_name, "accuracy": acc})
            print(f"    {method_name:<24}: accuracy = {acc:.3f}")

        for dec in decisions["MNSquared"]:
            if 1 <= dec <= L:
                conf_mat[ell, dec - 1] += 1
            else:
                conf_mat[ell, L] += 1  # rejected

    print(f"\nAll done in {(time.time() - t_total) / 60:.1f} min")

    df_acc = pd.DataFrame(data=acc_rows)
    df_acc.to_csv(RESULTS_DIR / "genomics_d_accuracy.csv", index=False)

    df_conf = pd.DataFrame(
        data=conf_mat,
        index=[org["name"] for org in ORGANISMS],
        columns=[org["name"] for org in ORGANISMS] + ["Rejected"],
    )
    df_conf.to_csv(RESULTS_DIR / "genomics_d_confusion_matrix.csv")

    print("\n=== Attribution accuracy (mean across organisms) ===")
    print(df_acc.groupby("method")["accuracy"].mean().sort_values(ascending=False))

    print(f"\nResults written to {RESULTS_DIR / 'genomics_d_accuracy.csv'}")
    print(f"Results written to {RESULTS_DIR / 'genomics_d_confusion_matrix.csv'}")
