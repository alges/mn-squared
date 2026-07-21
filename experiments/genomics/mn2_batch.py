"""
Thin wrapper around MNSquaredTest for the genomics experiments.

`mn2_decide_batch` groups histograms by their evidence size n (the row sum), builds one
`MNSquaredTest` per distinct n, and decides each group in turn. This is required for Experiment D,
where each real gene has its own length, but is equally correct (and more direct) when every row
shares a single n, as in Experiments A-C. This is a convenience layer over
`core.make_mn_squared_test` and `core.get_decisions_for_histograms`, which already implement the
underlying construction for the synthetic scenarios.
"""
from __future__ import annotations

import numpy as np

from experiments.core import make_mn_squared_test, get_decisions_for_histograms
from experiments.settings import FloatArray, IntArray


def mn2_decide_batch(
    histograms: IntArray,
    null_probabilities: FloatArray,
    alpha: float | FloatArray = 0.05,
    n_mc: int = 10_000,
    seed: int = 0,
) -> IntArray:
    """
    Return MNSquared decisions for a batch of histograms, grouped internally by evidence size.

    Parameters
    ----------
    histograms:
        Two-dimensional array of shape (m, k) with integer counts. Rows may have different sums;
        histograms are grouped by row sum and each group is decided with its own `MNSquaredTest`.
    null_probabilities:
        Two-dimensional array of shape (L, k) with null probability vectors.
    alpha:
        Scalar or array of shape (L,) with per-null significance levels.
    n_mc:
        Number of Monte Carlo realisations for the CDF approximation.
    seed:
        RNG seed for the Monte Carlo backend.

    Returns
    -------
    One-dimensional array of shape (m,) with decisions in {1, ..., L, -1}.
    """
    hist_arr: IntArray = np.asarray(a=histograms, dtype=np.int64)
    if hist_arr.ndim == 1:
        hist_arr = hist_arr[np.newaxis, :]

    null_p: FloatArray = np.asarray(a=null_probabilities, dtype=np.float64)
    n_nulls: int = null_p.shape[0]
    alpha_vec: FloatArray = np.broadcast_to(np.asarray(a=alpha, dtype=np.float64), (n_nulls,))

    row_sums: IntArray = hist_arr.sum(axis=1)
    decisions: IntArray = np.empty(shape=(hist_arr.shape[0],), dtype=np.int64)

    for n in np.unique(row_sums):
        idx = np.where(row_sums == n)[0]
        test = make_mn_squared_test(
            null_probabilities=null_p,
            alpha_vector=alpha_vec,
            evidence_size=int(n),
            cdf_method="mc_multinomial",
            mc_samples=n_mc,
            seed=seed,
        )
        decisions[idx] = get_decisions_for_histograms(test=test, histograms=hist_arr[idx])

    return decisions
