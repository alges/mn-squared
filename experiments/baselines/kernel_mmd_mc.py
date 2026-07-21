"""
Monte Carlo-calibrated Gaussian-kernel one-sample test for a multinomial null.

Unlike `kernel_mmd.py`, which reduces the one-sample problem to a two-sample permutation test via `hyppo`, this module
calibrates the null distribution of a Gaussian-kernel statistic directly against `Multinomial(n, p_null)`, avoiding the
permutation/two-sample reduction entirely. This mirrors settings in which no closed-form or off-the-shelf calibration
is available for a one-sample multinomial kernel test, so calibration is obtained by simulation from the null itself.

For a null probability vector p and a histogram h with n = sum(h), the statistic is

    d(h) = 2 * (1 - exp(-||h/n - p||^2 / (2 * sigma^2)))

where sigma^2 is set via the median heuristic on samples drawn from p itself, and the null distribution of d is
approximated by Monte Carlo draws from Multinomial(n, p). The p-value is the right-tail probability under this
simulated null.

The single-null p-value function has the plain `(histogram, p_null) -> float` signature expected by `BASELINE_SINGLE`
in `registry.py`, with no shared state across calls. This is deliberate: batched evaluation (see
`baselines/multiple_testing.py`) may run under a process pool, where module-level state configured in the parent
process would not be visible to worker processes. Per-(n, p_null) calibration is cached within each process via
`functools.lru_cache`, so repeated calls for the same size and null within a single worker still avoid redundant Monte
Carlo simulation.
"""
from __future__ import annotations

import functools

import numpy as np

from experiments.settings import FloatArray, IntArray


_N_MC: int = 3_000
_SEED: int = 0
_N_BANDWIDTH_SAMPLES: int = 30


@functools.lru_cache(maxsize=256)
def _calibrate(n: int, p_null_key: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
    """
    Estimate the kernel bandwidth and the sorted null statistics for a given (n, p_null).

    Cached per-process: `p_null_key` is a hashable tuple representation of the null probability vector, so repeated
    calls with the same size and null reuse the Monte Carlo calibration instead of resampling.

    Parameters
    ----------
    n:
        Histogram size used to draw calibration samples.
    p_null_key:
        Null probability vector, as a hashable tuple.

    Returns
    -------
    Tuple of (sigma2, sorted_null_stats), where `sorted_null_stats` is itself a tuple for hashability/caching.
    """
    p_null = np.asarray(a=p_null_key, dtype=np.float64)
    rng = np.random.default_rng(seed=_SEED)

    # Bandwidth via the median heuristic on samples drawn from the null itself.
    bandwidth_samples = rng.multinomial(n, p_null, size=_N_BANDWIDTH_SAMPLES) / n
    diff = bandwidth_samples[:, np.newaxis, :] - bandwidth_samples[np.newaxis, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    i_upper = np.triu_indices(n=bandwidth_samples.shape[0], k=1)
    sigma2 = max(float(np.median(sq_dist[i_upper])), 1e-10)

    # Monte Carlo null distribution of the statistic.
    mc_samples = rng.multinomial(n, p_null, size=_N_MC) / n
    sq = np.sum((mc_samples - p_null) ** 2, axis=-1)
    stats = 2.0 * (1.0 - np.exp(-sq / (2.0 * sigma2)))
    sorted_stats = tuple(np.sort(stats).tolist())

    return sigma2, sorted_stats


def mmd_gaussian_mc_pvalue(histogram: IntArray, p_null: FloatArray) -> float:
    """
    Monte Carlo-calibrated Gaussian-kernel one-sample p-value against a multinomial null.

    Parameters
    ----------
    histogram:
        Observed counts (k,).
    p_null:
        Null probability vector p_ell (k,).

    Returns
    -------
    p_value:
        Right-tail Monte Carlo p-value for H0: "multinomial with probabilities p_null".
    """
    h: IntArray = np.asarray(a=histogram, dtype=np.int64)
    p: FloatArray = np.asarray(a=p_null, dtype=np.float64)
    n = int(h.sum())

    p_key = tuple(np.round(p, decimals=12).tolist())
    sigma2, sorted_null_stats = _calibrate(n=n, p_null_key=p_key)
    sorted_null_stats_arr = np.asarray(sorted_null_stats, dtype=np.float64)

    h_norm = h / n
    sq_obs = float(np.sum((h_norm - p) ** 2))
    stat_obs = 2.0 * (1.0 - np.exp(-sq_obs / (2.0 * sigma2)))

    rank = int(np.searchsorted(sorted_null_stats_arr, stat_obs))
    return float((_N_MC - rank) / _N_MC)
