"""
Exact CDF backend
=================

Exhaustively enumerates all histograms in the non-normalized histogram space :math:`\\Delta'_{k,n}` to obtain the exact
distribution of JSd.

Complexity
----------
* For fixed :math:`k`: :math:`O(n^{k-1})` (stars-and-bars).
* For fixed :math:`n`: :math:`O(k^n)`.

Notes
-----
* Enumeration should be **cached per probability vector** so repeated calls avoid re-computation.
"""
from .base import CDFBackend

from multinull_jsd._jsd_distance import jsd
from multinull_jsd._validators import validate_probability_vector
from multinull_jsd.types import ScalarFloat, FloatArray, IntArray, FloatDType, IntDType, CDFCallable
from typing import Optional, cast

import numpy.typing as npt
import numpy as np

import itertools
import math


class ExactCDFBackend(CDFBackend):
    """
    Exhaustively enumerates all histograms in the non-normalized histogram space :math:`\\Delta'_{k,n}` to obtain the
    exact distribution of JSd.

    Complexity
    ----------
    :math:`O(n^{k-1})` for fixed :math:`k` (stars-and-bars enumeration) or :math:`O(k^n)` for fixed :math:`n`.

    Parameters
    ----------
    evidence_size
        Number of samples :math:`n`. See ``CDFBackend`` for details.

    Notes
    -----
    Enumeration is cached **per probability vector** so repeated calls with the same vector avoid re-computation.
    """
    def __init__(self, evidence_size: int):
        super().__init__(evidence_size=evidence_size)
        # Cache for histogram enumerations keyed by dimension (:math:`k`)
        self._histogram_cache: dict[int, IntArray] = {}
        # Cache for CDFs keyed by probability vector values (:math:`\\mathbf{p}`)
        self._cdf_cache: dict[tuple[float, ...], CDFCallable] = {}
        # Cache for log-factorial values
        self._lf_cache: Optional[FloatArray] = None

    def _enumerate_histograms(self, k: int) -> IntArray:
        """
        Enumerates all possible histograms for a given dimension.

        This function generates all possible histograms (frequency distributions) for the provided dimension. The
        output is a numpy array where each element represents a unique histogram configuration associated with the
        given dimension.

        Parameters
        ----------
        k
            Dimension for which the histograms are enumerated.

        Returns
        -------
        IntArray
            A 2-D numpy array where each row corresponds to a unique histogram configuration for the specified
            dimension.
        """
        if k in self._histogram_cache:
            return self._histogram_cache[k]

        n: int = self.evidence_size

        if k == 1:
            histogram_array: IntArray = np.array(object=[[n]], dtype=IntDType)
            self._histogram_cache[1] = histogram_array
            return histogram_array

        total_enumeration_positions: int = n + k - 1

        histograms: list[list[int]] = []

        # Iterates over all possible combinations of histogram positions: this follows a "stars-and-bars" enumeration
        for bar_positions in itertools.combinations(iterable=range(total_enumeration_positions), r=k - 1):
            histogram: list[int] = [0] * k
            histogram[0] = bar_positions[0]
            for bar_idx, bar_position in enumerate(bar_positions[1:]):
                histogram[bar_idx + 1] = bar_position - bar_positions[bar_idx] - 1
            histogram[-1] = total_enumeration_positions - bar_positions[-1] - 1
            histograms.append(histogram)

        histogram_array = np.array(object=histograms, dtype=IntDType)
        self._histogram_cache[k] = histogram_array
        return histogram_array


    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        prob_vector = validate_probability_vector(
            name="prob_vector", value=prob_vector, n_categories=None
        ).astype(FloatDType, copy=False)

        cdf_key: tuple[float, ...] = tuple(float(x) for x in prob_vector)
        if cdf_key in self._cdf_cache:
            return self._cdf_cache[cdf_key]

        histogram_array: IntArray = self._enumerate_histograms(k=prob_vector.shape[0])

        # If the probability vector has exact zeros, we discard histograms with non-zero counts on those positions
        p_zero_mask: npt.NDArray[np.bool_] = np.equal(prob_vector, 0.0)
        if np.any(p_zero_mask):
            histogram_array = histogram_array[np.all(histogram_array[:, p_zero_mask] == 0, axis=1)]

        n: int = self.evidence_size
        distances: FloatArray = jsd(p=prob_vector, q=histogram_array.astype(dtype=FloatDType, copy=False) / n)

        if self._lf_cache is None:
            self._lf_cache = np.fromiter(iter=(math.lgamma(h + 1) for h in range(n + 1)), dtype=FloatDType)

        lf_n: float = float(self._lf_cache[n])
        sum_lf_h: FloatArray = self._lf_cache[histogram_array].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sum_h_weighted_log_p: FloatArray = (
                histogram_array * np.where(prob_vector > 0, np.log(prob_vector), 0.0)
            ).sum(axis=1)

        histogram_probs: FloatArray = np.exp(lf_n + sum_h_weighted_log_p - sum_lf_h)
        histogram_probs /= histogram_probs.sum()

        distance_order: IntArray = np.argsort(a=distances)
        distance_values: FloatArray = distances[distance_order]
        cdf_values: FloatArray = np.clip(a=np.cumsum(a=histogram_probs[distance_order]), a_min=0.0, a_max=1.0)

        def cdf(tau: ScalarFloat | FloatArray) -> ScalarFloat | FloatArray:
            """
            Computes the cumulative distribution function (CDF) values for a given input.

            This function calculates the cumulative distribution function (CDF) using the input parameter tau. The
            parameter tau can be either a float or an array of floats. The returned value is an array of floats
            representing the calculated CDF values for the given input.

            Parameters
            ----------
            tau
                A float or an array of floats representing the input values for which the CDF is to be computed.

            Returns
            -------
            The CDF values for the given input.
            """
            if np.isscalar(tau):
                tau_idx: int = int(np.searchsorted(a=distance_values, v=tau, side="right")) - 1
                if tau_idx < 0:
                    return 0.0
                return float(cdf_values[tau_idx])

            tau_array: FloatArray = np.asarray(a=tau, dtype=FloatDType)
            tau_idx_array: IntArray = np.searchsorted(
                a=distance_values, v=tau_array, side="right"
            ).astype(dtype=IntDType) - 1

            output_array: FloatArray = np.zeros_like(a=tau_array, dtype=FloatDType)
            non_negative_tau_idx_mask: npt.NDArray[np.bool_] = tau_idx_array >= 0
            output_array[non_negative_tau_idx_mask] = cdf_values[tau_idx_array[non_negative_tau_idx_mask]]
            return np.clip(a=output_array, a_min=0.0, a_max=1.0)

        cdf_callable: CDFCallable = cast(CDFCallable, cdf)
        self._cdf_cache[cdf_key] = cdf_callable
        return cdf_callable

    def __repr__(self) -> str:
        return f"ExactCDFBackend(evidence_size={self.evidence_size})"
