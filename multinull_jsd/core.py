"""
High-level orchestrator for the Multi-Null JSd test.

Typical usage
-------------
>>> from multinull_jsd import MultiNullJSDTest
>>> test = MultiNullJSDTest(evidence_size=100, prob_dim=3, cdf_method="mc_multinomial", mc_samples=10_000, seed=0)
>>> test.add_nulls([0.5, 0.3, 0.2], target_alpha=0.05)  # Add a null hypothesis
>>> test.add_nulls([0.4, 0.4, 0.2], target_alpha=0.01)  # Add another null hypothesis
>>> h = [55, 22, 23]  # Observed histogram to test
>>> p_vals = test.infer_p_values(h)  # Array of p-values for each null hypothesis
>>> decisions = test.infer_decisions(h)  # Array of decisions (1 or 2 for each null hypothesis, -1 for the alternative)
"""
from multinull_jsd.null_structures import IndexedHypotheses
from multinull_jsd.cdf_backends import (
    CDF_BACKEND_FACTORY, MC_CDF_BACKENDS, CDFBackend, ExactCDFBackend, MultinomialMCCDFBackend, NormalMCCDFBackend
)
from multinull_jsd._validators import FLOAT_TOL, validate_int_value
from multinull_jsd.types import FloatArray, FloatDType
from typing import Optional, Sequence, overload

import numpy.typing as npt
import numpy as np


class MultiNullJSDTest:
    """
    Class that orchestrates the Multi-Null JSd test decision rule.

    Parameters
    ----------
    evidence_size
        Number of samples :math:`n` in each histogram.
    prob_dim
        Number of categories :math:`k`.
    cdf_method
        CDF computation backend to use. Available options are ``"exact"``, ``"mc_multinomial"``, and ``"mc_normal"``.
    mc_samples
        Monte-Carlo repetitions :math:`N` (only for MC backends). Ignored for the exact CDF backend.
    seed
        RNG seed for reproducibility of Monte-Carlo backends. Ignored for the exact CDF backend.

    Raises
    ------
    TypeError
        If any of the parameters are of incorrect type.
    ValueError
        If any of the parameters are invalid, such as negative or non-integer values.
    """

    def __init__(
        self, evidence_size: int, prob_dim: int, cdf_method: str = "exact", mc_samples: Optional[int] = None,
        seed: Optional[int] = None
    ) -> None:

        # Parameter validation
        validate_int_value(name="evidence_size", value=evidence_size, min_value=1)
        self._k: int = validate_int_value(name="prob_dim", value=prob_dim, min_value=1)

        if cdf_method not in CDF_BACKEND_FACTORY.keys():
            raise ValueError(f"Invalid CDF method '{cdf_method}'. Must be one of {CDF_BACKEND_FACTORY.keys()}.")

        if cdf_method in MC_CDF_BACKENDS:
            validate_int_value(name="mc_samples", value=mc_samples, min_value=1)
            validate_int_value(name="seed", value=seed, min_value=0)

        # Initialization of container for null hypotheses
        self._nulls: IndexedHypotheses = IndexedHypotheses(
            cdf_backend=CDF_BACKEND_FACTORY[cdf_method](evidence_size, mc_samples, seed)
        )

        raise NotImplementedError

    def add_nulls(self, prob_vector: npt.ArrayLike, target_alpha: float | Sequence[float]) -> None:
        """
        Add one or multiple null hypotheses.

        Rules
        -----
        * If *target_alpha* is a scalar, the same α is applied to all new nulls; if it is a 1-D sequence, it must match
          the number of probability vectors provided.
        * Probability vectors must sum to one. ``prob_vector``  must have shape ``(k,)`` or ``(m, k)``.

        Raises
        ------
        ValueError
            Shape mismatch, invalid probability vector, or invalid target significance level.
        """

        # Validation of the probability vector(s)
        prob_vector = np.asarray(prob_vector, dtype=FloatDType)
        if prob_vector.ndim == 1:
            prob_vector = prob_vector[np.newaxis, :]
        elif prob_vector.ndim != 2:
            raise ValueError("Probability vector must be 1-D or 2-D (batch of histograms).")
        if prob_vector.shape[1] != self._k:
            raise ValueError(f"Probability vector must have {self._k} categories. Got shape {prob_vector.shape}.")
        if np.any(a=prob_vector < 0):
            raise ValueError("Probability values must be non-negative.")
        if not np.all(a=np.isclose(a=np.sum(a=prob_vector, axis=1), b=1.0, atol=FLOAT_TOL)):
            raise ValueError("Probability vectors must sum to one.")

        # Validation of the target alpha(s)
        target_alpha_vec: FloatArray = np.atleast_1d(target_alpha).astype(dtype=FloatDType)
        if target_alpha_vec.ndim != 1:
            raise ValueError("Target alpha must be a scalar or a 1-D sequence.")
        if target_alpha_vec.size == 1:
            target_alpha_vec = np.broadcast_to(array=target_alpha_vec, shape=(prob_vector.shape[0], ))
        elif target_alpha_vec.shape[0] != prob_vector.shape[0]:
            raise ValueError("Target alpha vector and probability vector must have the same length.")

        raise NotImplementedError

    def remove_nulls(self, null_index: int | Sequence[int]) -> None:
        """
        Remove one or multiple null hypotheses.

        Parameters
        ----------
        null_index
            Index or sequence of indices of null hypotheses to remove. Must be valid indices of the current nulls. The
            indexing is one-based, i.e., the first null hypothesis has index 1.
        """
        if isinstance(null_index, int):
            null_index_set: set[int] = {null_index}
        try:
            null_index_set = set(null_index)
        except TypeError:
            raise TypeError("null_index must be an integer or a sequence of integers.")
        for idx in null_index_set:
            validate_int_value(name="null_index value or element", value=idx, min_value=1, max_value=len(self._nulls))
        raise NotImplementedError

    def get_nulls(self) -> IndexedHypotheses:
        """
        Return the current null hypotheses.

        Returns
        -------
        IndexedHypotheses
            Container with the current null hypotheses, providing access by index.
        """
        raise NotImplementedError

    def infer_p_values(self, query: npt.ArrayLike) -> npt.ArrayLike:
        """
        Compute per-null p-values for a histogram or batch of histograms.

        Parameters
        ----------
        query
            Histogram or batch of histograms to test. Must be a 1-D array of shape ``(k,)`` or a 2-D array of shape
            ``(m,k)``, where ``m`` is the number of histograms and ``k`` is the number of categories. The histograms
            must be not normalized, i.e., they need to be raw counts of samples in each category and sum to the
            evidence size.

        Returns
        -------
        npt.ArrayLike
            Array of p-values for each null hypothesis. If the input is a single histogram, the output will have shape
            ``(L,)``, where ``L`` is the number of null hypotheses. Each entry corresponds to the p-value for the
            respective null hypothesis. If the input is a batch, the output will have shape ``(m,L)``.
        """
        raise NotImplementedError

    def infer_decisions(self, query: npt.ArrayLike) -> int | npt.NDArray[int]:
        """
        Apply the decision rule and return an *integer label array* with the same batch shape as *query*:

        * Decision outputs the index ``k`` when the null hypothesis of index ``k`` is selected as the least-rejected
          (accepted).
        * Decision outputs ``-1`` when the alternative hypothesis is chosen (i.e., all nulls are rejected).

        Parameters
        ----------
        query
            Histogram or batch of histograms to test. Must be a 1-D array of shape ``(k,)`` or a 2-D array of shape
            ``(m,k)``, where ``m`` is the number of histograms and ``k`` is the number of categories. The histograms
            must be not normalized, i.e., they need to be raw counts of samples in each category and sum to the
            evidence size.

        Returns
        -------
        int | npt.NDArray[int]
            Array of decisions with the same batch shape as *query*. Each entry corresponds to the decision for the
            respective histogram in the batch. If the input is a single histogram, the output will be a scalar integer.
            If the input is a batch, the output will be a 1-D array of integers.
        """
        raise NotImplementedError

    @overload
    def get_alpha(self, null_index: int) -> float: ...
    @overload
    def get_alpha(self, null_index: Sequence[int]) -> FloatArray: ...

    def get_alpha(self, null_index: int | Sequence[int]) -> float | FloatArray:
        """
        Return the actual significance level (Type-I error probability) for a null hypothesis or a list of hypotheses.

        Parameters
        ----------
        null_index
            Index or sequence of indices of null hypotheses. Must be valid indices of the current nulls. The indexing
            is one-based, i.e., the first null hypothesis has index 1.

        Returns
        -------
        float | Sequence[float]
            The actual significance level for the specified null hypothesis or a list of significance levels for each
            specified null hypothesis. If a single index is provided, a scalar float is returned; if a sequence of
            indices is provided, a 1-D array of floats is returned.
        """
        raise NotImplementedError

    def get_beta(self, query: npt.ArrayLike) -> float | FloatArray:
        """
        Get the maximum Type-II error probability (:math:`\\beta`) over all null hypotheses for a given histogram
        or batch of histograms. This is the probability of failing to reject null hypotheses when the alternative is
        true.

        Parameters
        ----------
        query
            Histogram or batch of histograms to test. Must be a 1-D array of shape ``(k,)`` or a 2-D array of shape
            ``(m,k)``, where ``m`` is the number of histograms and ``k`` is the number of categories.

        Returns
        -------
        float | FloatArray
            Estimated maximum Type-II error probability over all null hypotheses. If the input is a single histogram,
            a scalar float is returned; if the input is a batch, a 1-D array of floats is returned.
        """
        raise NotImplementedError

    def get_fwer(self) -> float:
        """
        Returns the actual Family-Wise Error Rate (FWER) of the Multi-Null JSd test, i.e., the probability of making at
        least one Type-I error when any of the null hypotheses is true.

        Returns
        -------
        float
            The actual FWER of the Multi-Null JSd test.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
