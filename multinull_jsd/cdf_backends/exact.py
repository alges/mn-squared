from .base import CDFBackend

from multinull_jsd.types import FloatArray, CDFCallable


class ExactCDFBackend(CDFBackend):
    """
    Exhaustively enumerates all histograms in the non-normalized histogram space :math:`\\Delta'_{k,n}` to obtain the
    exact distribution of JSd.

    Complexity
    ----------
    :math:`O(n^{k-1})` for fixed :math:`k` (stars-and-bars enumeration) or :math:`O(k^n)` for fixed :math:`n`.

    Notes
    -----
    Enumeration is cached **per probability vector** so repeated calls with the same vector avoid re-computation.
    """
    def __init__(self, evidence_size: int):
        super().__init__(evidence_size)
        raise NotImplementedError

    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        raise NotImplementedError
