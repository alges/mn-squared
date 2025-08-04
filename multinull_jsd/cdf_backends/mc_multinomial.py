from .base import CDFBackend

from multinull_jsd.types import FloatArray, CDFCallable


class MultinomialMCCDFBackend(CDFBackend):
    """
    Monte-Carlo estimator that draws **multinomial** histograms exactly from the provided probability vector and builds
    an empirical CDF.

    Parameters
    ----------
    evidence_size
        Number of samples :math:`n`.
    mc_samples
        Number of Monte-Carlo repetitions :math:`N`. Must be positive.
    seed
        Random-state seed for reproducibility.

    Notes
    -----
    The estimator satisfies the Strong Law of Large Numbers; hence, it converges to the exact CDF as
    :math:`N\\rightarrow\\infty`.
    """
    def __init__(self, evidence_size: int, mc_samples: int, seed: int):
        super().__init__(evidence_size)
        # TODO: Incorporate Monte-Carlo elements
        raise NotImplementedError

    def get_cdf(self, prob_vector: FloatArray) -> CDFCallable:
        raise NotImplementedError
