"""
cdf_backends package
====================

Pluggable cumulative-distribution-function estimators supporting the Multi-Null JSd test.

Each back-end is a subclass of `multinull_jsd.cdf_backends.base.CDFBackend` and is automatically selected via the
``cdf_method`` argument in ``multinull_jsd.core.MultiNullJSDTest``.
"""
from .mc_multinomial import MultinomialMCCDFBackend
from .mc_normal import NormalMCCDFBackend
from .exact import ExactCDFBackend
from .base import CDFBackend

from typing import Callable, Optional, Final

#: Factory for CDF backends.
CDF_BACKEND_FACTORY: Final[dict[str, Callable[[int, Optional[int], Optional[int]], CDFBackend]]] = {
    "exact": lambda n, _m, _s: ExactCDFBackend(evidence_size=n),
    "mc_multinomial": lambda n, m, s: MultinomialMCCDFBackend(evidence_size=n, mc_samples=m, seed=s),
    "mc_normal": lambda n, m, s: NormalMCCDFBackend(evidence_size=n, mc_samples=m, seed=s)
}

#: Names of CDF backends that use Monte-Carlo sampling.
MC_CDF_BACKENDS: Final[tuple[str, ...]] = ("mc_multinomial", "mc_normal")

__all__ = [
    "CDF_BACKEND_FACTORY", "MC_CDF_BACKENDS", "CDFBackend", "ExactCDFBackend", "NormalMCCDFBackend",
    "MultinomialMCCDFBackend"
]
