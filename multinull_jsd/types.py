"""
Utility type aliases and protocols used across *multinull_jsd*.

The file purposefully contains **no runtime logic** so it can be imported without triggering heavy scientific routines
during static-typing or documentation builds.
"""
from typing import Literal, Protocol, runtime_checkable, overload

import numpy.typing as npt
import numpy as np


#: Names understood by :class:`multinull_jsd.cdf_backends.base.CDFBackend`.
CDFBackendName = Literal["exact", "mc_multinomial", "mc_normal"]

#: Alias for a NumPy array of ``float64`` with *any* shape.
FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class CDFCallable(Protocol):
    """
    Signature of a cumulative-distribution-function returned by backends.

    The callable must be *vectorised*: when passed a scalar, it returns a Python ``float``; when passed an
    ``array_like`` object, it must broadcast and return a ``FloatArray`` of the same shape as the input.

    Notes
    -----
    * Implementors **must** guarantee monotonicity and clipping to ``[0,1]``  because decision logic relies on those
      properties.

    * The protocol can be checked in runtime, so you can use ``isinstance(obj,CDFCallable)``.
    """
    @overload
    def __call__(self, tau: float) -> float: ...
    @overload
    def __call__(self, tau: npt.ArrayLike) -> FloatArray: ...
    def __call__(self, tau: float | npt.ArrayLike) -> float | FloatArray: ...
