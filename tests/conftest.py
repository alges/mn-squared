# TODO: Remove the type-ignore comments when mypy understands hypothesis better.
from hypothesis.extra import numpy as hnp
from hypothesis import strategies as st
from typing import Callable

import numpy.typing as npt
import numpy as np

FloatArray = npt.NDArray[np.float64]

FArrayST = st.SearchStrategy[FloatArray]
DrawFArrayFn = Callable[[FArrayST], FloatArray]

@st.composite
def _prob_vector(draw: DrawFArrayFn, k: int = 3) -> FArrayST:
    """
    Hypothesis strategy: generate a random probability vector of dimension k.
    """
    array: FloatArray = draw(
        hnp.arrays(
            dtype=np.float64, shape=(k,),
            elements=st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False)
        )
    )
    if not np.any(a=array):
        array = array.copy()
        array[0] = 1.0
    return array / array.sum()  # type: ignore[return-value]

@st.composite
def _prob_batch(draw: DrawFArrayFn, m: int, k: int = 3) -> FArrayST:
    """
    Hypothesis strategy: generate a batch of m random probability vectors of dimension k.
    """
    return np.stack([draw(_prob_vector(k=k)) for _ in range(m)], axis=0)  # type: ignore[return-value]
