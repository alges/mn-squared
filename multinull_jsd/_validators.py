"""
**Internal validation helpers** used across the *multinull-jsd* code-base. The module contains **only light-weight,
side-effect-free checks** so that importing it never triggers heavy numerical work (NumPy is imported lazily and only
for datatype inspection).
"""
from typing import Any, Optional

FLOAT_TOL: float = 1e-12

def validate_int_value(name: str, value: Any, min_value: Optional[int] = None, max_value: Optional[int] = None) -> int:
    """
    Check that the given value is an integer within the defined bounds (inclusive).

    Parameters
    ----------
    name
        Human-readable name of the parameter – used verbatim in the error message to ease debugging.
    value
        Object to validate. Usually the raw argument received by a public API.
    min_value
        Optional lower bound (inclusive). If not provided, no lower bound is enforced.
    max_value
        Optional upper bound (inclusive). If not provided, no upper bound is enforced.

    Raises
    ------
    TypeError
        If *value* is not an ``int``.
    ValueError
        If the integer is not strictly positive.
    """
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer. Got {type(value).__name__}.")
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be at least {min_value}. Got {value!r}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be at most {max_value}. Got {value!r}.")
    return value
