# _common.py
"""Common utilities and constants shared across quadtree implementations."""

from __future__ import annotations

from typing import Any

# Type aliases
Bounds = tuple[float, float, float, float]
"""Axis-aligned rectangle as (min_x, min_y, max_x, max_y)."""

Point = tuple[float, float]
"""2D point as (x, y)."""

# Dtype mappings
QUADTREE_DTYPE_TO_NP_DTYPE = {
    "f32": "float32",
    "f64": "float64",
    "i32": "int32",
    "i64": "int64",
}
"""Mapping from quadtree dtype strings to NumPy dtype strings."""


def _is_np_array(x: Any) -> bool:
    """
    Check if x is a NumPy array without importing NumPy.

    This allows dtype checking without forcing NumPy as a hard dependency.

    Args:
        x: Object to check.

    Returns:
        True if x is a NumPy array.
    """
    mod = getattr(x.__class__, "__module__", "")
    return mod.startswith("numpy") and hasattr(x, "ndim") and hasattr(x, "shape")


def validate_bounds(bounds: Any) -> Bounds:
    """
    Validate and normalize bounds to a tuple.

    Args:
        bounds: Bounds as sequence of 4 numbers.

    Returns:
        Validated bounds as tuple.

    Raises:
        ValueError: If bounds are invalid.
    """
    if type(bounds) is not tuple:
        bounds = tuple(bounds)
    if len(bounds) != 4:
        raise ValueError(
            "bounds must be a tuple of four numeric values (x min, y min, x max, y max)"
        )
    return bounds  # type: ignore[return-value]


def validate_np_dtype(geoms: Any, expected_dtype: str) -> None:
    """
    Validate that a NumPy array's dtype matches expected dtype.

    Args:
        geoms: NumPy array to validate.
        expected_dtype: Expected quadtree dtype ('f32', 'f64', 'i32', 'i64').

    Raises:
        TypeError: If dtype doesn't match.
    """
    expected_np_dtype = QUADTREE_DTYPE_TO_NP_DTYPE.get(expected_dtype)
    if geoms.dtype != expected_np_dtype:
        raise TypeError(
            f"NumPy array dtype {geoms.dtype} does not match quadtree dtype {expected_dtype}"
        )
