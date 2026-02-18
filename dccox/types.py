"""Type aliases for numpy arrays."""

from __future__ import annotations

from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import AfterValidator


def _validate_1d(v: np.ndarray) -> np.ndarray:
    if v.ndim != 1:
        raise ValueError(f"Expected 1D array, got {v.ndim}D")
    return v


def _validate_1d_like(v: ArrayLike1D) -> np.ndarray:
    if isinstance(v, (pd.Series, list)):
        v = np.array(v)
    if v.ndim != 1:
        raise ValueError(f"Expected 1D array, got {v.ndim}D")
    return v


def _validate_2d(v: np.ndarray) -> np.ndarray:
    if v.ndim != 2:
        raise ValueError(f"Expected 2D array, got {v.ndim}D")
    return v


Array1D = Annotated[np.ndarray, AfterValidator(_validate_1d)]
Array2D = Annotated[np.ndarray, AfterValidator(_validate_2d)]

ArrayLike1D = Annotated[
    Array1D | pd.Series | list[float], AfterValidator(_validate_1d_like)
]
ArrayLike2D = Annotated[
    Array2D | pd.DataFrame | list[list[float]] | list[Array1D],
    AfterValidator(_validate_2d),
]
ArrayLike = ArrayLike1D | ArrayLike2D
