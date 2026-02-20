"""Type aliases for numpy arrays."""

from __future__ import annotations

from typing import Annotated

import numpy as np
from pydantic import AfterValidator


def _validate_1d(v: np.ndarray) -> np.ndarray:
    if v.ndim != 1:
        raise ValueError(f"Expected 1D array, got {v.ndim}D")
    return v


def _validate_2d(v: np.ndarray) -> np.ndarray:
    if v.ndim != 2:
        raise ValueError(f"Expected 2D array, got {v.ndim}D")
    return v


Array1D = Annotated[np.ndarray, AfterValidator(_validate_1d)]
Array2D = Annotated[np.ndarray, AfterValidator(_validate_2d)]
