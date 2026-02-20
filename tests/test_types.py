"""Test cases for type validators."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ValidationError
import pytest

from dccox.types import (
    Array1D,
    Array2D,
    _validate_1d,
    _validate_2d,
)


class TestValidate1D:
    """Test cases for _validate_1d."""

    def test_valid_1d_array(self) -> None:
        """Test that valid 1D array passes."""
        arr = np.array([1, 2, 3])
        result = _validate_1d(arr)
        np.testing.assert_array_equal(result, arr)

    def test_invalid_2d_array(self) -> None:
        """Test that 2D array raises ValueError."""
        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="Expected 1D array, got 2D"):
            _validate_1d(arr)

    def test_invalid_3d_array(self) -> None:
        """Test that 3D array raises ValueError."""
        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="Expected 1D array, got 3D"):
            _validate_1d(arr)

    def test_empty_1d_array(self) -> None:
        """Test that empty 1D array passes."""
        arr = np.array([])
        result = _validate_1d(arr)
        np.testing.assert_array_equal(result, arr)


class TestValidate2D:
    """Test cases for _validate_2d."""

    def test_valid_2d_array(self) -> None:
        """Test that valid 2D array passes."""
        arr = np.array([[1, 2], [3, 4]])
        result = _validate_2d(arr)
        np.testing.assert_array_equal(result, arr)

    def test_invalid_1d_array(self) -> None:
        """Test that 1D array raises ValueError."""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="Expected 2D array, got 1D"):
            _validate_2d(arr)

    def test_invalid_3d_array(self) -> None:
        """Test that 3D array raises ValueError."""
        arr = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="Expected 2D array, got 3D"):
            _validate_2d(arr)

    def test_empty_2d_array(self) -> None:
        """Test that empty 2D array passes."""
        arr = np.array([]).reshape(0, 2)
        result = _validate_2d(arr)
        np.testing.assert_array_equal(result, arr)

    def test_single_row_2d_array(self) -> None:
        """Test that single row 2D array passes."""
        arr = np.array([[1, 2, 3]])
        result = _validate_2d(arr)
        np.testing.assert_array_equal(result, arr)

    def test_single_column_2d_array(self) -> None:
        """Test that single column 2D array passes."""
        arr = np.array([[1], [2], [3]])
        result = _validate_2d(arr)
        np.testing.assert_array_equal(result, arr)


class TestPydanticBaseModel:
    """Test cases for Pydantic BaseModel integration."""

    def test_basemodel_with_array1d(self) -> None:
        """Test BaseModel with Array1D field."""

        class Model1D(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            vector: Array1D

        arr = np.array([1.0, 2.0, 3.0])
        model = Model1D(vector=arr)
        np.testing.assert_array_equal(model.vector, arr)

    def test_basemodel_with_array1d_invalid(self) -> None:
        """Test BaseModel with Array1D rejects 2D array."""

        class Model1D(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            vector: Array1D

        arr = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValidationError):
            Model1D(vector=arr)

    def test_basemodel_with_array2d(self) -> None:
        """Test BaseModel with Array2D field."""

        class Model2D(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            matrix: Array2D

        arr = np.array([[1, 2], [3, 4]])
        model = Model2D(matrix=arr)
        np.testing.assert_array_equal(model.matrix, arr)

    def test_basemodel_with_array2d_invalid(self) -> None:
        """Test BaseModel with Array2D rejects 1D array."""

        class Model2D(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            matrix: Array2D

        arr = np.array([1, 2, 3])
        with pytest.raises(ValidationError):
            Model2D(matrix=arr)

    def test_basemodel_with_both_arrays(self) -> None:
        """Test BaseModel with both Array1D and Array2D fields."""

        class MixedModel(BaseModel):
            model_config = {"arbitrary_types_allowed": True}
            vector: Array1D
            matrix: Array2D

        vec = np.array([1.0, 2.0])
        mat = np.array([[1, 2], [3, 4]])
        model = MixedModel(vector=vec, matrix=mat)
        np.testing.assert_array_equal(model.vector, vec)
        np.testing.assert_array_equal(model.matrix, mat)
