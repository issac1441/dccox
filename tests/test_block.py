"""Test cases for BlockMatrix."""

from __future__ import annotations

import numpy as np
import pytest

from dccox.block import BlockMatrix


class TestBlockMatrix:
    """Test cases for BlockMatrix class."""

    @pytest.fixture
    def blocks_axis1(self) -> list[list[np.ndarray]]:
        """Create test blocks for axis=1 (different column sizes)."""
        x11 = np.array([1] * 4).reshape(2, 2)
        x12 = np.array([2] * 6).reshape(2, 3)
        x21 = np.array([3] * 6).reshape(3, 2)
        x22 = np.array([4] * 9).reshape(3, 3)
        return [[x11, x12], [x21, x22]]

    @pytest.fixture
    def blocks_axis0(self) -> list[list[np.ndarray]]:
        """Create test blocks for axis=0 (different row sizes)."""
        x11 = np.array([1] * 4).reshape(2, 2)
        x12 = np.array([2] * 6).reshape(3, 2)
        x21 = np.array([3] * 8).reshape(2, 4)
        x22 = np.array([4] * 12).reshape(3, 4)
        return [[x11, x12], [x21, x22]]

    def test_init(self, blocks_axis1: list[list[np.ndarray]]) -> None:
        """Test BlockMatrix initialization."""
        blk = BlockMatrix(blocks_axis1)
        assert blk.blocks == blocks_axis1
        assert blk.axis == 1

    def test_init_with_axis0(self, blocks_axis0: list[list[np.ndarray]]) -> None:
        """Test BlockMatrix initialization with axis=0."""
        blk = BlockMatrix(blocks_axis0, axis=0)
        assert blk.blocks == blocks_axis0
        assert blk.axis == 0

    def test_shape(self, blocks_axis1: list[list[np.ndarray]]) -> None:
        """Test shape property."""
        blk = BlockMatrix(blocks_axis1)
        assert blk.shape == (2, 2)

    def test_shape_single_block(self) -> None:
        """Test shape with single block."""
        x = np.array([[1, 2], [3, 4]])
        blk = BlockMatrix([[x]])
        assert blk.shape == (1, 1)

    def test_csizes(self, blocks_axis1: list[list[np.ndarray]]) -> None:
        """Test csizes property (number of rows per client)."""
        blk = BlockMatrix(blocks_axis1)
        assert blk.csizes == [2, 3]

    def test_dsizes(self, blocks_axis1: list[list[np.ndarray]]) -> None:
        """Test dsizes property (number of columns per dimension)."""
        blk = BlockMatrix(blocks_axis1)
        assert blk.dsizes == [2, 3]

    def test_subset_with_slice(self) -> None:
        """Test subset static method with slice."""
        ls = [1, 2, 3, 4, 5]
        assert BlockMatrix.subset(ls, slice(1, 3)) == [2, 3]
        assert BlockMatrix.subset(ls, slice(None)) == [1, 2, 3, 4, 5]

    def test_subset_with_int(self) -> None:
        """Test subset static method with int."""
        ls = [1, 2, 3, 4, 5]
        assert BlockMatrix.subset(ls, 0) == [1]
        assert BlockMatrix.subset(ls, 2) == [3]

    def test_subset_with_list(self) -> None:
        """Test subset static method with list."""
        ls = [1, 2, 3, 4, 5]
        assert BlockMatrix.subset(ls, [0, 2, 4]) == [1, 3, 5]

    def test_subset_with_tuple(self) -> None:
        """Test subset static method with tuple."""
        ls = [1, 2, 3, 4, 5]
        assert BlockMatrix.subset(ls, (1, 3)) == [2, 4]

    def test_getitem_row_slice_axis1(
        self, blocks_axis1: list[list[np.ndarray]]
    ) -> None:
        """Test __getitem__ with row index and column slice for axis=1."""
        blk = BlockMatrix(blocks_axis1)
        result = blk[0, :]
        expected = np.array([[1, 1, 2, 2, 2], [1, 1, 2, 2, 2]])
        np.testing.assert_array_equal(result, expected)

    def test_getitem_col_slice_axis1(
        self, blocks_axis1: list[list[np.ndarray]]
    ) -> None:
        """Test __getitem__ with row slice and column index for axis=1."""
        blk = BlockMatrix(blocks_axis1)
        result = blk[:, 0]
        expected = np.array([[1, 1], [1, 1], [3, 3], [3, 3], [3, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_getitem_full_slice_axis1(
        self, blocks_axis1: list[list[np.ndarray]]
    ) -> None:
        """Test __getitem__ with full slice for axis=1."""
        blk = BlockMatrix(blocks_axis1)
        result = blk[:, :]
        expected = np.array(
            [
                [1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [3, 3, 4, 4, 4],
                [3, 3, 4, 4, 4],
                [3, 3, 4, 4, 4],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_getitem_single_block_axis1(
        self, blocks_axis1: list[list[np.ndarray]]
    ) -> None:
        """Test __getitem__ to get a single block for axis=1."""
        blk = BlockMatrix(blocks_axis1)
        result = blk[0, 0]
        expected = np.array([[1, 1], [1, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_getitem_row_slice_axis0(
        self, blocks_axis0: list[list[np.ndarray]]
    ) -> None:
        """Test __getitem__ with row index and column slice for axis=0."""
        blk = BlockMatrix(blocks_axis0, axis=0)
        result = blk[0, :]
        expected = np.array([[1, 1], [1, 1], [2, 2], [2, 2], [2, 2]])
        np.testing.assert_array_equal(result, expected)

    def test_getitem_col_slice_axis0(
        self, blocks_axis0: list[list[np.ndarray]]
    ) -> None:
        """Test __getitem__ with row slice and column index for axis=0."""
        blk = BlockMatrix(blocks_axis0, axis=0)
        result = blk[:, 0]
        expected = np.array([[1, 1, 3, 3, 3, 3], [1, 1, 3, 3, 3, 3]])
        np.testing.assert_array_equal(result, expected)

    def test_getitem_with_list_indices(
        self, blocks_axis1: list[list[np.ndarray]]
    ) -> None:
        """Test __getitem__ with list indices."""
        blk = BlockMatrix(blocks_axis1)
        result = blk[[0], [0, 1]]
        expected = np.array([[1, 1, 2, 2, 2], [1, 1, 2, 2, 2]])
        np.testing.assert_array_equal(result, expected)

    def test_getitem_second_row(self, blocks_axis1: list[list[np.ndarray]]) -> None:
        """Test __getitem__ for second row."""
        blk = BlockMatrix(blocks_axis1)
        result = blk[1, :]
        expected = np.array(
            [
                [3, 3, 4, 4, 4],
                [3, 3, 4, 4, 4],
                [3, 3, 4, 4, 4],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_getitem_second_col(self, blocks_axis1: list[list[np.ndarray]]) -> None:
        """Test __getitem__ for second column."""
        blk = BlockMatrix(blocks_axis1)
        result = blk[:, 1]
        expected = np.array(
            [
                [2, 2, 2],
                [2, 2, 2],
                [4, 4, 4],
                [4, 4, 4],
                [4, 4, 4],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_larger_block_matrix(self) -> None:
        """Test with a larger block matrix (3x3)."""
        blocks = [
            [np.ones((2, 2)), np.ones((2, 3)) * 2, np.ones((2, 1)) * 3],
            [np.ones((3, 2)) * 4, np.ones((3, 3)) * 5, np.ones((3, 1)) * 6],
            [np.ones((1, 2)) * 7, np.ones((1, 3)) * 8, np.ones((1, 1)) * 9],
        ]
        blk = BlockMatrix(blocks)

        assert blk.shape == (3, 3)
        assert blk.csizes == [2, 3, 1]
        assert blk.dsizes == [2, 3, 1]

        result = blk[0, :]
        assert result.shape == (2, 6)

        result = blk[:, 0]
        assert result.shape == (6, 2)

    def test_non_square_blocks(self) -> None:
        """Test with non-square blocks."""
        x11 = np.array([[1, 2, 3], [4, 5, 6]])
        x12 = np.array([[7], [8]])
        x21 = np.array([[9, 10, 11]])
        x22 = np.array([[12]])
        blk = BlockMatrix([[x11, x12], [x21, x22]])

        assert blk.shape == (2, 2)
        assert blk.csizes == [2, 1]
        assert blk.dsizes == [3, 1]

        result = blk[:, :]
        expected = np.array(
            [
                [1, 2, 3, 7],
                [4, 5, 6, 8],
                [9, 10, 11, 12],
            ]
        )
        np.testing.assert_array_equal(result, expected)
