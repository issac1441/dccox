"""Block matrix module."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

from dccox.types import Array2D

T = TypeVar("T")


class BlockMatrix:
    """
    A block matrix class for handling matrices with different dimensions.

    Attributes
    ----------
    blocks : list of lists of Array2D
        The list stores the lists of arrays
    axis : int
        The axis of unequal sizes.
        For example, the matrix shapes in a Ms[0]:
        [(3,10), (3,2), (3,4)], the axis=1,
        [(10,3), (2,3), (4,2)], the axis=0.
        This will affect the indexing output.
    """

    def __init__(self, Ms: list[list[Array2D]], axis: int = 1) -> None:
        """
        Initialize a block matrix.

        Parameters
        ----------
        Ms : list of lists of Array2D
            The list stores the lists of arrays
        axis : int
            The axis of unequal sizes.
            For example, the matrix shapes in a Ms[0]:
            [(3,10), (3,2), (3,4)], the axis=1,
            [(10,3), (2,3), (4,2)], the axis=0.
            This will affect the indexing output.
        """
        self.blocks = Ms
        self.__nc = len(self.blocks)
        self.__nd = len(self.blocks[0])
        # Each matrix in the 2nd layer list has different shape in axis 1, the axis=1,
        # similarily, each matrix has different shape in axis 0, the axis=0.
        self.axis = axis

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the block matrix.

        Returns
        -------
            shape : tuple of int
                The shape of the block matrix, (number of clients, number of dimensions).
        """
        return (self.__nc, self.__nd)

    @property
    def csizes(self) -> list[int]:
        """
        The number of samples for each block.

        Returns
        -------
            csizes : list of int
                The number of samples for each block.
        """
        return [mc[0].shape[0] for mc in self.blocks]

    @property
    def dsizes(self) -> list[int]:
        """
        The number of features for each block.

        Returns
        -------
            dsizes : list of int
                The number of features for each block.
        """
        return [m.shape[1] for m in self.blocks[0]]

    @staticmethod
    def subset(ls: list[T], idx: slice | int | list[int] | tuple[int, ...]) -> list[T]:
        """
        Get the subset of the list.

        Parameters
        ----------
        ls : list
            The list to get the subset.
        idx : slice, int or list
            The index to indicate the subset.

        Returns
        -------
        subset : list
            The subset of the list.
        """
        if isinstance(idx, slice):
            return ls[idx]
        elif isinstance(idx, int):
            return [ls[idx]]
        elif isinstance(idx, (list, tuple)):
            return [ls[i] for i in idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx).__name__}")

    def __getitem__(self, indices: slice | int | list | tuple) -> Array2D:
        """
        Get the block matrix.

        Parameters
        ----------
        indices : slice, int or list
            The index to indicate the (cth, dth) block.

        Returns
        -------
        block : Array2D
            The concatenated matrix.

        Examples
        --------
        # The example of axis=1
        >>> x11 = np.array([1] * 4).reshape(2, 2)
        >>> x12 = np.array([2] * 6).reshape(2, 3)
        >>> x21 = np.array([3] * 6).reshape(3, 2)
        >>> x22 = np.array([4] * 9).reshape(3, 3)
        >>> np.block([[x11, x12], [x21, x22]])
        array([[1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [3, 3, 4, 4, 4],
                [3, 3, 4, 4, 4],
                [3, 3, 4, 4, 4]])
        >>> blk = BlockMatrix([[x11, x12], [x21, x22]])
        >>> blk[0, :]
        array([[1, 1, 2, 2, 2],
                [1, 1, 2, 2, 2]])
        >>> blk[:, 0]
        array([[1, 1],
                [1, 1],
                [3, 3],
                [3, 3],
                [3, 3]])

        # The sample of axis=0
        >>> x11 = np.array([1] * 4).reshape(2, 2)
        >>> x12 = np.array([2] * 6).reshape(3, 2)
        >>> x21 = np.array([3] * 8).reshape(2, 4)
        >>> x22 = np.array([4] * 12).reshape(3, 4)
        >>> blk = BlockMatrix([[x11, x12], [x21, x22]])
        >>> blk[0, :]
        array([[1, 1],
                [1, 1],
                [2, 2],
                [2, 2],
                [2, 2]])
        >>> blk[:, 0]
        array([[1, 1, 3, 3, 3, 3],
                [1, 1, 3, 3, 3, 3]])
        """
        M = [self.subset(m, indices[1]) for m in self.subset(self.blocks, indices[0])]

        if self.axis == 0:
            return np.concatenate([np.concatenate(mc) for mc in M], axis=1)
        elif self.axis == 1:
            return np.block(M)
        else:
            raise ValueError(f"Unsupported axis value: {self.axis}. Must be 0 or 1.")
