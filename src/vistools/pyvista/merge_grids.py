# The MIT License (MIT)
#
# Copyright (c) 2023-2026 Ivo Steinbrecher
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Merge multiple grids into a single grid."""

from typing import Any

import numpy as np
import pyvista as pv


def _union_field_names(
    grid_a: pv.UnstructuredGrid, grid_b: pv.UnstructuredGrid, attribute: str
) -> set[str]:
    """Return the union of array names from two grids.

    Args:
        grid_a: First grid.
        grid_b: Second grid.
        attribute: Data type name ("point_data" or "cell_data").

    Returns:
        Set of array names present in either grid for the data type name.
    """
    data_a = getattr(grid_a, attribute)
    data_b = getattr(grid_b, attribute)
    return set(data_a.keys()) | set(data_b.keys())


def _field_meta(
    grid_a: pv.UnstructuredGrid,
    grid_b: pv.UnstructuredGrid,
    attribute: str,
    name: str,
) -> tuple[np.dtype, tuple[int, ...]]:
    """Determine dtype and trailing shape of a data array.

    The trailing shape is:
        - () for scalar arrays of shape (n,)
        - (k,) for vector-like arrays of shape (n, k)

    Args:
        grid_a: First grid.
        grid_b: Second grid.
        attribute: Data type name ("point_data" or "cell_data").
        name: Array name.

    Returns:
        Tuple of (dtype, trailing shape).

    Raises:
        ValueError: If an array has more than 2 dimensions.
    """
    data_a = getattr(grid_a, attribute)
    data_b = getattr(grid_b, attribute)

    def _check_data(data: Any) -> np.ndarray | None:
        """Check if the data contains the array and return it if so.

        Also perform a check that the array has at most 2 dimensions.
        """
        if name not in data:
            return None

        array = data[name]
        if array.ndim > 2:
            raise ValueError(
                f"Field '{name}' has more than 2 dimensions, which is not supported."
            )
        return array

    array_a = _check_data(data_a)
    array_b = _check_data(data_b)
    if array_a is not None:
        array = array_a
    else:
        array = array_b

    return array.dtype, array.shape[1:] if array.ndim > 1 else ()


def _ensure_field(
    grid: pv.UnstructuredGrid,
    attribute: str,
    name: str,
    dtype: np.dtype,
    tail_shape: tuple[int, ...],
    fill_value: Any,
) -> None:
    """Ensure a field exists on a grid with the expected name and metadata.

    If the field is missing, it is created and filled with the given value.
    If it exists, its dtype and trailing shape are validated.

    Args:
        grid: Grid to modify.
        attribute: Data type name ("point_data" or "cell_data").
        name: Array name.
        dtype: Expected NumPy dtype.
        tail_shape: Expected trailing shape.
        fill_value: Value used to fill missing arrays.

    Raises:
        TypeError: If an existing array has a mismatched dtype.
        ValueError: If an existing array has a mismatched shape.
    """
    data = getattr(grid, attribute)
    n = grid.n_points if attribute == "point_data" else grid.n_cells

    if name in data:
        array = data[name]
        array_tail_shape = array.shape[1:] if array.ndim > 1 else ()
        if array.dtype != dtype:
            raise TypeError(
                f"Field '{name}' has dtype {array.dtype}, expected {dtype}."
            )
        if not array_tail_shape == tail_shape:
            raise ValueError(
                f"Field '{name}' has tail shape {array_tail_shape}, expected {tail_shape}."
            )
        return

    arr = np.full((n, *tail_shape), fill_value, dtype=dtype)
    data.set_array(arr, name)


def _patch_data_arrays(
    grid_a: pv.UnstructuredGrid,
    grid_b: pv.UnstructuredGrid,
    point_fill: dict[str, Any] | None = None,
    cell_fill: dict[str, Any] | None = None,
) -> None:
    """Ensure both grids share the same point and cell data schema.

    Missing arrays are created on each grid using provided fill values.

    Args:
        grid_a: First grid (modified in place).
        grid_b: Second grid (modified in place).
        point_fill: Mapping from point-data field names to fill values.
        cell_fill: Mapping from cell-data field names to fill values.
    """
    for attribute, fill_map in [
        ("point_data", point_fill or {}),
        ("cell_data", cell_fill or {}),
    ]:
        for name in _union_field_names(grid_a, grid_b, attribute):
            dtype, tail_shape = _field_meta(grid_a, grid_b, attribute, name)

            default_fill_value = 0 if tail_shape == () else [0] * tail_shape[0]
            fill_value = fill_map.get(name, default_fill_value)

            _ensure_field(grid_a, attribute, name, dtype, tail_shape, fill_value)
            _ensure_field(grid_b, attribute, name, dtype, tail_shape, fill_value)


def merge_grids(
    grid_a: pv.UnstructuredGrid,
    grid_b: pv.UnstructuredGrid,
    *,
    merge_inplace: bool = False,
    modify_grid_b_inplace: bool = False,
    point_fill: dict[str, Any] | None = None,
    cell_fill: dict[str, Any] | None = None,
) -> pv.UnstructuredGrid:
    """Merge two unstructured grids while preserving all points.

    Before merging, point and cell data arrays are aligned so both grids
    share the same set of fields. Missing fields are created and filled.

    Args:
        grid_a: First grid.
        grid_b: Second grid.
        merge_inplace: If True, modify grid_a in place during merge.
        modify_grid_b_inplace: If True, allow modification of grid_b when
            patching missing arrays. Otherwise, grid_b is copied.
        point_fill: Mapping from point-data field names to fill values.
        cell_fill: Mapping from cell-data field names to fill values.

    Returns:
        Merged unstructured grid.

    Raises:
        TypeError: If field dtypes do not match.
        ValueError: If field shapes are incompatible.
    """
    if not merge_inplace:
        grid_a = grid_a.copy(deep=True)
    if not modify_grid_b_inplace:
        grid_b = grid_b.copy(deep=True)

    _patch_data_arrays(grid_a, grid_b, point_fill=point_fill, cell_fill=cell_fill)

    return grid_a.merge(grid_b, merge_points=False, inplace=merge_inplace)
