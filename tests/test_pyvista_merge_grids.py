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
"""Test the functionality of merge_grids."""

import numpy as np
import pytest
import pyvista as pv

from vistools.pyvista.merge_grids import (
    _ensure_field,
    _field_meta,
    _patch_data_arrays,
    _union_field_names,
    merge_grids,
)


def create_line_grid(offset: float = 0.0) -> pv.UnstructuredGrid:
    """Create a minimal 2-point, 1-cell unstructured line grid."""
    points = np.array(
        [
            [0.0 + offset, 0.0, 0.0],
            [1.0 + offset, 0.0, 0.0],
        ]
    )
    cells = np.array([2, 0, 1])
    celltypes = np.array([pv.CellType.LINE], dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes, points)


def test_pyvista_merge_grids_union_field_names():
    """Test that the union of field names is correctly computed."""
    g1 = create_line_grid()
    g2 = create_line_grid()

    g1.point_data["a"] = np.array([1.0, 2.0])
    g1.point_data["b"] = np.array([2.0, 3.0])
    g2.point_data["b"] = np.array([3.0, 4.0])
    g2.point_data["c"] = np.array([4.0, 5.0])

    names = _union_field_names(g1, g2, "point_data")
    assert names == {"a", "b", "c"}


@pytest.mark.parametrize(
    "data_value,tail_shape_ref", (([[1.0, 0.0], [0.0, 1.0]], (2,)), ([1.0, 0.0], ()))
)
def test_pyvista_merge_grids_field_meta(data_value, tail_shape_ref):
    """Test that the field meta is correctly computed."""

    # Only the first grid has the data
    g1 = create_line_grid()
    g2 = create_line_grid()
    g1.point_data["data_name"] = np.array(data_value)
    dtype, tail_shape = _field_meta(g1, g2, "point_data", "data_name")
    assert dtype == np.float64
    assert tail_shape == tail_shape_ref

    # Only the second grid has the data
    g1 = create_line_grid()
    g2 = create_line_grid()
    g2.point_data["data_name"] = np.array(data_value)
    dtype, tail_shape = _field_meta(g1, g2, "point_data", "data_name")
    assert dtype == np.float64
    assert tail_shape == tail_shape_ref

    # Both grids have the data
    g1 = create_line_grid()
    g2 = create_line_grid()
    g1.point_data["data_name"] = np.array(data_value)
    g2.point_data["data_name"] = np.array(data_value)
    dtype, tail_shape = _field_meta(g1, g2, "point_data", "data_name")
    assert dtype == np.dtype(float)
    assert tail_shape == tail_shape_ref


def test_pyvista_merge_grids_ensure_field(assert_results_close):
    """Test that _ensure_field correctly adds missing fields and validates
    existing ones."""

    # Check that a missing scalar field is added with the correct shape and value
    g = create_line_grid()
    _ensure_field(
        g,
        "point_data",
        "temperature",
        np.dtype(float),
        (),
        2.1,
    )
    assert "temperature" in g.point_data
    assert g.point_data["temperature"].shape == (g.n_points,)
    assert_results_close(g.point_data["temperature"], [2.1, 2.1])

    # Check that a missing vector field is added with the correct shape and value
    g = create_line_grid()
    _ensure_field(g, "point_data", "velocity", np.dtype(float), (3,), [1.2, 2.3, 3.4])
    assert "velocity" in g.point_data
    assert g.point_data["velocity"].shape == (g.n_points, 3)
    assert_results_close(g.point_data["velocity"], [[1.2, 2.3, 3.4], [1.2, 2.3, 3.4]])

    # Check that an existing field with the correct dtype and shape is left unchanged
    g = create_line_grid()
    g.point_data["temperature"] = np.array([1.0, 2.0], dtype=float)
    _ensure_field(g, "point_data", "temperature", np.dtype(float), (), 2.1)
    assert "temperature" in g.point_data
    assert g.point_data["temperature"].shape == (g.n_points,)
    assert_results_close(g.point_data["temperature"], [1.0, 2.0])

    # Check that correct errors are raised
    with pytest.raises(TypeError, match="dtype"):
        g = create_line_grid()
        g.point_data["temperature"] = np.array([1, 2], dtype=int)
        _ensure_field(g, "point_data", "temperature", np.dtype(float), (), 0)
    with pytest.raises(ValueError, match="tail shape"):
        g = create_line_grid()
        g.point_data["temperature"] = np.array(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=float
        )
        _ensure_field(g, "point_data", "temperature", np.dtype(float), (), 2.1)


def test_pyvista_merge_grids_ensure_field_patch_data_arrays(assert_results_close):
    """Test that _patch_data_arrays correctly ensures both grids have the same
    fields with appropriate fill values."""

    g1 = create_line_grid()
    g2 = create_line_grid()

    g1_temperature = np.array([10.0, 20.0])
    g2_velocity = np.array([[1.0, 1.0], [1.0, 1.0]])
    g2_density = np.array([0.5, 0.5])
    g1.point_data["temperature"] = g1_temperature
    g2.point_data["velocity"] = g2_velocity
    g2.point_data["density"] = g2_density

    g1_mat_id = np.array([5], dtype=np.int32)
    g2_quality = np.array([-0.5], dtype=float)
    g1.cell_data["mat_id"] = g1_mat_id
    g2.cell_data["quality"] = g2_quality

    _patch_data_arrays(
        g1,
        g2,
        point_fill={"temperature": -273.15, "velocity": [9.0, 10.0]},
        cell_fill={"mat_id": -1, "quality": -2.0},
    )

    assert set(g1.point_data.keys()) == {"temperature", "velocity", "density"}
    assert set(g2.point_data.keys()) == {"temperature", "velocity", "density"}
    assert set(g1.cell_data.keys()) == {"mat_id", "quality"}
    assert set(g2.cell_data.keys()) == {"mat_id", "quality"}

    assert_results_close(g1.point_data["temperature"], g1_temperature)
    assert_results_close(g1.point_data["velocity"], [[9.0, 10.0], [9.0, 10.0]])
    assert_results_close(g1.point_data["density"], [0.0, 0.0])

    assert_results_close(g2.point_data["temperature"], [-273.15, -273.15])
    assert_results_close(g2.point_data["velocity"], g2_velocity)
    assert_results_close(g2.point_data["density"], g2_density)

    assert_results_close(g1.cell_data["mat_id"], g1_mat_id)
    assert_results_close(g1.cell_data["quality"], [-2.0, -2.0])

    assert_results_close(g2.cell_data["mat_id"], [-1, -1])
    assert_results_close(g2.cell_data["quality"], g2_quality)


def test_pyvista_merge_grids_preserves_duplicate_points():
    """Test that merge_grids does not attempt to merge duplicate points at the
    same positions."""
    g1 = create_line_grid()
    g2 = create_line_grid()
    merged = merge_grids(g1, g2)
    assert merged.n_points == g1.n_points + g2.n_points
    assert merged.n_cells == g1.n_cells + g2.n_cells


def test_pyvista_merge_grids_union_of_fields(assert_results_close):
    """Test that merge_grids correctly merges two grids with disjoint data
    fields."""
    g1 = create_line_grid()
    g2 = create_line_grid()

    g1.point_data["temperature"] = np.array([1.0, 2.0])
    g2.point_data["velocity"] = np.array([[1.0, 2.0], [3.0, 4.0]])

    merged = merge_grids(
        g1, g2, point_fill={"temperature": -1.0, "velocity": [0.0, 0.0]}
    )

    assert set(merged.point_data.keys()) == {"temperature", "velocity"}
    assert_results_close(merged.point_data["temperature"], [1.0, 2.0, -1.0, -1.0])
    assert_results_close(
        merged.point_data["velocity"],
        [[0.0, 0.0], [0.0, 0.0], [1.0, 2.0], [3.0, 4.0]],
    )


@pytest.mark.parametrize("merge_inplace", (True, False))
@pytest.mark.parametrize("modify_grid_b_inplace", (True, False))
def test_pyvista_merge_grids_inplace_modifications(
    merge_inplace, modify_grid_b_inplace, assert_grids_close
):
    """Test that merge_grids correctly modifies the input grids in place when
    requested, and leaves them unchanged otherwise."""
    g1_original = create_line_grid()
    g2_original = create_line_grid()

    g1_original.point_data["temperature"] = np.array([1.0, 2.0])
    g2_original.point_data["velocity"] = np.array([[1.0, 0.0], [0.0, 1.0]])

    g1 = g1_original.copy(deep=True)
    g2 = g2_original.copy(deep=True)

    g_merged = merge_grids(
        g1, g2, merge_inplace=merge_inplace, modify_grid_b_inplace=modify_grid_b_inplace
    )

    if merge_inplace:
        assert g_merged is g1
        assert set(g1.point_data.keys()) == {"temperature", "velocity"}
    else:
        assert g_merged is not g1
        assert_grids_close(g1, g1_original)

    if modify_grid_b_inplace:
        assert set(g2.point_data.keys()) == {"temperature", "velocity"}
    else:
        assert_grids_close(g2, g2_original)


def test_pyvista_merge_grids_raises_on_dtype_mismatch():
    """Test that merge_grids raises a TypeError when the same field has
    different dtypes in the two grids."""
    g1 = create_line_grid()
    g2 = create_line_grid()

    g1.point_data["field"] = np.array([1.0, 2.0], dtype=float)
    g2.point_data["field"] = np.array([1, 2], dtype=np.int32)

    with pytest.raises(TypeError, match="dtype"):
        merge_grids(g1, g2)


def test_pyvista_merge_grids_raises_on_shape_mismatch():
    """Test that merge_grids raises a ValueError when the same field has
    different shapes in the two grids."""
    g1 = create_line_grid()
    g2 = create_line_grid()

    g1.point_data["field"] = np.array([1.0, 2.0])
    g2.point_data["field"] = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="tail shape"):
        merge_grids(g1, g2)


@pytest.mark.parametrize("merge_inplace", (True, False))
@pytest.mark.parametrize("modify_grid_b_inplace", (True, False))
@pytest.mark.parametrize("merge_order", ("<<a,b>,c>", "<a,<b,c>>"))
def test_pyvista_merge_grids_mixed_cell_types(
    get_corresponding_reference_file_path,
    merge_inplace,
    merge_order,
    modify_grid_b_inplace,
    assert_grids_close,
):
    """Test that merge_grids correctly merges two unstructured grids with mixed
    cell types and point data, preserving all points and fields, and applying
    fill values as needed."""

    mixed_grid = pv.get_reader(
        get_corresponding_reference_file_path(
            reference_file_base_name="mixed_cell_types"
        )
    ).read()

    mesh_mixed_cells_a = mixed_grid.copy(deep=True)
    mesh_mixed_cells_b = mixed_grid.copy(deep=True)
    mesh_mixed_cells_c = mixed_grid.copy(deep=True)

    # Add data fields
    n_cells = mixed_grid.n_cells
    n_points = mixed_grid.n_points
    mesh_mixed_cells_a.point_data["point_field_a"] = np.arange(n_points * 4).reshape(
        n_points, 4
    )
    mesh_mixed_cells_b.cell_data["cell_field_b"] = np.arange(n_cells * 2).reshape(
        n_cells, 2
    )
    mesh_mixed_cells_c.point_data["point_field_c"] = np.arange(n_points)

    # Move the last copy of the mesh
    mesh_mixed_cells_c.points += [0.05, 0, 0]

    if merge_order == "<<a,b>,c>":
        merged_grid = merge_grids(
            mesh_mixed_cells_a,
            mesh_mixed_cells_b,
            merge_inplace=merge_inplace,
            modify_grid_b_inplace=modify_grid_b_inplace,
        )
        merged_grid = merge_grids(
            merged_grid,
            mesh_mixed_cells_c,
            merge_inplace=merge_inplace,
            modify_grid_b_inplace=modify_grid_b_inplace,
        )
    elif merge_order == "<a,<b,c>>":
        merged_grid = merge_grids(
            mesh_mixed_cells_b,
            mesh_mixed_cells_c,
            merge_inplace=merge_inplace,
            modify_grid_b_inplace=modify_grid_b_inplace,
        )
        merged_grid = merge_grids(
            mesh_mixed_cells_a,
            merged_grid,
            merge_inplace=merge_inplace,
            modify_grid_b_inplace=modify_grid_b_inplace,
        )
    else:
        raise ValueError(f"Invalid merge order")

    assert_grids_close(get_corresponding_reference_file_path(), merged_grid)
