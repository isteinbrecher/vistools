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
"""Test the compare grids functionality."""

from operator import attrgetter

import numpy as np
import pytest
import pyvista as pv
import vtk

from vistools.vtk.compare_grids import compare_grids


def get_test_grid_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get the data needed to initialize the test grid as a pyvista
    unstructured grid.

    The grid consist of:
        - 1 line
        - 1 quad
        - 1 tetra
        - 1 polyhedron (cube)
    """

    # Points
    points = np.array(
        [
            # Line
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            # Quad
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
            [1.0, 2.0, 0.0],  # 4
            [0.0, 2.0, 0.0],  # 5
            # Tetra
            [0.0, 0.0, 1.0],  # 6
            # Polyhedron cube
            [3.0, 0.0, 0.0],  # 7
            [4.0, 0.0, 0.0],  # 8
            [4.0, 1.0, 0.0],  # 9
            [3.0, 1.0, 0.0],  # 10
            [3.0, 0.0, 1.0],  # 11
            [4.0, 0.0, 1.0],  # 12
            [4.0, 1.0, 1.0],  # 13
            [3.0, 1.0, 1.0],  # 14
        ],
        dtype=np.float64,
    )

    # Standard cells (line, quad, tet)
    cells_basic = [
        # Line
        2,
        0,
        1,
        # Quad
        4,
        2,
        3,
        4,
        5,
        # Tetra
        4,
        0,
        2,
        3,
        6,
    ]

    cell_types_basic = [
        vtk.VTK_LINE,
        vtk.VTK_QUAD,
        vtk.VTK_TETRA,
    ]

    # Polyhedron (cube)

    # The faces of the polyhedron
    faces = [
        [7, 8, 9, 10],  # bottom
        [11, 12, 13, 14],  # top
        [7, 8, 12, 11],  # front
        [8, 9, 13, 12],  # right
        [9, 10, 14, 13],  # back
        [10, 7, 11, 14],  # left
    ]

    # Set the polyhedron connectivity data structure
    polyhedron_connectivity = []
    polyhedron_connectivity.append(len(faces))
    for face in faces:
        polyhedron_connectivity.append(len(face))
        polyhedron_connectivity.extend(face)
    polyhedron_connectivity = [len(polyhedron_connectivity), *polyhedron_connectivity]

    # Create the grid
    cells = np.array(cells_basic + polyhedron_connectivity)
    cell_types = np.array(cell_types_basic + [vtk.VTK_POLYHEDRON])

    return cells, cell_types, points


def get_test_grid() -> pv.UnstructuredGrid:
    """Create the test grid."""

    cells, cell_types, points = get_test_grid_data()
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    return grid


def test_vtk_compare_grids_equal(get_corresponding_reference_file_path):
    """Test that two identical grids compare as equal."""
    grid_1 = get_test_grid()
    grid_2 = get_test_grid()
    grid_ref = pv.read(get_corresponding_reference_file_path())
    assert compare_grids(grid_1, grid_2)
    assert compare_grids(grid_1, grid_ref)


def test_vtk_compare_grids_coordinates_mismatch():
    """Test that a mismatch in point coordinates is detected."""
    grid_1 = get_test_grid()
    grid_2 = get_test_grid()

    # Move a single point
    grid_1.points[0, 0] += 0.1

    # Compare the grids
    compare, lines = compare_grids(grid_1, grid_2, output=True)
    assert not compare
    assert any("point_coordinates: Data values do not match" in s for s in lines)


def test_vtk_compare_grids_cell_types_mismatch():
    """Test that a mismatch in cell types is detected."""
    grid_1 = get_test_grid()
    grid_2 = get_test_grid()

    # Change the cell type in grid_2 (e.g. from tetra to triangle) while keeping same connectivity
    # (This is intentionally inconsistent geometry, but should trigger type mismatch.)
    grid_2.celltypes[:] = vtk.VTK_TRIANGLE

    compare, lines = compare_grids(grid_1, grid_2, output=True)
    assert not compare
    assert any("cell_types: Data values do not match" in s for s in lines)


def test_vtk_compare_grids_cell_connectivity_mismatch():
    """Test that a mismatch in cell connectivity is detected."""

    grid_1 = get_test_grid()
    grid_2_cells_original, grid_2_cell_types, grid_2_points = get_test_grid_data()

    # Change the cell offsets
    grid_2_cells = grid_2_cells_original.copy()
    grid_2_cells[0] = 3
    grid_2_cells[4] = 3
    grid_2 = pv.UnstructuredGrid(grid_2_cells, grid_2_cell_types, grid_2_points)
    compare, lines = compare_grids(grid_1, grid_2, output=True)
    assert not compare
    assert any("cell_offsets: Data values do not match" in s for s in lines)

    # Change the connectivity
    grid_2_cells = grid_2_cells_original.copy()
    grid_2_cells[12] = 7
    grid_2 = pv.UnstructuredGrid(grid_2_cells, grid_2_cell_types, grid_2_points)
    compare, lines = compare_grids(grid_1, grid_2, output=True)
    assert not compare
    assert any("cell_connectivity: Data values do not match" in s for s in lines)

    # Change the face offsets
    grid_2_cells = grid_2_cells_original.copy()
    grid_2_cells[15] = 3
    grid_2_cells[19] = 5
    grid_2 = pv.UnstructuredGrid(grid_2_cells, grid_2_cell_types, grid_2_points)
    compare, lines = compare_grids(grid_1, grid_2, output=True)
    assert not compare
    assert any("face_cell_offsets: Data values do not match" in s for s in lines)

    # Change the face connectivity
    grid_2_cells = grid_2_cells_original.copy()
    grid_2_cells[16] = 5
    grid_2 = pv.UnstructuredGrid(grid_2_cells, grid_2_cell_types, grid_2_points)
    compare, lines = compare_grids(grid_1, grid_2, output=True)
    assert not compare
    assert any("face_cell_connectivity: Data values do not match" in s for s in lines)


@pytest.mark.parametrize(
    "test_mode,expect_pass,info_string",
    [
        ("ok", True, "Compares OK"),
        ("data_mismatch", False, "Data values do not match"),
        ("component_mismatch", False, "Number of components does not match"),
        ("data_type_mismatch", False, "Data types do not match"),
    ],
)
@pytest.mark.parametrize(
    "data_type,access_function,size_function",
    [
        ["point_data", attrgetter("point_data"), lambda grid: grid.n_points],
        ["cell_data", attrgetter("cell_data"), lambda grid: grid.n_cells],
        ["field_data", attrgetter("field_data"), lambda _: 1],
    ],
)
@pytest.mark.parametrize(
    "n_components",
    (1, 3),
)
def test_vtk_compare_grids_data(
    test_mode,
    expect_pass,
    info_string,
    data_type,
    access_function,
    size_function,
    n_components,
):
    """Test that cell/point/field data can be compared."""
    grid_1 = get_test_grid()
    grid_2 = get_test_grid()

    n_components_1 = n_components
    if test_mode == "component_mismatch":
        n_components_2 = n_components + 1
    else:
        n_components_2 = n_components

    data_type_1 = np.float64
    if test_mode == "data_type_mismatch":
        data_type_2 = np.float32
    else:
        data_type_2 = data_type_1

    size = size_function(grid_1)
    access_function(grid_1)["data_name"] = np.linspace(
        0.0, 1.0, size * n_components_1, dtype=data_type_1
    ).reshape(-1, n_components_1)
    access_function(grid_2)["data_name"] = np.linspace(
        0.0, 1.0, size * n_components_2, dtype=data_type_2
    ).reshape(-1, n_components_2)

    if not expect_pass:
        # If we want to fail the comparison, modify the data
        access_function(grid_2)["data_name"][-1] += 0.1

    compare, lines = compare_grids(grid_1, grid_2, output=True)
    assert compare == expect_pass
    assert any(f"{data_type}::data_name: {info_string}" in s for s in lines)


@pytest.mark.parametrize(
    "data_type,access_function,size_function",
    [
        ["point_data", attrgetter("point_data"), lambda grid: grid.n_points],
        ["cell_data", attrgetter("cell_data"), lambda grid: grid.n_cells],
        ["field_data", attrgetter("field_data"), lambda _: 1],
    ],
)
def test_vtk_compare_grids_data_names(data_type, access_function, size_function):
    """Test that mismatches of cell/point/field data names can be found."""
    grid_1 = get_test_grid()
    grid_2 = get_test_grid()

    size = size_function(grid_1)
    access_function(grid_1)["data_name_1"] = np.linspace(0.0, 1.0, size)
    access_function(grid_2)["data_name_2"] = np.linspace(0.0, 1.0, size)

    compare, lines = compare_grids(grid_1, grid_2, output=True)
    assert not compare
    assert any(f"{data_type}: Data fields do not match" in s for s in lines)
