# The MIT License (MIT)
#
# Copyright (c) 2023-2025 Ivo Steinbrecher
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
"""Test the functionality of sort_grid."""

import os

import numpy as np
import pyvista

from vistools.pyvista.sort_grid import sort_grid


def test_pyvista_sort_grid_complete(
    get_corresponding_reference_file_path, assert_results_equal
):
    """Test the sort grid function.

    This is done by comparing FEM grids that were generated with a
    different number of processors. In this test we completely sort the
    grid, i.e., we sort all points such that their ordering is
    completely defined by the sorting keys
    """

    mesh_serial, mesh_parallel = [
        pyvista.get_reader(
            get_corresponding_reference_file_path(reference_file_base_name=name)
        ).read()
        for name in ["sort_serial", "sort_parallel"]
    ]

    # Since each element writes it's own nodes, we first sort the nodes by the cell id
    mesh_serial = mesh_serial.cell_data_to_point_data(pass_cell_data=True)
    mesh_parallel = mesh_parallel.cell_data_to_point_data(pass_cell_data=True)

    # To make the sorting more challenging we split the cell and point IDs into individual
    # fields representing the digits
    parallel_node_gid = np.array(mesh_parallel.point_data["node_gid"], dtype=int)
    mesh_parallel.point_data["node_gid_001"] = parallel_node_gid % 10
    mesh_parallel.point_data["node_gid_010"] = parallel_node_gid % 100
    mesh_parallel.point_data["node_gid_100"] = parallel_node_gid % 1000
    parallel_cell_gid = np.array(mesh_parallel.cell_data["element_gid"], dtype=int)
    mesh_parallel.cell_data["element_gid_01"] = parallel_cell_gid % 10
    mesh_parallel.cell_data["element_gid_10"] = parallel_cell_gid % 100
    mesh_parallel_sorted = sort_grid(
        mesh_parallel,
        sort_point_field=[
            "element_gid",
            "node_gid_100",
            "node_gid_010",
            "node_gid_001",
        ],
        sort_cell_field=["element_gid_10", "element_gid_01"],
    )

    # The serial array also has to be sorted, because we also sort with respect to the
    # node ids above.
    mesh_serial_sorted = sort_grid(
        mesh_serial,
        sort_point_field=["element_gid", "node_gid"],
        sort_cell_field=["element_gid"],
    )

    # Remove the arbitrary created data
    del mesh_parallel_sorted.point_data["node_gid_001"]
    del mesh_parallel_sorted.point_data["node_gid_010"]
    del mesh_parallel_sorted.point_data["node_gid_100"]
    del mesh_parallel_sorted.cell_data["element_gid_01"]
    del mesh_parallel_sorted.cell_data["element_gid_10"]

    # Compare the two grids
    assert_results_equal(mesh_parallel_sorted, mesh_serial_sorted)


def test_pyvista_sort_grid_partial(
    get_corresponding_reference_file_path, assert_results_equal
):
    """Test the sort grid function.

    This is done by comparing FEM grids that were generated with a
    different number of processors. Since each element writes it's own
    nodes, we will have to sort by the cell IDs and the node IDs. This
    additionally tests that when sorting data where multiple entries
    have the same value, the ordering within these entries is preserved.
    """

    mesh_serial, mesh_parallel = [
        pyvista.get_reader(
            get_corresponding_reference_file_path(reference_file_base_name=name)
        ).read()
        for name in ["sort_serial", "sort_parallel"]
    ]

    # Sort the parallel grid w.r.t. to the cell id
    mesh_parallel = mesh_parallel.cell_data_to_point_data(pass_cell_data=True)
    mesh_parallel_sorted = sort_grid(
        mesh_parallel,
        sort_cell_field=["element_gid"],
        sort_point_field=["element_gid"],
    )

    # Remove data created for sorting
    del mesh_parallel_sorted.point_data["element_gid"]

    # Compare the two grids
    assert_results_equal(mesh_parallel_sorted, mesh_serial)


def test_pyvista_sort_grid_mixed_types(
    get_corresponding_reference_file_path, assert_results_equal
):
    """Test that the sort grid function can handle polyhedrons."""

    mesh_mixed_cells = pyvista.get_reader(
        get_corresponding_reference_file_path(
            reference_file_base_name="mixed_cell_types"
        )
    ).read()

    # Sort the parallel grid
    mesh_mixed_cells.point_data["sort_id"] = np.arange(
        mesh_mixed_cells.number_of_points - 1, -1, -1, dtype=np.int64
    )
    mesh_mixed_cells_sorted = sort_grid(mesh_mixed_cells, sort_point_field=["sort_id"])

    # Compare with the reference grid
    assert_results_equal(
        get_corresponding_reference_file_path(
            reference_file_base_name="mixed_cell_types_sorted"
        ),
        mesh_mixed_cells_sorted,
    )
