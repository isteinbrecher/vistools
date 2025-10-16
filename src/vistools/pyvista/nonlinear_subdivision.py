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
"""Generate higher order visualizations for quadrilateral meshes."""

import numpy as np
import pyvista as pv


def generate_nonlinear_subdivision(
    grid: pv.UnstructuredGrid, nonlinear_subdivision: int, *, delete_created_arrays=True
) -> tuple[pv.PolyData, pv.PolyData]:
    """Generate a nonlinear subdivision of a grid.

    This function generates a nonlinear subdivision of the surface of a
    given grid and also returns the edges of the initial grid, also
    nonlinearly subdivided.
    """
    # Add the edge flag to the input grid
    grid = grid.copy()
    edge_flag = np.zeros(grid.n_points)
    for i_cell in range(grid.n_cells):
        cell = grid.get_cell(i_cell)
        point_ids = cell.point_ids

        if len(point_ids) == 9:
            for id in range(9):
                if id == 8:
                    edge_flag[point_ids[id]] = 0.0
                else:
                    edge_flag[point_ids[id]] = 1.0
        elif len(point_ids) == 27:
            for id in range(27):
                if id in (20, 21, 22, 23, 24, 25, 26):
                    edge_flag[point_ids[id]] = 0.0
                else:
                    edge_flag[point_ids[id]] = 1.0
        else:
            raise ValueError("Only quad9 and hex27 cells are supported at the moment.")

    grid.point_data["edge_flag"] = edge_flag

    # Refine the surface with nonlinear subdivision
    surface_refined = grid.extract_surface(nonlinear_subdivision=nonlinear_subdivision)
    surface_refined.point_data["original_point_ids"] = surface_refined.point_data[
        "vtkOriginalPointIds"
    ].copy()

    # Flag cells at the "corner"
    corner_cells = surface_refined.point_data_to_cell_data()
    corner_cell_flag = np.abs(corner_cells.cell_data["edge_flag"] - 1) < 1e-10
    surface_refined.cell_data["corner_cell_flag"] = corner_cell_flag

    # Get all edges (we still have to filter the corner edges)
    edges = surface_refined.extract_feature_edges(
        feature_edges=False,
        boundary_edges=True,
        non_manifold_edges=False,
        manifold_edges=True,
    )
    edges = edges.point_data_to_cell_data()

    # Filter the edges by edge_flag.
    edges = edges.threshold((1 - 1e-9, 2), invert=False, scalars="edge_flag")

    # We now want to filter the corner edges that don't have a valid original_point_ids
    corner_cell_flag = edges.cell_data["corner_cell_flag"]
    original_point_ids = edges.cell_data["original_point_ids"]
    final_filter = np.zeros(len(corner_cell_flag), dtype=int)
    for i, (corner_flag, original_id) in enumerate(
        zip(corner_cell_flag, original_point_ids)
    ):
        if corner_flag and original_id < 0:
            final_filter[i] = 0
        else:
            final_filter[i] = 1
    edges.cell_data["final_filter"] = final_filter
    edges = edges.threshold((1, 1), invert=False, scalars="final_filter")

    if delete_created_arrays:
        del surface_refined.point_data["original_point_ids"]
        del surface_refined.point_data["edge_flag"]
        del surface_refined.point_data["vtkOriginalPointIds"]
        del surface_refined.cell_data["corner_cell_flag"]
        del surface_refined.cell_data["vtkOriginalCellIds"]

        del edges.cell_data["final_filter"]
        del edges.cell_data["original_point_ids"]
        del edges.cell_data["edge_flag"]
        del edges.cell_data["corner_cell_flag"]
        del edges.cell_data["vtkOriginalCellIds"]

    return surface_refined, edges
