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
"""Extrude a profile along a polyline."""

from typing import List

import numpy as np
import vtk
from numpy.typing import NDArray
from vtk.util import numpy_support as vtk_numpy_support

from vistools.vtk.vtk_data_structures_utils import vtk_id_to_list


def polyline_cross_section(
    grid: vtk.vtkUnstructuredGrid, cross_section_points, *, closed: bool = True
) -> vtk.vtkPolyData:
    """Extrude a profile defined by the cross section coordinates along a
    polyline.

    Args
    ----
    grid:
        Polyline defining the centerline of the extruded structure
    cross_section_points:
        In-cross-section coordinates defining the profile
    closed:
        Flag if the profile is open or closed
    """

    # Get information about input grid
    n_cells = grid.GetNumberOfCells()

    # Get number of cross-section points
    n_cross_section_points = len(cross_section_points)

    # Check that all cells are poly lines.
    for i in range(n_cells):
        if not grid.GetCellType(i) == 4:
            raise ValueError("Only poly lines (vtk type 4) are supported")

    # Data arrays
    point_data_input = grid.GetPointData()
    base_vector_data: List[NDArray] = [None] * 3
    point_data = {}
    for i_point_data in range(point_data_input.GetNumberOfArrays()):
        name = point_data_input.GetArrayName(i_point_data)
        if name.startswith("base_vector_"):
            i_base = int(name.split("_")[-1])
            base_vector_data[i_base - 1] = vtk_numpy_support.vtk_to_numpy(
                point_data_input.GetArray(i_point_data)
            )
        else:
            point_data[name] = vtk_numpy_support.vtk_to_numpy(
                point_data_input.GetArray(i_point_data)
            )

    # New points
    new_point_coordinates = vtk.vtkPoints()
    new_point_data = {}
    for data_name in point_data.keys():
        if point_data[data_name].ndim == 1:
            n_components = 1
        else:
            n_components = point_data[data_name].shape[1]
        new_point_data[data_name] = vtk.vtkDoubleArray()
        new_point_data[data_name].SetName(data_name)
        new_point_data[data_name].SetNumberOfComponents(n_components)
    new_polygons = []
    new_quad4 = []

    def extrude_cross_section_polyline(polyline: vtk.vtkPolyLine):
        """Extrude the cross section along the given polyline."""

        i_start = new_point_coordinates.GetNumberOfPoints()

        point_ids = vtk_id_to_list(polyline.GetPointIds())
        for i_point_centerline, point_centerline_id in enumerate(point_ids):
            i_start_inner = new_point_coordinates.GetNumberOfPoints()

            coordinates = np.array(grid.GetPoint(point_centerline_id))
            base_vectors = [
                base_vector_data[i_dir][point_centerline_id] for i_dir in range(3)
            ]
            for i_cross_section_point, cross_section_point in enumerate(
                cross_section_points
            ):
                new_coordinate = (
                    coordinates
                    + cross_section_point[0] * base_vectors[1]
                    + cross_section_point[1] * base_vectors[2]
                )
                new_point_coordinates.InsertNextPoint(new_coordinate)

                # Set the point data
                for data_name in new_point_data.keys():
                    n_components = new_point_data[data_name].GetNumberOfComponents()
                    if n_components == 1:
                        new_point_data[data_name].InsertNextValue(
                            point_data[data_name][point_centerline_id]
                        )
                    else:
                        for value in point_data[data_name][point_centerline_id]:
                            new_point_data[data_name].InsertNextValue(value)

                # Set the quad4 cells
                is_start_centerline = i_point_centerline == 0
                is_last_cross_section = (
                    n_cross_section_points - 1 == i_cross_section_point
                )
                if (not is_start_centerline) and not (
                    is_last_cross_section and not closed
                ):
                    new_cell = vtk.vtkQuad()
                    new_cell.GetPointIds().SetNumberOfIds(4)
                    new_cell.GetPointIds().SetId(
                        0,
                        i_start_inner - n_cross_section_points + i_cross_section_point,
                    )
                    new_cell.GetPointIds().SetId(
                        1, i_start_inner + i_cross_section_point
                    )
                    if i_cross_section_point == n_cross_section_points - 1:
                        i_cross_section_point = -1
                    new_cell.GetPointIds().SetId(
                        2, i_start_inner + i_cross_section_point + 1
                    )
                    new_cell.GetPointIds().SetId(
                        3,
                        i_start_inner
                        - n_cross_section_points
                        + i_cross_section_point
                        + 1,
                    )
                    new_quad4.append(new_cell)

        i_end = new_point_coordinates.GetNumberOfPoints() - 1

        # Set the front and end polygon
        if closed:
            for index, reverse in [[i_start, False], [i_end, True]]:
                new_cell = vtk.vtkPolygon()
                new_cell.GetPointIds().SetNumberOfIds(n_cross_section_points)
                for i_point_polygon in range(n_cross_section_points):
                    if not reverse:
                        new_cell.GetPointIds().SetId(
                            n_cross_section_points - i_point_polygon - 1,
                            index + i_point_polygon,
                        )
                    else:
                        new_cell.GetPointIds().SetId(
                            i_point_polygon, index - i_point_polygon
                        )
                new_polygons.append(new_cell)

    # Create the new data for each polyline
    for i_cell in range(n_cells):
        extrude_cross_section_polyline(grid.GetCell(i_cell))

    # Add the new points and cells
    output_grid = vtk.vtkPolyData()
    output_grid.Initialize()
    output_grid.SetPoints(new_point_coordinates)
    for name in new_point_data.keys():
        output_grid.GetPointData().AddArray(new_point_data[name])
    output_grid.Allocate(len(new_polygons) + len(new_quad4), 1)
    for cell in new_polygons + new_quad4:
        output_grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

    return output_grid
