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
"""Provide the functionality to convert vtk data structures to a dictionary."""

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from vistools.vtk.vtk_data_structures_utils import vtk_id_to_list


def vtk_to_dict(grid: vtk.vtkUnstructuredGrid) -> dict:
    """Convert a VTK unstructured grid to a dictionary.

    Args:
        grid: The VTK unstructured grid to convert.

    Returns:
        Dictionary containing points, cells, cell types, point data,
        cell data, and field data.
    """

    def data_attributes_to_dict(attributes) -> dict:
        """Convert VTK data attributes (point data, cell data, field data) to a
        dictionary."""
        result = {}
        for i in range(attributes.GetNumberOfArrays()):
            array = attributes.GetArray(i)
            name = array.GetName()
            result[name] = vtk_to_numpy(array)
        return result

    # Get the connectivity of each cell.
    cells = []
    cell_types = []
    faces: list[list[list[int]] | None] = []
    id_list = vtk.vtkIdList()
    for cell_id in range(grid.GetNumberOfCells()):
        grid.GetCellPoints(cell_id, id_list)
        cells.append(vtk_id_to_list(id_list))
        cell_type = grid.GetCellType(cell_id)
        cell_types.append(cell_type)
        if cell_type == vtk.VTK_POLYHEDRON:
            cell = grid.GetCell(cell_id)
            cell_faces = []
            for face_id in range(cell.GetNumberOfFaces()):
                face = cell.GetFace(face_id)
                face_ids = [face.GetPointId(i) for i in range(face.GetNumberOfPoints())]
                cell_faces.append(face_ids)

            faces.append(cell_faces)
        else:
            faces.append(None)

    return_dict = {
        "points": vtk_to_numpy(grid.GetPoints().GetData()),
        "cells": cells,
        "cell_types": cell_types,
        "point_data": data_attributes_to_dict(grid.GetPointData()),
        "cell_data": data_attributes_to_dict(grid.GetCellData()),
        "field_data": data_attributes_to_dict(grid.GetFieldData()),
    }
    if vtk.VTK_POLYHEDRON in return_dict["cell_types"]:
        return_dict["faces"] = faces
    return return_dict
