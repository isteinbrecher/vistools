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
"""Compare two grids to each other."""

import numpy as np
import vtk
from vtk.util import numpy_support as vtk_numpy_support


def _vtk_array_to_info(array: vtk.vtkDataArray | vtk.vtkPoints) -> dict:
    """Convert a vtk array to a dictionary with relevant information for
    comparison."""

    if isinstance(array, vtk.vtkPoints):
        array = array.GetData()
    elif isinstance(array, vtk.vtkDataArray):
        pass
    else:
        raise ValueError(f"Unsupported array type {type(array)}")

    return {
        "size": array.GetNumberOfTuples(),
        "components": array.GetNumberOfComponents(),
        "data_type": array.GetDataType(),
        "data": vtk_numpy_support.vtk_to_numpy(array),
    }


def compare_grids(
    grid_1,
    grid_2,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    output: bool = False,
) -> bool | tuple[bool, list[str]]:
    """Compare two grids to each other.

    Args:
        grid_1, grid_2: Input grids which are compared to each other
        rtol: Relative tolerance for comparing floating point data
        atol: Absolute tolerance for comparing floating point data
        output: If output of the results of the comparison should be given

    Returns:
        If output is False, only a boolean value indicating whether the grids are
        equal is returned. If output is True, a tuple of the boolean value and a
        list of strings with the results of the comparison is returned.
    """

    if rtol is None:
        rtol = 1e-8
    if atol is None:
        atol = 1e-8

    def compare_arrays(array_1, array_2, name):
        """Compare two arrays."""

        if array_1 is None and array_2 is None:
            return True, f"{name}: OK (empty)"
        elif array_1 is None or array_2 is None:
            return (
                False,
                f"{name}: Array 1 is {type(array_1)}, array 2 is {type(array_2)}",
            )

        array_1_info = _vtk_array_to_info(array_1)
        array_2_info = _vtk_array_to_info(array_2)

        n_1 = array_1_info["size"]
        n_2 = array_2_info["size"]
        if not n_1 == n_2:
            return (
                False,
                f"{name}: Number of tuples does not match, got {n_1} and {n_2}",
            )

        n_1 = array_1_info["components"]
        n_2 = array_2_info["components"]
        if not n_1 == n_2:
            return (
                False,
                f"{name}: Number of components does not match, got {n_1} and {n_2}",
            )

        # This list is used to specify certain types that are considered equivalent
        # for the comparison, e.g., because they are used interchangeably in different
        # versions of VTK or PyVista.
        equivalent_types = [
            {12, 16},  # Unsigned integer and vtk_id_type
        ]
        t_1 = array_1_info["data_type"]
        t_2 = array_2_info["data_type"]
        if t_1 == t_2:
            pass
        elif {t_1, t_2} in equivalent_types:
            pass
        else:
            return (
                False,
                f"{name}: Data types do not match, got {t_1} and {t_2}",
            )

        if not np.allclose(
            array_1_info["data"],
            array_2_info["data"],
            rtol=rtol,
            atol=atol,
        ):
            max_diff = np.max(np.abs(array_1_info["data"] - array_2_info["data"]))
            return (
                False,
                f"{name}: Data values do not match, maximum difference is {max_diff}",
            )

        return True, f"{name}: Compares OK"

    return_value = True
    lines = []

    # Compare the point coordinates
    compare_value, string = compare_arrays(
        grid_1.GetPoints(), grid_2.GetPoints(), "point_coordinates"
    )
    return_value = compare_value and return_value
    lines.append(string)

    # Compare the cells
    compare_value, string = compare_arrays(
        grid_1.GetCellTypes(), grid_2.GetCellTypes(), "cell_types"
    )
    return_value = compare_value and return_value
    lines.append(string)

    grid_1_offsets = grid_1.GetCells().GetOffsetsArray()
    grid_1_connectivity = grid_1.GetCells().GetConnectivityArray()
    grid_2_offsets = grid_2.GetCells().GetOffsetsArray()
    grid_2_connectivity = grid_2.GetCells().GetConnectivityArray()
    compare_value, string = compare_arrays(
        grid_1_offsets, grid_2_offsets, "cell_offsets"
    )
    return_value = compare_value and return_value
    lines.append(string)
    compare_value, string = compare_arrays(
        grid_1_connectivity, grid_2_connectivity, "cell_connectivity"
    )
    return_value = compare_value and return_value
    lines.append(string)

    faces_1 = grid_1.GetPolyhedronFaces()
    faces_2 = grid_2.GetPolyhedronFaces()
    if (faces_1 is None) != (faces_2 is None):
        return_value = False
        lines.append("face_connectivity: Could not find both face data arrays")
    elif faces_1 is not None and faces_2 is not None:
        faces_1_offsets = faces_1.GetOffsetsArray()
        faces_1_connectivity = faces_1.GetConnectivityArray()
        faces_2_offsets = faces_2.GetOffsetsArray()
        faces_2_connectivity = faces_2.GetConnectivityArray()
        compare_value, string = compare_arrays(
            faces_1_offsets, faces_2_offsets, "face_cell_offsets"
        )
        return_value = compare_value and return_value
        lines.append(string)
        compare_value, string = compare_arrays(
            faces_1_connectivity, faces_2_connectivity, "face_cell_connectivity"
        )
        return_value = compare_value and return_value
        lines.append(string)

    def compare_data_fields(data_1, data_2, name):
        """Compare multiple data sets grouped together."""

        names_1, names_2 = [
            set([data.GetArrayName(i) for i in range(data.GetNumberOfArrays())])
            for data in [data_1, data_2]
        ]

        if not names_1 == names_2:
            return (
                False,
                [f"{name}: Data fields do not match, got {names_1} and {names_2}"],
            )

        if len(names_1) == 0:
            return True, [f"{name}: OK (empty)"]

        lines = []
        return_value = True
        for field_name in names_1:
            compare_value, string = compare_arrays(
                data_1.GetArray(field_name),
                data_2.GetArray(field_name),
                f"{name}::{field_name}",
            )
            return_value = compare_value and return_value
            lines.append(string)

        return return_value, lines

    # Compare actual data
    compare_value, strings = compare_data_fields(
        grid_1.GetFieldData(), grid_2.GetFieldData(), "field_data"
    )
    return_value = compare_value and return_value
    lines.extend(strings)
    compare_value, strings = compare_data_fields(
        grid_1.GetCellData(), grid_2.GetCellData(), "cell_data"
    )
    return_value = compare_value and return_value
    lines.extend(strings)
    compare_value, strings = compare_data_fields(
        grid_1.GetPointData(), grid_2.GetPointData(), "point_data"
    )
    return_value = compare_value and return_value
    lines.extend(strings)

    if output:
        return return_value, lines
    else:
        return return_value
