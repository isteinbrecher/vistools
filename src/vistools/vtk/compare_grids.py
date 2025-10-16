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
"""Compare two grids to each other."""

from typing import Tuple, Union

import numpy as np
from vtk.util import numpy_support as vtk_numpy_support

from vistools.vtk.utils import get_vtk_version

# Comparison tolerances for values computed with single precision.
COMPARE_TOL_SINGLE_PRECISION = {"rtol": 1e-6, "atol": 1e-6}


def compare_grids(
    grid_1, grid_2, *, rtol=None, atol=None, output=False
) -> Union[bool, Tuple[bool, list[str]]]:
    """Compare two grids to each other.

    Args
    ----
    grid_1, grid_2:
        Input grids which are compared to each other
    rtol: float
        Relative tolerance for comparing floating point data
    atol: float
        Absolute tolerance for comparing floating point data
    output: bool
        If output of the results of the comparison should be given
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

        n_1 = array_1.GetNumberOfTuples()
        n_2 = array_2.GetNumberOfTuples()
        if not n_1 == n_2:
            return (
                False,
                f"{name}: Number of tuples does not match, got {n_1} and {n_2}",
            )

        n_1 = array_1.GetNumberOfComponents()
        n_2 = array_2.GetNumberOfComponents()
        if not n_1 == n_2:
            return (
                False,
                f"{name}: Number of components does not match, got {n_1} and {n_2}",
            )

        t_1 = array_1.GetDataType()
        t_2 = array_2.GetDataType()
        if not t_1 == t_2:
            return (
                False,
                f"{name}: Data type does not match, got {t_1} and {t_2}",
            )

        if not np.allclose(
            vtk_numpy_support.vtk_to_numpy(array_1),
            vtk_numpy_support.vtk_to_numpy(array_2),
            rtol=rtol,
            atol=atol,
        ):
            max_diff = np.max(
                np.abs(
                    vtk_numpy_support.vtk_to_numpy(array_1)
                    - vtk_numpy_support.vtk_to_numpy(array_2)
                )
            )
            return (
                False,
                f"{name}: Data values do not match, maximum difference is {max_diff}",
            )

        return True, f"{name}: Compares OK"

    return_value = True
    lines = []

    # Compare the point coordinates
    compare_value, string = compare_arrays(
        grid_1.GetPoints().GetData(),
        grid_2.GetPoints().GetData(),
        "point_coordinates",
    )
    return_value = compare_value and return_value
    lines.append(string)

    # Compare the cells
    compare_value, string = compare_arrays(
        grid_1.GetCellTypesArray(), grid_2.GetCellTypesArray(), "cell_types"
    )
    return_value = compare_value and return_value
    lines.append(string)

    compare_value, string = compare_arrays(
        grid_1.GetCells().GetData(), grid_2.GetCells().GetData(), "cell_connectivity"
    )
    return_value = compare_value and return_value
    lines.append(string)

    if get_vtk_version() < (9, 4, 0):
        compare_value, string = compare_arrays(
            grid_1.GetFaces(),
            grid_2.GetFaces(),
            "face_connectivity",
        )
    else:
        faces1 = grid_1.GetPolyhedronFaces()
        faces2 = grid_2.GetPolyhedronFaces()
        if (faces1 is None) != (faces2 is None):
            return_value = False
            lines.append("face_connectivity: Could not find both face data arrays")
        elif faces1 is not None and faces2 is not None:
            compare_value, string = compare_arrays(
                faces1.GetData(),
                faces2.GetData(),
                "face_connectivity",
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
