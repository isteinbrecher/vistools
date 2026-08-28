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
"""Provide functionality to convert between pyvista and vtk grids."""

from pathlib import Path
from typing import TypeAlias

from vtkmodules.vtkCommonDataModel import vtkUnstructuredGrid

try:
    import pyvista as pv

    pyvista_is_installed = True
except ImportError:
    pyvista_is_installed = False

GridLike: TypeAlias = vtkUnstructuredGrid
GridLikeFile: TypeAlias = GridLike | Path


def cast_output_to_input_type(input_grid: GridLike, output_grid: GridLike) -> GridLike:
    """Cast the output grid to the same type as the input grid."""
    if not pyvista_is_installed:
        return output_grid
    else:
        if isinstance(input_grid, pv.UnstructuredGrid):
            return pv.UnstructuredGrid(output_grid)
        else:
            return output_grid
