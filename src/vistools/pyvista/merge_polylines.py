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
"""Merge lines or polylines and return a PyVista grid."""

import pyvista as pv

from vistools.vtk.merge_polylines import merge_polylines as vtk_merge_polylines


def merge_polylines(
    grid: pv.UnstructuredGrid,
    *,
    smooth_angle: float | None = None,
    tol: float = 1e-8,
) -> pv.UnstructuredGrid:
    """Merge connected lines or polylines into continuous curves.

    This is the PyVista adapter for :func:`vistools.vtk.merge_polylines`.

    Args:
        grid: Input grid containing only lines or polylines.
        smooth_angle: Threshold for the maximum angle between successive
            segments. See the VTK implementation for details.
        tol: Maximum distance between points that should be merged.

    Returns:
        The merged grid as a PyVista unstructured grid.
    """
    output_grid = vtk_merge_polylines(
        grid,
        smooth_angle=smooth_angle,
        tol=tol,
    )
    if output_grid is None:
        raise RuntimeError("The VTK merge did not return an output grid")
    return pv.wrap(output_grid)
