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
"""Merge lines or polylines with each other that represent a continuous
curve."""

import pyvista as pv

from vistools.vtk.merge_polylines import (
    merge_polylines as vtk_merge_polylines,
)


def merge_polylines(
    grid: pv.UnstructuredGrid,
    smooth_angle: float | None = None,
    tol: float = 1e-8,
) -> pv.UnstructuredGrid:
    """Merge lines or polylines with each other that represent a continuous
    curve.

    Args:
        grid:
            Input grid (can only contain lines or polylines).
        smooth_angle:
            Threshold for maximum angle between successive segments along a continuous
            line. The angle is calculated between the tangents of the lines which are
            always outward pointing, thus we get the angle pi if the tangents represent
            a straight polyline.
        tol:
            Tolerance for merging points. If two points are closer than this value,
            they will be merged into one point.
    """
    return pv.UnstructuredGrid(
        vtk_merge_polylines(grid, smooth_angle=smooth_angle, tol=tol)
    )
