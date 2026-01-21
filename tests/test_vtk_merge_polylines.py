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
"""Test the functionality of merge_polylines."""

import numpy as np
import pytest
import pyvista as pv

from vistools.vtk.merge_polylines import merge_polylines

CLEAN_GRID_TOL = 1e-10


@pytest.mark.parametrize("clean", [False, True])
def test_vtk_merge_polylines(
    clean, get_corresponding_reference_file_path, assert_results_equal
):
    """Test the merge_polylines function."""

    grid = pv.get_reader(
        get_corresponding_reference_file_path(additional_identifier="raw")
    ).read()

    identifier = None
    if clean:
        # For some reason the default tolerance does not work here.
        grid = grid.clean(tolerance=CLEAN_GRID_TOL)
        identifier = "clean"

    grid_merged = merge_polylines(grid)

    assert_results_equal(
        get_corresponding_reference_file_path(additional_identifier=identifier),
        grid_merged,
    )


@pytest.mark.parametrize(
    ["smooth_angle", "smooth_angle_name"],
    [[None, "default"], [0.5 * np.pi, "large_angle"]],
)
@pytest.mark.parametrize("clean", [False, True])
def test_vtk_merge_polylines_closed(
    smooth_angle,
    smooth_angle_name,
    clean,
    get_corresponding_reference_file_path,
    assert_results_equal,
):
    """Test the merge_polylines function for a closed polygon."""

    identifier = [smooth_angle_name]

    grid = pv.get_reader(
        get_corresponding_reference_file_path(additional_identifier="raw")
    ).read()

    if clean:
        # For some reason the default tolerance does not work here.
        grid = grid.clean(tolerance=CLEAN_GRID_TOL)
        identifier.append("clean")

    grid_merged = merge_polylines(grid, smooth_angle=smooth_angle)

    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier="_".join(identifier)
        ),
        grid_merged,
    )
