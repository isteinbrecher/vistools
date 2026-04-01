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
"""Test the functionality of polyline_cross_section."""

import pytest
import pyvista

from vistools.vtk.merge_polylines import merge_polylines
from vistools.vtk.polyline_cross_section import polyline_cross_section


@pytest.mark.parametrize("closed", [True, False])
def test_vtk_polyline_cross_section(
    get_corresponding_reference_file_path, assert_grids_close, closed
):
    """Test the polyline_cross_section function."""

    grid = pyvista.get_reader(
        get_corresponding_reference_file_path(reference_file_base_name="helix_beam")
    ).read()

    # Merge the polylines
    grid = grid.clean()
    grid = merge_polylines(grid)

    # Sweep a cross section along the helix
    cross_section_points = [[0, 0], [1, 0], [1, 1], [0, 1], [0.7, 0.3]]
    helix_3d = pyvista.UnstructuredGrid(
        polyline_cross_section(grid, cross_section_points, closed=closed)
    )

    # Compare with reference result
    assert_grids_close(
        get_corresponding_reference_file_path(
            additional_identifier=("closed" if closed else "open")
        ),
        helix_3d,
    )
