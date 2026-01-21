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

from vistools.pyvista.polyline_cross_section import polyline_cross_section
from vistools.vtk.merge_polylines import merge_polylines


@pytest.mark.parametrize(
    ["closed", "separate_surfaces"],
    [[False, False], [False, True], [True, False], [True, True]],
)
def test_pyvista_polyline_cross_section(
    get_corresponding_reference_file_path,
    assert_results_equal,
    closed,
    separate_surfaces,
):
    """Test the polyline_cross_section function."""

    # Load the helix centerline
    grid = pyvista.get_reader(
        get_corresponding_reference_file_path(reference_file_base_name="helix_beam")
    ).read()
    grid = grid.clean()
    grid = pyvista.UnstructuredGrid(merge_polylines(grid))
    grid_copy = grid.copy(deep=True)
    grid_copy.points += [2, 0, 0]
    grid = pyvista.merge([grid, grid_copy])

    # Sweep a cross section along the helix
    cross_section_points = [[0, 0], [1, 0], [1, 1], [0, 1], [0.7, 0.3]]
    helix_3d = polyline_cross_section(
        grid,
        cross_section_points,
        closed=closed,
        separate_surfaces=separate_surfaces,
    )

    # Compare with reference result
    variant_name = "{}_{}".format(
        "closed" if closed else "open",
        "separate" if separate_surfaces else "connected",
    )
    assert_results_equal(
        get_corresponding_reference_file_path(additional_identifier=variant_name),
        helix_3d,
    )
