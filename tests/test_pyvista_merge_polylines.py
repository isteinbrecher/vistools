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
"""Test the functionality of merge_polylines in pyvista."""

import pyvista as pv

from vistools.pyvista.merge_polylines import merge_polylines


def test_pyvista_merge_polylines_types(
    get_corresponding_reference_file_path, assert_grids_close
):
    """Test that the merge_polylines function returns the correct type."""

    reference_file_base_name = "vtk_merge_polylines"
    grid = pv.get_reader(
        get_corresponding_reference_file_path(
            reference_file_base_name=reference_file_base_name,
            additional_identifier="raw",
        )
    ).read()

    grid_merged = merge_polylines(grid)

    assert_grids_close(
        get_corresponding_reference_file_path(
            reference_file_base_name=reference_file_base_name
        ),
        grid_merged,
    )
    assert type(grid_merged) == pv.UnstructuredGrid
