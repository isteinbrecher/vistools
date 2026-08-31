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
"""Test the pure vtk version of merge polylines."""

import vtk

from vistools.vtk.merge_polylines import merge_polylines


def test_vtk_merge_polylines(
    load_grid, get_corresponding_reference_file_path, assert_grids_close
):
    """Test the pure VTK implementation for merging polylines."""
    grid = load_grid(
        get_corresponding_reference_file_path(
            reference_file_base_name="pyvista_merge_polylines",
            additional_identifier="raw",
        ),
        import_type="vtk",
    )

    merged_grid = merge_polylines(grid)

    assert isinstance(merged_grid, vtk.vtkUnstructuredGrid)
    assert_grids_close(
        get_corresponding_reference_file_path(
            reference_file_base_name="pyvista_merge_polylines"
        ),
        merged_grid,
    )
