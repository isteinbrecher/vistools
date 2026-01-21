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
"""Test the functionality of temporal_interpolator."""

import os

import pyvista as pv

from vistools.pyvista.temporal_interpolator import temporal_interpolator


def test_pyvista_temporal_interpolator(
    get_corresponding_reference_file_path, assert_results_equal
):
    """Test the temporal_interpolator function."""

    # Get the pvd reader
    pvd_path = os.path.join(get_corresponding_reference_file_path(extension="pvd"))
    pvd_reader = pv.get_reader(pvd_path)

    # Check if we can extract at a given time value
    mesh_01 = temporal_interpolator(pvd_reader, 1.0)
    assert_results_equal(
        get_corresponding_reference_file_path(additional_identifier="01"), mesh_01
    )

    # Check if we can interpolate between time steps
    mesh_interpolated = temporal_interpolator(pvd_reader, 2.0)
    assert_results_equal(
        get_corresponding_reference_file_path(additional_identifier="interpolated"),
        mesh_interpolated,
    )
