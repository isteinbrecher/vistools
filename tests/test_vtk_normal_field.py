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
"""Test the functionality of the shell extrusion."""

import pytest
import pyvista

from vistools.vtk.normal_field import add_normal_field


@pytest.mark.parametrize("nonlinear_subdivision_level", [1, 2])
@pytest.mark.parametrize("clean", [True, False])
def test_vtk_normal_field(
    nonlinear_subdivision_level,
    clean,
    get_corresponding_reference_file_path,
    assert_results_equal,
    assert_results_equal_single_precision_tol,
):
    """Test the add_normal_field function."""
    shell = pyvista.get_reader(
        get_corresponding_reference_file_path(reference_file_base_name="shell")
    ).read()
    shell = shell.warp_by_vector(vectors="displacement")

    if clean:
        shell = shell.clean()

    add_normal_field(
        shell, tolerance=1e-6, nonlinear_subdivision_level=nonlinear_subdivision_level
    )

    name_list = []
    if clean:
        name_list.append("clean")
    if not nonlinear_subdivision_level == 1:
        name_list.append("subdivision")

    # Since the normal field is computed using single precision, we need to use a
    # higher tolerance for the comparison.
    assert_results_equal(
        get_corresponding_reference_file_path(
            additional_identifier="_".join(name_list)
        ),
        shell,
        **assert_results_equal_single_precision_tol,
    )
