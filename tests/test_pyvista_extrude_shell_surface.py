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
"""Test the functionality of shell extrusion."""

import pyvista

from vistools.pyvista.extrude_shell_surface import extrude_shell_surface


def test_pyvista_extrude_shell_surface(
    get_corresponding_reference_file_path, assert_results_equal
):
    """Test the extrude_shell_surface function."""

    shell = pyvista.get_reader(
        get_corresponding_reference_file_path(reference_file_base_name="shell")
    ).read()

    shell = shell.warp_by_vector(vectors="displacement")
    shell = shell.clean()
    shell_3d = extrude_shell_surface(shell, thickness=0.05)

    assert_results_equal(get_corresponding_reference_file_path(), shell_3d)
