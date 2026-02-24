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
"""PyVista scalar bar to TikZ export functionality."""

import pyvista as pv
from pyvista import examples

from vistools.pyvista.scalar_bar_to_tikz import export_to_tikz


def test_pyvista_scalar_bar_to_tikz(
    get_corresponding_reference_file_path, tmp_path, assert_tex_close
):
    """Test the scalar bar to TikZ functionality."""

    # Load an example mesh with scalar data
    mesh = examples.load_random_hills()

    # Create a plotter object
    plotter = pv.Plotter(off_screen=True, window_size=[800, 500])

    # Add the mesh with scalar data
    plotter.add_mesh(
        mesh,
        scalars=None,
        scalar_bar_args=dict(
            title="Title1",
            interactive=False,
            height=0.3,
            position_x=0.001,
            position_y=0.1,
            width=0.05,
            vertical=True,
        ),
    )

    plotter.add_mesh(
        mesh,
        scalars=None,
        scalar_bar_args=dict(
            title="Title2",
            interactive=False,
            height=0.1,
            position_x=0.3,
            position_y=0.4,
            width=0.4,
            vertical=False,
        ),
    )

    name = "plot_to_tikz"
    export_to_tikz(tmp_path / name, plotter, dpi=250, is_testing=True)

    # Compare the created TikZ code.
    with open(tmp_path / (name + ".tex"), "r") as tikz_file:
        tikz_code = tikz_file.read().strip()
    with open(get_corresponding_reference_file_path(extension="tex"), "r") as tikz_file:
        tikz_code_ref = tikz_file.read().strip()
    assert_tex_close(tikz_code_ref, tikz_code)
