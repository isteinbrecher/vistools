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
"""Test the functionality of sort_grid."""

import numpy as np
import pyvista as pv

from vistools.pyvista.camera import get_camera_settings, set_camera_settings

MOCK_SETTINGS = {
    "window_size": (800, 600),
    "camera_position": (1.0, 1.0, 1.0),
    "camera_focal_point": (0.0, 0.0, 0.0),
    "camera_view_up": (0.0, 1.0, 0.0),
    "parallel_projection": True,
    "parallel_scale": 2.5,
    "view_angle": 20.0,
}
TOL = 1e-10


def check_all_close(a, b):
    """Check if two values are close, with np.allclose."""
    return np.allclose(a, b, atol=TOL, rtol=TOL)


def setup_plotter():
    """Setup a minimal pyvista plotter."""
    mesh = pv.Sphere()
    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh)
    return plotter


def test_set_camera_settings_applies_correctly():
    """Check that set_camera_settings applies the correct settings."""

    plotter = setup_plotter()

    # Apply the mock settings.
    set_camera_settings(plotter, MOCK_SETTINGS)

    camera = plotter.camera
    assert check_all_close(plotter.window_size, MOCK_SETTINGS["window_size"])
    assert check_all_close(camera.GetPosition(), MOCK_SETTINGS["camera_position"])
    assert check_all_close(camera.GetFocalPoint(), MOCK_SETTINGS["camera_focal_point"])
    assert check_all_close(camera.GetViewUp(), MOCK_SETTINGS["camera_view_up"])
    assert check_all_close(
        camera.GetParallelProjection(), MOCK_SETTINGS["parallel_projection"]
    )
    assert check_all_close(camera.GetParallelScale(), MOCK_SETTINGS["parallel_scale"])
    assert check_all_close(camera.GetViewAngle(), MOCK_SETTINGS["view_angle"])


def test_get_camera_settings_returns_expected_keys():
    """Check that get_camera_settings returns the correct dictionary."""

    plotter = setup_plotter()
    plotter.show(auto_close=False)  # prepare the plotter but don't block

    # Set camera settings
    set_camera_settings(plotter, MOCK_SETTINGS)

    # Emulate capturing camera settings without key interaction
    settings = get_camera_settings(plotter, print_settings=False)

    assert len(settings) == len(MOCK_SETTINGS)
    for key in settings:
        assert check_all_close(settings[key], MOCK_SETTINGS[key])

    plotter.close()
