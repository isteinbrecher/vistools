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
"""Camera functionality for pyvista."""

import json

import pyvista as pv


def get_camera_settings(
    plotter: pv.Plotter, *, zoom_factor: float = 1.1, print_settings: bool = True
) -> dict:
    """Show a given plotter and return the chosen camera and window settings.

    After closing the viewer, a dictionary of settings is returned.

    This function attaches interactive hotkeys to the PyVista viewer:
    - "c": This will print the camera settings to the console.
    - "m": This will increase the view zoom.
    - "n": This will decrease the view zoom.

    Args:
        plotter: The plotter instance (should already contain the scene).
        zoom_factor: The amount by which to zoom in/out per keypress.
        print_settings: Whether to print the camera settings to the console on close.

    Returns:
        Dictionary of camera and window configuration.
    """

    def zoom_in():
        """Zoom in by decreasing the view angle."""
        camera = plotter.camera
        camera.SetViewAngle(camera.GetViewAngle() / zoom_factor)
        plotter.render()

    def zoom_out():
        """Zoom out by increasing the view angle."""
        camera = plotter.camera
        camera.SetViewAngle(camera.GetViewAngle() * zoom_factor)
        plotter.render()

    def get_camera_state() -> dict:
        """Capture current camera and window settings."""
        camera = plotter.camera
        return {
            "camera_focal_point": camera.GetFocalPoint(),
            "camera_position": camera.GetPosition(),
            "camera_view_up": camera.GetViewUp(),
            "parallel_projection": camera.GetParallelProjection(),
            "parallel_scale": camera.GetParallelScale(),
            "view_angle": camera.GetViewAngle(),
            "window_size": plotter.window_size,
        }

    def print_camera_state():
        """Print current camera settings as formatted JSON."""
        print("\nView configuration:\n")
        print(json.dumps(get_camera_state(), indent=2))

    # Attach interactive keys
    plotter.add_key_event("c", print_camera_state)
    plotter.add_key_event("m", zoom_in)
    plotter.add_key_event("n", zoom_out)
    plotter.add_text(
        "Hotkeys:\n[c] Print camera settings\n[m] Zoom in\n[n] Zoom out",
        position="lower_right",
        font_size=12,
        color="black",
    )

    # Start viewer
    plotter.show(auto_close=False)

    settings = get_camera_state()

    if print_settings:
        print_camera_state()

    return settings


def set_camera_settings(plotter: pv.Plotter, settings: dict):
    """Set previously captured camera and window settings to a plotter.

    Args:
        plotter: The target PyVista plotter instance.
        settings: A dictionary as returned by `get_camera_settings()`.
    """
    camera = plotter.camera
    if "camera_position" in settings:
        camera.SetPosition(settings["camera_position"])
    if "camera_focal_point" in settings:
        camera.SetFocalPoint(settings["camera_focal_point"])
    if "camera_view_up" in settings:
        camera.SetViewUp(settings["camera_view_up"])
    if "parallel_projection" in settings:
        camera.SetParallelProjection(settings["parallel_projection"])
    if "parallel_scale" in settings:
        camera.SetParallelScale(settings["parallel_scale"])
    if "view_angle" in settings:
        camera.SetViewAngle(settings["view_angle"])
    if "window_size" in settings:
        plotter.window_size = settings["window_size"]

    plotter.reset_camera_clipping_range()
    plotter.render()
