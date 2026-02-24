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
"""Interpolate between time steps in pvd time step series."""

from operator import attrgetter

import numpy as np
import pyvista as pv


def temporal_interpolator(
    pvd_collection: pv.PVDReader, time: float, *, tol=1e-10
) -> pv.UnstructuredGrid:
    """Interpolate between time steps in pvd time step series.

    Args
    ----
    pvd_collection:
        PVD reader containing the pvd collection file
    time:
        Time value for the desired interpolated data
    tol:
        Tolerance for deciding if a time step is exactly found
    """

    def get_mesh(time_index) -> pv.UnstructuredGrid:
        """Return the mesh at a given time index."""
        pvd_collection.set_active_time_point(time_index)
        return pvd_collection.read()[0]

    time_values = np.array(pvd_collection.time_values)

    # Check if we hit the time step exactly
    diff = time_values - time
    abs_diff = np.abs(diff)
    closest_time_step = np.argmin(abs_diff)
    if abs_diff[closest_time_step] < tol:
        return get_mesh(closest_time_step)

    positive_indices = np.where(diff > 0)[0]
    end_index = positive_indices[np.argmin(diff[positive_indices])]
    start_index = end_index - 1
    start = time_values[start_index]
    end = time_values[end_index]
    alpha = (time - start) / (end - start)
    factor = [1 - alpha, alpha]

    output_mesh = get_mesh(start_index)
    end_mesh = get_mesh(end_index)

    output_mesh.points = factor[0] * output_mesh.points + factor[1] * end_mesh.points
    for data_type in ["point_data", "cell_data", "field_data"]:
        getter = attrgetter(data_type)
        for key in getter(output_mesh).keys():
            getter(output_mesh)[key] = (
                factor[0] * getter(output_mesh)[key] + factor[1] * getter(end_mesh)[key]
            )

    return output_mesh
