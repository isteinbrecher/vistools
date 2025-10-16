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
"""Extrude a given shell surface in normal direction."""

import numpy as np
import pyvista as pv

from vistools.vtk.normal_field import add_normal_field


def extrude_shell_surface(
    shell: pv.UnstructuredGrid,
    *,
    thickness: float,
    add_normal_field_kwargs: None | dict = {},
) -> pv.UnstructuredGrid:
    """Extrude a given shell (quad9) in thickness direction.

    It is assumed that the shell is already in the deformed configuration.

    Args:
        shell: Input grid
        thickness: Thickness of the shell
        add_normal_field_kwargs: Arguments to be passed to `add_normal_field`.
            If this is `None`, the grid has to already contain the surface normals
            field `surface_normals`.
    """

    id_set = set(shell.celltypes)
    if not len(id_set) == 1 or not id_set.pop() == 28:
        raise ValueError("Extrude shell is only implemented for quad9")

    # Compute the normals on the surface
    if add_normal_field_kwargs is not None:
        add_normal_field(shell, **add_normal_field_kwargs)

    # Extrude the mesh
    n_points_shell = shell.number_of_points
    n_cells_shell = shell.number_of_cells
    points = np.tile(shell.points, (3, 1))
    for i, factor in enumerate([-1, 1]):
        points[(i + 1) * n_points_shell : (i + 2) * n_points_shell] = (
            shell.points
            + factor * 0.5 * thickness * shell.point_data["surface_normals"]
        )
    cell_types = np.repeat(29, n_cells_shell)
    cell_connectivity = []
    for i_cell in range(n_cells_shell):
        cell_connectivity.append(27)

        cell_ids_mid = np.array(shell.get_cell(i_cell).point_ids)
        cell_ids_bottom = cell_ids_mid + n_points_shell
        cell_ids_top = cell_ids_mid + 2 * n_points_shell

        # Set the new point IDs
        cell_connectivity.extend(
            [
                cell_ids_bottom[0],
                cell_ids_bottom[1],
                cell_ids_bottom[2],
                cell_ids_bottom[3],
                #
                cell_ids_top[0],
                cell_ids_top[1],
                cell_ids_top[2],
                cell_ids_top[3],
                #
                cell_ids_bottom[4],
                cell_ids_bottom[5],
                cell_ids_bottom[6],
                cell_ids_bottom[7],
                #
                cell_ids_top[4],
                cell_ids_top[5],
                cell_ids_top[6],
                cell_ids_top[7],
                #
                cell_ids_mid[0],
                cell_ids_mid[1],
                cell_ids_mid[2],
                cell_ids_mid[3],
                #
                cell_ids_mid[7],
                cell_ids_mid[5],
                cell_ids_mid[4],
                cell_ids_mid[6],
                #
                cell_ids_bottom[8],
                cell_ids_top[8],
                cell_ids_mid[8],
            ]
        )
    shell_extruded = pv.UnstructuredGrid(cell_connectivity, cell_types, points)

    # Add data fields
    for key, value in shell.point_data.items():
        shell_extruded.point_data[key] = np.tile(value, (3, 1))
    for key, value in shell.cell_data.items():
        shell_extruded.cell_data[key] = value

    return shell_extruded
