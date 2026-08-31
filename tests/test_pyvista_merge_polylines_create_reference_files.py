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
"""Create the reference files for the merge polylines filter.

Initially taken from pvutils (https://github.com/imcs-compsim/pvutils).
"""

import numpy as np
import pyvista as pv
from beamme.core.element_beam import Beam2, Beam3
from beamme.core.material import MaterialBeamBase
from beamme.core.mesh import Mesh
from beamme.core.rotation import Rotation
from beamme.mesh_creation_functions.beam_line import (
    create_beam_mesh_line,
)


def create_line_not_connected_elements(
    mesh, beam_class, material, start_point, end_point, n_el
):
    """Create a line of elements that are not connected to each other.

    This allows for more complicated test cases, and emulates also the
    "old" BeamMe output behavior.
    """
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    for i in range(n_el):
        create_beam_mesh_line(
            mesh,
            beam_class,
            material,
            start_point + (end_point - start_point) * i / n_el,
            start_point + (end_point - start_point) * (i + 1) / n_el,
            n_el=1,
        )


def get_grid_for_testing(mesh: Mesh) -> pv.UnstructuredGrid:
    """Get a grid representing the mesh and add some dummy cell data to it."""
    grid = mesh.get_vtu_representation()
    grid.cell_data["cell_data_scalar"] = np.arange(grid.number_of_cells)
    grid.cell_data["cell_data_vector"] = np.arange(3 * grid.number_of_cells).reshape(
        (grid.number_of_cells, 3)
    )
    return grid


def test_pyvista_merge_polylines_create_reference_files(
    get_corresponding_reference_file_path, assert_grids_close
) -> None:
    """Create different test meshes for the merge polyline filter."""

    mesh = Mesh()
    mat = MaterialBeamBase(radius=0.1)

    n_el = 2

    # Create a single beam.
    z = -2
    create_line_not_connected_elements(mesh, Beam3, mat, [0, 0, z], [1, 0, z], n_el=1)

    # Create two beams.
    z = -1
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 0, z], [1, 0, z], n_el=n_el
    )

    # Create a closed path.
    z = 0
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 0, z], [1, 0, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [1, 0, z], [1, 1, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [1, 1, z], [0, 1, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [0, 0, z], n_el=n_el
    )

    # Create check all possible connection points.
    z = 1
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 0, z], [0, 1, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [0.5, 2, z], n_el=n_el
    )

    z = 2
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 0, z], [0, 1, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0.5, 2, z], [0, 1, z], n_el=n_el
    )

    z = 3
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [0, 0, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [0.5, 2, z], n_el=n_el
    )

    z = 4
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [0, 0, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0.5, 2, z], [0, 1, z], n_el=n_el
    )

    # Create a simple bifurcation.
    z = 5
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 0, z], [0, 1, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [0.5, 2, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [-0.5, 2, z], n_el=n_el
    )

    # Create a bifurcation with a closed circle.
    z = 6
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 0, z], [0, 1, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [0.5, 2, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 1, z], [-0.5, 2, z], n_el=n_el
    )
    create_line_not_connected_elements(
        mesh, Beam3, mat, [0, 0, z], [-0.5, 2, z], n_el=n_el
    )

    assert_grids_close(
        get_corresponding_reference_file_path(
            reference_file_base_name="pyvista_merge_polylines_raw"
        ),
        get_grid_for_testing(mesh),
    )


def polygon_mesh(
    beam_type: type,
    mat: MaterialBeamBase,
    radius: float,
    segments: int,
    *,
    n_segments: int | None = None,
) -> Mesh:
    """Create a regular polygon."""

    mesh = Mesh()

    delta_phi = 2.0 * np.pi / segments
    if n_segments is None:
        n_segments = segments

    def get_pos(angle):
        """Return the position for the polygon."""
        return radius * np.array([np.cos(angle), np.sin(angle), 0])

    mesh_line = Mesh()
    create_line_not_connected_elements(
        mesh_line, beam_type, mat, get_pos(0), get_pos(delta_phi), n_el=1
    )

    for _ in range(n_segments):
        mesh_add = mesh_line.copy()
        mesh.add(mesh_add)
        mesh_line.rotate(Rotation([0, 0, 1], delta_phi))

    return mesh


def test_pyvista_merge_polylines_create_reference_files_closed(
    get_corresponding_reference_file_path, assert_grids_close
):
    """Create a closed polygon circle test case for the merge polyline
    filter."""

    radius = 2.0
    mat = MaterialBeamBase(radius=0.1)
    mesh = polygon_mesh(Beam2, mat, radius, 10)

    create_line_not_connected_elements(
        mesh, Beam2, mat, [radius, 0, 0], [radius, radius, 0], n_el=1
    )
    create_line_not_connected_elements(
        mesh, Beam2, mat, [-radius, 0, 0], [-2 * radius, 0, 0], n_el=1
    )

    assert_grids_close(
        get_corresponding_reference_file_path(
            reference_file_base_name="pyvista_merge_polylines_closed_raw"
        ),
        get_grid_for_testing(mesh),
    )
