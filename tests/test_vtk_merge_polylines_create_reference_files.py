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
"""Create the reference files for the merge polylines filter.

Initially taken from pvutils (https://github.com/imcs-compsim/pvutils).
"""

from pathlib import Path
from typing import Optional, Type

import numpy as np
import pyvista as pv

try:
    from meshpy.core.material import MaterialBeam
    from meshpy.core.mesh import Mesh
    from meshpy.core.rotation import Rotation
    from meshpy.four_c.element_beam import Beam3eb, Beam3rHerm2Line3
    from meshpy.four_c.material import MaterialEulerBernoulli, MaterialReissner
    from meshpy.mesh_creation_functions.beam_basic_geometry import (
        create_beam_mesh_line,
    )
except Exception as _:
    MaterialBeam = None
    Mesh = None
    Rotation = None
    Beam3eb = None
    Beam3rHerm2Line3 = None
    MaterialEulerBernoulli = None
    MaterialReissner = None
    create_beam_mesh_line = None


def create_visualization_file(mesh: Mesh, name) -> None:
    """Rename the visualization file created by MeshPy."""

    script_dir = Path(__file__).parent.resolve()
    target_dir = script_dir / "test_files"

    mesh.write_vtk(name, target_dir, binary=False)

    filename_beam = target_dir / f"{name}_beam.vtu"
    filename_new = target_dir / f"{name}.vtu"

    # We add cell data.
    grid = pv.get_reader(filename_beam).read()
    grid.cell_data["cell_data_scalar"] = np.arange(grid.number_of_cells)
    grid.cell_data["cell_data_vector"] = np.arange(3 * grid.number_of_cells).reshape(
        (grid.number_of_cells, 3)
    )
    grid.save(filename_beam, binary=False)

    if filename_beam.exists():
        filename_beam.rename(filename_new)
    else:
        raise FileNotFoundError(f"File {filename_beam} does not exist.")


def create_reference_file_merge_polyline() -> None:
    """Create different test meshes for the merge polyline filter."""

    mesh = Mesh()
    mat = MaterialReissner(radius=0.1)

    n_el = 2

    # Create a single beam.
    z = -2
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, z], [1, 0, z], n_el=1)

    # Create two beams.
    z = -1
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, z], [1, 0, z], n_el=n_el)

    # Create a closed path.
    z = 0
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, z], [1, 0, z], n_el=n_el)
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 0, z], [1, 1, z], n_el=n_el)
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [1, 1, z], [0, 1, z], n_el=n_el)
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 1, z], [0, 0, z], n_el=n_el)

    # Create check all possible connection points.
    z = 1
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, z], [0, 1, z], n_el=n_el)
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 1, z], [0.5, 2, z], n_el=n_el
    )

    z = 2
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, z], [0, 1, z], n_el=n_el)
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0.5, 2, z], [0, 1, z], n_el=n_el
    )

    z = 3
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 1, z], [0, 0, z], n_el=n_el)
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 1, z], [0.5, 2, z], n_el=n_el
    )

    z = 4
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 1, z], [0, 0, z], n_el=n_el)
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0.5, 2, z], [0, 1, z], n_el=n_el
    )

    # Create a simple bifurcation.
    z = 5
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, z], [0, 1, z], n_el=n_el)
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 1, z], [0.5, 2, z], n_el=n_el
    )
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 1, z], [-0.5, 2, z], n_el=n_el
    )

    # Create a bifurcation with a closed circle.
    z = 6
    create_beam_mesh_line(mesh, Beam3rHerm2Line3, mat, [0, 0, z], [0, 1, z], n_el=n_el)
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 1, z], [0.5, 2, z], n_el=n_el
    )
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 1, z], [-0.5, 2, z], n_el=n_el
    )
    create_beam_mesh_line(
        mesh, Beam3rHerm2Line3, mat, [0, 0, z], [-0.5, 2, z], n_el=n_el
    )

    # Create the file for the tests
    create_visualization_file(mesh, "vtk_merge_polylines_raw")


def polygon_mesh(
    beam_type: Type,
    mat: MaterialBeam,
    radius: float,
    segments: int,
    *,
    n_segments: Optional[int] = None,
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
    create_beam_mesh_line(
        mesh_line, beam_type, mat, get_pos(0), get_pos(delta_phi), n_el=1
    )

    for _ in range(n_segments):
        mesh_add = mesh_line.copy()
        mesh.add(mesh_add)
        mesh_line.rotate(Rotation([0, 0, 1], delta_phi))

    return mesh


def create_reference_file_merge_polyline_closed():
    """Create a closed polygon circle test case for the merge polyline
    filter."""

    radius = 2.0
    mat = MaterialEulerBernoulli(radius=0.1)
    mesh = polygon_mesh(Beam3eb, mat, radius, 10)

    create_beam_mesh_line(
        mesh, Beam3eb, mat, [radius, 0, 0], [radius, radius, 0], n_el=1
    )
    create_beam_mesh_line(
        mesh, Beam3eb, mat, [-radius, 0, 0], [-2 * radius, 0, 0], n_el=1
    )

    create_visualization_file(mesh, "vtk_merge_polylines_closed_raw")


if __name__ == "__main__":
    """Execution part of script."""

    create_reference_file_merge_polyline()
    create_reference_file_merge_polyline_closed()
