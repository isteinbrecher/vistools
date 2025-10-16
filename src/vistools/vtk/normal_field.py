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
"""Compute normal fields on surfaces."""

import numpy as np
import vtk
from scipy.spatial import KDTree
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def add_normal_field(
    shell: vtk.vtkUnstructuredGrid,
    *,
    nonlinear_subdivision_level: int = 1,
    tolerance: float = 1e-6,
) -> None:
    """Append a point data field to the input grid that contains the surface
    normals on the points.

    Note:
     - The displacement already has to be applied to the grid.
     - The surface normals can only be computed with single precision.
    """

    def extract_surface(input, nonlinear_subdivision_level=None):
        """Apply the extract surface filter."""
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(input)
        if nonlinear_subdivision_level is not None:
            surface_filter.SetNonlinearSubdivisionLevel(nonlinear_subdivision_level)
        surface_filter.Update()
        return surface_filter.GetOutput()

    def compute_normals(input):
        """Compute the normals on the input surface."""
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(input)
        normals_filter.ComputePointNormalsOn()
        normals_filter.ComputeCellNormalsOff()
        normals_filter.Update()
        return normals_filter.GetOutput()

    if nonlinear_subdivision_level == 1:
        # In this case the filtered number of points will be the same as
        # the input and we don't have to search for the point mapping.

        # Extract surface.
        surface = extract_surface(shell)

        # Compute normals.
        surface_normals = compute_normals(surface)

        # Get the normal array
        normal_array = vtk_to_numpy(surface_normals.GetPointData().GetArray("Normals"))

        # Add the normals to the input data.
        vtk_selected = numpy_to_vtk(normal_array)
        vtk_selected.SetName("surface_normals")
        shell.GetPointData().AddArray(vtk_selected)
    else:
        # In this case we refine the surface, which results in a different number
        # of points. We have to find the mapping between the original points and
        # the refined points. The problem is that there can be multiple points at
        # the same spatial location (e.g. at edges). We use the connectivity
        # filter to assign unique ids to the points that indicate which face they
        # belong to. This way we can uniquely identify the points.

        # Add the node ids.
        shell_with_ids = vtk.vtkUnstructuredGrid()
        shell_with_ids.DeepCopy(shell)
        n_points = shell.GetNumberOfPoints()
        point_ids = np.arange(n_points, dtype=np.float64)
        vtk_point_ids = numpy_to_vtk(point_ids, deep=True)
        vtk_point_ids.SetName("point_ids")
        shell_with_ids.GetPointData().AddArray(vtk_point_ids)

        # Get the connectivity of the input mesh.
        connectivity = vtk.vtkConnectivityFilter()
        connectivity.SetInputData(shell_with_ids)
        connectivity.SetExtractionModeToAllRegions()
        connectivity.ColorRegionsOn()  # This adds a cell array named "RegionId"
        connectivity.Update()
        shell_with_ids_connectivity = connectivity.GetOutput()

        # Extract surface.
        surface = extract_surface(
            shell_with_ids_connectivity,
            nonlinear_subdivision_level=nonlinear_subdivision_level,
        )

        # Compute normals.
        surface_normals = compute_normals(surface)

        # Map the normals to the original mesh.
        def get_extended_coordinates(grid):
            """Get the coordinates extended with connectivity as a single
            array."""
            coordinates = vtk_to_numpy(grid.GetPoints().GetData())
            connectivity = vtk_to_numpy(grid.GetPointData().GetArray("RegionId"))
            extended_coordinates = np.zeros((len(coordinates), 4))
            extended_coordinates[:, :3] = coordinates
            extended_coordinates[:, 3] = connectivity
            return extended_coordinates

        kd_tree = KDTree(get_extended_coordinates(surface_normals))
        distances, indices = kd_tree.query(
            get_extended_coordinates(shell_with_ids_connectivity), k=2
        )
        # The "closest" point has to be within the tolerance, the next closest one outside.
        # This means that we have unique matching points.
        if np.max(distances[:, 0]) > tolerance:
            raise ValueError("Not all closest points are within the tolerance.")
        if np.min(distances[:, 1]) < tolerance:
            raise ValueError("The closest points are not unique.")

        # Get the mapping to the original points.
        point_id_mapping_float = vtk_to_numpy(
            surface_normals.GetPointData().GetArray("point_ids")
        )[indices[:, 0]]
        point_id_mapping = np.round(point_id_mapping_float).astype(np.int64)
        if not np.all(np.isclose(point_id_mapping_float, point_id_mapping)):
            raise ValueError("Can not handle floating point indices.")
        point_id_mapping_inverse = [0] * len(point_id_mapping)
        for i, val in enumerate(point_id_mapping):
            point_id_mapping_inverse[val] = i

        # Add the normals to the input data.
        normal_on_shell_np = vtk_to_numpy(
            surface_normals.GetPointData().GetArray("Normals")
        )[indices[:, 0]][point_id_mapping_inverse]
        normal_on_shell = numpy_to_vtk(normal_on_shell_np)
        normal_on_shell.SetName("surface_normals")
        shell.GetPointData().AddArray(normal_on_shell)
