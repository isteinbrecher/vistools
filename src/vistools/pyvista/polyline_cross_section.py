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
"""Extrude a profile along a polyline."""

import pyvista as pv

from vistools.vtk.polyline_cross_section import (
    polyline_cross_section as vtk_polyline_cross_section,
)


def polyline_cross_section(
    grid: pv.PolyData,
    cross_section_points,
    *,
    closed: bool = True,
    separate_surfaces: bool = False,
) -> pv.UnstructuredGrid:
    """Extrude a profile defined by the cross section coordinates along a
    polyline.

    This function calls the vtk version under the hood, but adds some additional
    functionality.

    Args
    ----
    grid:
        Polyline defining the centerline of the extruded structure
    cross_section_points:
        In-cross-section coordinates defining the profile
    closed:
        Flag if the profile is open or closed
    separate_surfaces:
        Per default all surface cells for a single polyline extrusion are connected.
        If this is true, then the connectivity of the returned surfaces is modified.
        The polygons at the start and end (for closed==True) will not be connected to
        the rest. For the extruded surfaces, each segment between two points in
        cross_section_points will be extruded along the centerline and not be connected
        to the bounding surfaces. This can be useful if sharp edges should be shown in
        the visualization.
    """

    cross_section_grid = pv.UnstructuredGrid(
        vtk_polyline_cross_section(grid, cross_section_points, closed=closed)
    )

    if not separate_surfaces:
        return cross_section_grid
    else:
        surface_ids = cross_section_grid.celltypes != pv.CellType.POLYGON
        grid_surfaces = cross_section_grid.extract_cells(surface_ids)
        if closed:
            n_surfaces = len(cross_section_points)
        else:
            n_surfaces = len(cross_section_points) - 1
        n_cells = grid_surfaces.n_cells
        surfaces = [
            grid_surfaces.extract_cells(range(i_start, n_cells, n_surfaces))
            for i_start in range(n_surfaces)
        ]

        if closed:
            bound_ids = cross_section_grid.celltypes == pv.CellType.POLYGON
            grid_bound = cross_section_grid.extract_cells(bound_ids)
            surfaces.append(grid_bound)

        return pv.merge(surfaces, merge_points=False)
