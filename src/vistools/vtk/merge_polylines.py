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
"""Merge lines or polylines with each other that represent a continuous
curve."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import vtk
from beamme.geometric_search.find_close_points import point_partners_to_partner_indices
from beamme.geometric_search.scipy import find_close_points_scipy

from vistools.vtk.vtk_data_structures_utils import vtk_id_to_list


@dataclass
class _MergePolylineData:
    """Common data required to merge polylines."""

    old_cell_tracker: List[Optional[int]]
    partner_list: List[int]
    partner_grouped: List[List[int]]
    smooth_angle: float


@dataclass
class _MergePoint:
    """Structure to hold either one or two point IDs.

    If one ID is present, this point will be taken as is from the
    original mesh. If two IDs are present, the two points will be
    merged.
    """

    index_1: int
    index_2: Optional[int] = None

    # ID of this point in the new grid
    point_id = None


@dataclass
class _PossibleCell:
    """Structure for next possible cells to add to the current merged
    polyline."""

    cell_id: int
    point_id: int
    tangent = None

    def set_cell_tangent(self, grid: vtk.vtkUnstructuredGrid) -> None:
        """Evaluate and set the tangent of this possible next cell.

        Args:
            grid: Grid the cell belongs to.
            cell_id: Global index of the cell.
            point_id: Global index of the point where the tangent should be evaluated.
                Has to be either the start or endpoint of the cell.

        Returns:
            Tangent at the given point.
        """

        cell_point_ids = vtk_id_to_list(grid.GetCell(self.cell_id).GetPointIds())
        self.tangent = _get_indices_tangent(grid, cell_point_ids, self.point_id)


@dataclass
class _MergedPolyline:
    """Structure to hold a single merged polyline."""

    connected_cell_points: List[_MergePoint]
    merged_cell_ids: List[int] = field(default_factory=list)

    @property
    def last_cell_id(self):
        """Return the ID of the last cell added to the merged polyline."""
        return self.merged_cell_ids[-1]

    def check_closed(self, data: _MergePolylineData) -> None:
        """Check if first and last point are the same, either by node ID or by
        position.

        If so, we set the start and end point to be equal to each other,
        so they can the same point in the created grid.
        """

        if (
            self.connected_cell_points[0].index_1
            == self.connected_cell_points[-1].index_1
        ):
            self.connected_cell_points[-1] = self.connected_cell_points[0]
        elif not -1 == (
            partner_id := data.partner_list[self.connected_cell_points[0].index_1]
        ):
            if (
                self.connected_cell_points[-1].index_1
                in data.partner_grouped[partner_id]
            ):
                self.connected_cell_points[0].index_2 = self.connected_cell_points[
                    -1
                ].index_1
                self.connected_cell_points[-1] = self.connected_cell_points[0]


def _smooth_angle_check(
    tangent_1: np.ndarray, tangent_2: np.ndarray, smooth_angle: float
) -> bool:
    """Check if the angle between the two tangents is larger than
    `smooth_angle`.

    The tangents are always outward pointing, thus we get the angle pi
    if the tangents represent a straight polyline.
    """
    dot = np.dot(tangent_1, tangent_2)
    return np.cos(smooth_angle) > dot


def _get_indices_tangent(
    grid: vtk.vtkUnstructuredGrid, connectivity: List[int], point_id: int
) -> np.ndarray:
    """Get the tangent of a polygon defined by the connectivity list.

    Args:
        grid: Grid the cell belongs to.
        connectivity: Connectivity list of the polygon.
        point_id: Global index of the point where the tangent should be evaluated.
            Has to be either the start or endpoint of the polygon.

    Returns:
        Tangent at the given point.
    """

    if connectivity[-1] == point_id:
        tangent_point_indices = [-2, -1]
    elif connectivity[0] == point_id:
        tangent_point_indices = [1, 0]
    else:
        raise ValueError("Given point index does not match the connectivity array")
    points_for_tangent = [
        np.array(grid.GetPoint(connectivity[index])) for index in tangent_point_indices
    ]
    tangent = points_for_tangent[1] - points_for_tangent[0]
    tangent = tangent / np.linalg.norm(tangent)
    return tangent


def _add_next_cell(
    grid: vtk.vtkUnstructuredGrid,
    merge_polyline_data: _MergePolylineData,
    merged_polyline: _MergedPolyline,
    connected_point_id: int,
) -> Optional[int]:
    """Start at the initial point and loop over lines as long as a connectivity
    is found."""

    # Get all possible cells that are next in line.
    # First get the ones that are connected to the current cell via the connectivity entries.
    id_list = vtk.vtkIdList()
    grid.GetPointCells(connected_point_id, id_list)
    cell_connectivity = vtk_id_to_list(id_list)
    possible_next_cells = [
        _PossibleCell(cell_id, connected_point_id)
        for cell_id in cell_connectivity
        if not cell_id == merged_polyline.last_cell_id
    ]
    # Extend the possible connected cells by spatially close ones.
    partner_index = merge_polyline_data.partner_list[connected_point_id]
    if not partner_index == -1:
        partner_points = merge_polyline_data.partner_grouped[partner_index]
        for point_id in partner_points:
            if point_id == connected_point_id:
                continue
            id_list = vtk.vtkIdList()
            grid.GetPointCells(point_id, id_list)
            cell_connectivity = vtk_id_to_list(id_list)
            for cell_id in cell_connectivity:
                possible_next_cells.append(_PossibleCell(cell_id, point_id))

    # Check the angle between this line and all connected cells
    smooth_connected_cells: List[_PossibleCell] = []
    tangent = _get_indices_tangent(
        grid,
        [merge_point.index_1 for merge_point in merged_polyline.connected_cell_points],
        connected_point_id,
    )
    for possible_cell in possible_next_cells:
        possible_cell.set_cell_tangent(grid)
        if _smooth_angle_check(
            tangent, possible_cell.tangent, merge_polyline_data.smooth_angle
        ):
            smooth_connected_cells.append(possible_cell)

    if len(smooth_connected_cells) == 0 or len(smooth_connected_cells) > 1:
        # In this case there are either no or multiple lines connected which are
        # smooth to the given one. There is no unique way to continue this, so we
        # stop here.
        return None
    elif (
        merge_polyline_data.old_cell_tracker[smooth_connected_cells[0].cell_id] is None
    ):
        # The smooth connected cell is already accounted for in the new cells
        return None
    else:
        # We want to continue along the found smooth cell
        new_cell = smooth_connected_cells[0]

        # We have to check that the next cell is not smooth to another cell
        # in possible cells (the last cell is not in possible cells). This can
        # happen for bifurcations. There we could run into issues depending on which
        # search direction we have when reaching the bifurcation.
        for possible_cell in possible_next_cells:
            if not possible_cell.cell_id == new_cell.cell_id:
                if _smooth_angle_check(
                    new_cell.tangent,
                    possible_cell.tangent,
                    merge_polyline_data.smooth_angle,
                ):
                    # This means that at this point there is no unique way to
                    # continue the smooth curve, so we stop.
                    return None

    # Add the new cell and its points (in correct order).
    new_cell_point_ids = [
        _MergePoint(index)
        for index in vtk_id_to_list(grid.GetCell(new_cell.cell_id).GetPointIds())
    ]
    if (
        new_cell_point_ids[0].index_1 == new_cell.point_id
        and connected_point_id == merged_polyline.connected_cell_points[-1].index_1
    ):
        # First point of this cell is added to the last point of the last cell.
        extend = True
    elif (
        new_cell_point_ids[-1].index_1 == new_cell.point_id
        and connected_point_id == merged_polyline.connected_cell_points[-1].index_1
    ):
        # Last point of this cell is added to the last point of the last cell.
        extend = True
        new_cell_point_ids.reverse()
    elif (
        new_cell_point_ids[0].index_1 == new_cell.point_id
        and connected_point_id == merged_polyline.connected_cell_points[0].index_1
    ):
        # First point of this cell is added to the first point of the last cell.
        extend = False
        new_cell_point_ids.reverse()
    elif (
        new_cell_point_ids[-1].index_1 == new_cell.point_id
        and connected_point_id == merged_polyline.connected_cell_points[0].index_1
    ):
        # Last point of this cell is added to the first point of the last cell.
        extend = False
    else:
        raise ValueError("This should not happen")

    # Extend the merged poly line points
    if extend:
        if (
            not merged_polyline.connected_cell_points[-1].index_1
            == new_cell_point_ids[0].index_1
        ):
            merged_polyline.connected_cell_points[-1].index_2 = new_cell_point_ids[
                0
            ].index_1
        merged_polyline.connected_cell_points.extend(new_cell_point_ids[1:])
        next_start_index = -1
    else:
        if (
            not merged_polyline.connected_cell_points[0].index_1
            == new_cell_point_ids[-1].index_1
        ):
            merged_polyline.connected_cell_points[0].index_2 = new_cell_point_ids[
                -1
            ].index_1
        merged_polyline.connected_cell_points = (
            new_cell_point_ids[:-1] + merged_polyline.connected_cell_points
        )
        next_start_index = 0

    merge_polyline_data.old_cell_tracker[new_cell.cell_id] = None
    merged_polyline.merged_cell_ids.append(new_cell.cell_id)
    return new_cell_point_ids[next_start_index].index_1


def _find_next_connected_polyline(
    grid: vtk.vtkUnstructuredGrid, merge_polyline_data: _MergePolylineData
) -> Optional[_MergedPolyline]:
    """Start with the first old cell that was not found yet. Then search all
    cells connected to that one.

    Return all point ids that make up the new poly line. Return None if
    all cells have been found.
    """

    # Take the next available cell and look for all connected cells
    for i in merge_polyline_data.old_cell_tracker:
        if i is not None:
            next_cell_id = i
            break
    else:
        return None

    merge_polyline_data.old_cell_tracker[next_cell_id] = None
    merged_polyline = _MergedPolyline(
        [
            _MergePoint(index)
            for index in vtk_id_to_list(grid.GetCell(next_cell_id).GetPointIds())
        ],
    )
    start_id = merged_polyline.connected_cell_points[0].index_1
    end_id = merged_polyline.connected_cell_points[-1].index_1

    for start_index in [start_id, end_id]:
        merged_polyline.merged_cell_ids.append(next_cell_id)
        next_point_id: Optional[int] = start_index
        while next_point_id is not None:
            next_point_id = _add_next_cell(
                grid,
                merge_polyline_data,
                merged_polyline,
                next_point_id,
            )

    return merged_polyline


def _insert_point_by_index(
    source_grid: vtk.vtkUnstructuredGrid,
    target_grid: vtk.vtkUnstructuredGrid,
    merge_point: _MergePoint,
) -> int:
    """Insert a point into the target grid using one or two point indices
    (defined by the merge point) from the source grid.

    Args:
        source_grid: The grid containing the original points and point data.
        target_grid: The grid to which the new point will be added.
        merge_point: The object containing information about the point to be merged.

    Returns:
        int: The index of the newly inserted point in the target grid.
    """

    if merge_point.point_id is not None:
        return merge_point.point_id

    target_points = target_grid.GetPoints()
    new_point_index = target_points.GetNumberOfPoints()

    # Get coordinates from source
    coords1 = np.array(source_grid.GetPoint(merge_point.index_1))
    if merge_point.index_2 is None:
        new_coords = coords1
    else:
        coords2 = np.array(source_grid.GetPoint(merge_point.index_2))
        new_coords = 0.5 * (coords1 + coords2)

    # Insert point coordinates
    target_points.InsertNextPoint(new_coords)

    # Copy or average point data arrays
    source_point_data = source_grid.GetPointData()
    target_point_data = target_grid.GetPointData()

    for array_index in range(source_point_data.GetNumberOfArrays()):
        source_array = source_point_data.GetArray(array_index)
        target_array = target_point_data.GetArray(array_index)

        if merge_point.index_2 is None:
            data_tuple = source_array.GetTuple(merge_point.index_1)
        else:
            data1 = np.array(source_array.GetTuple(merge_point.index_1))
            data2 = np.array(source_array.GetTuple(merge_point.index_2))
            data_tuple = 0.5 * (data1 + data2)

        target_array.InsertNextTuple(data_tuple)

    merge_point.point_id = new_point_index

    return new_point_index


def merge_polylines(
    grid: vtk.vtkUnstructuredGrid,
    *,
    output_grid: vtk.vtkUnstructuredGrid = None,
    smooth_angle: Optional[float] = None,
    tol: float = 1e-8,
) -> Optional[vtk.vtkUnstructuredGrid]:
    """Merge lines or polylines with each other that represent a continuous
    curve.

    Args:
        grid:
            Input grid (can only contain lines or polylines).
        output_grid:
            If this is given, the merged grid will be stores in this grid. Otherwise,
            a new grid will be returned.
        smooth_angle:
            Threshold for maximum angle between successive segments along a continuous
            line. The angle is calculated between the tangents of the lines which are
            always outward pointing, thus we get the angle pi if the tangents represent
            a straight polyline.
    """

    # Check that all cells are lines or polylines.
    n_cells = grid.GetNumberOfCells()
    for i in range(n_cells):
        cell = grid.GetCell(i)
        if not (isinstance(cell, vtk.vtkLine) or isinstance(cell, vtk.vtkPolyLine)):
            raise ValueError(
                "Only lines (vtk type 3) and poly lines (vtk type 4) are supported. Got {} (vtk type {})".format(
                    type(cell), grid.GetCellType(i)
                )
            )

    if smooth_angle is None:
        smooth_angle = 135.0 * np.pi / 180.0

    if not (output_grid_given := (output_grid is not None)):
        output_grid = vtk.vtkUnstructuredGrid()
    output_grid.Initialize()

    # Prepare the output grid
    output_grid.Initialize()
    output_points = vtk.vtkPoints()
    output_points.SetDataType(grid.GetPoints().GetDataType())
    new_point_data = vtk.vtkPointData()
    for i in range(grid.GetPointData().GetNumberOfArrays()):
        old_array = grid.GetPointData().GetArray(i)
        name = old_array.GetName()
        new_array = old_array.NewInstance()
        new_array.SetName(name)
        new_array.SetNumberOfComponents(old_array.GetNumberOfComponents())
        new_point_data.AddArray(new_array)
    output_grid.SetPoints(output_points)
    output_grid.GetPointData().ShallowCopy(new_point_data)

    # Search matching points
    point_coordinates = np.array(
        [grid.GetPoint(i) for i in range(grid.GetNumberOfPoints())]
    )
    partner_list, n_partners = find_close_points_scipy(point_coordinates, tol)
    partner_grouped = point_partners_to_partner_indices(partner_list, n_partners)

    # Start with the first cell and search all cells connected to that cell and so on.
    # Then do the same with the first cell that was not found and so on.
    merge_polyline_data = _MergePolylineData(
        list(range(n_cells)), partner_list, partner_grouped, smooth_angle
    )
    merged_polylines = []
    while True:
        merged_polyline = _find_next_connected_polyline(grid, merge_polyline_data)
        if merged_polyline is None:
            break
        merged_polyline.check_closed(merge_polyline_data)
        merged_polylines.append(merged_polyline)

    # Create the found poly line
    new_cells = []
    for merged_polyline in merged_polylines:
        connectivity = merged_polyline.connected_cell_points
        new_cell = vtk.vtkPolyLine()
        new_cell.GetPointIds().SetNumberOfIds(len(connectivity))
        for i, merge_point in enumerate(connectivity):
            new_point_index = _insert_point_by_index(grid, output_grid, merge_point)
            new_cell.GetPointIds().SetId(i, new_point_index)
        new_cells.append(new_cell)

    # Output the averaged cell data arrays
    input_cell_data = grid.GetCellData()
    output_cell_data = output_grid.GetCellData()

    for i in range(input_cell_data.GetNumberOfArrays()):
        array_in = input_cell_data.GetArray(i)
        name = array_in.GetName()
        n_components = array_in.GetNumberOfComponents()

        array_out = vtk.vtkDoubleArray()
        array_out.SetName(name)
        array_out.SetNumberOfComponents(n_components)
        array_out.SetNumberOfTuples(len(new_cells))

        for i_cell, merged_polyline in enumerate(merged_polylines):
            # Average components
            values = [
                array_in.GetTuple(cell_id)
                for cell_id in merged_polyline.merged_cell_ids
            ]
            average = np.mean(values, axis=0)
            array_out.SetTuple(i_cell, average)

        output_cell_data.AddArray(array_out)

    # Add all new cells to the output data
    output_grid.Allocate(len(new_cells), 1)
    for cell in new_cells:
        output_grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())

    if not output_grid_given:
        return output_grid
    else:
        return None
