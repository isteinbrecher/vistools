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
"""Testing framework infrastructure."""

import os
from pathlib import Path
from typing import Callable, Optional, Union

import pytest
import pyvista as pv
import vtk

from vistools.vtk.compare_grids import compare_grids


@pytest.fixture(scope="session")
def test_file_directory() -> Path:
    """Provide the path to the test file directory.

    Returns:
        Path: A Path object representing the full path to the test file directory.
    """

    testing_path = Path(__file__).resolve().parent
    return testing_path / "test_files"


@pytest.fixture(scope="function")
def current_test_name(request: pytest.FixtureRequest) -> str:
    """Return the name of the current pytest test.

    Args:
        request: The pytest request object.

    Returns:
        str: The name of the current pytest test.
    """

    return request.node.originalname


@pytest.fixture(scope="function")
def current_test_name_no_prefix(current_test_name) -> str:
    """Return the name of the current pytest test without the leading "test_".

    Args:
        request: The pytest request object.

    Returns:
        str: The name of the current pytest test without the leading "test_".
    """

    split = current_test_name.split("test_")
    if len(split) > 2:
        raise ValueError("Split is not unique")
    return split[1]


@pytest.fixture(scope="function")
def get_corresponding_reference_file_path(
    test_file_directory, current_test_name_no_prefix
) -> Callable:
    """Return function to get path to corresponding reference file for each
    test.

    Necessary to enable the function call through pytest fixtures.
    """

    def _get_corresponding_reference_file_path(
        reference_file_base_name: Optional[str] = None,
        additional_identifier: Optional[str] = None,
        extension: str = "vtu",
    ) -> Path:
        """Get path to corresponding reference file for each test. Also check
        if this file exists. Basename, additional identifier and extension can
        be adjusted.

        Args:
            reference_file_base_name: Basename of reference file, if none is
                provided the current test name is utilized
            additional_identifier: Additional identifier for reference file, by default none
            extension: Extension of reference file, by default ".vtu"

        Returns:
            Path to reference file.
        """

        corresponding_reference_file = (
            reference_file_base_name or current_test_name_no_prefix
        )

        if additional_identifier:
            corresponding_reference_file += f"_{additional_identifier}"

        corresponding_reference_file += "." + extension

        corresponding_reference_file_path = (
            test_file_directory / corresponding_reference_file
        )

        if not os.path.isfile(corresponding_reference_file_path):
            raise AssertionError(
                f"File path: {corresponding_reference_file_path} does not exist"
            )

        return corresponding_reference_file_path

    return _get_corresponding_reference_file_path


@pytest.fixture(scope="function")
def assert_results_equal() -> Callable:
    """Return function to compare vtu grids.

    Necessary to enable the function call through pytest fixtures.

    Returns:
        Function to compare results.
    """

    def _assert_results_equal(
        reference: Union[Path, pv.UnstructuredGrid, vtk.vtkUnstructuredGrid],
        result: Union[Path, pv.UnstructuredGrid, vtk.vtkUnstructuredGrid],
        rtol: Optional[float] = 1e-10,
        atol: Optional[float] = 1e-10,
    ) -> None:
        """Comparison between reference and result with relative or absolute
        tolerance.

        If the comparison fails, an assertion is raised.

        Args:
            reference: The reference data.
            result: The result data.
            rtol: The relative tolerance.
            atol: The absolute tolerance.
        """

        def _get_grid(data) -> pv.UnstructuredGrid:
            """Return the grid, if the data is a path, load the file."""
            if isinstance(data, Path):
                return pv.get_reader(data).read()
            else:
                return data

        grid_1 = _get_grid(reference)
        grid_2 = _get_grid(result)
        result = compare_grids(grid_1, grid_2, output=True, rtol=rtol, atol=atol)

        if not isinstance(result, bool):
            if not result[0]:
                raise AssertionError("\n".join(result[1]))
        else:
            assert False, "Got unexpected return variable from 'compare_grids'"

    return _assert_results_equal


@pytest.fixture(scope="session")
def assert_results_equal_single_precision_tol() -> dict:
    """Return comparison tolerances for values computed with single precision.

    Returns:
        dict: Dictionary with keys 'rtol' and 'atol'.
    """
    return {"rtol": 1e-6, "atol": 1e-6}
