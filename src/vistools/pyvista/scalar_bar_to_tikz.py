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
"""Create a screenshot and the TikZ code to create the scalar bar."""

import os
from pathlib import Path

import pyvista as pv

# Inch to centimeter conversion
INCH = 2.54


def _dots_to_cm(pt, dpi):
    """Convert a distance in pt to cm."""
    return float(pt) / dpi * INCH


def _hide_scalar_bar(scalar_bar):
    """Hide the scalar bar and return the data needed to reset it."""
    data = {"title": scalar_bar.GetTitle()}
    scalar_bar.SetTitle("")
    scalar_bar.DrawTickLabelsOff()
    return data


def _show_scalar_bar(scalar_bar, data):
    """Show the scalar bar and reset the options."""
    scalar_bar.SetTitle(data["title"])
    scalar_bar.DrawTickLabelsOn()


def _get_scalar_bar_rectangles(plotter, scalar_bars, dpi):
    """Get the position of the TikZ color bars.

    We return the distance in cm where the scalar bar starts and the width
    and the height:
        [pos_x, pos_y, width, height]
    """

    rectangles = []

    for scalar_bar in scalar_bars:
        rectangle_pixel = [0, 0, 0, 0]
        scalar_bar.GetScalarBarRect(rectangle_pixel, plotter.renderer)
        rectangles.append([_dots_to_cm(val, dpi) for val in rectangle_pixel])

    return rectangles


def _get_tikz_data(scalar_bar):
    """Return the data required for TikZ for this scalar bar."""

    data = {}

    min_max = list(scalar_bar.GetLookupTable().GetRange())
    data["min_max"] = min_max

    ticks = []
    ticks.extend(min_max)
    ticks.sort()
    data["ticks"] = ticks

    data["title"] = scalar_bar.GetTitle()

    data["orientation"] = "v" if scalar_bar.GetOrientation() else "h"

    return data


def _get_tikz_string_continuous(rectangle, data, number_format):
    """Get the TikZ code for a continuous color bar."""

    # Add the code that is valid for all types of labels.
    tikz_code = [
        "\\begin{axis}[",
        "scale only axis,",
        "scaled x ticks=false,",
        "scaled y ticks=false,",
        f"at={{({rectangle[0]}cm,{rectangle[1]}cm)}},",
        "tick label style={font=\\footnotesize},",
        f"title={data['title']},",
        f"xticklabel={number_format},",
        f"yticklabel={number_format},",
        f"xmin={data['min_max'][0]},",
        f"xmax={data['min_max'][1]},",
        f"ymin={data['min_max'][0]},",
        f"ymax={data['min_max'][1]},",
        f"width={rectangle[2]}cm,",
        f"height={rectangle[3]}cm,",
    ]

    ticks_string = ",".join(map(str, data["ticks"]))

    if data["orientation"] == "v":
        tikz_code += [
            "xtick=\\empty,",
            f"ytick={{{ticks_string}}},",
            "ytick pos=right,",
            "ytick align=outside,",
        ]
    elif data["orientation"] == "h":
        tikz_code += [
            "ytick=\\empty,",
            f"xtick={{{ticks_string}}},",
            "xtick pos=right,",
            "xtick align=outside,",
            "title style={{yshift=10pt,}},",
        ]
    else:
        raise ValueError("Got unexpected value for orientation")

    tikz_code += ["]", "\\end{axis}"]
    return tikz_code


def export_to_tikz(
    name_or_path: Path,
    plotter: pv.Plotter,
    *,
    dpi: int = 300,
    figure_path: str = "",
    number_format: str = "{$\\pgfmathprintnumber[sci,precision=1,sci generic={mantissa sep=,exponent={\\mathrm{e}{##1}}}]{\\tick}$}",
    is_testing: bool = False,
):
    """Export a screenshot and also export TikZ code that overlays a TikZ
    scalar bar.

    We export the image with the raw scalar bar, i.e., no labels, ticks, title, etc..
    We also export TikZ code that can import this image and overlay a TikZ scalar bar
    such that the numbering of the scalar bar can be done directly within LaTeX.

    Args:
        name_or_path: Name of the output files. Can include directory paths.
        plotter: Pyvista plotter
        dpi: Desired resolution of image in dots per inch.
        figure_path: Relative path to figures in the TeX document structure. Default is empty.
        number_format: Number format to be used for the TeX ticks.
    """

    # Name of image and TikZ file.
    path_to_name = Path(name_or_path)
    base_dir = path_to_name.parent
    name = path_to_name.name
    image_name = name + ".png"
    image_path = base_dir / image_name
    tikz_path = base_dir / (name + ".tex")

    # Get all scalar bars in the plotter
    scalar_bars = [plotter.scalar_bars[key] for key in list(plotter.scalar_bars.keys())]

    # Hide the stuff we don't want in the image
    original_data = [_hide_scalar_bar(scalar_bar) for scalar_bar in scalar_bars]

    # Save the screenshot
    if not is_testing:
        plotter.screenshot(image_path)
    else:
        plotter.screenshot(None, return_img=True)

    # Reset the scalar bar properties
    for scalar_bar, data in zip(scalar_bars, original_data):
        _show_scalar_bar(scalar_bar, data)

    # Set the header of te TikZ code
    tikz_code = [
        "%% This file was created with vistools",
        "%% Use the following includes in the LaTeX header:",
        "%\\usepackage{tikz}",
        "%\\usepackage{pgfplots}",
        "%% Optional:",
        "%\\pgfplotsset{compat=1.16}",
        "%\\normalsize % Sometimes in figure environments a smaller font is selected -> uncomment this to activate the default font size",
        "\\begin{tikzpicture}",
        "% The -0.2pt here are needed, so the image is really placed at the origin.",
        f"\\node[anchor=south west,inner sep=-0.2pt] (image) at (0,0) {{\\includegraphics[scale={72.0 / dpi}]{{{os.path.join(figure_path, image_name)}}}}};",
    ]

    # Get the positions of the scalar bar
    rectangles = _get_scalar_bar_rectangles(plotter, scalar_bars, dpi)

    # Get the data we need for the TikZ plot from the scalar bar
    tikz_data_list = [_get_tikz_data(scalar_bar) for scalar_bar in scalar_bars]

    # Add the TikZ code for each color bar
    for rectangle, data in zip(rectangles, tikz_data_list):
        tikz_code.extend(_get_tikz_string_continuous(rectangle, data, number_format))

    # Finalize the TikZ code
    tikz_code.append("\\end{tikzpicture}%\n%}%")

    # Write TikZ code to file.
    with open(tikz_path, "w") as text_file:
        text_file.write("\n".join(tikz_code))
