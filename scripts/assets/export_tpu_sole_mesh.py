#!/usr/bin/env python3
"""Export the measured 10 mm magnetized TPU STEP as a metre-scaled STL.

Run this script with ``freecadcmd``.  The source STEP is kept as the design
authority.  The derived STL is centered at the origin and uses the canonical
foot axes expected by the Hall model: +X toe, +Y robot-left, +Z upward.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import FreeCAD as App
import Mesh
import Part


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    shape = Part.read(str(source))
    if shape.isNull() or not shape.isValid() or len(shape.Solids) != 1:
        raise RuntimeError("TPU STEP must contain one valid closed solid")

    # The measured STEP is X=width, Y=heel-to-toe, Z=thickness and carries a
    # large CAD assembly translation.  Recenter first, then map source +Y
    # (the broad toe end) to canonical +X with a -90 degree rotation about Z.
    bounds = shape.BoundBox
    center = App.Vector(
        0.5 * (bounds.XMin + bounds.XMax),
        0.5 * (bounds.YMin + bounds.YMax),
        0.5 * (bounds.ZMin + bounds.ZMax),
    )
    shape.translate(-center)
    shape.rotate(App.Vector(0.0, 0.0, 0.0), App.Vector(0.0, 0.0, 1.0), -90.0)
    scale = App.Matrix()
    scale.scale(App.Vector(1.0e-3, 1.0e-3, 1.0e-3))
    shape = shape.transformGeometry(scale)

    output.parent.mkdir(parents=True, exist_ok=True)
    document = App.newDocument("tpu_sole_export")
    feature = document.addObject("Part::Feature", "TPUSoleA40Grid35")
    feature.Label = "Magnetized TPU layer, effective Shore A40, Grid 35 percent"
    feature.Shape = shape
    document.recompute()
    Mesh.export([feature], str(output))

    final_bounds = shape.BoundBox
    print(
        {
            "source": str(source),
            "output": str(output),
            "solid_count": len(shape.Solids),
            "valid": shape.isValid(),
            "closed": shape.isClosed(),
            "volume_m3": shape.Volume,
            "bbox_m": [final_bounds.XLength, final_bounds.YLength, final_bounds.ZLength],
            "axis_convention": "+X toe, +Y left, +Z up",
        }
    )


if __name__ == "__main__":
    main()
