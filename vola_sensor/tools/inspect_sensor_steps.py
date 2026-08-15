"""Print deterministic geometry facts for the two supplied sensor STEP files.

Run with FreeCADCmd from the repository root.
"""

import json
from pathlib import Path

import Part


INPUT_DIR = Path("结构/机器人足底 -pcb外壳和软鞋垫")


def vec(vector):
    return [round(vector.x, 6), round(vector.y, 6), round(vector.z, 6)]


def inspect(step_path):
    shape = Part.read(str(step_path.resolve()))
    bbox = shape.BoundBox
    solids = []
    for index, solid in enumerate(shape.Solids, start=1):
        solid_bbox = solid.BoundBox
        solids.append(
            {
                "index": index,
                "volume_mm3": round(solid.Volume, 3),
                "center_of_mass_mm": vec(solid.CenterOfGravity),
                "bbox_min_mm": [
                    round(solid_bbox.XMin, 6),
                    round(solid_bbox.YMin, 6),
                    round(solid_bbox.ZMin, 6),
                ],
                "bbox_size_mm": [
                    round(solid_bbox.XLength, 6),
                    round(solid_bbox.YLength, 6),
                    round(solid_bbox.ZLength, 6),
                ],
            }
        )
    return {
        "file": str(step_path),
        "shape_type": shape.ShapeType,
        "valid": shape.isValid(),
        "solids_count": len(shape.Solids),
        "shells_count": len(shape.Shells),
        "faces_count": len(shape.Faces),
        "edges_count": len(shape.Edges),
        "bbox_min_mm": [
            round(bbox.XMin, 6),
            round(bbox.YMin, 6),
            round(bbox.ZMin, 6),
        ],
        "bbox_max_mm": [
            round(bbox.XMax, 6),
            round(bbox.YMax, 6),
            round(bbox.ZMax, 6),
        ],
        "bbox_size_mm": [
            round(bbox.XLength, 6),
            round(bbox.YLength, 6),
            round(bbox.ZLength, 6),
        ],
        "volume_mm3": round(shape.Volume, 3),
        "center_of_mass_mm": vec(shape.CenterOfGravity),
        "solids": solids,
    }


result = [inspect(path) for path in sorted(INPUT_DIR.glob("*.STEP"))]
print("SENSOR_STEP_FACTS_BEGIN")
print(json.dumps(result, ensure_ascii=False, indent=2))
print("SENSOR_STEP_FACTS_END")
