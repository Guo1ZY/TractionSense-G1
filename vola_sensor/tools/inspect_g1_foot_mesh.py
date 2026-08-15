"""Inspect the official Unitree G1 ankle-roll/foot STL meshes."""

import json
from pathlib import Path

import Mesh


MESH_DIR = Path(
    "结构/机器人足底 -pcb外壳和软鞋垫/reference/"
    "unitree_ros/robots/g1_description/meshes"
)

facts = []
for side in ("left", "right"):
    mesh_path = MESH_DIR / f"{side}_ankle_roll_link.STL"
    mesh = Mesh.Mesh(str(mesh_path.resolve()))
    bbox = mesh.BoundBox
    facts.append(
        {
            "file": str(mesh_path),
            "facets": mesh.CountFacets,
            "bbox_min_m": [
                round(bbox.XMin, 6),
                round(bbox.YMin, 6),
                round(bbox.ZMin, 6),
            ],
            "bbox_max_m": [
                round(bbox.XMax, 6),
                round(bbox.YMax, 6),
                round(bbox.ZMax, 6),
            ],
            "bbox_size_mm": [
                round(bbox.XLength * 1000, 3),
                round(bbox.YLength * 1000, 3),
                round(bbox.ZLength * 1000, 3),
            ],
        }
    )

print("G1_FOOT_MESH_FACTS_BEGIN")
print(json.dumps(facts, ensure_ascii=False, indent=2))
print("G1_FOOT_MESH_FACTS_END")
