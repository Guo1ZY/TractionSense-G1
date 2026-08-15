"""Validate generated connector B-reps, meshes, and assembled clearances."""

from pathlib import Path
import json

import FreeCAD as App
import Mesh
import Part


ROOT = Path("/home/mosense/guo_1/vola_sensor")
OUTPUT_DIR = ROOT / "结构/机器人足底 -pcb外壳和软鞋垫/设计输出"
ASSEMBLY = OUTPUT_DIR / "G1足底传感器_快拆连接件_装配.FCStd"

doc = App.openDocument(str(ASSEMBLY))
carrier_obj = doc.getObject("Sensor_Carrier")
shoe_obj = doc.getObject("Foot_Shoe")
quick_pin_obj = doc.getObject("Quick_Release_Pin")
heel_pin_obj = doc.getObject("Foot_Retaining_Pin")


def located_shape(obj):
    shape = obj.Shape.copy()
    shape.Placement = obj.Placement
    return shape


carrier = located_shape(carrier_obj)
shoe = located_shape(shoe_obj)
quick_pin = located_shape(quick_pin_obj)
heel_pin = located_shape(heel_pin_obj)

checks = {
    "assembly_file": str(ASSEMBLY),
    "parts": {},
    "assembled_interference_mm3": {
        "carrier_vs_foot_shoe": round(carrier.common(shoe).Volume, 6),
        "quick_pin_vs_carrier_and_shoe": round(
            quick_pin.common(carrier.fuse(shoe)).Volume, 6
        ),
        "heel_pin_vs_foot_shoe": round(heel_pin.common(shoe).Volume, 6),
    },
}

for step_path in sorted(OUTPUT_DIR.glob("0[1-4]_*.step")):
    shape = Part.read(str(step_path))
    checks["parts"][step_path.name] = {
        "format": "STEP",
        "valid": shape.isValid(),
        "solids": len(shape.Solids),
        "shells": len(shape.Shells),
        "volume_mm3": round(shape.Volume, 3),
    }

for stl_path in sorted(OUTPUT_DIR.glob("0[1-4]_*.stl")):
    mesh = Mesh.Mesh(str(stl_path))
    def mesh_metric(name):
        metric = getattr(mesh, name, None)
        return metric() if callable(metric) else None

    checks["parts"][stl_path.name] = {
        "format": "STL",
        "facets": mesh.CountFacets,
        "solid": mesh.isSolid(),
        "components": mesh.countComponents(),
        "has_non_manifolds": mesh_metric("hasNonManifolds"),
        "has_non_uniform_facets": mesh_metric("hasNonUniformOrientedFacets"),
        "duplicated_facets": mesh_metric("countDuplicatedFacets"),
        "degenerated_facets": mesh_metric("countDegeneratedFacets"),
        "non_uniform_facets": mesh_metric("countNonUniformOrientedFacets"),
    }
checks["pass"] = (
    all(
        part.get("valid", part.get("solid", False))
        for part in checks["parts"].values()
    )
    and all(
        part.get("solids", 1) == 1
        for part in checks["parts"].values()
        if part["format"] == "STEP"
    )
    and checks["assembled_interference_mm3"]["carrier_vs_foot_shoe"] < 0.01
    and checks["assembled_interference_mm3"]["quick_pin_vs_carrier_and_shoe"] < 0.01
    and checks["assembled_interference_mm3"]["heel_pin_vs_foot_shoe"] < 0.01
)

(OUTPUT_DIR / "export_validation_report.json").write_text(
    json.dumps(checks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("CONNECTOR_EXPORT_VALIDATION_BEGIN")
print(json.dumps(checks, ensure_ascii=False, indent=2))
print("CONNECTOR_EXPORT_VALIDATION_END")
App.closeDocument(doc.Name)
