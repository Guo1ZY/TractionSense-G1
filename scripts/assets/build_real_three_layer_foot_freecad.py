#!/usr/bin/env python3
"""Create a confusion-free FreeCAD copy of the real three-part foot stack.

Run with FreeCADCmd/freecadcmd.  The source alignment document is read-only.
The generated document contains exactly these geometric objects, top-to-bottom:

1. rigid robot sole;
2. rigid PCB enclosure, with the PCB represented as contained inside the shell;
3. magnetized TPU layer, with magnets represented as embedded inclusions.

There is deliberately no connector, adapter, spacer, or compliance layer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import FreeCAD as App
import Mesh


SOURCE_OBJECTS = {
    "Unitree_G1_Left_Foot": (
        "Rigid robot sole (top)",
        "rigid_robot_sole",
        "Existing Unitree G1 foot/ankle-roll mesh",
    ),
    "Sensor_PCB_Housing": (
        "Rigid PCB enclosure (PCB inside)",
        "rigid_PCB_enclosure_with_PCB_inside",
        "Rigid enclosure; the PCB is contained inside, not a separate layer",
    ),
    "Sensor_Soft_Insole": (
        "Magnetized TPU layer (magnets embedded, bottom)",
        "magnetized_TPU_with_embedded_magnets",
        "Only deformable layer; four magnets per Hall site are embedded in this material",
    ),
}


def _copy_object(source: object, target_document: object, target_name: str) -> object:
    if source.TypeId == "Mesh::Feature":
        target = target_document.addObject("Mesh::Feature", target_name)
        target.Mesh = source.Mesh.copy()
    elif hasattr(source, "Shape"):
        target = target_document.addObject("PartDesign::Feature", target_name)
        target.Shape = source.Shape.copy()
    else:
        raise TypeError(f"unsupported source type: {source.Name} ({source.TypeId})")
    target.Placement = source.Placement
    return target


def _world_z_bounds(obj: object) -> tuple[float, float]:
    """Return displayed Z bounds, including the object's placement."""
    # FreeCAD's Shape/Mesh BoundBox already includes its Feature placement.
    # Transforming those bounds a second time would double the source pose.
    bounds = obj.Mesh.BoundBox if obj.TypeId == "Mesh::Feature" else obj.Shape.BoundBox
    return bounds.ZMin, bounds.ZMax


def _translate_z(obj: object, delta_z: float) -> None:
    placement = obj.Placement
    placement.Base.z += delta_z
    obj.Placement = placement


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mesh-output",
        type=Path,
        default=None,
        help="optional combined STL for browser/CAD Viewer inspection",
    )
    args = parser.parse_args()

    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_document = App.openDocument(str(source_path))
    found = {obj.Name for obj in source_document.Objects}
    missing = set(SOURCE_OBJECTS) - found
    unexpected_geometry = {
        obj.Name
        for obj in source_document.Objects
        if obj.Name not in SOURCE_OBJECTS and (hasattr(obj, "Shape") or hasattr(obj, "Mesh"))
    }
    if missing or unexpected_geometry:
        raise RuntimeError(
            f"source must contain only the real three-part geometry; missing={sorted(missing)}, "
            f"unexpected_geometry={sorted(unexpected_geometry)}"
        )

    output_document = App.newDocument("RealThreeLayerHallFoot")
    # Insert objects in physical top-to-bottom order so the tree itself shows
    # the stack and cannot be mistaken for the older quick-release assembly.
    ordered_names = (
        "Unitree_G1_Left_Foot",
        "Sensor_PCB_Housing",
        "Sensor_Soft_Insole",
    )
    colors = ((0.22, 0.22, 0.24), (0.12, 0.16, 0.20), (0.10, 0.42, 0.82))
    output_names = ("RigidRobotSole", "RigidPCBEnclosure", "MagnetizedTPULayer")
    copied = []
    for stack_index, (source_name, output_name, color) in enumerate(
        zip(ordered_names, output_names, colors, strict=True), start=1
    ):
        source = source_document.getObject(source_name)
        label, physical_role, description = SOURCE_OBJECTS[source_name]
        target = _copy_object(source, output_document, output_name)
        target.Label = label
        target.addProperty("App::PropertyInteger", "StackOrder", "PhysicalStack")
        target.StackOrder = stack_index
        target.addProperty("App::PropertyString", "PhysicalRole", "PhysicalStack")
        target.PhysicalRole = physical_role
        target.addProperty("App::PropertyString", "Description", "PhysicalStack")
        target.Description = description
        target.addProperty("App::PropertyBool", "IsDeformable", "PhysicalStack")
        target.IsDeformable = source_name == "Sensor_Soft_Insole"
        target.addProperty("App::PropertyString", "InterfaceAbove", "PhysicalStack")
        target.InterfaceAbove = "none" if stack_index == 1 else "direct_contact_no_intermediate_layer"
        # FreeCADCmd has no GUI view provider.  Colors remain optional display
        # metadata and do not affect the physical object tree or geometry.
        if target.ViewObject is not None:
            target.ViewObject.ShapeColor = color
            if source_name == "Sensor_PCB_Housing":
                target.ViewObject.Transparency = 35
            elif source_name == "Sensor_Soft_Insole":
                target.ViewObject.Transparency = 18
        copied.append(target)

    # The source document was positioned for an older quick-release concept
    # and contains vertical assembly gaps.  Close those gaps in the corrected
    # sensor-stack copy: TPU top == PCB bottom, PCB top == robot-sole bottom.
    rigid_sole, pcb_enclosure, magnetized_tpu = copied
    output_document.recompute()
    _, tpu_top = _world_z_bounds(magnetized_tpu)
    pcb_bottom, _ = _world_z_bounds(pcb_enclosure)
    _translate_z(pcb_enclosure, tpu_top - pcb_bottom)
    output_document.recompute()
    # PartDesign can update the feature's active solid bounds on first
    # recompute.  Apply the exact interface constraint once more afterwards.
    pcb_bottom, _ = _world_z_bounds(pcb_enclosure)
    _translate_z(pcb_enclosure, tpu_top - pcb_bottom)
    output_document.recompute()
    _, pcb_top = _world_z_bounds(pcb_enclosure)
    sole_bottom, _ = _world_z_bounds(rigid_sole)
    _translate_z(rigid_sole, pcb_top - sole_bottom)
    output_document.recompute()

    output_document.addObject("App::FeaturePython", "StackDefinition")
    definition = output_document.getObject("StackDefinition")
    definition.Label = "STACK: sole -> PCB enclosure (PCB inside) -> magnetized TPU"
    definition.addProperty("App::PropertyString", "PhysicalStack", "Definition")
    definition.PhysicalStack = (
        "rigid robot sole -> rigid PCB enclosure (PCB inside) -> "
        "magnetized TPU (magnets embedded)"
    )
    definition.addProperty("App::PropertyBool", "HasIntermediateConnectorLayer", "Definition")
    definition.HasIntermediateConnectorLayer = False

    output_document.recompute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_document.saveAs(str(output_path))
    mesh_output = (
        args.mesh_output.expanduser().resolve()
        if args.mesh_output is not None
        else output_path.with_suffix(".stl")
    )
    mesh_output.parent.mkdir(parents=True, exist_ok=True)
    Mesh.export(copied, str(mesh_output))
    App.closeDocument(output_document.Name)
    App.closeDocument(source_document.Name)

    # Re-open the saved artifact so the report reflects FreeCAD's persisted
    # placements rather than a stale pre-save PartDesign bound cache.
    validation_document = App.openDocument(str(output_path))
    validation_objects = [validation_document.getObject(name) for name in output_names]
    world_z_bounds = {obj.Name: _world_z_bounds(obj) for obj in validation_objects}
    sole_bounds = world_z_bounds["RigidRobotSole"]
    pcb_bounds = world_z_bounds["RigidPCBEnclosure"]
    tpu_bounds = world_z_bounds["MagnetizedTPULayer"]
    print(
        {
            "source": str(source_path),
            "output": str(output_path),
            "mesh_output": str(mesh_output),
            "geometry_objects": list(output_names),
            "physical_stack": [SOURCE_OBJECTS[name][1] for name in ordered_names],
            "has_intermediate_connector_layer": False,
            "world_z_bounds_mm": world_z_bounds,
            "interface_gaps_mm": {
                "PCB_to_TPU": pcb_bounds[0] - tpu_bounds[1],
                "sole_to_PCB": sole_bounds[0] - pcb_bounds[1],
            },
        }
    )
    App.closeDocument(validation_document.Name)


if __name__ == "__main__":
    main()
