"""Build a printable quick-release connector and a Unitree G1 29DoF assembly.

Run inside the FreeCAD 1.1 GUI (directly or through the Robust MCP Bridge).
All dimensions are millimetres. The two supplied STEP files remain source
geometry; generated connector parts are exported as FCStd, STEP, and STL.
"""

from math import cos, sin
from pathlib import Path
import json
import re
import xml.etree.ElementTree as ET

import FreeCAD as App
import FreeCADGui as Gui
import Mesh
import MeshPart
import Part


ROOT = Path("/home/mosense/guo_1/vola_sensor")
INPUT_DIR = ROOT / "结构/机器人足底 -pcb外壳和软鞋垫"
OUTPUT_DIR = INPUT_DIR / "设计输出"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOUSING_STEP = INPUT_DIR / "足式底壳-安pcbzx-垫高版-9.4mm.STEP"
INSOLE_STEP = INPUT_DIR / "足式底壳-安pcbzx-垫高版-鞋垫-10mm.STEP"
G1_DIR = INPUT_DIR / "reference/unitree_ros/robots/g1_description"
G1_URDF = G1_DIR / "g1_29dof.urdf"


P = {
    "sensor_bbox_x": (1586.445176, 1666.484166),
    "sensor_bbox_y": (-177.438748, 37.582040),
    "housing_top_z": 10.5001,
    "sensor_bottom_z": -12.4,
    "foot_bbox_x": (-65.841, 142.365),
    "foot_bbox_y": (-37.742, 37.841),
    "foot_bbox_z": (-35.409, 23.584),
    "carrier_thickness": 2.7,
    "snap_clearance": 0.28,
    "snap_wall": 2.0,
    "snap_hook": 1.4,
    "dovetail_bottom_width": 8.0,
    "dovetail_top_width": 12.0,
    "dovetail_height": 3.8,
    "dovetail_clearance": 0.30,
    "shoe_bottom_z": 13.45,
    "shoe_top_z": 17.80,
    "foot_side_clearance": 0.46,
    "wall_thickness": 2.7,
    "wall_height": 8.2,
    "retaining_lip": 3.5,
    "quick_pin_diameter": 5.0,
    "quick_pin_hole": 5.3,
    "heel_pin_diameter": 5.0,
    "heel_pin_hole": 5.3,
}

sensor_center_x = sum(P["sensor_bbox_x"]) / 2.0
sensor_center_y = sum(P["sensor_bbox_y"]) / 2.0
foot_center_x = sum(P["foot_bbox_x"]) / 2.0
foot_center_y = sum(P["foot_bbox_y"]) / 2.0


def sensor_placement(z_offset=0.0):
    placement = App.Placement()
    placement.Base = App.Vector(
        foot_center_x + sensor_center_y,
        foot_center_y - sensor_center_x,
        z_offset,
    )
    placement.Rotation = App.Rotation(App.Vector(0, 0, 1), 90)
    return placement


def add_feature(doc, name, label, shape, color, group=None, transparency=0):
    obj = doc.addObject("PartDesign::Feature", name)
    obj.Label = label
    obj.Shape = shape
    obj.ViewObject.ShapeColor = color
    obj.ViewObject.LineColor = tuple(max(0.0, value - 0.22) for value in color)
    obj.ViewObject.Transparency = transparency
    if group:
        group.addObject(obj)
    return obj


def rounded_box(x_min, y_min, length, width, radius, z_min, height):
    radius = min(radius, length / 2.0, width / 2.0)
    shape = Part.makeBox(
        length - 2 * radius,
        width,
        height,
        App.Vector(x_min + radius, y_min, z_min),
    )
    shape = shape.fuse(
        Part.makeBox(
            length,
            width - 2 * radius,
            height,
            App.Vector(x_min, y_min + radius, z_min),
        )
    )
    for x in (x_min + radius, x_min + length - radius):
        for y in (y_min + radius, y_min + width - radius):
            shape = shape.fuse(
                Part.makeCylinder(radius, height, App.Vector(x, y, z_min))
            )
    return shape.removeSplitter()


def dovetail_prism(x_min, length, center_y, z_min, z_max, bottom_width, top_width):
    points = [
        App.Vector(x_min, center_y - bottom_width / 2.0, z_min),
        App.Vector(x_min, center_y + bottom_width / 2.0, z_min),
        App.Vector(x_min, center_y + top_width / 2.0, z_max),
        App.Vector(x_min, center_y - top_width / 2.0, z_max),
    ]
    wire = Part.makePolygon(points + [points[0]])
    return Part.Face(wire).extrude(App.Vector(length, 0, 0))


def largest_horizontal_face(shape, z_value, tolerance=0.02):
    candidates = []
    for face in shape.Faces:
        bbox = face.BoundBox
        if bbox.ZLength <= tolerance and abs(bbox.ZMin - z_value) <= tolerance:
            candidates.append(face)
    if not candidates:
        raise RuntimeError(f"No horizontal face found at z={z_value}")
    return max(candidates, key=lambda face: face.Area)


def make_carrier(housing_shape, insole_shape):
    # The largest top face of the insole gives a clean, exact supplied sole outline.
    top_face = largest_horizontal_face(insole_shape, -2.4)
    outline = Part.Face(top_face.OuterWire)
    outline.Placement = sensor_placement(P["housing_top_z"] + 2.4)
    plate = outline.extrude(App.Vector(0, 0, P["carrier_thickness"]))

    plate_top = P["housing_top_z"] + P["carrier_thickness"]
    rail_top = plate_top + P["dovetail_height"]
    rails = None
    for center_y in (-20.0, 20.0):
        rail = dovetail_prism(
            -61.5,
            197.5,
            center_y,
            plate_top,
            rail_top,
            P["dovetail_bottom_width"],
            P["dovetail_top_width"],
        )
        rails = rail if rails is None else rails.fuse(rail)

    # Locking tab sits behind the heel, outside the supplied sensor envelope.
    lock_tab = rounded_box(-75.2, -8.0, 12.0, 16.0, 4.0, P["housing_top_z"], 2.7)

    carrier = plate.fuse(rails).fuse(lock_tab)

    # Six flexible snap arms: four on the sides, one at toe, one at heel.
    arm_z = 0.20
    arm_height = plate_top - arm_z
    hook_z = -0.35
    hook_height = 1.65
    side_inner = 40.30
    side_outer = side_inner + P["snap_wall"]
    for clip_x in (-47.0, 101.0):
        for sign in (-1, 1):
            if sign > 0:
                arm_y = side_inner
                hook_y = side_inner - P["snap_hook"]
                hook_width = P["snap_wall"] + P["snap_hook"]
            else:
                arm_y = -side_outer
                hook_y = -side_outer
                hook_width = P["snap_wall"] + P["snap_hook"]
            arm = Part.makeBox(
                20.0,
                P["snap_wall"],
                arm_height,
                App.Vector(clip_x, arm_y, arm_z),
            )
            bridge_y = 30.0 if sign > 0 else -side_outer
            bridge = Part.makeBox(
                20.0,
                side_outer - 30.0,
                P["carrier_thickness"],
                App.Vector(clip_x, bridge_y, P["housing_top_z"]),
            )
            hook = Part.makeBox(
                20.0,
                hook_width,
                hook_height,
                App.Vector(clip_x, hook_y, hook_z),
            )
            carrier = carrier.fuse(bridge).fuse(arm).fuse(hook)

    for is_toe in (False, True):
        if is_toe:
            arm_x = 145.95
            hook_x = arm_x - P["snap_hook"]
        else:
            arm_x = -71.45
            hook_x = arm_x
        arm = Part.makeBox(
            P["snap_wall"],
            20.0,
            arm_height,
            App.Vector(arm_x, -10.0, arm_z),
        )
        hook = Part.makeBox(
            P["snap_wall"] + P["snap_hook"],
            20.0,
            hook_height,
            App.Vector(hook_x, -10.0, hook_z),
        )
        if is_toe:
            bridge_x = 130.0
            bridge_length = arm_x + P["snap_wall"] - bridge_x
        else:
            bridge_x = arm_x
            bridge_length = -55.0 - arm_x
        bridge = Part.makeBox(
            bridge_length,
            20.0,
            P["carrier_thickness"],
            App.Vector(bridge_x, -10.0, P["housing_top_z"]),
        )
        carrier = carrier.fuse(bridge).fuse(arm).fuse(hook)

    quick_hole = Part.makeCylinder(
        P["quick_pin_hole"] / 2.0,
        rail_top - P["housing_top_z"] + 2.0,
        App.Vector(-72.0, 0, P["housing_top_z"] - 1.0),
    )
    return carrier.cut(quick_hole).removeSplitter()


def make_foot_shoe():
    shoe_bottom = P["shoe_bottom_z"]
    shoe_top = P["shoe_top_z"]
    shoe = rounded_box(-69.7, -39.1, 214.0, 78.2, 28.0, shoe_bottom, shoe_top - shoe_bottom)
    shoe = shoe.fuse(
        rounded_box(-75.2, -8.0, 12.0, 16.0, 4.0, shoe_bottom, shoe_top - shoe_bottom)
    )

    clearance = P["dovetail_clearance"]
    for center_y in (-20.0, 20.0):
        channel = dovetail_prism(
            -70.5,
            209.7,
            center_y,
            shoe_bottom - 0.3,
            shoe_top - 0.48,
            P["dovetail_bottom_width"] + 2 * clearance,
            P["dovetail_top_width"] + 2 * clearance,
        )
        shoe = shoe.cut(channel)

    inner_y = max(abs(P["foot_bbox_y"][0]), abs(P["foot_bbox_y"][1])) + P["foot_side_clearance"]
    outer_y = inner_y + P["wall_thickness"]
    wall_x_min = -70.7
    wall_x_max = 145.6
    wall_z_top = shoe_top + P["wall_height"]
    wall_length = wall_x_max - wall_x_min

    # Fully support the wall roots. Without these strips, a rounded base can
    # meet a straight wall only along a face/edge near the toe and generate a
    # non-manifold STL even though the B-rep remains valid.
    support_height = shoe_top - shoe_bottom
    side_support = Part.makeBox(
        wall_length,
        P["wall_thickness"] + 0.4,
        support_height,
        App.Vector(wall_x_min, inner_y - 0.2, shoe_bottom),
    )
    side_support = side_support.fuse(
        Part.makeBox(
            wall_length,
            P["wall_thickness"] + 0.4,
            support_height,
            App.Vector(wall_x_min, -outer_y - 0.2, shoe_bottom),
        )
    )
    shoe = shoe.fuse(side_support)

    side_walls = Part.makeBox(
        wall_length,
        P["wall_thickness"],
        P["wall_height"],
        App.Vector(wall_x_min, inner_y, shoe_top),
    )
    side_walls = side_walls.fuse(
        Part.makeBox(
            wall_length,
            P["wall_thickness"],
            P["wall_height"],
            App.Vector(wall_x_min, -outer_y, shoe_top),
        )
    )
    lip_z = wall_z_top - 2.2
    lip_width = P["wall_thickness"] + P["retaining_lip"]
    side_lips = Part.makeBox(
        wall_length,
        lip_width,
        2.2,
        App.Vector(wall_x_min, inner_y - P["retaining_lip"], lip_z),
    )
    side_lips = side_lips.fuse(
        Part.makeBox(
            wall_length,
            lip_width,
            2.2,
            App.Vector(wall_x_min, -outer_y, lip_z),
        )
    )

    toe_stop = Part.makeBox(
        2.7,
        2 * outer_y,
        P["wall_height"],
        App.Vector(142.9, -outer_y, shoe_top),
    )
    toe_lip = Part.makeBox(
        5.2,
        2 * outer_y,
        2.2,
        App.Vector(140.4, -outer_y, lip_z),
    )
    shoe = shoe.fuse(side_walls).fuse(side_lips).fuse(toe_stop).fuse(toe_lip)

    quick_hole = Part.makeCylinder(
        P["quick_pin_hole"] / 2.0,
        shoe_top - P["housing_top_z"] + 2.0,
        App.Vector(-72.0, 0, P["housing_top_z"] - 1.0),
    )
    shoe = shoe.cut(quick_hole)

    heel_hole = Part.makeCylinder(
        P["heel_pin_hole"] / 2.0,
        2 * outer_y + 4.0,
        App.Vector(-68.6, -outer_y - 2.0, 22.0),
        App.Vector(0, 1, 0),
    )
    return shoe.cut(heel_hole).removeSplitter()


def make_vertical_quick_pin():
    shaft = Part.makeCylinder(
        P["quick_pin_diameter"] / 2.0,
        8.4,
        App.Vector(-72.0, 0, 10.25),
    )
    head = Part.makeCylinder(4.2, 2.0, App.Vector(-72.0, 0, 18.65))
    return shaft.fuse(head)


def make_heel_pin():
    outer_y = max(abs(P["foot_bbox_y"][0]), abs(P["foot_bbox_y"][1])) + P["foot_side_clearance"] + P["wall_thickness"]
    shaft = Part.makeCylinder(
        P["heel_pin_diameter"] / 2.0,
        2 * outer_y + 3.0,
        App.Vector(-68.6, -outer_y - 1.5, 22.0),
        App.Vector(0, 1, 0),
    )
    head = Part.makeCylinder(
        4.3,
        1.5,
        App.Vector(-68.6, -outer_y - 1.8, 22.0),
        App.Vector(0, 1, 0),
    )
    return shaft.fuse(head)


def quat_from_rpy(roll, pitch, yaw):
    cr, sr = cos(roll / 2.0), sin(roll / 2.0)
    cp, sp = cos(pitch / 2.0), sin(pitch / 2.0)
    cy, sy = cos(yaw / 2.0), sin(yaw / 2.0)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def origin_placement(origin):
    if origin is None:
        return App.Placement()
    xyz = [float(value) for value in origin.attrib.get("xyz", "0 0 0").split()]
    rpy = [float(value) for value in origin.attrib.get("rpy", "0 0 0").split()]
    qx, qy, qz, qw = quat_from_rpy(*rpy)
    return App.Placement(
        App.Vector(xyz[0] * 1000.0, xyz[1] * 1000.0, xyz[2] * 1000.0),
        App.Rotation(qx, qy, qz, qw),
    )


def placement_multiply(first, second):
    return first.multiply(second)


def parse_urdf_zero_pose():
    root = ET.parse(G1_URDF).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = []
    child_names = set()
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        child_names.add(child)
        joints.append((parent, child, origin_placement(joint.find("origin"))))
    root_link = next(name for name in links if name not in child_names)
    world = {root_link: App.Placement()}
    pending = list(joints)
    while pending:
        next_pending = []
        progressed = False
        for parent, child, local in pending:
            if parent in world:
                world[child] = placement_multiply(world[parent], local)
                progressed = True
            else:
                next_pending.append((parent, child, local))
        if not progressed:
            raise RuntimeError(f"URDF tree could not be resolved: {next_pending[:3]}")
        pending = next_pending
    return root, links, world


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def add_g1_robot(doc, group, target_left_foot):
    root, links, world = parse_urdf_zero_pose()
    left_foot_world = world["left_ankle_roll_link"]
    root_adjust = placement_multiply(target_left_foot, left_foot_world.inverse())

    material_colors = {}
    for material in root.findall("material"):
        color = material.find("color")
        if color is not None:
            rgba = [float(value) for value in color.attrib["rgba"].split()]
            material_colors[material.attrib["name"]] = tuple(rgba[:3])

    adjusted_world = {}
    for link_name, link in links.items():
        adjusted_world[link_name] = placement_multiply(root_adjust, world[link_name])
        visual = link.find("visual")
        if visual is None:
            continue
        mesh_node = visual.find("./geometry/mesh")
        if mesh_node is None:
            continue
        mesh_path = G1_DIR / mesh_node.attrib["filename"]
        if not mesh_path.exists():
            print(f"Missing G1 mesh: {mesh_path}")
            continue
        visual_mesh = Mesh.Mesh(str(mesh_path))
        scale = App.Matrix()
        scale.A11 = 1000.0
        scale.A22 = 1000.0
        scale.A33 = 1000.0
        visual_mesh.transform(scale)
        obj = doc.addObject("Mesh::Feature", f"G1_{safe_name(link_name)}")
        obj.Label = f"G1 29DoF · {link_name}"
        obj.Mesh = visual_mesh
        obj.addProperty("App::PropertyString", "URDFLink", "Source")
        obj.URDFLink = link_name
        obj.addProperty("App::PropertyString", "SourceMesh", "Source")
        obj.SourceMesh = str(mesh_path)
        obj.Placement = placement_multiply(
            adjusted_world[link_name], origin_placement(visual.find("origin"))
        )
        material = visual.find("material")
        color = material_colors.get(material.attrib.get("name"), (0.65, 0.65, 0.68)) if material is not None else (0.65, 0.65, 0.68)
        obj.ViewObject.ShapeColor = color
        group.addObject(obj)
    return adjusted_world


def duplicate_part(doc, source_obj, name, label, transform, group):
    duplicate = doc.addObject("PartDesign::Feature", name)
    duplicate.Label = label
    duplicate.Shape = source_obj.Shape
    duplicate.Placement = placement_multiply(transform, source_obj.Placement)
    duplicate.ViewObject.ShapeColor = source_obj.ViewObject.ShapeColor
    duplicate.ViewObject.Transparency = source_obj.ViewObject.Transparency
    group.addObject(duplicate)
    return duplicate


def build_detail_document(housing_shape, insole_shape, carrier_shape, shoe_shape, quick_pin_shape, heel_pin_shape):
    old = App.listDocuments().get("G1_Foot_Sensor_Quick_Release")
    if old:
        App.closeDocument(old.Name)
    doc = App.newDocument("G1_Foot_Sensor_Quick_Release")
    source_group = doc.addObject("App::Part", "Supplied_Geometry")
    source_group.Label = " supplied sensor and official G1 foot"
    connector_group = doc.addObject("App::Part", "Quick_Release_Connector")
    connector_group.Label = "Two-stage positive-lock quick release"

    housing = add_feature(
        doc,
        "Sensor_PCB_Housing",
        "Sensor PCB housing · supplied STEP",
        housing_shape,
        (0.16, 0.45, 0.88),
        source_group,
        12,
    )
    housing.Placement = sensor_placement()
    insole = add_feature(
        doc,
        "Sensor_Soft_Insole",
        "Sensor soft insole · supplied STEP",
        insole_shape,
        (0.18, 0.78, 0.40),
        source_group,
        25,
    )
    insole.Placement = sensor_placement()

    foot_path = G1_DIR / "meshes/left_ankle_roll_link.STL"
    foot_mesh = Mesh.Mesh(str(foot_path))
    scale = App.Matrix()
    scale.A11 = scale.A22 = scale.A33 = 1000.0
    foot_mesh.transform(scale)
    foot = doc.addObject("Mesh::Feature", "Unitree_G1_Left_Foot")
    foot.Label = "Unitree G1 left ankle-roll/foot · official mesh"
    foot.Mesh = foot_mesh
    target_left_foot = App.Placement(
        App.Vector(0, 0, P["shoe_top_z"] - P["foot_bbox_z"][0]),
        App.Rotation(),
    )
    foot.Placement = target_left_foot
    foot.ViewObject.ShapeColor = (0.72, 0.72, 0.76)
    foot.ViewObject.Transparency = 8
    source_group.addObject(foot)

    carrier = add_feature(
        doc,
        "Sensor_Carrier",
        "01 Sensor carrier · PA12/PA-CF snap cradle + male dovetails",
        carrier_shape,
        (0.95, 0.55, 0.12),
        connector_group,
    )
    shoe = add_feature(
        doc,
        "Foot_Shoe",
        "02 Foot shoe · captured dovetail + side/toe retention",
        shoe_shape,
        (0.66, 0.24, 0.84),
        connector_group,
    )
    quick_pin = add_feature(
        doc,
        "Quick_Release_Pin",
        "03 Ø5 heel quick-release detent pin",
        quick_pin_shape,
        (0.92, 0.78, 0.12),
        connector_group,
    )
    heel_pin = add_feature(
        doc,
        "Foot_Retaining_Pin",
        "04 Ø5 foot heel retaining detent pin",
        heel_pin_shape,
        (0.90, 0.74, 0.10),
        connector_group,
    )

    for obj in (carrier, shoe):
        obj.addProperty("App::PropertyString", "MaterialRecommendation", "Design")
        obj.MaterialRecommendation = "PA12 SLS preferred; PA-CF FDM for prototype"
        obj.addProperty("App::PropertyString", "DesignClearance", "Design")
        obj.DesignClearance = "Dovetail 0.30 mm/side; foot 0.46 mm/side; sensor snap 0.28 mm"

    doc.recompute()
    doc.saveAs(str(OUTPUT_DIR / "G1足底传感器_快拆连接件_装配.FCStd"))

    export_map = {
        "01_传感器载板_卡扣+燕尾": carrier,
        "02_G1脚套_燕尾槽+防脱边": shoe,
        "03_快拆定位销_直径5mm": quick_pin,
        "04_脚后跟防脱销_直径5mm": heel_pin,
    }
    for basename, obj in export_map.items():
        Part.export([obj], str(OUTPUT_DIR / f"{basename}.step"))
        export_mesh = MeshPart.meshFromShape(
            Shape=obj.Shape,
            LinearDeflection=0.08,
            AngularDeflection=0.35,
            Relative=False,
        )
        export_mesh.removeDuplicatedPoints()
        export_mesh.harmonizeNormals()
        export_mesh.write(str(OUTPUT_DIR / f"{basename}.stl"))

    Part.export([carrier, shoe, quick_pin, heel_pin], str(OUTPUT_DIR / "G1足底传感器_快拆连接件_四件套.step"))

    view = Gui.activeDocument().activeView()
    view.setAnimationEnabled(False)
    view.viewAxonometric()
    view.fitAll()
    view.saveImage(str(OUTPUT_DIR / "01_连接件_装配轴测.png"), 1800, 1400, "White")
    view.viewTop()
    view.fitAll()
    view.saveImage(str(OUTPUT_DIR / "01_连接件_装配俯视.png"), 1800, 1400, "White")
    view.viewRight()
    view.fitAll()
    view.saveImage(str(OUTPUT_DIR / "01_连接件_装配侧视.png"), 1800, 1400, "White")

    # Exploded review state; restore before final save.
    foot.Placement = placement_multiply(App.Placement(App.Vector(0, 0, 32), App.Rotation()), foot.Placement)
    shoe.Placement.Base = App.Vector(0, 0, 17)
    quick_pin.Placement.Base = App.Vector(-16, 0, 12)
    heel_pin.Placement.Base = App.Vector(-12, 0, 28)
    doc.recompute()
    view.viewAxonometric()
    view.fitAll()
    view.saveImage(str(OUTPUT_DIR / "02_连接件_爆炸视图.png"), 1800, 1400, "White")
    foot.Placement = target_left_foot
    shoe.Placement = App.Placement()
    quick_pin.Placement = App.Placement()
    heel_pin.Placement = App.Placement()
    doc.recompute()
    doc.save()
    return doc, (housing, insole, carrier, shoe, quick_pin, heel_pin), target_left_foot


def build_full_robot_document(detail_objects, target_left_foot):
    old = App.listDocuments().get("Unitree_G1_29DoF_Foot_Sensor_Assembly")
    if old:
        App.closeDocument(old.Name)
    doc = App.newDocument("Unitree_G1_29DoF_Foot_Sensor_Assembly")
    robot_group = doc.addObject("App::Part", "Unitree_G1_29DoF")
    robot_group.Label = "Unitree G1 29DoF · official URDF zero pose"
    left_group = doc.addObject("App::Part", "Left_Foot_Sensor")
    left_group.Label = "Left foot · sensor + quick-release connector"
    right_group = doc.addObject("App::Part", "Right_Foot_Sensor")
    right_group.Label = "Right foot · mirrored duplicate for assembly review"

    adjusted_world = add_g1_robot(doc, robot_group, target_left_foot)
    left_foot_world = adjusted_world["left_ankle_roll_link"]
    right_foot_world = adjusted_world["right_ankle_roll_link"]
    right_from_left = placement_multiply(right_foot_world, left_foot_world.inverse())

    names = (
        ("Housing", "Sensor housing"),
        ("Insole", "Soft insole"),
        ("Carrier", "Sensor carrier"),
        ("Shoe", "Foot shoe"),
        ("QuickPin", "Quick-release pin"),
        ("HeelPin", "Foot retaining pin"),
    )
    left_clones = []
    right_clones = []
    for source, (suffix, label) in zip(detail_objects, names):
        left_clones.append(
            duplicate_part(
                doc,
                source,
                f"Left_{suffix}",
                f"Left · {label}",
                App.Placement(),
                left_group,
            )
        )
        right_clones.append(
            duplicate_part(
                doc,
                source,
                f"Right_{suffix}",
                f"Right · {label}",
                right_from_left,
                right_group,
            )
        )

    floor = doc.addObject("PartDesign::Feature", "Reference_Floor")
    floor.Label = "Reference floor at sensor bottom"
    floor.Shape = Part.makeBox(
        600,
        500,
        1.0,
        App.Vector(-260, -250, P["sensor_bottom_z"] - 1.0),
    )
    floor.ViewObject.ShapeColor = (0.82, 0.84, 0.87)
    floor.ViewObject.Transparency = 78

    doc.recompute()
    doc.saveAs(str(OUTPUT_DIR / "Unitree_G1_29DoF_整机+双足传感器+快拆连接件.FCStd"))

    view = Gui.activeDocument().activeView()
    view.setAnimationEnabled(False)
    view.viewFront()
    view.fitAll()
    view.saveImage(str(OUTPUT_DIR / "03_G1整机_正视.png"), 1800, 1600, "White")
    view.viewAxonometric()
    view.fitAll()
    view.saveImage(str(OUTPUT_DIR / "03_G1整机_轴测.png"), 1800, 1600, "White")
    return doc


housing_shape = Part.read(str(HOUSING_STEP))
insole_shape = Part.read(str(INSOLE_STEP))
carrier_shape = make_carrier(housing_shape, insole_shape)
shoe_shape = make_foot_shoe()
quick_pin_shape = make_vertical_quick_pin()
heel_pin_shape = make_heel_pin()

shape_facts = {}
for name, shape in {
    "sensor_carrier": carrier_shape,
    "foot_shoe": shoe_shape,
    "quick_release_pin": quick_pin_shape,
    "foot_retaining_pin": heel_pin_shape,
}.items():
    bbox = shape.BoundBox
    shape_facts[name] = {
        "valid": shape.isValid(),
        "solids": len(shape.Solids),
        "volume_mm3": round(shape.Volume, 3),
        "bbox_min_mm": [round(bbox.XMin, 3), round(bbox.YMin, 3), round(bbox.ZMin, 3)],
        "bbox_size_mm": [round(bbox.XLength, 3), round(bbox.YLength, 3), round(bbox.ZLength, 3)],
    }

detail_doc, detail_objects, target_left_foot = build_detail_document(
    housing_shape,
    insole_shape,
    carrier_shape,
    shoe_shape,
    quick_pin_shape,
    heel_pin_shape,
)
full_doc = build_full_robot_document(detail_objects, target_left_foot)

validation = {
    "freecad_version": ".".join(App.Version()[:3]),
    "source_housing": str(HOUSING_STEP),
    "source_insole": str(INSOLE_STEP),
    "source_g1_urdf": str(G1_URDF),
    "design_parameters_mm": P,
    "generated_shape_facts": shape_facts,
    "dovetail_nominal_clearance_per_side_mm": P["dovetail_clearance"],
    "foot_side_clearance_per_side_mm": P["foot_side_clearance"],
    "sensor_snap_clearance_mm": P["snap_clearance"],
    "notes": [
        "Positive vertical/lateral retention is provided by twin captured dovetails.",
        "Longitudinal rail escape is blocked by a removable 5 mm detent pin.",
        "The G1 foot is captured by side lips, a toe stop, and a rear 5 mm detent pin.",
        "Sensor housing retention uses six flexible perimeter snap arms and must be fit-tested on PA12 prototypes.",
    ],
}
(OUTPUT_DIR / "validation_report.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("BUILD_RESULT_BEGIN")
print(json.dumps(validation, ensure_ascii=False, indent=2))
print("BUILD_RESULT_END")
print(f"Outputs saved in {OUTPUT_DIR}")
