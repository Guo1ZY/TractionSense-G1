"""Load and align the supplied sensor stack and official G1 left foot in FreeCAD.

This script is intended to run inside the FreeCAD GUI through the MCP bridge.
"""

from pathlib import Path

import FreeCAD as App
import FreeCADGui as Gui
import Mesh
import Part


ROOT = Path("/home/mosense/guo_1/vola_sensor")
INPUT_DIR = ROOT / "结构/机器人足底 -pcb外壳和软鞋垫"
OUTPUT_DIR = INPUT_DIR / "设计输出"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

doc = App.newDocument("G1_Foot_Sensor_Input_Review")

housing_path = INPUT_DIR / "足式底壳-安pcbzx-垫高版-9.4mm.STEP"
insole_path = INPUT_DIR / "足式底壳-安pcbzx-垫高版-鞋垫-10mm.STEP"
foot_path = (
    INPUT_DIR
    / "reference/unitree_ros/robots/g1_description/meshes/left_ankle_roll_link.STL"
)

sensor_center_x = (1586.445176 + 1666.484166) / 2.0
sensor_center_y = (-177.438748 + 37.582040) / 2.0
foot_center_x = (-65.841 + 142.365) / 2.0


def sensor_placement():
    # Translate the original SolidWorks coordinates to the local origin, rotate
    # so the sole length follows +X, and align the sensor/foot bounding-box centers.
    placement = App.Placement()
    placement.Base = App.Vector(foot_center_x + sensor_center_y, -sensor_center_x, 0)
    placement.Rotation = App.Rotation(App.Vector(0, 0, 1), 90)
    return placement


housing = doc.addObject("PartDesign::Feature", "Sensor_PCB_Housing")
housing.Label = "Sensor PCB housing (supplied STEP)"
housing.Shape = Part.read(str(housing_path))
housing.Placement = sensor_placement()
housing.ViewObject.ShapeColor = (0.18, 0.46, 0.88)
housing.ViewObject.Transparency = 10

insole = doc.addObject("PartDesign::Feature", "Sensor_Soft_Insole")
insole.Label = "Sensor soft insole (supplied STEP)"
insole.Shape = Part.read(str(insole_path))
insole.Placement = sensor_placement()
insole.ViewObject.ShapeColor = (0.20, 0.78, 0.42)
insole.ViewObject.Transparency = 25

foot_mesh = Mesh.Mesh(str(foot_path))
scale = App.Matrix()
scale.A11 = 1000.0
scale.A22 = 1000.0
scale.A33 = 1000.0
foot_mesh.transform(scale)
foot = doc.addObject("Mesh::Feature", "Unitree_G1_Left_Foot")
foot.Label = "Official Unitree G1 left ankle-roll/foot mesh"
foot.Mesh = foot_mesh
# Put the lowest foot point 7 mm above the housing top for the connector volume.
foot.Placement.Base = App.Vector(0, 0, 10.5001 + 7.0 + 35.409)
foot.ViewObject.ShapeColor = (0.72, 0.72, 0.76)
foot.ViewObject.Transparency = 5

doc.recompute()
doc.saveAs(str(OUTPUT_DIR / "00_输入件与G1左脚对齐.FCStd"))

view = Gui.activeDocument().activeView()
view.setAnimationEnabled(False)
view.viewTop()
view.fitAll()
view.saveImage(str(OUTPUT_DIR / "00_输入对齐_俯视.png"), 1600, 1200, "White")
view.viewRight()
view.fitAll()
view.saveImage(str(OUTPUT_DIR / "00_输入对齐_侧视.png"), 1600, 1200, "White")
view.viewAxonometric()
view.fitAll()
view.saveImage(str(OUTPUT_DIR / "00_输入对齐_轴测.png"), 1600, 1200, "White")

print(f"Saved input review document and snapshots to {OUTPUT_DIR}")
