"""Enable the local-only Robust MCP Bridge in FreeCAD.

Run with FreeCADCmd, not the system Python interpreter.
"""

import FreeCAD


PARAM_PATH = "User parameter:BaseApp/Preferences/Mod/RobustMCPBridge"

preferences = FreeCAD.ParamGet(PARAM_PATH)
preferences.SetBool("AutoStart", True)
preferences.SetBool("StatusBarEnabled", True)
preferences.SetInt("XMLRPCPort", 9875)
preferences.SetInt("SocketPort", 9876)

print("Robust MCP Bridge configured: AutoStart=true, XML-RPC=9875, socket=9876")
