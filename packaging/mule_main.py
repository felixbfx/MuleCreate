"""Entry point for the packaged program (PyInstaller)."""
import os
from pathlib import Path
import sys

# PyInstaller relocates USD's library, so its relative plugin search paths no longer find the bundled metadata.
# Configure discovery before importing the app, which imports pxr.
if getattr(sys, "frozen", False):
    plugin_info = Path(sys._MEIPASS) / "pxr" / "pluginfo" / "plugInfo.json"
    if plugin_info.is_file():
        paths = [str(plugin_info)]
        if os.environ.get("PXR_PLUGINPATH_NAME"):
            paths.append(os.environ["PXR_PLUGINPATH_NAME"])
        os.environ["PXR_PLUGINPATH_NAME"] = os.pathsep.join(paths)

from sketch_machine.launch import main

main()
