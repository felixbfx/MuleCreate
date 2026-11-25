# PyInstaller: Mule as a folder with an executable. Build in the repo root:
#   .venv/Scripts/python.exe -m PyInstaller packaging/mule.spec --noconfirm
# Result: dist/Mule. Check: dist/Mule/Mule.exe --selftest result.json
# PyInstaller only builds for the system it runs on (Windows, Linux, Mac each separately).
import os
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(SPECPATH, "mule.ico")  # from the last frame of the logo animation (with lettering)

datas = [(os.path.join(ROOT, "bible", "sketch_machine_bible.md"), "bible"),
         (os.path.join(ROOT, "sketch_machine", "assets", "mule_start.webp"), os.path.join("sketch_machine", "assets")),
         (os.path.join(ROOT, "sketch_machine", "assets", "mule_icon.png"), os.path.join("sketch_machine", "assets")),
         # reference objects (File > Insert reference object): person 1.80 m, car, house
         *[(os.path.join(ROOT, "sketch_machine", "assets", "reference", f), os.path.join("sketch_machine", "assets", "reference"))
           for f in ("person_1m80.usda", "car.usda", "house.usda")],
         # Review: runs as its own program in the Python with faster-whisper, so it must be bundled as a .py file
         (os.path.join(ROOT, "sketch_machine", "review_transcribe.py"), "sketch_machine")]
# Working-method package of the Sketch Brief (rules, roles, hooks, visibility map) – the export copies it into the brief folder
for base, _dirs, files in os.walk(os.path.join(ROOT, "sketch_machine", "brief_package")):
    if os.sep + "tests" in base or "__pycache__" in base:
        continue
    datas += [(os.path.join(base, f), os.path.relpath(base, ROOT)) for f in files]
binaries = []
hidden = collect_submodules("sketch_machine") + collect_submodules("glcontext") + collect_submodules("moderngl")
for package in ("pxr",):  # USD finds its plugins via plugInfo.json next to the libraries
    d, b, h = collect_all(package)
    datas, binaries, hidden = datas + d, binaries + b, hidden + h

a = Analysis([os.path.join(SPECPATH, "mule_main.py")], pathex=[ROOT], datas=datas, binaries=binaries, hiddenimports=hidden,
             excludes=["tkinter", "tests", "matplotlib", "IPython", "PySide6.scripts"])
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Mule", console=False,
          icon=ICON if os.path.exists(ICON) else None)
coll = COLLECT(exe, a.binaries, a.datas, name="Mule")
sys.path.insert(0, SPECPATH)
import third_party  # noqa: E402 – licenses of third-party software next to Mule.exe (aborts if a text is missing)
third_party.write(os.path.join(DISTPATH, "Mule"), [e[0] for e in a.pure], [e[0] for e in a.binaries + a.datas])
if sys.platform == "darwin":
    app = BUNDLE(coll, name="Mule.app", icon=None, bundle_identifier="com.felixbfx.mule")
