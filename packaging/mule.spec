# PyInstaller: Mule as a folder with an executable. Build in the repo root:
#   .venv/Scripts/python.exe -m PyInstaller packaging/mule.spec --noconfirm     (Windows)
#   .venv/bin/python -m PyInstaller packaging/mule.spec --noconfirm             (macOS, Linux)
# Result: dist/Mule (Windows/Linux), dist/Mule.app on macOS (unsigned, arm64 = the Python it is built with).
# Check: dist/Mule/Mule.exe --selftest result.json – macOS: dist/Mule.app/Contents/MacOS/Mule --selftest result.json
# PyInstaller only builds for the system it runs on (Windows, Linux, Mac each separately).
import os
import shutil
import subprocess
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
DARWIN = sys.platform == "darwin"
ICON = os.path.join(SPECPATH, "mule.icns" if DARWIN else "mule.ico")  # from the last frame of the logo animation (with lettering)
sys.path.insert(0, SPECPATH)
import third_party  # noqa: E402 – licenses of third-party software in the package (aborts if a text is missing)

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
excludes = ["tkinter", "tests", "matplotlib", "IPython", "PySide6.scripts"]
if not DARWIN:
    hidden = [name for name in hidden if not name.startswith("sketch_machine.metal")]
    excludes.append("sketch_machine.metal")
if DARWIN:
    # Build Metal only on macOS; retain OpenGL for explicit selection and unsupported GPUs.
    metal_built = os.path.join(workpath, "native_metal")
    subprocess.run([sys.executable, "-m", "sketch_machine.metal.build", metal_built], cwd=ROOT, check=True)
    binaries.append((os.path.join(metal_built, "libmule_metal.dylib"), os.path.join("sketch_machine", "metal", "built")))
    datas.append((os.path.join(metal_built, "builtin.metallib"), os.path.join("sketch_machine", "metal", "built")))
    datas.append((os.path.join(metal_built, "scene.metallib"), os.path.join("sketch_machine", "metal", "built")))
    datas.append((os.path.join(ROOT, "sketch_machine", "metal", "programs.json"), os.path.join("sketch_machine", "metal")))
    # Guides inside the bundle (Contents/Resources): nothing but code may sit next to the executable (paths.doc_path)
    for name in ("ANLEITUNG.md", "GUIDE.md", "AI_HANDBOOK.md", "CHANGELOG.md", "CONTRIBUTING.md"):
        if os.path.isfile(os.path.join(ROOT, name)):
            datas.append((os.path.join(ROOT, name), "."))
    if os.path.isdir(os.path.join(ROOT, "docs")):
        datas.append((os.path.join(ROOT, "docs"), "docs"))
for package in ("pxr",):  # USD finds its plugins via plugInfo.json next to the libraries
    d, b, h = collect_all(package)
    datas, binaries, hidden = datas + d, binaries + b, hidden + h

a = Analysis([os.path.join(SPECPATH, "mule_main.py")], pathex=[ROOT], datas=datas, binaries=binaries, hiddenimports=hidden,
             excludes=excludes)
if DARWIN:  # licenses into the bundle (Contents/Resources, next to the guides) – written before COLLECT so BUNDLE has them
    notices = os.path.join(workpath, "third_party")
    shutil.rmtree(notices, ignore_errors=True)
    third_party.write(notices, [e[0] for e in a.pure], [e[0] for e in a.binaries + a.datas])
    for base, _dirs, files in os.walk(notices):
        for f in files:
            src = os.path.join(base, f)
            a.datas.append((os.path.relpath(src, notices), src, "DATA"))
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Mule", console=False,
          icon=ICON if os.path.exists(ICON) else None)
coll = COLLECT(exe, a.binaries, a.datas, name="Mule")
if not DARWIN:  # licenses next to Mule.exe
    third_party.write(os.path.join(DISTPATH, "Mule"), [e[0] for e in a.pure], [e[0] for e in a.binaries + a.datas])


if DARWIN:
    app = BUNDLE(coll, name="Mule.app", icon=ICON, bundle_identifier="com.felixbfx.mule",
                 info_plist={"NSHighResolutionCapable": True, "LSMinimumSystemVersion": "13.0"})
