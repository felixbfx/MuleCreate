"""Portable builds retain OpenGL, and packaged startup discovers bundled USD plugins."""
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

LEVEL = "core"


def test_frozen_entrypoint_configures_usd_plugins_before_launch(h):
    entrypoint = Path(__file__).parents[1] / "packaging" / "mule_main.py"
    launch = ModuleType("sketch_machine.launch")
    observed = []
    launch.main = lambda: observed.append(os.environ.get("PXR_PLUGINPATH_NAME"))
    with tempfile.TemporaryDirectory() as folder:
        plugins = Path(folder) / "pxr" / "pluginfo"
        plugins.mkdir(parents=True)
        plugin_info = plugins / "plugInfo.json"
        plugin_info.write_text('{"Includes": ["*/resources/"]}')
        with patch.object(sys, "frozen", True, create=True), patch.object(sys, "_MEIPASS", folder, create=True):
            with patch.dict(os.environ, {"PXR_PLUGINPATH_NAME": "/custom/usd/plugins"}):
                with patch.dict(sys.modules, {"sketch_machine.launch": launch}):
                    runpy.run_path(str(entrypoint))
    assert observed == [str(plugin_info) + os.pathsep + "/custom/usd/plugins"]


def test_windows_linux_spec_skips_metal_build_and_assets(h):
    spec = Path(__file__).parents[1] / "packaging" / "mule.spec"
    source = compile(spec.read_text(), str(spec), "exec")
    for platform in ("win32", "linux"):
        collected = []
        analyzed = []
        notices = []
        hooks = ModuleType("PyInstaller.utils.hooks")
        def submodules(package):
            collected.append(package)
            return [package, "sketch_machine.metal", "sketch_machine.metal.api"] if package == "sketch_machine" else [package]

        hooks.collect_submodules = submodules
        hooks.collect_all = lambda package: ([], [], [package])
        licenses = ModuleType("third_party")
        licenses.write = lambda *args: notices.append(args)

        def analysis(*args, **kwargs):
            analyzed.append(kwargs)
            return SimpleNamespace(pure=[], binaries=[], datas=[], scripts=[])

        def forbidden(*args, **kwargs):
            raise AssertionError("Portable builds must not compile or bundle native Metal")

        with tempfile.TemporaryDirectory() as folder:
            namespace = {
                "SPECPATH": str(spec.parent), "DISTPATH": folder, "workpath": folder,
                "Analysis": analysis, "PYZ": lambda *a, **k: None,
                "EXE": lambda *a, **k: None, "COLLECT": lambda *a, **k: None,
                "BUNDLE": forbidden,
            }
            original_path = list(sys.path)
            try:
                with patch.object(sys, "platform", platform), patch("subprocess.run", forbidden):
                    with patch.dict(sys.modules, {"PyInstaller.utils.hooks": hooks, "third_party": licenses}):
                        exec(source, namespace)
            finally:
                sys.path[:] = original_path

        assert "moderngl" in collected and "glcontext" in collected
        assert not {"moderngl", "glcontext"} & set(analyzed[0]["excludes"])
        assert "sketch_machine.metal" in analyzed[0]["excludes"]
        assert not any(name.startswith("sketch_machine.metal") for name in analyzed[0]["hiddenimports"])
        assets = [str(path) for path, _ in analyzed[0]["datas"] + analyzed[0]["binaries"]]
        assert not any(path.endswith((".metallib", ".dylib")) for path in assets)
        assert not any(Path(path).name == "programs.json" for path in assets)
        assert notices, "Portable builds must still generate their license notices"
