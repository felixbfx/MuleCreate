"""Automatic renderer selection and native shader independence, without a window."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

LEVEL = "core"


def test_native_selection_requires_supported_gpu(h):
    from sketch_machine import metal
    with patch.dict(os.environ, {"MULE_GPU": ""}), patch.object(sys, "platform", "darwin"):
        with patch.object(metal, "supported_os", return_value=True):
            for available in (False, True):
                with patch.object(metal, "capabilities", return_value={"available": available, "reason": "GPU capability"}):
                    assert metal.wanted() is available


def test_mac_bundle_defaults_to_metal_and_retains_opengl_fallback(h):
    code = '''
import os
import sys
from unittest.mock import patch
from PySide6 import QtWidgets, QtOpenGLWidgets
from sketch_machine import metal
sys.platform = "darwin"
sys.frozen = True
sys.executable = "/Applications/Mule.app/Contents/MacOS/Mule"
if sys.argv[2]:
    os.environ["MULE_GPU"] = sys.argv[2]
else:
    os.environ.pop("MULE_GPU", None)
available = sys.argv[1] == "1"
with patch.object(metal, "supported_os", return_value=True):
    with patch.object(metal, "capabilities", return_value={"available": available, "reason": "Unsupported GPU"}):
        from sketch_machine.vp.backend import GPUBackend, backend_name
expected = "metal" if available and not sys.argv[2] else "gl"
assert backend_name() == expected
assert issubclass(GPUBackend, QtWidgets.QWidget if expected == "metal" else QtOpenGLWidgets.QOpenGLWidget)
'''
    for available, choice in (("1", ""), ("0", ""), ("1", "gl")):
        subprocess.run([sys.executable, "-c", code, available, choice], check=True, capture_output=True, text=True)


def test_native_constants_match_moderngl(h):
    import moderngl
    from sketch_machine.metal import types
    for name, value in vars(types).items():
        if name.isupper() and hasattr(moderngl, name):
            assert getattr(moderngl, name) == value, name


def test_explicit_portable_backend_skips_native_probe(h):
    from sketch_machine import metal
    with patch.object(metal, "capabilities", side_effect=AssertionError("Native probe should not run")):
        for choice in ("gl", "opengl", " GL "):
            with patch.dict(os.environ, {"MULE_GPU": choice}):
                assert not metal.wanted()
        for platform in ("win32", "linux"):
            with patch.object(sys, "platform", platform), patch.dict(os.environ, {"MULE_GPU": ""}):
                assert not metal.wanted()


def test_forced_native_reports_unsupported_machine(h):
    from sketch_machine import metal
    with patch.object(metal, "capabilities", return_value={"available": False, "reason": "GPU lacks Metal 4"}):
        with patch.dict(os.environ, {"MULE_GPU": "metal"}):
            try:
                metal.wanted()
            except RuntimeError as exc:
                assert str(exc) == "GPU lacks Metal 4"
            else:
                raise AssertionError("Explicit native selection must report the unavailable renderer")


def test_old_os_does_not_load_native_library(h):
    from sketch_machine import metal
    from sketch_machine.metal import api
    metal.capabilities.cache_clear()
    try:
        with patch.object(metal, "supported_os", return_value=False):
            with patch.object(api, "API", side_effect=AssertionError("Native library must not load")):
                assert not metal.capabilities()["available"]
    finally:
        metal.capabilities.cache_clear()


def test_native_initialization_failure_can_select_opengl(h):
    from sketch_machine import metal
    from sketch_machine.metal import api
    metal.capabilities.cache_clear()
    try:
        with patch.object(metal, "supported_os", return_value=True):
            with patch.object(api, "API", side_effect=OSError("Library cannot load")):
                result = metal.capabilities()
                assert not result["available"] and result["reason"] == "Library cannot load"
    finally:
        metal.capabilities.cache_clear()


def test_frozen_app_uses_bundled_libraries_without_compiler(h):
    from sketch_machine.metal import build
    with tempfile.TemporaryDirectory() as folder:
        module = Path(folder) / "metal" / "build.py"
        bundled = module.with_name("built")
        bundled.mkdir(parents=True)
        for name in build.NAMES:
            (bundled / name).touch()
        with patch.object(build, "__file__", str(module)), patch.object(sys, "frozen", True, create=True):
            with patch.object(build, "build", side_effect=AssertionError("Frozen app must not compile")):
                assert build.locate() == bundled
                (bundled / "scene.metallib").unlink()
                try:
                    build.locate()
                except RuntimeError as exc:
                    assert "not bundled" in str(exc)
                else:
                    raise AssertionError("Incomplete native assets must not be selected")


def test_native_programs_cover_portable_renderer(h):
    from sketch_machine.renderer import PROGRAMS
    from sketch_machine.metal.programs import definitions
    specs = definitions()
    assert set(specs) == set(PROGRAMS) | {"resolve2", "resolve4", "resolve8"}
    sources = Path(__file__).parents[1] / "sketch_machine" / "metal" / "shaders"
    for spec in specs.values():
        stages = [spec["vertex"], spec["fragment"]]
        if spec["wide"]:
            stages.append(spec["wide"]["vertex"])
        for stage in stages:
            # Layout checks must tolerate formatter line wrapping.
            text = " ".join((sources / stage["source"]).read_text().split())
            assert stage["entry"] + "(" in text
            # The compiler checks MSL field offsets; also ensure those checks match the host schema.
            if "Uniform size" in text:
                assert f'== {spec["block_size"]}, "Uniform size"' in text
                for member in spec["members"]:
                    assert f'== {member["offset"]}, "{member["name"]} offset"' in text


def test_native_renderer_imports_without_rhi_or_glsl(h):
    code = '''
import importlib.abc
import sys
from types import SimpleNamespace
class BlockPortable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("sketch_machine.rhi", "sketch_machine.glsl_shaders")):
            raise ImportError("Portable shader toolchain imported: " + fullname)
sys.meta_path.insert(0, BlockPortable())
from sketch_machine.metal.programs import Program
from sketch_machine.renderer import Renderer
from sketch_machine.metal.context import Context
class NativeContext:
    native_metal = True
    def program(self, name): return Program(self, name)
    def _function(self, spec): return SimpleNamespace(entry=spec["entry"], handle=1)
    def vertex_array(self, *a, **k): return object()
Renderer(NativeContext(), SimpleNamespace(glDepthMask=lambda *a: None))
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_windows_linux_renderer_imports_without_native_metal(h):
    code = '''
import importlib.abc
import os
import sys
from types import SimpleNamespace
# Load this machine's extension modules before simulating the target platform.
import numpy
from PySide6 import QtCore, QtGui, QtWidgets, QtOpenGLWidgets
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith("sketch_machine.metal"):
            raise ImportError("Native Metal dependency imported: " + fullname)
sys.meta_path.insert(0, BlockNative())
sys.platform = sys.argv[1]
os.environ.pop("MULE_GPU", None)
from sketch_machine.vp.backend import GPUBackend, backend_name
assert backend_name() == "gl"
assert issubclass(GPUBackend, QtOpenGLWidgets.QOpenGLWidget)
class Surface:
    def setFormat(self, fmt): self.fmt = fmt
surface = Surface()
GPUBackend._setup_surface(surface)
assert surface.fmt.version() == (3, 3)
assert surface.fmt.profile() == QtGui.QSurfaceFormat.CompatibilityProfile
assert surface.fmt.depthBufferSize() == 24
from sketch_machine.renderer import Renderer
from sketch_machine.renderer import PROGRAMS
from sketch_machine.renderer import program
class PortableContext:
    def program(self, vertex_shader, fragment_shader):
        assert vertex_shader.lstrip().startswith("#version 330")
        assert fragment_shader.lstrip().startswith("#version 330")
        return object()
    def vertex_array(self, *a, **k): return object()
ctx = PortableContext()
Renderer(ctx, SimpleNamespace(glDepthMask=lambda *a: None))
for name in PROGRAMS:
    program(ctx, name)
'''
    for platform in ("win32", "linux"):
        subprocess.run([sys.executable, "-c", code, platform], check=True, capture_output=True, text=True)
