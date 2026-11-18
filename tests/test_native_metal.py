"""Native Metal resource ordering and real MetalRT nearest-hit queries, including instances and scene changes."""

from types import SimpleNamespace

import numpy as np


def _native(h):
    ctx = h.v.renderer.ctx
    return ctx if getattr(ctx, "native_metal", False) else None


def _triangle():
    class Geometry:
        pass

    g = Geometry()
    g.vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], "f4")
    g.faces = np.array([[0, 1, 2]], "u4")
    g.colors = np.ones((3, 3), "f4")
    return g


def test_metalrt_nearest_hit_instances_and_scene_changes(h):
    ctx = _native(h)
    if ctx is None or not ctx.supports_raytracing:
        return "Native MetalRT unavailable - skipped"
    geo = _triangle()
    far = SimpleNamespace(geo=geo, xform=np.eye(4))
    near = SimpleNamespace(geo=geo, xform=np.eye(4))
    near.xform[2, 3] = 1

    def trace(meshes):
        return h.v._with_gpu(
            lambda: ctx.ray_pick(
                meshes, [[0, 0, 3], [8, 0, 3]], [[0, 0, -1], [0, 0, -1]]
            )
        )

    data, prims = trace([far, near])
    assert data[0, 3] == 2 and np.allclose(data[0, :3], [0, 0, 1]) and prims[0] == 0
    assert data[1, 3] == 0
    # Reuse the geometry BLAS while changing the instanced transform, including nonuniform scaling.
    near.xform[:3, :3] = np.diag([2.0, 0.5, 1.0])
    near.xform[2, 3] = -1
    data, _ = trace([far, near])
    assert data[0, 3] == 1 and np.allclose(data[0, :3], [0, 0, 0])
    # Hide/remove the front object; the remaining instance becomes the nearest hit.
    data, _ = trace([near])
    assert data[0, 3] == 1 and np.allclose(data[0, :3], [0, 0, -1])
    # Replacing geometry must build a new acceleration structure instead of reusing stale triangles.
    near.geo = _triangle()
    near.geo.vertices[:, 2] = 2
    data, _ = trace([near])
    assert data[0, 3] == 1 and np.allclose(data[0, :3], [0, 0, 1])
    assert ctx.diagnostics()["ray_queries"] >= 8


def test_native_upload_draw_readback_ordering(h):
    ctx = _native(h)
    if ctx is None:
        return "Native Metal unavailable - skipped"

    def work():
        buffer = ctx.buffer(np.arange(16, dtype="u4"))
        tex = ctx.texture((4, 4), 4)
        fbo = ctx.framebuffer([tex])
        try:
            for i in range(4):
                values = np.arange(16, dtype="u4") + i
                buffer.write(values)
                # Reading after an asynchronous commit must still wait for the committed copy.
                ctx.api.call("submit", ctx.handle, 0)
                assert np.array_equal(np.frombuffer(buffer.read(), "u4"), values)
                fbo.clear(0.25, 0.5, 0.75, 1.0)
                pixel = fbo.read(components=4)[:4]
                assert (
                    max(abs(int(a) - b) for a, b in zip(pixel, (64, 128, 191, 255)))
                    <= 1
                )
        finally:
            fbo.release()
            tex.release()
            buffer.release()

    h.v._with_gpu(work)


def test_split_views_share_geometry_after_original_view_closes(h):
    if _native(h) is None:
        return "Native Metal unavailable - skipped"
    from PySide6.QtCore import QCoreApplication, QEvent, Qt
    from sketch_machine.scene import MeshObject
    from tests.test_mesh_block import _box

    h.sc.meshes.append(MeshObject("Box", "", _box()))
    original = h.v
    original._changed()
    area = h.w.views
    second = area.split(original, Qt.Horizontal)
    area.split(
        second, Qt.Vertical
    )  # Reparents an existing native view into a nested splitter.
    h.pump(0.05)
    for vp in area.viewports():
        assert not vp.gpu_grab().isNull()
        assert vp.renderer.ctx.diagnostics()["presented_frames"] > 0
    area.close_viewport(original)
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    h.pump(0.05)
    for vp in area.viewports():
        assert not vp.gpu_grab().isNull()
        assert vp.renderer.ctx.diagnostics()["presented_frames"] > 0
    while area.count() > 1:
        area.close_viewport(area.viewports()[-1])


def test_editor_draws_with_metal41_shaders(h):
    ctx = _native(h)
    if ctx is None:
        return "Native Metal unavailable - skipped"
    import os
    import re
    import shutil
    import subprocess
    from unittest.mock import patch
    from PySide6.QtCore import QPointF, Qt
    from tests.harness import SCENES
    from sketch_machine.metal.programs import definitions

    assert ctx.version_code == 410 and "MSL 4.1" in ctx.info["GL_VERSION"], ctx.info
    if shutil.which("xcrun"):
        disasms = {}
        for lib_name in ("builtin.metallib", "scene.metallib"):
            out = subprocess.check_output(
                ["xcrun", "--sdk", "macosx", "metal-objdump", "-d", str(ctx.api.folder / lib_name)],
                text=True,
            )
            metas = set(re.findall(r'!\{!"Metal",[^\}]+\}', out))
            assert metas == {'!{!"Metal", i32 4, i32 1, i32 0}'}, (lib_name, metas)
            disasms[lib_name] = out
        expected = set()
        for spec in definitions().values():
            expected.update((spec["vertex"]["entry"], spec["fragment"]["entry"]))
            if spec["wide"]:
                expected.add(spec["wide"]["vertex"]["entry"])
        found = set(re.findall(r"@([A-Za-z0-9_]+)\(", disasms["scene.metallib"]))
        missing = expected - found
        assert not missing, f"Missing MSL 4.1 shader entries: {sorted(missing)}"

    h.load(os.path.join(SCENES, "watering_can.sketch.json"))
    h.grab()
    h.v.set_light(True)
    h.grab()
    h.v.set_xray(True)
    h.grab()
    h.v.set_light(False)
    h.v.set_xray(False)
    center = QPointF(h.v.width() / 2, h.v.height() / 2)
    h.v._mesh_samples([center])
    with patch.dict(os.environ, {"MULE_METALRT": "0"}):
        h.v._mesh_samples([center])

    h.new_scene()
    h.v._activate(h.sc.planes[0])
    h.stroke_frames([h.scrf([-1.5, 0, -1.0]), h.scrf([1.5, 0, -1.0])])
    h.key(Qt.Key_S)
    h.v.set_draw_tool("brush")
    h.v.set_paint_color((0.2, 0.45, 0.85))
    h.stroke_world([[x, 0, 0.5] for x in np.linspace(-1.5, 1.5, 12)])
    h.key(Qt.Key_S)
    h.edit_mode("point")
    h.v._knot_list()
    h.draw_mode()
    h.grab()

    assert ctx._library is not None and ctx._library.version == 41 and len(ctx._library.binary) > 0
    diag = ctx.diagnostics()
    assert diag["libraries"] >= 2 and diag["pipelines"] >= 8 and diag["draws"] > 20 and diag["presented_frames"] > 0, diag

    def check_fbo():
        tex = ctx.texture((4, 4), 4)
        fbo = ctx.framebuffer([tex])
        try:
            fbo.use()
            assert ctx._fbo is fbo
            fbo.release()
            assert ctx._fbo is ctx.screen
        finally:
            if not fbo._released:
                fbo.release()
            tex.release()

    h.v._with_gpu(check_fbo)


def test_inline_note_editor_visible_above_metal_view(h):
    if _native(h) is None:
        return "Native Metal unavailable - skipped"
    import ctypes
    from PySide6.QtCore import Qt
    from sketch_machine.scene import Note

    h.new_scene()
    note = Note("Heading", np.array([0.5, 0.8, 0.0]))
    h.sc.notes.append(note)
    h.v.set_notes_mode(True)
    h.v.ws.active_note = note.id
    h.grab()
    h.v.notes_tool.edit(note.id)
    h.pump(0.05)
    try:
        ed = h.v.notes_tool.editor[0]
        assert ed.isVisible() and ed.testAttribute(Qt.WA_NativeWindow)
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.object_getClassName.restype = ctypes.c_char_p
        objc.object_getClassName.argtypes = [ctypes.c_void_p]
        msg_ptr = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p))
        msg_ulong = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p, ctypes.c_void_p))
        msg_idx = ctypes.cast(
            objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong),
        )
        subviews = msg_ptr(int(h.v.winId()), objc.sel_registerName(b"subviews"))
        n = msg_ulong(subviews, objc.sel_registerName(b"count"))
        obj_at = objc.sel_registerName(b"objectAtIndex:")
        ed_ptr = int(ed.winId())
        metal_index = None
        ed_index = None
        for i in range(n):
            sv = msg_idx(subviews, obj_at, i)
            if objc.object_getClassName(sv) == b"MuleMetalView":
                metal_index = i
            if sv == ed_ptr:
                ed_index = i
        assert metal_index is not None and ed_index is not None and ed_index > metal_index, (
            metal_index,
            ed_index,
        )
    finally:
        h.v.notes_tool.finish_edit(cancel=True)
