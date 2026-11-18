"""Self-test of the finished package: mule --selftest <result.json>. On a fresh machine it checks what can get lost
during packaging (GPU rendering, MetalRT, USD, bible, brief export) and exits like normal; real settings stay
untouched, no dialogs."""
import json
import os
import sys
import tempfile
import traceback

from PySide6.QtCore import QSettings, QTimer


def use_throwaway_settings():
    """Before the first read of the settings: throwaway file instead of the machine's real settings."""
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, tempfile.mkdtemp(prefix="mule_selftest_settings_"))


def _usd_file(folder):
    from pxr import Usd, UsdGeom
    path = os.path.join(folder, "selftest.usda")
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.Xform.Define(stage, "/root")
    mesh = UsdGeom.Mesh.Define(stage, "/root/quad")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    stage.GetRootLayer().Save()
    return path


def run(window, app, out):
    """Checks one after another; a failure does not stop the later ones."""
    from . import brief, paths
    result = {"frozen": paths.FROZEN, "checks": {}}
    folder = tempfile.mkdtemp(prefix="mule_selftest_")

    def check(name, fn):
        try:
            result["checks"][name] = {"ok": True, "info": fn()}
        except Exception:
            result["checks"][name] = {"ok": False, "info": traceback.format_exc()[-1500:]}

    def pump():
        for _ in range(20):
            app.processEvents()

    def opengl():
        window.show()
        pump()
        image = window.viewport.grabFramebuffer()
        renderer = window.viewport.renderer
        assert renderer is not None, window.viewport._gl_error
        assert image.width() > 0 and image.height() > 0
        result["gpu"] = dict(renderer.ctx.info)
        from .vp.backend import backend_name
        result["gpu"]["backend"] = backend_name()
        if os.environ.get("MULE_GPU") in ("metal","native"):
            assert getattr(renderer.ctx,"native_metal",False), "Native Metal renderer was not initialized"
        return f"{image.width()}x{image.height()}"

    def usd():
        tree = window.viewport.import_mesh(_usd_file(folder))
        parts = [tree] if hasattr(tree, "geo") else [o for o in window.viewport.scene.descendants(tree) if hasattr(o, "geo")]
        assert parts, "no meshes from the USD file"
        return len(parts)

    def bible():
        assert os.path.isfile(brief.BIBLE) and os.access(brief.BIBLE, os.W_OK), brief.BIBLE
        return brief.BIBLE

    def metalrt():
        import numpy as np
        ctx=window.viewport.renderer.ctx
        if not getattr(ctx,"native_metal",False): return "Qt/GL fallback"
        stats=ctx.diagnostics()
        assert stats["draws"]>0 and stats["frames"]>0,stats
        if not ctx.supports_raytracing or os.environ.get("MULE_METALRT","1")=="0": return stats
        meshes=window.viewport._pick_items(window.viewport._pickables())
        mesh=next(m for m in meshes if m.geo is not None and len(m.geo.faces))
        triangle=mesh.geo.vertices[mesh.geo.faces[0]].astype(float)
        triangle=triangle@mesh.xform[:3,:3].T+mesh.xform[:3,3]
        center=triangle.mean(axis=0);normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
        data,_=window.viewport._with_gpu(lambda:ctx.ray_pick(meshes,[center+normal],[ -normal]))
        assert data[0,3]>0,data
        result["gpu"]["native_diagnostics"]=ctx.diagnostics()
        return ctx.diagnostics()

    def export():
        target = os.path.join(folder, "selftest_brief")
        files = brief.export(window.viewport.scene, window.viewport, target, "selftest")
        for name in ("brief.md", "brief.json", "scene.usd"):
            assert os.path.isfile(os.path.join(target, name)), name
        return len(files)

    def manual():
        from .manual import ManualWindow
        result["_manual"] = ManualWindow(window)  # with its parent, as in the program: Qt deletes it with the window
        return result["_manual"].tabs.count()

    def scene_file():
        path = os.path.join(folder, "selftest.sketch.json")
        window.viewport.scene.save(path)
        window.load_path(path)
        assert window.path == path
        return os.path.getsize(path)

    def videos():
        from PySide6.QtMultimedia import QMediaFormat
        from .video import available
        assert available()
        codecs = QMediaFormat().supportedVideoCodecs(QMediaFormat.Decode)
        assert QMediaFormat.VideoCodec.H264 in codecs, codecs
        return len(codecs)

    def animation():
        from PySide6.QtGui import QMovie
        from . import startup
        movie = QMovie(startup.ANIMATION)
        assert movie.isValid() and movie.frameCount() > 200, startup.ANIMATION
        return movie.frameCount()

    def recording():
        from PySide6.QtWebSockets import QWebSocket  # OBS connection of the recording
        from . import recorder
        assert QWebSocket() is not None and os.path.isdir(recorder.recordings_dir())
        return "websocket"

    def licenses():
        if not getattr(sys, "frozen", False):
            return "only in the package"
        bundle = paths.doc_path()  # next to Mule.exe; in Mule.app: Contents/Resources
        with open(os.path.join(bundle, "THIRD_PARTY_NOTICES.md"), encoding="utf-8") as fh:
            text = fh.read()
        for needed in ("PySide6", "usd-core", "numpy", "Python", "FFmpeg", "General Public License v3.0"):
            assert needed in text, needed
        with open(os.path.join(bundle, "LICENSE.txt"), encoding="utf-8") as fh:
            assert "GNU GENERAL PUBLIC LICENSE" in fh.read(200), "LICENSE.txt"
        for name in ("LGPL-3.0.txt", "GPL-3.0.txt", "LGPL-2.1.txt", "Python-LICENSE.txt"):
            assert os.path.isfile(os.path.join(bundle, "licenses", name)), name
        return len(os.listdir(os.path.join(bundle, "licenses")))

    for name, fn in (("opengl", opengl), ("usd", usd), ("metalrt",metalrt), ("bible", bible), ("brief", export), ("manual", manual),
                     ("scene", scene_file), ("video", videos), ("start_animation", animation), ("recording", recording),
                     ("licenses", licenses)):
        check(name, fn)
    keep = result.pop("_manual", None)  # noqa: F841 – stays alive until the end
    result["ok"] = all(c["ok"] for c in result["checks"].values())
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=1)
    sys.stdout.flush()
    window.viewport.scene.dirty = False  # no "Save?" dialog on close
    window.close()  # as with a normal exit: closing is part of the test too
    QTimer.singleShot(0, app.quit)
    app.exec()
    keep = None  # noqa: F841
    return 0 if result["ok"] else 1
