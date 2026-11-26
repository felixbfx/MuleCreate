"""Speed: what a frame must not recompute without a change to the scene - and that it still looks right."""
import numpy as np
from PySide6.QtCore import QEvent, QPointF
from PySide6.QtGui import QMouseEvent

from tests.harness import H, Qt
from tests.test_material import _square


def _orbit(h: H, steps=6):
    """Alt + left mouse drag (orbit), every frame as in the app (without a forced rebuild)."""
    v = h.v

    def send(kind, pos, buttons):
        button = Qt.LeftButton if kind != QEvent.MouseMove else Qt.NoButton
        h.app.sendEvent(v, QMouseEvent(kind, pos, pos, button, buttons, Qt.AltModifier))

    send(QEvent.MouseButtonPress, QPointF(500, 400), Qt.LeftButton)
    for i in range(steps):
        send(QEvent.MouseMove, QPointF(500 + 6 * (i + 1), 400 + 2 * i), Qt.LeftButton)
        img = v.gpu_grab()
    assert v._mode == "orbit", f"dragging with Alt should orbit (mode {v._mode})"
    return img, lambda: send(QEvent.MouseButtonRelease, QPointF(540, 410), Qt.NoButton)


def test_orbiting_does_not_rebuild_strokes_and_surfaces(h: H):
    """Orbiting changes only the camera: draw lists and surface sync do not run every frame (otherwise orbiting costs a
    multiple with many lines/surfaces), and the image is still exactly the freshly built one."""
    h.new_scene()
    _square(h, y=0.0, x0=-1.0, size=2.0)
    h.sc.add_curve(np.linspace([-1.0, 0.5, 0.0], [1.0, 0.5, 1.0], 12), None, np.full(12, 0.03))
    h.v._changed()
    h.camera([0.0, 0.0, 0.5], 6, [0.3, 0.8, 0.5])
    v, sc = h.v, h.sc
    calls = []
    orig = (v._surface_draw_list, v._stroke_draw_list, sc.sync_lofts)
    v._surface_draw_list = lambda: calls.append("surfaces") or orig[0]()
    v._stroke_draw_list = lambda: calls.append("strokes") or orig[1]()
    sc.sync_lofts = lambda *a, **k: calls.append("sync") or orig[2](*a, **k)
    try:
        v.gpu_grab()  # last frame before orbiting
        calls.clear()
        cached, release = _orbit(h)
        # once on engaging (click, mode switch), after that in none of the 6 frames
        many = {k: calls.count(k) for k in set(calls) if calls.count(k) > 1}
        assert not many, f"rebuilt every frame while orbiting: {many}"
        fresh = v.grabFramebuffer()  # forces a rebuild
        n = h.diff(cached, fresh, threshold=2)
        assert n == 0, f"image while orbiting differs from the freshly built one ({n} pixels)"
        release()
    finally:
        v._surface_draw_list, v._stroke_draw_list, sc.sync_lofts = orig


def test_engaging_and_releasing_the_camera_rebuilds_nothing(h: H):
    """Alt+click, orbit, release: no rebuild of the draw lists and no surface sync - not on engaging either
    (otherwise the first frame visibly stutters in large scenes) and not in the other viewports."""
    h.new_scene()
    _square(h, y=0.0, x0=-1.0, size=2.0)
    h.sc.add_curve(np.linspace([-1.0, 0.5, 0.0], [1.0, 0.5, 1.0], 12), None, np.full(12, 0.03))
    h.v._changed()
    h.camera([0.0, 0.0, 0.5], 6, [0.3, 0.8, 0.5])
    v, sc = h.v, h.sc
    calls = []
    orig = (v._surface_draw_list, v._scene_stroke_list, sc.sync_lofts)
    v._surface_draw_list = lambda: calls.append("surfaces") or orig[0]()
    v._scene_stroke_list = lambda: calls.append("strokes") or orig[1]()
    sc.sync_lofts = lambda *a, **k: calls.append("sync") or orig[2](*a, **k)
    try:
        v.gpu_grab()
        calls.clear()
        _, release = _orbit(h)
        release()
        v.gpu_grab()
        assert not calls, f"engaging/releasing the camera rebuilt: {calls}"
        assert v._mode is None, f"no mode any more after release ({v._mode})"
    finally:
        v._surface_draw_list, v._scene_stroke_list, sc.sync_lofts = orig


def test_hover_rebuilds_only_on_change(h: H):
    """Edit mode: moving the mouse without a button rebuilds strokes/surfaces only when the highlighted edge changes -
    the colouring on hover still appears (and goes away again)."""
    h.new_scene()
    h.sc.new_stroke()
    h.sc.add_curve(np.linspace([-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], 20), None, np.full(20, 0.03))
    line = h.sc.curves[-1]
    h.v._changed()
    h.camera([0.0, 0.0, 0.0], 4, [0.0, 1.0, 0.02])
    h.edit_mode("edge")
    v = h.v

    def move(p):
        h.app.sendEvent(v, QMouseEvent(QEvent.MouseMove, p, p, Qt.NoButton, Qt.NoButton, Qt.NoModifier))
        return v.gpu_grab()

    on = QPointF(h.scr(line.points[10]))
    off = on + QPointF(0, 120)
    base = move(off)
    move(off + QPointF(3, 0))
    assert not v._lists_dirty, "moving without a hover change: no new draw lists"
    hot = move(on)
    assert v._select_hover is not None, "over the line: hover"
    assert h.diff(hot, base) > 0, "line under the cursor is highlighted"
    cold = move(off)
    assert v._select_hover is None
    assert h.diff(cold, base) == 0, "highlight disappears again"
    h.draw_mode()


def test_drawing_rebuilds_only_the_stroke(h: H):
    """During the stroke only the running stroke changes: surface sync and the scene's lines do not run every
    frame (in glider_owl 42 -> 11 ms per frame) - and the image in the middle of the stroke is exactly the freshly built one;
    after release the stroke is a line of the scene."""
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    h.new_scene()
    _square(h, y=0.0, x0=-1.0, size=2.0)
    h.sc.add_curve(np.linspace([-1.0, 0.5, 0.0], [1.0, 0.5, 1.0], 12), None, np.full(12, 0.03))
    h.v._changed()
    h.camera([0.0, 0.0, 0.5], 6, [0.3, 0.8, 0.5])
    v, sc = h.v, h.sc
    sc.activate(sc.presets[1] if len(sc.presets) > 1 else sc.presets[0])
    sc.new_stroke()
    v._changed()
    v.gpu_grab()
    n_lines = len(h.curves())
    calls = []
    orig = (v._surface_draw_list, v._scene_stroke_list, sc.sync_lofts)
    v._surface_draw_list = lambda: calls.append("surfaces") or orig[0]()
    v._scene_stroke_list = lambda: calls.append("strokes") or orig[1]()
    sc.sync_lofts = lambda *a, **k: calls.append("sync") or orig[2](*a, **k)
    try:
        c = QPointF(v.width() / 2, v.height() / 2)
        pts = [c + QPointF(-120 + 8 * k, 30 * np.sin(k / 4)) for k in range(30)]
        h.move(pts[0])
        QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, pts[0].toPoint())
        for p in pts[1:]:
            h.move(p, Qt.LeftButton)
            cached = v.gpu_grab()
        assert v._mode == "draw", f"stroke is running ({v._mode})"
        many = {k: calls.count(k) for k in set(calls) if calls.count(k) > 2}
        assert not many, f"rebuilt every frame while drawing: {many}"
        fresh = v.grabFramebuffer()  # forces a rebuild
        n = h.diff(cached, fresh, threshold=2)
        assert n == 0, f"image in the middle of the stroke differs from the freshly built one ({n} pixels)"
        QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, pts[-1].toPoint())
        h.pump(0.02)
        after = v.gpu_grab()
        assert len(h.curves()) == n_lines + 1, "the stroke is a line of the scene"
        assert h.diff(after, v.grabFramebuffer(), threshold=2) == 0, "after release: image = freshly built"
    finally:
        v._surface_draw_list, v._scene_stroke_list, sc.sync_lofts = orig
