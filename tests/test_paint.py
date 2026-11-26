"""Paint mode: dead, flat-lying strokes – newest on top, not selectable or editable, erasable, they stick."""
import os

import numpy as np

from tests.harness import H, OUT, Qt
from sketch_machine import components, edges
from sketch_machine.scene import LoftSurface, Scene


def _paint_cross(h: H):
    sc = h.sc
    ground = sc.planes[0]
    h.v._activate(ground)
    h.camera([0, 0, 0], 9, [0, 0.8, 0.6])
    h.v.ws.pen_world = 0.35
    h.key(Qt.Key_S)
    assert h.v.ws.paint_mode and h.w.paint_mode_action.isChecked()
    h.v.set_draw_tool("brush")
    h.v.set_paint_color((0.85, 0.2, 0.1))
    h.stroke_world([[x, 0, 0.3 * np.sin(x)] for x in np.linspace(-3, 3, 30)])
    h.v.set_paint_color((0.15, 0.35, 0.85))
    h.stroke_world([[1.0 + 0.2 * np.sin(z * 2), 0, z] for z in np.linspace(-2.5, 2.5, 30)])  # crosses at x = 1
    return [c for c in h.curves() if c.paint]


def test_paint_strokes_flat_and_newest_on_top(h: H):
    paint = _paint_cross(h)
    assert len(paint) == 2 and [c.seq for c in paint] == [1, 2]
    assert all(c.plane_id == h.sc.planes[0].id for c in paint)
    assert all(c.normals is not None and len(c.normals) == len(c.points) for c in paint)
    img = h.save_image("paint_kreuz.png")
    r, g, b = h.pixel(img, h.scr([1.0, 0.0, 0.3 * np.sin(1.0)]))  # away from the origin (the world axes lie there)
    assert b > r + 40, f"at the crossing the newer blue stroke is on top ({r}, {g}, {b})"


def test_paint_strokes_not_selectable_and_without_points(h: H):
    paint = _paint_cross(h)
    h.key(Qt.Key_S)
    h.edit_mode("edge")
    assert h.v._curve_under(h.scrf(paint[0].points[20]), 8)[0] is None
    h.v.ws.component = "point"
    h.v._changed()
    assert not any(c.paint for _, c, _ in h.v._knot_list())
    assert not any(c.paint for _, c, _ in components.KnotIndex(h.sc).entries)


def test_double_click_on_plane_selects_paint(h: H):
    paint = _paint_cross(h)
    h.edit_mode("edge")
    h.v.set_planes_visible(True)
    h.v.select_loop(h.scrf([2.5, 0, 2.0]), "replace")
    sel = [edges.ref_to_curve(h.sc, r) for r in h.v.edge_selection]
    assert sum(1 for c in sel if c is not None and c.paint) == len(paint)


def test_erasing_and_saving(h: H):
    paint = _paint_cross(h)
    n0 = sum(len(c.points) for c in paint)
    h.v.set_draw_tool("eraser")
    h.stroke_world([[x, 0, 0.0] for x in np.linspace(-1.6, -0.4, 8)])
    paint = [c for c in h.curves() if c.paint]
    assert sum(len(c.points) for c in paint) < n0 and {c.seq for c in paint} == {1, 2}
    path = os.path.join(OUT, "paint_rt.sketch.json")
    h.sc.save(path)
    s2, _ = Scene.load(path)
    loaded = [c for c in s2.curves if c.paint]
    assert len(loaded) == len(paint), f"{len(loaded)} paint strokes loaded, {len(paint)} saved"
    assert {c.seq for c in loaded} == {1, 2} and all(c.plane_id is not None for c in loaded)


def test_paint_follows_the_loft(h: H):
    sc = h.sc
    sc.new_stroke()
    t = np.linspace(0, 1, 12)
    secs = []
    for z in (0.0, 2.0):
        sc.add_curve(np.stack([t * 3 - 1.5, 0.5 * np.sin(t * np.pi), np.full_like(t, z)], axis=1), None)
        secs.append(sc.curves[-1])
    lo = sc.create_loft([("lines", [c.id]) for c in secs], [c.points for c in secs])
    h.v._changed()
    h.v._activate(lo)
    verts = lo.fill_mesh()[0]
    h.frame(verts, back=[0.3, 0.8, 0.5])
    h.v.set_paint_mode(True)
    h.v.set_draw_tool("brush")
    c = verts.mean(axis=0)
    q = h.scr(c)
    from tests.harness import QPoint
    h.stroke([QPoint(q.x() - 50 + 5 * k, q.y()) for k in range(21)])
    pc = [x for x in sc.curves if x.paint][-1]
    h.v._changed()
    assert pc.plane_id == lo.id and pc.id in lo.attached
    before = pc.points.copy()
    sc.checkpoint()
    secs[1].translate(np.array([0.0, 0.6, 0.0]))
    h.v._changed()
    pc = sc.curve_by_id(pc.id)
    assert float(np.abs(pc.points - before).max()) > 1e-3, "paint stroke deforms with the loft"
    assert isinstance(lo, LoftSurface)


def _patch_zum_malen(h: H):
    """Curve surface from four border lines as the active drawing surface, paint mode on, camera in front."""
    sc = h.new_scene()
    sc.new_stroke()
    t = np.linspace(0, 1, 12)
    corners = [([-2, 0, 0], [2, 0, 0]), ([2, 0, 0], [2, 3, 0]), ([2, 3, 0], [-2, 3, 0]), ([-2, 3, 0], [-2, 0, 0])]
    for a, b in corners:
        sc.add_curve(np.array(a, float) + t[:, None] * (np.array(b, float) - np.array(a, float)), None)
    h.v._changed()
    surf = sc.create_patch(sc.curves[-4:])
    h.v._changed()
    h.v._activate(surf)
    h.camera([0, 1.5, 0], 8, [0, 0, 1])
    h.v.set_paint_mode(True)
    h.v.set_draw_tool("brush")
    return surf


def _male(h: H, y):
    h.stroke_world([[x, y, 0] for x in np.linspace(-1, 1, 20)])
    return [c for c in h.curves() if c.paint][-1]


def test_paint_stroke_sticks_and_deforms_along(h: H):
    """A paint stroke is bound to its surface immediately when put down, even without a full reconciliation, and follows it."""
    surf = _patch_zum_malen(h)
    c = _male(h, 1.5)
    assert c.plane_id == surf.id and c.id in surf.attached, "bound right after putting down"
    before = c.points.copy()
    rand = [x for x in h.curves() if not x.paint][2]
    h.sc.checkpoint()
    rand.points = rand.points + np.array([0.0, 0.0, 1.0])
    h.v._changed()
    assert float(np.abs(h.sc.curve_by_id(c.id).points - before).max()) > 1e-3, "follows the deformed surface"


def test_paint_stroke_undo_redo_without_scene_copy(h: H):
    from sketch_machine.scene import _PaintStep
    surf = _patch_zum_malen(h)
    a, b = _male(h, 1.0), _male(h, 2.0)
    assert [type(e) for e in h.sc._undo[-2:]] == [_PaintStep, _PaintStep], "no copy of the scene per paint stroke"
    pa, pb = a.points.copy(), b.points.copy()
    h.v.undo()
    assert h.sc.curve_by_id(b.id) is None and b.id not in surf.attached and h.sc.curve_by_id(a.id) is not None
    h.v.redo()
    assert np.array_equal(h.sc.curve_by_id(b.id).points, pb) and b.id in surf.attached
    assert h.sc.curve_by_id(b.id).seq == 2, "stays the newest stroke"
    h.v.undo()
    h.v.undo()
    assert not [c for c in h.curves() if c.paint] and not surf.attached
    h.v.redo()
    assert np.array_equal(h.sc.curve_by_id(a.id).points, pa)
    assert h.sc.curve_by_id(b.id) is None


def test_paint_stroke_undo_alternating_with_other_changes(h: H):
    """Light steps and full scene copies in one history: the copy recreates objects, the paint stroke is
    still found."""
    surf = _patch_zum_malen(h)
    a = _male(h, 1.0)
    pts = a.points.copy()
    h.sc.checkpoint()  # full copy, then a change to the surface
    surf.name = "Blatt"
    b = _male(h, 2.0)
    h.v.undo()  # b gone
    h.v.undo()  # copy: all objects new
    assert h.sc.curve_by_id(a.id) is not None and h.sc.curve_by_id(b.id) is None
    h.v.undo()  # a gone, although the object is a different one by now
    assert not [c for c in h.curves() if c.paint]
    for _ in range(3):
        h.v.redo()
    assert np.array_equal(h.sc.curve_by_id(a.id).points, pts) and h.sc.curve_by_id(b.id) is not None
    surf = h.sc.object_by_id(surf.id)
    assert a.id in surf.attached and b.id in surf.attached


def test_paint_stroke_with_new_stroke_object_is_full_step(h: H):
    """If painting first creates a stroke object, undo must take that back too."""
    from sketch_machine.scene import _PaintStep
    h.new_scene()
    h.v._activate(h.sc.planes[0])
    h.camera([0, 0, 0], 9, [0, 0.8, 0.6])
    h.v.set_paint_mode(True)
    h.v.set_draw_tool("brush")
    assert not h.sc.strokes
    h.stroke_world([[x, 0, 0] for x in np.linspace(-2, 2, 20)])
    assert len(h.sc.strokes) == 1 and not isinstance(h.sc._undo[-1], _PaintStep)
    h.v.undo()
    assert not h.sc.strokes


def test_paint_stroke_does_not_rebuild_lists_and_outliner_needlessly(h: H):
    """Starting and moving do not rebuild the draw lists, the outliner stays put, its line count is right."""
    from PySide6.QtCore import QPoint, QPointF
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QTreeWidgetItemIterator
    from sketch_machine.outliner import ID_ROLE, INFO, Outliner
    _patch_zum_malen(h)
    _male(h, 1.0)
    h.v.repaint()
    h.grab()
    tree = h.w.findChild(Outliner)
    calls, rebuilt = [], []
    list_fn, rebuild_fn = type(h.v)._scene_stroke_list, Outliner.rebuild
    type(h.v)._scene_stroke_list = lambda s: (calls.append(s._mode), list_fn(s))[1]
    Outliner.rebuild = lambda s: (rebuilt.append(1), rebuild_fn(s))[1]
    try:
        h.pump(0.05)
        rebuilt.clear()
        frame = lambda: h.v.gpu_grab()
        frame()
        calls.clear()
        pts = [h.scrf([x, 2.0, 0]) for x in np.linspace(-1, 1, 12)]
        QTest.mousePress(h.v, Qt.LeftButton, Qt.NoModifier, pts[0].toPoint())
        frame()
        for p in pts[1:]:
            h.move(QPointF(p), Qt.LeftButton)
            frame()
        assert not calls, f"lists rebuilt while drawing: {calls}"
        QTest.mouseRelease(h.v, Qt.LeftButton, Qt.NoModifier, pts[-1].toPoint())
        h.pump(0.05)
        assert not rebuilt, "outliner rebuilt"
        frame()
        assert calls, "after putting down, the lists show the new stroke"
    finally:
        type(h.v)._scene_stroke_list, Outliner.rebuild = list_fn, rebuild_fn
    stroke = h.sc.active_stroke
    it = QTreeWidgetItemIterator(tree)
    while it.value() and it.value().data(0, ID_ROLE) != stroke.id:
        it += 1
    assert it.value().text(INFO).startswith(str(len(stroke.curves))), "line count up to date"


def test_paint_on_mesh_lies_flat_and_slightly_lifted(h: H):
    """On an imported mesh: paint lies flat on its surface (normal of the mesh, towards the side it was painted
    from – also with wrongly wound triangles) and slightly lifted; older scenes are adjusted once
    on loading."""
    import json
    import tempfile
    from sketch_machine.geometry import Geometry
    from sketch_machine.scene import MeshObject, PAINT_LIFT
    h.new_scene()
    v = np.array([[-3, 0.5, -3], [3, 0.5, -3], [3, 0.5, 3], [-3, 0.5, 3]], float)
    mesh = MeshObject("Platte", "", Geometry(v, np.array([[0, 1, 2], [0, 2, 3]])))  # normal wound downwards
    h.sc.meshes.append(mesh)
    h.v._changed()
    h.camera([0, 0.5, 0], 9, [0.1, 0.9, 0.4])
    h.v._activate(mesh)
    h.v.ws.pen_world = 0.1
    h.key(Qt.Key_S)
    h.v.set_draw_tool("brush")
    h.stroke_world([[x, 0.5, 0.3 * np.sin(x)] for x in np.linspace(-2, 2, 30)])
    h.key(Qt.Key_S)
    c = [c for c in h.curves() if c.paint][0]
    assert c.plane_id == mesh.id
    lift = PAINT_LIFT * float(np.mean(c.widths))
    assert c.lift > 0 and abs(c.lift - lift) < 1e-9
    assert np.allclose(c.normals, [0, 1, 0], atol=1e-6), "normal of the mesh, towards the camera"
    assert np.allclose(c.points[:, 1], 0.5 + lift, atol=1e-4), "just above the surface"
    # older scene: paint without lift, normal towards the camera of that time
    path = os.path.join(tempfile.mkdtemp(), "malerei_mesh.sketch.json")
    h.sc.save(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    for st in raw["strokes"]:
        for cd in st["curves"]:
            if cd.get("paint"):
                cd.pop("lift")
                cd["points"] = [[x, 0.5, z] for x, _, z in cd["points"]]
                cd["normals"] = [[0.1, 0.9, 0.4]] * len(cd["points"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    from sketch_machine.geometry import Geometry as G
    import sketch_machine.geometry as geo_mod
    real = geo_mod.load
    geo_mod.load = lambda p, prim=None: G(v, np.array([[0, 1, 2], [0, 2, 3]]))  # mesh without file
    try:
        loaded = Scene.load(path)[0]
        again = [c for c in loaded.curves if c.paint][0]
        assert np.allclose(again.points[:, 1], 0.5 + lift, atol=1e-4) and np.allclose(again.normals, [0, 1, 0], atol=1e-6)
        loaded.save(path)
        twice = [c for c in Scene.load(path)[0].curves if c.paint][0]
        assert np.allclose(twice.points[:, 1], 0.5 + lift, atol=1e-4), "lifted only once"
    finally:
        geo_mod.load = real
