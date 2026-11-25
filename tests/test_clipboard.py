"""Copy and paste: pasted surfaces follow their pasted lines, the original stays untouched."""
import os

import numpy as np

from tests.harness import H, Qt, line
from sketch_machine import edges
from sketch_machine.clipboard import Clipboard, paste


def _bend(c, h):  # raise the middle, ends stay (outline stays closed)
    c.points = c.points + np.outer(np.sin(np.linspace(0, np.pi, len(c.points))), [0.0, h, 0.0])


def _check(sc, surf, lines, move_index):
    clip = Clipboard()
    clip.copy(sc, [surf], [("line", sc.owner_of(c).id, c.id) for c in lines])
    objs, refs = paste(sc, clip, "elements")
    new_surf = next(o for o in objs if o.name.endswith("_copy"))
    new_lines = [edges.ref_to_curve(sc, r) for r in refs]
    sc.sync_lofts()
    orig, new = surf.fill_mesh()[0].copy(), new_surf.fill_mesh()[0].copy()
    _bend(new_lines[move_index], 0.7)
    sc.sync_lofts()
    # (the displayed net depends on the curvature: bent, it also has a different point count)
    same = lambda a, b: a.shape == b.shape and np.allclose(a, b)
    assert not same(new_surf.fill_mesh()[0], new), "pasted surface follows its pasted line"
    assert same(surf.fill_mesh()[0], orig), "original stays untouched"


def _square(sc):
    sc.new_stroke()
    P = [np.array(p, float) for p in ([0, 0, 0], [2, 0, 0], [2, 0, 2], [0, 0, 2])]
    out = []
    for a, b in zip(P, P[1:] + P[:1]):
        sc.add_curve(line(a, b, 10), None)
        out.append(sc.curves[-1])
    return out


def test_fill_surface(h: H):
    sc = h.sc
    lines = _square(sc)
    f = sc.create_fill([(c, False) for c in lines])
    sc.sync_lofts()
    _check(sc, f, lines, 1)


def test_loft_with_rails(h: H):
    sc = h.sc
    sc.new_stroke()
    t = np.linspace(0, 1, 12)
    secs = []
    for z in (0.0, 1.0, 2.0):
        sc.add_curve(np.stack([t * 2, 0.4 * np.sin(t * np.pi), np.full_like(t, z)], axis=1), None)
        secs.append(sc.curves[-1])
    lo = sc.create_loft([("lines", [c.id]) for c in secs], [c.points for c in secs])
    sc.add_loft_rails(lo, np.array([0, 0, 1.0]))
    sc.sync_lofts()
    rails = [sc.curve_by_id(cid) for cid, _ in lo.rails]
    _check(sc, lo, secs + rails, 2)


def test_curve_patch(h: H):
    sc = h.sc
    sc.new_stroke()
    t = np.linspace(0, 1, 12)
    sides = [np.stack([t * 2, 0.3 * np.sin(t * np.pi), t * 0], 1), np.stack([np.full_like(t, 2), 0.3 * np.sin(t * np.pi), t * 2], 1),
             np.stack([2 - t * 2, 0.3 * np.sin(t * np.pi), np.full_like(t, 2)], 1), np.stack([t * 0, 0.3 * np.sin(t * np.pi), 2 - t * 2], 1)]
    lines = []
    for s in sides:
        sc.add_curve(s, None)
        lines.append(sc.curves[-1])
    ps = sc.create_patch(lines)
    sc.sync_lofts()
    _check(sc, ps, lines, 1)


def test_copy_paste_in_the_program(h: H):
    sc = h.sc
    lines = _square(sc)
    for c in lines:
        c.plane_id = sc.planes[0].id
    f = sc.create_fill([(c, False) for c in lines])
    h.v._changed()
    h.camera([1, 0, 1], 6, [0.2, 0.9, 0.4])
    h.edit_mode("face")
    h.v.select_loop(h.scrf([1.0, 0.0, 1.0]), "replace")
    assert h.v.selection == [f.id] and len(h.v.edge_selection) == 4
    h.key(Qt.Key_C, Qt.ControlModifier)
    h.key(Qt.Key_V, Qt.ControlModifier)
    new = sc.surfaces[-1]
    originals = {c.id for c in lines}
    linked = [cid for cid, _ in new.links]
    assert len(sc.surfaces) == 2 and len(linked) == 4
    assert all(cid not in originals and h.sc.curve_by_id(cid) is not None for cid in linked)


def test_duplicate_lines_with_surfaces_own_object(h: H):
    """Ctrl+D with lines and surfaces (e.g. a whole wing): the copied lines go into a new stroke object
    next to the original (same group), not into the original object; copy moved -> original stays."""
    sc = h.sc
    lines = _square(sc)
    st = sc.owner_of(lines[0])
    h.v.edge_selection = [("line", st.id, c.id) for c in lines]
    h.v.do_patch()
    surf = sc.surfaces[-1]
    h.v._changed()
    h.edit_mode("edge")
    h.v.selection = [surf.id]
    h.v.edge_selection = [("line", st.id, c.id) for c in lines]
    before = {c.id: c.points.copy() for c in lines}
    h.v.duplicate("elements")
    new_st = {r[1] for r in h.v.edge_selection}
    assert len(new_st) == 1 and st.id not in new_st, "copied lines in a new object"
    assert len(st.curves) == 4, "original object keeps only its lines"
    h.v.transform_target.translate(np.array([0.0, 1.0, 0.0]))
    h.v._changed()
    assert all(np.allclose(c.points, before[c.id]) for c in lines), "original lines stay"


def test_copy_line_object_into_another_scene(h: H):
    """Line object copied in the outliner, pasted into a new scene: it takes its surfaces along (owner = the copy),
    and they are still connected to the pasted lines."""
    from sketch_machine.scene import Scene
    sc = h.sc
    lines = _square(sc)
    st = sc.owner_of(lines[0])
    st.name = "U_Boot"
    sc.create_fill([(c, False) for c in lines])
    h.v._changed()
    assert sc.surfaces[-1].owner == st.id
    h.v.selection = [st.id]
    assert h.v.copy_selection(objects_only=True)
    h.v.set_scene(Scene())  # like File > New / Open: the clipboard stays
    h.v.paste("objects")
    new = h.sc
    new_st = next(s for s in new.strokes if s.name == st.name)
    assert len(new_st.curves) == 4 and len(new.surfaces) == 1
    surf = new.surfaces[0]
    assert surf.owner == new_st.id
    assert {cid for cid, _ in surf.links} == {c.id for c in new_st.curves}


def test_reference_objects(h: H):
    """File > Insert reference object: person 1.80 m tall, standing on the ground; found again from another Mule folder."""
    import os
    from sketch_machine import paths
    from sketch_machine.scene import Scene
    sizes = {}
    for name, _ in paths.REFERENCES:
        h.w.import_reference(name)
        lo, hi = h.sc.meshes[-1].world_bounds()
        assert abs(lo[1]) < 1e-6, name
        sizes[name] = hi - lo
    assert abs(sizes["person_1m80"][1] - 1.80) < 1e-4
    assert 4.4 < sizes["car"][0] < 4.6 and 9 < sizes["house"][1] < 10
    data = h.sc.to_dict(os.path.join(paths.ROOT, "elsewhere"))
    for m in data["meshes"]:  # as if the scene was saved by a Mule in another folder
        m["path"] = "C:/Old/Mule/sketch_machine/assets/reference/" + os.path.basename(m["path"])
    again = Scene.from_dict(data, paths.ROOT)
    assert not again.load_warnings and all(not m.missing for m in again.meshes)


def test_copy_between_two_mule_windows(h: H):
    """Second Mule window: the copy goes through the system clipboard as a small scene – a null with a line object,
    its surface and a reference mesh below arrives whole (hierarchy, surface owner, links to the pasted lines)."""
    from sketch_machine import clipboard as cb, paths
    from sketch_machine.clipboard import Clipboard
    from sketch_machine.scene import Scene
    sc = h.sc
    lines = _square(sc)
    st = sc.owner_of(lines[0])
    st.name = "Sub_0"
    sc.create_fill([(c, False) for c in lines])
    h.w.import_reference("person_1m80")
    mesh = sc.meshes[-1]
    from sketch_machine.scene import NullObject
    null = NullObject("U_Boot", np.zeros(3), np.eye(3))
    sc.nulls.append(null)
    st.parent = mesh.parent = null.id
    h.v._changed()
    h.v.selection = [null.id]
    assert h.v.copy_selection(objects_only=True)
    own = cb.INSTANCE
    try:
        cb.INSTANCE = "other window"
        h.v.set_scene(Scene())
        h.v.ws.clipboard = Clipboard()  # the other window has nothing of its own
        h.v.paste("objects")
    finally:
        cb.INSTANCE = own
    new = h.sc
    new_null = next(n for n in new.nulls if n.name == "U_Boot")
    new_st = next(s for s in new.strokes if s.name == "Sub_0")
    assert new_st.parent == new_null.id and len(new_st.curves) == 4
    assert len(new.meshes) == 1 and new.meshes[0].parent == new_null.id and not new.meshes[0].missing
    assert os.path.samefile(new.meshes[0].path, os.path.join(paths.REFERENCE_DIR, "person_1m80.usda"))
    surf = new.surfaces[0]
    assert surf.owner == new_st.id and {cid for cid, _ in surf.links} == {c.id for c in new_st.curves}
