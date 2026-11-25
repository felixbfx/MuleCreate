"""Lofts (Shift+F) as a curve surface: robustness of the shape and the grid (feedback loft_test).

An open loft becomes a curve surface (border: first cross section, side line, last cross section, side line);
the middle cross sections are lines on it. The expected shape is that of a loft: across the cross sections smooth
curves through all cross sections (blended circular arcs), no tent between them – also after editing.
"""
import numpy as np

from tests.harness import H
from sketch_machine import components, loft, network
from sketch_machine.scene import PatchSurface


def _arc(center, rx, ry, z, n=40, flat=False, a0=0.0, a1=np.pi):
    """Arc in the plane z (upright) or on the ground (flat: in XZ, apex in +z)."""
    t = np.linspace(a0, a1, n)
    cx, cy = center
    if flat:
        return np.stack([cx + rx * np.cos(t), np.zeros(n), z + ry * np.sin(t)], axis=1)
    return np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t), np.full(n, z)], axis=1)


def _make_loft(h: H, sections):
    """Draw cross sections as lines and loft them like the artist (Shift+F in edge mode, in this order)."""
    sc = h.sc
    sc.new_stroke()
    lines = []
    for pts in sections:
        sc.add_curve(np.asarray(pts, float), None)
        lines.append(sc.curves[-1])
    h.v._changed()
    network.sync(sc)
    st = sc.owner_of(lines[0]).id
    h.v.set_select_mode(True)
    h.v.ws.component = "edge"
    h.v.edge_selection = [("line", st, c.id) for c in lines]
    h.v._pick_rank = {c.id: k for k, c in enumerate(lines)}
    h.v._track_pick_order = lambda: None
    h.v.do_loft()
    del h.v._track_pick_order
    surf = sc.surfaces[-1]
    assert isinstance(surf, PatchSurface), f"open loft is {type(surf).__name__}, not a curve surface"
    sc.sync_lofts()
    return surf, [sc.curve_by_id(c.id) for c in lines]


def _net_row(surf, k=0):
    """Grid row of the k-th cross net line (e.g. middle cross section)."""
    return surf.net["u"][k][1]


def _row_hard(surf, k=1):
    """Hardness along the k-th cross net line (0: bottom border)."""
    return surf._net_geo["hard_u"][k]


def _size(g):
    return float(np.linalg.norm(np.ptp(np.asarray(g).reshape(-1, 3), axis=0)))


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
    return np.linalg.norm(a + ab * t[:, None] - p, axis=1)


def _dist_to_mesh(pts, verts, tris):
    """Distance of each point to the triangle mesh (exact: inside the face, otherwise nearest edge)."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    n = np.cross(b - a, c - a)
    nn = np.maximum(np.einsum("ij,ij->i", n, n), 1e-18)
    out = []
    for p in np.asarray(pts, float):
        dot = np.einsum("ij,ij->i", p - a, n)
        q = p - n * (dot / nn)[:, None]
        w0 = np.einsum("ij,ij->i", np.cross(c - b, q - b), n) / nn
        w1 = np.einsum("ij,ij->i", np.cross(a - c, q - c), n) / nn
        inside = (w0 >= 0) & (w1 >= 0) & (1 - w0 - w1 >= 0)
        d = np.where(inside, np.abs(dot) / np.sqrt(nn), np.inf)
        pp = np.broadcast_to(p, a.shape)
        d = np.minimum(d, np.minimum(_seg_dist(pp, a, b), np.minimum(_seg_dist(pp, b, c), _seg_dist(pp, c, a))))
        out.append(float(d.min()))
    return np.array(out)


def _ideal(lines):
    """This is what the loft would look like: smooth curves through the (current) cross sections."""
    secs, _ = loft.prepare([c.points for c in lines])
    return loft.grid(secs)


def _deviation(surf, ideal, inner=0.15):
    """Largest distance of the surface from the ideal loft shape (without the border strips at the side lines)."""
    m = ideal.shape[1]
    lo, hi = int(m * inner), int(m * (1 - inner))
    verts, tris = surf.fill_mesh()[:2]
    pts = ideal[:, lo:hi + 1:3].reshape(-1, 3)
    return float(_dist_to_mesh(pts, verts, tris).max()) / _size(ideal)


UNEVEN = [_arc((0, 0), 1.0, 1.0, 0.0), _arc((0.4, 0), 2.2, 2.8, 1.4, n=55), _arc((0, 0), 1.2, 0.8, 4.0, n=31)]
FLAT_HIGH_FLAT = [_arc((0, 0), 4.0, 3.0, 0.0, flat=True), _arc((0, 0), 4.0, 5.0, 4.0), _arc((0, 0), 4.0, 3.0, 8.0, flat=True)]


def test_loft_shape_smooth_through_uneven_arcs(h: H):
    """Three uneven arcs (size, point count, distance): the surface swings round through all three like a loft."""
    surf, lines = _make_loft(h, UNEVEN)
    dev = _deviation(surf, _ideal(lines))
    assert dev < 0.01, f"surface deviates {dev:.1%} from the loft shape (tent instead of rounding?)"


def test_loft_shape_flat_ends_high_middle_arc(h: H):
    """Like the artist's curve surface 4: first and last arc lie on the ground, the middle one stands high – seen from
    the side a round arc, not a pyramid."""
    surf, lines = _make_loft(h, FLAT_HIGH_FLAT)
    dev = _deviation(surf, _ideal(lines))
    assert dev < 0.01, f"surface deviates {dev:.1%} from the loft shape (tent instead of rounding?)"


def test_loft_grid_follows_the_cross_sections(h: H):
    """Topology: the middle cross section lies on a grid row whose ends sit exactly on the points of the
    side lines; the grid rows do not cross it obliquely."""
    surf, lines = _make_loft(h, UNEVEN)
    g = surf.grid()
    att = surf.attached[lines[1].id]
    rows = np.asarray(att["uv"])[:, 0]
    assert np.ptp(rows) < 0.05, f"middle cross section runs obliquely through the grid (rows {rows.min():.2f} … {rows.max():.2f})"
    row = int(round(rows.mean()))
    ends = lines[1].points[[0, -1]]
    corner = np.array([g[row, 0], g[row, -1]])
    gap = min(np.linalg.norm(corner - ends, axis=1).max(), np.linalg.norm(corner - ends[::-1], axis=1).max())
    assert gap < 1e-3 * _size(g), f"grid row ends {gap:.4f} beside the points of the side lines"


def test_loft_raising_middle_arc_stays_loft(h: H):
    """Raise the middle arc as an edge: the side lines go along at their points, the surface stays on the
    arc and side lines and keeps swinging round (no tent, no gap to the outer line)."""
    surf, lines = _make_loft(h, FLAT_HIGH_FLAT)
    sc = h.sc
    mid = lines[1]
    sc.checkpoint()
    components.ComponentGroup(sc, [], [(mid, 0, len(mid.points) - 1)]).translate(np.array([0.0, 3.0, 0.0]))
    sc.sync_lofts()
    rails = [c for c in sc.patch_border(surf) if c.id not in (lines[0].id, lines[2].id)]
    assert len(rails) == 2
    for r in rails:
        knot = network.knot_at(r, mid.points[0]) if np.linalg.norm(r.points - mid.points[0], axis=1).min() < 1e-6 \
            else network.knot_at(r, mid.points[-1])
        assert np.linalg.norm(r.points[knot] - mid.points[[0, -1]], axis=1).min() < 1e-6, "side line goes along"
    verts, tris = surf.fill_mesh()[:2]
    size = _size(verts)
    for c in [mid] + rails:
        d = _dist_to_mesh(c.points, verts, tris).max()
        assert d < 0.005 * size, f"surface detaches from line {c.id} ({d / size:.2%})"
    # stay round: no kink at the (soft) cross section across it – the surface goes through cross sections and the
    # softly co-dragged side lines (the difference from the pure loft is distributed smoothly across the width)
    # (compared with the pure loft through the cross sections at the same subdivision: a tight rounding is not a tent)
    g = surf.grid()
    rows = [0, _net_row(surf), g.shape[0] - 1]
    secs, _ = loft.prepare([c.points for c in lines])
    ref = loft.interp_rows(secs, rows)
    row = rows[1]
    for f in (0.25, 0.5, 0.75):
        col, cr = int(round(f * (g.shape[1] - 1))), int(round(f * (ref.shape[1] - 1)))
        k = components.kink_deg(g[row::-1, col], g[row:, col])
        k_ref = components.kink_deg(ref[row::-1, cr], ref[row:, cr])
        assert k < k_ref + 8, f"tent at the cross section: {k:.0f}° kink, loft {k_ref:.0f}° (column {col})"


def test_loft_insert_point_and_slide_changes_nothing(h: H):
    """Insert a point on the middle arc and let it slide (S): the line keeps its shape – so does the surface."""
    surf, lines = _make_loft(h, UNEVEN)
    sc = h.sc
    mid = lines[1]
    before = surf.grid().copy()
    k = components.insert_point(mid, len(mid.points) // 2, 0.5)
    sc.sync_lofts()
    after_insert = surf.grid()
    d = float(np.abs(after_insert - before).max()) / _size(before)
    assert d < 1e-3, f"inserting a point changes the surface ({d:.2%})"
    st = sc.owner_of(mid).id
    h.v.point_selection = [("pt", st, mid.id, k)]
    h.frame(before.reshape(-1, 3))
    start = h.scrf(mid.points[k])
    h.v._begin_slide(start)
    for dx in (6, 14, 25, 40, 25, -10, -30):
        h.v._slide_update(type(start)(start.x() + dx, start.y()))
        sc.sync_lofts()
        g = surf.grid()
        d = float(np.abs(g - before).max()) / _size(before)
        assert d < 2e-3, f"sliding by {dx} px moves the surface by {d:.2%}"
    h.v.finish_slide(True)


def test_loft_dragging_point_shapes_softly(h: H):
    """Drag an inserted point on the middle arc: the surface follows the arc, the grid does not fold."""
    surf, lines = _make_loft(h, UNEVEN)
    sc = h.sc
    mid = lines[1]
    k = components.insert_point(mid, len(mid.points) // 3, 0.5)
    sc.sync_lofts()
    components.ComponentGroup(sc, [(mid, k)], []).translate(np.array([0.0, 0.8, 0.3]))
    sc.sync_lofts()
    verts, tris = surf.fill_mesh()[:2]
    size = _size(verts)
    d = _dist_to_mesh(mid.points, verts, tris).max()
    assert d < 0.005 * size, f"surface lies {d / size:.2%} beside the dragged arc"
    g = surf.grid()
    for axis in (0, 1):  # neighbouring cell edges never point against each other (no fold)
        e = np.diff(g, axis=axis)
        dots = np.einsum("...i,...i->...", e[:-1] if axis == 0 else e[:, :-1], e[1:] if axis == 0 else e[:, 1:])
        assert dots.min() > 0, f"grid folds (direction {axis})"


def test_loft_save_load_stays_loft(h: H):
    """Saving and loading: the curve surface still knows it is a loft (same shape, cross sections shape it)."""
    import os, tempfile
    from sketch_machine.scene import Scene
    surf, lines = _make_loft(h, UNEVEN)
    g = surf.grid().copy()
    path = os.path.join(tempfile.mkdtemp(), "loft.sketch.json")
    h.sc.save(path)
    new = Scene.load(path)[0]
    new.sync_lofts()
    s2 = next(s for s in new.surfaces if isinstance(s, PatchSurface))
    assert s2.net and len(s2.net["u"]) == 1, "cross section known after loading"
    d = float(np.abs(s2.grid() - g).max()) / _size(g)
    assert d < 1e-3, f"shape different after loading ({d:.2%})"


def test_loft_old_scene_is_recognized(h: H):
    """The artist's loft_test (saved before v0.99.4, without loft_rows): the lofts are recognized on loading and swing round
    through their middle arc. (Their side lines are saved lines – with the loft shape of that time; the surface
    follows them, so no comparison with a freshly computed loft shape here.)"""
    import os
    path = os.path.join(os.path.dirname(__file__), "scenes", "loft_old.sketch.json")
    h.load(path)
    sc = h.sc
    sc.sync_lofts()
    lofts = [s for s in sc.surfaces if isinstance(s, PatchSurface) and s.net and s.net["u"]]
    assert len(lofts) == 4, f"{len(lofts)} of 4 lofts recognized"
    for s in lofts:
        verts, tris = s.fill_mesh()[:2]
        size = _size(verts)
        mids = [sc._chain_points([i for cid in ids for i in sc.split_pieces(cid)]) for ids, _ in s.net["u"]]
        for c in mids + [b.points for b in sc.patch_border(s)]:
            d = _dist_to_mesh(c, verts, tris).max() / size
            assert d < 0.005, f"{s.name}: {d:.1%} beside one of its lines"
        g = s.grid()
        row = _net_row(s)
        bend = max(components.kink_deg(g[row::-1, c], g[row:, c]) for c in range(2, g.shape[1] - 2, 4))
        # the saved side lines still have the Catmull-Rom shape (before v0.99.14): blended with the circular-arc loft
        # a slight kink remains at the arc (up to ~15°); freshly created lofts: see below (< 5°)
        assert bend < 20, f"{s.name}: kinks at the middle arc ({bend:.0f}°) instead of passing round through it"


def test_loft_fresh_round_through_the_middle_arc(h: H):
    """Freshly created loft (the artist's profiles: flat, high, flat): the surface runs through the middle cross
    section without a kink (circular arcs blended; with Catmull-Rom it was ~19°)."""
    for sections in (FLAT_HIGH_FLAT, UNEVEN):
        h.new_scene()
        surf, _ = _make_loft(h, sections)
        g = surf.grid()
        row = _net_row(surf)
        bend = max(components.kink_deg(g[row::-1, c], g[row:, c]) for c in range(2, g.shape[1] - 2, 4))
        assert bend < 5, f"kinks at the middle cross section ({bend:.1f}°)"


def test_loft_delete_cross_section(h: H):
    """Delete the middle cross section: the surface loses it and stays on all remaining lines (border), nothing
    breaks."""
    from tests.test_topology import _delete_lines
    surf, lines = _make_loft(h, UNEVEN)
    _delete_lines(h, [lines[1]])
    assert surf in h.sc.surfaces and surf.net is not None and not surf.net["u"]
    g = surf.grid()
    assert np.isfinite(g).all()
    verts, tris = surf.fill_mesh()[:2]
    size = _size(verts)
    for c in h.sc.patch_border(surf):
        d = _dist_to_mesh(c.points, verts, tris).max()
        assert d < 0.005 * size, f"surface detaches from border line {c.id} ({d / size:.2%})"


def test_loft_raise_undo(h: H):
    """Raise and undo: the same shape as before."""
    surf, lines = _make_loft(h, FLAT_HIGH_FLAT)
    sc = h.sc
    before = surf.grid().copy()
    sc.checkpoint()
    components.ComponentGroup(sc, [], [(lines[1], 0, len(lines[1].points) - 1)]).translate(np.array([0.0, 2.0, 1.0]))
    sc.sync_lofts()
    sc.undo()
    sc.sync_lofts()
    s = next(x for x in sc.surfaces if x.id == surf.id)
    d = float(np.abs(s.grid() - before).max()) / _size(before)
    assert d < 1e-6, f"after undo {d:.2%} different"


def test_line_on_surface_slide_changes_nothing(h: H):
    """An ordinary line on the surface (lengthwise, from cross section to cross section): insert a point and slide – the
    surface stays as it is (the line keeps its shape)."""
    surf, lines = _make_loft(h, UNEVEN)
    sc = h.sc
    g = surf.grid()
    col = g[:, g.shape[1] // 3].copy()
    sc.add_curve(col, surf.id)
    line = sc.curves[-1]
    sc.sync_lofts()
    assert line.id in surf.attached
    before = surf.grid().copy()
    k = components.insert_point(line, len(line.points) // 2, 0.5)
    sc.sync_lofts()
    st = sc.owner_of(line).id
    h.v.point_selection = [("pt", st, line.id, k)]
    h.frame(before.reshape(-1, 3))
    start = h.scrf(line.points[k])
    h.v._begin_slide(start)
    for dy in (5, 12, 20, -8):
        h.v._slide_update(type(start)(start.x(), start.y() + dy))
        sc.sync_lofts()
        d = float(np.abs(surf.grid() - before).max()) / _size(before)
        assert d < 2e-3, f"sliding by {dy} px moves the surface by {d:.2%}"
    h.v.finish_slide(True)


def test_loft_hard_cross_section_kinks_the_surface(h: H):
    """Set the middle cross section hard (edge mode, Shift+K): the side lines kink at it, the loft is split there
    – the surface has a fold exactly at the cross section and stays on all lines. Soft (Alt+K): round again."""
    surf, lines = _make_loft(h, UNEVEN)
    sc = h.sc
    mid = lines[1]

    def kink():
        g = surf.grid()
        col, row = g.shape[1] // 2, _net_row(surf)
        return components.kink_deg(g[row::-1, col], g[row:, col])
    soft_kink = kink()
    # coarse grid, strong rounding: a few degrees between two grid pieces are not a kink (hard only from 15°)
    assert soft_kink < 12 and not _row_hard(surf).any(), f"round at first ({soft_kink:.0f}°)"
    h.frame(surf.grid().reshape(-1, 3))
    h.edit_mode("edge")
    h.v.edge_selection = [("line", sc.owner_of(mid).id, mid.id)]
    before = mid.points.copy()
    h.v.set_hardness(1.0)
    sc.sync_lofts()
    assert np.array_equal(mid.points, before), "the cross section itself stays"
    # fully hard = both sides straight towards the point; how hard that is, the angle tells (a flat side line
    # kinks only ~31° -> 0.55), proportionally across the cross section
    assert _row_hard(surf).max() > 0.99 and _row_hard(surf).min() > 0.5, "cross section is hard"
    assert kink() > 30, f"fold at the cross section ({kink():.0f}°)"
    verts, tris = surf.fill_mesh()[:2]
    size = _size(verts)
    for c in [mid] + sc.patch_border(surf):
        d = _dist_to_mesh(c.points, verts, tris).max()
        assert d < 0.005 * size, f"surface detaches from line {c.id} ({d / size:.2%})"
    h.v.set_hardness(0.0)
    sc.sync_lofts()
    assert not _row_hard(surf).any() and kink() < 12, f"round again ({kink():.0f}°)"


def test_loft_raise_cross_section_as_whole_line(h: H):
    """loft_test_v002: select the middle cross section as a whole line (edge mode) and raise it with the gizmo – the
    side lines run through its ends and must go along (same points stay connected), the surface
    stays on all lines."""
    from sketch_machine import edges
    surf, lines = _make_loft(h, FLAT_HIGH_FLAT)
    sc = h.sc
    mid = lines[1]
    network.sync(sc)
    ends = [network.pid_of(mid, 0), network.pid_of(mid, len(mid.points) - 1)]
    group = edges.handles_for(sc, [edges.curve_to_ref(sc, mid)])[0]
    for _ in range(3):  # like the gizmo: in steps
        group.translate(np.array([0.0, 1.0, 0.3]))
    sc.sync_lofts()
    network.sync(sc)
    assert [network.pid_of(mid, 0), network.pid_of(mid, len(mid.points) - 1)] == ends, "ends stay the same points"
    refs = network.refs_by_pid(sc)
    for pid in ends:
        pts = [o.points[j] for o, j in refs[pid]]
        assert len(pts) >= 2 and max(np.linalg.norm(p - pts[0]) for p in pts) < 1e-9, "side line went along"
    verts, tris = surf.fill_mesh()[:2]
    size = _size(verts)
    for c in [mid] + sc.patch_border(surf):
        d = _dist_to_mesh(c.points, verts, tris).max()
        assert d < 0.005 * size, f"surface detaches from line {c.id} ({d / size:.2%})"


def test_fold_gets_hard_light(h: H):
    """Light normals per triangle corner: at a fold (kink from 45°) each side gets its own normal, a
    smooth rounding stays smooth (as averaged)."""
    from sketch_machine import renderer
    u, v = np.meshgrid(np.linspace(-1, 1, 21), np.linspace(0, 1, 11), indexing="ij")
    roof = np.stack([u, 0.8 * (1 - np.abs(u)), v], axis=2)  # roof: 77° kink in the middle
    tube = np.stack([np.cos(u * 1.5), np.sin(u * 1.5), v], axis=2)  # smooth rounding
    for grid, folded in ((roof, True), (tube, False)):
        verts, tris, _ = loft.build(None, False, g=grid)
        corner, split = renderer._corner_normals(verts, tris)
        smooth = renderer._smooth_normals(verts, tris)[tris]
        dev = np.degrees(np.arccos(np.clip(np.einsum("tki,tki->tk", corner, smooth), -1, 1)))
        if folded:
            on_ridge = np.abs(verts[tris][..., 0]) < 1e-9
            a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
            own = np.cross(b - a, c - a)
            own = np.repeat((own / np.linalg.norm(own, axis=1, keepdims=True))[:, None], 3, axis=1)
            off = np.degrees(np.arccos(np.clip(np.einsum("tki,tki->tk", corner, own), -1, 1)))
            assert off[on_ridge].max() < 1.0 and dev[on_ridge].min() > 20, "at the fold: normal of its own side"
            assert split[on_ridge].min() > 0.3 and dev[~on_ridge].max() < 1e-3, (split[on_ridge].min(), dev[~on_ridge].max())
        else:
            assert dev.max() < 0.5 and split.max() < 1e-9, "smooth stays smooth"


def test_loft_hard_cross_section_with_lengthwise_line(h: H):
    """loft_test_v002: hard cross section, above it a line lengthwise from cross section to cross section, connected at the
    crossing. The cross-section row stays exactly on the cross section (no bulge between support points), and
    raising the crossing point deforms the surface softly – no needle, the surface stays on all lines. The cross section
    has more points than support points are taken (as in the scene)."""
    surf, lines = _make_loft(h, [FLAT_HIGH_FLAT[0], _arc((0, 0), 4.0, 5.0, 4.0, n=60), FLAT_HIGH_FLAT[2]])
    sc = h.sc
    mid = lines[1]
    h.frame(surf.grid().reshape(-1, 3))
    h.edit_mode("edge")
    h.v.edge_selection = [("line", sc.owner_of(mid).id, mid.id)]
    h.v.set_hardness(1.0)
    sc.sync_lofts()
    g = surf.grid()
    col = g[:, g.shape[1] // 2 + 3].copy()  # lengthwise, crosses the cross section
    sc.add_curve(col, surf.id)
    line = sc.curves[-1]
    sc.sync_lofts()
    network.connect_crossings(sc, line, [mid], 1e-3 * _size(g))
    sc.sync_lofts()
    g, b = surf.grid(), surf.base_grid()
    size = _size(g)
    row = _net_row(surf)
    dev = float(np.linalg.norm(g[row] - b[row], axis=1).max()) / size
    assert dev < 1e-3, f"cross-section row deviates {dev:.2%}"
    shared = set(line.pids.values()) & set(mid.pids.values())
    assert len(shared) == 1, "crossing is a shared point"
    k = next(j for j, p in line.pids.items() if p in shared)
    top = g[..., 1].max()
    sc.checkpoint()
    components.ComponentGroup(sc, [(line, k)], []).translate(np.array([0.0, 1.0, 0.0]))
    sc.sync_lofts()
    g = surf.grid()
    assert g[..., 1].max() < top + 1.0 + 0.02 * size, f"needle: surface rises {g[..., 1].max() - top:.2f} instead of 1"
    verts, tris = surf.fill_mesh()[:2]
    for c in [mid, line] + sc.patch_border(surf):
        d = _dist_to_mesh(c.points[::2], verts, tris).max()
        assert d < 0.01 * size, f"surface detaches from line {c.id} ({d / size:.2%})"


def test_crossing_next_to_corner_one_point(h: H):
    """If a new line crosses another very close to its own corner (stroke kinks over a fold), this becomes
    one point – no second one right beside it (which would be pulled into a needle when dragging)."""
    sc = h.sc
    sc.new_stroke()
    t = np.linspace(0, 1, 81)
    sc.add_curve(np.stack([t * 4, 1 - np.abs(t * 4 - 2) * 0.5, np.zeros_like(t)], 1), None)  # roof, corner at x = 2
    roof = sc.curves[-1]
    sc.add_curve(np.stack([np.full(21, 2.06), np.linspace(-1, 2, 21), np.zeros(21)], 1), None)  # crosses at x = 2.06
    other = sc.curves[-1]
    network.sync(sc)
    inner = [k for k in components.knots(roof) if 0 < k < len(roof.points) - 1]
    assert len(inner) == 1 and abs(roof.points[inner[0], 0] - 2.0) < 0.06, "corner recognized"
    network.connect_crossings(sc, roof, [other], 0.05, radius=0.2)
    network.sync(sc)
    inner = [k for k in components.knots(roof) if 0 < k < len(roof.points) - 1]
    assert len(inner) == 1, f"{len(inner)} points at the crossing"
    assert network.pid_of(roof, inner[0]) in set(other.pids.values()), "the one point is the crossing"


def test_loft_v002_cross_section_row_without_bulge(h: H):
    """The artist's loft_test_v002 (hard cross section with 53 points, lengthwise line above it): the cross-section row lies exactly
    on the cross section – before, a grid point between two support points bulged out by 0.39."""
    import os
    h.load(os.path.join(os.path.dirname(__file__), "scenes", "loft_v002.sketch.json"))
    sc = h.sc
    sc.sync_lofts()
    s = next(x for x in sc.surfaces if isinstance(x, PatchSurface) and x.net and x.net["u"])
    g, b = s.grid(), s.base_grid()
    row = _net_row(s)
    dev = float(np.linalg.norm(g[row] - b[row], axis=1).max())
    assert dev < 0.01, f"cross-section row deviates {dev:.3f}"


def test_stroke_over_hard_cross_section(h: H):
    """Stroke with the mouse over a loft with a hard cross section (the artist's procedure): every point of the stroke gets
    its own position on the surface – also on the outer side of the fold, where the perpendicular projection falls beside both
    adjacent triangles (before: nearest grid point, several points on the same spot, surface distorted)."""
    surf, lines = _make_loft(h, FLAT_HIGH_FLAT)
    sc = h.sc
    mid = lines[1]
    h.frame(surf.grid().reshape(-1, 3))
    h.edit_mode("edge")
    h.v.edge_selection = [("line", sc.owner_of(mid).id, mid.id)]
    h.v.set_hardness(1.0)
    sc.sync_lofts()
    h.v.edge_selection = []
    h.draw_mode()
    h.v._activate(surf)
    h.frame(surf.grid().reshape(-1, 3), back=(0.15, 1.0, 0.1), margin=1.4)
    g = surf.grid()
    col = g[:, g.shape[1] // 2 + 4]
    before = {c.id for c in h.curves()}
    h.stroke_world(np.vstack([(col[0] + (col[0] - col[2]) * 0.6)[None], col, (col[-1] + (col[-1] - col[-3]) * 0.6)[None]]))
    sc.sync_lofts()
    new = [c for c in h.curves() if c.id not in before and not c.paint and c.id in surf.attached]
    assert new, "stroke lies on the surface"
    size = _size(surf.grid())
    for c in new:
        uv = surf.attached[c.id]["uv"]
        dup = len(uv) - len(np.unique(np.round(uv, 6), axis=0))
        assert dup == 0, f"{dup} points share one spot of the surface"
        dev = np.linalg.norm(loft.evaluate_grid(uv, surf.grid(), False) - c.points, axis=1).max() / size
        assert dev < 0.002, f"stroke {dev:.2%} beside the surface"


def test_loft_divide_on_curve_surface(h: H):
    """Alt+D on an open loft (curve surface): whole surface selected -> one new cross line between every pair of
    neighbouring cross sections, exactly on the surface, connected as a net line; the shape stays. Two neighbouring cross
    sections selected -> only between them."""
    from tests.harness import Qt
    surf, lines = _make_loft(h, UNEVEN)
    sc = h.sc
    before = surf.grid().copy()
    h.edit_mode("face")
    h.v.selection = [surf.id]
    h.v.edge_selection = []
    h.v.divide_loft()
    assert len(surf.net["u"]) == 3, f"{len(surf.net['u'])} cross lines instead of 3"
    verts, tris = surf.fill_mesh()[:2]
    dev = float(_dist_to_mesh(before.reshape(-1, 3)[::7], verts, tris).max()) / _size(before)
    assert dev < 0.005, f"shape changes ({dev:.2%})"
    h.edit_mode("edge")
    h.v.selection = []
    h.v.edge_selection = [("line", sc.owner_of(c).id, c.id) for c in (lines[1],)] + \
        [("line", sc.owner_of(sc.curve_by_id(i)).id, i) for i in sorted(sc.net_rows_lines(surf)[3])]
    h.v.divide_loft()
    assert len(surf.net["u"]) == 4, f"{len(surf.net['u'])} cross lines instead of 4"


def test_fold_stays_with_finer_net(h: H):
    """Net resolution finer/coarser (also odd factors): the fold at a hard cross section stays exactly on one
    grid row (before, resampling rounded it off when no new row lay on it)."""
    surf, lines = _make_loft(h, UNEVEN)
    sc = h.sc
    mid = lines[1]
    h.frame(surf.grid().reshape(-1, 3))
    h.edit_mode("edge")
    h.v.edge_selection = [("line", sc.owner_of(mid).id, mid.id)]
    h.v.set_hardness(1.0)
    sc.sync_lofts()
    g = surf.grid()
    found = {}
    for factor in (1.0, 1.7, 0.7):
        surf.resolution = factor
        verts = surf.fill_mesh()[0]
        m2 = int(round((g.shape[1] - 1) * factor)) + 1
        g2 = verts.reshape(-1, m2, 3)
        col = m2 // 2
        found[factor] = max(components.kink_deg(g2[i::-1, col], g2[i:, col]) for i in range(2, len(g2) - 2))
        dist = _dist_to_mesh(mid.points[::3], *surf.fill_mesh()[:2]).max() / _size(verts)
        assert dist < 0.003, f"net x{factor}: surface {dist:.2%} beside the cross section"
    assert found[1.0] > 30, found
    for factor in (1.7, 0.7):
        assert found[factor] > 0.8 * found[1.0], f"net x{factor}: fold {found[factor]:.0f}° instead of {found[1.0]:.0f}°"


def test_loft_lengthwise_lines_at_knots_from_5_degrees(h: H):
    """Cross sections with a point in the middle: kink 3° -> only the side edges, one surface; 10° -> a
    lengthwise line there too, one surface; 20° -> lengthwise line and split along it into two curve surfaces."""
    for deg, rails, faces in ((3.0, 2, 1), (10.0, 3, 1), (20.0, 3, 2)):
        h.new_scene()
        t = np.radians(deg)
        secs = []
        for z in (0.0, 2.0):
            a = np.linspace([-2.0, 0, z], [0.0, 0, z], 21)
            b = np.linspace([0.0, 0, z], [2 * np.cos(t), 2 * np.sin(t), z], 21)[1:]
            secs.append(np.vstack([a, b]))
        sc = h.sc
        sc.new_stroke()
        lines = []
        for pts in secs:
            sc.add_curve(pts, None)
            lines.append(sc.curves[-1])
            lines[-1].knots = (len(pts), (20,))
        h.v._changed()
        network.sync(sc)
        st = sc.owner_of(lines[0]).id
        h.v.set_select_mode(True)
        h.v.ws.component = "edge"
        h.v.edge_selection = [("line", st, c.id) for c in lines]
        h.v._pick_rank = {c.id: k for k, c in enumerate(lines)}
        h.v._track_pick_order = lambda: None
        h.v.do_loft()
        del h.v._track_pick_order
        along = [c for c in sc.curves if abs((c.points[-1] - c.points[0])[2]) > 1.9]
        assert len(along) == rails, f"{deg}°: {len(along)} lengthwise lines instead of {rails}"
        assert len(sc.surfaces) == faces, f"{deg}°: {len(sc.surfaces)} surfaces instead of {faces}"
        assert all(isinstance(f, PatchSurface) for f in sc.surfaces)


def test_loft_square_four_surfaces_circle_one_tube(h: H):
    """Closed: square cross sections (corners 90°) -> four curve surfaces; circles without a hard point -> one tube."""
    from tests.harness import circle
    for shape, faces in (("square", 4), ("circle", 1)):
        h.new_scene()
        secs = []
        for y in (0.0, 2.0):
            if shape == "square":
                q = [[0, y, 0], [1, y, 0], [1, y, 1], [0, y, 1], [0, y, 0]]
                secs.append(np.vstack([np.linspace(a, b, 11)[:-1] for a, b in zip(q, q[1:])] + [np.array([q[0]], float)]))
            else:
                secs.append(circle([0, y, 0], 1.0, 65))
        surf, _lines = _make_loft_any(h, secs)
        assert len(h.sc.surfaces) == faces, f"{shape}: {len(h.sc.surfaces)} surfaces instead of {faces}"
        if shape == "square":
            for f in h.sc.surfaces:
                v = f.fill_mesh()[0]
                p = v - v.mean(axis=0)
                assert np.linalg.svd(p, full_matrices=False)[1][2] < 1e-4, f"wall {f.name} not flat"


def _make_loft_any(h: H, sections):
    sc = h.sc
    sc.new_stroke()
    lines = []
    for pts in sections:
        sc.add_curve(np.asarray(pts, float), None)
        lines.append(sc.curves[-1])
    h.v._changed()
    network.sync(sc)
    st = sc.owner_of(lines[0]).id
    h.v.set_select_mode(True)
    h.v.ws.component = "edge"
    h.v.edge_selection = [("line", st, c.id) for c in lines]
    h.v._pick_rank = {c.id: k for k, c in enumerate(lines)}
    h.v._track_pick_order = lambda: None
    h.v.do_loft()
    del h.v._track_pick_order
    sc.sync_lofts()
    return sc.surfaces[-1], lines


def _rings_with_points(h: H):
    """Two rings (closed lines) with points at every quarter, like the hull rings of the harbor submarine."""
    sc = h.sc
    sc.new_stroke()
    t = np.linspace(0, 2 * np.pi, 121)
    rings = []
    for z, r in ((0.0, 1.0), (2.0, 1.6)):
        sc.add_curve(np.stack([r * np.cos(t), r * np.sin(t), np.full_like(t, z)], axis=1), None)
        rings.append(sc.curves[-1])
        for k in (30, 60, 90):
            components.insert_point(rings[-1], k, 0.0)
    h.v._changed()
    network.sync(sc)
    h.v.set_select_mode(True)
    h.v.ws.component = "edge"
    return rings


def test_loft_through_selected_pieces_takes_only_the_pieces(h: H):
    """Artist 2026-11-24 (harbor, recording 15:49): one piece each of two rings selected, Shift+F – the loft took both
    whole rings and filled a twisted cone. A selected piece is the cross-section: the surface spans only between the
    two pieces. One Ctrl+Z takes back the surface and the split."""
    sc = h.sc
    rings = _rings_with_points(h)
    st = sc.owner_of(rings[0]).id
    assert components.segments(rings[0])[1] == (30, 60)
    lines_before = len(sc.curves)
    h.v.edge_selection = []
    h.v.seg_selection = [("seg", st, c.id, 30, 60) for c in rings]  # the quarter from 90° to 180°: x <= 0, y >= 0
    h.v.do_loft()
    assert len(sc.surfaces) == 1, "one loft"
    h.v._changed()
    verts = sc.surfaces[-1].fill_mesh()[0]
    assert verts[:, 0].max() < 0.02 and verts[:, 1].min() > -0.02, \
        f"surface leaves the selected quarter: x up to {verts[:, 0].max():.2f}, y down to {verts[:, 1].min():.2f}"
    assert verts[:, 0].min() < -1.5 and verts[:, 1].max() > 1.5, "surface reaches over the whole piece"
    assert np.ptp(verts[:, 2]) > 1.99, "from ring to ring"
    h.v.undo()
    assert not sc.surfaces and len(sc.curves) == lines_before, "one undo step: surface gone, rings whole again"


def test_loft_that_fails_leaves_the_lines_unsplit(h: H):
    """A single selected piece is no loft (two cross-sections needed): the line it belongs to stays whole."""
    sc = h.sc
    rings = _rings_with_points(h)
    st = sc.owner_of(rings[0]).id
    ids = [c.id for c in sc.curves]
    undo = len(sc._undo)
    seg = ("seg", st, rings[0].id, 30, 60)
    h.v.edge_selection = []
    h.v.seg_selection = [seg]
    h.v.do_loft()
    assert not sc.surfaces
    assert [c.id for c in sc.curves] == ids, "lines unchanged"
    assert len(sc._undo) == undo, "no undo step left behind"
    assert h.v.seg_selection == [seg], "selection as before"
