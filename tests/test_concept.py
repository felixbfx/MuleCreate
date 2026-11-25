"""Topology concept (plan document, section "test criteria"): the artist's test scene topology.sketch.json, tasks 1-5.

Hierarchy: points -> real lines -> surfaces -> fog/painting; nothing further down moves anything further up.
The scene's red brush strokes are the templates: real lines are drawn along them, the way the artist would
do it (stroke a little past the surface edge). All checks of a task must pass.
"""
import numpy as np

from PySide6.QtCore import QPointF

from sketch_machine.i18n import tr
from tests.harness import H, Qt
from sketch_machine import components, network
from sketch_machine.scene import PatchSurface

SCENE = "tests/scenes/topology.sketch.json"
MOVE_TOL = 1e-6  # how far points that are not moved may shift
ON_TOL = 0.01  # fraction of the surface size: how close the surface must lie to its lines


# --- Scene and queries ---

def _surface(h: H, name):
    return next(s for s in h.sc.surfaces if s.name == name)


def _guides(h: H, surface, red_only=True):
    """Red brush strokes (painting) above the surface, sorted by position."""
    lo, hi = _box(surface)
    out = []
    for c in h.curves():
        if not c.paint or len(c.points) < 10:
            continue
        if red_only and c.color is not None and c.color[1] > 0.4:  # yellow hatching
            continue
        m = c.points.mean(axis=0)
        if np.all(m[[0, 2]] >= lo[[0, 2]]) and np.all(m[[0, 2]] <= hi[[0, 2]]):
            out.append(c)
    return out


def _box(surface):
    v = surface.fill_mesh()[0]
    return v.min(axis=0), v.max(axis=0)


def _size(surface):
    lo, hi = _box(surface)
    return float(np.linalg.norm(hi - lo))


def _border(h: H, surface):
    """Boundary lines of a curve surface (split ones as their pieces)."""
    out = []
    for cid in surface.patch_lines:
        for sub in h.sc.split_pieces(cid) or []:
            c = h.sc.curve_by_id(sub)
            if c is not None:
                out.append(c)
    return out


def _border_pids(h: H, surface):
    return {pid for c in _border(h, surface) for pid in (c.pids or {}).values()}


def _positions(h: H):
    """{point ID: position} of all net points."""
    return {pid: refs[0][0].points[refs[0][1]].copy() for pid, refs in network.refs_by_pid(h.sc).items()}


def _top_view(h: H, surface):
    lo, hi = _box(surface)
    h.camera((lo + hi) / 2, float(np.linalg.norm(hi - lo)) * 1.6, [0.0, 1.0, 0.01])


def _draw_along(h: H, surface, guide, over=0.06):
    """Draw a real line along a template on the surface, extending a little past the template at both ends
    (as if dragged over the edge). Returns the new line."""
    h.draw_mode()
    h.v._activate(surface)
    _top_view(h, surface)
    pts = guide.points
    size = _size(surface)
    a = pts[0] + (pts[0] - pts[min(4, len(pts) - 1)]) / max(np.linalg.norm(pts[0] - pts[min(4, len(pts) - 1)]), 1e-9) * over * size
    b = pts[-1] + (pts[-1] - pts[-min(5, len(pts))]) / max(np.linalg.norm(pts[-1] - pts[-min(5, len(pts))]), 1e-9) * over * size
    idx = np.linspace(0, len(pts) - 1, 24).astype(int)
    before = {c.id for c in h.curves()}
    h.stroke_world(np.vstack([a[None], pts[idx], b[None]]))
    new = [c for c in h.curves() if c.id not in before and not c.paint]
    assert len(new) == 1, f"{len(new)} new lines instead of one"
    return new[0]


def _members(h: H, surface):
    """Real lines of the surface: boundary and those drawn on it."""
    return _border(h, surface) + [c for c in h.curves() if not c.paint and c.plane_id == surface.id]


def _dist_to_mesh(points, verts, tris):
    """Distance of each point to the triangle surface (rough enough for tests: nearest point on each triangle)."""
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    out = []
    for p in points:
        out.append(float(np.min(_point_tri(p, a, b, c))))
    return np.array(out)


def _point_tri(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum("ij,ij->i", ab, ap), np.einsum("ij,ij->i", ac, ap)
    n = np.cross(ab, ac)
    nn = np.einsum("ij,ij->i", n, n)
    ok = nn > 1e-18
    # projection onto the plane, then barycentric; outside: nearest point of the edges
    best = np.full(len(a), np.inf)
    w = np.where(ok[:, None], np.cross(ab, ap), 0)
    v = np.where(ok[:, None], np.cross(ap, ac), 0)
    s = np.einsum("ij,ij->i", v, n) / np.maximum(nn, 1e-18)
    t = np.einsum("ij,ij->i", w, n) / np.maximum(nn, 1e-18)
    inside = ok & (s >= 0) & (t >= 0) & (s + t <= 1)
    proj = a + s[:, None] * ab + t[:, None] * ac
    best[inside] = np.linalg.norm(proj[inside] - p, axis=1)
    for x, y in ((a, b), (b, c), (c, a)):
        xy = y - x
        u = np.clip(np.einsum("ij,ij->i", p - x, xy) / np.maximum(np.einsum("ij,ij->i", xy, xy), 1e-18), 0, 1)
        best = np.minimum(best, np.linalg.norm(x + u[:, None] * xy - p, axis=1))
    del d1, d2
    return best


def _off_surface(h: H, surface):
    """Largest distance of the surface to its real lines (fraction of the surface size)."""
    verts, tris = surface.fill_mesh()[:2]
    worst = 0.0
    for c in _members(h, surface):
        pts = c.points[np.linspace(0, len(c.points) - 1, min(len(c.points), 15)).astype(int)]
        worst = max(worst, float(_dist_to_mesh(pts, verts, tris).max()))
    return worst / _size(surface)


def _covers(surface, point, tol=0.05):
    """Does the surface lie over this point (seen from above)?"""
    verts, tris = surface.fill_mesh()[:2]
    p = np.asarray(point, float)
    flat = verts.copy()
    flat[:, 1] = 0
    q = p.copy()
    q[1] = 0
    return float(_dist_to_mesh([q], flat, tris)[0]) < tol


def _move_point(h: H, pid, delta):
    """Drag a net point as in point mode (with all lines that share it)."""
    refs = network.refs_by_pid(h.sc)[pid]
    h.sc.checkpoint()
    group = components.ComponentGroup(h.sc, [refs[0]], [])
    group.translate(np.asarray(delta, float))
    h.v._changed()


def _pid_at(h: H, point, tol=0.05):
    pos = _positions(h)
    pid = min(pos, key=lambda p: np.linalg.norm(pos[p] - point))
    assert np.linalg.norm(pos[pid] - point) < tol, f"no net point at {point}"
    return pid


def _check_connected(h: H, surface, lines):
    """Every end of every new line is a shared point (boundary or other line); crossings are points."""
    refs = network.refs_by_pid(h.sc)
    border = _border_pids(h, surface)
    for c in lines:
        for k in (0, len(c.points) - 1):
            pid = network.pid_of(c, k)
            assert pid in border or network.degree(refs[pid]) > 1, f"line end {c.points[k].round(2)} hangs loose"
    for i, a in enumerate(lines):
        for b in lines[:i]:
            near = network.crossings(a, b, 0.01 * _size(surface))
            if near:
                assert set(a.pids.values()) & set(b.pids.values()), "crossing is not a shared point"


def _lift_and_check(h: H, surface, pid, delta):
    before = _positions(h)
    count = len(before)
    _move_point(h, pid, delta)
    after = _positions(h)
    assert len(after) == count, f"{len(after) - count:+d} points"
    moved = [p for p in before if p in after and np.linalg.norm(after[p] - before[p]) > MOVE_TOL]
    assert moved == [pid], f"{len(moved)} points moved, expected only the dragged one"
    off = _off_surface(h, surface)
    assert off < ON_TOL, f"surface lies {off * 100:.1f} % beside its lines"


# --- Task 1: raise a corner ---

def test_task1_raise_a_corner(h: H):
    h.load(SCENE)
    surf = _surface(h, "Curve surface 1")
    lines = [_draw_along(h, surf, g) for g in _guides(h, surf)]
    assert len(lines) == 2
    _check_connected(h, surf, lines)
    corner = _pid_at(h, [22.76, 0.0, 3.83])
    _lift_and_check(h, surf, corner, [0, 2.0, 0])
    _check_connected(h, surf, lines)
    h.save_image("konzept_task1.png")


# --- Task 2: raise a crossing ---

def test_task2_raise_a_crossing(h: H):
    h.load(SCENE)
    surf = _surface(h, "Curve surface 3")
    lines = [_draw_along(h, surf, g) for g in _guides(h, surf)[:2]]
    _check_connected(h, surf, lines)
    shared = set(lines[0].pids.values()) & set(lines[1].pids.values())
    assert len(shared) == 1, "one crossing"
    _lift_and_check(h, surf, shared.pop(), [0, 1.5, 0])
    _check_connected(h, surf, lines)
    h.save_image("konzept_task2.png")


# --- Tasks 3 and 4: delete a corner ---

def _delete_point(h: H, point):
    h.edit_mode("point")
    h.click(h.scr(point))
    assert h.v.point_selection, "point selected"
    h.key(Qt.Key_Delete, Qt.ControlModifier)  # Ctrl+Del: delete (Del dissolves)


def test_task3_delete_corner_bridge(h: H):
    h.load(SCENE)
    surf = _surface(h, "Curve surface 5")
    _top_view(h, surf)
    a, b, gone = np.array([-18.88, 0, 3.83]), np.array([-13.0, 0, -6.7]), np.array([-21.18, 0, -1.74])
    _delete_point(h, gone)
    near = [c for c in h.curves() if not c.paint and np.linalg.norm(c.points - gone, axis=1).min() < 0.3]
    assert not near, "adjoining pieces are gone"
    bridge = [c for c in h.curves() if not c.paint and {tuple(np.round(c.points[0], 1)), tuple(np.round(c.points[-1], 1))}
              == {tuple(np.round(a, 1)), tuple(np.round(b, 1))}]
    assert len(bridge) == 1, "neighbouring points connected by a new line"
    assert not _covers(surf, [-17.7, 0, -1.5]), "surface without the deleted corner"
    assert _covers(surf, [-12.0, 0, 0.0]), "rest of the surface stays"


def test_task4_delete_corner_outline_via_grid_lines(h: H):
    h.load(SCENE)
    surf = _surface(h, "Curve surface 6")
    guides = [g for g in _guides(h, surf) if np.linalg.norm(np.ptp(g.points, axis=0)) > 0.3 * _size(surf)]  # without the circle
    lines = [_draw_along(h, surf, g) for g in guides]
    assert len(lines) == 2, f"{len(lines)} grid lines"
    _check_connected(h, surf, lines)
    _delete_point(h, np.array([-47.38, 0, -1.74]))
    assert not _covers(surf, [-43.4, 0, -0.8]), "yellow area is gone"
    assert _covers(surf, [-38.0, 0, -2.0]), "rest of the surface stays"
    a, b = np.array([-45.1, 0, 3.8]), np.array([-39.2, 0, -6.7])
    bridge = [c for c in h.curves() if not c.paint and np.linalg.norm(c.points[0] - a) < 0.2 and np.linalg.norm(c.points[-1] - b) < 0.2]
    assert not bridge, "no bridge: the outline over the grid lines applies"
    _check_connected(h, surf, [c for c in lines if h.sc.curve_by_id(c.id) is not None])


# --- Task 5: several connected surfaces ---

def test_task5_move_every_point(h: H):
    h.load(SCENE)
    names = ("Curve surface 7", "Curve surface 8", "Curve surface 9")
    surfs = [_surface(h, n) for n in names]
    lo = np.min([_box(s)[0] for s in surfs], axis=0)
    hi = np.max([_box(s)[1] for s in surfs], axis=0)
    guides = [c for c in h.curves() if c.paint and len(c.points) > 60
              and np.all(c.points.mean(axis=0)[[0, 2]] >= lo[[0, 2]]) and np.all(c.points.mean(axis=0)[[0, 2]] <= hi[[0, 2]])]
    assert len(guides) == 7, f"{len(guides)} templates"
    lines = []
    for g in guides:  # every template lies on a surface
        mid = g.points[len(g.points) // 2]
        surf = min(surfs, key=lambda s: float(_dist_to_mesh([mid], *s.fill_mesh()[:2])[0]))
        lines.append((surf, _draw_along(h, surf, g, over=0.01)))
    for surf in surfs:
        _check_connected(h, surf, [c for s, c in lines if s is surf])
    pids = set()
    for surf in surfs:
        for c in _members(h, surf):
            pids |= set((c.pids or {}).values())
    for k, pid in enumerate(sorted(pids)):
        before = _positions(h)
        _move_point(h, pid, [0.0, 0.4 if k % 2 else -0.3, 0.0])
        after = _positions(h)
        assert len(after) == len(before), "no new points"
        moved = [p for p in before if np.linalg.norm(after[p] - before[p]) > MOVE_TOL]
        assert moved == [pid], f"point {k}: {len(moved)} points moved"
        for surf in surfs:
            off = _off_surface(h, surf)
            assert off < ON_TOL, f"{surf.name} lies {off * 100:.1f} % beside its lines"
    for surf in surfs:
        _check_connected(h, surf, [c for s, c in lines if s is surf])
    h.save_image("konzept_task5.png")


def test_scene_loads_all_surfaces(h: H):
    h.load(SCENE)
    assert sum(isinstance(s, PatchSurface) for s in h.sc.surfaces) == 9


def test_edge_connects_even_with_smoothing_and_a_fast_stroke(h: H):
    """debugging_topo: with smoothing the stroke ends at the trailing tip, fast strokes hit the surface
    last a bit before the edge - still the start (placed outside) and the end (dragged over the edge) must land on
    the boundary line."""
    h.load(SCENE)
    surf = _surface(h, "Curve surface 1")
    h.draw_mode()
    h.v._activate(surf)
    _top_view(h, surf)
    h.v.set_smooth(0.7)
    try:
        guide = _guides(h, surf)[0]
        pts = guide.points
        size = _size(surf)
        a = pts[0] + (pts[0] - pts[6]) / np.linalg.norm(pts[0] - pts[6]) * 0.08 * size
        b = pts[-1] + (pts[-1] - pts[-7]) / np.linalg.norm(pts[-1] - pts[-7]) * 0.05 * size
        before = {c.id for c in h.curves()}
        h.stroke([h.scrf(p) for p in np.vstack([a[None], pts[::6], b[None]])], step_px=25)
    finally:
        h.v.set_smooth(0.0)
    new = [c for c in h.curves() if c.id not in before and not c.paint]
    assert len(new) == 1
    _check_connected(h, surf, new)
    border = _border_pids(h, surf)
    c = new[0]
    assert network.pid_of(c, 0) in border and network.pid_of(c, len(c.points) - 1) in border, "both ends on the edge"


def test_close_lines_without_strong_waves(h: H):
    """debugging_topo (artist): four short lines close together, one raised by 0.184. On the coarse grid
    (up to v0.84.1) the surface swung from -2.6 to +1.1; on the full grid only the small overshoot of the
    smooth surface remains."""
    from sketch_machine import netsurf
    assert netsurf.sparse is not None, "scipy missing (requirements.txt)"
    h.load("tests/scenes/net_close_lines.sketch.json")
    surf = h.sc.surfaces[0]
    y = surf.grid()[..., 1]
    assert -0.35 < y.min() and y.max() < 0.25, f"surface swings out: {y.min():.2f} .. {y.max():.2f}"
    assert _off_surface(h, surf) < 0.002


def test_edge_points_sideways_and_corners_without_outliers(h: H):
    """debugging_topo (2nd state, artist): drag edge points sideways, raise corners, push a raised line over
    another (crossing at different heights). Formerly: the surface swung out by up to 18 units (parameter slid
    away under the lines, Coons dent, contradiction). Now: rest shape + smooth deformation, contradictions averaged."""
    moves = [([4.29, 0, -6.5], [1.0, 0, 0]), ([4.29, 0, -6.5], [0, 0, 1.0]), ([8.38, 0, -10.98], [1.0, 0, 0]),
             ([8.38, 0, -10.98], [0, 1.0, 0]), ([-3.99, 0, -0.55], [0, 0, 1.0]), ([1.08, 0.57, 1.65], [0, 0, 1.0])]
    for point, delta in moves:
        h.load("tests/scenes/net_border_points.sketch.json")
        surf = h.sc.surfaces[0]
        _move_point(h, _pid_at(h, point, tol=0.02), delta)
        ys = np.concatenate([c.points[:, 1] for c in _members(h, surf)])
        y = surf.grid()[..., 1]
        over = max(y.max() - ys.max(), ys.min() - y.min(), 0.0)
        assert over < 0.5, f"{point} by {delta}: surface swings {over:.2f} beyond the lines"
        assert _off_surface(h, surf) < 0.01, f"{point} by {delta}: surface lies beside its lines"


def test_dissolving_at_the_surface_edge_keeps_shape(h: H):
    """Del on the joint of two boundary lines of a curve surface: the two lines become one, the shape of line and
    surface stays, the surface knows the new boundary line."""
    h.load(SCENE)
    surf = _surface(h, "Curve surface 5")
    _top_view(h, surf)
    lines_before = {c.id for c in h.curves() if not c.paint}
    grid_before = surf.grid().copy()
    pos = _positions(h)
    joint = pos[_pid_at(h, [-18.88, 0, 3.83])]  # joint of two boundary lines, not a corner of the curve surface
    pts_before = np.vstack([c.points for c in _border(h, surf)])
    h.edit_mode("point")
    h.click(h.scr(joint))
    assert h.v.point_selection, "point selected"
    h.key(Qt.Key_Delete)
    border = _border(h, surf)
    assert len(border) == len(surf.patch_lines) == 4, f"{len(border)} boundary lines (5 before)"
    after = np.vstack([c.points for c in border])
    gap = max(np.linalg.norm(after - p, axis=1).min() for p in pts_before)
    assert gap < 1e-9, f"line points stay ({gap:.2g})"
    assert np.abs(surf.grid() - grid_before).max() < 1e-3 * _size(surf), "surface stays as it was"
    merged = [c for c in h.curves() if not c.paint and c.id not in lines_before]
    assert len(merged) == 1 and np.linalg.norm(merged[0].points - joint, axis=1).min() < 1e-9
    assert all(np.linalg.norm(merged[0].points[k] - joint) > 1e-6 for k in components.knots(merged[0])), \
        "no point left at the joint"


def test_crossing_is_not_dissolved(h: H):
    h.load(SCENE)
    surf = _surface(h, "Curve surface 3")
    lines = [_draw_along(h, surf, g) for g in _guides(h, surf)[:2]]
    x = (set(lines[0].pids.values()) & set(lines[1].pids.values())).pop()
    pos = _positions(h)[x]
    n_before = len(h.curves())
    h.edit_mode("point")
    h.click(h.scr(pos))
    h.key(Qt.Key_Delete)
    assert x in _positions(h), "crossing stays a point"
    assert len(h.curves()) == n_before


def _folded(g):
    du, dv = g[1:, :-1] - g[:-1, :-1], g[:-1, 1:] - g[:-1, :-1]
    n = np.cross(du, dv)[..., 1]
    return int(((np.sign(n) != np.sign(np.median(n))) & (np.abs(n) > 1e-9)).sum())


def test_corner_far_sideways_without_folds(h: H):
    """Edge nodes hang at fixed places of the boundary lines: if a corner is dragged far sideways, the surface stays
    unfolded (even redistribution at the edge formerly shifted all edge nodes of the side: 133 folded cells)."""
    h.load("tests/scenes/net_border_points.sketch.json")
    surf = h.sc.surfaces[0]
    rest = _folded(surf.grid())
    _move_point(h, _pid_at(h, [8.38, 0, -10.98], tol=0.02), [8.0, 0, -6.0])
    assert _folded(surf.grid()) <= rest + 2, f"{_folded(surf.grid())} folded cells"
    assert _off_surface(h, surf) < 0.002


def test_redistributing_keeps_shape(h: H):
    """Curve net: the grid points hang at fixed places of the lines - after a long drag nothing is
    reclassified (the surface depends on the classification and would jump after release)."""
    from sketch_machine import netsurf
    h.load("tests/scenes/net_border_points.sketch.json")
    surf = h.sc.surfaces[0]
    _move_point(h, _pid_at(h, [8.38, 0, -10.98], tol=0.02), [8.0, 0, -6.0])
    g_old = surf.grid().copy()
    old, netsurf.REREST = netsurf.REREST, 1.0
    try:
        assert h.sc.rerest_patches() == 0
    finally:
        netsurf.REREST = old
    h.v._changed()
    verts, tris = surf.fill_mesh()[:2]
    assert _dist_to_mesh(g_old.reshape(-1, 3)[::23], verts, tris).max() < 0.005 * _size(surf), "shape stays"
    assert _off_surface(h, surf) < 0.002


def _multi(h: H, names):
    surfs = [_surface(h, n) for n in names]
    lo = np.min([_box(s)[0] for s in surfs], axis=0)
    hi = np.max([_box(s)[1] for s in surfs], axis=0)
    h.draw_mode()
    h.camera((lo + hi) / 2, float(np.linalg.norm(hi - lo)) * 1.4, [0.0, 1.0, 0.01])
    h.v.set_multi_mode(True)
    h.sc.multi_ids = [s.id for s in surfs]
    h.v._changed()
    return surfs


def _stroke_new(h: H, a, b, n=40):
    a, b = np.asarray(a, float), np.asarray(b, float)
    before = {c.id for c in h.curves()}
    h.stroke([h.scrf(a + t * (b - a)) for t in np.linspace(0, 1, n)])
    return [c for c in h.curves() if c.id not in before and not c.paint]


def test_multi_across_surface_boundaries_connected(h: H):
    """Multi mode (artist): a stroke across three surfaces with shared boundary lines and past the outer edge.
    Each piece ends on a boundary line of its surface; transitions are shared points; crossings too."""
    h.load(SCENE)
    surfs = _multi(h, ("Curve surface 7", "Curve surface 8", "Curve surface 9"))
    try:
        first = _stroke_new(h, [-71.0, 0, 1.0], [-49.0, 0, -7.5])
        second = _stroke_new(h, [-64.0, 0, -9.0], [-57.0, 0, 6.5])
    finally:
        h.v.set_multi_mode(False)
    assert len(first) >= 2 and len(second) >= 2, (len(first), len(second))
    for c in first + second:
        surf = h.sc.object_by_id(c.plane_id)
        border = _border_pids(h, surf)
        for k in (0, len(c.points) - 1):
            assert network.pid_of(c, k) in border, f"end {c.points[k].round(2)} does not lie on the edge of {surf.name}"
    for i, a in enumerate(first):  # transitions: consecutive pieces share a point
        for b in first[i + 1:i + 2]:
            assert set(a.pids.values()) & set(b.pids.values()), "transition is not a shared point"
    for a in second:
        for b in first:
            if a.plane_id == b.plane_id and network.crossings(a, b, 0.01 * _size(h.sc.object_by_id(a.plane_id))):
                assert set(a.pids.values()) & set(b.pids.values()), "crossing is not a shared point"
    _check_connected(h, surfs[0], [c for c in first + second if c.plane_id == surfs[0].id])


def test_multi_artist_scene(h: H):
    """The artist's multi scene (net_multi): two curve surfaces with a shared boundary line; stroke across both and past both
    outer edges, a second one crosses an existing line. Everything connected."""
    h.load("tests/scenes/net_multi.sketch.json")
    surfs = list(h.sc.surfaces)
    lo = np.min([_box(s)[0] for s in surfs], axis=0)
    hi = np.max([_box(s)[1] for s in surfs], axis=0)
    h.draw_mode()
    h.camera((lo + hi) / 2, float(np.linalg.norm(hi - lo)) * 1.4, [0.0, 1.0, 0.01])
    h.v.set_multi_mode(True)
    h.sc.multi_ids = [s.id for s in surfs]
    h.v._changed()
    try:
        a = _stroke_new(h, [-4.5, 0, -7.5], [11.5, 0, 12.5])  # next to the artist's old line
        b = _stroke_new(h, [2.0, 0, -6.0], [1.5, 0, 0.5])  # crosses the existing line at z ~ -3
    finally:
        h.v.set_multi_mode(False)
    assert len(a) == 2, f"{len(a)} pieces"
    assert set(a[0].pids.values()) & set(a[1].pids.values()), "transition is a shared point"
    shared = (set(a[0].pids.values()) & set(a[1].pids.values())).pop()
    assert all(shared in _border_pids(h, s) for s in surfs), "transition lies on the shared boundary line"
    for c in a:
        surf = h.sc.object_by_id(c.plane_id)
        for k in (0, len(c.points) - 1):
            assert network.pid_of(c, k) in _border_pids(h, surf), f"end {c.points[k].round(2)} not on the edge"
    old = [c for c in h.curves() if not c.paint and c not in a + b and c.plane_id == b[0].plane_id]
    hits = [o for o in old if network.crossings(b[0], o, 0.05)]
    assert hits, "test setup: second stroke crosses an existing line"
    assert all(set(b[0].pids.values()) & set(o.pids.values()) for o in hits), "crossing is a shared point"


def test_recognised_shape_across_surface_boundaries_connected(h: H):
    """netz_multi_v002 (artist): a stroke held still at the end is recognised as an arc and laid anew onto the surfaces
    - transitions and edges were lost in the process (all ends loose). Multi and single mode."""
    scene = "tests/scenes/net_multi_v002.sketch.json"
    h.load(scene)
    path = np.vstack([h.curves()[-2].points, h.curves()[-1].points])  # the artist's arc across both surfaces
    d0 = (path[0] - path[5]) / np.linalg.norm(path[0] - path[5])
    d1 = (path[-1] - path[-6]) / np.linalg.norm(path[-1] - path[-6])
    pts = np.vstack([path[0] + d0 * 0.6, path[::8], path[-1] + d1 * 0.6])
    for multi in (True, False):
        h.load(scene)
        sc = h.sc
        sc.strokes[-1].curves = sc.strokes[-1].curves[:-2]  # remove the artist's loose lines
        h.v._changed()
        surfs = list(sc.surfaces)
        lo = np.min([_box(s)[0] for s in surfs], axis=0)
        hi = np.max([_box(s)[1] for s in surfs], axis=0)
        h.draw_mode()
        h.camera((lo + hi) / 2, float(np.linalg.norm(hi - lo)) * 1.4, [0.0, 1.0, 0.01])
        if multi:
            h.v.set_multi_mode(True)
            sc.multi_ids = [s.id for s in surfs]
        else:
            h.v._activate(surfs[0])
        h.v._changed()
        before = {c.id for c in h.curves()}
        try:
            h.stroke([h.scrf(p) for p in pts], hold=True)
        finally:
            h.v.set_multi_mode(False)
        new = [c for c in h.curves() if c.id not in before and not c.paint]
        assert new, "arc drawn"
        assert len({tuple(np.round(np.diff(c.points, axis=0).std(axis=0), 3)) for c in new}) >= 1
        for c in new:
            surf = sc.object_by_id(c.plane_id)
            border = _border_pids(h, surf)
            for k in (0, len(c.points) - 1):
                pid = network.pid_of(c, k)
                refs = network.refs_by_pid(sc)[pid]
                assert pid in border or network.degree(refs) > 1, \
                    f"{'Multi' if multi else 'Single'}: end {c.points[k].round(2)} hangs loose"
        if multi:
            assert len(new) == 2 and set(new[0].pids.values()) & set(new[1].pids.values()), "transition connected"


def test_cleanup_connects_loose_ends_of_old_scenes(h: H):
    """Cleanup "connect ends and crossings" applies the connection rule to existing lines: the artist's scenes
    netz_multi_v002/_v003 (drawn before the rule) afterwards have no loose ends close to lines any more."""
    for scene in ("tests/scenes/net_multi_v002.sketch.json", "tests/scenes/net_multi_v003.sketch.json"):
        h.load(scene)
        _top_view(h, h.sc.surfaces[0])
        assert network.sync(h.sc)["free_ends"] > 0, "test setup: loose ends present"
        h.v.set_cleanup_global(True)
        try:
            h.v.run_cleanup("crossings")
        finally:
            h.v.set_cleanup_global(False)
        assert network.sync(h.sc)["free_ends"] == 0, f"{scene}: {network.sync(h.sc)['free_ends']} loose ends"


# --- Fill surfaces as a net surface (netfill.py) ---

FILL = "tests/scenes/net_fill_cross.sketch.json"


def _fill_members(h: H, surf):
    return h.v._edge_lines(surf) + [c for c in h.curves() if not c.paint and c.plane_id == surf.id]


def _fill_off(h: H, surf):
    verts, tris = surf.fill_mesh()[:2]
    worst = 0.0
    for c in [c for c in h.curves() if not c.paint and c.plane_id == surf.id]:
        pts = c.points[np.linspace(0, len(c.points) - 1, min(len(c.points), 15)).astype(int)]
        worst = max(worst, float(_dist_to_mesh(pts, verts, tris).max()))
    return worst / _size(surf)


def _fill_draw(h: H, surf, a, b):
    h.draw_mode()
    h.v._activate(surf)
    _top_view(h, surf)
    return _stroke_new(h, a, b)


def test_fill_surface_raise_a_crossing(h: H):
    """Like task 2, on a fill surface: two lines criss-cross (past the edge), raise the crossing - the
    surface bulges along, only the dragged point moves, everything stays connected."""
    h.load(FILL)
    surf = h.sc.surfaces[0]
    a = _fill_draw(h, surf, [-8.5, 0, -3.0], [7.0, 0, -0.5])
    b = _fill_draw(h, surf, [-3.0, 0, -9.5], [-1.0, 0, 5.0])
    lines = a + b
    assert len(a) == 1 and len(b) == 1, (len(a), len(b))
    border = {pid for c in h.v._edge_lines(surf) for pid in (c.pids or {}).values()}
    for c in lines:
        for k in (0, len(c.points) - 1):
            assert network.pid_of(c, k) in border, f"end {c.points[k].round(2)} not on the outline"
    shared = set(a[0].pids.values()) & set(b[0].pids.values())
    assert shared, "crossing connected"
    pid = shared.pop()
    before = _positions(h)
    _move_point(h, pid, [0, 1.5, 0])
    after = _positions(h)
    moved = [p for p in before if p in after and np.linalg.norm(after[p] - before[p]) > MOVE_TOL]
    assert moved == [pid], f"{len(moved)} points moved"
    assert surf.netmesh is not None, "fill surface is a net surface"
    assert float(surf.fill_mesh()[0][:, 1].max()) > 1.2, "surface bulges along"
    off = _fill_off(h, surf)
    assert off < 0.01, f"surface lies {off * 100:.1f} % beside its lines"
    h.save_image("fuellflaeche_kreuzung.png")


def test_fill_surface_raise_corner_lines_stay(h: H):
    """Raise a corner of the outline: the surface deforms, the lines on it stay where they are, and the surface keeps
    passing through them."""
    h.load(FILL)
    surf = h.sc.surfaces[0]
    inner = _fill_draw(h, surf, [-3.0, 0, -9.5], [-1.0, 0, 5.0])
    keep = {c.id: c.points.copy() for c in h.curves() if c.plane_id == surf.id and not c.paint}
    corner = _pid_at(h, [-6.53, 0, -8.44], tol=0.05)
    _move_point(h, corner, [0, 2.0, 0])
    for cid, pts in keep.items():
        c = h.sc.curve_by_id(cid)
        assert c is not None and np.array_equal(c.points, pts), "line on the surface stayed put"
    assert float(surf.fill_mesh()[0][:, 1].max()) > 1.5, "surface follows the corner"
    assert _fill_off(h, surf) < 0.01


def test_loft_raise_a_crossing(h: H):
    """Like task 2, on a loft: two lines on it cross, raise the crossing - the loft goes along, only the
    dragged point moves, the lines stay connected and on the surface; rails stay put."""
    from tests.test_topology import _loft3
    lo, secs = _loft3(h)
    rails = {cid: h.sc.curve_by_id(cid).points.copy() for cid, _ in lo.rails}
    h.draw_mode()
    h.v._activate(lo)
    h.camera([1, 1.2, 1], 5, [0.05, 1, 0.05])
    a = _stroke_new(h, [-0.3, 1.0, 0.5], [2.3, 1.0, 0.5])
    b = _stroke_new(h, [1.0, 1.0, -0.3], [1.0, 1.0, 2.3])
    assert len(a) == 1 and len(b) == 1, (len(a), len(b))
    shared = set(a[0].pids.values()) & set(b[0].pids.values())
    assert shared, "crossing connected"
    pid = shared.pop()
    before = _positions(h)
    _move_point(h, pid, [0, 0.6, 0])
    after = _positions(h)
    moved = [p for p in before if p in after and np.linalg.norm(after[p] - before[p]) > MOVE_TOL]
    assert moved == [pid], f"{len(moved)} points moved"
    verts, tris = lo.fill_mesh()[:2]
    for c in (h.sc.curve_by_id(a[0].id), h.sc.curve_by_id(b[0].id)):
        pts = c.points[np.linspace(0, len(c.points) - 1, 15).astype(int)]
        assert _dist_to_mesh(pts, verts, tris).max() < 0.01 * _size(lo), "loft passes through the lines"
    for cid, pts in rails.items():  # shape stays (snapped strokes only insert points)
        rail = h.sc.curve_by_id(cid)
        gap = max(h.sc._closest_on(type(rail)(pts, None), p)[0] for p in rail.points)
        assert gap < 1e-9, f"rail has moved ({gap:.2g})"


def test_fill_surface_delete_corner_outline_over_line(h: H):
    """Like task 4, on a fill surface: a line across a corner, then delete the corner (Ctrl+Del) - the
    outline runs over the line, the surface loses the corner, no bridge; without a line: straight bridge (like task 3)."""
    h.load(FILL)
    surf = h.sc.surfaces[0]
    corner = np.array([-6.53, 0, -8.44])
    _fill_draw(h, surf, [-7.5, 0, -5.0], [-1.5, 0, -9.8])  # cuts off the corner
    assert _covers(surf, [-5.6, 0, -7.6]), "test setup: corner belongs to the surface"
    n_lines = len([c for c in h.curves() if not c.paint])
    _top_view(h, surf)
    _delete_point(h, corner)
    assert not _covers(surf, [-5.6, 0, -7.6]), "corner is gone"
    assert _covers(surf, [0.0, 0, -2.0]), "rest stays"
    assert len([c for c in h.curves() if not c.paint]) >= n_lines - 2, "no bridge instead of the outline"
    assert not any(np.linalg.norm(c.points - corner, axis=1).min() < 0.05 for c in h.curves() if not c.paint)


def test_open_loft_becomes_a_curve_surface_with_whole_lines(h: H):
    """loft_test (artist): a cross-section has points (pieces) and is selected as a whole line - the whole
    line belongs to the loft. An open loft then becomes a curve surface like any other (edge: first cross-section,
    rail, last cross-section, rail), the middle cross-sections shape it as lines on it. (Until 2026-11-24 a single
    selected piece stood for its whole line; now a piece is a piece: tests/test_loft.py.)"""
    from sketch_machine import components as comp
    from sketch_machine.scene import PatchSurface
    sc = h.sc
    sc.new_stroke()
    t = np.linspace(0, 1, 40)
    rows = []
    for z, lift in ((0.0, 0.0), (3.0, 1.2), (6.0, 0.3)):
        sc.add_curve(np.stack([t * 6 - 3, lift + 0.8 * np.sin(t * np.pi), np.full_like(t, z)], axis=1), None)
        rows.append(sc.curves[-1])
    mid = rows[1]
    for k in (13, 26):  # points on the middle cross-section -> three pieces
        comp.insert_point(mid, k, 0.0)
    h.v._changed()
    network.sync(sc)
    st = sc.owner_of(mid).id
    segs = comp.segments(mid)
    h.v.set_select_mode(True)
    h.v.ws.component = "edge"
    assert len(segs) == 3
    h.v.edge_selection = [("line", st, rows[0].id), ("line", st, mid.id), ("line", st, rows[2].id)]
    h.v.do_loft()
    surf = sc.surfaces[-1]
    assert isinstance(surf, PatchSurface), f"open loft is {type(surf).__name__}, not a curve surface"
    assert sc.curve_by_id(mid.id) is not None and len(sc.curve_by_id(mid.id).points) == len(mid.points), \
        "middle line stays whole (not split)"
    assert sc.curve_by_id(mid.id).plane_id == surf.id, "middle cross-section belongs to the surface as a line"
    h.v._changed()
    verts, tris = surf.fill_mesh()[:2]
    size = float(np.linalg.norm(np.ptp(verts, axis=0)))
    for c in rows:
        pts = sc.curve_by_id(c.id).points[::4]
        assert _dist_to_mesh(pts, verts, tris).max() < 0.01 * size, "surface passes through all cross-sections (whole)"
    border = {c.id for c in sc.patch_border(surf)}
    assert rows[0].id in border and rows[2].id in border and mid.id not in border


def test_loft_from_profiles_in_pieces_stays_a_ruled_surface(h: H):
    """Skyscraper: cross-sections made of several lines (U profile with kinks). The curve surfaces must lie exactly like
    the ruled surface between them - rest shape and edge take the same corners in the same order."""
    from sketch_machine import loft
    sc = h.sc
    sc.new_stroke()

    def u_profile(z, w, d, shift):
        a = np.linspace([-w, 0, z], [-w, d, z], 15)
        b = np.linspace([-w, d, z], [w, d, z], 30)[1:]
        c = np.linspace([w, d, z], [w, 0, z], 15)[1:]
        return np.vstack([a, b, c]) + [shift, 0, 0]

    secs = [u_profile(0, 2, 1.5, 0), u_profile(5, 1.2, 3, 0.8)]
    lines = []
    for s in secs:
        for part in (s[:15], s[14:44], s[43:]):
            sc.add_curve(part, None)
            lines.append(sc.curves[-1])
    h.v._changed()
    network.sync(sc)
    h.v.set_select_mode(True)
    h.v.ws.component = "edge"
    h.v.edge_selection = [("line", sc.owner_of(c).id, c.id) for c in lines]
    h.v.do_loft()
    h.v._changed()
    # the kinks of the U profile (90°) are hard points: three curve surfaces (artist 2026-11-09), together the ruled surface
    assert len(sc.surfaces) == 3 and all(isinstance(f, PatchSurface) for f in sc.surfaces), \
        [type(f).__name__ for f in sc.surfaces]
    ruled = loft.grid(np.array(loft.prepare(secs)[0]), 12).reshape(-1, 3)
    size = float(np.linalg.norm(np.ptp(ruled, axis=0)))
    d = np.min([_dist_to_mesh(ruled[::3], *f.fill_mesh()[:2]) for f in sc.surfaces], axis=0).max()
    assert d < 0.01 * size, f"surfaces deviate {d / size:.1%} from the ruled surface"


def test_extruding_open_becomes_a_curve_surface(h: H):
    """Extruding with drawing surface: open profile -> curve surface (edge: original, connecting line, copy,
    connecting line), closed -> tube (loft)."""
    from sketch_machine.scene import LoftSurface
    sc = h.sc
    for closed in (False, True):
        sc.new_stroke()
        t = np.linspace(0, 1, 40)
        if closed:
            a = t * 2 * np.pi
            pts = np.stack([2 * np.cos(a), np.zeros_like(a), 2 * np.sin(a) + 10], axis=1)
        else:
            pts = np.stack([t * 6 - 3, np.zeros_like(t), np.sin(t * np.pi * 1.5)], axis=1)
        sc.add_curve(pts, None)
        c = sc.curves[-1]
        h.v._changed()
        h.v.ws.extrude_surface = True
        h.v.set_select_mode(True)
        h.v.ws.component = "edge"
        h.v.edge_selection = [("line", sc.owner_of(c).id, c.id)]
        h.v.start_extrude()
        h.v._extrude.update(axis=np.array([0, 1.0, 0]), axis_name="y", length=3.0)
        h.v._finish_extrude()
        h.v._mode = None
        h.v._changed()
        surf = sc.surfaces[-1]
        if closed:
            assert type(surf) is LoftSurface and surf.closed, f"closed: {type(surf).__name__}"
        else:
            assert isinstance(surf, PatchSurface), f"open: {type(surf).__name__}"
            assert c.id in {x.id for x in sc.patch_border(surf)}, "original is the edge of the curve surface"


def test_old_open_lofts_become_curve_surfaces_on_loading(h: H):
    """Older scene with an open loft: after loading a curve surface, the cross-sections keep shaping it."""
    import json, os, tempfile
    from sketch_machine.scene import Scene, LoftSurface
    from sketch_machine import loft
    sc = h.sc
    sc.new_stroke()
    t = np.linspace(0, 1, 40)
    rows = []
    for z, lift in ((0.0, 0.0), (3.0, 1.2), (6.0, 0.3)):
        sc.add_curve(np.stack([t * 6 - 3, lift + 0.8 * np.sin(t * np.pi), np.full_like(t, z)], axis=1), None)
        rows.append(sc.curves[-1])
    h.v._changed()
    network.sync(sc)
    old = sc.loft_to_patch
    sc.loft_to_patch = lambda s: None  # this is how it used to arise: as a loft
    try:
        h.v.set_select_mode(True)
        h.v.ws.component = "edge"
        h.v.edge_selection = [("line", sc.owner_of(c).id, c.id) for c in rows]
        h.v.do_loft()
    finally:
        del sc.loft_to_patch
    surf = sc.surfaces[-1]
    assert type(surf) is LoftSurface
    h.v._changed()
    g = surf.grid()
    path = os.path.join(tempfile.mkdtemp(), "alt.sketch.json")
    sc.save(path)
    new = Scene.load(path)[0]
    patch = next(s for s in new.surfaces if s.name.startswith(tr("Kurvenfläche")))
    assert isinstance(patch, PatchSurface) and not any(type(s) is LoftSurface for s in new.surfaces)
    new.sync_lofts()
    verts, tris = patch.fill_mesh()[:2]
    size = float(np.linalg.norm(np.ptp(g.reshape(-1, 3), axis=0)))
    assert _dist_to_mesh(g.reshape(-1, 3)[::7], verts, tris).max() < 0.01 * size, "shape stays"


def test_free_stroke_connects_explicitly(h: H):
    """Free mode (Q): stroke from the end of a line to a line at another depth - both ends are afterwards connected
    with shared points (via the single connection rule, not via the position)."""
    sc = h.sc
    sc.new_stroke()
    sc.add_curve(np.linspace([-3, 0, 0], [0, 0, 0], 20), None)
    a = sc.curves[-1]
    sc.add_curve(np.linspace([2, -2, 4], [2, 3, 4], 20), None)
    b = sc.curves[-1]
    h.v._changed()
    h.frame(np.vstack([a.points, b.points]), back=(0.2, 0.3, 1.0))
    h.draw_mode()
    h.v.ws.magnet = True
    h.v.set_free_draw(True)
    try:
        start, end = h.scrf(a.points[-1]), h.scrf(b.points[10])
        mid = (start + end) / 2 + QPointF(0, -40)
        h.stroke([start, mid, end])
    finally:
        h.v.set_free_draw(False)  # the window is reused by the next tests
    network.sync(sc)
    new = [c for c in h.curves() if c is not a and c is not b and not c.paint]
    assert len(new) == 1, f"{len(new)} new lines"
    c = new[0]
    refs = network.refs_by_pid(sc)
    p0, p1 = network.pid_of(c, 0), network.pid_of(c, len(c.points) - 1)
    assert p0 == network.pid_of(a, len(a.points) - 1), "start shares the point with the line end"
    assert network.degree(refs[p1]) >= 2 and p1 in (b.pids or {}).values(), "end is a point on the other line"
    assert np.linalg.norm(c.points[-1] - sc.curve_by_id(b.id).points[[k for k, p in b.pids.items() if p == p1][0]]) < 1e-9


def _square_with_gap(h: H, gap):
    """Square made of four lines (edge 4), open by gap at one corner; the other corners are shared points."""
    sc = h.sc
    sc.new_stroke()
    c = [[0, 0, 0], [4, 0, 0], [4, 0, 4], [0, 0, 4]]
    lines = []
    for i in range(4):
        a, b = np.array(c[i], float), np.array(c[(i + 1) % 4], float)
        if i == 3:
            b = b + [gap, 0, 0]  # last line ends just before the corner
        sc.add_curve(np.linspace(a, b, 30), None)
        lines.append(sc.curves[-1])
    network.sync(sc)
    for i in range(3):
        network.join(sc, lines[i], len(lines[i].points) - 1, lines[i + 1], 0)
    h.v._changed()
    h.v.set_select_mode(True)
    h.v.ws.component = "edge"
    h.v.edge_selection = [("line", sc.owner_of(x).id, x.id) for x in lines]
    return lines


def test_surface_small_gap_becomes_a_shared_point(h: H):
    """Gap rule: gap under 2 % of the shorter line -> the ends become a shared point (in the middle),
    no new line; the surface is created."""
    lines = _square_with_gap(h, 0.04)  # 1 % of 4
    before = len(h.curves())
    h.v.do_fill()
    assert len(h.curves()) == before, "no bridge"
    a, b = lines[3], lines[0]
    assert network.pid_of(a, len(a.points) - 1) == network.pid_of(b, 0), "corner is a shared point"
    assert np.allclose(a.points[-1], [0.02, 0, 0], atol=1e-6), "meeting in the middle"
    assert any(type(s).__name__ == "SurfaceObject" for s in h.sc.surfaces)


def test_surface_large_gap_becomes_a_real_line(h: H):
    """Gap rule: larger gap -> a real straight line, connected to shared points at both ends."""
    lines = _square_with_gap(h, 1.0)
    before = len(h.curves())
    h.v.do_fill()
    new = [c for c in h.curves() if not any(c is x for x in lines)]
    assert len(h.curves()) == before + 1 and len(new) == 1, "one bridge"
    br = new[0]
    ends = {network.pid_of(br, 0), network.pid_of(br, len(br.points) - 1)}
    assert ends == {network.pid_of(lines[3], len(lines[3].points) - 1), network.pid_of(lines[0], 0)}
    assert np.allclose(lines[0].points[0], [0, 0, 0]), "the rest stays where it is"
    surf = h.sc.surfaces[-1]
    assert br.id in {cid for cid, _ in surf.links}, "bridge belongs to the outline"


def test_curve_surface_with_gap(h: H):
    """Alt+F uses the same gap rule: an open square yields a curve surface including a bridge."""
    lines = _square_with_gap(h, 1.0)
    h.v.do_patch()
    surf = h.sc.surfaces[-1]
    assert isinstance(surf, PatchSurface), "curve surface is created despite the gap"
    assert len(h.sc.patch_border(surf)) == 5


def _outline(h: H, poly):
    """Closed, connected outline made of straight lines in the XZ plane, all selected."""
    sc = h.sc
    sc.new_stroke()
    pts = [np.array([x, 0.0, z]) for x, z in poly]
    lines = []
    for i in range(len(pts)):
        sc.add_curve(np.linspace(pts[i], pts[(i + 1) % len(pts)], 25), None)
        lines.append(sc.curves[-1])
    network.sync(sc)
    for i in range(len(lines)):
        network.join(sc, lines[i], 24, lines[(i + 1) % len(lines)], 0)
    h.v._changed()
    h.camera([2, 0, 2], 9, [0, 1, 0.01])
    h.edit_mode("edge")
    h.v.edge_selection = [("line", sc.owner_of(c).id, c.id) for c in lines]
    return lines


def test_f_is_a_surface_command(h: H):
    """F: curve surface if its grid does not fold (also triangle, hexagon); inward-kinked outline (L shape)
    -> fill surface, without folds. One undo step takes everything back."""
    from sketch_machine.scene import SurfaceObject
    hexagon = [[2 * np.cos(a), 2 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 7)[:-1]]
    for poly, kind in (([[0, 0], [4, 0], [2, 3.5]], PatchSurface), (hexagon, PatchSurface),
                       ([[0, 0], [4, 0], [4, 1.5], [1.5, 1.5], [1.5, 4], [0, 4]], SurfaceObject)):
        h.new_scene()
        _outline(h, poly)
        h.key(Qt.Key_F)
        assert len(h.sc.surfaces) == 1, f"{len(h.sc.surfaces)} surfaces"
        s = h.sc.surfaces[0]
        assert type(s) is kind, f"{len(poly)}-gon: {type(s).__name__} instead of {kind.__name__}"
        v, t = s.fill_mesh()[:2]
        n = np.cross(v[t][:, 1] - v[t][:, 0], v[t][:, 2] - v[t][:, 0])[:, 1]
        p = np.array(poly)
        true = 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
        assert abs(0.5 * np.abs(n).sum() - true) < 0.01 * true, "surface covers exactly the outline (no fold)"
    h.v.undo()
    assert not h.sc.surfaces, "one undo step"


def _pick_lines(h: H, lines):
    """Select lines one after another (like Shift+click in edge mode) - the order is remembered."""
    h.edit_mode("edge")
    h.v.selection = []
    h.v.edge_selection = []
    h.v._track_pick_order()
    for c in lines:
        h.v.edge_selection = h.v.edge_selection + [("line", h.sc.owner_of(c).id, c.id)]
        h.v._track_pick_order()


def test_loft_order_as_selected_and_middle_shapes(h: H):
    """Skyscraper test 2 (artist): two cross-sections at the bottom, a strongly curved one at the top whose centre lies between the
    others. Selected bottom, bottom, top -> exactly this order (guessed, the upper one would be in the middle). The
    middle cross-section then lies exactly on the curve surface, even if it deviates far from the raw surface."""
    sc = h.sc
    sc.new_stroke()
    t = np.linspace(0, 1, 60)
    a = np.stack([6 + 1.5 * t, np.zeros_like(t), -0.6 + 10.6 * t], 1)
    b = np.stack([2.2 - 3.7 * t, np.full_like(t, 2.5), 1.2 + 7.9 * t], 1)
    c = np.stack([2.1 + 0.9 * t + 1.2 * np.sin(t * 3 * np.pi), np.full_like(t, 3.8), 5.4 - 5.6 * t], 1)
    for p in (a, b, c):
        sc.add_curve(p, None)
    la, lb, lc = sc.curves[-3:]
    h.v._changed()
    from sketch_machine import loft as L
    guess = L.order([la.points, lb.points, lc.points])
    assert guess[1] == 2, "test setup: guessed, the upper one would be in the middle"
    _pick_lines(h, [la, lb, lc])
    h.v.do_loft()
    h.v._changed()
    s = sc.surfaces[-1]
    assert isinstance(s, PatchSurface)
    border = {x.id for x in sc.patch_border(s)}
    assert la.id in border and lc.id in border and lb.id not in border, "first and last as selected"
    assert sc.curve_by_id(lb.id).plane_id == s.id, "middle cross-section shapes the surface"
    verts, tris = s.fill_mesh()[:2]
    size = float(np.linalg.norm(np.ptp(verts, axis=0)))
    dd = _dist_to_mesh(sc.curve_by_id(lb.id).points, verts, tris)
    from sketch_machine import netsurf, components
    att = s.attached[lb.id]
    idx = netsurf.samples(lb.points, att["uv"], att.get("on"), components.knots(lb))
    d = dd.max()
    # right next to the rail, edge and cross-section hold the same grid node: there the solution averages slightly
    assert d < 0.005 * size, f"middle cross-section {d / size:.2%} beside the surface (at support points {dd[idx].max() / size:.2%}, at point {int(np.argmax(dd))} of {len(dd)}, support points {len(idx)})"


def test_loft_strongly_arched_middle_lies_on_the_surface(h: H):
    """Middle cross-section far above the raw surface (Coons from the edge alone): it is not bound by proximity,
    but by its position in the loft - the curve surface passes exactly through it, at every point of the line."""
    sc = h.sc
    sc.new_stroke()
    rows = []
    for z, lift in ((0.0, 0.0), (3.0, 3.0), (6.0, 0.0)):
        x = np.linspace(-3, 3, 11)  # roughly drawn: few points, distinct corners of the polyline
        sc.add_curve(np.stack([x, lift * (1 - (x / 3) ** 2), np.full_like(x, z)], 1), None)
        rows.append(sc.curves[-1])
    h.v._changed()
    _pick_lines(h, rows)
    h.v.do_loft()
    h.v._changed()
    s = sc.surfaces[-1]
    mid = sc.curve_by_id(rows[1].id)
    assert mid.plane_id == s.id
    verts, tris = s.fill_mesh()[:2]
    size = float(np.linalg.norm(np.ptp(verts, axis=0)))
    d = _dist_to_mesh(mid.points, verts, tris)
    assert d.max() < 0.003 * size, f"points beside the surface: {np.round(d / size, 4)}"


def test_loft_selected_surface_does_not_count_in_edge_mode(h: H):
    """After a loft the new surface stays selected. If lines for the next loft are then selected in edge mode,
    it must not count as an additional cross-section."""
    sc = h.sc
    sc.new_stroke()
    x = np.linspace(-2, 2, 20)
    lines = []
    for z in (0.0, 3.0, 10.0, 13.0):
        sc.add_curve(np.stack([x, np.zeros_like(x), np.full_like(x, z)], 1), None)
        lines.append(sc.curves[-1])
    h.v._changed()
    _pick_lines(h, lines[:2])
    h.v.do_loft()
    first = sc.surfaces[-1]
    assert h.v.selection == [first.id]
    h.v.edge_selection = []
    h.v._track_pick_order()
    for c in lines[2:]:
        h.v.edge_selection = h.v.edge_selection + [("line", sc.owner_of(c).id, c.id)]
        h.v._track_pick_order()
    h.v.do_loft()
    second = sc.surfaces[-1]
    assert second is not first
    verts = second.fill_mesh()[0]
    assert verts[:, 2].min() > 9.9, "second loft only through its two lines"


def test_loft_cross_section_from_pieces_via_points(h: H):
    """A cross-section made of two pieces that share a point is one cross-section - combined via the
    curve net, also in the middle of the selection."""
    sc = h.sc
    sc.new_stroke()
    sc.add_curve(np.linspace([-2, 0, 0], [2, 0, 0], 20), None)
    a = sc.curves[-1]
    sc.add_curve(np.linspace([-2, 1, 3], [0, 1.5, 3], 10), None)
    m1 = sc.curves[-1]
    sc.add_curve(np.linspace([0, 1.5, 3], [2, 1, 3], 10), None)
    m2 = sc.curves[-1]
    sc.add_curve(np.linspace([-2, 0, 6], [2, 0, 6], 20), None)
    b = sc.curves[-1]
    network.sync(sc)
    network.join(sc, m1, len(m1.points) - 1, m2, 0)
    h.v._changed()
    _pick_lines(h, [a, m1, m2, b])
    h.v.do_loft()
    h.v._changed()
    s = sc.surfaces[-1]
    assert isinstance(s, PatchSurface)
    border = {x.id for x in sc.patch_border(s)}
    assert a.id in border and b.id in border and m1.id not in border and m2.id not in border
    assert sc.curve_by_id(m1.id).plane_id == s.id and sc.curve_by_id(m2.id).plane_id == s.id
