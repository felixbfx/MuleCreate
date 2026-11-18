"""Shared imports, constants and geometry helpers of all viewport parts."""
import math
import types
import time

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QRectF, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF, QSurfaceFormat
from .backend import GPUBackend, USE_METAL  # Native MetalKit or OpenGL
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QHBoxLayout,
                               QInputDialog, QLabel, QMenu, QSlider, QWidget, QWidgetAction)

from ..camera import Camera, view_for_back
from ..clipboard import Clipboard
from ..gizmo import TOOLS, TransformGizmo
from ..note_tool import NotesTool
from .. import light as light_mod
from ..renderer import Renderer, ortho_matrix, projection_matrix
from .. import clipboard, components, edges, extrude, fill, loft, network, shapes
from ..scene import (SYMMETRY_PLANES, Curve, ExtrudeSurface, LoftSurface, MeshObject, NullObject, PatchSurface,
                    PivotHandle, Plane, StrokeObject, SurfaceObject, TransformSet, inherit)
from ..workspace import SHARED, Workspace
from .. import theme
from .. import timing as timing_mod
from ..theme import (AXIS_COLORS, CONNECT_COLOR, DRAWING_COLOR, ERASER_COLOR, MESH_TINT_ALPHA, PIVOT_COLOR,
                    PLANE_COLORS, SELECTED_STROKE_COLOR, STRAIGHT_COLOR, STROKE_COLOR)
from ..i18n import tr

FOV = math.radians(45.0)
NEAR = 0.05
NEAR_PER_DIST = 0.001  # front clipping plane grows along from NEAR at NEAR_PER_DIST = 50 m distance (ratio to far stays)
FLIGHT_SECONDS = 0.3
FOCUS_DIST = 20.0  # at plane size 10
PLANE_STEP = 0.25  # per mouse wheel notch at home distance, scaled with the camera distance
PIXELS_PER_STEP = 40  # Shift + RMB: this many pixels correspond to one notch
LIGHT_BUTTON = 11  # radius of the light buttons at the right viewport edge
SUN_DISC = 62  # radius of the sky disc (edit sun)
CLICK_PX = 4  # up to this mouse movement a press counts as a click
TABLET_CLICK_PX = 10  # with the tablet pen (it wobbles when tapping) …
TABLET_CLICK_S = 0.25  # … and put down this briefly: click instead of stroke
TABLET_MIN_STROKE_PX = 3  # shorter strokes with the pen are discarded (tapping)
SPOT_PX = 70  # radius of the highlight around the pen tip on the drawing plane
SPOT_ALPHA, SPOT_ALPHA_OUTSIDE = 35, 80  # subtle on the plane, clearer at infinity (outside the rectangle)
NOTE_GLOW_SCALE = 0.6  # artist: 60 % of the first version (0.2 was too weak)
NOTE_GLOW = {k: tuple(x * NOTE_GLOW_SCALE for x in v) for k, v in
             {1.0: (0.22, 0.75, 0.32, 0.89), 0.5: (0.12, 0.5, 0.18, 0.78)}.items()}  # selected note: connected parts
# in magenta – strength (1 = part at the end of a pointer line, 0.5 = arm of the octopus) -> (tint of the material, lines toward
# the color, tint of the meshes, opacity of the outline)
FAINT_MIX = 0.45  # sketch brief: parts also meant by a note (octopus) highlighted this much weaker than the anchor
EDIT_GLOW = 0.1  # edit mode: lines (edge mode) or surfaces (surface mode) this much toward the selection color
PICK_PX = 8  # click tolerance for lines
POINT_PX = 9  # point mode: a point (line end, corner) must be this close to the cursor
COMPONENT_NAMES = {"point": tr("Punkte"), "edge": tr("Kanten"), "face": tr("Flächen")}
END_SNAP_PX = 7  # line ends only snap this close – beyond that the line under the cursor wins (dense grids)
END_GAP_PX = 10
FREE_POINT_PX = 10  # free mode: points (ends, corners, inserted ones) snap this close – before lines
FREE_SPAN_MIN = 0.05  # free mode: the start–target connection points almost along the view direction (sin < this), no spanning plane
FREE_POINTS_MAX = 6000  # free mode: show at most this many points as small rings
LINE_EXACT_PX = 3  # click to activate / double click: a line must be this close, otherwise it applies to the surface below
ON_SURFACE_TOL = 0.004  # a line counts as "on the drawing surface" if it is this close everywhere (fraction of the surface size)
ON_SURFACE_SAMPLES = 12
NOTICE_S = 4.0  # a hint stays this long in the status line
TIE_SCREEN_PX = 1.5  # lines this close together on the screen count as on top of each other when picking
CONTINUE_PX = 24  # jump in space (measured in pixels) up to which a stroke continues onto a neighboring surface  # a stroke ends this far beside the drawing surface (shorter dropouts are bridged)
PRESSURE_MIN, PRESSURE_MAX = 0.1, 2.0  # pressure sensitivity to 1: lightest touch 10 %, firmest 200 %
TABLET_FRESH_S = 0.25  # mouse events this soon after a tablet event come from the pen
HOLD_MS = 250  # holding still at the stroke end -> line or shape (artist 11.11.: half as long as before)
HOLD_PX = 4
SURFACE_SAMPLE_PX = 2  # sampling distance when a straight line is projected onto a mesh
NEW_PLANE_VIEW_FRACTION = 0.6  # new plane fills this fraction of the image height
FUSE_MAX_DEG = 15  # Fuse only connects where the line continues without a kink (corners stay separate)
FUSE_MEASURE_PX = 12  # measure direction at the line end over this many pixels (evens out jitter, hardly any curvature)
BORDER_SNAP = 0.015  # stroke beyond the surface edge: ends on a border line this close (fraction of the surface size)
BORDER_REACH = 0.08  # … or on the border line it hits in its direction within this distance (fast stroke, smoothing)
BORDER_DIR = 0.02  # stroke direction at the end: measured over this length
BORDER_HIT = 0.006  # the extended direction must pass the border line this closely
SNAP_PX = 12  # snap tolerance: line ends closer than this count as connected when filling
LOOP_CHOICES = 4  # this many different loops per line are available when clicking repeatedly
PEN_FEEDBACK_SECONDS = 1.0  # the size circle stays visible this long after adjusting
ORTHO_DEPTH_MIN = 100.0  # orthographic visible: this far in front of and behind the camera ...
ORTHO_DEPTH_FACTOR = 20.0  # ... at least, otherwise this many camera distances
ERASE_SAMPLE_PX = 1.0  # accuracy of the eraser cuts in pixels
ERASE_MAX_SAMPLES = 2000  # at most this many test points per line piece
OCCLUDE_TOL = 0.002  # a line counts as occluded if a mesh lies more than this (x camera distance) in front of it
PIVOT_SNAP_PX = 16  # axis mode, V held: the pivot snaps to points this close to the cursor
BOX_SAMPLE_PX = 2.0  # selection rectangle: check lines at this distance for contact
BOX_MAX_CHECKS = 24  # check at most this many points per line for visibility
# axis views: view direction (back) -> name as in Houdini
VIEW_NAMES = (((0, 1, 0), "Top"), ((0, -1, 0), "Bottom"), ((0, 0, 1), "Front"), ((0, 0, -1), "Back"),
              ((1, 0, 0), "Right"), ((-1, 0, 0), "Left"))

CLOSE_BUTTON = 16  # size of the X for closing split viewports (top right)
GIZMO_RADIUS = 38
GIZMO_MARGIN = 62
AXES = tuple((name, axis, color) for (name, axis), color in zip(
    (("X", np.array([1.0, 0, 0])), ("Y", np.array([0, 1.0, 0])), ("Z", np.array([0, 0, 1.0]))), AXIS_COLORS))

# colors (scheme "Zeichenbüro") are in theme.py: active = Prussian blue, selected = terracotta,
# both = sage green
GENERATED_PAPER_MIX = 0.45  # connecting lines added by filling: lighter (toward paper)


class _TriangleGrid:
    """Grid over the triangles of a surface: which triangles are near a point?"""

    def __init__(self, verts, tris):
        v0, v1, v2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
        self.v0, self.e1, self.e2 = v0, v1 - v0, v2 - v0
        n = np.cross(self.e1, self.e2)
        size = np.linalg.norm(n, axis=1)
        self.ok = size > 1e-9 * max(float(size.max()), 1e-18) if len(size) else size > 0  # slivers without direction
        self.n = np.where(self.ok[:, None], n / np.maximum(size, 1e-30)[:, None], 0.0)
        self.cent = (v0 + v1 + v2) / 3
        lo = np.minimum(np.minimum(v0, v1), v2)
        hi = np.maximum(np.maximum(v0, v1), v2)
        size = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) if len(verts) else 1.0
        self.cell = max(float(np.percentile(np.max(hi - lo, axis=1), 95)) * 2, size * 1e-3, 1e-9)
        a = np.floor(lo / self.cell).astype(np.int64)
        b = np.floor(hi / self.cell).astype(np.int64)
        self.cells = {}
        for k in range(len(tris)):
            for x in range(a[k, 0], b[k, 0] + 1):
                for y in range(a[k, 1], b[k, 1] + 1):
                    for z in range(a[k, 2], b[k, 2] + 1):
                        self.cells.setdefault((x, y, z), []).append(k)
        self.cells = {key: np.array(v, np.int64) for key, v in self.cells.items()}

    def near(self, point):
        return self.near_cell(tuple(int(c) for c in np.floor(np.asarray(point) / self.cell)))

    def near_cell(self, key):
        """Triangles in cell key and its neighbors (computed once per cell)."""
        memo = self.__dict__.setdefault("_near", {})
        if key not in memo:
            x, y, z = key
            parts = [self.cells[k] for k in ((x + dx, y + dy, z + dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                             for dz in (-1, 0, 1)) if k in self.cells]
            memo[key] = np.unique(np.concatenate(parts)) if parts else None
        return memo[key]


# (id points, id triangles) -> grid. The shape is in both: net fill surfaces keep the same triangle array (rest net)
# on every shape change, only the points are new – by triangles alone the pen would hit the old shape
_TRI_GRIDS = {}


def _triangle_grid(mesh):
    key = (id(mesh[0]), id(mesh[1]))
    entry = _TRI_GRIDS.get(key)
    if entry is None or entry[0] is not mesh[0] or entry[1] is not mesh[1]:
        if len(_TRI_GRIDS) > 64:
            _TRI_GRIDS.clear()
        entry = _TRI_GRIDS[key] = (mesh[0], mesh[1], _TriangleGrid(np.asarray(mesh[0], float), np.asarray(mesh[1])))
    return entry[2]


def _grid_step(verts):
    """Grid size for a drawing surface: about a tenth of its size, as a round number (1, 2, 5 …)."""
    size = float(np.max(np.ptp(verts, axis=0))) / 10 if len(verts) else 1.0
    base = 10 ** math.floor(math.log10(max(size, 1e-6)))
    return min((m * base for m in (1, 2, 5, 10)), key=lambda x: abs(math.log(x / max(size, 1e-6))))


def _dist_to_triangles_fast(points, tri, block=512):
    """Smallest distance of each point (N,3) to a set of triangles (M,3,3), vectorized in blocks."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(b - a, c - a)
    raw = np.einsum("ij,ij->i", n, n)
    valid = raw > 1e-12 * max(float(raw.max()), 1e-30)  # sliver triangles (area 0) have no "inside"
    nn = np.maximum(raw, 1e-18)
    out = np.full(len(points), np.inf)
    for k in range(0, len(points), block):
        p = points[k:k + block][:, None, :]  # (B, 1, 3)
        ap = p - a[None]
        t = np.einsum("bmj,mj->bm", ap, n) / nn  # distance to the plane (scaled)
        q = p - t[..., None] * n[None]
        c0 = np.einsum("bmj,mj->bm", np.cross(b - a, q - a[None]), n)
        c1 = np.einsum("bmj,mj->bm", np.cross(c - b, q - b[None]), n)
        c2 = np.einsum("bmj,mj->bm", np.cross(a - c, q - c[None]), n)
        inside = (c0 >= 0) & (c1 >= 0) & (c2 >= 0) & valid[None]
        d_plane = np.abs(t) * np.sqrt(nn)[None]
        best = np.where(inside, d_plane, np.inf).min(axis=1)
        for s, e in ((a, b), (b, c), (c, a)):
            se = e - s
            u = np.clip(np.einsum("bmj,mj->bm", p - s[None], se) / np.maximum(np.einsum("mj,mj->m", se, se), 1e-18), 0, 1)
            best = np.minimum(best, np.linalg.norm(s[None] + u[..., None] * se[None] - p, axis=2).min(axis=1))
        out[k:k + block] = best
    return out


def _near_mesh(points, mesh, tol):
    """Which points are at most tol from the surface? Each point is checked only against the triangles
    of its grid cell and the neighboring cells (_TriangleGrid), grouped by cell – instead of against all triangles
    (curve surfaces have over 13,000; that cost up to a minute on double click)."""
    points = np.asarray(points, float)
    out = np.zeros(len(points), bool)
    if not len(points):
        return out
    grid = _triangle_grid(mesh)
    if tol > grid.cell:  # margin larger than a cell: the neighboring cells are not enough, check all
        verts, tris = mesh[0], mesh[1]
        return _dist_to_triangles_fast(points, np.asarray(verts, float)[tris]) <= tol
    cells = np.floor(points / grid.cell).astype(np.int64)
    uniq, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    for u, key in enumerate(uniq.tolist()):
        cand = grid.near_cell(tuple(key))
        if cand is None:
            continue
        members = np.flatnonzero(inverse == u)
        v0 = grid.v0[cand]
        tri = np.stack([v0, v0 + grid.e1[cand], v0 + grid.e2[cand]], axis=1)
        out[members] = _dist_to_triangles_fast(points[members], tri) <= tol
    return out


def _closest_on_polyline(points, p):
    """Point on the polyline closest to p in space."""
    a, b = points[:-1], points[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0, 1)
    q = a + ab * t[:, None]
    return q[int(np.argmin(np.linalg.norm(q - p, axis=1)))].copy()


def _round_nice(x):
    """Round to two significant digits (e.g. 7.43 -> 7.4, 123 -> 120)."""
    if x <= 0:
        return 1.0
    digits = 1 - math.floor(math.log10(x))
    return round(x, digits)


def _rgba(color):
    return color.redF(), color.greenF(), color.blueF(), color.alphaF()


def _capsule_dist(points, a, b):
    """Screen distance of the points (N, 2) to the segment a-b (the path of the eraser)."""
    ab = b - a
    length = ab @ ab
    t = np.clip((points - a) @ ab / length, 0, 1) if length > 1e-12 else np.zeros(len(points))
    return np.linalg.norm(a + t[:, None] * ab - points, axis=1)


def _shared(name):
    """Attribute that lives in the workspace and applies to all viewports."""
    return property(lambda self: getattr(self.ws, name), lambda self, value: setattr(self.ws, name, value))


__all__ = [_n for _n in dict(globals()) if not _n.startswith("__")]
