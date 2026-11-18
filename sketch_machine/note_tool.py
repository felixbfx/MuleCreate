"""Notes mode (N): draw, create, move, connect and rotate notes in the viewport.

The heading is projected into the note plane with QPainter in perspective (stays sharp) and lies
over the drawing, like labels on a plan. Brackets at the top left and bottom right, the
leader lines run at right angles (see notes.leader).

Note octopus: from every anchor point arms point to further parts of the same kind (NoteMember) – Shift+drag from the
anchor point or "Add selection to octopus" in the right-click menu. The arms are curved and magenta and only visible on the
selected note, so that they are clearly distinguished from the leader lines.
"""
import math
import time

import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen, QPolygonF, QTransform
from PySide6.QtWidgets import QLineEdit, QMenu

from . import notes, theme
from .scene import Note, NoteAnchor, NoteMember
from .i18n import tr

FONT_PX = 64.0  # reference size of the text: layout in these pixels, then scaled into the world
FAMILIES = ["Bahnschrift Light", "Segoe UI Light", "Helvetica Neue", "Noto Sans", "DejaVu Sans"]
PAD = 0.32  # distance of the brackets around the text (fraction of the line height)
BRACKET = 0.5  # leg length of the brackets (fraction of the line height)
NEW_OFFSET_PX = (40, 72)  # new note on a line: this far to the right and above the anchor point
ICON_R = 9  # radius of the + and maximize buttons
HANDLE_R = 5  # ring at the lower left corner: drag = new leader line
DOT_R = 2.8
NEAR_DEPTH = 0.05  # point at the end of a leader line
HIT_PX = 8
STICKY_PX = 34  # maximize button stays as long as the cursor is this close to the note or button …
STICKY_S = 0.6  # … or this long after leaving
ROTATE_PER_PX = 0.01  # rotate (R): radians per pixel of mouse movement
ROTATE_SNAP = math.radians(15)  # with Ctrl
LINE_W = 1.3
ARM_W = 1.8  # arms of the note octopus
ARM_ALPHA = 210  # slightly transparent
ARM_GLOW = (6.0, 40)  # soft glow beneath: width, opacity
ARM_BEND = 0.22  # bend of the arms (fraction of their length on screen)
ARM_RING_R = 3.6  # open ring at the end of an arm
OCTO_DOT_R = 3.8  # selected note: anchor point with octopus – point, ring around it, soft glow (width, opacity)
OCTO_RING_R = 7.0
OCTO_GLOW = (5.0, 60)
OCCL_ARM = 24  # occlusion: samples per arm …
ARM_LIFT_PX = 4.0  # … checked this many pixels toward the camera (arms often lie on the surface they connect)
ARM_SAMPLES = 4000  # selection to octopus: at most this many points per part, to find the place nearest to the anchor
OCCL_GRID = (24, 6)  # occlusion: samples over the note plate (columns, rows)
OCCL_LEADER = 24  # … and per piece of a leader line
HIDDEN_ALPHA = 55  # notes mode: occluded parts pale (for finding them again), otherwise invisible
PLACEHOLDER = tr("Notiz")
PLACEHOLDER_ALPHA = 165
MIN_LINE_PX = {"s": 2.0, "m": 2.4, "l": 2.8, "xl": 3.4}  # never smaller from a distance (line height in px; artist: may become very small)
# … but in notes mode legible and grabbable: the text is fixed in world units (22–90 cm); in a scene of
# kilometers a dot was left of it at normal distance, which could neither be read nor clicked (artist 12.11.,
# harbor scene). Outside the mode the very small limit above still applies.
MIN_LINE_PX_NOTES = {"s": 11.0, "m": 13.0, "l": 15.0, "xl": 18.0}
# Backing of the plate in paper color (dark theme: dark gray, light: light), so that the text stays legible even in front of a
# light or dark object (artist 12.11.) – opacity 0…1
NOTE_BACK_ALPHA = 0.55  # (the artist named 10 %: in the image that is not enough – light text in front of a white surface stays illegible; from about half it works)
NOTE_SCALES = (1, 2, 5, 10, 25, 50, 100)  # note size of the scene: presets in the right-click menu
# Text is rendered into an image of roughly screen size (raster engine, clean anti-aliasing and hinting)
# and then drawn in perspective – directly projected text was barely legible when small
TEXT_BUCKETS = (2, 3, 4, 6, 8, 10, 12, 14, 17, 20, 24, 29, 35, 42, 50, 60, 72, 86, 104, 125, 150)  # line heights in px
TEXT_CACHE_MAX = 400


def camera_glyph(p, c, r, color):
    """Camera symbol around the center c (radius r): body, viewfinder, lens."""
    p.save()
    p.setPen(QPen(color, max(1.0, r * 0.14)))
    p.setBrush(Qt.NoBrush)
    w, h = r * 1.15, r * 0.78
    body = QRectF(c.x() - w / 2, c.y() - h / 2 + r * 0.08, w, h)
    p.drawRoundedRect(body, r * 0.12, r * 0.12)
    p.drawRect(QRectF(c.x() - w * 0.18, body.top() - r * 0.16, w * 0.36, r * 0.16))
    p.drawEllipse(QPointF(c.x(), body.center().y()), h * 0.3, h * 0.3)
    p.restore()


def note_font(px):
    font = QFont()
    font.setFamilies(FAMILIES)
    font.setWeight(QFont.Light)
    font.setPixelSize(max(1, int(round(px))))
    return font


_METRICS = {}
_TEXT_IMAGES = {}


def _text_image(text, color, line_px, dpr):
    """Heading as an image with line height ~line_px (screen pixels), cached."""
    bucket = next((b for b in TEXT_BUCKETS if b >= line_px), TEXT_BUCKETS[-1])
    key = (text, color.rgba(), bucket, dpr)
    image = _TEXT_IMAGES.get(key)
    if image is None:
        w, h, ascent = _layout(text)
        scale = bucket * dpr / h
        image = QImage(max(1, int(w * scale + 3)), max(1, int(h * scale + 2)), QImage.Format_ARGB32_Premultiplied)
        image.fill(0)
        p = QPainter(image)
        p.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        p.setFont(note_font(FONT_PX * scale))
        p.setPen(color)
        p.drawText(QPointF(1, ascent * scale + 1), text)
        p.end()
        if len(_TEXT_IMAGES) > TEXT_CACHE_MAX:
            _TEXT_IMAGES.clear()
        image = _TEXT_IMAGES[key] = (image, scale)
    return image


def _layout(title):
    """(Width, height, ascender) of the heading in reference pixels."""
    text = title or PLACEHOLDER
    if text not in _METRICS:
        fm = QFontMetricsF(note_font(FONT_PX))
        _METRICS[text] = (max(fm.horizontalAdvance(text), FONT_PX * 0.8), fm.ascent() + fm.descent(), fm.ascent())
    return _METRICS[text]


class NotesTool:
    def __init__(self, vp):
        self.vp = vp
        self.hover = None  # (kind, note id, index) under the cursor
        self.sticky = None  # (note id, time): note at which the maximize button still stands
        self.drag = None  # running drag: dict with "kind"
        self.editor = None  # (QLineEdit, note id, new?)
        self.last_size = "m"
        self.dismiss_click = False  # the next click only ends typing
        self._screen = {}  # note id -> screen geometry of the last image (for hits)
        self.occlusion = None  # note id -> (plate grid, [leader lines]) visible? – per image, None: everything visible

        # --- Geometry ---------------------------------------------------------
    def geometry(self, note):
        w, h, ascent = _layout(note.title)
        line = notes.SIZES.get(note.size, notes.SIZES["m"]) * float(getattr(self.vp.scene, "note_scale", 1.0))
        floor = MIN_LINE_PX_NOTES if self.vp.notes_mode else MIN_LINE_PX
        line = max(line, floor.get(note.size, 12) * self._wpp(note.pos))  # minimum size on screen
        k = line / h  # world units per reference pixel
        r = notes.right_vector(note, self.vp.cam.right)
        pad = PAD * h

        def world(x, y):  # reference pixel (y down, 0 = top edge of the text) -> world
            return note.pos + r * (x * k) + notes.UP * ((h - y) * k)

        box = [world(-pad, -pad), world(w + pad, -pad), world(w + pad, h + pad), world(-pad, h + pad)]
        left, right = world(-pad, h + pad), world(w + pad, h + pad)
        leaders = [notes.leader(left, right, a.point) for a in note.anchors]
        return dict(w=w, h=h, ascent=ascent, k=k, r=r, pad=pad, world=world, box=box, leaders=leaders)

    @staticmethod
    def _cells(g):
        """Centers of the sample cells over the plate in reference pixels (rows × columns) and the cell size."""
        cols, rows = OCCL_GRID
        x0, y0, cw, ch = -g["pad"], -g["pad"], (g["w"] + 2 * g["pad"]) / cols, (g["h"] + 2 * g["pad"]) / rows
        return [(x0 + (i + 0.5) * cw, y0 + (j + 0.5) * ch) for j in range(rows) for i in range(cols)], (x0, y0, cw, ch)

    @staticmethod
    def _leader_samples(lead):
        start, knee, end = (np.asarray(q, float) for q in lead)
        t = np.linspace(0, 1, OCCL_LEADER + 1)[:, None]
        return np.concatenate([start + (knee - start) * t, (knee + (end - knee) * t)[1:]])

    def visibility(self, points_visible):
        """Which parts of the notes in the image just rendered are not occluded by surfaces/meshes – the same
        coverage map as for lines and points. points_visible: renderer.points_visible (directly after render)."""
        pts, slots = [], []
        for note in self.vp.note_list():
            g = self.geometry(note)
            cells, _ = self._cells(g)
            box = [g["world"](x, y) for x, y in cells]
            leads = [self._leader_samples(lead) for lead in g["leaders"]]
            slots.append((note.id, len(pts), len(box), [len(s) for s in leads]))
            pts += box
            for s in leads:
                pts += list(s)
        arms = []  # arms of the octopus: only visible at the clicked anchor of the selected note
        focus = self.vp.ws.active_anchor
        note = self.vp.scene.note_by_id(focus[0]) if focus and self.vp.notes_active else None
        if note is not None and focus[0] == self.vp.ws.active_note and focus[1] < len(note.anchors):
            for i, e in self._arm_ends(note, focus[1]):
                arms.append(((note.id, focus[1], i), len(pts)))
                samples = self.arm_samples(note.anchors[focus[1]].point, e)
                eye = self.vp.cam.eye  # a few pixels toward the camera: the arm often lies exactly on the surface
                lift = np.broadcast_to(self.vp.cam.back, samples.shape) if self.vp.cam.ortho else eye - samples
                lift = lift / np.maximum(np.linalg.norm(lift, axis=1, keepdims=True), 1e-9)
                pts += list(samples + lift * np.array([[ARM_LIFT_PX * self._wpp(q)] for q in samples]))
        if not pts:
            return {}
        vis = points_visible(np.array(pts))
        out = {}
        for nid, at, nbox, nleads in slots:
            box = vis[at:at + nbox].reshape(OCCL_GRID[1], OCCL_GRID[0])
            at += nbox
            leads = []
            for n in nleads:
                leads.append(vis[at:at + n])
                at += n
            out[nid] = (box, leads)
        out["_arms"] = {key: vis[at:at + OCCL_ARM + 1] for key, at in arms}
        return out

    def _depth(self, point):
        return float((point - self.vp.cam.eye) @ -self.vp.cam.back)

    def _wpp(self, point):
        return float(self.vp._world_per_px(max(self._depth(point), 1e-3)))

        # --- Drawing ----------------------------------------------------------
    def draw(self, p):
        vp = self.vp
        self._screen = {}
        if not vp.ws.notes_visible or not vp.note_list():
            return
        active = vp.ws.active_note
        editing = self.editor[1] if self.editor else None
        hover_id = self.hover[1] if self.hover else None
        for note in sorted(vp.note_list(), key=lambda n: -self._depth(n.pos)):  # back to front
            g = self.geometry(note)
            pts = np.array(g["box"] + [q for lead in g["leaders"] for q in lead])
            s, ok = vp.project(pts)
            if not ok[:4].all():  # plate behind the camera: do not draw the note – but the octopus of the clicked
                self._arms_only(p, note, s, ok)  # anchor, yes (one is close to the parts)
                continue
            quad = QPolygonF([QPointF(*s[i]) for i in range(4)])
            in_mode = vp.notes_active
            color = QColor(theme.GRAPHITE)
            if in_mode and note.id == active:
                color = QColor(theme.TERRACOTTA)
            elif in_mode and note.id == hover_id:
                color = QColor(theme.OCHRE)
            elif not in_mode:
                color.setAlpha(205)
            if NOTE_BACK_ALPHA > 0:
                back = QColor(theme.PAPER)
                back.setAlphaF(NOTE_BACK_ALPHA * (color.alphaF() if not in_mode else 1.0))
                p.setPen(Qt.NoPen)
                p.setBrush(back)
                p.drawPolygon(quad)
            info = {"quad": quad, "anchors": [], "icons": {}, "handle": None, "members": []}
            occl = self.occlusion.get(note.id) if self.occlusion else None
            faint = QColor(color)
            faint.setAlpha(HIDDEN_ALPHA)
            # Leader lines (right-angled): start, bend, target – occluded pieces like with any line
            for j, a in enumerate(note.anchors):
                idx = 4 + 3 * j
                if not ok[idx:idx + 3].all():
                    continue
                end = QPointF(*s[idx + 2])
                info["anchors"].append((j, end))
                seen = occl[1][j] if occl is not None and j < len(occl[1]) else None
                for part_color, show in ((faint, False), (color, True)) if seen is not None else ((color, True),):
                    if not show and not in_mode:
                        continue  # outside notes mode occluded parts are invisible
                    pen = QPen(part_color, LINE_W, Qt.DashLine if a.lost else Qt.SolidLine)
                    pen.setCosmetic(True)
                    p.setPen(pen)
                    p.setBrush(Qt.NoBrush)
                    if seen is None:
                        p.drawPolyline(QPolygonF([QPointF(*s[idx + m]) for m in range(3)]))
                    else:
                        samples = self._leader_samples(g["leaders"][j])
                        q, qok = vp.project(samples)
                        run = []
                        for k in range(len(samples)):  # contiguous pieces of the same visibility
                            if qok[k] and bool(seen[k]) == show:
                                run.append(QPointF(*q[k]))
                            else:
                                if len(run) > 1:
                                    p.drawPolyline(QPolygonF(run))
                                run = []
                        if len(run) > 1:
                            p.drawPolyline(QPolygonF(run))
                    if seen is None or bool(seen[-1]) == show:
                        octo = in_mode and note.id == active and bool(a.members)
                        if octo:  # selected note, anchor with octopus: magenta, larger, with glow – "click here"
                            self._draw_octo_dot(p, end, show, vp.ws.active_anchor == (note.id, j))
                        else:
                            p.setBrush(Qt.NoBrush if a.lost else part_color)
                            p.drawEllipse(end, DOT_R, DOT_R)
                        if show and in_mode and note.id == active and vp.ws.active_anchor == (note.id, j) and                                 not octo:  # clicked: ring
                            ring = QPen(theme.OCTOPUS, 1.6)
                            p.setPen(ring)
                            p.setBrush(Qt.NoBrush)
                            p.drawEllipse(end, DOT_R + 3.2, DOT_R + 3.2)
            focus = vp.ws.active_anchor
            if in_mode and note.id == active and focus and focus[0] == note.id:  # arms only at the clicked anchor
                info["members"] = self._draw_arms(p, note, s, ok, focus[1])
            # Heading and brackets in the note plane
            w, h, pad = g["w"], g["h"], g["pad"]
            src = QPolygonF([QPointF(-pad, -pad), QPointF(w + pad, -pad), QPointF(w + pad, h + pad), QPointF(-pad, h + pad)])
            xf = QTransform()
            if not QTransform.quadToQuad(src, quad, xf):
                continue
            if occl is None or occl[0].all():  # fully visible: without clipping (brackets lie on the edge)
                self._draw_label(p, note, g, s, xf, color, editing)
            elif not occl[0].any():  # fully occluded: pale in notes mode, otherwise nothing
                if in_mode:
                    self._draw_label(p, note, g, s, xf, faint, editing)
            else:  # partly occluded: only the visible cells (in notes mode the occluded pale)
                for part_color, show in ((faint, False), (color, True)):
                    if show or in_mode:
                        self._draw_label(p, note, g, s, xf, part_color, editing, clip=self._clip(g, occl[0] == show))
            # Buttons and connection ring: only in notes mode, on the note under the cursor or the active one
            sticky_id = self.sticky[0] if self.sticky else None
            if in_mode and note.id in (hover_id, active, sticky_id):
                top_right, bottom_left = QPointF(*s[1]), QPointF(*s[3])
                info["icons"]["max"] = top_right + QPointF(ICON_R + 8, 0)
                info["icons"]["cam"] = top_right + QPointF(ICON_R + 8, 2 * ICON_R + 6)  # view as image onto the board
                info["handle"] = bottom_left
                self._draw_buttons(p, info, color)
            self._screen[note.id] = info
        self._draw_drag(p)

        # --- Note octopus --------------------------------------------------------
    def member_end(self, m):
        """End of an arm in the world: the remembered place – if it no longer lies on the part (moved), its center.
        None: the part is deleted and no place known."""
        scene = self.vp.scene
        if m.curve_id is not None:
            curve = scene.curve_by_id(m.curve_id)
            return notes.curve_point(curve.points, m.t) if curve is not None else m.point
        obj = scene.object_by_id(m.obj_id)
        pts = _member_points(obj, scene) if obj is not None else None
        if pts is None or not len(pts):
            return m.point
        lo, hi = pts.min(0), pts.max(0)
        slack = 0.02 * (hi - lo) + 1e-6
        if m.point is not None and np.all(m.point >= lo - slack) and np.all(m.point <= hi + slack):
            return m.point
        return (lo + hi) / 2

    def member_alive(self, m):
        scene = self.vp.scene
        return scene.object_by_id(m.obj_id) is not None and (m.curve_id is None
                                                              or scene.curve_by_id(m.curve_id) is not None)

    @staticmethod
    def _arm_path(a, b):
        """Curved arm from a to b (screen, while dragging): quadratic curve, arc to the side."""
        path = QPainterPath(a)
        d = b - a
        c = (a + b) / 2 + QPointF(-d.y(), d.x()) * ARM_BEND
        path.quadTo(c, b)
        return path

    def _arms_only(self, p, note, s, ok):
        """Note without plate in the image: only anchor point and arms of the clicked anchor (stays clickable)."""
        vp = self.vp
        focus = vp.ws.active_anchor
        if not (vp.notes_active and note.id == vp.ws.active_note and focus and focus[0] == note.id
                and focus[1] < len(note.anchors)):
            return
        j = focus[1]
        info = {"quad": QPolygonF(), "anchors": [], "icons": {}, "handle": None, "members": []}
        info["members"] = self._draw_arms(p, note, s, ok, j)
        if ok[4 + 3 * j + 2]:
            end = QPointF(*s[4 + 3 * j + 2])
            info["anchors"].append((j, end))
            self._draw_octo_dot(p, end, True, True)
        self._screen[note.id] = info

    def _draw_octo_dot(self, p, end, show, chosen):
        """Anchor point with octopus on the selected note: magenta, larger, soft ring around it (clicked: stronger
        ring). Occluded: pale like everything occluded."""
        p.save()
        p.setRenderHint(QPainter.Antialiasing)
        scale = 1.0 if show else HIDDEN_ALPHA / 255
        ink = QColor(theme.OCTOPUS)
        for width, alpha, r in ((OCTO_GLOW[0], OCTO_GLOW[1], OCTO_RING_R), (1.4 if not chosen else 2.0,
                                                                            150 if not chosen else 235, OCTO_RING_R)):
            c = QColor(ink)
            c.setAlpha(int(alpha * scale))
            pen = QPen(c, width)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(end, r, r)
        ink.setAlpha(int(255 * scale))
        p.setPen(Qt.NoPen)
        p.setBrush(ink)
        p.drawEllipse(end, OCTO_DOT_R, OCTO_DOT_R)
        p.restore()

    def arm_samples(self, anchor, end):
        """Arm from the anchor to the end of an arm as points in the world: quadratic arc, bent to the side on screen
        (bend ARM_BEND of its length) – in the world, so that surfaces can occlude it like a line."""
        cam = self.vp.cam
        a, b = np.asarray(anchor, float), np.asarray(end, float)
        mid = (a + b) / 2
        d = b - a
        dx, dy = float(d @ cam.right), -float(d @ cam.up)  # screen direction (y down), in world lengths
        bend = (cam.right * -dy - cam.up * dx) * ARM_BEND  # (-dy, dx) on screen, back into the world
        c = mid + bend
        ray = cam.eye - c  # arched toward the camera (along the line of sight: the same on screen), otherwise the
        gap = float(np.linalg.norm(ray))  # arc would run behind the surface – near the camera at most up to half the
        if getattr(cam, "ortho", False):  # orthographic: sight rays parallel
            c = c + cam.back * float(np.linalg.norm(bend))
        else:
            c = c + ray / max(gap, 1e-9) * min(float(np.linalg.norm(bend)), 0.5 * gap)  # distance (never behind it)
        t = np.linspace(0.0, 1.0, OCCL_ARM + 1)[:, None]
        return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b

    def _arm_ends(self, note, j):
        """[(arm index, world end)] of the living or still known ends at anchor j."""
        anchor = note.anchors[j]
        return [(i, e) for i, m in enumerate(anchor.members) for e in [self.member_end(m)] if e is not None]

    def _draw_arms(self, p, note, s, ok, j):
        """Draw arms at the clicked anchor j of the selected note – occluded pieces pale like with the
        leader lines -> [((anchor, arm), screen point)] for the hits."""
        out = []
        if j >= len(note.anchors):
            return out
        anchor = note.anchors[j]
        seen_all = (self.occlusion or {}).get("_arms", {})
        hover = self.hover
        for i, e in self._arm_ends(note, j):
            m = anchor.members[i]
            samples = self.arm_samples(anchor.point, e)
            q, qok = self.vp.project(samples)
            if qok.sum() < 2:  # fully behind the camera
                continue
            end = QPointF(*q[-1]) if qok[-1] else None  # end behind the camera: only the visible piece
            seen = seen_all.get((note.id, j, i))
            hot = hover is not None and hover[0] == "member" and hover[1] == note.id and hover[2] == (j, i)
            p.save()
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(Qt.NoBrush)
            for show in ((False, True) if seen is not None else (True,)):
                k = 1.0 if show else HIDDEN_ALPHA / ARM_ALPHA
                runs, run = [], []
                for n in range(len(samples)):  # contiguous pieces of the same visibility
                    if qok[n] and (seen is None or bool(seen[n]) == show):
                        run.append(QPointF(*q[n]))
                    else:
                        if len(run) > 1:
                            runs.append(run)
                        run = [QPointF(*q[n])] if qok[n] and run else []
                if len(run) > 1:
                    runs.append(run)
                ink = QColor(theme.TERRACOTTA if hot else theme.OCTOPUS)
                for width, alpha, style in ((ARM_GLOW[0] + (2.0 if hot else 0.0), ARM_GLOW[1], Qt.SolidLine),
                                            (ARM_W + (0.6 if hot else 0.0), ARM_ALPHA,
                                             Qt.SolidLine if self.member_alive(m) else Qt.DashLine)):
                    c = QColor(ink)
                    c.setAlpha(int(alpha * k))
                    pen = QPen(c, width, style, Qt.RoundCap, Qt.RoundJoin)
                    pen.setCosmetic(True)
                    p.setPen(pen)
                    for r in runs:
                        p.drawPolyline(QPolygonF(r))
            if end is None:
                p.restore()
                continue
            end_seen = seen is None or bool(seen[-1])
            ring = QColor(theme.TERRACOTTA if hot else theme.OCTOPUS)
            ring.setAlpha(ARM_ALPHA if end_seen else HIDDEN_ALPHA)
            pen = QPen(ring, ARM_W)
            pen.setCosmetic(True)
            p.setPen(pen)
            paper = QColor(theme.PAPER)
            paper.setAlpha(255 if end_seen else HIDDEN_ALPHA)
            p.setBrush(paper)
            p.drawEllipse(end, ARM_RING_R, ARM_RING_R)
            p.restore()
            out.append(((j, i), end))
        return out

    def _member_target(self, pos, group=False):
        """Part under the cursor for an arm: line (of a stroke object) or surface/mesh – never a plane.
        Copies under symmetry/array are not clickable: the original is hit. group: the null the
        part lies in (whole group; without a null the part itself)."""
        vp = self.vp
        curve, point, _ = vp._front_curve(pos, exact_over_surface=True)  # the frontmost surface wins
        if curve is not None:
            owner = vp.scene.owner_of(curve)
            if owner is not None:
                if group:
                    return self._group_member(owner, point)
                _, t = notes.curve_param(curve.points, point)
                return NoteMember(owner.id, curve.id, notes.curve_point(curve.points, t), t)
        obj, hit, _ = vp._mesh_under(pos)
        if obj is None or hit is None or getattr(obj, "aid", False):
            return None
        if group:
            return self._group_member(obj, hit)
        return NoteMember(obj.id, None, np.asarray(hit, float))

    def _group_member(self, obj, point):
        null = self.vp.scene.parent_of(obj)
        return NoteMember((null or obj).id, None, np.asarray(point, float))

    def click_member(self, pos, group=False):
        """Shift+click in notes mode on a part: onto the octopus of the selected anchor (if the note is selected and has
        only one leader line: onto that one) – Shift+click again removes it. True: click consumed."""
        vp = self.vp
        note = vp.scene.note_by_id(vp.ws.active_note) if vp.ws.active_note is not None else None
        if note is None or not any(n is note for n in vp.note_list()):
            return False
        target = self._member_target(pos, group)
        if target is None:
            return False
        if vp.ws.active_anchor and vp.ws.active_anchor[0] == note.id and vp.ws.active_anchor[1] < len(note.anchors):
            j = vp.ws.active_anchor[1]
        elif len(note.anchors) == 1:
            j = 0
            vp.ws.active_anchor = (note.id, 0)  # so its arms are visible
        else:
            vp.notify(tr("Krake: erst den Ankerpunkt anklicken, an den die Teile sollen (die Notiz hat mehrere)"))
            return True
        anchor = note.anchors[j]
        i = next((k for k, m in enumerate(anchor.members) if _same(m, target)), None)
        if i is not None:
            self.remove_member(note.id, j, i)
            return True
        if target.curve_id is None and target.obj_id in self._anchor_parts(note, j):
            vp.notify(tr("Krake: das ist schon das Teil am Ende der Hinweislinie"))
            return True
        if self.add_member(note.id, j, target):
            obj = vp.scene.object_by_id(target.obj_id)
            what = tr("Gruppe „{0}“ (alles darin)").format(obj.name) if obj in vp.scene.nulls else f"„{obj.name}“"
            vp.notify(tr("{0}: {1} zur Krake – {2} Arm(e) · Shift+Klick nochmal: wieder heraus · Strg+Shift+Klick: die ganze Gruppe").format(note.title or PLACEHOLDER, what, len(anchor.members)))
        return True

    def _anchor_parts(self, note, j):
        a = note.anchors[j]
        if a.curve_id is not None:
            return set()
        oid = self.vp._anchor_object(note.id, j, a.point)
        return {oid} if oid is not None else set()

    def add_member(self, nid, j, member):
        """Attach an arm to anchor j of the note (one undo step). False: already there."""
        note = self.vp.scene.note_by_id(nid)
        if note is None or j >= len(note.anchors) or member is None:
            return False
        anchor = note.anchors[j]
        if any(_same(m, member) for m in anchor.members) or \
                (member.curve_id is not None and member.curve_id == anchor.curve_id) or \
                (member.curve_id is None and member.obj_id in self._anchor_parts(note, j)):  # the part at the anchor itself
            return False
        self.vp.scene.checkpoint()
        anchor.members.append(member)
        self.vp.ws.active_note = nid
        self.vp._changed()
        return True

    def add_selection(self, nid, j):
        """Selection to octopus: selected objects (nulls: everything inside) and selected lines as arms at anchor j – one
        undo step. End of each arm: the place of the part nearest to the anchor. -> number of new arms."""
        from . import edges
        vp = self.vp
        scene = vp.scene
        note = scene.note_by_id(nid)
        if note is None or j >= len(note.anchors):
            return 0
        anchor = note.anchors[j]
        found = []
        for oid in vp.ws.selection:
            found += _expand(scene, scene.object_by_id(oid))
        members = []
        for obj in found:
            pts = _member_points(obj, scene)
            point = None
            if pts is not None and len(pts):
                if len(pts) > ARM_SAMPLES:
                    pts = pts[:: len(pts) // ARM_SAMPLES + 1]
                point = pts[int(np.argmin(np.linalg.norm(pts - anchor.point, axis=1)))].astype(float)
            members.append(NoteMember(obj.id, None, point))
        for ref in vp.ws.edge_selection:
            curve = edges.ref_to_curve(scene, ref)
            owner = scene.owner_of(curve) if curve is not None else None
            if owner is not None:
                _, t = notes.curve_param(curve.points, anchor.point)
                members.append(NoteMember(owner.id, curve.id, notes.curve_point(curve.points, t), t))
        new = []
        for m in members:
            if not any(_same(o, m) for o in anchor.members + new) and not (
                    m.curve_id is not None and m.curve_id == anchor.curve_id):
                new.append(m)
        if not new:
            vp.notify(tr("Krake: nichts Neues in der Auswahl (Objekte oder Linien wählen, dann hinzufügen)"))
            return 0
        scene.checkpoint()
        anchor.members += new
        vp.ws.active_note, vp.ws.active_anchor = nid, (nid, j)  # its arms show
        vp.notify(tr("{0}: {1} Teil(e) zur Krake hinzugefügt ({2} Arm(e)) – Strg+Z nimmt es zurück").format(note.title or PLACEHOLDER, len(new), len(anchor.members)))
        vp._changed()
        return len(new)

    def remove_member(self, nid, j, i):
        note = self.vp.scene.note_by_id(nid)
        if note is None or j >= len(note.anchors) or i >= len(note.anchors[j].members):
            return
        self.vp.scene.checkpoint()
        del note.anchors[j].members[i]
        self.vp.notify(tr("Arm der Krake entfernt – Strg+Z holt ihn zurück"))
        self.vp._changed()

    def clear_members(self, nid, j):
        note = self.vp.scene.note_by_id(nid)
        if note is None or j >= len(note.anchors) or not note.anchors[j].members:
            return
        self.vp.scene.checkpoint()
        note.anchors[j].members = []
        self.vp._changed()

    def _clip(self, g, mask):
        """Cells of the plate (mask: rows × columns) as a path in reference pixels; edge cells extend beyond the edge
        so that the brackets on the edge are not half cut off."""
        _, (x0, y0, cw, ch) = self._cells(g)
        path = QPainterPath()
        rows, cols = mask.shape
        out = 0.25 * g["h"]
        for j in range(rows):
            for i in range(cols):
                if mask[j, i]:
                    r = QRectF(x0 + i * cw, y0 + j * ch, cw, ch).adjusted(-0.01 * cw, -0.01 * ch, 0.01 * cw, 0.01 * ch)
                    r.adjust(-out if i == 0 else 0, -out if j == 0 else 0,
                             out if i == cols - 1 else 0, out if j == rows - 1 else 0)
                    path.addRect(r)
        return path.simplified()

    def _draw_label(self, p, note, g, s, xf, color, editing, clip=None):
        """Brackets, content point and heading in the note plane (clip: only this part of the plate)."""
        w, h, pad = g["w"], g["h"], g["pad"]
        p.save()
        p.setTransform(xf, True)
        if clip is not None:
            p.setClipPath(clip)
        pen = QPen(color, LINE_W)
        pen.setCosmetic(True)
        if note.review:  # review draft (not yet adopted): dashed brackets
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        b = BRACKET * h
        p.drawPolyline(QPolygonF([QPointF(-pad, -pad + b), QPointF(-pad, -pad), QPointF(-pad + b, -pad)]))
        p.drawPolyline(QPolygonF([QPointF(w + pad, h + pad - b), QPointF(w + pad, h + pad), QPointF(w + pad - b, h + pad)]))
        if note.board.get("items"):  # note board has content: small dot at the top right
            p.setBrush(color)
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(w + pad, -pad), 0.09 * h, 0.09 * h)
        if note.id != editing:
            text_color = QColor(color)
            if not note.title:
                text_color.setAlpha(min(text_color.alpha(), PLACEHOLDER_ALPHA))
            line_px = math.hypot(*(s[3] - s[0])) * h / (h + 2 * pad)  # line height on screen
            dpr = round(self.vp.devicePixelRatioF(), 2)
            image, scale = _text_image(note.title or PLACEHOLDER, text_color, line_px, dpr)
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(QRectF(-1 / scale, -1 / scale, image.width() / scale, image.height() / scale), image)
        p.restore()

    def _draw_buttons(self, p, info, color):
        hover = self.hover
        for name, c in info["icons"].items():
            hot = hover is not None and hover[0] == name
            bg = QColor(theme.PAPER)
            bg.setAlpha(235)
            p.setBrush(bg)
            p.setPen(QPen(theme.TERRACOTTA if hot else color, 1.2))
            p.drawEllipse(c, ICON_R, ICON_R)
            q = ICON_R * 0.45
            if name == "plus":
                p.drawLine(c + QPointF(-q, 0), c + QPointF(q, 0))
                p.drawLine(c + QPointF(0, -q), c + QPointF(0, q))
            elif name == "cam":
                camera_glyph(p, c, ICON_R * 0.62, QColor(theme.TERRACOTTA) if hot else QColor(color))
            else:  # maximize: two corners
                for sx, sy in ((-1, -1), (1, 1)):
                    corner = c + QPointF(sx * q, sy * q)
                    p.drawLine(corner, corner - QPointF(sx * q * 0.9, 0))
                    p.drawLine(corner, corner - QPointF(0, sy * q * 0.9))
        if info["handle"] is not None:
            hot = hover is not None and hover[0] == "handle"
            p.setBrush(QColor(theme.PAPER))
            p.setPen(QPen(theme.TERRACOTTA if hot else color, 1.4))
            p.drawEllipse(info["handle"], HANDLE_R, HANDLE_R)

    def _draw_drag(self, p):
        d = self.drag
        if d is not None and d["kind"] == "arm" and d.get("cursor") is not None:
            self._draw_arm_drag(p, d)
            return
        if d is None or d["kind"] not in ("leader", "anchor") or d.get("cursor") is None:
            return
        info = self._screen.get(d["note"])
        if info is None:
            return
        start = info["handle"] or info["quad"][3]
        if d["kind"] == "anchor":
            note = self.vp.scene.note_by_id(d["note"])
            g = self.geometry(note)
            s, ok = self.vp.project(np.array(g["leaders"][d["index"]][:1]))
            start = QPointF(*s[0]) if ok[0] else start
        target = d.get("target")
        pen = QPen(theme.TERRACOTTA if target is not None else theme.GRAPHITE_SOFT, 1.4, Qt.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawLine(start, d["cursor"])
        if target is not None:
            s, ok = self.vp.project(target.point[None])
            if ok[0]:
                p.setBrush(theme.TERRACOTTA)
                p.drawEllipse(QPointF(*s[0]), DOT_R + 1.5, DOT_R + 1.5)

    def _draw_arm_drag(self, p, d):
        """Shift+drag from the anchor point: arm to the cursor, snaps to the part underneath."""
        note = self.vp.scene.note_by_id(d["note"])
        s, ok = self.vp.project(note.anchors[d["index"]].point[None])
        if not ok[0]:
            return
        target = d.get("target")
        color = QColor(theme.OCTOPUS)
        if target is None:
            color.setAlpha(150)
        p.setPen(QPen(color, 1.4, Qt.DashLine))
        p.setBrush(Qt.NoBrush)
        end = d["cursor"]
        if target is not None:
            q, qok = self.vp.project(target.point[None])
            end = QPointF(*q[0]) if qok[0] else end
        p.drawPath(self._arm_path(QPointF(*s[0]), end))
        if target is not None:
            p.setBrush(theme.OCTOPUS)
            p.drawEllipse(end, ARM_RING_R + 1, ARM_RING_R + 1)

        # --- Hits -----------------------------------------------------------
    def hit(self, pos):
        """(Kind, note id, index) under the cursor: plus, max, handle, member (index (anchor, arm)), anchor, note –
        or None."""
        order = sorted(self.vp.note_list(), key=lambda n: self._depth(n.pos))  # front first
        near = lambda q, r: (q - pos).manhattanLength() <= r * 1.5
        for note in order:
            info = self._screen.get(note.id)
            if info is None:
                continue
            for name, c in info["icons"].items():
                if near(c, ICON_R + 6):
                    return name, note.id, None
            if info["handle"] is not None and near(info["handle"], HANDLE_R + 3):
                return "handle", note.id, None
            for ji, end in info.get("members", ()):
                if near(end, HIT_PX):
                    return "member", note.id, ji
            for j, end in info["anchors"]:
                if near(end, HIT_PX):
                    return "anchor", note.id, j
        for note in order:
            info = self._screen.get(note.id)
            if info is not None and info["quad"].containsPoint(pos, Qt.OddEvenFill):
                return "note", note.id, None
        return None

        # --- Input (from the viewport, only in notes mode) -----------------------
    def press(self, e):
        vp = self.vp
        pos = e.position()
        if self.drag is not None and self.drag["kind"] == "rotate":
            self._end_rotate(e.button() == Qt.LeftButton)
            return True
        if self.editor is not None:  # click only ends typing (creates no new note)
            self.finish_edit()
            self.drag = None
            return True
        if vp._in_gizmo(pos):
            return False
        hit = self.hit(pos)
        if e.button() == Qt.RightButton:
            if hit is None:
                return False
            self._menu(hit, e.globalPosition().toPoint())
            return True
        if e.button() != Qt.LeftButton:
            return False
        ctrl = bool(e.modifiers() & Qt.ControlModifier)
        if hit is None:
            self.drag = {"kind": "create", "press": QPointF(pos), "mods": e.modifiers()}
        elif hit[0] == "max":
            self.drag = None
            self.open_board(hit[1])
        elif hit[0] == "cam":
            self.drag = None
            self.snapshot(hit[1])
        elif hit[0] == "handle" or (hit[0] == "note" and ctrl):
            self.drag = {"kind": "leader", "note": hit[1], "cursor": QPointF(pos), "target": None}
        elif hit[0] == "anchor" and e.modifiers() & Qt.ShiftModifier:  # Shift: drag arm of the note octopus …
            if vp.ws.active_anchor == (hit[1], hit[2]):
                self.drag = {"kind": "arm", "note": hit[1], "index": hit[2], "cursor": QPointF(pos), "target": None}
            else:  # … only from the clicked anchor (whose arms are visible): otherwise select first
                vp.ws.active_note = hit[1]
                vp.ws.active_anchor = (hit[1], hit[2])
                self.drag = None
                vp.notify(tr("Ankerpunkt gewählt – seine Krakenarme sind zu sehen; Shift+Ziehen hängt weitere an"))
        elif hit[0] == "member":
            self.drag = None  # (Del or right click removes the arm)
        elif hit[0] == "anchor":
            self.drag = {"kind": "anchor", "note": hit[1], "index": hit[2], "press": QPointF(pos), "cursor": None,
                         "target": None}
        else:
            note = vp.scene.note_by_id(hit[1])
            vp.ws.active_note = note.id
            vp.ws.active_anchor = None  # the whole note: all its parts glow
            self.drag = {"kind": "move", "note": note.id, "press": QPointF(pos), "start": note.pos.copy(),
                         "moved": False}
        return True

    def move(self, e):
        vp = self.vp
        pos = e.position()
        d = self.drag
        if d is None:
            self.hover = self.hit(pos)
            self._update_sticky(pos)
            return
        if d["kind"] == "rotate":
            note = vp.scene.note_by_id(d["note"])
            yaw = d["yaw0"] - (pos.x() - d["x0"]) * ROTATE_PER_PX
            if e.modifiers() & Qt.ControlModifier:
                yaw = round(yaw / ROTATE_SNAP) * ROTATE_SNAP
            note.yaw = yaw
            vp.notify(tr("Notiz drehen: {0:.0f}° – Klick: fertig · Esc/Rechtsklick: zurück · Strg: 15°-Schritte").format(math.degrees(yaw) % 360))
        elif d["kind"] == "move":
            if not d["moved"] and (pos - d["press"]).manhattanLength() < 4:
                return
            note = vp.scene.note_by_id(d["note"])
            if not d["moved"]:
                vp.scene.checkpoint()
                d["moved"] = True
            delta = self._screen_delta(d["start"], d["press"], pos)
            if e.modifiers() & Qt.ShiftModifier:  # Shift: vertical only
                delta = notes.UP * float(delta @ notes.UP)
            note.pos = d["start"] + delta
        elif d["kind"] == "arm":
            d["cursor"] = QPointF(pos)
            d["target"] = self._member_target(pos)
        elif d["kind"] in ("leader", "anchor"):
            if d["kind"] == "anchor" and d["cursor"] is None and (pos - d["press"]).manhattanLength() < 4:
                return
            d["cursor"] = QPointF(pos)
            d["target"] = self._anchor_target(pos)

    def release(self, e):
        vp = self.vp
        pos = e.position()
        d, self.drag = self.drag, None
        if d is None:
            return
        if d["kind"] == "rotate":
            self.drag = d  # rotating continues until the click
            return
        if d["kind"] == "create":  # click into the void: no note active any more (new note: double click)
            click = (pos - d["press"]).manhattanLength() < 4
            if click and d.get("mods", 0) & Qt.ShiftModifier and \
                    self.click_member(pos, bool(d["mods"] & Qt.ControlModifier)):
                pass  # Shift+click on a part: onto the octopus or off it
            elif click and vp.ws.active_note is not None:
                vp.ws.active_note = vp.ws.active_anchor = None
        elif d["kind"] == "leader":
            target = d.get("target") or self._anchor_target(pos)
            note = vp.scene.note_by_id(d["note"])
            if target is not None and note is not None:
                vp.scene.checkpoint()
                note.anchors.append(target)
                vp.ws.active_note = note.id
                vp.notify(tr("{0}: verbunden ({1} Verbindung(en))").format(note.title or PLACEHOLDER, len(note.anchors)))
            else:
                vp.notify(tr("Hinweislinie: auf einer Linie, einem Punkt oder einer Fläche loslassen"))
        elif d["kind"] == "arm":
            target = d.get("target") or self._member_target(pos)
            note = vp.scene.note_by_id(d["note"])
            if target is None:
                vp.notify(tr("Krake: auf einer Linie, einer Fläche oder einem Mesh loslassen"))
            elif self.add_member(d["note"], d["index"], target):
                vp.notify(tr("{0}: Arm angehängt ({1} Arm(e)) – Entf am Ring entfernt ihn").format(note.title or PLACEHOLDER, len(note.anchors[d['index']].members)))
            else:
                vp.notify(tr("Krake: dieses Teil ist schon dabei (oder ist das Teil am Anker selbst)"))
        elif d["kind"] == "anchor" and d["cursor"] is None:  # click on the anchor point: only its parts glow,
            vp.ws.active_note = d["note"]  # its arms appear – again: off again (whole note)
            pick = (d["note"], d["index"])
            vp.ws.active_anchor = None if vp.ws.active_anchor == pick else pick
        elif d["kind"] == "anchor" and d["cursor"] is not None:
            note = vp.scene.note_by_id(d["note"])
            vp.scene.checkpoint()
            if d["target"] is not None:
                note.anchors[d["index"]] = d["target"]
            else:
                del note.anchors[d["index"]]
                vp.notify(tr("Verbindung gelöst – Strg+Z holt sie zurück"))
        self.hover = self.hit(pos)

    def _update_sticky(self, pos):
        """The maximize button does not disappear immediately: it stays as long as the cursor is near the
        note or the button, and for a short time after."""
        now = time.monotonic()
        if self.hover is not None:
            self.sticky = (self.hover[1], now)
            return
        if self.sticky is None:
            return
        info = self._screen.get(self.sticky[0])
        if info is not None:
            area = info["quad"].boundingRect()
            for c in info["icons"].values():
                area = area.united(QRectF(c.x() - ICON_R, c.y() - ICON_R, 2 * ICON_R, 2 * ICON_R))
            if area.adjusted(-STICKY_PX, -STICKY_PX, STICKY_PX, STICKY_PX).contains(pos):
                self.sticky = (self.sticky[0], now)
                return
        if now - self.sticky[1] > STICKY_S:
            self.sticky = None

    def double_click(self, e):
        hit = self.hit(e.position())
        self.drag = None
        if hit is None:  # double click into the void or onto a line: new note
            self._create(e.position())
            return True
        if hit[0] != "note":
            return False
        self.edit(hit[1])
        return True

    def key(self, e):
        """Keys in notes mode; True = handled."""
        vp = self.vp
        k = e.key()
        if self.drag is not None and self.drag["kind"] == "rotate":
            if k == Qt.Key_Escape:
                self._end_rotate(False)
                return True
            if k in (Qt.Key_Return, Qt.Key_Enter):
                self._end_rotate(True)
                return True
        target = self.hover[1] if self.hover else vp.ws.active_note
        note = vp.scene.note_by_id(target) if target is not None else None
        if k in (Qt.Key_Delete, Qt.Key_Backspace) and note is not None:
            if self.hover and self.hover[0] == "member":
                self.remove_member(note.id, *self.hover[2])
            elif self.hover and self.hover[0] == "anchor":
                vp.scene.checkpoint()
                del note.anchors[self.hover[2]]
                vp.notify(tr("Verbindung gelöst – Strg+Z holt sie zurück"))
            else:
                self.delete(note.id)
            self.hover = None
            vp._changed()
            return True
        if k == Qt.Key_R and note is not None and not e.isAutoRepeat():
            self.start_rotate(note.id)
            return True
        if k == Qt.Key_F2 and note is not None:  # (not Enter: it arrives here from confirming the heading)
            self.edit(note.id)
            return True
        if k == Qt.Key_Escape:
            if vp.ws.active_anchor is not None and vp.ws.active_anchor[0] == vp.ws.active_note:
                vp.ws.active_anchor = None  # first back to the whole note
                vp._changed()
            elif vp.ws.active_note is not None:
                vp.ws.active_note = vp.ws.active_anchor = None
                vp._changed()
            elif vp.ws.timing_mode:
                return False  # shots: Esc continues (gizmo, camera)
            else:
                vp.set_notes_mode(False)
            return True
        return False

    def cancel(self):
        """Leave mode or new scene: end running drags and inputs."""
        if self.drag is not None and self.drag["kind"] == "rotate":
            self._end_rotate(True)
        self.drag = None
        self.hover = None
        self.sticky = None
        self.finish_edit()

        # --- Actions ----------------------------------------------------------
    def _screen_delta(self, start, press, pos):
        """Displacement in the image plane through start that corresponds to the mouse movement press -> pos."""
        vp = self.vp
        n = vp.cam.back
        hits = []
        for q in (press, pos):
            eye, d = vp.ray(q)
            denom = float(d @ n)
            if abs(denom) < 1e-9:
                return np.zeros(3)
            hits.append(eye + d * float((start - eye) @ n) / denom)
        return hits[1] - hits[0]

    def _anchor_target(self, pos):
        """Target of a leader line under the cursor: line (ends snap) or point on a surface. In shots: a
        stroke of the currently shown flipbook drawing or a point on a flipbook plane."""
        vp = self.vp
        if vp.ws.timing_mode:
            return vp._shot_anchor_target(pos)
        curve, point, _ = vp._front_curve(pos, exact_over_surface=True)  # the frontmost surface wins
        if curve is not None:
            _, t = notes.curve_param(curve.points, point)
            ends, ok = vp.project(curve.points[[0, -1]])
            for end, flag, value in zip(ends, ok, (0.0, 1.0)):
                if flag and math.hypot(end[0] - pos.x(), end[1] - pos.y()) <= HIT_PX + 2:
                    t = value
            return NoteAnchor(notes.curve_point(curve.points, t), curve.id, t)
        point = vp._surface_under(pos)[1]
        return NoteAnchor(np.asarray(point, float)) if point is not None else None

    def _create(self, pos):
        vp = self.vp
        anchor = self._anchor_target(pos)
        if anchor is not None and (anchor.curve_id is not None or vp.ws.timing_mode):  # shots: also on the plane
            r = notes.right_vector(Note("", anchor.point), vp.cam.right)
            wpp = self._wpp(anchor.point)
            where = anchor.point + r * NEW_OFFSET_PX[0] * wpp + notes.UP * NEW_OFFSET_PX[1] * wpp
        else:
            anchor = None
            where = self._free_spot(pos)
        vp.scene.checkpoint()
        note = Note("", where, self.last_size, anchors=[anchor] if anchor is not None else [])
        vp.note_list().append(note)
        vp.ws.active_note = note.id
        vp._changed()
        self.edit(note.id, new=True)

    def _free_spot(self, pos):
        """Note in empty space: under the cursor at the depth of the line nearest on screen
        (without lines: depth of the camera pivot). A drawing surface or mesh directly under
        the cursor wins if it is closer to the camera – then the note sits on that surface."""
        vp = self.vp
        cam = vp.cam
        view = -cam.back
        depth = float((cam.target - cam.eye) @ view)
        curves = vp.scene.pickable_curves()
        if curves:
            pts = np.vstack([c.points for c in curves])
            screen, ok = vp.project(pts)
            dist = np.hypot(screen[:, 0] - pos.x(), screen[:, 1] - pos.y())
            dist[~ok | np.isnan(dist)] = np.inf
            k = int(np.argmin(dist))
            if np.isfinite(dist[k]):
                depth = float((pts[k] - cam.eye) @ view)
        surface, hit, _ = vp._mesh_under(pos)  # drawing surfaces and meshes (no planes)
        if hit is not None and NEAR_DEPTH < float((hit - cam.eye) @ view) < depth:
            return np.asarray(hit, float)
        origin, d = vp.ray(pos)
        s = (depth - float((origin - cam.eye) @ view)) / max(float(d @ view), 1e-9)
        return origin + d * s

    def _drop_checkpoint(self):
        """Undo the step just saved, without keeping it for Ctrl+Y."""
        scene = self.vp.scene
        if scene.undo() and scene._redo:
            scene._redo.pop()

    def delete(self, nid):
        vp = self.vp
        note = vp.scene.note_by_id(nid)
        if note is None:
            return
        vp.scene.checkpoint()
        vp.note_list().remove(note)
        if vp.ws.timing_mode:  # shots: their cameras go along (they lie only on their board)
            for v in vp.ws.viewports:
                if v._look is not None and vp.scene.timing.note_of(vp.scene.timing.camera_by_id(v._look)) is note:
                    v.leave_camera()
            vp.scene.timing.cams = [c for c in vp.scene.timing.cams if c.note != note.id]
            vp.ws.timing_changed.emit()
        if vp.ws.active_note == nid:
            vp.ws.active_note = None
        vp.notify(tr("Notiz {0} gelöscht – Strg+Z holt sie zurück").format(note.title or PLACEHOLDER))
        vp._changed()

    def set_scene_scale(self, factor):
        """Note size of the scene: factor on all font sizes (stored with the scene)."""
        factor = max(0.01, min(float(factor), 10000.0))
        if abs(factor - float(getattr(self.vp.scene, "note_scale", 1.0))) < 1e-9:
            return
        self.vp.scene.note_scale = factor
        self.vp.scene.dirty = True
        self.vp.notify(tr("Notizgröße der Szene: × {0:g}").format(factor))
        self.vp._changed()

    def ask_scene_scale(self):
        from PySide6.QtWidgets import QInputDialog
        value, ok = QInputDialog.getDouble(self.vp, tr("Notizgröße der Szene"), tr("Faktor auf alle Notizen dieser Szene:"),
                                           float(getattr(self.vp.scene, "note_scale", 1.0)), 0.01, 10000.0, 2)
        if ok:
            self.set_scene_scale(value)

    def set_size(self, nid, size):
        note = self.vp.scene.note_by_id(nid)
        if note is None or note.size == size:
            return
        self.vp.scene.checkpoint()
        note.size = size
        self.last_size = size
        self.vp._changed()

    def set_follow(self, nid, follow):
        note = self.vp.scene.note_by_id(nid)
        if note is None:
            return
        self.vp.scene.checkpoint()
        note.yaw = None if follow else notes.yaw_of(notes.right_vector(note, self.vp.cam.right))
        self.vp._changed()

    def start_rotate(self, nid):
        vp = self.vp
        note = vp.scene.note_by_id(nid)
        x = vp._last_pos.x() if vp._last_pos is not None else vp.width() / 2
        vp.scene.checkpoint()
        yaw0 = notes.yaw_of(notes.right_vector(note, vp.cam.right))
        note.yaw = yaw0
        self.drag = {"kind": "rotate", "note": nid, "yaw0": yaw0, "x0": x}
        vp.ws.active_note = nid
        vp.notify(tr("Notiz drehen: Maus links/rechts · Strg: 15°-Schritte · Klick: fertig · Esc/Rechtsklick: zurück"))
        vp.update()

    def _end_rotate(self, keep):
        self.drag = None
        if not keep:
            self._drop_checkpoint()
        self.vp._changed()

    def clear_anchors(self, nid):
        note = self.vp.scene.note_by_id(nid)
        if note is None or not note.anchors:
            return
        self.vp.scene.checkpoint()
        note.anchors = []
        self.vp._changed()

    def open_board(self, nid):
        from .note_board import open_board
        self.vp.ws.active_note = nid
        open_board(self.vp, nid)

    def snapshot(self, nid):
        """Camera button: view (without controls) as image onto the board of the note – in shots a new camera."""
        from .note_board import insert_snapshot
        note = self.vp.scene.note_by_id(nid)
        if self.vp.ws.timing_mode and note is not None:
            self.vp.ws.active_note = nid
            self.vp.new_timing_camera(note)
            return
        if insert_snapshot(self.vp, nid) is not None and note is not None:
            self.vp.notify(tr("{0}: Ansicht ins Notizbrett eingefügt – Doppelklick aufs Bild fliegt hierher zurück").format(note.title or PLACEHOLDER))

    def _menu(self, hit, global_pos):
        vp = self.vp
        nid = hit[1]
        note = vp.scene.note_by_id(nid)
        menu = QMenu(vp)
        if hit[0] == "member":
            menu.addAction(tr("Aus Krake entfernen\tEntf"), lambda: self.remove_member(nid, *hit[2]))
            menu.exec(global_pos)
            return
        if hit[0] == "anchor":
            j = hit[2]
            menu.addAction(tr("Auswahl zur Krake hinzufügen"), lambda: self.add_selection(nid, j))
            menu.addAction(tr("Krake leeren"), lambda: self.clear_members(nid, j)).setEnabled(bool(note.anchors[j].members))
            menu.addSeparator()
            menu.addAction(tr("Verbindung lösen\tEntf"), lambda: self._remove_anchor(nid, j))
            menu.exec(global_pos)
            return
        vp.ws.active_note = nid
        for size, name in notes.SIZE_NAMES.items():
            a = menu.addAction(name)
            a.setCheckable(True)
            a.setChecked(note.size == size)
            a.triggered.connect(lambda _=False, s=size: self.set_size(nid, s))
        # Note size of the whole scene: factor on all font sizes (large scenes need larger notes)
        scale = float(getattr(vp.scene, "note_scale", 1.0))
        sub = menu.addMenu(tr("Notizgröße der Szene: × {0:g}").format(scale))
        for k in NOTE_SCALES:
            a = sub.addAction(f"× {k:g}")
            a.setCheckable(True)
            a.setChecked(abs(scale - k) < 1e-6)
            a.triggered.connect(lambda _=False, k=k: self.set_scene_scale(k))
        sub.addSeparator()
        sub.addAction(tr("Eigener Wert …"), self.ask_scene_scale)
        menu.addSeparator()
        follow = menu.addAction(tr("Mit der Kamera drehen"))
        follow.setCheckable(True)
        follow.setChecked(note.yaw is None)
        follow.toggled.connect(lambda on: self.set_follow(nid, on))
        menu.addAction(tr("Drehen\tR"), lambda: self.start_rotate(nid))
        menu.addSeparator()
        menu.addAction(tr("Überschrift bearbeiten\tDoppelklick"), lambda: self.edit(nid))
        menu.addAction(tr("Notizbrett öffnen"), lambda: self.open_board(nid))
        menu.addAction(tr("Neue Kamera (Screenshot ins Brett)") if vp.ws.timing_mode else tr("Ansicht ins Notizbrett (Bild)"),
                       lambda: self.snapshot(nid))
        menu.addAction(tr("Verbindungen lösen"), lambda: self.clear_anchors(nid)).setEnabled(bool(note.anchors))
        if len(note.anchors) == 1:  # one anchor: the octopus right here (with several: right click on the anchor point)
            menu.addAction(tr("Auswahl zur Krake hinzufügen"), lambda: self.add_selection(nid, 0))
        menu.addSeparator()
        menu.addAction(tr("Löschen\tEntf"), lambda: self.delete(nid))
        menu.exec(global_pos)
        vp.update()

    def _remove_anchor(self, nid, index):
        note = self.vp.scene.note_by_id(nid)
        if note is None or index >= len(note.anchors):
            return
        self.vp.scene.checkpoint()
        del note.anchors[index]
        self.vp._changed()

        # --- Typing the heading --------------------------------------------------
    def edit(self, nid, new=False):
        vp = self.vp
        note = vp.scene.note_by_id(nid)
        if note is None:
            return
        self.finish_edit()
        g = self.geometry(note)
        s, ok = vp.project(np.array([g["world"](0, 0), g["world"](0, g["h"])]))
        px = abs(s[1][1] - s[0][1]) if ok.all() else 24
        px = min(max(px * 0.8, 14), 48)
        ed = QLineEdit(vp)
        if vp.testAttribute(Qt.WA_NativeWindow):
            ed.setAttribute(Qt.WA_NativeWindow, True)
        ed.setFrame(False)
        ed.setFont(note_font(px))
        ed.setText(note.title)
        ed.setPlaceholderText(tr("Überschrift"))
        paper, ink = QColor(theme.PAPER), QColor(theme.GRAPHITE)
        ed.setStyleSheet(f"QLineEdit {{ background: rgba({paper.red()}, {paper.green()}, {paper.blue()}, 225); "
                         f"color: rgb({ink.red()}, {ink.green()}, {ink.blue()}); border: none; "
                         f"border-bottom: 1px solid rgb({ink.red()}, {ink.green()}, {ink.blue()}); padding: 0 4px; }}")
        width = max(220, int(QFontMetricsF(ed.font()).horizontalAdvance(note.title or tr("Überschrift")) + 40))
        x, y = (s[0] if ok.all() else (vp.width() / 2, vp.height() / 2))
        ed.setGeometry(int(x) - 4, int(y), width, int(px * 1.5))
        ed.returnPressed.connect(self.finish_edit)
        ed.installEventFilter(_EscapeFilter(ed, self.cancel_edit, self._focus_lost))
        self.editor = (ed, nid, new, note.title)
        ed.show()
        ed.raise_()
        ed.setFocus()
        ed.selectAll()
        vp.notify(tr("Überschrift tippen – Enter: fertig · Esc: abbrechen"))
        vp.update()

    def finish_edit(self, cancel=False):
        if self.editor is None:
            return
        ed, nid, new, old = self.editor
        self.editor = None
        vp = self.vp
        text = ed.text().strip()
        ed.hide()
        ed.deleteLater()
        note = vp.scene.note_by_id(nid)
        if note is not None:
            if new and cancel and not text:
                self._drop_checkpoint()  # Esc on an empty new note: gone again
                vp.ws.active_note = None
            elif not cancel and text != old:
                if not new:
                    vp.scene.checkpoint()
                note.title = text
        vp.setFocus()
        vp._changed()

    def _focus_lost(self, reason):
        """Focus lost (click elsewhere): end typing. If it was a mouse click into the viewport, this click
        only ends it – it creates no new note."""
        if self.editor is None:
            return
        self.dismiss_click = reason == Qt.MouseFocusReason
        self.finish_edit()

    def cancel_edit(self):
        self.finish_edit(cancel=True)


def _same(a, b):
    return a.obj_id == b.obj_id and a.curve_id == b.curve_id


def _member_points(obj, scene=None):
    """Points of a part of the octopus in the world (mesh: corners, surface: net, stroke object: lines, null: everything inside)
    or None."""
    if scene is not None and obj in scene.nulls:
        pts = [q for c in scene.descendants(obj) if not isinstance(c, type(obj))
               for q in [_member_points(c)] if q is not None and len(q)]
        return np.vstack(pts) if pts else None
    if hasattr(obj, "geo"):
        if obj.missing or not len(obj.geo.vertices):
            return None
        return obj.geo.vertices.astype(float) @ obj.xform[:3, :3].T + obj.xform[:3, 3]
    if hasattr(obj, "fill_mesh"):
        mesh = obj.fill_mesh()
        return None if mesh is None else np.asarray(mesh[0], float)
    if hasattr(obj, "curves"):
        pts = [c.points for c in obj.curves if len(c.points)]
        return np.vstack(pts) if pts else None
    return None


def _expand(scene, obj):
    """Selection -> parts of the octopus: meshes, surfaces (no helper surfaces), stroke objects and nulls (one arm for the
    whole group, meaning everything inside)."""
    if obj is None or getattr(obj, "aid", False):
        return []
    if hasattr(obj, "geo") or hasattr(obj, "fill_mesh") or hasattr(obj, "curves") or obj in scene.nulls:
        return [obj]
    return []  # planes


class _EscapeFilter(QObject):
    """Esc in the heading: cancel (QLineEdit does not know this by itself)."""

    def __init__(self, parent, callback, focus_lost):
        super().__init__(parent)
        self._callback = callback
        self._focus_lost = focus_lost

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
            self._callback()
            return True
        if event.type() == QEvent.FocusOut:
            self._focus_lost(event.reason())
        return False
