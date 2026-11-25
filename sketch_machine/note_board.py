"""Note board: the 2D surface behind a note – text blocks and images freely arranged, as in PureRef.

Lies semi-transparent over the viewports (the sketch stays visible) or as its own window (e.g. on
the second monitor). Ctrl+V pastes images and text, files can be dragged in. Every change lands
in the note immediately (one undo step in the shared history, Ctrl+Z/Ctrl+Y also in the board); images live
in the scene (scene.assets) and are saved with it in the same file (like PureRef).
"""
import base64
import copy
import hashlib
import json
import math
import time
import os
import re
import urllib.request

import numpy as np
from PySide6.QtCore import QBuffer, QByteArray, QEvent, QIODevice, QMimeData, QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPainterPath, QPainterPathStroker, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (QApplication, QFrame, QGraphicsItem, QGraphicsPixmapItem, QGraphicsProxyWidget,
                               QGraphicsScene, QGraphicsTextItem, QGraphicsView, QHBoxLayout, QLabel, QLineEdit, QMenu,
                               QSlider, QToolButton, QVBoxLayout, QWidget)

from . import board_links, shapes, timing, ui, video
from .note_tool import camera_glyph, note_font
from .scene import unique_name
from .vp.common import HOLD_MS, HOLD_PX
from .i18n import tr

INK = QColor(ui.TEXT)  # notes window in the interface style (anthracite), independent of the viewport mode
INK_DIM = QColor(ui.TEXT_DIM)
WINDOW_BG = QColor(ui.PANEL)

MARGIN = 28  # distance of the overlay to the edge of the viewports
BG_ALPHA = 242  # opacity of the overlay (the sketch shows through slightly)
TEXT_WIDTH = 340  # width of new text blocks
TEXT_PX = 17
IMAGE_MAX = 520  # new images at most this wide/high (the original stays stored)
ZOOM_STEP = 1.15
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
POSTER_MS = 1000  # preview image of a video: at this position (shorter videos: after a tenth – frame 0 is often black)
SNAP_MAX = 1600  # screenshots from the viewport at most this large (pixels), as JPEG
SNAP_QUALITY = 90
HANDLE_PX = 7  # handles at the corners of selected images/texts (screen pixels): dragging scales
REDO_PX = 11  # ↻ at the top right of a screenshot (radius, screen pixels): retake
HERO_LABEL = "hero camera"  # checkbox left of ↻: screenshot is a hero camera for the sketch brief
BADGE_SHARE = 0.25  # ↻ and checkbox together at most this wide (fraction of the image width on screen)
CAPTION_SHARE = 0.035  # font size of the image caption as a fraction of the image width (scales with the image)
HERO_FILL, HERO_LINE, HERO_TEXT = QColor(186, 238, 214), QColor(62, 146, 126), QColor(28, 64, 54)  # like helper planes
INK_STEP_PX = 1.0  # new stroke point from this distance (screen pixels)
INK_CORNER = math.radians(50)  # stronger bend between two stroke pieces: real corner, is not rounded
ZOOM_DRAG = 0.006  # Alt + right mouse button: zoom per pixel (as in the 3D view)
ZOOM_RANGE = (0.02, 40.0)
WEB_TIMEOUT = 8  # seconds: load an image from a web address (the clipboard only had the link)
BACKDROP_TITLE_PX = 22
BACKDROP_MIN = (120.0, 80.0)  # smallest size when dragging at the corners
BACKDROP_PAD = 36.0  # margin around the selection when creating (extra room at the top for the title)
BACKDROP_SHADE = (34, 62)  # darkening (alpha of black) of the surface and the title bar
AUTO_PAD = 16.0  # margin of the auto backdrops: with title bar below the distance of the groups (board_links.SECTION_GAP),
#                  otherwise neighbors overlap
BOARD_PEN_DEFAULT = 4.0  # brush width in the board in board units (independent of zoom), Ctrl+Shift scales
BOARD_PEN_RANGE = (0.3, 200.0)
_board_pen = None  # remembered width (QSettings board_pen)


def board_pen():
    global _board_pen
    if _board_pen is None:
        from PySide6.QtCore import QSettings
        _board_pen = QSettings("SketchMachine", "SketchMachine").value("board_pen", BOARD_PEN_DEFAULT, type=float)
    return _board_pen


def set_board_pen(width):
    global _board_pen
    from PySide6.QtCore import QSettings
    _board_pen = min(max(float(width), BOARD_PEN_RANGE[0]), BOARD_PEN_RANGE[1])
    QSettings("SketchMachine", "SketchMachine").setValue("board_pen", _board_pen)
    return _board_pen


SETTLE_MS = 500  # painting: this long after the last stroke without a new one, only then into the note (undo step) and show the connection
LINK_SHOW_MS = 2500  # this long the board shows after painting what the stroke connects (selected: permanent)
LINK_COLOR = QColor(96, 196, 206)
# pen colors of the board (None = ink: follows the interface); plus a custom color via the color wheel
BOARD_COLORS = ((None, tr("Tinte")), ((0.80, 0.22, 0.20), tr("Rot")), ((0.20, 0.42, 0.80), tr("Blau")),
                ((0.24, 0.58, 0.30), tr("Grün")), ((0.92, 0.56, 0.16), "Orange"))


def _rgba(color, alpha=None):
    c = QColor(color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {c.alpha() if alpha is None else alpha})"


def _on_scene(item, gscene):
    """Is the element (still) on this board? (After clear() the C++ object is already deleted.)"""
    try:
        return item.scene() is gscene
    except RuntimeError:
        return False


class _HeroNameEdit(QLineEdit):
    """Name field of a hero camera in the board. Also closes on clicking away without a change – Qt 6 does not report
    editingFinished then, and the field would otherwise stay open (after Ctrl+Z even without its image)."""

    def __init__(self, text, parent, done):
        super().__init__(text, parent)
        self._done = done

    def focusOutEvent(self, e):
        super().focusOutEvent(e)
        if e.reason() != Qt.PopupFocusReason:  # (context menu of the field: stays open)
            self._done()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.setText("")  # Esc: name stays as it was
            return self._done()
        super().keyPressEvent(e)


class TextBlock(QGraphicsTextItem):
    def __init__(self, text=""):
        super().__init__(text)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemIsFocusable)
        self.setTextWidth(TEXT_WIDTH)
        self.setFont(note_font(TEXT_PX))
        self.setDefaultTextColor(INK)

    def start_editing(self):
        self.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.setFocus()

    def mouseDoubleClickEvent(self, e):
        self.start_editing()
        super().mouseDoubleClickEvent(e)

    def focusOutEvent(self, e):
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        super().focusOutEvent(e)
        for view in self.scene().views() if self.scene() is not None else ():
            view.board.commit()  # finished writing: one undo step

    def paint(self, p, option, widget=None):
        super().paint(p, option, widget)
        r = self.boundingRect()
        b = 10
        pen = QPen(QColor(ui.ACCENT) if self.isSelected() else INK_DIM, 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawPolyline([QPointF(r.left(), r.top() + b), r.topLeft(), QPointF(r.left() + b, r.top())])
        p.drawPolyline([QPointF(r.right(), r.bottom() - b), r.bottomRight(), QPointF(r.right() - b, r.bottom())])


def camera_icon(color, size=20):
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    camera_glyph(p, QPointF(size / 2, size / 2), size * 0.42, QColor(color))
    p.end()
    return QIcon(pix)


def backdrop_icon(color, size=20):
    """Backdrop button: flat tab (title bar) on a taller frame."""
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 1.3)
    p.setPen(pen)
    w, x = size * 0.72, size * 0.14
    tab = QRectF(x, size * 0.2, w, size * 0.16)
    body = QRectF(x, tab.bottom() + size * 0.06, w, size * 0.44)
    fill = QColor(color)
    fill.setAlpha(110)
    p.setBrush(fill)
    p.drawRoundedRect(tab, 1.5, 1.5)
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(body, 1.5, 1.5)
    p.end()
    return QIcon(pix)


def _dot_icon(color, current, size=18):
    """Color button: filled dot (color None: color circle for "custom color"), current: with ring."""
    pix = QPixmap(size * 2, size * 2)
    pix.setDevicePixelRatio(2.0)
    pix.fill(QColor(0, 0, 0, 0))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    r = QRectF(3.5, 3.5, size - 7, size - 7)
    if color is None:
        from PySide6.QtGui import QConicalGradient
        g = QConicalGradient(r.center(), 90)
        for k, hue in enumerate((0, 60, 120, 180, 240, 300, 360)):
            g.setColorAt(k / 6, QColor.fromHsv(hue % 360, 170, 230))
        p.setBrush(g)
    else:
        p.setBrush(color)
    p.setPen(QPen(QColor(ui.LINE), 1))
    p.drawEllipse(r)
    if current:
        p.setPen(QPen(QColor(ui.ACCENT), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(r.adjusted(-2.5, -2.5, 2.5, 2.5))
    p.end()
    return QIcon(pix)


def camera_view(vp):
    """Camera of the view as dict (stored on the image: double click flies back, the brief names it)."""
    target, dist, rot = vp.cam.state()
    return {"target": [round(float(x), 5) for x in target], "dist": round(float(dist), 5),
            "rot": [[round(float(x), 6) for x in row] for row in rot], "ortho": bool(vp.cam.ortho)}


def retake(vp, view, aspect=None, look=None):
    """JPEG bytes of the view from the camera view (without controls) – the own camera stays as it is.
    aspect: aspect ratio of the old image (strokes on it fit again); X-ray, wireframe and light as currently
    set (none of this is remembered on the image). look: different rendering (see Viewport.clean_image)."""
    import numpy as np
    saved, ortho = vp.cam.state(), vp.cam.ortho
    try:
        vp.cam.set_state(np.array(view["target"], float), float(view["dist"]), np.array(view["rot"], float))
        vp.cam.ortho = bool(view.get("ortho"))
        ws = vp.ws
        look = dict({"xray": ws.xray_opacity if ws.xray else None, "wire": ws.wire_opacity if ws.wireframe else None},
                    **(look or {}))
        return snapshot(vp, aspect, look)[0]
    finally:
        vp.cam.set_state(*saved)
        vp.cam.ortho = ortho
        vp.update_view()


def snapshot(vp, aspect=None, look=None):
    """(JPEG bytes, camera) of the view without controls (aspect, look: see Viewport.clean_image)."""
    img = vp.clean_image(aspect=aspect, look=look)
    longest = max(img.width(), img.height())
    if longest > SNAP_MAX:
        img = img.scaled(img.width() * SNAP_MAX // longest, img.height() * SNAP_MAX // longest,
                         Qt.KeepAspectRatio, Qt.SmoothTransformation)
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    img.convertToFormat(QImage.Format_RGB32).save(buf, "JPG", SNAP_QUALITY)
    return bytes(buf.data()), camera_view(vp)


def fly_to_view(vp, view):
    """Fly the view back into the perspective of a screenshot."""
    import numpy as np
    vp.cam.ortho = False
    vp._flight_ortho = True if view.get("ortho") else None
    vp.fly_to(np.array(view["target"], float), float(view["dist"]), np.array(view["rot"], float))


class ImageItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, file, view=None, missing=False, hero=False, hero_name=""):
        super().__init__(pixmap)
        self.file = file
        self.view = view  # screenshot from the viewport: camera (double click flies there)
        # Hero camera (checkbox at the top right): key view for the sketch brief – own image and USD camera
        self.hero, self.hero_name = bool(hero and view), hero_name
        self.hero_prompt = ""  # image prompt that the artist approved (MCP set_hero_prompt)
        self.caption, self._caption_rect = "", QRectF()  # image caption: what the image stands for, what matters
        self.caption_parts = []  # object ids of the parts the image belongs to (confirmed with the caption)
        self.missing = missing  # image not found (placeholder) – screenshots can be retaken
        self.shot = None  # shots: {"cam", "pose"} – the image is a pose of this camera (settings below)
        self.setZValue(-1)  # images always lie below strokes and text (drawing happens on top)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.setTransformationMode(Qt.SmoothTransformation)
        self.setTransformOriginPoint(self.image_rect().center())

    def image_rect(self):
        """The image itself (without caption) – frame, arrow heads (uv) and pivot refer to it."""
        return QRectF(self.offset(), self.pixmap().deviceIndependentSize())

    def boundingRect(self):
        return super().boundingRect().united(self._caption_rect) if self.caption else super().boundingRect()

    def _caption_font(self):
        return note_font(max(self.image_rect().width() * CAPTION_SHARE, 6))

    def set_caption(self, text):
        """Set caption (empty: none) – it appears as a gray line under the image, scaled with the image."""
        from PySide6.QtGui import QFontMetricsF
        self.prepareGeometryChange()
        self.caption = (text or "").strip()
        r = self.image_rect()
        fm = QFontMetricsF(self._caption_font())
        text_h = fm.boundingRect(QRectF(0, 0, r.width(), 1e6), Qt.TextWordWrap, self.caption).height()
        self._caption_rect = QRectF(r.left(), r.bottom(), r.width(), fm.height() * 0.4 + text_h)
        self.update()

    def setPixmap(self, pixmap):
        super().setPixmap(pixmap)
        if self.caption:  # new image size: set caption anew
            self.set_caption(self.caption)

    def paint(self, p, option, widget=None):
        super().paint(p, option, widget)
        if self.caption:
            p.setFont(self._caption_font())
            p.setPen(INK_DIM)
            p.drawText(self._caption_rect.adjusted(0, p.fontMetrics().height() * 0.4, 0, 0),
                       Qt.TextWordWrap | Qt.AlignLeft | Qt.AlignTop, self.caption)
        if self.hero:  # hero camera: turquoise frame (the checkbox itself only on mouse or selection)
            pen = QPen(HERO_LINE, 3)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(self.image_rect())
        if self.isSelected():
            pen = QPen(QColor(ui.ACCENT), 1.5)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(self.image_rect())


class VideoItem(ImageItem):
    """Reference video on the board: an image (preview in scene.assets, counts for arrows, captions, sections and the
    brief like any image) with playback. The video's frames only replace the display during playback, at the same
    pixel size – places on the image (uv of the arrows) stay valid. The board controls playback (NoteBoard.toggle_video)."""
    BAR_SHARE, BUTTON_SHARE = 0.045, 0.11  # timeline and button as a fraction of the shorter image side

    def __init__(self, pixmap, file, path, duration, missing=False):
        super().__init__(pixmap, file, None, missing)
        self.path, self.duration = path, int(duration)
        self.position, self.playing = 0, False

    @property
    def absent(self):
        return not os.path.isfile(self.path)

    def _bar(self):
        r = self.image_rect()
        h = min(r.width(), r.height()) * self.BAR_SHARE
        return QRectF(r.left(), r.bottom() - h, r.width(), h)

    def _button(self):
        r = self.image_rect()
        return r.center(), min(r.width(), r.height()) * self.BUTTON_SHARE

    def control_at(self, local):
        """"bar" (timeline), "button" (play/pause) or None at a place of the element."""
        if self.absent:
            return None
        if self._bar().contains(local):
            return "bar"
        center, radius = self._button()
        return "button" if math.hypot(local.x() - center.x(), local.y() - center.y()) <= radius else None

    def fraction_at(self, local):
        bar = self._bar()
        return min(max((local.x() - bar.left()) / max(bar.width(), 1e-6), 0.0), 1.0)

    def show_frame(self, image, position):
        """Show a frame of the video (size of the element stays)."""
        self.position = position
        size = self.pixmap().size()
        self.setPixmap(QPixmap.fromImage(image.scaled(size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)))

    def paint(self, p, option, widget=None):
        super().paint(p, option, widget)
        bar = self._bar()
        p.setPen(Qt.NoPen)
        p.fillRect(bar, QColor(0, 0, 0, 150))
        if self.duration > 0:
            done = QRectF(bar)
            done.setWidth(bar.width() * min(self.position / self.duration, 1.0))
            p.fillRect(done, QColor(ui.ACCENT))
        font = note_font(max(bar.height() * 0.7, 4))
        p.setFont(font)
        p.setPen(QColor(255, 255, 255))
        p.drawText(bar.adjusted(bar.height() * 0.4, 0, -bar.height() * 0.4, 0), Qt.AlignRight | Qt.AlignVCenter,
                   f"{video.time_label(self.position)} / {video.time_label(self.duration)}")
        if self.absent:
            p.drawText(self.image_rect(), Qt.AlignCenter, tr("Video fehlt"))
        elif not self.playing:
            center, radius = self._button()
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawEllipse(center, radius, radius)
            p.setBrush(QColor(255, 255, 255, 235))
            k = radius * 0.5
            p.drawPolygon([center + QPointF(-k * 0.7, -k), center + QPointF(-k * 0.7, k), center + QPointF(k, 0)])


def _ink_piece(path, a, b, c):
    """Piece from the middle a-b to the middle b-c: arc around b, at a real corner (rectangle) sharp through b."""
    end = (b + c) / 2
    u, v = b - a, c - b
    cross = u.x() * v.y() - u.y() * v.x()
    dot = u.x() * v.x() + u.y() * v.y()
    if abs(math.atan2(cross, dot)) > INK_CORNER:
        path.lineTo(b)
        path.lineTo(end)
    else:
        path.quadTo(b, end)


def _ink_paths(pts, widths):
    """Stroke as smoothed paths [(QPainterPath, width)]; equal widths in one piece."""
    out = []
    n = len(pts)

    def piece(k):
        path = QPainterPath(pts[0] if k == 0 else (pts[k - 1] + pts[k]) / 2)
        if k == 0:
            path.lineTo((pts[0] + pts[1]) / 2)
        elif k == n - 1:
            path.lineTo(pts[k])
        else:
            _ink_piece(path, pts[k - 1], pts[k], pts[k + 1])
        return path
    for k in range(n):
        path, w = piece(k), widths[k]
        if out and abs(out[-1][1] - w) < 1e-3:
            out[-1][0].connectPath(path)
        else:
            out.append((path, w))
    return out


class StrokeItem(QGraphicsItem):
    """Painted stroke in the board (without color in the text color of the board, otherwise in its color), width per
    point. style: "pen" (older boards also "brush" – drawn the same)."""

    def __init__(self, pts, widths, style="pen", color=None):
        super().__init__()
        self.pts, self.widths = [QPointF(p) for p in pts], [float(w) for w in widths]
        self.style, self.color = style, color  # color: (r, g, b) 0..1 or None (ink)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.live = False  # is being painted right now: the view shows it (BoardView._ink), the element stays hidden
        self._rect = QRectF()
        self._paths = None
        self._update_rect()

    def begin_live(self):
        self.live = True
        self.setVisible(False)

    def end_live(self):
        self.live = False
        self.prepareGeometryChange()
        self._update_rect()
        self._paths = None
        self.setVisible(True)

    def add(self, p, w):
        self.pts.append(QPointF(p))
        self.widths.append(float(w))
        if self.live:  # bounding rect only at the end: walking the whole list per point would be quadratic
            return
        self.prepareGeometryChange()
        self._update_rect()
        self._paths = None

    def set_points(self, pts, widths):
        self.prepareGeometryChange()
        self.pts, self.widths = [QPointF(p) for p in pts], [float(w) for w in widths]
        self._update_rect()
        self._paths = None
        self.update()

    def _update_rect(self):
        if not self.pts:
            self._rect = QRectF()
            return
        xs, ys = [p.x() for p in self.pts], [p.y() for p in self.pts]
        m = max(self.widths) if self.widths else 1.0
        self._rect = QRectF(min(xs) - m, min(ys) - m, max(xs) - min(xs) + 2 * m, max(ys) - min(ys) + 2 * m)

    def boundingRect(self):
        return self._rect

    def shape(self):
        path = QPainterPath()
        if self.pts:
            path.moveTo(self.pts[0])
            for p in self.pts[1:]:
                path.lineTo(p)
        stroker = QPainterPathStroker()
        stroker.setWidth(max(max(self.widths, default=1.0), 6.0))
        stroker.setCapStyle(Qt.RoundCap)
        return stroker.createStroke(path)

    def qcolor(self):
        return QColor.fromRgbF(*self.color) if self.color is not None else QColor(INK)

    def paint(self, p, option, widget=None):
        if not self.pts:
            return
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor(ui.ACCENT) if self.isSelected() else self.qcolor()
        if len(self.pts) == 1:
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawEllipse(self.pts[0], self.widths[0] / 2, self.widths[0] / 2)
            return
        p.setBrush(Qt.NoBrush)
        if self._paths is None:
            self._paths = _ink_paths(self.pts, self.widths)
        for path, w in self._paths:
            p.setPen(QPen(color, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)

    def data(self):
        """Stored in board coordinates (moved: the position is included)."""
        o = self.pos()
        entry = {"kind": "stroke", "style": self.style,
                 "pts": [[round(p.x() + o.x(), 1), round(p.y() + o.y(), 1)] for p in self.pts],
                 "w": [round(w, 2) for w in self.widths]}
        if self.color is not None:
            entry["color"] = [round(float(c), 4) for c in self.color]
        return entry

    def erase(self, center, radius):
        """Remove points in the circle. Returns the remaining pieces [(points, widths)] (empty: gone entirely) or None if
        the circle does not hit the stroke."""
        o = self.pos()
        # densify between points far apart: the eraser also hits between two points
        step = max(radius * 0.5, 1e-3)
        pts, widths = [self.pts[0]], [self.widths[0]]
        for a, b, wa, wb in zip(self.pts, self.pts[1:], self.widths, self.widths[1:]):
            n = int(((b.x() - a.x()) ** 2 + (b.y() - a.y()) ** 2) ** 0.5 / step)
            for k in range(1, n + 1):
                t = k / (n + 1)
                pts.append(a + (b - a) * t)
                widths.append(wa + (wb - wa) * t)
            pts.append(b)
            widths.append(wb)
        hit = [(p.x() + o.x() - center.x()) ** 2 + (p.y() + o.y() - center.y()) ** 2 <= radius ** 2 for p in pts]
        if not any(hit):
            return None
        runs, cur = [], []
        for p, w, h in zip(pts, widths, hit):
            if h:
                if len(cur) >= 2:
                    runs.append(cur)
                cur = []
            else:
                cur.append((QPointF(p + o), w))
        if len(cur) >= 2:
            runs.append(cur)
        return [([p for p, _ in r], [w for _, w in r]) for r in runs]


ARROW_HEAD = 14.0  # length of the arrow head (board units)
ARROW_REACH = 30.0  # the arrow start must be this close to a text on loading to follow it


def _arrow_start(rect, tip):
    """Start of an arrow from the text (rectangle in the board) toward tip: just next to the text, on the side facing the target."""
    sx = rect.left() - 8 if tip.x() < rect.left() else rect.right() + 8 if tip.x() > rect.right() else tip.x()
    sy = min(max(tip.y(), rect.top() + 12), rect.bottom() - 12) if sx != tip.x() else (
        rect.top() - 8 if tip.y() < rect.top() else rect.bottom() + 8)
    return QPointF(sx, sy)


class ArrowItem(StrokeItem):
    """Arrow set by Mule (dictated text, review): from the text to a place (uv) in an image. It follows
    both when they are moved or scaled (follow). Stored as a stroke (pts = shaft, tip at the end)
    with "arrow": {to: image file, uv}; on loading it finds text and image again via position (_bind)."""

    def __init__(self, pts, widths, to, uv, color=None):
        super().__init__(pts, widths, "pen", color)
        self.to, self.uv = to, (float(uv[0]), float(uv[1]))
        self.src = self.dst = None  # TextBlock, ImageItem
        self.setFlag(QGraphicsItem.ItemIsMovable, False)  # fixed to text and image

    def _bind(self, gscene):
        """Find the text at the start and the image at the tip (the nearest image of this file)."""
        a, b = self.pts[0] + self.pos(), self.pts[-1] + self.pos()
        texts = [(_rect_gap(t.sceneBoundingRect(), a), t) for t in gscene.items() if isinstance(t, TextBlock)]
        pics = [(_rect_gap(i.mapRectToScene(i.image_rect()), b), i) for i in gscene.items()
                if isinstance(i, ImageItem) and i.file == self.to]
        d, t = min(texts, key=lambda e: e[0], default=(None, None))
        self.src = t if d is not None and d <= ARROW_REACH else None
        self.dst = min(pics, key=lambda e: e[0])[1] if pics else None
        self.setFlag(QGraphicsItem.ItemIsMovable, self.dst is None)  # image gone: an ordinary stroke

    def follow(self):
        """Re-lay the shaft from text to image place. -> True if it moved."""
        dst = self.dst
        if dst is None or dst.scene() is not self.scene():
            return False
        ir = dst.mapRectToScene(dst.image_rect())
        tip = QPointF(ir.left() + ir.width() * self.uv[0], ir.top() + ir.height() * self.uv[1])
        src = self.src if self.src is not None and self.src.scene() is self.scene() else None
        start = _arrow_start(src.sceneBoundingRect(), tip) if src is not None else self.pts[0] + self.pos()
        o = self.pos()
        if all(abs(p.x() - q.x()) < 0.05 and abs(p.y() - q.y()) < 0.05
               for p, q in ((self.pts[0] + o, start), (self.pts[-1] + o, tip))):
            return False
        self.setPos(0, 0)
        self.set_points([start, tip], [self.widths[0]] * 2)
        return True

    def _update_rect(self):
        super()._update_rect()
        if self.pts:
            self._rect = self._rect.adjusted(-ARROW_HEAD, -ARROW_HEAD, ARROW_HEAD, ARROW_HEAD)

    def _head(self):
        a, b = self.pts[0], self.pts[-1]
        ang = math.atan2(b.y() - a.y(), b.x() - a.x())
        return [QPointF(b.x() + ARROW_HEAD * math.cos(ang + math.pi - s * 0.45),
                        b.y() + ARROW_HEAD * math.sin(ang + math.pi - s * 0.45)) for s in (-1, 1)]

    def paint(self, p, option, widget=None):
        super().paint(p, option, widget)
        if len(self.pts) < 2:
            return
        color = QColor(ui.ACCENT) if self.isSelected() else self.qcolor()
        p.setPen(QPen(color, self.widths[-1], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        for end in self._head():
            p.drawLine(self.pts[-1], end)

    def data(self):
        entry = super().data()
        entry["arrow"] = {"to": self.to, "uv": [round(self.uv[0], 4), round(self.uv[1], 4)]}
        return entry


def _rect_gap(r, p):
    """Distance of a point to a rectangle (0 inside)."""
    dx = max(r.left() - p.x(), 0.0, p.x() - r.right())
    dy = max(r.top() - p.y(), 0.0, p.y() - r.bottom())
    return math.hypot(dx, dy)


def follow_arrows(gscene):
    """Lay all Mule arrows in the board onto their text and their image place. -> number of those moved."""
    return sum(1 for i in gscene.items() if isinstance(i, ArrowItem) and i.follow())


class BackdropTitle(QGraphicsTextItem):
    """Title of a backdrop: double click writes, focus away = done (one undo step)."""

    def __init__(self, text, parent):
        super().__init__(text, parent)
        self.setFont(note_font(BACKDROP_TITLE_PX))
        self.setDefaultTextColor(INK)
        self.setPos(10, 4)

    def start_editing(self, select_all=False):
        self.setTextInteractionFlags(Qt.TextEditorInteraction)
        self.setFocus()
        if select_all:
            cursor = self.textCursor()
            cursor.select(cursor.SelectionType.Document)
            self.setTextCursor(cursor)

    def mouseDoubleClickEvent(self, e):
        self.start_editing()
        super().mouseDoubleClickEvent(e)

    def focusOutEvent(self, e):
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        super().focusOutEvent(e)
        for view in self.scene().views() if self.scene() is not None else ():
            view.board.commit()


def _top_level(item):
    """Element of the board (not the title of a backdrop, not a camera bar under a shot). Not via parentItem():
    in PySide 6.12 the call deletes an element without a parent from the scene (backdrop content vanished on
    clicking)."""
    return not isinstance(item, (BackdropTitle, QGraphicsProxyWidget))


class ShotRow(QFrame):
    """Camera settings under a shot screenshot (shots): color, name, frame of the pose, look through,
    ⟲ change pose, ➜ new pose from the current frame, focal length of this pose. Sits as a proxy on the image
    (stays the same size when zooming). Everything that reloads the board runs delayed (the bar belongs to the board)."""

    def __init__(self, board, item):
        super().__init__()
        self.board, self.item = board, item
        self.setObjectName("shotrow")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        top = QHBoxLayout()
        top.setSpacing(2)
        self.color = QToolButton()
        self.color.setToolTip(tr("Farbe der Kamera (Kategorien) – neue Kameras haben immer die Standardfarbe"))
        self.color.clicked.connect(self._color_menu)
        top.addWidget(self.color)
        self.name = QToolButton()
        self.name.setToolTip(tr("Name der Kamera – klicken zum Umbenennen"))
        self.name.clicked.connect(self._rename)
        self.name.setObjectName("camname")
        self.name.setToolButtonStyle(Qt.ToolButtonTextOnly)
        top.addWidget(self.name)
        top.addStretch(1)
        lay.addLayout(top)  # name above everything, frames and buttons below
        top = QHBoxLayout()
        top.setSpacing(2)
        self.frame = QToolButton()
        self.frame.setToolTip(tr("Frames dieser Stellung (bis zur nächsten) – klicken: zum ersten springen, Rechtsklick: ändern"))
        self.frame.clicked.connect(lambda: self._later(lambda vp, cam, pose: vp.set_frame(pose.frame)))
        self.frame.setContextMenuPolicy(Qt.CustomContextMenu)
        self.frame.customContextMenuRequested.connect(lambda _p: self._set_frame())
        top.addWidget(self.frame)
        top.addStretch(1)
        self.buttons = {}
        for key, text, tip in (("look", "👁", tr("Durch diese Kamera schauen (bei dieser Stellung) – Esc verlässt sie")),
                               ("set", "⟲", tr("Stellung ändern: bekommt die Ansicht, durch die du gerade schaust – keine Animation")),
                               ("add", "➜", tr("Neue Stellung ab dem aktuellen Frame aus der Ansicht, durch die du gerade schaust (Fahrt) – neues Bild"))):
            b = QToolButton(text=text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, k=key: self._action(k))
            self.buttons[key] = b
            top.addWidget(b)
        lay.addLayout(top)
        focal = QHBoxLayout()
        focal.setSpacing(4)
        self.focal = QSlider(Qt.Horizontal)
        self.focal.setRange(0, 1000)
        self.focal.setToolTip(tr("Brennweite dieser Stellung – hält den Ausschnitt (Alt: reiner Zoom)"))
        self.focal.sliderPressed.connect(self._focal_pressed)
        self.focal.valueChanged.connect(self._focal_moved)
        self.focal.sliderReleased.connect(self._focal_released)
        focal.addWidget(self.focal, 1)
        self.mm = QLabel()
        self.mm.setFixedWidth(46)
        focal.addWidget(self.mm)
        lay.addLayout(focal)
        self.setFixedWidth(270)
        self.setStyleSheet(
            f"QFrame#shotrow {{ background: {_rgba(QColor(ui.PANEL), 235)}; border: 1px solid {_rgba(INK, 50)}; "
            f"border-radius: 4px; }} QToolButton {{ background: transparent; color: {_rgba(INK)}; border: none; "
            f"font-size: 12px; padding: 1px 4px; }} QToolButton:hover {{ color: {_rgba(ui.ACCENT)}; }} "
            f"QToolButton:disabled {{ color: {_rgba(INK, 70)}; }} "
            f"QToolButton#camname {{ font-size: 14px; font-weight: bold; }} "
            f"QLabel {{ color: {_rgba(INK, 170)}; background: transparent; font-size: 11px; }}")
        self.refresh()

    def _ref(self):
        """(Viewport, camera, pose) or None (e.g. just deleted)."""
        vp = self.board.vp
        if vp is None or not self.item.shot:
            return None
        cam = vp.scene.timing.camera_by_id(self.item.shot["cam"])
        pose = timing.pose_by_id(cam, self.item.shot["pose"]) if cam is not None else None
        return (vp, cam, pose) if pose is not None else None

    def refresh(self):
        ref = self._ref()
        if ref is None:
            return
        vp, cam, pose = ref
        pix = QPixmap(12, 12)
        pix.fill(QColor.fromRgbF(*cam.rgb))
        self.color.setIcon(QIcon(pix))
        self.name.setText(cam.name)
        self.frame.setText(timing.pose_range_label(cam, pose, vp.scene.timing.end))
        self.focal.blockSignals(True)
        self.focal.setValue(_mm_to_slider(pose.mm))
        self.focal.blockSignals(False)
        self.mm.setText(f"{pose.mm:.0f} mm")
        inside = any(v._look == cam.id for v in vp.ws.viewports)
        for key in ("set", "add"):  # poses are only set from inside
            self.buttons[key].setEnabled(inside)
        self.setStyleSheet(self.styleSheet().split(" /*in*/")[0] +
                           (f" /*in*/ QFrame#shotrow {{ border: 2px solid {QColor.fromRgbF(*cam.rgb).name()}; }}"
                            if inside else ""))

    def _later(self, fn):
        """Run the action after this click – it can rebuild the board (and with it this bar)."""
        ref = self._ref()
        if ref is not None:
            QTimer.singleShot(0, lambda: fn(*ref))

    def _action(self, key):
        if key == "look":
            def look(vp, cam, pose):
                vp.set_frame(pose.frame)
                vp.look_through(cam.id)
                vp.setFocus()
            return self._later(look)
        self._later(lambda vp, cam, pose: vp.commit_pose(key == "add", cam=cam, pose=pose))

    def _color_menu(self):
        ref = self._ref()
        if ref is None:
            return
        menu = QMenu(self)
        for rgb in timing.CAM_COLORS:
            pix = QPixmap(16, 16)
            pix.fill(QColor.fromRgbF(*rgb))
            menu.addAction(QIcon(pix), "", lambda rgb=rgb: self._later(
                lambda vp, cam, pose: vp.set_cam_props(cam, color=None if rgb == timing.CAM_COLOR else tuple(rgb))))
        menu.exec(self.color.mapToGlobal(self.color.rect().bottomLeft()))

    def _rename(self):
        from PySide6.QtWidgets import QInputDialog
        ref = self._ref()
        if ref is None:
            return
        text, ok = QInputDialog.getText(self.board, tr("Kamera umbenennen"), tr("Name"), text=ref[1].name)
        if ok and text.strip() and text.strip() != ref[1].name:
            self._later(lambda vp, cam, pose: vp.set_cam_props(cam, name=text.strip()))

    def _set_frame(self):
        from PySide6.QtWidgets import QInputDialog
        ref = self._ref()
        if ref is None:
            return
        vp, cam, pose = ref
        tm = vp.scene.timing
        value, ok = QInputDialog.getInt(self.board, tr("Stellung ab Frame"), tr("Frame"), pose.frame, tm.start, tm.end)
        if ok and value != pose.frame:
            self._later(lambda vp, cam, pose: vp.move_pose(cam, pose, value))

    def _focal_pressed(self):
        ref = self._ref()
        if ref is not None:
            ref[0].begin_focal(ref[1])  # one undo step per slider drag

    def _focal_moved(self, v):
        ref = self._ref()
        if ref is None:
            return
        vp, cam, pose = ref
        if not self.focal.isSliderDown():  # click into the bar / keyboard: own undo step
            vp.begin_focal(cam)
        keep = not (QApplication.keyboardModifiers() & Qt.AltModifier)
        vp.set_cam_focal(cam, _slider_to_mm(v), keep, pose=pose)
        self.mm.setText(f"{pose.mm:.0f} mm")
        if not self.focal.isSliderDown():
            self._later(lambda vp, cam, pose: vp.end_focal(cam, pose))

    def _focal_released(self):
        self._later(lambda vp, cam, pose: vp.end_focal(cam, pose))


def _mm_to_slider(mm):
    lo, hi = timing.MM_RANGE
    return round(1000 * (math.log(mm) - math.log(lo)) / (math.log(hi) - math.log(lo)))


def _slider_to_mm(v):
    lo, hi = timing.MM_RANGE
    return math.exp(math.log(lo) + v / 1000 * (math.log(hi) - math.log(lo)))


class BackdropItem(QGraphicsItem):
    """Frame with title behind images and texts (like the Network Box in Houdini): marks what belongs together.
    If moved, what lies inside moves along; in the sketch brief every backdrop is its own section with
    its title. Without color: slightly darker than the board, the title bar a bit more."""

    def __init__(self, rect, title=""):
        super().__init__()
        self._rect = QRectF(0, 0, rect.width(), rect.height())
        self.setPos(rect.topLeft())
        self.setZValue(-2)  # behind the images (-1)
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable)
        self.title = BackdropTitle(title, self)
        self._carry, self._last = [], None

    def boundingRect(self):
        return self._rect

    def _band(self):
        """Title bar (tab): only here is the backdrop grabbed – in the frame one drags out a box."""
        return QRectF(self._rect.left(), self._rect.top(), self._rect.width(), self.title.boundingRect().height() + 8)

    def shape(self):
        path = QPainterPath()
        path.addRect(self._band())
        return path

    def set_rect(self, rect):
        """Position and size in board coordinates (drag corners)."""
        self.prepareGeometryChange()
        self._carry = []
        self.setPos(rect.topLeft())
        self._rect = QRectF(0, 0, rect.width(), rect.height())
        self.update()

    def paint(self, p, option, widget=None):
        fill, head = QColor(0, 0, 0, BACKDROP_SHADE[0]), QColor(0, 0, 0, BACKDROP_SHADE[1])
        edge = QColor(INK_DIM)
        edge.setAlpha(70)
        pen = QPen(QColor(ui.ACCENT) if self.isSelected() else edge, 1.4 if self.isSelected() else 1)
        pen.setCosmetic(True)
        p.setPen(Qt.NoPen)
        p.setBrush(fill)
        p.drawRoundedRect(self._rect, 6, 6)
        p.setBrush(head)
        p.drawRoundedRect(self._band(), 6, 6)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(self._rect, 6, 6)

    def contents(self):
        """Texts, images, strokes (and smaller backdrops) whose center lies in the frame."""
        r = self.sceneBoundingRect()
        return [i for i in self.scene().items(r) if i is not self and _top_level(i) and
                r.contains(i.sceneBoundingRect().center())]

    def mousePressEvent(self, e):
        # what lies inside moves along – unless it is also selected (Qt moves that itself)
        self._carry = [i for i in self.contents() if not i.isSelected()]
        self._last = self.pos()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        # only follow after Qt's own move (not from within itemChange: moved there, the
        # content stayed invisible until release)
        super().mouseMoveEvent(e)
        if self._carry and self._last is not None:
            d = self.pos() - self._last
            for item in self._carry:
                item.moveBy(d.x(), d.y())
        self._last = self.pos()

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self._carry, self._last = [], None

    def data(self):
        r = self.sceneBoundingRect()
        return {"kind": "backdrop", "x": round(r.x(), 1), "y": round(r.y(), 1), "w": round(r.width(), 1),
                "h": round(r.height(), 1), "title": self.title.toPlainText().strip()}


class BoardView(QGraphicsView):
    def __init__(self, board):
        super().__init__()
        self.board = board
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("background: transparent; border: none;")
        self.viewport().setAutoFillBackground(False)
        self.setAcceptDrops(True)
        self._pan = None
        self._zoom = None  # Alt + right mouse button: (last position, anchor point on screen, in the board)
        self._right_zoom = False  # last right click was an Alt zoom: no context menu afterwards
        self._scaling = None  # handle dragged: scale image/text
        self._scrub = None  # timeline of a video dragged
        self._stroke = None  # stroke currently being painted
        self._ink = None  # its ink (layer in viewport size, only while painting)
        self._shape = None  # recognized shape (holding still at the stroke end), readjusted until release
        self._straight = False  # stroke is recognized (shape or straight line)
        self._last = None  # last cursor position while painting (board)
        self._hold_anchor = None
        self._hold = QTimer(self, singleShot=True, interval=HOLD_MS)  # holding still: shape or straight line, as in the 3D view
        self._hold.timeout.connect(self._recognize)
        self._erasing = False
        self._pressure = None  # pen pressure of the last tablet event (mouse: None)
        self._cursor = None  # cursor position (eraser circle)
        self._mods = Qt.NoModifier  # Ctrl/Shift of the last mouse event
        self._flash = None  # stroke just painted: briefly show what it connects (board_links)
        self._hover_shot = None  # screenshot under the cursor: shows its ↻ button
        self._hero_edit = None  # (screenshot, QLineEdit): name of the hero camera is being written
        self._pen_drag = None  # Ctrl (+ Shift) + right mouse button: drag brush width (thickness of the strokes)
        self._pen_all = False
        self._pen_show = None  # (place, until when) circle of the brush width
        self._flash_timer = QTimer(self, singleShot=True, interval=LINK_SHOW_MS)
        self._flash_timer.timeout.connect(self._end_flash)
        self.setMouseTracking(True)

    def event(self, e):
        # Ctrl+V/Z/Y belong to the board – otherwise the shortcuts of the main window take over (Ctrl+V pasted the
        # 3D clipboard instead of the image)
        if e.type() == QEvent.ShortcutOverride and e.modifiers() & Qt.ControlModifier and \
                e.key() in (Qt.Key_V, Qt.Key_Z, Qt.Key_Y, Qt.Key_C, Qt.Key_X):
            e.accept()
            return True
        return super().event(e)

    def tabletEvent(self, e):
        """Remember pressure, Qt turns it into mouse events; while painting the board takes the pen position directly
        (fractions of pixels, every position) – otherwise as in the viewport."""
        self._pressure = float(e.pressure())
        if self._stroke is not None and e.type() == QEvent.TabletMove:
            self._mods = e.modifiers()
            self._stroke_to(e.position())
            e.accept()
            return
        e.ignore()

        # --- Stroke while painting: ink on its own layer ------------------------------
        # Per point only the new piece is added and only its surroundings are redrawn – the cost depends neither
        # on the length of the stroke nor on the screen content (cursive writing should follow without delay).
    def _exact_scene(self, pos):
        return self.viewportTransform().inverted()[0].map(QPointF(pos))

    def _ink_begin(self):
        self._compress_tablet = QApplication.testAttribute(Qt.AA_CompressTabletEvents)
        QApplication.setAttribute(Qt.AA_CompressTabletEvents, False)
        self._stroke.begin_live()
        self._ink_clear()
        self._ink_dirty()

    def _ink_clear(self):
        vp = self.viewport()
        ratio = vp.devicePixelRatioF()
        self._ink = QPixmap(vp.size() * ratio)
        self._ink.setDevicePixelRatio(ratio)
        self._ink.fill(Qt.transparent)
        self._ink_view = (QTransform(self.viewportTransform()), vp.size())
        self._ink_done = 0  # this many points already lie on the layer

    def _ink_end(self):
        QApplication.setAttribute(Qt.AA_CompressTabletEvents, self._compress_tablet)
        self._ink = None
        self.viewport().update()

    def _ink_extend(self):
        """Finished pieces onto the layer; the last, still open piece is drawn by drawForeground."""
        s = self._stroke
        n = len(s.pts) - 1
        if self._ink_done >= n:
            return
        p = QPainter(self._ink)
        p.setRenderHint(QPainter.Antialiasing)
        p.setTransform(self.viewportTransform())
        p.setBrush(Qt.NoBrush)
        color = s.qcolor()
        for k in range(self._ink_done, n):
            path = QPainterPath(s.pts[0] if k == 0 else (s.pts[k - 1] + s.pts[k]) / 2)
            if k == 0:
                path.lineTo((s.pts[0] + s.pts[1]) / 2)
            else:
                _ink_piece(path, s.pts[k - 1], s.pts[k], s.pts[k + 1])
            p.setPen(QPen(color, s.widths[k], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawPath(path)
        p.end()
        self._ink_done = n

    def _ink_dirty(self):
        s = self._stroke
        last = s.pts[-3:]
        xs, ys = [q.x() for q in last], [q.y() for q in last]
        pad = max(s.widths[-3:]) + 2 * self._px()
        box = QRectF(min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad)
        self.viewport().update(self.viewportTransform().mapRect(box).toAlignedRect())

    def _draw_ink(self, p):
        s = self._stroke
        if self._ink_view != (self.viewportTransform(), self.viewport().size()):
            self._ink_clear()  # window size or view changed: apply anew
        self._ink_extend()
        p.save()
        p.resetTransform()
        p.drawPixmap(0, 0, self._ink)
        p.restore()
        p.setRenderHint(QPainter.Antialiasing)
        w = s.widths[-1]
        if len(s.pts) == 1:
            p.setPen(Qt.NoPen)
            p.setBrush(s.qcolor())
            p.drawEllipse(s.pts[0], w / 2, w / 2)
        else:
            p.setPen(QPen(s.qcolor(), w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine((s.pts[-2] + s.pts[-1]) / 2, s.pts[-1])

    def _stroke_to(self, view_pos):
        """The stroke follows the pen (screen position with fractions of pixels)."""
        pos = self._exact_scene(view_pos)
        self._last = pos
        if self._straight:
            self._update_straight(pos)
            return
        d = pos - self._stroke.pts[-1]
        if math.hypot(d.x(), d.y()) >= INK_STEP_PX * self._px():  # no swarm of points with a slow hand
            self._stroke.add(pos, self._width())
            self._ink_dirty()
        if (view_pos - self._hold_anchor).manhattanLength() > HOLD_PX:
            self._hold_anchor = QPointF(view_pos)
            self._hold.start()  # restart: holding still counts from here

    def _px(self):
        """Board units per screen pixel (pen and eraser are as large on screen as in the 3D view)."""
        return 1.0 / max(self.transform().m11(), 1e-6)

    def _width(self):
        """Stroke width in board units: one width for the whole board, the same at every zoom (times pen pressure)."""
        vp = self.board.vp
        f = vp._pressure_factor(self._pressure, "pen") if vp is not None and self._pressure is not None else 1.0
        return max(board_pen() * float(f), 0.1)

    def _scale_strokes(self, factor):
        """Ctrl+Shift (mouse wheel, right mouse button drag): thickness of the selected strokes, without selection of all strokes of the
        board – as in the 3D view. False: no strokes."""
        strokes = [i for i in self.scene().selectedItems() if isinstance(i, StrokeItem)] or             [i for i in self.scene().items() if isinstance(i, StrokeItem) and not i.live]
        for st in strokes:
            st.set_points(st.pts, [max(w * factor, 0.05) for w in st.widths])
        if strokes:
            self.board.hint.setText((tr("Dicke der gewählten Striche ({0})") if self.scene().selectedItems() else
                                     tr("Dicke aller Striche im Brett ({0})")).format(len(strokes)))
        return bool(strokes)

    def _scale_pen(self, factor, at=None):
        """Ctrl (mouse wheel, right mouse button drag): brush width of the board, circle shows it briefly."""
        w = set_board_pen(board_pen() * factor)
        self._pen_show = (at if at is not None else self._cursor, time.monotonic() + 1.2)
        QTimer.singleShot(1300, self.viewport().update)
        self.board.hint.setText(tr("Pinselbreite im Brett: {0:.2g} (gilt bei jedem Zoom)").format(w))
        self.viewport().update()

    def _erase_at(self, pos):
        """Erase as in the 3D view: what the circle hits; Ctrl+Shift: whole strokes."""
        whole = bool(self._mods & Qt.ControlModifier and self._mods & Qt.ShiftModifier)
        radius = float(getattr(self.board.vp.ws, "eraser_px", 24.0)) / 2 * self._px() if self.board.vp else 12 * self._px()
        for item in list(self.scene().items()):
            if not isinstance(item, StrokeItem):
                continue
            rest = item.erase(pos, radius)
            if rest is None:
                continue
            self.scene().removeItem(item)
            for pts, widths in rest if not whole else ():
                self.scene().addItem(StrokeItem(pts, widths, item.style, item.color))

        # --- Shape recognition (holding still at the stroke end, as in the 3D view) -----------------
    def _recognize(self):
        s = self._stroke
        if s is None or self._straight or len(s.pts) < 2 or self._last is None:
            return
        self._straight = True
        s.end_live()  # recognized shape: few points, the element draws itself
        self._ink_end()
        pts = np.array([[p.x(), -p.y()] for p in s.pts])  # shapes computes with y up
        shape = shapes.recognize(pts, (0.0,))
        self._shape = shapes.prepare(shape, self._to2d(self._last)) if shape is not None else None
        self._stroke_w = float(np.mean(s.widths))
        self._start = QPointF(s.pts[0])
        if self._shape is not None:
            extra = tr(" · Strg: Quadrat") if self._shape["kind"] == "rect" else tr(" · Strg: Winkelschritte") \
                if self._shape["kind"] == "arc" else ""
            self.board.hint.setText(tr("{0} erkannt – Maus: Größe{1} · Loslassen: fertig").format(shapes.NAMES[self._shape['kind']], extra))
        else:
            self.board.hint.setText(tr("Gerade – Strg: Winkelschritte · Loslassen: fertig"))
        self._update_straight(self._last)

    @staticmethod
    def _to2d(p):
        return np.array([p.x(), -p.y()])

    def _snap_deg(self):
        vp = self.board.vp
        return float(getattr(vp.ws, "snap_angle", 15.0)) if vp is not None else 15.0

    def _update_straight(self, pos):
        ctrl = bool(self._mods & Qt.ControlModifier)
        if self._shape is not None:
            self._shape = shapes.adjust(self._shape, self._to2d(pos), square=ctrl,
                                        snap_deg=self._snap_deg() if ctrl else None)
            lines = shapes.polylines(self._shape)
            joined = list(lines[0])
            for line in lines[1:]:  # rectangle: four sides as one stroke
                joined += list(line[1:]) if np.allclose(line[0], joined[-1]) else list(line)
            pts = [QPointF(float(x), -float(y)) for x, y in joined]
        else:
            end = QPointF(pos)
            if ctrl:  # angle steps as with straight lines in the 3D view
                d = pos - self._start
                length, step = math.hypot(d.x(), d.y()), math.radians(self._snap_deg())
                angle = round(math.atan2(d.y(), d.x()) / step) * step
                end = self._start + QPointF(length * math.cos(angle), length * math.sin(angle))
            pts = [self._start, end]
        self._stroke.set_points(pts, [self._stroke_w] * len(pts))

        # --- Handles: scale selected images and texts at the corners (like PureRef) ---------
    def _scalable(self):
        return [i for i in self.scene().selectedItems() if isinstance(i, (ImageItem, TextBlock, BackdropItem))] \
            if self.board.paint_tool is None else []

    @staticmethod
    def _corners(rect):
        return [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]

    def _redo_shots(self):
        """Screenshots with ↻ button: selected ones and the one under the cursor."""
        if self.board.paint_tool is not None or self.board.vp is None:
            return []
        shots = [i for i in self.scene().selectedItems() if isinstance(i, ImageItem) and i.view]
        hover = self._hover_shot
        if hover is not None and hover.scene() is self.scene() and hover not in shots:
            shots.append(hover)
        return shots

    def _badge_scale(self, item):
        """(Image rectangle on screen, scale) for ↻, checkbox and name on the screenshot: together at most BADGE_SHARE
        of the image width, never larger than at full size – so they always hang the same on the image, however far zoomed."""
        from PySide6.QtGui import QFontMetrics
        rect = self.mapFromScene(item.sceneBoundingRect()).boundingRect()
        full = QFontMetrics(note_font(11)).horizontalAdvance(HERO_LABEL) + 34 + 2 * REDO_PX + 26  # checkbox + ↻ + margins
        return rect, max(0.15, min(1.0, BADGE_SHARE * rect.width() / full))

    def _redo_center(self, item, scaled=None):
        """Center and radius of the ↻ button (screen): top right in the image, scaled with the image."""
        rect, k = scaled or self._badge_scale(item)
        return QPointF(rect.right() - (REDO_PX + 10) * k, rect.top() + (REDO_PX + 10) * k), REDO_PX * k

    def _hero_rects(self, item):
        """(Checkbox, name, scale) on the screenshot (screen): left of ↻, the name below – same scale."""
        from PySide6.QtGui import QFontMetrics
        scaled = self._badge_scale(item)
        k = scaled[1]
        c, r = self._redo_center(item, scaled)
        fm = QFontMetrics(note_font(11))
        w, hgt = (fm.horizontalAdvance(HERO_LABEL) + 34) * k, (2 * REDO_PX - 4) * k  # room: rounded text
        box = QRectF(c.x() - r - 6 * k - w, c.y() - hgt / 2, w, hgt)
        name_w = max(w, (fm.horizontalAdvance(item.hero_name or tr("Name …")) + 14) * k)
        return box, QRectF(box.right() - name_w, box.bottom() + 4 * k, name_w, hgt), k

    def _hero_shots(self):
        """Screenshots with visible checkbox: like ↻ – selected or under the cursor, or the name is being written
        (hero images are otherwise recognized by the turquoise frame). Camera images in shots have no checkbox."""
        shots = [i for i in self._redo_shots() if not i.shot]
        if self._hero_edit is not None and not _on_scene(self._hero_edit[0], self.scene()):
            self._end_hero_edit(apply=False)  # image gone (Ctrl+Z, board reloaded): do not leave the field standing
        if self._hero_edit is not None and self._hero_edit[0] not in shots:
            shots.append(self._hero_edit[0])
        return shots

    def _hero_at(self, pos):
        """("box" | "name", screenshot) under the cursor or None."""
        for item in self._hero_shots():
            box, name, _k = self._hero_rects(item)
            if box.adjusted(-2, -2, 2, 2).contains(pos):
                return "box", item
            if item.hero and name.contains(pos):
                return "name", item
        return None

    def toggle_hero(self, item):
        """Toggle checkbox (one undo step); on: write the name right away."""
        item.hero = not item.hero
        if not item.hero:
            self._end_hero_edit(apply=False)  # checkbox off: no open name field any more
        if item.hero:
            taken = self.hero_names(skip=item)
            if not item.hero_name:  # default: smallest free number in the whole scene
                item.hero_name = next(f"Hero {n}" for n in range(1, len(taken) + 2) if f"Hero {n}" not in taken)
            item.hero_name = unique_name(item.hero_name, taken)
        self.board.commit()
        self.viewport().update()
        if item.hero:
            self.edit_hero_name(item)

    def hero_names(self, skip=None):
        """Names of all hero cameras of the scene – this board as it is right now, and the other notes."""
        names = {i.hero_name for i in self.scene().items() if isinstance(i, ImageItem) and i.hero and i is not skip}
        vp = self.board.vp
        if vp is not None:
            for note in vp.scene.notes + [vp.scene.global_note]:
                if note.id != self.board.note_id:
                    names |= {i.get("hero_name") for i in note.board.get("items", []) if i.get("hero")}
        return names - {None, ""}

    def edit_hero_name(self, item):
        """Small name field under the checkbox; Enter or clicking away accepts."""
        self._end_hero_edit()
        edit = _HeroNameEdit(item.hero_name, self.viewport(), self._end_hero_edit)
        edit.font_px = None
        self._place_hero_edit(item, edit)
        edit.selectAll()
        edit.editingFinished.connect(self._end_hero_edit)
        self._hero_edit = (item, edit)
        edit.show()
        edit.setFocus()

    def _place_hero_edit(self, item, edit):
        """Name field exactly over the name, in its scale (follows zoom and moving)."""
        _box, name, k = self._hero_rects(item)
        px = max(round(11 * k), 4)
        if px != edit.font_px:  # font size in its own style (the app style gives input fields a fixed size);
            # own background: the board is styled transparent, that would otherwise show through onto the field
            edit.setStyleSheet(f"background: {HERO_FILL.name()}; color: {HERO_TEXT.name()}; border: 1px solid "
                               f"{HERO_LINE.name()}; border-radius: 3px; padding: 0 1px; font-size: {px}px;")
            edit.font_px = px
        edit.setGeometry(name.adjusted(0, -1 * k, 40 * k, 1 * k).toRect())

    def _end_hero_edit(self, apply=True):
        """Close name field (Enter, clicking away, Esc, checkbox off, image gone); apply: accept name."""
        if self._hero_edit is None:
            return
        item, edit = self._hero_edit
        self._hero_edit = None
        name = edit.text().strip()
        edit.hide()
        edit.deleteLater()
        if apply and _on_scene(item, self.scene()) and name and name != item.hero_name:
            item.hero_name = unique_name(name, self.hero_names(skip=item))  # duplicate: append _1, _2 …
            self.board.commit()
        self.viewport().update()

    def _redo_at(self, pos):
        for item in self._redo_shots():
            c, r = self._redo_center(item)
            if math.hypot(pos.x() - c.x(), pos.y() - c.y()) <= r + 2:
                return item
        return None

    def _handle_at(self, pos):
        """(Element, corner) under the cursor (screen) or None."""
        for item in self._scalable():
            for k, c in enumerate(self._corners(item.sceneBoundingRect())):
                q = self.mapFromScene(c)
                if abs(q.x() - pos.x()) <= HANDLE_PX and abs(q.y() - pos.y()) <= HANDLE_PX:
                    return item, k
        return None

    def _scale_to(self, pos):
        item, k, anchor, diag, s0 = (self._scaling[key] for key in ("item", "corner", "anchor", "diag", "scale"))
        if isinstance(item, BackdropItem):  # backdrop: drag out the frame, the opposite corner stays
            q = self.mapToScene(pos.toPoint())
            sx, sy = (1 if q.x() >= anchor.x() else -1), (1 if q.y() >= anchor.y() else -1)
            w, h = max(abs(q.x() - anchor.x()), BACKDROP_MIN[0]), max(abs(q.y() - anchor.y()), BACKDROP_MIN[1])
            item.set_rect(QRectF(anchor, anchor + QPointF(sx * w, sy * h)).normalized())
            return
        d = self.mapToScene(pos.toPoint()) - anchor
        f = (d.x() * diag.x() + d.y() * diag.y()) / max(diag.x() ** 2 + diag.y() ** 2, 1e-9)
        item.setScale(max(s0 * f, 0.02))
        moved = self._corners(item.sceneBoundingRect())[(k + 2) % 4]  # opposite corner stays put
        item.moveBy(anchor.x() - moved.x(), anchor.y() - moved.y())

    def drawForeground(self, p, rect):
        if self._stroke is not None and self._stroke.live:
            self._draw_ink(p)  # only the ink: handles and buttons are off while painting, tags come afterwards
            return
        if self._pen_show is not None and self._pen_show[0] is not None and time.monotonic() < self._pen_show[1]:
            pen = QPen(INK, 1)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self._pen_show[0], board_pen() / 2, board_pen() / 2)
        if self.board.paint_tool == "eraser" and self._cursor is not None and self.board.vp is not None:
            r = float(self.board.vp.ws.eraser_px) / 2 * self._px()
            pen = QPen(INK, 1)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(self._cursor, r, r)
        items = self._scalable()
        if items:
            pen = QPen(QColor(ui.ACCENT), 1)
            pen.setCosmetic(True)
            p.setPen(pen)
            p.setBrush(QColor(ui.PANEL))
            h = HANDLE_PX * 0.8 * self._px()
            for item in items:
                for c in self._corners(item.sceneBoundingRect()):
                    p.drawRect(QRectF(c.x() - h, c.y() - h, 2 * h, 2 * h))

        heroes = self._hero_shots()
        if heroes:
            p.save()
            p.resetTransform()
            for item in heroes:
                box, name, k = self._hero_rects(item)
                p.setFont(note_font(max(11 * k, 4)))
                p.setPen(QPen(HERO_LINE, max(1.2 * k, 0.6)))
                p.setBrush(HERO_FILL)
                p.drawRoundedRect(box, 4 * k, 4 * k)
                tick = QRectF(box.left() + 5 * k, box.center().y() - 6 * k, 12 * k, 12 * k)
                p.setBrush(HERO_LINE if item.hero else QColor(255, 255, 255))
                p.drawRect(tick)
                if item.hero:
                    p.setPen(QColor(255, 255, 255))
                    p.drawText(tick, Qt.AlignCenter, "✓")
                p.setPen(HERO_TEXT)
                p.drawText(box.adjusted(22 * k, 0, -4 * k, 0), Qt.AlignVCenter | Qt.AlignLeft, HERO_LABEL)
                if self._hero_edit is not None and self._hero_edit[0] is item:  # field follows zoom and moving
                    self._place_hero_edit(item, self._hero_edit[1])
                elif item.hero:
                    p.setPen(Qt.NoPen)
                    p.setBrush(HERO_FILL)
                    p.drawRoundedRect(name, 4 * k, 4 * k)
                    p.setPen(HERO_TEXT)
                    p.drawText(name.adjusted(7 * k, 0, -4 * k, 0), Qt.AlignVCenter | Qt.AlignLeft,
                               item.hero_name or tr("Name …"))
            p.restore()

        shots = self._redo_shots()
        if shots:
            p.save()
            p.resetTransform()
            for item in shots:
                c, r = self._redo_center(item)
                p.setFont(note_font(max(13 * r / REDO_PX, 4)))
                p.setPen(QPen(QColor(ui.ACCENT), max(1.5 * r / REDO_PX, 0.6)))
                p.setBrush(QColor(ui.PANEL))
                p.drawEllipse(c, r, r)
                p.setPen(INK)
                p.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, "↻")
            p.restore()

        strokes = [s for s in self.scene().selectedItems() if isinstance(s, StrokeItem)][:12]
        if self._flash is not None and _on_scene(self._flash, self.scene()) and self._flash not in strokes:
            strokes.append(self._flash)
        for s in strokes:
            link = board_links.stroke_link(self.scene(), s)
            if link is not None:
                self._draw_link(p, link)

    def show_link(self, stroke):
        """After painting briefly show what the stroke connects (label, image, place of the tip) – only if
        it connects something: writing and doodling in the open stays quiet (artist, 2026-11-07). Selected strokes
        keep showing every result, also "unclear"."""
        if not board_links.found(board_links.stroke_link(self.scene(), stroke)):
            return
        self._flash = stroke
        self._flash_timer.start()
        self.viewport().update()

    def _end_flash(self):
        self._flash = None
        self.viewport().update()

    def _draw_link(self, p, link):
        """Connection of a stroke: source and target framed, ring at the tip and a tag in screen size
        (so the brief reads it too)."""
        px = self._px()
        source, target, tip = link["source"], link["target"], link["tip"]
        pen = QPen(LINK_COLOR, 2)
        pen.setCosmetic(True)
        fill = QColor(LINK_COLOR)
        fill.setAlpha(36)
        p.setPen(pen)
        p.setBrush(fill)
        for item in (source, target):
            if item is not None:
                p.drawRoundedRect(item.sceneBoundingRect().adjusted(-4 * px, -4 * px, 4 * px, 4 * px), 4 * px, 4 * px)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(tip, 7 * px, 7 * px)

        def what(item):
            return ("Screenshot" if item.view else tr("Bild")) if isinstance(item, ImageItem) else                 "Text" if isinstance(item, TextBlock) else tr("ins Leere")
        kind = link["kind"]
        text = (f"„{board_links.label_of(source)}“ → {what(target)}" if kind == "link" else
                f"{what(source)} → {what(target)}" if kind == "image" else
                tr("Markierung auf {0}").format(what(target)) if kind == "mark" and target is not None else
                tr("unklar – kein Ende an Text oder Bild"))
        at = self.mapFromScene(tip)
        p.save()
        p.resetTransform()
        p.setFont(note_font(12))
        w = p.fontMetrics().horizontalAdvance(text) + 14
        box = QRectF(at.x() + 12, at.y() + 10, w, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(ui.PANEL))
        p.drawRoundedRect(box, 4, 4)
        p.setPen(QPen(LINK_COLOR, 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(box, 4, 4)
        p.setPen(INK)
        p.drawText(box, Qt.AlignCenter, text)
        p.restore()

    def sizeHint(self):  # otherwise as large as the (huge) scene: the panel would push out the viewport
        return QSize(360, 260)

    def editing(self):
        item = self.scene().focusItem()  # text block or title of a backdrop
        return isinstance(item, QGraphicsTextItem) and item.textInteractionFlags() & Qt.TextEditorInteraction

    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120
        item = self.itemAt(e.position().toPoint())
        ctrl, shift = bool(e.modifiers() & Qt.ControlModifier), bool(e.modifiers() & Qt.ShiftModifier)
        if ctrl and shift:  # as in the 3D view: Ctrl+Shift = thickness of the strokes (selected, otherwise all)
            if self._scale_strokes(1.1 ** steps):
                self.board.commit(burst="board-widths")
            return
        if ctrl and (self.board.paint_tool is not None or not isinstance(item, (ImageItem, TextBlock))):
            return self._scale_pen(1.15 ** steps, self.mapToScene(e.position().toPoint()))  # as in the 3D view: brush
        if ctrl and isinstance(item, (ImageItem, TextBlock)):  # Ctrl+mouse wheel on image/text (without painting): scale
            item.setScale(max(0.02, item.scale() * ZOOM_STEP ** steps))
            self.board.commit(burst="board-wheel")  # notches of a rotation: one undo step
        else:
            self._zoom_by(ZOOM_STEP ** steps)

    def _zoom_by(self, f):
        lo, hi = ZOOM_RANGE
        m = self.transform().m11()
        f = max(lo / m, min(hi / m, f))
        self.scale(f, f)

    def contextMenuEvent(self, e):
        """Right click on an image: image caption (not after Alt + right mouse button – that zooms)."""
        if self._right_zoom or self.board.paint_tool is not None:
            return
        item = next((i for i in self.items(e.pos()) if isinstance(i, ImageItem)), None)
        can_paste = QApplication.clipboard().mimeData().hasFormat(BOARD_MIME)
        if item is None:
            if can_paste:  # into the void: paste copied elements here
                menu = QMenu(self)
                menu.addAction(tr("Einfügen\tStrg+V"), self.board.paste)
                menu.exec(e.globalPos())
            return
        menu = QMenu(self)
        row = getattr(item, "row", None) if item.shot else None
        if row is not None and row._ref() is not None:
            _vp, cam, _pose = row._ref()
            menu.addAction(tr("Durch die Kamera schauen\tDoppelklick"), lambda: row._action("look"))
            menu.addAction(tr("Kamera umbenennen …"), row._rename)
            menu.addAction(tr("Stellung ab Frame …"), row._set_frame)
            sizes = menu.addMenu(tr("Symbolgröße"))
            for f in (0.25, 0.5, 1.0, 2.0, 4.0):
                a = sizes.addAction(f"× {f:g}")
                a.setCheckable(True)
                a.setChecked(abs(cam.size - f) < 1e-6)
                a.triggered.connect(lambda _=False, f=f: row._later(lambda vp, c, p: vp.set_cam_props(c, size=f)))
            menu.addSeparator()
        menu.addAction(tr("Bildunterschrift …") if not item.caption else tr("Bildunterschrift ändern …"),
                       lambda: self.edit_caption(item))
        if isinstance(item, VideoItem) and not item.absent:
            menu.addAction(tr("Standbild ins Brett"), lambda: self.board.video_still(item))
        menu.addSeparator()
        if not item.isSelected():  # right click on an unselected image: this one is meant
            self.scene().clearSelection()
            item.setSelected(True)
        menu.addAction(tr("Kopieren\tStrg+C"), self.board.copy_selection)
        menu.addAction(tr("Ausschneiden\tStrg+X"), lambda: self.board.copy_selection(cut=True))
        if can_paste:
            menu.addAction(tr("Einfügen\tStrg+V"), self.board.paste)
        menu.exec(e.globalPos())

    def edit_caption(self, item):
        """Write an image's caption – applies to the same image in all notes (one undo step)."""
        from PySide6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getMultiLineText(
            self, tr("Bildunterschrift"), tr("Wofür steht das Bild hier, und worauf kommt es an? (z. B. „nur das Scharnier links – die Proportionen des Fotos gelten nicht“; leer: keine Unterschrift)"),
            item.caption)
        if not ok or text.strip() == item.caption:
            return
        if self.board.vp is not None:
            set_image_captions(self.board.vp, {item.file: text})
        else:
            item.set_caption(text)
            self.board.commit()
        self.viewport().update()

    def mousePressEvent(self, e):
        self._mods = e.modifiers()
        self._right_zoom = e.button() == Qt.RightButton and bool(e.modifiers() & Qt.AltModifier)
        if e.button() == Qt.RightButton and e.modifiers() & Qt.AltModifier:  # Alt + right mouse button: zoom
            self._zoom = (e.position(), e.position(), self.mapToScene(e.position().toPoint()))
            self.setCursor(Qt.SizeVerCursor)
            return
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and e.modifiers() & Qt.AltModifier):
            self._pan = e.position()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if e.button() == Qt.RightButton and e.modifiers() & Qt.ControlModifier:
            # as in the 3D view: Ctrl + right mouse button drag = brush width, with Shift = thickness of the strokes
            self._pen_drag = e.position()
            self._pen_all = bool(e.modifiers() & Qt.ShiftModifier)
            self._pen_at = self.mapToScene(e.position().toPoint())
            self._right_zoom = True  # no context menu afterwards
            return
        tool = self.board.paint_tool
        if tool is not None and e.button() == Qt.LeftButton:  # painting: pen paints, eraser deletes
            pos = self._exact_scene(e.position())
            if tool == "eraser":
                self._erasing = True
                self._erase_at(pos)
            else:
                self._stroke = StrokeItem([pos], [self._width()], "pen", self.board.pen_color())
                self._shape, self._straight, self._last = None, False, pos
                self._hold_anchor = e.position()
                self._hold.start()
                self.scene().addItem(self._stroke)
                self._ink_begin()
            return
        if e.button() == Qt.LeftButton:
            hero = self._hero_at(e.position())
            if hero is not None:  # checkbox: hero camera on/off; name: rename
                (self.toggle_hero if hero[0] == "box" else self.edit_hero_name)(hero[1])
                return
            shot = self._redo_at(e.position())
            if shot is not None:  # ↻: retake (Shift: from the current view of the viewport)
                self.board.retake_item(shot, current=bool(e.modifiers() & Qt.ShiftModifier))
                return
            hit = self._handle_at(e.position())
            if hit is not None:
                item, k = hit
                corners = self._corners(item.sceneBoundingRect())
                self._scaling = {"item": item, "corner": k, "anchor": corners[(k + 2) % 4],
                                 "diag": corners[k] - corners[(k + 2) % 4], "scale": item.scale()}
                return
            control = self._video_control(e.position())
            if control is not None:
                item, what, local = control
                if what == "button":
                    self.board.toggle_video(item)
                else:  # timeline: stop and scrub as long as dragged
                    self._scrub = item
                    self.board.seek_video(item, item.fraction_at(local))
                return
        super().mousePressEvent(e)

    def _video_control(self, pos):
        """(Video, "bar"/"button", place in the element) at a place of the board – text above takes precedence."""
        item = next((i for i in self.items(pos.toPoint()) if isinstance(i, (VideoItem, TextBlock))), None)
        if not isinstance(item, VideoItem):
            return None
        local = item.mapFromScene(self.mapToScene(pos.toPoint()))
        what = item.control_at(local)
        return (item, what, local) if what else None

    def mouseMoveEvent(self, e):
        self._mods = e.modifiers()
        pos = self.mapToScene(e.position().toPoint())
        if self.board.paint_tool == "eraser":
            self._cursor = pos
            self.viewport().update()
        if self._zoom is not None:
            last, start, anchor = self._zoom
            d = e.position() - last
            self._zoom = (e.position(), start, anchor)
            self.setTransformationAnchor(QGraphicsView.NoAnchor)
            self._zoom_by(math.exp((d.x() - d.y()) * ZOOM_DRAG))
            now = self.mapToScene(start.toPoint())  # the anchor point stays under the cursor
            self.translate(now.x() - anchor.x(), now.y() - anchor.y())
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            return
        if self._scaling is not None:
            self._scale_to(e.position())
            return
        if self._pen_drag is not None:
            d = e.position() - self._pen_drag
            self._pen_drag = e.position()
            if self._pen_all:
                self._scale_strokes(math.exp((d.x() - d.y()) * 0.01))
            else:
                self._scale_pen(math.exp((d.x() - d.y()) * 0.01), self._pen_at)
            return
        if self._scrub is not None:
            self.board.seek_video(self._scrub, self._scrub.fraction_at(self._scrub.mapFromScene(pos)))
            return
        if self._stroke is not None:
            self._stroke_to(e.position())
            return
        if self._erasing:
            self._erase_at(pos)
            return
        if self._pan is not None:
            d = e.position() - self._pan
            self._pan = e.position()
            self.setTransformationAnchor(QGraphicsView.NoAnchor)
            self.translate(d.x() / self.transform().m11(), d.y() / self.transform().m22())
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            return
        if not e.buttons():
            under = self.itemAt(e.position().toPoint())
            under = under if isinstance(under, ImageItem) and under.view else None
            if under is not self._hover_shot:
                self._hover_shot = under
                self.viewport().update()
            hit = self._handle_at(e.position())
            self.viewport().setCursor((Qt.SizeFDiagCursor if hit[1] % 2 == 0 else Qt.SizeBDiagCursor) if hit else
                                      Qt.PointingHandCursor if self._redo_at(e.position()) is not None
                                      or self._hero_at(e.position()) is not None
                                      or (self.board.paint_tool is None
                                          and self._video_control(e.position()) is not None) else
                                      Qt.CrossCursor if self.board.paint_tool == "pen" else
                                      Qt.BlankCursor if self.board.paint_tool == "eraser" else Qt.ArrowCursor)
        super().mouseMoveEvent(e)
        if e.buttons() & Qt.LeftButton and self._scalable():
            self.viewport().update()  # handles move along

    def mouseReleaseEvent(self, e):
        if self._pen_drag is not None:
            self._pen_drag = None
            if self._pen_all:
                self.board.commit()  # thickness of the strokes: one undo step
            return
        if self._stroke is not None or self._erasing:
            self._hold.stop()
            if self._straight:
                self.board.set_paint_tool(self.board.paint_tool)  # hint line back
            done = self._stroke
            if done is not None and done.live:
                done.end_live()
                self._ink_end()
            self._stroke, self._erasing, self._pressure = None, False, None
            self._shape, self._straight = None, False
            self.board.commit_soon(done)  # nothing heavy on release: only after a pause
            return
        if self._zoom is not None:
            self._zoom = None
            self.unsetCursor()
            return
        if self._scaling is not None:
            self._scaling = None
            self.board.commit()
            return
        if self._scrub is not None:
            self._scrub = None
            return
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
            return
        super().mouseReleaseEvent(e)
        self.board.commit()  # moved (only if something changed)

    def mouseDoubleClickEvent(self, e):
        if self.board.paint_tool is not None:  # while painting: second click is another stroke
            return self.mousePressEvent(e)
        item = self.itemAt(e.position().toPoint())
        if item is None or isinstance(item, BackdropItem):  # into the void (also in a backdrop): new text block
            self.board.add_text(self.mapToScene(e.position().toPoint()), edit=True)
            return
        if isinstance(item, ImageItem) and item.shot and self.board.vp is not None:  # shots: look through the camera
            row = getattr(item, "row", None)
            if row is not None:
                row._action("look")
            return
        if isinstance(item, ImageItem) and item.view and self.board.vp is not None:  # screenshot: fly there
            fly_to_view(self.board.vp, item.view)
            return
        super().mouseDoubleClickEvent(e)

    def keyPressEvent(self, e):
        if self.editing():
            if e.key() == Qt.Key_Escape:
                self.scene().focusItem().clearFocus()
                return
            return super().keyPressEvent(e)
        mods = e.modifiers()
        if e.key() == Qt.Key_V and mods & Qt.ControlModifier:
            return self.board.paste()
        if e.key() in (Qt.Key_C, Qt.Key_X) and mods & Qt.ControlModifier:  # elements from board to board
            return self.board.copy_selection(cut=e.key() == Qt.Key_X)
        if mods & Qt.ControlModifier and e.key() in (Qt.Key_Z, Qt.Key_Y):  # same history as in the 3D view
            return self.board.undo(redo=e.key() == Qt.Key_Y or bool(mods & Qt.ShiftModifier))
        if e.key() == Qt.Key_A and not mods:  # pen
            return self.board.set_paint_tool("pen")
        if e.key() == Qt.Key_D and not mods:  # eraser
            return self.board.set_paint_tool("eraser")
        if e.key() == Qt.Key_B and not mods:  # backdrop around the selection
            return self.board.add_backdrop()
        if e.key() == Qt.Key_B and mods == Qt.ShiftModifier:  # auto backdrops by proximity
            return self.board.auto_backdrops()
        if e.key() == Qt.Key_Escape and self.board.paint_tool is not None:  # first painting off, then close
            return self.board.set_paint_tool(None)
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            for item in self.scene().selectedItems():
                self.scene().removeItem(item)
            self.board.commit()
            return
        if e.key() == Qt.Key_Escape:
            return self.board.close_board()
        if e.key() == Qt.Key_Space and not mods:  # video: running or selected – play/pause
            chosen = self.board.video_item or next((i for i in self.scene().selectedItems()
                                                    if isinstance(i, VideoItem)), None)
            if chosen is not None:
                return self.board.toggle_video(chosen)
        if e.key() == Qt.Key_F:  # show everything
            return self.board.fit()
        super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() or e.mimeData().hasImage() or e.mimeData().hasText():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        e.acceptProposedAction()

    def dropEvent(self, e):
        self.board.insert_mime(e.mimeData(), self.mapToScene(e.position().toPoint()))
        e.acceptProposedAction()


class NoteBoard(QFrame):
    """Note board of a note – as a large overlay over the viewports (maximize) or in the "Note" panel below
    the outliner (only in notes mode; dockable like any panel, e.g. below the viewport, or floating)."""

    def __init__(self, host, overlay=True, on_maximize=None, on_scope=None, on_pin=None):
        super().__init__(host if overlay else None)
        self.area = host
        self.overlay = overlay
        self.vp = None
        self.note_id = None
        self.loaded = None
        self.loaded_title = ""
        self.committing = False
        # Strokes: only after a short pause into the note – writing stays fluid, one pause = one undo step
        self._settle_timer = QTimer(self, singleShot=True, interval=SETTLE_MS)
        self._settle_timer.timeout.connect(self._settle)
        self._settle_strokes = []
        self.setObjectName("noteboard")
        self.gscene = QGraphicsScene(self)
        self.gscene.setSceneRect(QRectF(-20000, -20000, 40000, 40000))
        self.video_item, self._player, self._sink = None, None, None  # at most one video runs at any time
        self.view = BoardView(self)
        self.view.setScene(self.gscene)
        self.gscene.selectionChanged.connect(self.view.viewport().update)  # selected strokes show their connection
        self.gscene.changed.connect(lambda _rects: follow_arrows(self.gscene))  # Mule arrows follow text and image
        self.title = QLineEdit()
        self.title.setPlaceholderText(tr("Überschrift"))
        self.title.setFont(note_font(24 if overlay else 18))
        head = QHBoxLayout()
        head.setContentsMargins(18 if overlay else 10, 12 if overlay else 8, 12 if overlay else 6, 4)
        if not overlay:  # panel: Local (clicked note) | Global (note of the whole scene)
            self.scope_buttons = {}
            for key, text, tip in (("local", tr("Lokal"), tr("Die im Viewport angeklickte Notiz")),
                                   ("global", "Global", tr("Notiz der ganzen Szene: Projekt-Infos, Moodboard, allgemeine Prompts"))):
                b = QToolButton()
                b.setText(text)
                b.setCheckable(True)
                b.setChecked(key == "local")
                b.setToolTip(tip)
                b.clicked.connect(lambda _=False, k=key: on_scope and on_scope(k))
                self.scope_buttons[key] = b
                head.addWidget(b)
            head.addSpacing(8)
        head.addWidget(self.title, 1)
        self.cam_button = QToolButton()
        self.cam_button.setIcon(camera_icon(INK))
        self.cam_button.setToolTip(tr("Ansicht als Bild einfügen (aktuelle Perspektive, ohne Bedienelemente) – Doppelklick aufs Bild fliegt dorthin zurück"))
        self.cam_button.clicked.connect(self.add_snapshot)
        head.addWidget(self.cam_button)
        self.backdrop_button = QToolButton()
        self.backdrop_button.setIcon(backdrop_icon(INK))
        self.backdrop_button.setToolTip(tr("Backdrop (B): Rahmen mit Titel um die Auswahl (sonst in der Mitte) – markiert, was zusammengehört; am Reiter verschieben nimmt den Inhalt mit, im Sketch Brief ein eigener Abschnitt · Rechtsklick: Auto-Backdrops (Shift+B)"))
        self.backdrop_button.clicked.connect(self.add_backdrop)
        self.backdrop_button.setContextMenuPolicy(Qt.CustomContextMenu)
        self.backdrop_button.customContextMenuRequested.connect(self._backdrop_menu)
        head.addWidget(self.backdrop_button)
        self.retake_button = QToolButton()  # only visible with screenshots: retake missing or all
        self.retake_button.setText("↻")
        self.retake_button.clicked.connect(self.retake_missing)
        self.retake_button.hide()
        head.addWidget(self.retake_button)
        self.paint_tool = None  # painting in the board: "pen", "eraser" or off
        self.paint_button = QToolButton()
        self.paint_button.setText("✎")
        self.paint_button.setCheckable(True)
        self.paint_button.setToolTip(tr("Malen (A: Stift, D: Radierer, Esc: aus) – Größe und Druck wie im 3D-View, Farbe daneben"))
        self.paint_button.clicked.connect(lambda on: self.set_paint_tool("pen" if on else None))
        head.addWidget(self.paint_button)
        self.color_buttons = []  # pen colors of the board: fixed colors and a custom one (color wheel)
        head.addSpacing(4)
        for rgb, name in BOARD_COLORS:
            b = QToolButton(objectName="swatch")
            b.setIconSize(QSize(18, 18))
            b.setToolTip(tr("Stiftfarbe: {0}").format(name))
            b.clicked.connect(lambda _=False, c=rgb: self.set_pen_color(c))
            self.color_buttons.append((b, rgb))
            head.addWidget(b)
        self.custom_button = QToolButton(objectName="swatch")
        self.custom_button.setIconSize(QSize(18, 18))
        self.custom_button.setToolTip(tr("Eigene Stiftfarbe (Farbrad)"))
        self.custom_button.clicked.connect(self._open_color_popup)
        head.addWidget(self.custom_button)
        head.addSpacing(4)
        self._custom_color = None
        self._color_popup = None
        if overlay:
            button = QToolButton()
            button.setText("✕")
            button.setToolTip(tr("Schließen (Esc) – der Inhalt landet in der Notiz"))
            button.clicked.connect(self.close_board)
        else:
            self.pin_button = QToolButton()  # pinned: panel stays visible also while drawing, painting and in edit
            self.pin_button.setText("📌")
            self.pin_button.setCheckable(True)
            self.pin_button.setToolTip(tr("Anheften: das Notizbrett bleibt in jedem Modus sichtbar (Zeichnen, Malen, Edit) – als Referenzbrett"))
            self.pin_button.clicked.connect(lambda on: on_pin and on_pin(on))
            head.addWidget(self.pin_button)
            button = QToolButton()
            button.setText("⤢")
            button.setToolTip(tr("Maximieren: das Brett füllt das ganze Programmfenster (Esc: zurück)"))
            button.clicked.connect(lambda: on_maximize and on_maximize())
        head.addWidget(button)
        self.empty = QLabel(tr("Im Viewport eine Notiz anklicken – hier steht dann alles, was zu ihr gehört."))
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignCenter)
        self.hint = QLabel(tr("Strg+V: Bild oder Text einfügen · Dateien hineinziehen · Doppelklick: Text · Kamera: Ansicht als Bild (Doppelklick darauf fliegt zurück) · A/✎: malen, D: radieren · B: Backdrop (Shift+B: automatisch) · Ecken ziehen: skalieren · Entf: löschen · Strg+Z/Y · Mausrad/Alt+Rechts: zoomen · Mitte/Alt+Ziehen: verschieben · F: alles zeigen · Esc: schließen") if overlay else
                           tr("Strg+V / hineinziehen: Bild, Text · Doppelklick: Text · Kamera: Ansicht · A: malen · B: Backdrop · ⤢: groß"))
        self.hint.setWordWrap(True)
        self._hint_text = self.hint.text()
        self.review_bar = self._make_review_bar(overlay)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 8)
        lay.addLayout(head)
        lay.addWidget(self.review_bar)
        lay.addWidget(self.view, 1)
        lay.addWidget(self.empty, 1)
        foot = QHBoxLayout()
        foot.setContentsMargins(18 if overlay else 10, 0, 18 if overlay else 10, 0)
        foot.addWidget(self.hint)
        lay.addLayout(foot)
        self._style()
        if overlay:
            host.installEventFilter(self)
            self.hide()
        else:
            self._show_empty(True)
            QApplication.instance().focusChanged.connect(self._focus_changed)

        # --- Appearance ------------------------------------------------------------
    def _style(self):
        paper, ink = QColor(WINDOW_BG), QColor(INK)
        alpha = BG_ALPHA if self.overlay else 255
        radius = 6 if self.overlay else 0
        self.setStyleSheet(
            f"QFrame#noteboard {{ background: {_rgba(paper, alpha)}; border: 1px solid {_rgba(ink, 60)}; "
            f"border-radius: {radius}px; }}"
            f"QLineEdit {{ background: transparent; color: {_rgba(ink)}; border: none; "
            f"border-bottom: 1px solid {_rgba(ink, 90)}; padding: 2px 0; font-size: {22 if self.overlay else 17}px; }}"
            f"QToolButton {{ background: transparent; color: {_rgba(ink)}; border: none; font-size: 16px; padding: 4px 8px; }}"
            f"QToolButton:hover {{ color: {_rgba(ui.ACCENT)}; }}"
            f"QToolButton:checked {{ color: {_rgba(ui.ACCENT)}; border-bottom: 2px solid {_rgba(ui.ACCENT)}; }}"
            f"QToolButton#swatch {{ padding: 4px 1px; }}"
            f"QLabel {{ color: {_rgba(ink, 140)}; background: transparent; font-size: 11px; }}")
        for item in self.gscene.items():
            if isinstance(item, TextBlock):
                item.setDefaultTextColor(INK)
        self._refresh_colors()

    def _refresh_colors(self):
        """Color buttons: dots in their color, the current one with a ring; the last shows the custom color (otherwise a
        color circle)."""
        current = self.pen_color()
        same = lambda a, b: (a is None) == (b is None) and (a is None or max(abs(x - y) for x, y in zip(a, b)) < 1e-3)
        for b, rgb in self.color_buttons:
            b.setIcon(_dot_icon(QColor.fromRgbF(*rgb) if rgb is not None else INK, same(rgb, current)))
        custom = self._custom_color
        self.custom_button.setIcon(_dot_icon(QColor.fromRgbF(*custom) if custom is not None else None,
                                             custom is not None and same(custom, current)))

    def pen_color(self):
        return getattr(self.vp.ws, "board_color", None) if self.vp is not None else None

    def set_pen_color(self, rgb, custom=False):
        """Pen color (shared by board and panel); turns the pen on."""
        if self.vp is None:
            return
        self.vp.ws.board_color = None if rgb is None else tuple(float(x) for x in rgb)
        if custom:
            self._custom_color = self.vp.ws.board_color
        self._refresh_colors()
        if self.paint_tool != "pen":
            self.set_paint_tool("pen")

    def _open_color_popup(self):
        from .colorpick import ColorPopup
        if self._color_popup is None:
            self._color_popup = ColorPopup(self)
            self._color_popup.picked.connect(lambda rgb, final: self.set_pen_color(rgb, custom=True))
        start = self._custom_color or self.pen_color() or (0.24, 0.23, 0.22)
        recent = list(getattr(self.vp.ws, "recent_colors", [])) if self.vp is not None else []
        self._color_popup.show_at(self.custom_button.mapToGlobal(self.custom_button.rect().bottomLeft()), start, recent)

        # --- Review draft (review.py) ----------------------------------------------------------------------------
    def _make_review_bar(self, overlay):
        bar = QFrame(objectName="reviewbar")
        row = QHBoxLayout(bar)
        row.setContentsMargins(18 if overlay else 10, 2, 12 if overlay else 6, 4)
        self.review_label = QLabel()
        self.review_label.setWordWrap(True)
        self.review_label.setStyleSheet("color: #b0602a;")
        row.addWidget(self.review_label, 1)
        self.review_play = QToolButton()
        self.review_play.setText(tr("▶ Anhören"))
        self.review_play.setToolTip(tr("Die Stelle der Review-Aufnahme abspielen, aus der dieser Entwurf stammt (nochmal: Stopp)"))
        self.review_play.clicked.connect(self._review_play)
        row.addWidget(self.review_play)
        accept = QToolButton()
        accept.setText(tr("✓ Übernehmen"))
        accept.setToolTip(tr("Entwurf durchgesehen: wird eine normale Notiz (Strg+Z nimmt es zurück)"))
        accept.clicked.connect(self._review_accept)
        row.addWidget(accept)
        self._review_player = self._review_audio = None
        self._review_stop = QTimer(self, singleShot=True)
        self._review_stop.timeout.connect(self._review_halt)
        bar.hide()
        return bar

    def _review_note(self):
        note = self.vp.scene.note_by_id(self.note_id) if self.vp is not None and self.note_id is not None else None
        return note if note is not None and note.review else None

    def _update_review_bar(self):
        note = self._review_note()
        self.review_bar.setVisible(note is not None)
        if note is None:
            self._review_halt()
            return
        from . import review
        r = note.review
        quote = (r.get("quote") or "").strip()
        short = quote if len(quote) <= 160 else quote[:157] + "…"
        self.review_label.setText(tr("Review-Entwurf · {0}").format(review.clock(r.get("t", 0)))
                                  + (f" · „{short}“" if short else ""))
        self.review_label.setToolTip(quote)
        self.review_play.setEnabled(os.path.isfile(self._review_audio_path(r)))

    @staticmethod
    def _review_audio_path(r):
        from . import review
        return os.path.join(review.reviews_dir(), r.get("review") or "-", review.AUDIO)

    def _review_play(self):
        note = self._review_note()
        if note is None:
            return
        if self._review_player is not None and self._review_stop.isActive():
            return self._review_halt()
        from PySide6.QtCore import QUrl
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
        r = note.review
        if self._review_player is None:
            self._review_player, self._review_audio = QMediaPlayer(self), QAudioOutput(self)
            self._review_player.setAudioOutput(self._review_audio)
        start = max(float(r.get("t", 0)) - 1.0, 0.0)
        end = float(r.get("t_end", float(r.get("t", 0)) + 8)) + 1.0
        player = self._review_player
        player.setSource(QUrl.fromLocalFile(self._review_audio_path(r)))
        player.setPosition(int(start * 1000))
        player.play()
        self._review_stop.start(int(max(end - start, 1.0) * 1000))
        self.review_play.setText(tr("■ Stopp"))

    def _review_halt(self):
        self._review_stop.stop()
        if self._review_player is not None:
            self._review_player.stop()
        self.review_play.setText(tr("▶ Anhören"))

    def _review_accept(self):
        note = self._review_note()
        if note is None:
            return
        self.commit()
        self.vp.scene.checkpoint()
        note.review = None
        self.vp._changed()
        self._update_review_bar()
        self.vp.notify(tr("{0}: Entwurf übernommen – Strg+Z nimmt es zurück").format(note.title or tr("Notiz")))

    def _show_empty(self, empty):
        if empty:
            self.review_bar.hide()
            self._review_halt()
        self.empty.setVisible(empty)
        self.view.setVisible(not empty)
        self.title.setEnabled(not empty)
        self.hint.setVisible(not empty)

    def eventFilter(self, obj, e):
        if obj is self.area and e.type() == QEvent.Resize:
            self._place()
        return False

    def _place(self):
        self.setGeometry(self.area.rect().adjusted(MARGIN, MARGIN, -MARGIN, -MARGIN))

    def _focus_changed(self, old, new):
        """Panel: if focus leaves the panel, the content lands in the note."""
        inside = lambda w: w is not None and (w is self or self.isAncestorOf(w))
        if inside(old) and not inside(new):
            self.commit()

        # --- Open / close ----------------------------------------------------
    def show_note(self, vp, nid):
        """Panel: show this note (None: empty). Changes made beforehand go into the previous note."""
        if self._settle_timer.isActive() and self.vp is not None:
            self.commit()  # waiting strokes belong to the previous note
        self.vp = vp
        note = vp.scene.note_by_id(nid) if nid is not None else None
        same = note is not None and note.id == self.note_id  # undo or similar: new state of the same note, view stays
        self.note_id = note.id if note is not None else None
        self._style()
        if note is None:
            self._clear()
            self.title.setText("")
            self.loaded, self.loaded_title = {}, ""
            self._show_empty(True)
            self.retake_button.hide()
            return
        self._show_empty(False)
        self._load(note, keep_view=same)

    def open(self, vp, nid):
        """Open overlay (maximize)."""
        if self.note_id is not None and self.note_id != nid:
            self.commit()
        self.vp = vp
        note = vp.scene.note_by_id(nid)
        if note is None:
            return
        self.note_id = nid
        self._show_empty(False)
        self._load(note)
        self._style()
        self._place()
        self.show()
        self.raise_()
        self.view.setFocus()

    def _clear(self):
        """Clear the board without a text writing the half-empty state into the note on losing focus."""
        was, self.committing = self.committing, True
        self.stop_video()
        self.view._end_hero_edit(apply=False)  # open name field belongs to an image that is about to disappear
        # references of the view to elements that are about to be deleted (otherwise every drawing fails afterwards)
        v = self.view
        v._flash, v._hover_shot, v._scrub, v._scaling = None, None, None, None
        self._settle_strokes = []
        try:
            self.gscene.clear()
        finally:
            self.committing = was

    def _load(self, note, keep_view=False):
        if self._settle_timer.isActive():  # strokes not yet adopted first into their note
            self.commit()
        transform = self.view.transform()
        center = self.view.mapToScene(self.view.viewport().rect().center())
        if self.view._ink is not None:
            self.view._ink_end()
        self.view._stroke = self.view._scaling = None
        self._clear()
        self.view.resetTransform()
        self.title.setText(note.title)
        _populate(self.gscene, self.vp.scene, note.board)
        if self.vp.scene.is_shot_note(note):
            self._attach_shot_rows()
        self.loaded, self.loaded_title = self.serialize(), note.title
        self._update_review_bar()
        if keep_view:
            self.view.setTransform(transform)
            self.view.centerOn(center)
        else:
            self.fit()
        self._update_retake()

    def _attach_shot_rows(self):
        """Shots: under every camera image its settings (ShotRow)."""
        for item in self.gscene.items():
            if isinstance(item, ImageItem) and item.shot:
                row = ShotRow(self, item)
                proxy = QGraphicsProxyWidget(item)
                proxy.setWidget(row)
                proxy.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                r = item.image_rect()
                proxy.setPos(r.left(), item.boundingRect().bottom() + 2 / max(item.scale(), 1e-3))
                proxy.setZValue(5)
                item.row = row

    def refresh_shot_rows(self):
        for item in self.gscene.items():
            if isinstance(item, ImageItem) and getattr(item, "row", None) is not None:
                item.row.refresh()

    def is_shots(self):
        return self.vp is not None and self.note_id is not None and \
            self.vp.scene.is_shot_note(self.vp.scene.note_by_id(self.note_id))

    def _shots(self):
        return [i for i in self.gscene.items(Qt.AscendingOrder) if isinstance(i, ImageItem) and i.view]

    def _missing_shots(self):
        return [i for i in self._shots() if i.missing]

    def _update_retake(self):
        shots, missing = self._shots(), len(self._missing_shots())
        self.retake_button.setVisible(bool(shots))
        self.retake_button.setToolTip(
            tr("{0} fehlende Screenshot(s) aus ihrer gespeicherten Kamera neu aufnehmen").format(missing) if missing else
            tr("Alle {0} Screenshots dieses Bretts neu aufnehmen – je Bild: gespeicherte Kamera, gleiches Seitenverhältnis (Striche passen wieder), X-Ray, Drahtgitter und Licht wie gerade eingestellt. Einzeln: ↻ oben rechts am Bild").format(len(shots)))

    def retake_missing(self):
        """Retake missing screenshots (image file lost) from their camera – if none is missing, all. Position and
        size in the board stay; one undo step."""
        shots = self._missing_shots() or self._shots()
        if self.vp is None or not shots:
            return
        for item in shots:
            self._retake(item)
        self.commit()
        self._update_retake()
        self.vp.notify(tr("{0} Screenshot(s) neu aufgenommen – Strg+Z nimmt es zurück").format(len(shots)))

    def retake_item(self, item, current=False):
        """↻ on a screenshot: retake, position and size in the board stay. current: from the current
        view of the viewport (camera, X-ray, light as set) instead of from the stored camera."""
        if self.vp is None or item.scene() is not self.gscene:
            return
        self._retake(item, current)
        self.commit()
        self._update_retake()
        self.vp.notify(tr("Screenshot neu aufgenommen") + (tr(" aus der jetzigen Ansicht") if current else "") +
                       tr(" – Strg+Z nimmt es zurück"))

    def _retake(self, item, current=False, look=None):
        """Replace the image of a screenshot: in the same aspect ratio as before (placeholder: like the viewport),
        same width in the board."""
        old = item.pixmap()
        aspect = None if item.missing else old.width() / max(old.height(), 1)
        cam = self.vp.scene.timing.camera_by_id(item.shot["cam"]) if item.shot else None
        pose = timing.pose_by_id(cam, item.shot["pose"]) if cam is not None else None
        if pose is not None:  # shots: through the camera at its pose (image format, focal length, drawings)
            data, view = self.vp.camera_jpeg(cam, pose.frame, pose=pose), timing.pose_view(pose)
        elif current:
            data, view = snapshot(self.vp, aspect)
        else:
            data, view = retake(self.vp, item.view, aspect, look), item.view
        pix = QPixmap()
        if not pix.loadFromData(QByteArray(data)):
            return
        name = hashlib.sha1(data).hexdigest()[:16] + ".jpg"
        self.vp.scene.assets[name] = data
        corner = item.sceneBoundingRect().topLeft()
        if not item.missing:  # same width as before (the new image may have more or fewer pixels)
            item.setScale(item.scale() * old.width() / max(pix.width(), 1))
        item.setPixmap(pix)
        item.setTransformOriginPoint(item.image_rect().center())
        if not item.missing:  # upper left corner stays: strokes on it sit at the same place again
            moved = item.sceneBoundingRect().topLeft()
            item.moveBy(corner.x() - moved.x(), corner.y() - moved.y())
        item.file, item.missing, item.view = name, False, view
        item.update()

    def undo(self, redo=False):
        """Ctrl+Z/Ctrl+Y in the board: same history as in the 3D view; the board shows the state of the note afterwards."""
        if self.vp is None or self.note_id is None:
            return
        self.commit()
        nid = self.note_id
        (self.vp.redo if redo else self.vp.undo)()
        note = self.vp.scene.note_by_id(nid)
        if note is None:  # the note itself was undone
            if self.overlay:
                self.note_id = None
                return self.close_board()
            return self.show_note(self.vp, None)
        if note.board != self.loaded or note.title != self.loaded_title:
            self._load(note, keep_view=True)

    def edited(self):
        return self.note_id is not None and (self.serialize() != self.loaded or
                                             self.title.text().strip() != (self.loaded_title or "").strip())

    def fit(self):
        rect = self.gscene.itemsBoundingRect()
        self.view.resetTransform()
        if rect.isEmpty():
            self.view.centerOn(0, 0)
            return
        self.view.fitInView(rect.adjusted(-60, -60, 60, 60), Qt.KeepAspectRatio)
        if self.view.transform().m11() > 1.0:  # little content: do not blow it up
            self.view.resetTransform()
            self.view.centerOn(rect.center())

    def _entry(self, item):
        """Stored entry of a board element (text, image/video, stroke, backdrop) or None."""
        if isinstance(item, TextBlock):
            text = item.toPlainText()
            if not text.strip():
                return None
            entry = {"kind": "text", "x": round(item.x(), 1), "y": round(item.y(), 1),
                     "w": round(item.textWidth(), 1), "text": text}
            if abs(item.scale() - 1.0) > 1e-4:
                entry["scale"] = round(item.scale(), 4)
            return entry
        if isinstance(item, ImageItem):
            entry = {"kind": "image", "x": round(item.x(), 1), "y": round(item.y(), 1),
                     "scale": round(item.scale(), 4), "file": item.file}
            if item.view:
                entry["view"] = item.view
            if item.shot:
                entry["shot"] = dict(item.shot)
            if item.hero:
                entry["hero"], entry["hero_name"] = True, item.hero_name
                if item.hero_prompt:
                    entry["hero_prompt"] = item.hero_prompt
            if item.caption:
                entry["caption"] = item.caption
                if item.caption_parts:
                    entry["caption_parts"] = list(item.caption_parts)
            if isinstance(item, VideoItem):
                if self.vp is not None:  # copied next to the scene on saving: from now on the copy
                    item.path = self.vp.scene.video_moved.get(item.path, item.path)
                entry["video"] = {"path": item.path, "duration": item.duration}
            return entry
        if isinstance(item, StrokeItem) and item.pts:
            return item.data()
        if isinstance(item, BackdropItem):
            return item.data()
        return None

    def serialize(self):
        follow_arrows(self.gscene)  # just moved (changed only comes later): arrows already at the new place
        items = [e for e in (self._entry(item) for item in self.gscene.items(Qt.AscendingOrder)) if e is not None]
        return {"items": items} if items else {}

        # --- Elements from board to board: copy, cut, paste (Global ↔ Local, Local ↔ Local) ---
    def copy_selection(self, cut=False):
        """Selected elements (images, videos, texts, strokes, backdrops) to the clipboard – as board entries
        (for every board of this scene), plus as image or text for other programs. cut: then remove here (one
        undo step). Returns whether anything was copied."""
        chosen = [i for i in self.gscene.items(Qt.AscendingOrder) if i.isSelected()]
        pairs = [(i, e) for i, e in ((i, self._entry(i)) for i in chosen) if e is not None]
        if not pairs:
            if self.vp is not None:
                self.vp.notify(tr("Kopieren: erst Bilder oder andere Elemente im Brett wählen"))
            return False
        mime = QMimeData()
        # the image data comes along (names are content hashes): pasting works in another scene or Mule window too
        files = {e["file"] for _, e in pairs if e.get("kind") == "image" and e.get("file")}
        assets = {f: base64.b64encode(data).decode("ascii") for f in sorted(files)
                  if self.vp is not None and (data := self.vp.scene.asset_bytes(f)) is not None}
        mime.setData(BOARD_MIME, json.dumps({"cut": bool(cut), "items": [e for _, e in pairs], "assets": assets})
                     .encode("utf-8"))
        images = [i for i, _ in pairs if isinstance(i, ImageItem)]
        if len(pairs) == 1 and images and not images[0].missing:
            mime.setImageData(images[0].pixmap().toImage())
        texts = [e["text"] for _, e in pairs if e["kind"] == "text"]
        if texts:
            mime.setText("\n\n".join(texts))
        QApplication.clipboard().setMimeData(mime)
        if cut:
            for item, _ in pairs:
                self.gscene.removeItem(item)
            self.commit()
        if self.vp is not None:
            self.vp.notify((tr("{0} Element(e) ausgeschnitten") if cut else tr("{0} Element(e) kopiert")).format(len(pairs))
                           + tr(" – in einem anderen Brett mit Strg+V einfügen"))
        return True

    def _paste_entries(self, clip, pos):
        """Paste copied board entries here: their arrangement stays, the upper left corner lies at pos; the
        pasted items are then selected (immediately movable). One undo step.
        Hero camera and shot binding belong to exactly one image: they only move along when cutting (and the
        shot binding only into a shots board) – a copy is an ordinary image."""
        entries = copy.deepcopy(clip.get("items") or [])
        moved = bool(clip.get("cut"))
        if self.vp is None or not entries:
            return
        for name, text in (clip.get("assets") or {}).items():  # images from another scene / Mule window
            if self.vp.scene.asset_bytes(name) is None:
                try:
                    self.vp.scene.assets[name] = base64.b64decode(text)
                except (ValueError, TypeError):
                    pass
        lost = [e for e in entries if e.get("kind") == "image" and self.vp.scene.asset_bytes(e.get("file")) is None]
        entries = [e for e in entries if not any(e is x for x in lost)]
        xs, ys = [], []
        for e in entries:
            pts = e.get("pts") if e.get("kind") == "stroke" else [[e.get("x", 0.0), e.get("y", 0.0)]]
            xs += [p[0] for p in pts]
            ys += [p[1] for p in pts]
        if entries:
            dx, dy = pos.x() - min(xs), pos.y() - min(ys)
            for e in entries:
                if e.get("kind") == "stroke":
                    e["pts"] = [[round(x + dx, 1), round(y + dy, 1)] for x, y in e["pts"]]
                else:
                    e["x"], e["y"] = round(e["x"] + dx, 1), round(e["y"] + dy, 1)
                if not moved:
                    for key in ("hero", "hero_name", "hero_prompt"):
                        e.pop(key, None)
                if not (moved and self.is_shots()):
                    e.pop("shot", None)
            before = set(self.gscene.items())
            _populate(self.gscene, self.vp.scene, {"items": entries})
            self.gscene.clearSelection()
            for item in self.gscene.items():
                if item not in before and isinstance(item, (TextBlock, ImageItem, StrokeItem, BackdropItem)):
                    item.setSelected(True)
            if self.is_shots():
                self._attach_shot_rows_new(before)
            self.commit()
            self._update_retake()
        if moved:  # cut is now pasted: pasting again is a copy
            again = QMimeData()
            again.setData(BOARD_MIME, json.dumps({"cut": False, "items": clip.get("items") or [],
                                                  "assets": clip.get("assets") or {}}).encode("utf-8"))
            QApplication.clipboard().setMimeData(again)
        text = tr("{0} Element(e) eingefügt").format(len(entries))
        if lost:
            text += tr(" – {0} Bild(er) fehlen in dieser Szene und wurden ausgelassen").format(len(lost))
        self.vp.notify(text)

    def _attach_shot_rows_new(self, before):
        """Shots board: pasted camera images get their settings (as on loading)."""
        for item in self.gscene.items():
            if item not in before and isinstance(item, ImageItem) and item.shot and getattr(item, "row", None) is None:
                row = ShotRow(self, item)
                proxy = QGraphicsProxyWidget(item)
                proxy.setWidget(row)
                proxy.setFlag(QGraphicsItem.ItemIgnoresTransformations)
                r = item.image_rect()
                proxy.setPos(r.left(), item.boundingRect().bottom() + 2 / max(item.scale(), 1e-3))
                proxy.setZValue(5)
                item.row = row

    def commit_soon(self, stroke=None):
        """After a stroke: after SETTLE_MS without a new stroke into the note (commit, light) and show what the last
        stroke connects. Every other commit takes waiting strokes along right away."""
        if stroke is not None:
            self._settle_strokes.append(stroke)
        self._settle_timer.start()

    def _settle(self):
        strokes, self._settle_strokes = self._settle_strokes, []
        self.commit(light=True)
        for st in reversed(strokes):
            if st.scene() is self.gscene:
                self.view.show_link(st)
                break

    def commit(self, burst=None, light=False):
        """Content and heading into the note (one undo step, only if something changed; burst: the same
        identifier within one second belongs to the same step). light: only board content changed (painting) – the
        3D view rebuilds nothing, only window title and other open boards of this note."""
        self._settle_timer.stop()
        if self.vp is None or self.note_id is None or self.committing:
            return
        note = self.vp.scene.note_by_id(self.note_id)
        if note is None:
            return
        board, title = self.serialize(), self.title.text().strip()
        if board != (note.board or {}) or (title and title != note.title):
            self.committing = True
            title_same = not title or title == note.title
            try:
                self.vp.scene.checkpoint(burst)
                note.board = board
                if title:
                    note.title = title
                self.loaded, self.loaded_title = board, note.title
                if self.vp.scene.is_shot_note(note):  # shots: deleted camera images take their pose along
                    tm = self.vp.scene.timing
                    timing.sync_boards(tm)
                    for v in self.vp.ws.viewports:
                        if v._look is not None and tm.camera_by_id(v._look) is None:
                            v.leave_camera()
                    self.vp.ws.timing_changed.emit()
                if light and title_same and not self.vp.scene.is_shot_note(note):
                    self._light_refresh(note)
                else:
                    self.vp._changed()
            finally:
                self.committing = False

    def _light_refresh(self, note):
        """After a light commit: star in the window title, other open boards of this note, note overlay."""
        win = self.vp.window()
        if hasattr(win, "_update_title"):
            win._update_title()
        for b in _open_boards(self.vp, note.id):
            if b is not self and not b.edited():
                b._load(note, keep_view=True)
        self.vp.update_view(all_views=True)

        # --- Video ---
    def _load_video(self, item):
        """item becomes the video of the player (always a new one: a player does not reliably switch files)."""
        from PySide6.QtCore import QUrl
        from PySide6.QtMultimedia import QMediaPlayer, QVideoSink
        self.stop_video()
        self._player, self._sink = QMediaPlayer(self), QVideoSink(self)
        self._player.setVideoSink(self._sink)
        self._player.setLoops(QMediaPlayer.Infinite)  # without QAudioOutput: muted
        self._sink.videoFrameChanged.connect(self._video_frame)
        self.video_item = item
        self._player.setSource(QUrl.fromLocalFile(os.path.abspath(item.path)))
        video.wait(lambda: self._player.duration() > 0, video.LOAD_MS)  # before that, seeking would go nowhere
        self._player.setPosition(item.position)

    def _video_frame(self, frame):
        item = self.video_item
        if item is None or self._player is None:
            return
        if item.scene() is not self.gscene:  # deleted while it is running
            return self.stop_video()
        image = frame.toImage()
        if not image.isNull():
            item.show_frame(image, self._player.position())

    def toggle_video(self, item):
        """Play/pause; another video stops the running one."""
        from PySide6.QtMultimedia import QMediaPlayer
        if item is not self.video_item:
            self._load_video(item)
        playing = self._player.playbackState() != QMediaPlayer.PlayingState
        (self._player.play if playing else self._player.pause)()
        item.playing = playing
        item.update()

    def seek_video(self, item, fraction):
        """Jump to a position and stop there (timeline)."""
        if item is not self.video_item:
            self._load_video(item)
        self._player.pause()
        item.playing = False
        item.position = int(item.duration * fraction)
        self._player.setPosition(item.position)
        item.update()

    def stop_video(self):
        """End playback (the element keeps its last image until the board is next loaded)."""
        player, sink, item = self._player, self._sink, self.video_item
        self._player, self._sink, self.video_item = None, None, None
        if player is None:
            return
        sink.videoFrameChanged.disconnect(self._video_frame)
        player.stop()
        player.setVideoSink(None)
        player.deleteLater()
        sink.deleteLater()
        if item is not None:
            item.playing = False

    def add_video(self, path, pos):
        """Insert a video file as an element with a preview image. None: not readable (message in the viewport)."""
        def fail(text):
            if self.vp is not None:
                self.vp.notify(text)
        if not video.available():
            return fail(tr("Videos brauchen PySide6-Addons – in dieser Installation nicht vorhanden."))
        frames, duration = video.read_frames(path, lambda d: [min(POSTER_MS, d * 0.1)])
        if not frames:
            return fail(tr("Video konnte nicht gelesen werden: {0}").format(os.path.basename(path)))
        data, ext = _encode(frames[0][1])
        name = hashlib.sha1(data).hexdigest()[:16] + ext
        self.vp.scene.assets[name] = data
        pix = QPixmap.fromImage(frames[0][1])
        item = VideoItem(pix, name, os.path.abspath(path), duration)
        item.position = frames[0][0]
        longest = max(pix.width(), pix.height())
        if longest > IMAGE_MAX:
            item.setScale(IMAGE_MAX / longest)
        item.setPos(pos)
        self.gscene.addItem(item)
        return item

    def video_still(self, item):
        """Image of the current position as its own image next to the video (caption: file and time, can be extended)."""
        frames, _ = video.read_frames(item.path, lambda d: [item.position], video.STILL_MAX)
        if not frames:
            return None
        rect = item.sceneBoundingRect()
        img = self.add_image(*_encode(frames[0][1]), QPointF(rect.right() + 24, rect.top()))
        if img is not None:
            img.setScale(rect.width() / max(img.pixmap().width(), 1))
            img.set_caption(f"{os.path.basename(item.path)} @ {video.time_label(item.position)}")
            self.commit()
        return img

    def close_board(self):
        self.stop_video()
        if not self.overlay:  # panel: Esc returns focus to the viewport (the content is adopted)
            self.commit()
            if self.vp is not None:
                self.vp.setFocus()
            return
        self.commit()
        self.set_paint_tool(None)
        nid, self.note_id = self.note_id, None
        self._clear()
        self.hide()
        panel = getattr(self.vp.ws, "note_panel", None) if self.vp is not None else None
        if panel is not None and panel.note_id == nid:
            panel.show_note(self.vp, nid)  # panel shows the new state
        if self.vp is not None:
            self.vp.setFocus()
            self.vp.update()

        # --- Painting ---------------------------------------------------------------------
    def set_paint_tool(self, tool):
        """Painting in the board: pen, eraser or off (None) – off: move images, texts, strokes."""
        self.paint_tool = tool
        self.paint_button.setChecked(tool is not None)
        self.view.setDragMode(QGraphicsView.NoDrag if tool else QGraphicsView.RubberBandDrag)
        self.view.viewport().setCursor(Qt.CrossCursor if tool == "pen" else
                                       Qt.BlankCursor if tool == "eraser" else Qt.ArrowCursor)
        self.view._cursor = None
        self.view.viewport().update()
        self.hint.setText(tr("Malen: Stift · am Strichende stillhalten: Gerade/Form · D: Radierer · Esc oder ✎: aus")
                          if tool == "pen" else
                          tr("Radierer · Strg+Shift: ganze Striche · A: Stift · Esc oder ✎: aus") if tool == "eraser" else
                          self._hint_text)
        if tool is not None:
            self.gscene.clearSelection()
            self.view.setFocus()

        # --- Insert content ---------------------------------------------------------
    def _free_spot(self, near):
        """Space next to existing content (to the right of it), otherwise near."""
        rect = self.gscene.itemsBoundingRect()
        return QPointF(rect.right() + 40, rect.top()) if not rect.isEmpty() else near

    def add_text(self, pos, text="", edit=False):
        block = TextBlock(text)
        block.setPos(pos)
        self.gscene.addItem(block)
        if edit:
            self.gscene.clearSelection()
            block.start_editing()
        return block

    def add_backdrop(self):
        """Backdrop around the selected elements (otherwise in the middle of the view); the title is immediately ready
        for writing."""
        if self.vp is None or self.note_id is None:
            return None
        chosen = [i for i in self.gscene.selectedItems() if _top_level(i)]
        if chosen:
            rect = chosen[0].sceneBoundingRect()
            for i in chosen[1:]:
                rect = rect.united(i.sceneBoundingRect())
            rect = rect.adjusted(-BACKDROP_PAD, -BACKDROP_PAD - BACKDROP_TITLE_PX * 1.8, BACKDROP_PAD, BACKDROP_PAD)
        else:
            c = self.view.mapToScene(self.view.viewport().rect().center())
            rect = QRectF(c.x() - 300, c.y() - 200, 600, 400)
        item = BackdropItem(rect, "Backdrop")
        self.gscene.addItem(item)
        self.gscene.clearSelection()
        self.commit()
        item.title.start_editing(select_all=True)
        return item

    def _backdrop_menu(self, at):
        """Right click on the backdrop button: auto backdrops."""
        menu = QMenu(self)
        menu.addAction(tr("Auto-Backdrops (Shift+B) – um alles, was nah beieinander liegt"), self.auto_backdrops)
        menu.exec(self.backdrop_button.mapToGlobal(at))

    def auto_backdrops(self):
        """Backdrops by the old principle (proximity, board_links.sections) around everything that is not yet in one – only around
        groups of at least two texts/images; title: the first text in it (for renaming). One undo step."""
        if self.vp is None or self.note_id is None:
            return []
        def frame(items):
            rect = items[0].sceneBoundingRect()
            for i in items[1:]:
                rect = rect.united(i.sceneBoundingRect())
            return rect.adjusted(-AUTO_PAD, -AUTO_PAD - BACKDROP_TITLE_PX * 1.8, AUTO_PAD, AUTO_PAD)

        groups = [sec["items"] for sec in board_links.sections(self.gscene)
                  if sec["title"] is None and len(sec["items"]) >= 2]  # not already in a backdrop, not alone
        def grabs(items, others):  # would the frame around items enclose an element of the other group?
            r = frame(items)
            return any(r.contains(o.sceneBoundingRect().center()) for o in others)

        merged = True
        while merged:  # interlocking groups become one backdrop – otherwise an image would end up in the wrong
            merged = False  # frame (only overlapping edges are harmless)
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if grabs(groups[i], groups[j]) or grabs(groups[j], groups[i]):
                        groups[i] = sorted(groups[i] + groups.pop(j), key=board_links._reading_key)
                        merged = True
                        break
                if merged:
                    break
        made = []
        for items in groups:
            first = next((i for i in items if isinstance(i, TextBlock)), None)
            item = BackdropItem(frame(items), board_links.label_of(first, 40) if first is not None else tr("Bilder"))
            self.gscene.addItem(item)
            made.append(item)
        if made:
            self.commit()
        return made

    def add_snapshot(self):
        """Camera button: image of the view (without controls) next to the content, remembers the camera. In shots: a
        new camera from this view (its image goes onto the board)."""
        if self.vp is None or self.note_id is None:
            return None
        if self.is_shots():
            return self.vp.new_timing_camera(self.vp.scene.note_by_id(self.note_id))
        data, view = snapshot(self.vp)
        img = self.add_image(data, ".jpg", self._free_spot(self.view.mapToScene(self.view.viewport().rect().center())),
                             view=view)
        if img is not None:
            self.gscene.clearSelection()
            img.setSelected(True)
            self.view.ensureVisible(img)
            self.commit()
            self._update_retake()
        return img

    def add_image(self, data, ext, pos, view=None):
        pix = QPixmap()
        if not pix.loadFromData(QByteArray(data)):
            return None
        name = hashlib.sha1(data).hexdigest()[:16] + ext
        self.vp.scene.assets[name] = bytes(data)
        img = ImageItem(pix, name, view)
        longest = max(pix.width(), pix.height())
        if longest > IMAGE_MAX:
            img.setScale(IMAGE_MAX / longest)
        img.setPos(pos)
        self.gscene.addItem(img)
        return img

    def _cursor_spot(self):
        local = self.view.mapFromGlobal(self.cursor().pos())
        if self.view.viewport().rect().contains(local):
            return self.view.mapToScene(local)
        return self._free_spot(self.view.mapToScene(self.view.viewport().rect().center()))

    def paste(self):
        self.insert_mime(QApplication.clipboard().mimeData(), self._cursor_spot())

    def insert_mime(self, mime, pos):
        """Paste/drag in: elements copied within a board (copy_selection), image files, image data (in the
        browser "Copy image"), otherwise an image link from the web (is loaded), otherwise text. One undo step."""
        if mime.hasFormat(BOARD_MIME):
            try:
                clip = json.loads(bytes(mime.data(BOARD_MIME)).decode("utf-8"))
            except ValueError:
                clip = None
            if isinstance(clip, dict):
                return self._paste_entries(clip, pos)
        placed = False
        if mime.hasUrls():
            for url in mime.urls():
                path = url.toLocalFile()
                if path and video.is_video(path) and os.path.isfile(path):
                    if self.add_video(path, pos) is not None:
                        pos = pos + QPointF(40, 40)
                    placed = True  # not readable: the message is enough, the path does not become text
                elif path and os.path.splitext(path)[1].lower() in IMAGE_EXT and os.path.isfile(path):
                    with open(path, "rb") as f:
                        data = f.read()
                    ext = os.path.splitext(path)[1].lower()
                    if self.add_image(data, ".jpg" if ext == ".jpeg" else ext, pos) is not None:
                        pos = pos + QPointF(40, 40)
                        placed = True
        if not placed and mime.hasImage():
            image = QImage(mime.imageData())
            if not image.isNull():
                placed = self.add_image(*_encode(image), pos) is not None
        if not placed:
            link = _image_link(mime)
            if link is not None:
                data = _download(link)
                placed = data is not None and self.add_image(data, _ext_of(data), pos) is not None
                if not placed and self.vp is not None:
                    self.vp.notify(tr("Bild konnte nicht geladen werden: {0}").format(link[:80]))
        if not placed and mime.hasText() and mime.text().strip():
            self.add_text(pos, mime.text())
        self.commit()

    @staticmethod
    def _missing_pixmap():
        pix = QPixmap(160, 110)
        pix.fill(QColor(0, 0, 0, 0))
        p = QPainter(pix)
        p.setPen(QPen(INK_DIM, 1, Qt.DashLine))
        p.drawRect(0, 0, 159, 109)
        p.drawText(pix.rect(), Qt.AlignCenter, tr("Bild fehlt"))
        p.end()
        return pix


def _encode(image):
    """(Bytes, extension) of a pasted image: without transparency as JPEG (photos from the web stay small in the
    scene file), otherwise PNG."""
    argb = image.convertToFormat(QImage.Format_ARGB32)
    alpha = np.frombuffer(argb.constBits(), np.uint8)[3::4]
    opaque = alpha.size == 0 or int(alpha.min()) == 255
    buf = QBuffer()
    buf.open(QIODevice.WriteOnly)
    if opaque:
        image.convertToFormat(QImage.Format_RGB32).save(buf, "JPG", SNAP_QUALITY)
    else:
        image.save(buf, "PNG")
    return bytes(buf.data()), ".jpg" if opaque else ".png"


def _image_link(mime):
    """Web address of an image in the clipboard (link, <img src> in the HTML or text), otherwise None."""
    candidates = [u.toString() for u in mime.urls()] if mime.hasUrls() else []
    if mime.hasHtml():
        candidates += re.findall(r"<img[^>]+src=[\"']([^\"']+)[\"']", mime.html(), re.I)
    if mime.hasText():
        candidates.append(mime.text().strip())
    for c in candidates:
        if re.match(r"https?://\S+$", c):
            return c.replace("&amp;", "&")
    return None


def _download(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Mule)"})
        with urllib.request.urlopen(req, timeout=WEB_TIMEOUT) as r:
            data = r.read()
    except (OSError, ValueError):
        return None
    image = QImage.fromData(data)
    if image.isNull():  # no image file (e.g. page instead of image)
        return None
    return data if _ext_of(data) else _encode(image)[0]


def _ext_of(data):
    """Extension for image data (JPEG/PNG/GIF/WebP/BMP stay as they are), otherwise ""."""
    heads = ((b"\xff\xd8", ".jpg"), (b"\x89PNG", ".png"), (b"GIF8", ".gif"), (b"BM", ".bmp"))
    for head, ext in heads:
        if data[:len(head)] == head:
            return ext
    return ".webp" if data[:4] == b"RIFF" and data[8:12] == b"WEBP" else ""


def set_hero_prompt(vp, hero, prompt):
    """Save an approved image prompt on the hero camera with this name (empty: remove) – one undo step.
    If its board is open, on the image there (otherwise the board would overwrite the note on the next save)."""
    scene = vp.scene
    for note in scene.notes + [scene.global_note]:
        items = note.board.get("items", [])
        k = next((k for k, i in enumerate(items) if i.get("kind") == "image" and i.get("hero")
                  and i.get("hero_name") == hero), None)
        if k is None:
            continue
        for board in (getattr(vp.window(), "_note_board", None), getattr(vp.ws, "note_panel", None)):
            if board is not None and board.note_id == note.id:
                shot = next((i for i in board.gscene.items() if isinstance(i, ImageItem) and i.hero
                             and i.hero_name == hero), None)
                if shot is not None:
                    shot.hero_prompt = prompt
                    board.commit()
                    return True
        scene.checkpoint()
        entry = dict(items[k])
        if prompt:
            entry["hero_prompt"] = prompt
        else:
            entry.pop("hero_prompt", None)
        note.board = {**note.board, "items": items[:k] + [entry] + items[k + 1:]}
        vp._changed()
        return True
    return False


def set_image_captions(vp, captions, parts=None):
    """Image captions {file: text} (file as in the brief, with or without "notes/"; empty text removes them together with
    parts) on every occurrence of the image in all notes – one undo step. parts {file: [object ids]}: the parts
    the image belongs to (hard link image ↔ component); files without an entry keep their parts. Open
    boards show it right away (otherwise the board would overwrite the note on the next save). -> files that are in
    no board."""
    scene = vp.scene
    wanted = {f.split("/")[-1]: (t or "").strip() for f, t in captions.items()}
    parts = {f.split("/")[-1]: list(dict.fromkeys(p)) for f, p in (parts or {}).items()}
    boards = [b for b in (getattr(vp.window(), "_note_board", None), getattr(vp.ws, "note_panel", None))
              if b is not None and b.note_id is not None]
    found, changed = set(), []
    for note in scene.notes + [scene.global_note]:
        items = note.board.get("items", [])
        hits = [k for k, i in enumerate(items) if i.get("kind") == "image" and i.get("file") in wanted]
        if not hits:
            continue
        found |= {items[k]["file"] for k in hits}
        open_boards = [b for b in boards if b.note_id == note.id]
        for b in open_boards:
            for img in b.gscene.items():
                if isinstance(img, ImageItem) and img.file in wanted:
                    img.set_caption(wanted[img.file])
                    if not img.caption:
                        img.caption_parts = []
                    elif img.file in parts:
                        img.caption_parts = parts[img.file]
        if open_boards:
            changed.append((note, open_boards[0].serialize(), open_boards))
        else:
            new = [dict(i) for i in items]
            for k in hits:
                f = new[k]["file"]
                if wanted[f]:
                    new[k]["caption"] = wanted[f]
                    if f in parts:
                        new[k].pop("caption_parts", None)
                        if parts[f]:
                            new[k]["caption_parts"] = parts[f]
                else:
                    new[k].pop("caption", None)
                    new[k].pop("caption_parts", None)
            changed.append((note, {**note.board, "items": new}, []))
    changed = [c for c in changed if c[1] != (c[0].board or {})]
    if changed:
        scene.checkpoint()
        for note, board, open_boards in changed:
            note.board = board
            for b in open_boards:
                b.loaded = board
        vp._changed()
    return sorted(set(wanted) - found)


DICTATE_GAP = 80.0  # distance of the dictated text to the image it points to


def _open_boards(vp, note_id=None):
    """Open boards (overlay, panel) – only those of this note, if note_id."""
    return [b for b in (getattr(vp.window(), "_note_board", None), getattr(vp.ws, "note_panel", None))
            if b is not None and b.note_id is not None and (note_id is None or b.note_id == note_id)]


def open_note_id(vp):
    """Note whose board is currently open (overlay before panel), otherwise None."""
    boards = _open_boards(vp)
    return boards[0].note_id if boards else None


def _dictated_entries(gscene, text, links):
    """Entries for a dictated text on the board in gscene: the text block (next to the first image it
    points to, otherwise to the right of everything) and per image an arrow from the text to the place uv. links: [(ImageItem, (u, v))]."""
    block = TextBlock(text)
    size = block.boundingRect().size()
    others = [i.sceneBoundingRect().adjusted(-20, -20, 20, 20) for i in gscene.items() if _top_level(i)]
    if links:
        r = links[0][0].sceneBoundingRect()
        pos = QPointF(r.right() + DICTATE_GAP, r.top())
    else:
        r = gscene.itemsBoundingRect()
        pos = QPointF(r.right() + 40, r.top()) if not r.isEmpty() else QPointF(0, 0)
    for _ in range(60):  # cover nothing: keep moving down
        rect = QRectF(pos, size)
        hit = next((o for o in others if o.intersects(rect)), None)
        if hit is None:
            break
        pos = QPointF(pos.x(), hit.bottom() + 1)
    rect = QRectF(pos, size)
    entries = [{"kind": "text", "x": round(pos.x(), 1), "y": round(pos.y(), 1), "w": float(TEXT_WIDTH), "text": text}]
    for img, (u, v) in links:
        ir = img.mapRectToScene(img.image_rect())
        tip = QPointF(ir.left() + ir.width() * u, ir.top() + ir.height() * v)
        a = _arrow_start(rect, tip)  # just next to the text, on the side facing the target
        entries.append({"kind": "stroke", "style": "pen", "w": [3.0, 3.0],
                        "pts": [[round(a.x(), 1), round(a.y(), 1)], [round(tip.x(), 1), round(tip.y(), 1)]],
                        "arrow": {"to": img.file, "uv": [round(u, 4), round(v, 4)]}})  # follows text and image
    return entries


def add_dictated_text(vp, note, text, images=()):
    """Place dictated (already processed) text as a text block on the board of the note, with arrows to images
    of this board – one undo step. images: [(file as in the brief, with or without "notes/", (u, v) or None =
    image center)]. If the board is open, it lands there (otherwise the board would overwrite the note on the next save).
    -> (number of arrows, [files that are not on this board])."""
    open_boards = _open_boards(vp, note.id)
    gscene = open_boards[0].gscene if open_boards else QGraphicsScene()
    if not open_boards:
        _populate(gscene, vp.scene, note.board)
    pics = {}
    for i in gscene.items(Qt.AscendingOrder):
        if isinstance(i, ImageItem):
            pics.setdefault(i.file, i)
    links, missing = [], []
    for f, uv in images:
        img = pics.get(f.replace("\\", "/").split("/")[-1])
        if img is None:
            missing.append(f)
        else:
            u, v = uv if uv else (0.5, 0.5)
            links.append((img, (min(max(float(u), 0.0), 1.0), min(max(float(v), 0.0), 1.0))))
    entries = _dictated_entries(gscene, text, links)
    if open_boards:
        _populate(gscene, vp.scene, {"items": entries})
        open_boards[0].commit()
        for b in open_boards[1:]:  # the same note open twice (overlay and panel)
            b._load(note, keep_view=True)
    else:
        gscene.clear()
        vp.scene.checkpoint()
        note.board = {**(note.board or {}), "items": list((note.board or {}).get("items", [])) + entries}
        vp._changed()
    return len(links), missing


def review_items(scene, gscene, text, shots):
    """Images of a review spot and the text for it as board entries: the images one below the other to the right of the
    content in gscene, the text next to it with one arrow per image to the place that was pointed at. shots: [(JPEG
    bytes, camera as camera_view or None, (u, v) or None[, caption])]. The images land in
    scene.assets (name from content and file type). -> (image entries, text entries) – the image entries then already
    lie in gscene."""
    r = gscene.itemsBoundingRect()
    x, y = (r.right() + 60, r.top()) if not r.isEmpty() else (0.0, 0.0)
    images = []
    for shot in shots:
        data, view = shot[0], shot[1]
        caption = shot[3] if len(shot) > 3 else None
        name = hashlib.sha1(data).hexdigest()[:16] + (_ext_of(data) or ".jpg")
        scene.assets[name] = data
        img = QImage.fromData(data)
        scale = min(1.0, IMAGE_MAX / max(img.width(), img.height(), 1))
        entry = {"kind": "image", "x": round(x, 1), "y": round(y, 1), "scale": round(scale, 4), "file": name}
        if view:
            entry["view"] = view
        if caption:
            entry["caption"] = caption
        images.append(entry)
        y += img.height() * scale + (60 if caption else 30)
    _populate(gscene, scene, {"items": images})
    by_file = {i.file: i for i in gscene.items(Qt.AscendingOrder) if isinstance(i, ImageItem)}
    links = []
    for entry, shot in zip(images, shots):
        uv = shot[2]
        if uv is not None and entry["file"] in by_file:
            u, v = uv
            links.append((by_file[entry["file"]], (min(max(float(u), 0.0), 1.0), min(max(float(v), 0.0), 1.0))))
    texts = _dictated_entries(gscene, text, links) if text.strip() else []
    return images, texts


def image_file_bytes(path):
    """Image file -> bytes as when dragging in (JPEG/PNG/GIF/WebP/BMP stay as they are; anything else Qt can read
    becomes JPEG or PNG) or None."""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    if _ext_of(data):
        return data if not QImage.fromData(data).isNull() else None
    image = QImage(path)
    return None if image.isNull() else _encode(image)[0]


def add_review_entries(vp, note, text, shots, checkpoint=True):
    """Review spot into the board of an existing note: images + text with arrows (review_items) – one undo step
    (checkpoint False: belongs to an already saved step, e.g. all drafts of a review at once).
    If the board is open, it lands there (otherwise the board would overwrite the note)."""
    open_boards = _open_boards(vp, note.id)
    if open_boards:
        board = open_boards[0]
        board.commit()
        _, texts = review_items(vp.scene, board.gscene, text, shots)
        _populate(board.gscene, vp.scene, {"items": texts})
        board.commit()
        for b in open_boards[1:]:
            b._load(note, keep_view=True)
        return
    gscene = QGraphicsScene()
    _populate(gscene, vp.scene, note.board or {})
    images, texts = review_items(vp.scene, gscene, text, shots)
    gscene.clear()
    if checkpoint:
        vp.scene.checkpoint()
    note.board = {**(note.board or {}), "items": list((note.board or {}).get("items", [])) + images + texts}
    vp._changed()


def edit_note_text(vp, note, old, new, title=None):
    """Change text on the board of a note (AI on explicit request): old must occur exactly once in exactly one
    text block and is replaced by new (empty = delete; an empty block disappears); title: new
    heading. One undo step; if the board is open, there (otherwise it would overwrite the note). -> old block text
    or None (heading only). ValueError if old is missing or ambiguous."""
    old = old or ""
    for b in _open_boards(vp, note.id)[:1]:
        b.commit()  # whatever is on the board right now counts
    if old:
        texts = [i.get("text", "") for i in (note.board or {}).get("items", []) if i.get("kind") == "text"]
        n = sum(t.count(old) for t in texts)
        if n != 1:
            raise ValueError(("not found" if n == 0 else f"found {n} times – quote more so it is unique")
                             + f" on this board: {old!r}")
    open_boards = _open_boards(vp, note.id)
    if open_boards:
        b = open_boards[0]
        before = None
        if old:
            block = next(i for i in b.gscene.items() if isinstance(i, TextBlock) and old in i.toPlainText())
            before = block.toPlainText()
            block.setPlainText(before.replace(old, new))
            if not block.toPlainText().strip():
                b.gscene.removeItem(block)
        if title is not None and title.strip():
            b.title.setText(title.strip())
        b.commit()
        for o in open_boards[1:]:
            o._load(note, keep_view=True)
        return before
    vp.scene.checkpoint()
    before, items = None, []
    for i in (note.board or {}).get("items", []):
        if old and before is None and i.get("kind") == "text" and old in i.get("text", ""):
            before = i["text"]
            i = {**i, "text": before.replace(old, new)}
            if not i["text"].strip():
                continue
        items.append(i)
    note.board = {**(note.board or {}), "items": items}
    if title is not None and title.strip():
        note.title = title.strip()
    vp._changed()
    return before


def board_for(vp):
    host = vp.window()  # maximize: the whole program window
    board = getattr(host, "_note_board", None)
    if board is None:
        board = host._note_board = NoteBoard(host, overlay=True)
    return board


BOARD_MIME = "application/x-mule-board"  # clipboard: board entries (NoteBoard.copy_selection)


def _populate(gscene, scene, board):
    """Content of a board (stored entries) into a QGraphicsScene: texts, images, painted strokes."""
    arrows = []
    _fill(gscene, scene, board, arrows)
    for a in arrows:  # only when everything is placed: find text and image of the Mule arrows
        a._bind(gscene)
        a.follow()


def _fill(gscene, scene, board, arrows):
    for item in board.get("items", []):
        kind = item.get("kind")
        if kind == "text":
            block = TextBlock(item.get("text", ""))
            block.setTextWidth(float(item.get("w", TEXT_WIDTH)))
            block.setScale(float(item.get("scale", 1.0)))
            block.setPos(float(item["x"]), float(item["y"]))
            gscene.addItem(block)
        elif kind == "image":
            data = scene.asset_bytes(item["file"])
            pix = QPixmap()
            missing = data is None or not pix.loadFromData(QByteArray(data))
            if missing:
                pix = NoteBoard._missing_pixmap()
            if item.get("video"):
                img = VideoItem(pix, item["file"], item["video"]["path"], item["video"].get("duration", 0), missing)
            else:
                img = ImageItem(pix, item["file"], item.get("view"), missing, item.get("hero", False),
                                item.get("hero_name", ""))
            img.hero_prompt = item.get("hero_prompt", "")
            img.shot = dict(item["shot"]) if item.get("shot") else None
            if item.get("caption"):
                img.set_caption(item["caption"])
                img.caption_parts = list(item.get("caption_parts", []))
            img.setScale(float(item.get("scale", 1.0)))
            img.setPos(float(item["x"]), float(item["y"]))
            gscene.addItem(img)
        elif kind == "stroke" and item.get("pts"):
            color = tuple(item["color"]) if item.get("color") else None
            pts, widths = [QPointF(x, y) for x, y in item["pts"]], item.get("w", [2.0] * len(item["pts"]))
            tie = item.get("arrow")
            if tie and len(pts) >= 2:
                arrow = ArrowItem(pts, widths, tie.get("to", ""), tie.get("uv", (0.5, 0.5)), color)
                arrows.append(arrow)
                gscene.addItem(arrow)
            else:
                gscene.addItem(StrokeItem(pts, widths, item.get("style", "pen"), color))
        elif kind == "backdrop":
            gscene.addItem(BackdropItem(QRectF(float(item["x"]), float(item["y"]), float(item["w"]), float(item["h"])),
                                        item.get("title", "")))


def render_board(scene, board, max_px=2400):
    """Board as one image (images, painting, text on the board background) – for the brief export, so that the
    AI sees the annotations. None: empty."""
    gscene = QGraphicsScene()
    _populate(gscene, scene, board)
    rect = gscene.itemsBoundingRect()
    if rect.isEmpty():
        return None
    rect = rect.adjusted(-24, -24, 24, 24)
    img = render_region(gscene, rect, min(2.0, max_px / max(rect.width(), rect.height())))
    gscene.clear()
    return img


def render_region(gscene, rect, k, badges=()):
    """Section rect of a board as an image, k pixels per board unit; badges: [(board point, text)] – ring and
    tag at the place (brief: P1, P2 … at the arrow tips)."""
    img = QImage(max(int(rect.width() * k), 1), max(int(rect.height() * k), 1), QImage.Format_RGB32)
    img.fill(QColor(WINDOW_BG))
    p = QPainter(img)
    p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.TextAntialiasing)
    gscene.render(p, QRectF(0, 0, img.width(), img.height()), rect)
    font = note_font(15)
    font.setBold(True)
    p.setFont(font)
    for pt, text in badges:
        at = QPointF((pt.x() - rect.left()) * k, (pt.y() - rect.top()) * k)
        p.setPen(QPen(LINK_COLOR, 2.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(at, 9, 9)
        box = QRectF(at.x() + 8, at.y() + 6, p.fontMetrics().horizontalAdvance(text) + 12, 22)
        p.setPen(Qt.NoPen)
        p.setBrush(LINK_COLOR)
        p.drawRoundedRect(box, 4, 4)
        p.setPen(QColor(20, 24, 26))
        p.drawText(box, Qt.AlignCenter, text)
    p.end()
    return img


def insert_snapshot(vp, nid):
    """Camera button on the note in the viewport: image of the view into its board. If it is currently open (panel or
    maximized), it goes in there; otherwise directly into the note (one undo step, the panel reloads)."""
    for board in (getattr(vp.window(), "_note_board", None), getattr(vp.ws, "note_panel", None)):
        if board is not None and board.note_id == nid and (board.overlay is False or board.isVisible()):
            return board.add_snapshot()
    note = vp.scene.note_by_id(nid)
    if note is None:
        return None
    data, view = snapshot(vp)
    name = hashlib.sha1(data).hexdigest()[:16] + ".jpg"
    vp.scene.assets[name] = data
    img = QImage.fromData(data)
    scale = min(1.0, IMAGE_MAX / max(img.width(), img.height(), 1))
    right, top = _content_right_top(vp.scene, note.board.get("items", []))
    vp.scene.checkpoint()
    note.board = {"items": list(note.board.get("items", [])) + [
        {"kind": "image", "x": round(right + 40 if right is not None else 0.0, 1), "y": round(top or 0.0, 1),
         "scale": round(scale, 4), "file": name, "view": view}]}
    vp.ws.active_note = nid
    vp._changed()
    return name


def _content_right_top(scene, items):
    """Right edge and top edge of the content of a board (from the stored entries) – or (None, None)."""
    right = top = None
    for it in items:
        x, y = float(it.get("x", 0)), float(it.get("y", 0))
        if it.get("kind") == "image":
            data = scene.asset_bytes(it["file"])
            size = QImage.fromData(data).size() if data else QSize(160, 110)
            w = size.width() * float(it.get("scale", 1.0))
        elif it.get("kind") == "stroke":
            pts = it.get("pts") or [[x, y]]
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            x, y, w = min(xs), min(ys), max(xs) - min(xs)
        else:
            w = float(it.get("w", TEXT_WIDTH))
        right = x + w if right is None else max(right, x + w)
        top = y if top is None else min(top, y)
    return right, top


def open_boards(vp, nid):
    """Boards currently showing this note (Note panel, Shots panel, maximized board)."""
    boards = (getattr(vp.window(), "_note_board", None), getattr(vp.ws, "note_panel", None),
              getattr(vp.ws, "shots_panel", None))
    return [b for b in boards if b is not None and b.note_id == nid and b.vp is not None]


def shot_edit(vp, nid, fn, checkpoint=True):
    """Shots: change the board of a shot note from outside (new camera, pose, image) – fn(entries) changes the list of
    board entries (dicts) and the camera data in the process. Open boards adopt their state beforehand and show the new
    one afterwards; one undo step (checkpoint=False: the caller has already set it)."""
    boards = open_boards(vp, nid)
    for b in boards:
        b.commit()
    note = vp.scene.note_by_id(nid)
    if note is None:
        return None
    if checkpoint:
        vp.scene.checkpoint()
    items = [dict(i) for i in note.board.get("items", [])]
    result = fn(items)
    note.board = {**note.board, "items": items} if items else {}
    timing.sync_boards(vp.scene.timing)
    for b in boards:
        b._load(note, keep_view=True)
    return result


def shot_image(vp, cam, pose):
    """(File name, width, height) of a fresh image through the camera at its pose – in scene.assets."""
    data = vp.camera_jpeg(cam, pose.frame, pose=pose)
    name = hashlib.sha1(data).hexdigest()[:16] + ".jpg"
    vp.scene.assets[name] = data
    img = QImage.fromData(data)
    return name, img.width(), img.height()


def open_board(vp, nid):
    for panel in (getattr(vp.ws, "note_panel", None), getattr(vp.ws, "shots_panel", None)):
        if panel is not None and panel.note_id == nid:
            panel.commit()  # changed in the panel first into the note
    board_for(vp).open(vp, nid)
