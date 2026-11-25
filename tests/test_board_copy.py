"""Elements from board to board: copy or cut several images (and texts, strokes) and paste them into another
board – Global ↔ Local, Local ↔ Local. Layout stays, the pasted items are selected, one undo step each."""
import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QPointF, Qt
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QApplication

from sketch_machine.note_board import BOARD_MIME, ImageItem, TextBlock, board_for
from sketch_machine.scene import Note
from tests.harness import H
from tests.test_notes import _close, _note_scene


def _png(color):
    img, buf = QImage(120, 80, QImage.Format_RGB32), QBuffer()
    img.fill(QColor(*color))
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    return bytes(buf.data())


def _filled_board(h):
    """Local note with two images (the second a hero camera) and one text; plus a second local note."""
    note = _note_scene(h)
    other = Note("Lager", np.array([-0.5, 0.8, 0.0]))
    h.sc.notes.append(other)
    board = board_for(h.v)
    board.open(h.v, note.id)
    h.pump(0.05)
    a = board.add_image(_png((200, 60, 60)), ".png", QPointF(0, 0))
    b = board.add_image(_png((60, 60, 200)), ".png", QPointF(300, 150))
    b.view = {"target": [0.0, 0.0, 0.0], "dist": 5.0, "rot": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}  # screenshot
    b.hero, b.hero_name = True, "Totale"
    board.add_text(QPointF(0, 300), "Kran")
    board.commit()
    return note, other, board, (a, b)


def _images(board):
    return [i for i in board.gscene.items(Qt.AscendingOrder) if isinstance(i, ImageItem)]


def test_copy_several_images_local_to_global(h: H):
    note, other, board, (a, b) = _filled_board(h)
    try:
        board.gscene.clearSelection()
        a.setSelected(True)
        b.setSelected(True)
        assert board.copy_selection()
        assert QApplication.clipboard().mimeData().hasFormat(BOARD_MIME)
        assert len(_images(board)) == 2, "copying leaves the board as it is"
        board.open(h.v, h.sc.global_note.id)  # the “Global” panel is the same board with the project note
        h.pump(0.05)
        assert not _images(board)
        undo = len(h.sc._undo)
        board.insert_mime(QApplication.clipboard().mimeData(), QPointF(1000, 500))
        got = _images(board)
        assert len(got) == 2 and {i.file for i in got} == {a.file, b.file}, "both images in the global board"
        pos = sorted((i.x(), i.y()) for i in got)
        assert pos[0] == (1000.0, 500.0) and pos[1] == (1300.0, 650.0), "layout stays, top left corner at the target"
        assert all(i.isSelected() for i in got), "the pasted items are selected"
        assert not any(i.hero for i in got), "a copy is not a second hero camera"
        assert len(h.sc._undo) == undo + 1, "one undo step"
        assert len([e for e in h.sc.global_note.board["items"] if e["kind"] == "image"]) == 2, "stored in the note"
        assert len([e for e in h.sc.note_by_id(note.id).board["items"] if e["kind"] == "image"]) == 2, "source untouched"
    finally:
        _close(h, board)


def test_cut_and_paste_into_another_local_note(h: H):
    note, other, board, (a, b) = _filled_board(h)
    try:
        board.gscene.clearSelection()
        for item in board.gscene.items():
            if isinstance(item, (ImageItem, TextBlock)):
                item.setSelected(True)
        assert board.copy_selection(cut=True)
        assert not h.sc.note_by_id(note.id).board, "cut: the source is empty"
        board.open(h.v, other.id)
        h.pump(0.05)
        board.insert_mime(QApplication.clipboard().mimeData(), QPointF(0, 0))
        items = h.sc.note_by_id(other.id).board["items"]
        assert sorted(e["kind"] for e in items) == ["image", "image", "text"], "images and text arrived"
        assert any(e.get("hero") and e.get("hero_name") == "Totale" for e in items), "cut: hero camera moves along"
        board.insert_mime(QApplication.clipboard().mimeData(), QPointF(900, 0))  # pasting again = copy
        items = h.sc.note_by_id(other.id).board["items"]
        assert len(items) == 6 and sum(1 for e in items if e.get("hero")) == 1, "second paste: copy without hero"
        h.sc.undo()
        h.sc.undo()
        h.sc.undo()
        assert len(h.sc.note_by_id(note.id).board["items"]) == 3, "undo brings the cut items back"
    finally:
        _close(h, board)


def test_keyboard_shortcuts_in_board(h: H):
    from PySide6.QtTest import QTest
    note, other, board, (a, b) = _filled_board(h)
    try:
        board.gscene.clearSelection()
        a.setSelected(True)
        board.view.setFocus()
        QTest.keyClick(board.view, Qt.Key_C, Qt.ControlModifier)
        mime = QApplication.clipboard().mimeData()
        assert mime.hasFormat(BOARD_MIME) and mime.hasImage(), "one image: also as an image for other programs"
        board.open(h.v, other.id)
        h.pump(0.05)
        board.view.setFocus()
        QTest.keyClick(board.view, Qt.Key_V, Qt.ControlModifier)
        assert len(_images(board)) == 1, "Ctrl+V pastes the copied image"
    finally:
        _close(h, board)


def test_copy_images_into_another_scene(h: H):
    """Board elements into another scene (or Mule window): the image data travels in the clipboard."""
    from sketch_machine.scene import Scene
    note, other, board, (a, b) = _filled_board(h)
    try:
        board.gscene.clearSelection()
        a.setSelected(True)
        assert board.copy_selection()
        _close(h, board)
        h.v.set_scene(Scene())  # new scene: no image data of the old one
        board = board_for(h.v)
        board.open(h.v, h.sc.global_note.id)
        h.pump(0.05)
        board.insert_mime(QApplication.clipboard().mimeData(), QPointF(0, 0))
        got = _images(board)
        assert len(got) == 1 and got[0].file == a.file and not got[0].missing
        assert h.sc.asset_bytes(a.file) is not None, "the image is stored in the new scene"
    finally:
        _close(h, board)
