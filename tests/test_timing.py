"""Shots (prototype, internally “Timing”): flipbook planes with drawings per frame, shot notes like notes (Shots panel),
cameras as images on their boards, saving/undo, frame changes without rebuilding the draw lists."""
import os

import numpy as np
from PySide6.QtCore import Qt

from tests.harness import H, OUT
from sketch_machine import timing


def _flip_scene(h: H):
    """New scene in Timing with a flipbook plane in front of the camera -> plane."""
    h.new_scene()
    h.camera((0, 0, 0), 12.0, (0, 0, 1))
    h.v.set_timing_mode(True)
    h.v.set_draw_tool("brush")
    h.v.new_plane()
    plane = h.sc.active_plane
    assert plane.flipbook, "Space+N in Timing creates a flipbook plane"
    h.pump(0.05)
    return plane


def _on_plane(plane, pts2d):
    return np.array([plane.origin + x * plane.axis_u + y * plane.axis_v for x, y in pts2d])


def _draw(h: H, plane, frame, pts2d):
    h.v.set_frame(frame)
    before = {c.id for c in h.sc.curves}
    h.stroke_world(_on_plane(plane, pts2d))
    new = [c for c in h.sc.curves if c.id not in before]
    assert new, f"no stroke was created at {frame}"
    return new


def _leave(h: H):
    if h.v.ws.timing_mode:
        h.v.set_select_mode(False)


def _shot_note(h: H, plane=None, title="Shot 1", text=""):
    """Shot note on the flipbook plane (leader line to the plane), clicked -> note."""
    from sketch_machine.scene import Note, NoteAnchor
    anchors = [NoteAnchor(plane.origin.copy())] if plane is not None else []
    board = {"items": [{"kind": "text", "x": 0.0, "y": 0.0, "w": 340.0, "text": text}]} if text else {}
    note = Note(title, (plane.origin if plane is not None else np.zeros(3)) + np.array([0, 2.0, 0]), anchors=anchors,
                board=board)
    h.sc.timing.notes.append(note)
    h.v.ws.active_note = note.id
    h.v._changed()
    return note


def _add_pose(h: H, cam, frame, state, mm=None):
    """Pose with its image on the board (like ➜, without looking through the camera)."""
    from sketch_machine.note_board import shot_edit, shot_image
    from sketch_machine.vp.timing_view import _shot_entry

    def edit(items):
        p = timing.add_pose(cam, frame, state)
        if mm is not None:
            p.mm = mm
        name, _w, _h = shot_image(h.v, cam, p)
        items.append(_shot_entry(cam, p, name, 500.0 * len(cam.poses), 0.0, 0.3))
        return p
    return shot_edit(h.v, h.sc.timing.note_of(cam).id, edit)


def test_drawings_hold_until_the_next(h: H):
    """Painting at a frame creates a drawing there; it holds until the next drawing of the same plane."""
    plane = _flip_scene(h)
    try:
        a = _draw(h, plane, 1010, [(-2, 0), (0, 1), (2, 0)])
        b = _draw(h, plane, 1015, [(-2, -1), (0, -2), (2, -1)])
        assert all(c.frame == 1010 and c.paint for c in a) and all(c.frame == 1015 for c in b)
        assert timing.flip_keys(h.sc)[plane.id] == [1010, 1015]
        shown = lambda c: h.v._layer_visible(c)
        for frame, want_a, want_b in ((1001, False, False), (1010, True, False), (1013, True, False), (1015, False, True),
                                      (1090, False, True)):
            h.v.set_frame(frame)
            assert shown(a[0]) == want_a and shown(b[0]) == want_b, f"at {frame}"
        # visible in the image: at 1001 the plane is empty, at 1012 it is not
        h.sc.timing.onion = False
        h.v.set_frame(1001)
        empty = h.grab()
        h.v.set_frame(1012)
        drawn = h.grab()
        assert h.diff(empty, drawn) > 50, "the drawing of 1010 applies at 1012"
        h.v.set_frame(1003)
        assert h.diff(empty, h.grab()) == 0, "before the first drawing the plane is empty"
    finally:
        _leave(h)


def test_frame_change_does_not_rebuild_draw_lists(h: H):
    """Performance: a frame change only selects ranges in the flipbook buffer – no recomputation of the stroke list,
    no new upload."""
    plane = _flip_scene(h)
    try:
        for k, frame in enumerate(range(1002, 1030, 3)):
            _draw(h, plane, frame, [(-2 + k * 0.2, 0), (0, 1 + k * 0.1), (2, 0)])
        paint = lambda: h.v.gpu_grab()  # a frame like paintGL, without forcing a rebuild
        h.v.set_frame(1001)
        h.grab()  # build everything once
        paint()
        r = h.v.renderer
        calls = []
        orig = h.v._scene_stroke_list
        h.v._scene_stroke_list = lambda: calls.append(1) or orig()
        rev, key = r._stroke_rev, r._flip_key
        try:
            for frame in range(1001, 1040):
                h.v.set_frame(frame)
                paint()
                assert r._flip_jobs, f"at {frame} the applicable drawing is drawn"

        finally:
            del h.v._scene_stroke_list
        assert not calls, f"stroke list rebuilt {len(calls)}×"
        assert r._stroke_rev == rev, f"strokes uploaded again ({rev} -> {r._stroke_rev})"
        assert r._flip_key is key, "flipbook buffer prepared again"
    finally:
        _leave(h)


def test_flipbook_locked_outside_timing(h: H):
    """Outside Timing: flipbook plane cannot be activated, its drawings cannot be erased; in Timing painting happens only on
    flipbook planes."""
    plane = _flip_scene(h)
    a = _draw(h, plane, 1005, [(-2, 0), (2, 0)])
    _leave(h)
    assert not h.sc.active_plane.flipbook, "on leaving, the previous drawing surface becomes active again"
    h.v._activate(plane)
    assert h.sc.active_plane is not plane
    h.v.set_draw_tool("eraser")
    pts = h.scr(_on_plane(plane, [(0, 0)])[0])
    h.v.set_frame(1005)
    from PySide6.QtCore import QPointF
    h.stroke([QPointF(pts.x() - 5, pts.y() - 20), QPointF(pts.x() + 5, pts.y() + 20)])
    assert all(any(c is x for x in h.sc.curves) for c in a), "eraser outside Timing left the flipbook alone"
    h.v.set_draw_tool("pen")
    # in Timing on a normal plane: no stroke
    h.v.set_timing_mode(True)
    try:
        h.sc.activate(h.sc.presets[1])
        n = len(h.sc.curves)
        h.stroke_world(np.array([[-1, 0, 0], [1, 0.5, 0]]))
        assert len(h.sc.curves) == n, "in Timing one does not paint on normal planes"
    finally:
        _leave(h)


def test_undo_of_a_new_drawing(h: H):
    plane = _flip_scene(h)
    try:
        _draw(h, plane, 1020, [(-1, 0), (1, 0)])
        assert timing.flip_keys(h.sc)[plane.id] == [1020]
        h.v.undo()
        assert timing.flip_keys(h.sc)[plane.id] == []
        assert h.sc.timing.frame == 1020, "undo does not change the frame"
    finally:
        _leave(h)


def test_continue_painting_erase_blank_drawing_move(h: H):
    """Painting/erasing at a frame without a drawing first copies the applicable one there (the earlier one stays); blank
    drawing, move, delete."""
    plane = _flip_scene(h)
    tl = h.w.timeline
    try:
        a = _draw(h, plane, 1010, [(-2, 0), (2, 0)])
        b = _draw(h, plane, 1013, [(-2, 1.5), (2, 1.5)])
        assert timing.flip_keys(h.sc)[plane.id] == [1010, 1013]
        at13 = [c for _, c in timing.key_curves(h.sc, plane.id, 1013)]
        assert len(at13) == len(b) == len(a) + 1, "the pose of 1010 is still there at 1013 (continue painting)"
        assert np.allclose(at13[0].points, a[0].points) and at13[0] is not a[0]
        assert [c for _, c in timing.key_curves(h.sc, plane.id, 1010)] == a, "1010 stays as it was"
        # erase at 1016 (1013 applies there): copy to 1016, erased in it
        h.v.set_frame(1016)
        h.v.set_draw_tool("eraser")
        from PySide6.QtCore import QPointF
        mid = h.scrf(_on_plane(plane, [(0, 0)])[0])
        h.stroke([QPointF(mid.x() - 3, mid.y() - 25), QPointF(mid.x() + 3, mid.y() + 25)])
        h.v.set_draw_tool("brush")
        assert timing.flip_keys(h.sc)[plane.id] == [1010, 1013, 1016]
        assert len(timing.key_curves(h.sc, plane.id, 1013)) == len(at13), "1013 stays intact when erasing at 1016"
        assert len(timing.key_curves(h.sc, plane.id, 1016)) == len(at13) + 1, "at 1016 the lower line is cut in two"
        h.v.set_frame(1020)
        tl._blank()
        assert timing.flip_keys(h.sc)[plane.id] == [1010, 1013, 1016, 1020]
        h.v.set_frame(1022)
        assert not any(h.v._layer_visible(c) for c in h.sc.curves if c.frame is not None), "after the blank drawing: nothing"
        timing.move_key(h.sc, plane, 1013, 1014)
        h.v._changed()
        assert timing.flip_keys(h.sc)[plane.id] == [1010, 1014, 1016, 1020]
        h.v.set_frame(1012)
        tl._delete_key()  # at 1012, 1010 applies
        assert timing.flip_keys(h.sc)[plane.id] == [1014, 1016, 1020]
    finally:
        _leave(h)


def test_clicking_camera_in_viewport(h: H):
    """Clicking the camera symbol selects camera and note, double-click looks through it, a faded pose jumps to its frame."""
    plane = _flip_scene(h)
    tm = h.sc.timing
    try:
        note = _shot_note(h, plane)
        h.camera((0, 0, 0), 10.0, (0.6, 0.3, 0.75))
        cam = h.v.new_timing_camera(note)
        _add_pose(h, cam, 1030, (cam.poses[0].target, cam.poses[0].dist * 1.5, cam.poses[0].rot))
        h.v.ws.timing_cam = h.v.ws.active_note = None
        h.camera((0, 0, 0), 40.0, (0.2, 0.5, 0.85))
        h.v.set_frame(1001)
        eye, corners = timing.frustum(cam.poses[0], timing.base_size(cam.poses[0]))
        at = h.scrf((corners[0] + corners[1]) / 2)
        h.click(at)
        assert h.v.ws.timing_cam == cam.id and h.v.ws.active_note == note.id, "click selects camera and its note"
        eye2, corners2 = timing.frustum(cam.poses[1], timing.base_size(cam.poses[1]))
        h.click(h.scrf((corners2[1] + corners2[2]) / 2))
        assert tm.frame == 1030, "click on the faded pose jumps to its frame"
        h.dclick(h.scrf((corners2[1] + corners2[2]) / 2))
        assert h.v._look == cam.id
        assert not any(c.frame is not None for c in h.sc.curves), "clicks on cameras paint nothing"
        h.v.leave_camera()
    finally:
        _leave(h)


def test_camera_poses_only_from_inside(h: H):
    """Camera from the view; ⟲ changes the applicable pose (no move), ➜ creates a new one from the frame on; when
    looking through, the viewport jumps along; new poses inherit the focal length."""
    _flip_scene(h)
    tm = h.sc.timing
    try:
        h.v.set_frame(1001)
        cam = h.v.new_timing_camera(tm.global_note)
        shots = lambda: timing.shot_entries(tm, cam)
        assert len(cam.poses) == 1 and len(shots()) == 1, "new camera: one pose, its image on the board"
        assert cam.color is None, "new cameras have the default color"
        h.v.commit_pose(False, cam=cam)  # not inside: nothing happens
        assert len(cam.poses) == 1
        h.v.look_through(cam.id)
        assert h.v._look == cam.id and h.v._cam_tan is not None
        h.v.set_cam_focal(cam, 80.0)
        h.v.set_frame(1010)
        h.v.cam.orbit(40, 0)
        assert h.v.look_modified()
        h.v.commit_pose(False)
        assert len(cam.poses) == 1, "⟲ makes no move"
        assert not h.v.look_modified()
        first = cam.poses[0].state()
        h.v._confirm_discard = lambda: True
        h.v.set_frame(1020)
        h.v.cam.orbit(-60, 0)
        h.v.commit_pose(True)
        assert [p.frame for p in cam.poses] == [1001, 1020] and abs(cam.poses[1].mm - 80.0) < 1e-6
        assert len(shots()) == 2, "➜: new pose with its own image"
        assert {timing.shot_ref(e)[1] for _, _, e in shots()} == {p.id for p in cam.poses}
        h.v.set_frame(1005)
        assert np.allclose(h.v.cam.rot, first[2]), "jumps to the first pose at 1005"
        h.v.set_frame(1040)
        assert np.allclose(h.v.cam.rot, cam.poses[1].rot)
        # focal length: keep the framing -> subject size at the target point stays
        p = cam.poses[1]
        before = p.dist * timing.mm_to_tan(p.mm)
        h.v.set_cam_focal(cam, 35.0)
        assert abs(p.dist * timing.mm_to_tan(p.mm) - before) < 1e-9 and abs(p.mm - 35.0) < 1e-9
        h.v.leave_camera()
        assert h.v._look is None and h.v._cam_tan is None
        # delete the image of a pose from the board: the pose is gone
        from sketch_machine.note_board import shot_edit
        shot_edit(h.v, tm.global_note.id, lambda items: items.pop(next(
            k for k, e in enumerate(items) if timing.shot_ref(e) == (cam.id, cam.poses[1].id))))
        assert [q.frame for q in cam.poses] == [1001], "image deleted -> pose deleted"
        shot_edit(h.v, tm.global_note.id, lambda items: items.clear())
        assert tm.camera_by_id(cam.id) is None, "last image deleted -> camera deleted"
    finally:
        del h.v._confirm_discard
        _leave(h)


def test_camera_image_in_image_format(h: H):
    plane = _flip_scene(h)
    try:
        _draw(h, plane, 1003, [(-2, 0), (2, 1)])
        cam = h.v.new_timing_camera(h.sc.timing.global_note)
        from PySide6.QtGui import QImage
        img = QImage.fromData(h.sc.asset_bytes(timing.shot_entries(h.sc.timing, cam)[0][2]["file"]))
        assert abs(img.width() / img.height() - timing.GATE) < 0.02, "camera images in 16:9 format"
        img.save(os.path.join(OUT, "timing_kamerabild.png"))
    finally:
        _leave(h)


def test_save_and_load(h: H):
    plane = _flip_scene(h)
    try:
        _draw(h, plane, 1004, [(-2, 0), (2, 0)])
        tm = h.sc.timing
        note = _shot_note(h, plane, text="Explosion")
        h.v.set_frame(1002)
        cam = h.v.new_timing_camera(note)
        _add_pose(h, cam, 1030, h.v.cam.state())
        timing.add_blank_key(plane, 1040)
        path = os.path.join(OUT, "timing_test.sketch.json")
        h.sc.save(path)
        from sketch_machine import timing_export
        from sketch_machine.scene import Scene
        sc = Scene.load(path)[0]
        p2 = next(p for p in sc.planes if p.flipbook)
        assert p2.blank_keys == [1040]
        assert timing.flip_keys(sc)[p2.id] == [1004, 1040]
        assert not sc.notes, "shot notes are not normal notes"
        n2 = sc.timing.notes[0]
        assert n2.title == "Shot 1" and timing.note_layers(sc, n2) == [p2] and timing_export.note_text(n2) == "Explosion"
        c2 = sc.timing.cams_of(n2)[0]
        assert [p.frame for p in c2.poses] == [1002, 1030]
        entries = timing.shot_entries(sc.timing, c2)
        assert len(entries) == 2 and all(sc.asset_bytes(e["file"]) for _, _, e in entries), "camera images in the file"
        assert any(s.flipbook == p2.id for s in sc.strokes)
    finally:
        _leave(h)


def test_old_prototype_scene_is_converted(h: H):
    """Files of the first prototype (timing notes with text, range, cameras with images): shot notes with text and
    camera images on the board, on their plane."""
    import json
    plane = _flip_scene(h)
    _draw(h, plane, 1004, [(-2, 0), (2, 0)])
    _leave(h)
    path = os.path.join(OUT, "timing_alt.sketch.json")
    h.sc.save(path)
    data = json.load(open(path, encoding="utf-8"))
    pose = {"frame": 1001, "target": [0, 0, 0], "dist": 10.0, "rot": list(np.eye(3).ravel()), "mm": 35.0}
    some = next(iter(data["assets"]), None)
    data["assets"]["alt.jpg"] = data["assets"][some] if some else ""
    data["timing"] = {"start": 1001, "end": 1100, "fps": 24, "frame": 1001,
                      "global_note": {"id": 1, "title": "Global", "text": "What it is about", "cameras": []},
                      "notes": [{"id": 2, "title": "Shot 1", "text": "Explosion", "start": 1001, "end": 1040,
                                 "layers": [plane.id], "cameras": [{"id": 3, "name": "Kamera 1", "poses": [pose],
                                                                    "images": [{"frame": 1001, "file": "alt.jpg"}]}]}]}
    json.dump(data, open(path, "w", encoding="utf-8"))
    from sketch_machine import timing_export
    from sketch_machine.scene import Scene
    sc = Scene.load(path)[0]
    tm = sc.timing
    assert timing_export.note_text(tm.global_note) == "What it is about"
    n = tm.notes[0]
    assert n.title == "Shot 1" and "Explosion" in timing_export.note_text(n) and "1001–1040" in timing_export.note_text(n)
    assert [p.name for p in timing.note_layers(sc, n)] == [plane.name]
    cam = tm.cams_of(n)[0]
    assert [e["file"] for _, _, e in timing.shot_entries(tm, cam)] == ["alt.jpg"]


def test_old_scene_without_timing_stays_the_same(h: H):
    h.new_scene()
    assert "timing" not in h.sc.to_dict(OUT), "scenes without timing save no timing"


def test_panel_and_playback(h: H):
    """Shots panel shows the board of the clicked shot note (empty without one, Global on a button), under each camera image
    its settings; playback runs frame by frame (none is skipped) and stops."""
    from sketch_machine.note_board import ImageItem
    plane = _flip_scene(h)
    panel = h.w.shots_panel
    try:
        _draw(h, plane, 1002, [(-1, 0), (1, 0)])
        h.v.set_frame(1001)
        h.v.ws.active_note = None
        h.w._sync_shots_panel()
        assert panel.note_id is None, "nothing clicked: panel empty"
        note = _shot_note(h, plane)
        h.w._sync_shots_panel()
        assert panel.note_id == note.id
        panel.add_snapshot()  # camera button on the board: new camera
        cam = h.sc.timing.cams_of(note)[0]
        rows = [i.row for i in panel.gscene.items() if isinstance(i, ImageItem) and i.shot]
        assert len(rows) == 1 and not rows[0].buttons["add"].isEnabled(), "⟲/➜ only from inside"
        assert rows[0].name.text() == cam.name
        rows[0]._action("look")
        h.pump(0.05)
        assert h.v._look == cam.id
        panel.refresh_shot_rows()
        rows = [i.row for i in panel.gscene.items() if isinstance(i, ImageItem) and i.shot]
        assert rows[0].buttons["add"].isEnabled()
        h.v.leave_camera()
        h.w._set_shot_scope("global")
        assert panel.note_id == h.sc.timing.global_note.id
        h.w._set_shot_scope("local")
        assert h.w.note_panel.note_id != note.id, "shot notes not in the Note panel"
        seen = []
        h.v.ws.frame_changed.connect(seen.append)
        h.w.timeline.set_playing(True)
        for _ in range(40):
            h.v.repaint()
            h.v._frame_painted = True  # the test window is off-screen: the frame counts as painted
            h.pump(0.03)
        h.w.timeline.set_playing(False)
        h.v.ws.frame_changed.disconnect(seen.append)
        assert len(seen) >= 5, f"only {len(seen)} frames played"
        assert all(b - a == 1 or (a == h.sc.timing.end and b == h.sc.timing.start) for a, b in zip(seen, seen[1:])), \
            "no frame skipped"
        assert not h.v.ws.playing
    finally:
        _leave(h)


def _two_pose_camera(h: H, plane):
    """Shot note with camera: pose at 1001 (24 mm) and 1020 (50 mm, other direction)."""
    note = _shot_note(h, plane, text="Explosion grows")
    h.camera((0, 0, 0), 12.0, (0.3, 0.2, 0.9))
    h.v.set_frame(1001)
    cam = h.v.new_timing_camera(note)
    h.camera((0, 0, 0), 15.0, (-0.6, 0.3, 0.7))
    _add_pose(h, cam, 1020, h.v.cam.state(), mm=50.0)
    return note, cam


def test_usd_cameras_stepped_and_linear(h: H):
    """Timing cameras in scene.usd: stepped holds each pose until the next, linear lies in between; view direction
    and focal length are right."""
    from pxr import Usd, UsdGeom
    from sketch_machine import usd_export
    plane = _flip_scene(h)
    try:
        _draw(h, plane, 1005, [(-1, 0), (1, 0)])
        note, cam = _two_pose_camera(h, plane)
        p0, p1 = cam.poses
        eye = lambda p: p.target + p.dist * p.rot[:, 2]
        for interp in ("step", "linear"):
            h.sc.timing.interp = interp
            path = os.path.join(OUT, f"timing_{interp}.usda")
            usd_export.write(h.sc, path)
            stage = Usd.Stage.Open(path)
            assert stage.GetStartTimeCode() == h.sc.timing.start and stage.GetTimeCodesPerSecond() == 24
            prim = next(p for p in stage.Traverse() if p.GetTypeName() == "Camera")
            assert str(prim.GetPath()).startswith("/sketch/shots/")
            xf = UsdGeom.Xformable(prim)
            focal = UsdGeom.Camera(prim).GetFocalLengthAttr()

            def at(t):
                m = np.array(xf.ComputeLocalToWorldTransform(t))
                return m[3, :3], m[:3, :3]  # row vectors: row 3 = position, rows 0..2 = right, up, back
            pos, rows = at(1001)
            assert np.allclose(pos, eye(p0), atol=1e-4) and np.allclose(rows[2], p0.rot[:, 2], atol=1e-4), "pose 1"
            assert abs(focal.Get(1001) - p0.mm) < 1e-3
            pos, rows = at(1020)
            assert np.allclose(pos, eye(p1), atol=1e-4) and np.allclose(rows[2], p1.rot[:, 2], atol=1e-4), "pose 2"
            assert abs(focal.Get(1030) - 50.0) < 1e-3
            mid = at(1010)[0]
            if interp == "step":
                assert np.allclose(mid, eye(p0), atol=1e-4) and np.allclose(at(1019)[0], eye(p0), atol=1e-4), "stepped holds"
                assert abs(focal.Get(1019) - p0.mm) < 1e-3
            else:
                assert not np.allclose(mid, eye(p0), atol=1e-3) and not np.allclose(mid, eye(p1), atol=1e-3), "linear in between"
    finally:
        _leave(h)


def test_brief_with_timing(h: H):
    """Sketch Brief: Timing section (shots, cameras with poses, flipbook drawings), brief.json → timing,
    fresh images per camera at poses and drawing changes."""
    import json
    import shutil
    from sketch_machine import brief
    plane = _flip_scene(h)
    try:
        _draw(h, plane, 1005, [(-1, 0), (1, 0)])
        _draw(h, plane, 1012, [(-1, 1), (1, 1)])
        h.v.set_frame(1030)
        h.w.timeline._blank()
        h.sc.timing.global_note.board = {"items": [{"kind": "text", "x": 0.0, "y": 0.0, "w": 340.0,
                                                    "text": "A small explosion at the engine"}]}
        note, cam = _two_pose_camera(h, plane)
        folder = os.path.join(OUT, "timing_brief")
        shutil.rmtree(folder, ignore_errors=True)
        brief.export(h.sc, h.v, folder, "timing")
        md = open(os.path.join(folder, "brief.md"), encoding="utf-8").read()
        assert "## Shots – what happens when" in md and "A small explosion at the engine" in md
        assert "S1 – Shot 1" in md and "camera move" in md and "Explosion grows" in md
        assert f"layers: {plane.name}" in md
        data = json.load(open(os.path.join(folder, "brief.json"), encoding="utf-8"))["shots"]
        assert data["notes"][0]["board"] and os.path.isfile(os.path.join(folder, data["notes"][0]["board"])), \
            "board of the shot note as an image"
        assert data["camera_interpolation"] == "step"
        lay = data["layers"][0]
        assert lay["drawings"] == [1005, 1012] and lay["empty"] == [1030]
        c = data["notes"][0]["cameras"][0]
        assert [p["frame"] for p in c["poses"]] == [1001, 1020] and c["poses"][0]["holds_until"] == 1019
        frames = [i["frame"] for i in c["images"]]
        assert {1001, 1005, 1012, 1020, 1030} <= set(frames), frames
        assert all(os.path.isfile(os.path.join(folder, i["file"])) for i in c["images"])
        assert "Flipbook" not in json.dumps(json.load(open(os.path.join(folder, "brief.json"), encoding="utf-8"))["lines"]), \
            "flipbook strokes are not in the lines"
    finally:
        _leave(h)


def test_brief_without_timing_content_without_section(h: H):
    """Without action no timing in the brief: empty notes, unpainted flipbook planes do not count; in the section
    only notes with content and planes with drawings appear; without cameras no cameras in the USD."""
    import json
    import shutil
    from sketch_machine import brief, timing_export
    plane = _flip_scene(h)
    try:
        _shot_note(h, plane, title="Empty")
        assert not timing_export.has_timing(h.sc), "empty note and empty plane: no timing"
        folder = os.path.join(OUT, "timing_brief_leer")
        shutil.rmtree(folder, ignore_errors=True)
        brief.export(h.sc, h.v, folder, "leer")
        assert "## Shots" not in open(os.path.join(folder, "brief.md"), encoding="utf-8").read()
        assert "shots" not in json.load(open(os.path.join(folder, "brief.json"), encoding="utf-8"))
        assert not os.path.isdir(os.path.join(folder, "shots"))
        assert "/shots" not in open(os.path.join(folder, "scene.usd"), "rb").read().decode("latin-1")
        h.v.new_plane()  # second, unpainted plane
        h.v.activate_flipbook(plane)
        _draw(h, plane, 1004, [(-1, 0), (1, 0)])
        assert timing_export.has_timing(h.sc)
        data = timing_export.brief_data(h.sc)
        assert data["notes"] == [] and [l["name"] for l in data["layers"]] == [plane.name]
        assert timing_export.usd_lines(h.sc, 1) == [] and timing_export.usd_header(h.sc) == []
    finally:
        _leave(h)


def test_flipbook_plane_move_only_with_gizmo(h: H):
    """Flipbook planes are fixed objects: mouse wheel/[ ] do not move them; T/R/E in Timing shows the gizmo, dragging moves
    the plane with the drawings of all frames; Esc removes the gizmo."""
    from PySide6.QtCore import QPointF
    plane = _flip_scene(h)
    try:
        a = _draw(h, plane, 1003, [(-1, 0), (1, 0)])
        b = _draw(h, plane, 1008, [(-1, 1), (1, 1)])
        flip = [c for c in h.sc.curves if c.frame is not None]
        origin = plane.origin.copy()
        h.v._shift_plane(1.0, burst=("wheel", plane.id))
        assert np.allclose(plane.origin, origin), "mouse wheel/[ ] do not move flipbook planes"
        assert h.v.gizmo_target is None, "without T/R/E no gizmo"
        h.key(Qt.Key_T)
        target = h.v.gizmo_target
        assert target is not None and len(target.items) == 1 + len(flip), "plane with all drawings"
        # real dragging on the gizmo: find a handle next to the pivot
        c = h.scrf(plane.origin)
        grab = next((QPointF(c.x() + dx, c.y() + dy) for r in range(12, 160, 3) for dx, dy in ((r, 0), (0, -r), (-r, 0), (0, r))
                     if h.v.gizmo.hit(QPointF(c.x() + dx, c.y() + dy), target)), None)
        assert grab is not None, "gizmo handle found"
        before = [cv.points.copy() for cv in flip]
        h.stroke([grab, QPointF(grab.x() + 60, grab.y() + 40)])
        moved = plane.origin - origin
        assert np.linalg.norm(moved) > 1e-3, "plane moved"
        for cv, old in zip(flip, before):
            assert np.allclose(cv.points - old, moved, atol=1e-6), "drawings of all frames move along"
        assert not any(cv.frame is not None and all(cv is not f for f in flip) for cv in h.sc.curves), "no stroke was created"
        # rotate and scale via the target (same paths as the gizmo)
        from sketch_machine.camera import rot_axis
        t2 = h.v.gizmo_target
        t2.rotate(rot_axis(np.array([0, 1.0, 0]), 0.5), t2.pivot())
        t2.scale(np.array([2.0, 2.0, 1.0]), t2.local_axes(), t2.pivot())
        d = [float((cv.points - plane.origin) @ plane.normal @ np.ones(len(cv.points))) for cv in flip]
        assert all(abs(x) < 1e-6 for x in d), "drawings stay in the plane"
        h.key(Qt.Key_Escape)
        assert h.v.gizmo.tool is None and h.v.gizmo_target is None
    finally:
        _leave(h)


def test_shot_notes_like_notes(h: H):
    """Shots: double-click creates a shot note (on a drawing: pinned to it) – own list, not with the
    normal notes; a simple stroke keeps painting; deleting the note takes its cameras along; in the Notes tab
    shot notes are not visible."""
    plane = _flip_scene(h)
    tm = h.sc.timing
    try:
        a = _draw(h, plane, 1003, [(-2, 0), (2, 0)])
        normal = len(h.sc.notes)
        h.dclick(h.scrf(a[0].points[len(a[0].points) // 2]))
        h.v.notes_tool.finish_edit()
        assert len(tm.notes) == 1 and len(h.sc.notes) == normal, "shot note in its own list"
        note = tm.notes[0]
        assert note.anchors and note.anchors[0].curve_id == a[0].id, "pinned to the drawing"
        assert timing.note_layers(h.sc, note) == [plane]
        assert h.v.ws.active_note == note.id
        n = len(h.sc.curves)
        _draw(h, plane, 1003, [(-2, -2), (2, -2.5)])
        assert len(h.sc.curves) == n + 1, "painting continues next to the note"
        cam = h.v.new_timing_camera(note)
        assert tm.note_of(cam) is note
        h.v.notes_tool.delete(note.id)
        assert not tm.notes and not tm.cams, "note deleted -> its cameras too"
        h.v.undo()
        tm = h.sc.timing  # (undo creates a new timing)
        assert len(tm.notes) == 1 and len(tm.cams) == 1, "undo brings back note and camera"
        _leave(h)
        h.v.set_notes_mode(True)
        assert h.v.note_list() is h.sc.notes and h.v.notes_tool.hit(h.scrf(tm.notes[0].pos)) is None, \
            "no shot notes in the Notes tab"
        h.v.set_notes_mode(False)
    finally:
        _leave(h)


def test_flipbook_locked_everywhere_outside(h: H):
    """Outside Timing: outliner shows flipbook only greyed out (not selectable), delete/rename/reparent and
    edit gizmo have no effect; in Timing deleting the plane takes its drawings along. In a timing camera no
    axis view."""
    plane = _flip_scene(h)
    _draw(h, plane, 1002, [(-1, 0), (1, 0)])
    stroke = next(s for s in h.sc.strokes if s.flipbook == plane.id)
    tm = h.sc.timing
    cam = h.v.new_timing_camera(tm.global_note)
    h.v.look_through(cam.id)
    h.v.axis_view(np.array([0, 1.0, 0]))
    h.pump(0.4)
    assert not h.v.cam.ortho and h.v._look == cam.id, "no axis view in the camera"
    h.v.leave_camera()
    _leave(h)
    h.v.set_selection([plane.id, stroke.id])
    assert h.v.selection == [], "outliner does not select flipbook outside Timing"
    h.w.outliner.rebuild()
    from PySide6.QtWidgets import QTreeWidgetItemIterator
    from sketch_machine import outliner as ol
    rows = {}
    it = QTreeWidgetItemIterator(h.w.outliner)
    while it.value():
        rows[it.value().data(0, ol.ID_ROLE)] = it.value().flags()
        it += 1
    assert not rows[plane.id] & Qt.ItemIsSelectable and not rows[stroke.id] & Qt.ItemIsSelectable
    h.v.delete_objects([plane.id, stroke.id])
    h.v.rename(plane.id, "Other")
    assert plane in h.sc.custom and stroke in h.sc.strokes and plane.name != "Other"
    h.v.set_select_mode(True)
    h.v.selection = [plane.id]  # e.g. brought back via undo
    h.v.gizmo.tool = "translate"
    assert h.v.gizmo_target is None, "edit gizmo does not move a flipbook plane"
    h.v.selection = []
    h.v.set_select_mode(False)
    h.v.set_timing_mode(True)
    try:
        h.w.outliner.rebuild()
        h.v.delete_objects([plane.id])
        assert plane not in h.sc.custom and all(s.flipbook != plane.id for s in h.sc.strokes), \
            "deleting the flipbook plane takes its drawings along"
        assert not any(c.frame is not None for c in h.sc.curves)
    finally:
        _leave(h)


def test_fully_erased_stays_blank_drawing(h: H):
    """Everything erased (here with Ctrl+Shift: whole strokes) – the frame stays a blank drawing, the previous one
    does not come back; afterwards the new pose is painted there."""
    from PySide6.QtCore import QPointF
    plane = _flip_scene(h)
    try:
        _draw(h, plane, 1010, [(-2, 0), (2, 0)])
        h.v.set_frame(1014)
        h.v.set_draw_tool("eraser")
        mid = h.scrf(_on_plane(plane, [(0, 0)])[0])
        h.stroke([QPointF(mid.x() - 3, mid.y() - 15), QPointF(mid.x() + 3, mid.y() + 15)],
                 mods=Qt.ControlModifier | Qt.ShiftModifier)
        h.v.set_draw_tool("brush")
        assert timing.flip_keys(h.sc)[plane.id] == [1010, 1014] and 1014 in plane.blank_keys
        assert not timing.key_curves(h.sc, plane.id, 1014), "at 1014 the drawing is blank"
        assert len(timing.key_curves(h.sc, plane.id, 1010)) == 1, "1010 stays"
        assert not any(h.v._layer_visible(c) for c in h.sc.curves if c.frame is not None), "nothing comes back"
        new = _draw(h, plane, 1014, [(-1, 2), (1, 2)])
        assert all(c.frame == 1014 for c in new) and len(timing.key_curves(h.sc, plane.id, 1014)) == 1, \
            "new pose at 1014, without a copy of the old one"
    finally:
        _leave(h)
