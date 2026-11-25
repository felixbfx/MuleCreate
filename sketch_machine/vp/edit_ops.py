"""Undo/redo, set scene, copy/paste/duplicate, group."""
from .common import *  # noqa: F401,F403
from ..i18n import tr
from PySide6.QtCore import QByteArray, QMimeData


class EditOpsMixin:
    def _queued_press(self):
        """Did this Ctrl+Z/Y only arrive in the queue while the previous step was still computing? Then it does not count
        (harbor test: undo hung, the user pressed several times – afterwards all the queued presses ran and much more was
        lost than intended). Detectable: arrives immediately after the end of a slow step."""
        last = getattr(self.ws, "_undo_busy", None)
        return last is not None and last[1] > 0.25 and time.monotonic() - last[0] < 0.25

    def _undo_step(self, fn):
        if self._queued_press():
            return self.notify(tr("Rückgängig: gestaute Tastendrücke verworfen – der vorige Schritt hat gerechnet"))
        t = time.monotonic()
        self.ws.selection_pivot = None  # geometry jumps back: pivot anew from the bounding box
        if fn():
            self._changed()
            self.repaint()  # image now, not only after the queued presses
        self.ws._undo_busy = (time.monotonic(), time.monotonic() - t)

    def undo(self):
        self._undo_step(self.scene.undo)

    def redo(self):
        self._undo_step(self.scene.redo)

    def set_scene(self, scene):
        from ..note_board import board_for
        board_for(self).close_board()
        for vp in self.ws.viewports:
            vp.notes_tool.cancel()
        self.ws.active_note = None
        for vp in self.ws.viewports:  # timing: the cameras of the old scene no longer exist
            vp._look, vp._cam_tan = None, None
        self.ws.playing, self.ws.timing_cam = False, None
        self.scene = scene
        if self.ws.timing_mode:
            self._timing_pick_layer()
        self.ws.timing_changed.emit()
        self.ws.frame_changed.emit(scene.timing.frame)
        scene.selection_io = (self._selection_state, self._set_selection_state)
        self.selection = []
        self.edge_selection = []
        self.ws.detached = frozenset()
        self._changed()

    # --- Copy & paste -----------------------------------------------------
    def copy_selection(self, objects_only=False, clip=None):
        self._materialize_segments()
        return self._copy_selection(objects_only, clip)

    def _copy_selection(self, objects_only=False, clip=None):
        """Selection to the clipboard: objects and (unless objects_only) selected lines/edges."""
        objs = self.selected_objects
        refs = [] if objects_only else list(dict.fromkeys(self.edge_selection))
        if not objs and not refs:
            self.status_changed.emit(tr("Kopieren: nichts ausgewählt."))
            return False
        (clip or self.ws.clipboard).copy(self.scene, objs, refs)
        parts = [tr("{0} Objekt(e)").format(len(objs))] if objs else []
        parts += [tr("{0} Linie(n)").format(len(refs))] if refs else []
        if clip is None:
            self._to_system_clipboard()
            self.status_changed.emit(tr("Kopiert: ") + ", ".join(parts))
        return True

    def _to_system_clipboard(self):
        """Also into the system clipboard, so that a second Mule window can paste it (clipboard.MIME)."""
        try:
            raw = self.ws.clipboard.to_bytes(self.scene)
        except Exception:  # noqa: BLE001 – within this window the copy still works
            return
        mime = QMimeData()
        mime.setData(clipboard.MIME, QByteArray(raw))
        QApplication.clipboard().setMimeData(mime)

    def _from_system_clipboard(self):
        """Copy of another Mule window, if it is the latest thing in the system clipboard; otherwise None."""
        mime = QApplication.clipboard().mimeData()
        if mime is None or not mime.hasFormat(clipboard.MIME):
            return None
        return Clipboard.from_bytes(mime.data(clipboard.MIME))

    def paste(self, mode="elements", clip=None, duplicate=False):
        """Paste at the same place; the pasted items are then selected, translate gizmo active.
        mode "elements" (viewport): lines into the active stroke object, "objects" (outliner): new objects."""
        clip = clip or self._from_system_clipboard() or self.ws.clipboard
        if clip.empty:
            self.status_changed.emit(tr("Einfügen: Zwischenablage ist leer."))
            return
        self._resolve_pending_paste(force=True)  # a previous, unmoved copy is dropped
        self.scene.checkpoint()
        objects, refs = clipboard.paste(self.scene, clip, mode, duplicate)
        self.selection = [o.id for o in objects]
        self.edge_selection = refs
        self.ws.detached = frozenset(refs)
        self.ws.pending_paste = frozenset(refs) if refs and not objects else None  # only counts once moved
        if self._paste_restore is None:
            self._paste_restore = (self.gizmo.tool,)
        self.gizmo.tool = "translate"
        if not self.select_mode:  # placing with the gizmo belongs in selection mode
            self.set_select_mode(True)
        self._changed()
        target = self.transform_target
        if target is not None:  # for "repeat duplicate": where the pasted items initially lay
            lo, hi = target.bounds()
            self._dup_last = (self._selection_key(), (lo + hi) / 2)

    def duplicate(self, mode="elements"):
        """Ctrl+D: copy and paste right away, without changing the clipboard."""
        clip = Clipboard()
        if self.copy_selection(objects_only=mode == "objects", clip=clip):
            self.paste(mode, clip, duplicate=True)
            return True
        return False

    def duplicate_repeat(self, mode="elements"):
        """Ctrl+Shift+D: duplicate again and move by the same offset as last time
        (Ctrl+D, move the copy, then Ctrl+Shift+D as often as needed – e.g. floors)."""
        target = self.transform_target
        if target is not None and self._dup_last is not None and self._dup_last[0] == self._selection_key():
            lo, hi = target.bounds()
            delta = (lo + hi) / 2 - self._dup_last[1]
            if np.linalg.norm(delta) > 1e-9:
                self._dup_step = delta
        if self._dup_step is None:
            self.status_changed.emit(tr("Duplizieren wiederholen: erst Strg+D und die Kopie verschieben."))
            return
        if not self.duplicate(mode):
            return
        target = self.transform_target
        if target is not None:
            target.translate(self._dup_step.copy())  # belongs to the same undo step as the paste
        self.notify(tr("Dupliziert und verschoben um ") + " ".join(f"{x:+.3g}" for x in self._dup_step))
        self._changed()

    def group_selection(self):
        self._materialize_segments()
        return self._group_selection()

    def _group_selection(self):
        """Ctrl+G: move selected lines into a new stroke object (e.g. podium, tower, band
        separately) – for order in the outliner and for export.
        Surfaces follow their lines: a surface whose lines ALL move along then belongs to the new object (together with
        its painting) – whether it is also selected or not. A surface of which only some of the lines move stays
        where it is. If other objects are selected as well (e.g. a mesh), the new object and they go together into a
        new null – there in the hierarchy where the object the lines came from lay."""
        curves = [c for c in dict.fromkeys(edges.ref_to_curve(self.scene, r) for r in self.edge_selection)
                  if c is not None]
        if not curves:  # no lines selected, but stroke objects: merge their lines into one object
            # (also-selected surfaces do not count as line objects – they follow their lines anyway)
            picked = [o for o in self._tree_selection() if isinstance(o, StrokeObject) and not isinstance(o, SurfaceObject)]
            if len(picked) >= 2:
                return self._merge_strokes(picked)
            self.status_changed.emit(tr("Neues Objekt: erst Linien auswählen (Edit/Tab, Klick / Rechteck / L) – "
                                        "oder mehrere Stroke-Objekte, um sie zusammenzulegen."))
            return
        self.scene.checkpoint()
        owners = {id(o): o for o in (self.scene.owner_of(c) for c in curves) if o is not None}
        home = self.scene.common_parent(list(owners.values()))
        # surfaces never go into the null: they hang on their lines (move along or stay)
        extra = [o for o in self._tree_selection() if id(o) not in owners and not isinstance(o, SurfaceObject)]
        active = self.scene.active_stroke
        stroke = self.scene.new_stroke(parent=home)
        self.scene.active_stroke = active  # drawing continues as before (if it became empty: new object)
        for c in curves:
            owner = self.scene.owner_of(c)
            if owner is not None and owner is not stroke:
                owner.curves.remove(c)
            stroke.curves.append(c)
        faces = self._adopt_surfaces(stroke, {c.id for c in curves})
        for owner in owners.values():  # objects that became empty disappear
            if not owner.curves and owner in self.scene.strokes:
                self.scene.strokes.remove(owner)
        self.edge_selection = [("line", stroke.id, c.id) for c in curves]
        self.selection = []
        group = ""
        if extra:  # lines + selected objects: together into a null at the place of origin
            members = [stroke] + extra
            boxes = [o.bounds() for o in self._with_descendants(members) if not isinstance(o, NullObject)
                     and not (isinstance(o, StrokeObject) and not o.curves)]
            origin = ((np.min([b[0] for b in boxes], axis=0) + np.max([b[1] for b in boxes], axis=0)) / 2
                      if boxes else None)
            null = self.scene.new_null("null", origin, parent=home)
            self.scene.set_parent(members, null)
            group = tr(", zusammen mit {0} Objekt(en) in „{1}“").format(len(extra), null.name)
        if faces:
            group = tr(", {0} Fläche(n) ziehen mit um").format(len(faces)) + group
        self.notify(tr("{0} Linie(n) in neues Objekt „{1}“{2} – Doppelklick im Outliner zum Umbenennen.").format(len(curves), stroke.name, group))
        self._changed()

    def _adopt_surfaces(self, stroke, moved_ids):
        """Surfaces whose lines (border, outline, cross-sections, longitudinal lines) ALL just moved into stroke now belong
        to stroke: owner and place in the hierarchy like the object, their painting moves along (painting hangs on
        its surface and cannot be selected individually). Surfaces of which only some of the lines moved
        stay with their previous object. Returns the taken-over surfaces."""
        sc = self.scene
        taken = []
        for f in sc.surfaces:
            if f.aid:
                continue
            ids = sc.defining_line_ids(f)
            if not ids or not all(i in moved_ids for i in ids):
                continue
            f.owner, f.parent = stroke.id, stroke.parent
            f._owner_parent = stroke.parent
            taken.append(f)
            for c in sc.curves_on(f.id):
                owner = sc.owner_of(c)
                if c.paint and owner is not None and owner is not stroke:
                    owner.curves.remove(c)
                    stroke.curves.append(c)
        return taken

    def _merge_strokes(self, picked):
        """Ctrl+G with several selected stroke objects (without line selection): all their lines and their painting go
        into the first of them (in outliner order, name and place stay); the others disappear. Their surfaces
        then all belong to the one object and sit below it. Other selected objects (meshes) stay."""
        sc = self.scene
        picked = sorted(picked, key=sc.strokes.index)
        keep, rest = picked[0], picked[1:]
        sc.checkpoint()
        n = sum(len(o.curves) for o in rest)
        gone = {o.id for o in rest}
        merged = list(keep.curves)
        for o in rest:
            merged += o.curves
            o.curves = []
            sc.strokes.remove(o)
        keep.curves = merged
        faces = [f for f in sc.surfaces if f.owner in gone]
        for f in faces:  # surfaces of the dissolved objects: now with the one object, at its place in the hierarchy
            f.owner, f.parent = keep.id, keep.parent
            f._owner_parent = keep.parent
        if sc.active_stroke in rest:
            sc.active_stroke = keep
        self.edge_selection = []
        self.selection = [keep.id]
        text = tr("{0} Stroke-Objekte zu „{1}“ zusammengelegt ({2} Linien dazu).").format(len(picked), keep.name, n)
        if faces:
            text += " " + tr("{0} Fläche(n) ziehen mit um").format(len(faces))
        self.notify(text)
        self._changed()

    # --- Color and type of selected lines (edit) --------------------------------------
    def _chosen_curves(self):
        """Selected lines (edges, partial pieces – each the whole line – and selected stroke objects), without paint strokes."""
        refs = list(self.edge_selection) + [("line", r[1], r[2]) for r in self.seg_selection]
        curves = [edges.ref_to_curve(self.scene, r) for r in refs]
        for obj in self.selected_objects:
            if isinstance(obj, StrokeObject) and not isinstance(obj, SurfaceObject):  # (surfaces: their shape, not lines)
                curves += obj.curves
        return [c for c in dict.fromkeys(curves) if c is not None and not c.paint]

    def apply_color(self, rgb=None, reset=False):
        """Edit, "Apply" in the color picker: everything selected gets the color rgb (otherwise the current one) – surfaces as
        tint, brush lines as stroke color. Pencil lines stay as they are (for constructing; convert via
        right click), helper surfaces have no material. reset: tint of the selected surfaces back to their material
        (lines stay). One undo step. Returns whether anything was colored."""
        rgb = tuple(float(x) for x in (self.ws.paint_color if rgb is None else rgb))
        surfaces = [o for o in self.selected_objects if isinstance(o, SurfaceObject) and not o.aid]
        curves = self._chosen_curves()
        brush = [] if reset else [c for c in curves if c.style == "brush"]
        pencil = len(curves) - len(brush)
        if not surfaces and not brush:
            self.notify(tr("Bleistiftlinien werden nicht umgefärbt (Rechtsklick: in Pinselstrich umwandeln)")
                        if curves and not reset else tr("Anwenden: erst Flächen oder Pinsellinien wählen"))
            return False
        self.scene.checkpoint()
        for s in surfaces:
            s.tint = None if reset else rgb
        for c in brush:
            c.color = rgb
        if reset:
            text = tr("{0} Fläche(n): Färbung zurückgesetzt").format(len(surfaces))
        else:
            parts = ([tr("{0} Fläche(n)").format(len(surfaces))] if surfaces else []) + \
                    ([tr("{0} Pinsellinie(n)").format(len(brush))] if brush else [])
            text = tr("{0} gefärbt").format(" + ".join(parts))
            if pencil:
                text += tr(" – {0} Bleistiftlinie(n) unverändert (Rechtsklick: in Pinselstrich umwandeln)").format(pencil)
        self.notify(text)
        self._changed()
        return True

    def convert_lines(self, style):
        """Right click in edit: convert selected lines into brush strokes (current color) or pencil – shape,
        connections and width stay."""
        curves = [c for c in self._chosen_curves() if c.style != style]
        if not curves:
            return self.notify(tr("Umwandeln: erst Linien wählen") if not self._chosen_curves() else
                               tr("Die gewählten Linien sind schon {0}").format(tr("Pinselstriche") if style == 'brush' else tr("Bleistift")))
        self.scene.checkpoint()
        for c in curves:
            c.style = style
            c.color = tuple(float(x) for x in self.ws.paint_color) if style == "brush" else None
        self.notify(tr("{0} Linie(n) in {1} umgewandelt").format(len(curves), tr("Pinselstriche") if style == 'brush' else tr("Bleistift")))
        self._changed()
