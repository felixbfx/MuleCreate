"""Selection: objects, edges, points; selection state and undo, pivot, rectangle and loop selection."""
from .common import *  # noqa: F401,F403
from .. import extruding
from ..i18n import tr


class SelectionMixin:
    # --- Selection ----------------------------------------------------------
    @property
    def selected_objects(self):
        return [o for o in (self.scene.object_by_id(i) for i in self.selection) if o is not None]

    def _with_descendants(self, objs):
        """objs plus everything under selected nulls (each object once, order kept) - what a null moves."""
        out, seen = [], set()
        for o in objs:
            for x in [o] + (self.scene.descendants(o) if isinstance(o, NullObject) else []):
                if x.id not in seen:
                    seen.add(x.id)
                    out.append(x)
        return out

    @property
    def selected(self):
        """The only selected object, otherwise None."""
        objs = self.selected_objects
        return objs[0] if len(objs) == 1 else None

    @property
    def transform_target(self):
        """What the transform gizmo moves. The fixed planes 1-3 are not transformed."""
        # Needed on every mouse move and every frame: cached until the selection
        # or scene (checkpoint, undo) changes. During a drag the target stays the same this way.
        key = (id(self.scene), self.scene.revision, tuple(self.selection), tuple(self.edge_selection),
               tuple(self.point_selection), tuple(self.seg_selection),
               self.ws.detached, self.ws.axis_mode, self._free_move, self.ws.knot_mode)
        if self.ws.target_cache[0] == key:
            return self.ws.target_cache[1]
        picked = [o for o in self.selected_objects if not (isinstance(o, Plane) and o.preset)
                  and not self._flip_locked(o)]  # flipbook moves only with the gizmo in timing (along with its drawings)
        items = self._with_descendants(picked)  # a null moves everything under it along
        selected = set(self.selection) | {o.id for o in items}
        refs = [r for r in self.edge_selection if r[1] not in selected]  # do not duplicate wholly selected objects
        # frozen helper surfaces move only themselves, never their lines
        for fill_obj in [o for o in items if type(o) is SurfaceObject and o.links and o.frozen is None]:  # outline along
            refs += [r for r in (edges.curve_to_ref(self.scene, self.scene.curve_by_id(i)) for cid, _ in fill_obj.links
                                 for i in self.scene.link_ids(cid)
                                 if cid is not None) if r is not None and r not in refs and r[1] not in selected]
        for loft_obj in [o for o in items if isinstance(o, LoftSurface) and o.frozen is None]:  # cross-sections along
            for src in loft_obj.sources:
                if src is None:
                    continue
                if src[0] == "lines":
                    refs += [r for r in (edges.curve_to_ref(self.scene, self.scene.curve_by_id(cid)) for cid in src[1])
                             if r is not None and r not in refs and r[1] not in selected]
                else:
                    other = self.scene.object_by_id(src[1])
                    if other is not None and other not in items:
                        items.append(other)
        # Moving a surface means: moving ALL its lines - border, net, cross and longitudinal lines, the lines lying on
        # it, and its painting. The surface results from its lines (never the other way round); moved alone it would lie
        # beside them and jump back at the next reconciliation. Exception: frozen helper surfaces (above).
        have = set(refs)
        for surf in [o for o in items if isinstance(o, SurfaceObject) and o.frozen is None]:
            for curve in self.scene.surface_lines(surf):  # the same net that also comes along when copying
                r = edges.curve_to_ref(self.scene, curve)
                if r is not None and r not in have and r[1] not in selected:
                    have.add(r)
                    refs.append(r)
        pts, segs = self._valid_points(), self._valid_segs()
        if (pts or segs) and refs:
            # Mixed (whole lines + sub-edges/points): the whole lines as all their sub-edges into the SAME
            # group - two groups side by side moved shared points twice (harbor test 09.11.: lines
            # ran up to 2x as far)
            have = {(r[2], r[3], r[4]) for r, _c in segs}
            rest = []
            for r in refs:
                c = edges.ref_to_curve(self.scene, r)
                if c is None or c.paint:
                    rest.append(r)
                    continue
                for a, b in components.segments(c):
                    if (c.id, a, b) not in have:
                        have.add((c.id, a, b))
                        segs.append((("seg", r[1], c.id, a, b), c))
            refs = rest
        if pts or segs:  # points/sub-edges: soft, welded points move along (Alt: only the selected)
            pasted = {r[2] for r in self.ws.detached}
            only = pasted if pasted and all(c.id in pasted for _, c in list(pts) + list(segs)) else None
            items = [components.ComponentGroup(self.scene, [(c, r[3]) for r, c in pts],
                                               [(c, r[3], r[4]) for r, c in segs], weld=not self._free_move,
                                               mode=self.ws.knot_mode, only=only)] + items
        if refs:
            # freshly pasted lines: welded only among themselves (not with the original they lie on)
            only = self.ws.detached if self.ws.detached and set(refs) <= self.ws.detached else None
            items = edges.handles_for(self.scene, refs, not self._free_move, only) + items  # Alt: without welding
        target = None
        null = picked[0] if len(picked) == 1 and isinstance(picked[0], NullObject) and not (pts or segs or refs) else None
        if null is not None:
            # A null: its axis cross is the pivot (like the object axis in Cinema 4D); in axis mode the
            # gizmo moves only the null, the objects below it stay put
            target = TransformSet(items, override=lambda: (null.origin.copy(), null.axes.copy(), True), scene=self.scene)
            if self.ws.axis_mode:
                target = PivotHandle(target, lambda point, axes: self._place_null(null, point, axes))
        elif items:
            sel_key = self._selection_key()
            stored = lambda: self.ws.selection_pivot[1:] if self.ws.selection_pivot and self.ws.selection_pivot[0] == sel_key else None
            store = lambda point, axes, own: setattr(self.ws, "selection_pivot", (sel_key, point, axes, own))
            target = TransformSet(items, override=stored, store=store, scene=self.scene)
            if self.ws.axis_mode:  # gizmo moves only the axis
                target = PivotHandle(target, lambda point, axes: store(point, axes, True))
        self.ws.target_cache = (key, target)
        return target

    @property
    def gizmo_target(self):
        """Transform target for the gizmo - only in selection mode. Draw mode is for drawing;
        moving, rotating, scaling drawing templates belongs in edit mode (Tab)."""
        if self.ws.timing_mode:  # timing: the active flipbook plane with all drawings (T/R/E)
            return self._timing_gizmo_target()
        return self.transform_target if self.select_mode and not self.ws.value_tool else None

    def _selection_key(self):
        return (tuple(self.selection), tuple(self.edge_selection), tuple(self.point_selection),
                tuple(self.seg_selection))

    # --- Axis mode (Alt + A): own pivot for the current selection ---
    def set_axis_mode(self, on):
        self.ws.axis_mode = on
        if on and self.gizmo.tool not in ("translate", "rotate"):
            self.gizmo.tool = "translate"
        self.toggles_changed.emit()
        self._changed()

    @staticmethod
    def _place_null(null, point, axes):
        """Axis mode on a null: only offset/rotate the null (keep axes orthonormal)."""
        null.origin = np.array(point, float)
        if axes is not None:
            q, r = np.linalg.qr(np.asarray(axes, float))
            null.axes = q * np.sign(np.diag(r))  # remove noise from the rotation, direction stays

    def has_custom_pivot(self):
        pivot = self.ws.selection_pivot
        return bool(pivot) and pivot[0] == self._selection_key() and pivot[3]

    def reset_pivot(self):
        self.ws.selection_pivot = None
        self._changed()

    def _move_snap(self, pos):
        """V while moving: (source point, target point) or None. The target is a line end, a
        line middle or plane corner under the cursor, the source the line end of the selection (position at
        the start) nearest to the target - e.g. a chair leg exactly at the seat corner."""
        drag = self.gizmo._drag
        if not self._vsnap or drag is None or self.gizmo.tool != "translate" or isinstance(drag["target"], PivotHandle):
            return None
        if "vpts" not in drag:
            curves = [c for c in (edges.ref_to_curve(self.scene, r) for r in self.edge_selection) if c is not None]
            curves += [c for o in self._with_descendants(self.selected_objects)
                       if isinstance(o, StrokeObject) and not isinstance(o, SurfaceObject) for c in o.curves]
            if not curves:
                return None
            pts = np.array([c.points[e] for c in curves for e in (0, -1)])
            applied, handle = drag["applied"], drag["handle"]
            shift = drag["axes"][:, handle[1]] * applied if handle[0] == "axis" else np.asarray(applied, float)
            drag["vpts"] = pts - shift
            drag["vsrc_ids"] = {id(c) for c in curves}
        dst = self._pivot_snap_point(pos, exclude=drag["vsrc_ids"])
        if dst is None:
            return None
        self._snap_dot = dst
        return drag["vpts"][int(np.argmin(np.linalg.norm(drag["vpts"] - dst, axis=1)))], dst

    def _pivot_snap_point(self, pos, exclude=()):
        """V: nearest line start/end, line middle or plane corner/middle at the cursor.
        exclude: id(line) of the lines that are out of the question (the moved ones themselves)."""
        candidates = []
        for c in self._visible_curves():
            if id(c) in exclude:
                continue
            arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c.points, axis=0), axis=1))])
            mid = np.array([np.interp(arc[-1] / 2, arc, c.points[:, i]) for i in range(3)])
            candidates += [c.points[0], c.points[-1], mid]
        for plane in self._visible_planes():
            candidates += list(plane.corners()) + [plane.origin]
        if not candidates:
            return None
        pts = np.array(candidates)
        screen, ok = self.project(pts)
        d = np.hypot(screen[:, 0] - pos.x(), screen[:, 1] - pos.y())
        d[~ok | np.isnan(d)] = np.inf
        k = int(np.argmin(d))
        return pts[k].copy() if d[k] < PIVOT_SNAP_PX else None

    def set_selection(self, ids, activate=True):
        self.selection = [i for i in ids if self.scene.object_by_id(i) is not None
                          and self._timing_allows(self.scene.object_by_id(i))]  # outliner: flipbook only in timing
        if activate and self.selection:
            self._activate(self.scene.object_by_id(self.selection[-1]))
        self._changed()
        self._commit_selection()  # outliner click: own undo step

    # --- Selection in the undo history ---
    def _selection_state(self):
        """The whole selection as an immutable tuple (objects, edges, points, sub-edges)."""
        return (tuple(self.selection), tuple(self.edge_selection), tuple(self.point_selection),
                tuple(self.seg_selection))

    def _set_selection_state(self, state):
        sel, edges_, pts, segs = state[:4]
        self.selection, self.edge_selection = list(sel), list(edges_)
        self.point_selection, self.seg_selection = list(pts), list(segs)
        self.ws.target_cache = (None, None)
        self._sel_last = state

    def _commit_selection(self):
        """After an input: if only the selection changed, the previous one becomes its own undo step.
        Changes of the scene (checkpoint, undo/redo) remember the selection themselves."""
        scene = self.scene
        if scene.selection_io is None:  # e.g. the start scene (did not come in via set_scene)
            scene.selection_io = (self._selection_state, self._set_selection_state)
        stamp = (id(scene), scene.revision, scene.sel_revision)
        state = self._selection_state()
        last = getattr(self, "_sel_last", None)
        if getattr(self, "_sel_stamp", None) == stamp and last is not None and state != last:
            scene.checkpoint_selection(last)
        self._sel_last = state
        self._sel_stamp = (id(scene), scene.revision, scene.sel_revision)

    def _activate(self, obj):
        if not self._timing_allows(obj):  # flipbook planes only in timing, in timing only them
            return
        if isinstance(obj, Plane):
            self.scene.activate(obj)
        elif isinstance(obj, (MeshObject, SurfaceObject)):
            self.scene.activate_mesh(obj)  # meshes and surfaces are drawing surfaces
        elif isinstance(obj, StrokeObject) and not isinstance(obj, SurfaceObject):
            self.scene.active_stroke = obj  # surfaces take no new strokes

    # Selection as in classic 3D software: click replaces, Shift adds, Ctrl deselects
    @staticmethod
    def _how(mods):
        if mods & Qt.ControlModifier and not mods & Qt.ShiftModifier:
            return "remove"
        return "add" if mods & Qt.ShiftModifier else "replace"

    def _apply_selection(self, ids=(), refs=(), how="replace", activate=True, pts=(), segs=()):
        """Select objects (ids), lines (refs), points (pts) and sub-edges (segs) ("replace"),
        add ("add") or deselect ("remove"). Points at the same place count as one."""
        ids, refs, pts, segs = list(ids), list(refs), list(pts), list(segs)
        ids = [i for i in ids if self._timing_allows(self.scene.object_by_id(i))]  # flipbook only in timing
        if how == "replace":
            self.selection, self.edge_selection = list(dict.fromkeys(ids)), list(dict.fromkeys(refs))
            self.point_selection, self.seg_selection = self._unique_points(pts), list(dict.fromkeys(segs))
        elif how == "add":
            self.selection = list(dict.fromkeys(self.selection + ids))
            self.edge_selection = list(dict.fromkeys(self.edge_selection + refs))
            self.point_selection = self._unique_points(self.point_selection + pts)
            self.seg_selection = list(dict.fromkeys(self.seg_selection + segs))
        else:
            self.selection = [i for i in self.selection if i not in ids]
            self.edge_selection = [r for r in self.edge_selection if r not in refs]
            gone = [p for p in (self._point_pos(r) for r in pts) if p is not None]
            self.point_selection = [r for r in self.point_selection if (q := self._point_pos(r)) is None or
                                    not any(np.linalg.norm(q - g) < components.WELD_TOL for g in gone)]
            self.seg_selection = [r for r in self.seg_selection if r not in segs]
        if activate and how != "remove" and ids:
            self._activate(self.scene.object_by_id(ids[-1]))
        elif activate and how != "remove" and (refs or pts or segs):  # line/point clicked: its object becomes active
            owner = self.scene.object_by_id((refs or pts or segs)[-1][1])
            if isinstance(owner, StrokeObject) and not isinstance(owner, SurfaceObject):  # drawing surface stays
                self._activate(owner)
        self._end_paste_gizmo()

    def _end_paste_gizmo(self):
        """The move gizmo after pasting applies only to what was pasted: if something else is selected, the
        previous tool is active again (otherwise it is in the way when drawing on)."""
        if self._paste_restore is not None:
            self.gizmo.tool = self._paste_restore[0]
            self._paste_restore = None

    # --- Edit mode (Tab) and edges ---
    def set_select_mode(self, on):
        """Base states: draw (default) and edit (Tab, as in Blender); plus notes (N)."""
        if self._mode == "extrude":
            self.cancel_extrude()
        if on and self.ws.draw_tool == "eraser":  # S leaves the eraser
            self.ws.draw_tool = self.ws.last_draw_tool
        if self.notes_mode:
            self._leave_notes()
        if self.ws.timing_mode:  # draw/edit leave timing
            self._leave_timing()
        self.select_mode = on
        if not on:
            self.ws.value_tool = None  # hardness/smoothing exist only in edit mode
            self.ws.edit_set = frozenset()  # remembered set of edited line objects (_active_strokes)
        self._select_hover = self._band_end = None
        self.toggles_changed.emit()
        self._changed()

    def set_notes_mode(self, on):
        """Third base state notes (N): create, connect, edit notes; Esc/N back to drawing."""
        if self._mode == "extrude":
            self.cancel_extrude()
        if self.ws.timing_mode:
            self._leave_timing()
        if on:
            self.select_mode = False
            self._select_hover = self._band_end = None
            self.ws.notes_visible = True
        else:
            self._leave_notes()
        self.ws.notes_mode = on
        self.toggles_changed.emit()
        self._changed()

    def _leave_notes(self):
        for vp in self.ws.viewports:
            vp.notes_tool.cancel()
        self.ws.notes_mode = False

    def set_notes_visible(self, on):
        self.ws.notes_visible = on
        if not on and self.notes_mode:
            self._leave_notes()
        self.toggles_changed.emit()
        self._changed()

    def _edge_under(self, pos, candidates=None, tolerance=PICK_PX, visible=False):
        """(ref, points, 3D point) of the nearest line within tolerance pixels, otherwise None.
        visible: lines behind the surface under the cursor do not count (except X-ray) - otherwise on double-click
        a see-through line far behind wins just because it is 1 px closer to the cursor (harbor test 09.11.)."""
        if candidates is None:
            found = self._near_segments(pos, None, tolerance, cache="edges")
        else:
            found = self._near_segments(pos, list(candidates), tolerance)
        if not found:
            return None
        pts_of = dict(self._visible_edges() if candidates is None else candidates)
        if visible and not self.xray:
            # per candidate at ITS pixel: does a surface lie clearly in front there? (compared at the cursor pixel, at
            # a flat view lines on the same surface dropped out - depth changes strongly per pixel there)
            view = -self.cam.back
            pts3, idx = [], []
            for n, f in enumerate(found):
                pts = pts_of.get(f[1])
                if pts is not None and f[2] + 1 < len(pts):
                    pts3.append(pts[f[2]] + (pts[f[2] + 1] - pts[f[2]]) * f[3])
                    idx.append(n)
            if pts3:
                screen, ok = self.project(np.array(pts3))
                hits = self._mesh_samples([QPointF(float(x), float(y)) for x, y in screen])
                drop = set()
                for n, p, (x, y), visible_px, (_m, front, _d) in zip(idx, pts3, screen, ok, hits):
                    if not visible_px or front is None:
                        continue
                    depth = float((front - self.cam.eye) @ view)
                    margin = max(0.01 * abs(depth), 4 * float(self._world_per_px(abs(depth))))
                    if float((p - self.cam.eye) @ view) > depth + margin:
                        drop.add(n)
                found = [f for n, f in enumerate(found) if n not in drop]
                if not found:
                    return None
        d, ref, i, t = min(found, key=lambda f: f[0])
        pts = pts_of.get(ref)
        if pts is None or i + 1 >= len(pts):  # cache outdated: once without it
            self.ws.seg_cache = None
            return self._edge_under(pos, list(self._visible_edges()), tolerance)
        return ref, pts, self._segment_point(pos, pts[i], pts[i + 1], t)

    def _select_click(self, obj, how):
        """Object by click (click on empty space without a key clears everything)."""
        self._apply_selection([obj.id] if obj is not None else [], (), how)

    def _select_edge_click(self, ref, how):
        self._apply_selection((), [ref], how)

    def select_linked(self, mode="object"):
        """L ("object"), Shift+L ("all"), Ctrl+L ("loop") - see _select_linked / _select_flow_loop."""
        if self.ws.component == "point" and self.point_selection:  # via the lines of the points
            self._convert_selection("edge")
            self._select_linked(mode)
            return self._convert_selection("point")
        if mode == "loop":
            return self._select_flow_loop()
        self.edge_selection = list(dict.fromkeys(self.edge_selection + [("line", r[1], r[2]) for r in self.seg_selection]))
        self.seg_selection = []
        return self._select_linked(mode)

    def _select_linked(self, mode="object"):
        """L: also select everything connected to the selection - connected is what shares a point id (ends,
        crossings, T-junctions), never by position. "object": only within the own stroke object, "all" (Shift+L):
        also across object boundaries. Without a selection nothing happens."""
        curves = self._visible_curves()
        owner = {c.id: s.id for s in self.scene.strokes for c in s.curves}
        selected = set(self.selection)
        start = {c.id for c in (edges.ref_to_curve(self.scene, r) for r in self.edge_selection) if c is not None and not c.paint}
        start |= {c.id for c in curves if owner.get(c.id) in selected}
        if not start:
            return
        at_pid = {}
        for c in curves:
            for pid in set((c.pids or {}).values()):
                at_pid.setdefault(pid, []).append(c)
        by_id = {c.id: c for c in curves}
        seen, stack = set(start), [cid for cid in start if cid in by_id]
        while stack:
            cid = stack.pop()
            for pid in set((by_id[cid].pids or {}).values()):
                for other in at_pid[pid]:
                    if other.id not in seen and (mode == "all" or owner.get(other.id) == owner.get(cid)):
                        seen.add(other.id)
                        stack.append(other.id)
        refs = [edges.curve_to_ref(self.scene, c) for c in curves if c.id in seen]
        self._apply_selection((), [r for r in refs if r is not None], "add", activate=False)

    def _select_flow_loop(self):
        """Ctrl+L: loop following the flow of the net (like an edge loop in a poly modeler) - from the selected edges
        straight on at every point: at a crossing (4 edges) the opposite one, at a point with 2 edges the
        other; at points with 3 or more than 4 edges (T-junction, star) and at free ends it stops."""
        curves = self._visible_curves()
        pieces = self._line_pieces(curves)
        start = set()
        for k, (c, a, b, _, _) in enumerate(pieces):
            for r in self.edge_selection:
                if r[0] == "line" and r[2] == c.id:
                    start.add(k)
            for r in self.seg_selection:
                if r[2] == c.id and a < r[4] and b > r[3]:
                    start.add(k)
        if not start:
            return
        ends_at = {}
        for k, (_, _, _, na, nb) in enumerate(pieces):
            ends_at.setdefault(na, []).append((k, 0))
            ends_at.setdefault(nb, []).append((k, 1))

        def leave(k, end):  # direction in which the piece leaves the node
            c, a, b, _, _ = pieces[k]
            d = c.points[a + 1] - c.points[a] if end == 0 else c.points[b - 1] - c.points[b]
            return d / max(float(np.linalg.norm(d)), 1e-12)

        chosen = set(start)
        for k0 in start:
            for end0 in (0, 1):
                k, end = k0, end0
                while True:
                    node = pieces[k][3 + end]
                    others = [(q, e) for q, e in ends_at[node] if (q, e) != (k, end)]
                    if len(others) + 1 not in (2, 4):
                        break
                    come = -leave(k, end)
                    q, e = max(others, key=lambda qe: float(come @ leave(*qe)))
                    if q in chosen:
                        break
                    chosen.add(q)
                    k, end = q, 1 - e
        refs, segs = [], []
        whole = {}
        for k, (c, a, b, _, _) in enumerate(pieces):
            whole.setdefault(c.id, []).append(k in chosen)
        for k in sorted(chosen):
            c, a, b, _, _ = pieces[k]
            ref = edges.curve_to_ref(self.scene, c)
            if ref is None:
                continue
            if all(whole[c.id]):
                refs.append(ref)
            else:
                segs.append(("seg", ref[1], ref[2], a, b))
        self._apply_selection((), list(dict.fromkeys(refs)), "add", activate=False, segs=segs)

    def _rect_samples(self, pts, rect):
        """Points of a polyline that lie in the screen rectangle: (3D points, screen).
        Lines that merely cross the rectangle also hit (pieces are sampled finely)."""
        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        screen, ok = self.project(pts)
        if len(pts) < 2 or not ok.any():
            return None
        a, b = screen[:-1], screen[1:]
        near = (ok[:-1] & ok[1:] & (np.maximum(a[:, 0], b[:, 0]) >= x0) & (np.minimum(a[:, 0], b[:, 0]) <= x1)
                & (np.maximum(a[:, 1], b[:, 1]) >= y0) & (np.minimum(a[:, 1], b[:, 1]) <= y1))
        if not near.any():
            return None
        # sample the affected pieces in one step: n+1 points per piece, t from 0 to 1
        idx = np.flatnonzero(near)
        n = np.clip(np.ceil(np.linalg.norm(b[idx] - a[idx], axis=1) / BOX_SAMPLE_PX), 1, 512).astype(int)
        seg = np.repeat(idx, n + 1)
        starts = np.concatenate([[0], np.cumsum(n + 1)[:-1]])
        t = (np.arange(len(seg)) - np.repeat(starts, n + 1)) / np.repeat(n, n + 1)
        P = pts[seg] + t[:, None] * (pts[seg + 1] - pts[seg])
        s, sok = self.project(P)
        inside = sok & (s[:, 0] >= x0) & (s[:, 0] <= x1) & (s[:, 1] >= y0) & (s[:, 1] <= y1)
        return (P[inside], s[inside]) if inside.any() else None

    def _edges_in_rect(self, rect):
        """All lines and surface edges that touch the rectangle - only visible ones (except X-ray)."""
        found = []  # (ref, 3D test points, screen)
        for ref, pts in self._visible_edges():
            hit = self._rect_samples(pts, rect)
            if hit is not None:
                k = np.linspace(0, len(hit[0]) - 1, min(len(hit[0]), BOX_MAX_CHECKS)).astype(int)
                found.append((ref, hit[0][k], hit[1][k]))
        if self.xray or not found:
            return [f[0] for f in found]
        checks = [(i, j) for i, f in enumerate(found) for j in range(len(f[1]))]
        positions = [QPointF(*found[i][2][j]) for i, j in checks]
        visible = set()
        for (i, j), pos, (mesh, _, m_dist) in zip(checks, positions, self._mesh_samples(positions, exact=False)):
            if i in visible:
                continue
            if mesh is None:
                visible.add(i)
                continue
            origin, d = self.ray(pos)
            if m_dist >= (found[i][1][j] - origin) @ d - OCCLUDE_TOL * self.cam.dist:
                visible.add(i)
        return [f[0] for i, f in enumerate(found) if i in visible]

    def _meshes_in_rect(self, rect):
        """Meshes visible in the rectangle (pick buffer). X-ray: every mesh checked individually,
        so occluded ones are found too."""
        meshes = self._visible_meshes()
        if not meshes or self.renderer is None:
            return []
        x0, y0, _ = self._physical(rect.bottomLeft())
        x1, y1, size = self._physical(rect.topRight())
        groups = [[m] for m in meshes] if self.xray else [self._pick_items(self._pickables())]
        def gpu():
            found = []
            for group in groups:
                self.renderer.update_pick(size, self.cam, self._projection(), group)
                data, _, _ = self.renderer.read_pick(x0, y0, x1 - x0 + 1, y1 - y0 + 1)
                found += [self.renderer.pick_mesh(int(i)) for i in np.unique(np.round(data[..., 3])) if i > 0]
            return found
        found = self._with_gpu(gpu)
        return [m for m in found if isinstance(m, MeshObject)]

    def _band_select(self, rect, how):
        """Selection rectangle depending on mode: points (1) in it; edges (2): touched lines or sub-edges,
        planes (at their border) and visible meshes; surfaces (3): drawing surfaces, planes, meshes."""
        comp = self.ws.component
        if comp == "point":
            return self._apply_selection((), (), how, activate=False, pts=self._points_in_rect(rect))
        ids = [p.id for p in (self.scene.custom if self.planes_visible else [])
               if self._rect_samples(np.vstack([p.corners(), p.corners()[:1]]), rect) is not None]
        ids += [m.id for m in self._meshes_in_rect(rect)]
        if comp == "face":
            ids += [s.id for s in self._visible_surfaces()
                    if any(self._rect_samples(o, rect) is not None for o in s.outlines())]
            return self._apply_selection(ids, (), how, activate=False)
        lines, segs = [], []
        for ref in self._edges_in_rect(rect):
            c = edges.ref_to_curve(self.scene, ref)
            parts = components.segments(c) if c is not None else []
            hit = [(a, b) for a, b in parts if self._rect_samples(c.points[a:b + 1], rect) is not None]
            if len(parts) <= 1 or len(hit) == len(parts) or not hit:
                lines.append(ref)
            else:
                segs += [("seg", ref[1], ref[2], a, b) for a, b in hit]
        self._apply_selection(ids, lines, how, activate=False, segs=segs)

    # --- Extrude (X) from the edge selection ---
    def _surface_normal(self, curves):
        """Default direction for extruding: the same direction as just now if the selection is the copy of the
        last extrusion; otherwise the normal of the plane the lines were drawn on; otherwise the
        normal of the best-fit plane of all selected lines together (a ring split into pieces lies
        as a whole in one plane, a single piece often not unambiguously); for straight lines world Y (or X
        if they stand vertically)."""
        last = self._last_extrude
        if last is not None and {c.id for c in curves} == last[0]:
            return last[1]
        planes = {c.plane_id for c in curves}
        plane = self.scene.plane_by_id(next(iter(planes))) if len(planes) == 1 else None
        if plane is not None:
            return plane.normal
        allp = np.vstack([c.points for c in curves])
        pts = allp - allp.mean(axis=0)
        _, sv, vt = np.linalg.svd(pts, full_matrices=False)
        if len(sv) > 1 and sv[1] > 1e-3 * max(sv[0], 1e-12):
            n = vt[2]
            up = n[1] if abs(n[1]) > 0.1 else n @ self.cam.back  # upward or toward the camera
            return n if up >= 0 else -n
        direction = (allp[-1] - allp[0]) / max(np.linalg.norm(allp[-1] - allp[0]), 1e-12)
        return np.array([0, 1.0, 0]) if abs(direction[1]) < 0.9 else np.array([1.0, 0, 0])

    def _face_normal(self, faces):
        """Extrude surfaces: mean normal of the selected surfaces (area-weighted), upward or toward the camera."""
        total = np.zeros(3)
        for f in faces:
            mesh = f.fill_mesh()
            if mesh is None:
                continue
            v, t = mesh[0], mesh[1]
            total += np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]]).sum(axis=0)
        n = float(np.linalg.norm(total))
        if n < 1e-12:
            return self._surface_normal([c for f in faces for c in extruding.face_outline(self.scene, f) or []])
        total /= n
        up = total[1] if abs(total[1]) > 0.1 else total @ self.cam.back
        return total if up >= 0 else -total

    # --- Fill (F) from the edge selection ---
    def jump_to_object(self, pos, how):
        """Double-click in point/edge mode on another object: it becomes the edited one - selected in the outliner and
        viewport, active, its points appear. Line: its line object; surface: the surface together with the
        line objects of its lines (surfaces are references to lines); mesh: the mesh. False if the object is already
        being edited (then the double-click is meant for the loop)."""
        sc = self.scene
        hit = self._edge_under(pos, tolerance=LINE_EXACT_PX, visible=True)
        surface = None
        if hit is not None:
            owner = sc.object_by_id(hit[0][1])
            strokes = [owner] if isinstance(owner, StrokeObject) and not isinstance(owner, SurfaceObject) else []
        else:
            surface = self._mesh_under(pos)[0]
            if surface is None:
                return False
            # frozen helper surface: only itself (move with the gizmo, its lines stay)
            lines = self._lines_on_surface(surface) if isinstance(surface, SurfaceObject) and surface.frozen is None \
                else []
            count = {}
            for c in lines:
                if (o := sc.owner_of(c)) is not None:
                    count[o.id] = count.get(o.id, 0) + 1
            strokes = [sc.object_by_id(i) for i in sorted(count, key=count.get)]  # the most lines last: active
        editing = self._active_strokes()
        if strokes and all(s.id in editing for s in strokes):
            return False  # their lines are already being edited: loop or surface with its lines
        if not strokes and (surface is None or surface.id in self.selection):
            return False
        ids = ([surface.id] if surface is not None else []) + [s.id for s in strokes]
        self._apply_selection(ids, (), how)
        if surface is not None and strokes:
            sc.activate_mesh(surface)  # drawing surface as with a click in draw mode, line object stays active
        names = ", ".join(sc.object_by_id(i).name for i in ids)
        self.notify(tr("Bearbeitet jetzt: {0} – Doppelklick auf etwas anderes springt weiter").format(names))
        return True

    def select_loop(self, pos, how):
        """Double-click: smallest loop through the edge under the cursor, again: next loop.
        No line, but a drawing surface below: the surface with all lines lying on it."""
        # lines only exactly under the cursor (in dense drawings there is almost one everywhere within the normal snap
        # radius) - beside it the double-click is meant for the drawing surface
        hit = self._edge_under(pos, tolerance=LINE_EXACT_PX, visible=True)
        if hit is None:
            surface = self._mesh_under(pos)[0]
            if surface is None:
                plane = self._plane_under(pos)[0]
                if plane is not None:  # plane: with lines and painting on it
                    curves = self._lines_on_plane(plane)
                    refs = [r for r in (edges.curve_to_ref(self.scene, c) for c in curves) if r is not None]
                    self._apply_selection([plane.id] if not plane.preset else [], refs, how)
                    self.notify(tr("{0} mit {1} Linie(n) darauf gewählt – gemeinsam transformierbar").format(plane.name, len(refs)))
                    return
            if isinstance(surface, SurfaceObject) and surface.frozen is not None:  # helper surface: only it, without lines
                self._apply_selection([surface.id], (), how)
                self.notify(tr("{0} gewählt (eingefroren) – mit dem Gizmo bewegen, ihre Linien bleiben").format(surface.name))
                return
            if isinstance(surface, SurfaceObject):
                refs = [edges.curve_to_ref(self.scene, c) for c in self._lines_on_surface(surface)]
                self._apply_selection([surface.id], [r for r in refs if r is not None], how)
                self.notify(tr("{0} mit {1} Linie(n) darauf gewählt – gemeinsam transformierbar").format(surface.name, len(refs)))
                return
            hit = self._edge_under(pos, visible=True)
        curve = edges.ref_to_curve(self.scene, hit[0]) if hit else None
        if curve is None:
            return
        graph = self._edge_graph(self._visible_curves())
        _, i, _ = self.scene._closest_on(curve, hit[2])
        start = next((e for e in graph if e[0] == curve.id and e[1] <= i < e[2]),
                     next(e for e in graph if e[0] == curve.id))
        filled = [{c.id for c in sf.curves} for sf in self.scene.surfaces]
        cycles = [c for c in fill.shortest_cycles(graph, start, LOOP_CHOICES + len(filled))
                  if {e[0] for e in c} not in filled]  # do not offer already filled outlines again
        cycles = cycles[:LOOP_CHOICES] or [[start]]
        last_id, index = self._fill_last
        index = index + 1 if last_id == start and how == "replace" else 0
        self._fill_last = (start, index)
        refs, segs = [], []
        for cid, a, b in cycles[index % len(cycles)]:
            c = self.scene.curve_by_id(cid)
            ref = edges.curve_to_ref(self.scene, c) if c is not None else None
            if ref is None:
                continue
            if a == 0 and b == len(c.points) - 1:
                refs.append(ref)
            else:  # piece up to the shared point (T-junction): sub-edge, F/X/Del split the line there
                segs.append(("seg", ref[1], ref[2], a, b))
        self._apply_selection((), refs, how, segs=segs)

    def _line_pieces(self, curves):
        """Lines split at points they share with other lines (crossing, T-junction) into pieces
        [(line, a, b, node_a, node_b)]. Node = point id - connected is only what shares one, never by
        position (an end without an id is its own node)."""
        count = {}
        for c in curves:
            for pid in set((c.pids or {}).values()):
                count[pid] = count.get(pid, 0) + 1
        out = []
        for c in curves:
            n = len(c.points)
            if n < 2:
                continue
            closed = components.is_closed(c.points)

            def node(k):
                if closed and k == n - 1:
                    k = 0
                pid = network.pid_of(c, k)
                return pid if pid is not None else ("frei", c.id, k)

            ks = components.knots(c)
            cuts = [k for k in ks[1:-1] if count.get(network.pid_of(c, k), 0) > 1]
            bounds = [0, *cuts, n - 1]
            out += [(c, a, b, node(a), node(b)) for a, b in zip(bounds[:-1], bounds[1:])]
        return out

    def _edge_graph(self, curves):
        """Curve net as a graph for loops: {(line_id, a, b): (node_a, node_b, length)} from _line_pieces."""
        graph = {}
        for c, a, b, na, nb in self._line_pieces(curves):
            length = float(np.linalg.norm(np.diff(c.points[a:b + 1], axis=0), axis=1).sum())
            graph[(c.id, a, b)] = (na, nb, length)
        return graph

    def _lines_on_plane(self, plane):
        """Lines (and painting) lying entirely in the plane - within its rectangle or painted on it."""
        size = max(plane.size_u, plane.size_v, 1e-9)
        tol = ON_SURFACE_TOL * size
        out = []
        for c in self._visible_curves(include_paint=True):
            if c.paint and c.plane_id == plane.id:
                out.append(c)
                continue
            d = (c.points - plane.origin) @ plane.normal
            if len(c.points) and np.all(np.abs(d) <= tol) and (plane.preset or all(plane.contains(p) for p in c.points)):
                out.append(c)
        return out

    def _lines_on_surface(self, surface):
        """Lines connected to the drawing surface: lying entirely on it (outline, lines drawn
        on it, lines along its edges) - not those that merely touch it at one point
        or pierce it."""
        mesh = surface.fill_mesh()
        if mesh is None:
            return []
        verts = mesh[0]
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        size = max(float(np.linalg.norm(hi - lo)), 1e-9)
        tol = ON_SURFACE_TOL * size
        out = []
        for c in self._visible_curves(include_paint=True):
            # purely geometric: a line once drawn on it no longer lies on it after the surface is moved
            # (without its lines)
            if np.any(c.points.min(axis=0) > hi + tol) or np.any(c.points.max(axis=0) < lo - tol):
                continue
            arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c.points, axis=0), axis=1))])
            if arc[-1] <= 0:
                continue
            t = np.linspace(0, arc[-1], ON_SURFACE_SAMPLES)
            samples = np.stack([np.interp(t, arc, c.points[:, i]) for i in range(3)], axis=1)
            if np.all(_near_mesh(samples, mesh, tol)):
                out.append(c)
        return out
