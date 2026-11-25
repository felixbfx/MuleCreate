"""Surface operations (loft, curve surface, fill, split) and object commands (delete, rename, planes)."""
from .common import *  # noqa: F401,F403
from ..i18n import tr
import hashlib
import os
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QFileDialog
from .. import pictures


GAP_SNAP = 0.02  # gap in the outline shorter than this fraction of the shorter line: shared point instead of bridge


class SurfaceOpsMixin:
    def _track_pick_order(self):
        """In which order were lines selected? {line id: rank}. What was added in one move (rectangle,
        double click) shares a rank; deselected ones drop out. The loft takes its cross-sections in this
        order."""
        ids = [r[2] for r in self.edge_selection if r[0] == "line"] + [r[2] for r in self.seg_selection]
        rank = getattr(self, "_pick_rank", None)
        if rank is None:
            rank = self._pick_rank = {}
            self._pick_count = 0
        now = set(ids)
        for cid in [c for c in rank if c not in now]:
            del rank[cid]
        new = [c for c in dict.fromkeys(ids) if c not in rank]
        if new:
            self._pick_count += 1
            for cid in new:
                rank[cid] = self._pick_count

    def do_loft(self, wrap=False):
        # A selected piece of a line is the cross-section – exactly this piece, like F and Alt+F take it (artist
        # 2026-11-24: two pieces of two rings selected to bridge them, the loft took both whole rings). The line is
        # split at the points of the piece; whoever wants the whole line selects the whole line.
        self._track_pick_order()  # order of the selection (even if nothing was drawn since the last click)
        segs = self._valid_segs()
        if not segs:
            self.seg_selection = []
            return self._do_loft(wrap)
        rank = dict(getattr(self, "_pick_rank", None) or {})
        keep = (list(self.edge_selection), list(self.seg_selection))
        parents = {c.id for _r, c in segs}
        count = len(self.scene.surfaces)
        self._materialize_segments()  # (saves the undo step; the loft's own checkpoint merges into it)
        picked = {r[2] for r in self.edge_selection}
        for cid in parents:  # the pieces stand where their line stood in the order of the selection
            for piece in self.scene.split_map.get(cid, ()):
                if piece in picked and cid in rank:
                    self._pick_rank[piece] = rank[cid]
        self._do_loft(wrap)
        if len(self.scene.surfaces) == count:  # no loft came of it: the lines stay as they were
            self.scene.undo()
            self.scene._redo.clear()
            self.scene._burst = (None, 0.0)
            self.edge_selection, self.seg_selection = keep
            self._changed()

    def _do_loft(self, wrap=False):
        """Shift+F: curved drawing surface through the selected cross-sections – lines (connected
        pieces count as one cross-section) and/or drawing surfaces (their outline). The loft surface follows
        its cross-sections when they are moved later."""
        curves = [c for c in dict.fromkeys(edges.ref_to_curve(self.scene, r) for r in self.edge_selection) if c is not None and not c.paint]
        sources, sections = [], []
        for group in self._connected_groups(curves):
            tol = 1e-6 * max(float(np.linalg.norm(np.ptp(np.vstack([c.points for c in group]), axis=0))), 1.0)
            chains = extrude.chains([c.points for c in group], tol)
            if len(chains) != 1:
                self.status_changed.emit(tr("Loft: ein Querschnitt verzweigt sich – bitte nur einfache Linienzüge wählen."))
                return
            sources.append(("lines", [c.id for c in group]))
            sections.append(chains[0][0])
        # drawing surfaces as cross-section only in surface mode (or without selected lines) – otherwise a still
        # selected surface (e.g. the loft just created) would count unintentionally
        use_surfaces = self.ws.component == "face" or not curves
        for obj in (self.selected_objects if use_surfaces else []):
            if isinstance(obj, SurfaceObject) and obj.outlines():
                sources.append(("surface", obj.id))
                sections.append(obj.outlines()[0])
        if len(sections) < (3 if wrap else 2):
            self.status_changed.emit(tr("Loft: mindestens zwei Querschnitte wählen (Linien oder Zeichenflächen), dann Shift+F – ringförmig (Strg+Shift+F) mindestens drei."))
            return
        order = self._loft_order(sources, sections)
        self.scene.checkpoint()
        before = ({c.id for c in self.scene.curves}, {x.id for x in self.scene.surfaces})
        owner = next((self.scene.owner_of(c) for c in curves if self.scene.owner_of(c) is not None), None)
        parts = self._loft_hard_parts([sources[i] for i in order], [sections[i] for i in order], wrap)
        if parts:  # hard points (kink from 15°): between them a loft of its own per piece = own curve surface
            made = []
            for srcs, secs in parts:
                piece = self.scene.create_loft(srcs, secs)
                self.scene.add_loft_rails(piece, self.cam.back, owner)
                made.append(self.scene.loft_to_patch(piece) or piece)
            self._consolidate_new(before)
            made = [self._replaced.get(m.id, m) for m in made]
            self.scene.activate_mesh(made[0])
            self.selection = [f.id for f in made]
            self.edge_selection = []
            self._changed()
            self.notify(tr("Loft an harten Punkten geteilt: {0} Flächen durch {1} Querschnitte").format(len(made), len(sections)))
            return
        surface = self.scene.create_loft([sources[i] for i in order], [sections[i] for i in order], wrap=wrap)
        self.scene.add_loft_rails(surface, self.cam.back, owner)
        patch = self.scene.loft_to_patch(surface)
        if patch is not None:  # open loft: from now on a curve surface like any other
            surface = patch
        self._consolidate_new(before)
        if surface.id in self._replaced:  # unambiguous only after merging: now curve surface
            surface = patch = self._replaced[surface.id]
        self.scene.activate_mesh(surface)  # be able to draw right afterwards
        self.selection = [surface.id]
        self.edge_selection = []
        self._changed()
        kind = tr("Kurvenfläche (als Loft entstanden)") if patch is not None else \
            (tr("ringförmiger Loft") if wrap else tr("Röhre") if surface.closed else "Loft")
        self.notify(tr("{0}: {1} durch {2} Querschnitte").format(surface.name, kind, len(sections)))

    def _consolidate_new(self, before):
        """After loft/extrude: weld new lines to existing points exactly beneath them and merge duplicates
        with the existing line (topology.consolidate) – never two lines between the same points."""
        from .. import topology
        sc = self.scene
        self._replaced = {}
        old = set(before[0])
        grew = True
        while grew:  # pieces of split existing lines are not new (extrude splits foreign lines)
            add = {p for cid in old for p in sc.split_map.get(cid, ())} - old
            grew = bool(add)
            old |= add
        new = [c for c in sc.curves if c.id not in old and not c.paint]
        if not new:
            return []
        size = max(float(np.linalg.norm(np.ptp(np.vstack([c.points for c in new]), axis=0))), 1e-9)
        gone = topology.consolidate(sc, new, weld_tol=1e-5 * size)
        if gone:
            # lofts of this operation that had to stay loft because of the double line (side not unambiguous): now
            # curve surface – they then follow their border lines like the neighboring wall
            for s in [x for x in sc.surfaces if x.id not in before[1]]:
                if isinstance(s, LoftSurface) and not s.closed and not s.wrap and len(s.sources or []) >= 2                         and all(src is not None and src[0] == "lines" for src in s.sources):
                    patch = sc.loft_to_patch(s)
                    if patch is not None:
                        self._replaced[s.id] = patch
            self.notify(tr("{0} doppelte Linie(n) mit vorhandenen zusammengelegt").format(len(gone)))
        return gone

    def _loft_hard_parts(self, sources, sections, wrap):
        """Hard points of the first cross-section (knots with a kink from extruding.HARD_KNOT_DEG): split all cross-sections
        there (at the corresponding place, snapped to a point of their own nearby) -> [(sources, cross-sections)]
        per piece in between, or None (no hard points, ring, drawing surfaces as cross-section; closed: only from
        two hard points, otherwise it stays a tube)."""
        from .. import extruding
        sc = self.scene
        if wrap or any(src[0] != "lines" for src in sources):
            return None
        lines_of = lambda src: [c for cid in src[1] for sub in (sc.split_pieces(cid) or []) if (c := sc.curve_by_id(sub))]
        aligned, closed = loft.prepare(sections)
        aligned = np.asarray(aligned, float)
        m = aligned.shape[1]
        size = max(float(np.linalg.norm(np.ptp(aligned[0], axis=0))), 1e-9)
        cols = set()
        for p, kink, _ in extruding.knot_marks(sc, lines_of(sources[0])):
            if not extruding.HARD_KNOT_DEG <= kink < 179.0:  # 180 = free end (is already border)
                continue
            d = np.linalg.norm(aligned[0] - p, axis=1)
            col = int(np.argmin(d))
            if d[col] <= 0.03 * size and (closed or 0 < col < m - 1):
                cols.add(col)
        cols = sorted(cols)
        if not cols or closed and len(cols) < 2:
            return None
        tol = 1e-6 * size
        for k, src in enumerate(sources):  # split each cross-section at the corresponding places
            lines = lines_of(src)
            knots = [c.points[j] for c in lines for j in components.knots(c)]
            sec_size = max(float(np.linalg.norm(np.ptp(aligned[k], axis=0))), 1e-9)
            for col in cols:
                p = aligned[k][col]
                near = min(knots, key=lambda q: float(np.linalg.norm(q - p)), default=None)
                if near is not None and np.linalg.norm(near - p) <= 0.05 * sec_size:
                    p = near  # there the cross-section itself has a point (its corner)
                for c in lines_of(src):
                    d, i, t = sc._closest_on(c, p)
                    if d <= max(tol, 1e-9 * sec_size) + 1e-6 * sec_size:
                        sc._split_in(sc.owner_of(c), c, np.asarray(p, float), i, t)
                        break
        network.sync(sc)
        bounds = cols if closed else [0] + cols + [m - 1]
        ranges = [(bounds[i], bounds[(i + 1) % len(bounds)]) for i in range(len(bounds) if closed else len(bounds) - 1)]

        def inside(col, a, b):
            return a < col < b if a < b else (col > a or col < b)

        parts = []
        for a, b in ranges:
            srcs, secs = [], []
            for k, src in enumerate(sources):
                chosen = []
                for c in lines_of(src):
                    mid = c.points[len(c.points) // 2]
                    col = int(np.argmin(np.linalg.norm(aligned[k] - mid, axis=1)))
                    if inside(col, a, b):
                        chosen.append(c)
                if not chosen:
                    return None  # cross-section does not fit (very unevenly distributed): better unsplit
                tol_k = 1e-6 * max(float(np.linalg.norm(np.ptp(np.vstack([c.points for c in chosen]), axis=0))), 1.0)
                chain = extrude.chains([c.points for c in chosen], tol_k)
                if len(chain) != 1:
                    return None
                srcs.append(("lines", [c.id for c in chosen]))
                secs.append(chain[0][0])
            parts.append((srcs, secs))
        return parts

    def _loft_order(self, sources, sections):
        """Order of the cross-sections: as selected (rank of the first selected line of a cross-section). Only if that is
        not unambiguous (selected in one move, drawing surfaces as cross-section) is it guessed from the position."""
        rank = getattr(self, "_pick_rank", {}) or {}
        keys = []
        for src in sources:
            r = [rank[cid] for cid in src[1] if cid in rank] if src[0] == "lines" else []
            keys.append(min(r) if r else None)
        if None not in keys and len(set(keys)) == len(keys):
            return sorted(range(len(sources)), key=lambda i: keys[i])
        return loft.order(sections)

    def _connected_groups(self, curves):
        """Group lines that share a point at their ends (one cross-section per group) –
        via the curve net (point IDs), not via position. In the order of the selection."""
        network.sync(self.scene)
        ends = {id(c): {network.pid_of(c, 0), network.pid_of(c, len(c.points) - 1)} - {None} for c in curves}
        groups = []
        for c in curves:
            hits = [i for i, g in enumerate(groups) if any(ends[id(c)] & ends[id(o)] for o in g)]
            if not hits:
                groups.append([c])
                continue
            merged = [o for i in hits for o in groups[i]] + [c]
            at = hits[0]  # where the earliest part stood
            groups = [g for i, g in enumerate(groups) if i not in hits]
            groups.insert(at, merged)
        return groups

    def do_patch(self):
        """Alt+F: curve surface – smooth, curved drawing surface from the selected border curves (Coons patch)."""
        self._materialize_segments()
        curves = [c for c in dict.fromkeys(edges.ref_to_curve(self.scene, r) for r in self.edge_selection) if c is not None and not c.paint]
        if not curves:
            return self.notify(tr("Kurvenfläche: erst die Randkurven wählen (Edit, Kanten), dann Alt+F"))
        self.scene.checkpoint()
        curves, bridges, snapped = self._close_outline(curves)  # gap rule as with filling
        surface = self.scene.create_patch(curves)
        if surface is None:
            self.scene.undo()
            self.scene._redo.clear()
            return self.notify(tr("Kurvenfläche: die gewählten Kurven bilden keinen geschlossenen Umriss"))
        self.scene.activate_mesh(surface)
        self.selection = [surface.id]
        self.edge_selection = []
        self._changed()
        self.notify(tr("{0}: glatte Fläche aus {1} Randkurve(n){2}").format(surface.name, len(curves), self._gap_report(bridges, snapped)))

    def divide_loft(self):
        """Alt+D: split loft – whole loft selected: a new one between each pair of cross-sections; two or more
        adjacent cross-sections selected: only between them."""
        sc = self.scene
        lofts = [o for o in self.selected_objects if isinstance(o, LoftSurface)]
        chosen = {c.id for c in (edges.ref_to_curve(sc, r) for r in self.edge_selection) if c is not None}
        jobs = []
        for f in lofts:
            n = len(sc.net_rows_lines(f)) if isinstance(f, PatchSurface) else len(f.sections())
            if n > 1:
                jobs.append((f, list(range(n - 1))))
        if not jobs and chosen:
            for f in sc.surfaces:
                if isinstance(f, PatchSurface):  # open loft = curve surface: adjacent cross rows of the net
                    hit = [i for i, ids in enumerate(sc.net_rows_lines(f)) if ids & chosen]
                    pairs = [i for i in hit if (i + 1) in hit]
                    if pairs:
                        jobs.append((f, pairs))
                    continue
                if not isinstance(f, LoftSurface):
                    continue
                hit = [i for i, src in enumerate(f.sources) if src and src[0] == "lines" and set(src[1]) & chosen]
                k = len(f.curves)
                pairs = [i for i in hit if (i + 1) in hit]
                if f.wrap and 0 in hit and (k - 1) in hit and k > 2:
                    pairs.append(k - 1)  # ring: last and first cross-section are neighbors
                if pairs:
                    jobs.append((f, pairs))
        if not jobs:
            return self.notify(tr("Loft teilen: ganzen Loft (Flächenmodus) oder zwei benachbarte Querschnitte wählen, dann Alt+D"))
        sc.checkpoint()
        made = 0
        for f, spans in jobs:
            src_line = next((sc.curve_by_id(i) for s in f.sources if s and s[0] == "lines" for i in s[1]
                             if sc.curve_by_id(i) is not None), None)
            if src_line is None and isinstance(f, PatchSurface):
                src_line = next(iter(sc.patch_border(f)), None)
            owner = sc.owner_of(src_line) if src_line is not None else None
            owner = owner or (sc.active_stroke if sc.active_stroke in sc.strokes else sc.new_stroke())
            made += len(sc.divide_patch(f, spans, owner) if isinstance(f, PatchSurface) else sc.divide_loft(f, spans, owner))
        sc.sync_lofts()
        self._changed()
        self.notify(tr("Loft geteilt: {0} neue(r) Querschnitt(e)").format(made))

    def set_surface_material(self, oids, key):
        """Set the material (appearance) of a surface or several (list of ids) – one undo step."""
        oids = oids if isinstance(oids, (list, tuple, set)) else [oids]
        surfaces = [s for s in (self.scene.object_by_id(i) for i in oids)
                    if isinstance(s, SurfaceObject) and not s.aid and s.material != key]
        if not surfaces:
            return
        self.scene.checkpoint()
        for s in surfaces:
            s.material = key
        name = dict(theme.MATERIALS).get(key, key)
        self.notify(f"{surfaces[0].name}: {name}" if len(surfaces) == 1 else tr("{0} Flächen: {1}").format(len(surfaces), name))
        self._changed()

    def set_binding(self, oids, level):
        """Set the binding level for the brief (outliner) on meshes, surfaces and nulls (None: inherit) – one undo step.
        Changes nothing about the appearance."""
        objs = [o for o in (self.scene.object_by_id(i) for i in oids)
                if isinstance(o, (MeshObject, NullObject)) or isinstance(o, SurfaceObject) and not o.aid]
        objs = [o for o in objs if o.binding != level]
        if not objs:
            return
        self.scene.checkpoint()
        for o in objs:
            o.binding = level
        from ..brief import TASK_LEVELS
        name = tr(TASK_LEVELS[level][0]) if level else tr("vom Auftrag bzw. Null darüber")
        self.notify(f"Brief – {objs[0].name if len(objs) == 1 else tr('{0} Objekte').format(len(objs))}: {name}")
        self.scene_changed.emit()
        self.emit_status()

    def set_surface_opacity(self, oid, value):
        """Opacity of a surface (slider in the outliner) – one undo step per drag on the slider."""
        surface = self.scene.object_by_id(oid)
        if not isinstance(surface, SurfaceObject):
            return
        self.scene.checkpoint(burst=("opacity", oid))
        surface.opacity = max(0.0, min(1.0, float(value)))
        self.update_all()

    def set_display_resolution(self, factor):
        """Viewport quality: resolution of all surfaces (× 0.25 … 4) in addition to each surface's slider – only the
        displayed net, the shape stays (no undo step, app setting)."""
        factor = max(0.25, min(4.0, float(factor)))
        if factor == SurfaceObject.display_res:
            return
        SurfaceObject.display_res = factor
        for surface in self.scene.surfaces:
            surface._net_key = None
        self.update_all()

    def set_surface_resolution(self, oid, factor):
        """Resolution of the net of a surface (×0.25 … ×4, slider in the outliner) – the shape stays, only the net becomes
        finer or coarser. One undo step per drag on the slider."""
        surface = self.scene.object_by_id(oid)
        if not isinstance(surface, SurfaceObject):
            return
        self.scene.checkpoint(burst=("resolution", oid))
        surface.resolution = max(0.25, min(4.0, float(factor)))
        surface._net_key = None
        self.update_all()

    def load_picture_dialog(self, oid):
        path, _ = QFileDialog.getOpenFileName(self, tr("Bild laden"), "", f"{tr('Bilder')} ({pictures.FILE_FILTER})")
        if path:
            self.load_picture_file(oid, path)

    def load_picture_file(self, oid, path):
        """Image onto the plane/helper surface: stored in the scene; new from surface direction, otherwise the projection stays."""
        obj = self.scene.object_by_id(oid)
        with open(path, "rb") as fh:
            data = fh.read()
        image = QImage.fromData(data)
        if image.isNull():
            return self.notify(tr("Bild konnte nicht geladen werden: {0}").format(os.path.basename(path)))
        self.scene.checkpoint()
        name = f"picture_{hashlib.sha1(data).hexdigest()[:16]}{os.path.splitext(path)[1].lower()}"
        self.scene.assets[name] = data
        aspect = image.width() / max(image.height(), 1)
        if obj.picture:
            obj.picture.update(asset=name, image_aspect=aspect, on=True)
        else:
            pictures.create(obj, name, aspect)
        self._changed()

    def project_picture(self, oid):
        """Project: image from the current camera onto the object. Lock blocks that – so nothing jumps by accident."""
        obj = self.scene.object_by_id(oid)
        if not obj.picture:
            return
        if obj.picture.get("locked"):
            return self.notify(tr("Bild ist gesperrt – erst Lock lösen, dann neu projizieren"))
        self.scene.checkpoint()
        pictures.set_projection(obj, *pictures.from_camera(self.cam, self._projection(), self.width() / max(self.height(), 1)))
        self._changed()

    def set_picture_on(self, oid, on):
        obj = self.scene.object_by_id(oid)
        if obj.picture and obj.picture.get("on") != bool(on):
            self.scene.checkpoint()
            obj.picture["on"] = bool(on)
            self._changed()

    def set_picture_locked(self, oid, on):
        obj = self.scene.object_by_id(oid)
        if obj.picture and obj.picture.get("locked") != bool(on):
            self.scene.checkpoint()
            obj.picture["locked"] = bool(on)
            self._changed()

    def set_surface_aid(self, oid, on):
        """Surface <-> helper surface: as a helper surface it is only a projection surface to draw on (like a
        helper plane, hidden with the drawing aids, hides nothing). One undo step."""
        surface = self.scene.object_by_id(oid)
        if not isinstance(surface, SurfaceObject) or surface.aid == bool(on):
            return
        self.scene.checkpoint()
        self.notify(f"{surface.name}: " + self._set_aid(surface, bool(on)))
        self._changed()

    def _set_aid(self, surface, on):
        """Helper surface on (frozen) or off (reconnected if its lines are still in place);
        -> hint text."""
        surface.aid = on
        if on:
            self.scene.freeze_surface(surface)
            return tr("Hilfsfläche – eingefroren, nur zum Draufzeichnen; bewegen per Doppelklick + Gizmo")
        if self.scene.unfreeze_surface(surface):
            return tr("wieder eine Fläche mit Material, an ihren Linien")
        return tr("wieder eine Fläche mit Material – bleibt in ihrer Form (ihre Linien liegen nicht mehr an ihr)")

    def toggle_surface_aid(self):
        """Turn selected surfaces into helper surfaces (or back, if all are already helper surfaces)."""
        surfaces = [o for o in self.selected_objects if isinstance(o, SurfaceObject)]
        if not surfaces:
            return self.notify(tr("Hilfsfläche: erst Flächen wählen"))
        on = not all(s.aid for s in surfaces)
        self.scene.checkpoint()
        texts = [self._set_aid(s, on) for s in surfaces]
        self.notify(texts[0] if len(surfaces) == 1 else
                    tr("{0} Fläche(n) ").format(len(surfaces)) + (tr("sind jetzt Hilfsflächen (eingefroren, nur zum Draufzeichnen)") if on
                                                    else tr("haben wieder ihr Material")))
        self._changed()

    def update_all(self):
        for vp in self.ws.viewports:
            vp.update()

    def colorize_surfaces(self, reset=False):
        """Draw mode F: tint the active surface (or all in the multi set) in the current color – Shift+F: back
        to the material as it is."""
        targets = [s for s in self._drawing_surfaces() if isinstance(s, SurfaceObject)]
        if not targets:
            return self.notify(tr("Einfärben: erst eine Fläche zum Zeichnen wählen (Klick), dann F"))
        self.scene.checkpoint()
        for s in targets:
            s.tint = None if reset else tuple(float(x) for x in self.ws.paint_color)
        names = ", ".join(s.name for s in targets)
        self.notify(tr("{0}: Färbung zurückgesetzt").format(names) if reset else tr("{0}: eingefärbt").format(names))
        self._changed()

    def do_surface(self):
        """F: the one surface command. From the selected lines (gaps closed by the gap rule) a
        curve surface is created; if its grid would fold (outline kinked inward: L, U shape, star …), a fill
        surface instead. Decided once on creation – afterwards the surface stays what it is."""
        from .. import patch as patch_geo
        self._materialize_segments()
        curves = [c for c in dict.fromkeys(edges.ref_to_curve(self.scene, r) for r in self.edge_selection)
                  if c is not None and not c.paint]
        if not curves:
            return self.notify(tr("Fläche: erst Linien wählen (Edit, Kanten – Doppelklick für einen Loop), dann F"))
        self.scene.checkpoint()
        closed, bridges, snapped = self._close_outline(curves)
        surface = self.scene.create_patch(closed)
        if surface is not None and patch_geo.folds(np.array([c.points for c in surface.curves])) == 0:
            self.scene.activate_mesh(surface)
            self.selection = [surface.id]
            self.edge_selection = []
            self._changed()
            return self.notify(tr("{0}: Kurvenfläche{1}").format(surface.name, self._gap_report(bridges, snapped)))
        if surface is not None:
            self.scene.surfaces.remove(surface)  # grid folds: fill surface takes any shape
        refs = [r for r in (edges.curve_to_ref(self.scene, c) for c in closed) if r is not None]
        self.edge_selection = refs
        self.scene.merge_next()  # bridges and fill surface: one undo step (if it fails, _do_fill rolls everything back)
        self._do_fill(bridged=(bridges, snapped))

    def do_fill(self):
        self._materialize_segments()
        return self._do_fill()

    def _outline_graph(self, curves):
        """Selected lines as a net via their points (point IDs, explicitly connected – not via position):
        ({line id: (point a, point b)}, {point: (line, end)} for ends)."""
        network.sync(self.scene)
        graph, ends = {}, {}
        for c in curves:
            a, b = network.pid_of(c, 0), network.pid_of(c, len(c.points) - 1)
            if components.is_closed(c.points):
                b = a  # closed line: start and end are one point
            graph[c.id] = (a, b)
            ends.setdefault(a, (c, 0))
            ends.setdefault(b, (c, len(c.points) - 1))
        return graph, ends

    def _close_outline(self, curves):
        """Gap rule for the surface command: close open ends of the selected lines pairwise (the nearest pair
        of different pieces first). If the gap is shorter than GAP_SNAP of the shorter line, the ends become a
        shared point (meeting in the middle; an already connected point stays put, the lines bend softly toward it
        on their last piece), otherwise a real, straight line bridges the gap. -> (lines including bridges,
        number of bridges, number of connected points)"""
        sc = self.scene
        graph, ends = self._outline_graph(curves)
        nodes = {}
        for cid, (a, b) in graph.items():
            nodes[a] = nodes.get(a, 0) + 1
            nodes[b] = nodes.get(b, 0) + 1
        points = {pid: c.points[k].copy() for pid, (c, k) in ends.items()}
        pairs = fill.close_open_ends({cid: (a, b, 0.0) for cid, (a, b) in graph.items()}, points)
        refs = network.refs_by_pid(sc)
        length = lambda c: float(np.linalg.norm(np.diff(c.points, axis=0), axis=1).sum())
        owner = next((sc.owner_of(c) for c in curves if sc.owner_of(c) is not None), None) or sc.active_stroke \
            or sc.new_stroke()
        out, bridges, snapped = list(curves), 0, 0
        for p, q in pairs:
            (ca, ka), (cb, kb) = ends[p], ends[q]
            pa, pb = ca.points[ka].copy(), cb.points[kb].copy()
            gap = float(np.linalg.norm(pa - pb))
            if gap < GAP_SNAP * min(length(ca), length(cb)) and ca is not cb:
                held_a = network.degree(refs.get(p, [])) >= 2  # already connected to other lines: stays put
                held_b = network.degree(refs.get(q, [])) >= 2
                target = pa if held_a else pb if held_b else (pa + pb) / 2
                for c, k, here, held in ((ca, ka, pa, held_a), (cb, kb, pb, held_b)):
                    if not held and np.linalg.norm(target - here) > 0:
                        group = components.ComponentGroup(sc, [(c, k)], [], weld=True, mode="soft")
                        group.translate(target - here)
                network.join(sc, ca, ka, cb, kb)
                snapped += 1
            else:
                pts = np.array([pa, pb])
                # gap between two line ends: their width there, in between proportionally
                line = Curve(pts, None, components.widths_between(pts, [ca], [cb], float(self._widths_for(pts)[0])))
                owner.curves.append(line)
                network.sync(sc)
                network.join(sc, ca, ka, line, 0)
                network.join(sc, cb, kb, line, 1)
                out.append(line)
                bridges += 1
            refs = network.refs_by_pid(sc)
        return out, bridges, snapped

    def _gap_report(self, bridges, snapped):
        parts = ([tr("{0} Lücke(n) mit einer Linie überbrückt").format(bridges)] if bridges else []) + \
                ([tr("{0} Punkt(e) verbunden").format(snapped)] if snapped else [])
        return (" – " + ", ".join(parts)) if parts else ""

    def _do_fill(self, bridged=None):
        """Fill surface from the selected lines. Gaps in the outline are closed by the gap rule (_close_outline) with
        real lines or shared points; then every closed outline becomes a fill surface. bridged: the
        gaps are already closed (do_surface) – (bridges, connected points) for the message."""
        curves = [c for c in (edges.ref_to_curve(self.scene, r) for r in self.edge_selection) if c is not None and not c.paint]
        if not curves:
            self.status_changed.emit(tr("Zeichenfläche: erst Linien auswählen (Edit/Tab, Klick / Doppelklick für Loop)."))
            return
        self.scene.checkpoint()
        if bridged is None:
            curves, bridges, snapped = self._close_outline(curves)
        else:
            bridges, snapped = bridged
        graph, _ = self._outline_graph(curves)
        by_id = {c.id: c for c in curves}
        groups = fill.cycles_in([(eid, a, b) for eid, (a, b) in graph.items()])
        loops = [fill.order_cycle(g) for g in groups]

        def fail(text):
            self.scene.undo()  # bridges and connections back again
            self.scene._redo.clear()
            self.status_changed.emit(text)

        if not loops or any(loop is None for loop in loops):
            return fail(tr("Füllen nicht möglich: die gewählten Linien verzweigen sich (mehr als zwei Linien an einem Punkt)."))

        def loop_points(loop):
            parts = [by_id[eid].points[::-1] if flipped else by_id[eid].points for eid, flipped in loop]
            return np.vstack(parts)

        loops = [loop for loop in loops if not fill.is_degenerate(loop_points(loop))]
        if not loops:
            return fail(tr("Füllen nicht möglich: die gewählten Linien schließen keine Fläche ein."))
        created = [self.scene.create_fill([(by_id[eid], flipped) for eid, flipped in loop]) for loop in loops]
        self._fill_last = (None, 0)
        self.edge_selection = []
        self.selection = [s.id for s in created]
        self._changed()
        why = tr(" (Umriss nach innen geknickt)") if bridged is not None else ""
        self.notify(tr("{0}: Füllfläche{1}{2}").format(', '.join(s.name for s in created), why, self._gap_report(bridges, snapped)))

    def clear_all_curves(self):
        self.scene.checkpoint()
        self.scene.clear_curves()
        self._changed()

    def new_stroke(self):
        """Space+B (right-click "New Stroke"): new, empty stroke object – from now on drawing goes into it. Lies in
        the group currently being worked in: in the selected null, otherwise in the null of the active stroke object."""
        sc = self.scene
        sc.checkpoint()
        nulls = [o for o in self.selected_objects if isinstance(o, NullObject)]
        stroke = sc.new_stroke(parent=nulls[0].id) if len(nulls) == 1 else sc.new_stroke()
        self.selection = [stroke.id]
        where = sc.object_by_id(stroke.parent).name if stroke.parent is not None else None
        self.notify(tr("Neues Stroke-Objekt „{0}“{1} – es wird jetzt hinein gezeichnet").format(
            stroke.name, tr(" in „{0}“").format(where) if where else ""))
        self._changed()

    def import_mesh(self, path):
        """Load geometry, select it and bring it into view. Errors are raised as an exception."""
        self.scene.checkpoint()
        try:
            mesh = self.scene.import_mesh(path)
        except Exception:
            self.scene.undo()  # roll back the checkpoint, nothing has changed
            self.scene._redo.clear()
            raise
        self.selection = [mesh.id]
        self._changed()
        self.focus(mesh)
        return mesh

    def delete_selection(self, hard=False, remove=False):
        """Del: delete selection. Point mode: dissolve points (the shape stays, two line ends become one line);
        Shift+Del: remove points (the line is re-run through); Ctrl+Del: delete points (adjacent pieces gone,
        surface gets a new outline)."""
        if self.select_mode and self.ws.component == "point" and self.point_selection:
            return self.delete_points() if hard else self.remove_points() if remove else self.dissolve_points()
        self._materialize_segments()
        return self._delete_selection()

    def _selected_pids(self):
        network.sync(self.scene)
        pids = list(dict.fromkeys(network.pid_of(c, r[3]) for r, c in self._valid_points()))
        return [p for p in pids if p is not None]

    def delete_points(self):
        """Ctrl+Del in point mode: delete selected points (netedit.py) – the pieces up to the neighboring points
        disappear; a curve surface takes the outline that is still closed via its lines, otherwise a
        straight line connects the neighboring points."""
        from .. import netedit
        pids = self._selected_pids()
        if not pids:
            return
        self.scene.checkpoint()
        done, bridges, rebuilt = netedit.delete_points(self.scene, pids)
        self.point_selection = []
        self.ws.target_cache = (None, None)
        parts = [tr("{0} Punkt(e) gelöscht").format(done)]
        if bridges:
            parts.append(tr("{0} Brücke(n)").format(bridges))
        if rebuilt:
            parts.append(tr("{0} Fläche(n) mit neuem Umriss").format(rebuilt))
        self.notify(", ".join(parts))
        self._changed()

    def remove_points(self):
        """Shift+Del in point mode: remove selected points – the line is re-run between the neighboring points
        (smooth); two line ends become one line in the process. Crossings and T-junctions stay (hint)."""
        from .. import netedit
        pids = self._selected_pids()
        if not pids:
            return
        self.scene.checkpoint()
        done, refused = netedit.remove_points(self.scene, pids)
        if not done:
            self.scene.undo()  # nothing changed: roll back the checkpoint
            self.scene._redo.clear()
            self.notify(tr("Kreuzungen, T-Stöße und offene Enden lassen sich nicht entfernen – Strg+Entf löscht den Punkt"))
            return
        self.point_selection = []
        self.ws.target_cache = (None, None)
        text = tr("{0} Punkt(e) entfernt – die Linie läuft neu durch").format(done)
        if refused:
            text += tr("; {0} nicht (Kreuzung/T-Stoß/offenes Ende)").format(refused)
        self.notify(text)
        self._changed()

    def dissolve_points(self):
        """Del in point mode: dissolve selected points – the lines keep their shape; if two line ends meet,
        they become one line. Crossings and T-junctions stay (hint)."""
        from .. import netedit
        pids = self._selected_pids()
        if not pids:
            return
        self.scene.checkpoint()
        done, merged, refused = netedit.dissolve_points(self.scene, pids)
        if not done:
            self.scene.undo()  # nothing changed: roll back the checkpoint
            self.scene._redo.clear()
            self.notify(tr("Kreuzungen, T-Stöße und offene Enden lassen sich nicht auflösen – Strg+Entf löscht den Punkt"))
            return
        self.point_selection = []
        self.ws.target_cache = (None, None)
        text = tr("{0} Punkt(e) aufgelöst – die Linien bleiben, wie sie sind").format(done)
        if merged:
            text += tr(", {0}× zwei Linien zu einer verbunden").format(merged)
        if refused:
            text += tr("; {0} nicht (Kreuzung/T-Stoß/offenes Ende)").format(refused)
        self.notify(text)
        self._changed()

    def _delete_selection(self):
        """Del: delete selected objects (drawing surfaces: lines on them stay) and selected lines."""
        objs = [o for o in self.selected_objects if not (isinstance(o, Plane) and o.preset)]
        if not self.select_mode:  # draw mode does not show selected surfaces/planes: then do not delete them either
            objs = [o for o in objs if not isinstance(o, (Plane, SurfaceObject, MeshObject, NullObject))]
        lines = {(r[1], r[2]) for r in self.edge_selection if r[0] == "line"}
        if not objs and not lines:
            return
        self.scene.checkpoint()
        self.scene.delete(objs)
        for stroke in self.scene.strokes:
            stroke.curves = [c for c in stroke.curves if (stroke.id, c.id) not in lines]
        self.selection = []
        self.edge_selection = []
        self._changed()

    def delete_objects(self, ids):
        """Outliner: delete objects (a null with everything below it, stroke objects with their lines)."""
        objs = [o for o in (self.scene.object_by_id(i) for i in ids) if o is not None
                and not (isinstance(o, Plane) and o.preset) and not self._flip_locked(o)]
        if not objs:
            return
        self.scene.checkpoint()
        self.scene.delete(objs)
        self.selection, self.edge_selection, self.point_selection, self.seg_selection = [], [], [], []
        self._changed()

    def rename(self, obj_id, name):
        obj = self.scene.object_by_id(obj_id)
        if obj is None or not name.strip() or obj.name == name or (isinstance(obj, Plane) and obj.preset)                 or self._flip_locked(obj):
            return
        self.scene.checkpoint()
        obj.name = name.strip()
        self._changed()

    def activate_preset(self, index):
        if self.multi_mode:  # multi mode: 1/2/3 add the ground plane to or remove it from the set
            return self._toggle_multi(self.scene.presets[index])
        if self.scene.active_surface is not self.scene.presets[index]:
            self.scene.checkpoint()
        self.scene.activate(self.scene.presets[index])
        self.selection = []
        self._changed()

    def activate_last_custom(self):
        plane = self.scene.last_custom
        if plane is None:
            self.status_changed.emit(tr("Noch keine eigene Ebene – mit Space + N oder Rechtsklick → New Plane anlegen."))
            return
        if self.scene.active_surface is not plane:
            self.scene.checkpoint()
        self.scene.activate(plane)
        self.selection = [plane.id]
        self._changed()
