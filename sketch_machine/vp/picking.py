"""Projection, rays and picking: what lies under the cursor (lines, surfaces, meshes, planes)."""
import os
from .common import *  # noqa: F401,F403
from ..i18n import tr


class PickingMixin:
    # --- Projection -------------------------------------------------------
    def _tan_half(self):
        """tan of half the (vertical) angle of view; _fov_zoom > 1 widens it for one image only (clean_image with
        a foreign aspect ratio: capture wider, then crop top/bottom)."""
        cam_tan = getattr(self, "_cam_tan", None)  # timing: through a camera (its focal length, image format fitted)
        if cam_tan is None:
            base = math.tan(FOV / 2)
        elif getattr(self, "_cam_exact", False):  # image of the camera: exactly its vertical angle of view
            base = cam_tan
        else:
            base = timing_mod.gate_fit(cam_tan, self.width(), self.height())[0]
        return base * getattr(self, "_fov_zoom", 1.0)

    def _half_height(self):
        """Orthographic: half image height in world units (as large as perspective at target)."""
        return self.cam.dist * self._tan_half()

    def _ortho_depth(self):
        return max(ORTHO_DEPTH_MIN, self.cam.dist * ORTHO_DEPTH_FACTOR)

    def _projection(self):
        aspect = self.width() / max(self.height(), 1)
        if self.cam.ortho:
            return ortho_matrix(self._half_height(), aspect, self._ortho_depth())
        return projection_matrix(2 * math.atan(self._tan_half()), aspect, self._near(), self._far())

    def _focal(self):
        """Pixel scale for the stroke width in the shader (see RIBBON_VS)."""
        return 1.0 / self._half_height() if self.cam.ortho else 1.0 / self._tan_half()

    def project(self, points):
        """(N,3) world points -> (N,2) screen (logical pixels), plus mask for visible points."""
        c = self.cam
        v = points - c.eye
        x, y, z = v @ c.right, v @ c.up, -(v @ c.back)
        w, h = self.width(), self.height()
        aspect = w / max(h, 1)
        if c.ortho:
            hh = self._half_height()
            sx = w / 2 + x / (hh * aspect) * (w / 2)
            sy = h / 2 - y / hh * (h / 2)
            return np.stack([sx, sy], axis=1), np.abs(z) < self._ortho_depth()
        t = self._tan_half()
        near = self._near()
        z_safe = np.where(z > near, z, np.nan)
        sx = w / 2 + (x / (z_safe * t * aspect)) * (w / 2)
        sy = h / 2 - (y / (z_safe * t)) * (h / 2)
        return np.stack([sx, sy], axis=1), z > near

    def ray(self, pos):
        """(origin, direction) of the view ray. Orthographic: parallel, with the origin behind everything visible."""
        c = self.cam
        w, h = self.width(), self.height()
        nx = (pos.x() / w) * 2 - 1
        ny = 1 - (pos.y() / h) * 2
        if c.ortho:
            hh = self._half_height()
            origin = c.eye + nx * hh * (w / max(h, 1)) * c.right + ny * hh * c.up + c.back * self._ortho_depth()
            return origin, -c.back
        t = self._tan_half()
        d = -c.back + nx * t * (w / max(h, 1)) * c.right + ny * t * c.up
        return c.eye, d / np.linalg.norm(d)

    def _near(self):
        """Front clipping plane: grows live with the camera distance (from ~50 m) so that terrain of kilometers also lies cleanly
        in depth; below that fixed as always (the depth offsets of the strokes are tuned to it)."""
        return max(NEAR, self.cam.dist * NEAR_PER_DIST)

    def _far(self):
        return max(1000.0, self.cam.dist * 100)

    def _ray_plane(self, pos, plane):
        """Intersection point and ray parameter with the (infinite) plane."""
        eye, d = self.ray(pos)
        n = plane.normal
        denom = d @ n
        if abs(denom) < 1e-6:
            return None, None
        s = ((plane.origin - eye) @ n) / denom
        return (eye + s * d, s) if s > 0 else (None, None)

    def _visible_planes(self):
        """Displayed planes. Drawing aids off: none at all, not even the base planes – the result raw
        (drawing still continues on the active plane)."""
        if not self.planes_visible:
            return []
        planes = list(self.scene.custom)
        if self.scene.active_plane.preset:
            planes.append(self.scene.active_plane)
        if self.multi_mode:  # base planes in the multi set are visible
            planes += [s for s in self._multi_surfaces() if isinstance(s, Plane) and s.preset and not any(s is q for q in planes)]
        return [p for p in planes if not p.hidden]

    def _visible_meshes(self):
        return [m for m in self.scene.meshes if not m.missing and not self.scene.is_hidden(m)]

    def _plane_under(self, pos):
        """(plane, point, distance) of the nearest visible plane within its limits."""
        best = (None, None, None)
        for plane in self._visible_planes():
            hit, s = self._ray_plane(pos, plane)
            if hit is not None and plane.contains(hit) and (best[2] is None or s < best[2]):
                best = (plane, hit, s)
        return best

    # --- GPU picking on meshes ---
    def _physical(self, pos):
        dpr = self.devicePixelRatioF()
        pw, ph = int(self.width() * dpr), int(self.height() * dpr)
        return int(pos.x() * dpr), ph - 1 - int(pos.y() * dpr), (pw, ph)

    def _mesh_samples(self, samples, meshes=None, exact=True):
        """For each screen point (mesh, point, distance) of the frontmost hit under
        meshes (default: all visible), otherwise (None, None, None). One GPU readback."""
        meshes = self._pick_items(self._pickables() if meshes is None else meshes)
        misses = [(None, None, None)] * len(samples)
        if not meshes or self.renderer is None:
            return misses
        phys = np.array([self._physical(s)[:2] for s in samples])
        size = self._physical(QPointF())[2]
        ctx=self.renderer.ctx
        use_rt=bool(getattr(ctx,"supports_raytracing",False)) and os.environ.get("MULE_METALRT","1")!="0"
        def gpu():
            if use_rt:
                dpr=self.devicePixelRatioF()
                # Match raster pixel centers, including Retina. Exact drawing surfaces are refined below.
                rays=[self.ray(QPointF((int(s.x()*dpr)+.5)/dpr,(int(s.y()*dpr)+.5)/dpr)) for s in samples]
                origins=np.asarray([r[0] for r in rays]);dirs=np.asarray([r[1] for r in rays])
                if self.cam.ortho:
                    near,far=0.,2*self._ortho_depth()
                else:
                    forward=np.maximum(dirs@(-self.cam.back),1e-8)
                    near,far=self._near()/forward,self._far()/forward
                return ctx.ray_pick(meshes,origins,dirs,near,far)
            self.renderer.update_pick(size, self.cam, self._projection(), meshes)
            return self.renderer.read_pick_points(phys)  # only these pixels, not the rectangle around them
        data, prims = self._with_gpu(gpu)
        out, rays, exact_jobs = [], [], {}
        for k, s in enumerate(samples):
            px = data[k]
            index=int(round(float(px[3])))
            mesh = (meshes[index-1] if 0<index<=len(meshes) else None) if use_rt else self.renderer.pick_mesh(index)
            mesh = getattr(mesh, "owner", mesh)  # surface proxy -> surface
            if mesh is None:
                out.append((None, None, None))
                rays.append(None)
                continue
            origin, d = self.ray(s)  # distance as with the planes: along the view ray
            out.append([mesh, px[:3].astype(float), None])
            rays.append((origin, d))
            if exact and isinstance(mesh, SurfaceObject):
                exact_jobs.setdefault(id(mesh), (mesh, []))[1].append(k)
        for mesh, ks in exact_jobs.values():  # exact hits: all rays of a surface at once
            pts = self._exact_hits(mesh, np.array([rays[k][0] for k in ks]), np.array([rays[k][1] for k in ks]),
                                   np.array([out[k][1] for k in ks]), prims[ks])
            for k, pt in zip(ks, pts):
                out[k][1] = pt
        for k, entry in enumerate(out):
            if entry[0] is not None:
                origin, d = rays[k]
                out[k] = (entry[0], entry[1], float((entry[1] - origin) @ d))
        return out

    @staticmethod
    def _exact_hits(surface, origins, dirs, approx, prims=None):
        """Exact hit points of many view rays with a drawing surface (lines projected onto surfaces).
        prims: the triangle the GPU hit for each ray – first intersect only with it (one test per ray);
        what narrowly misses there (triangle edges) searches as before in the triangles of the surroundings."""
        mesh = surface.fill_mesh()
        if mesh is None:
            return approx
        grid = _triangle_grid(mesh)
        if prims is not None and len(prims):
            ti = np.clip(np.asarray(prims, np.int64), 0, len(grid.v0) - 1)
            v0, e1, e2 = grid.v0[ti], grid.e1[ti], grid.e2[ti]
            pv = np.cross(dirs, e2)
            det = np.einsum("ij,ij->i", e1, pv)
            ok = np.abs(det) > 1e-14
            inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
            tv = origins - v0
            u = np.einsum("ij,ij->i", tv, pv) * inv
            qv = np.cross(tv, e1)
            w = np.einsum("ij,ij->i", qv, dirs) * inv
            t = np.einsum("ij,ij->i", e2, qv) * inv
            eps = 1e-6
            hit = ok & (u >= -eps) & (w >= -eps) & (u + w <= 1 + eps) & (t > 0)
            out = np.array(approx, float, copy=True)
            out[hit] = origins[hit] + dirs[hit] * t[hit][:, None]
            rest = np.flatnonzero(~hit)
            if len(rest):
                out[rest] = PickingMixin._exact_hits(surface, origins[rest], dirs[rest], approx[rest])
            return out
        pairs_s, pairs_t = [], []
        # group rays by grid cell: neighboring sample points share their candidate triangles
        cells = np.floor(np.asarray(approx) / grid.cell).astype(np.int64)
        uniq, inverse = np.unique(cells, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        for u, key in enumerate(uniq.tolist()):
            cand = grid.near_cell(tuple(key))
            if cand is None:
                continue
            members = np.flatnonzero(inverse == u)
            pairs_s.append(np.repeat(members, len(cand)))
            pairs_t.append(np.tile(cand, len(members)))
        if not pairs_s:
            return approx
        si, ti = np.concatenate(pairs_s), np.concatenate(pairs_t)
        o, d = origins[si], dirs[si]
        v0, e1, e2 = grid.v0[ti], grid.e1[ti], grid.e2[ti]
        pv = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pv)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0
        u = np.einsum("ij,ij->i", tv, pv) * inv
        qv = np.cross(tv, e1)
        w = np.einsum("ij,ij->i", qv, d) * inv
        t = np.einsum("ij,ij->i", e2, qv) * inv
        eps = 1e-6
        hit = ok & (u >= -eps) & (w >= -eps) & (u + w <= 1 + eps) & (t > 0)
        out = approx.copy()
        if not hit.any():
            return out
        t_approx = np.einsum("ij,ij->i", approx - origins, dirs)
        si, t = si[hit], t[hit]
        err = np.abs(t - t_approx[si])
        order = np.lexsort((err, si))  # per ray the intersection near the GPU depth first
        first = order[np.r_[True, si[order][1:] != si[order][:-1]]]
        out[si[first]] = origins[si[first]] + dirs[si[first]] * t[first][:, None]
        return out

    def _mesh_under(self, pos, meshes=None):
        """(mesh, point, distance) under the cursor, otherwise (None, None, None)."""
        return self._mesh_samples([pos], meshes)[0]

    @staticmethod
    def _line_samples(p0, p1):
        n = max(2, int((p1 - p0).manhattanLength() / SURFACE_SAMPLE_PX) + 1)
        return [p0 + (p1 - p0) * (i / (n - 1)) for i in range(n)]

    def _mesh_line(self, p0, p1, mesh):
        """Project the screen segment p0 -> p1 onto the surface of mesh.
        Returns the points from the start to the first place where the mesh is left – an adjoining
        surface does not continue the line (drawing target is only the active surface)."""
        points = []
        for hit_mesh, point, _ in self._mesh_samples(self._line_samples(p0, p1), [mesh]):
            if hit_mesh is not mesh:
                break
            points.append(point)
        return points

    def _continues(self, prev, point):
        """Does point lie in space directly next to prev (stroke changes seamlessly to an adjoining surface)?"""
        depth = (prev - self.cam.eye) @ -self.cam.back
        return np.linalg.norm(point - prev) < CONTINUE_PX * float(self._world_per_px(depth))

    def _pickables(self):
        """Everything you can click and draw on like on a mesh: meshes and drawing surfaces
        (hidden ones only while being drawn on)."""
        surfaces = self._visible_surfaces()
        active = self.scene.active_mesh
        if isinstance(active, SurfaceObject) and active not in surfaces:
            surfaces.append(active)
        return self._visible_meshes() + surfaces

    @staticmethod
    def _pick_items(objs):
        """Replace surfaces by their picking proxies (meshes stay as they are)."""
        out = []
        for o in objs:
            item = o.pick_proxy() if isinstance(o, SurfaceObject) else o  # (copies already come as proxies)
            if item is not None:
                out.append(item)
        return out

    # --- Multi-plane mode (B) ---
    @staticmethod
    def _is_drawing_surface(o):
        return isinstance(o, (Plane, SurfaceObject)) or isinstance(o, MeshObject) and not o.missing

    def _multi_surfaces(self):
        """The multi set: planes, surfaces and meshes drawn on in multi mode. Independent,
        independent of the selection (is kept when switching to selection mode)."""
        objs = (self.scene.object_by_id(i) for i in self.scene.multi_ids)
        found = [o for o in objs if o is not None and self._is_drawing_surface(o)]
        return found or [self.scene.active_surface]  # empty set: the active surface

    def _toggle_multi(self, obj):
        """Multi mode: clicking a surface adds it to the set or removes it again."""
        self.scene.checkpoint()  # which surfaces are active belongs to the history (Ctrl+Z)
        ids = self.scene.multi_ids
        if not any(self.scene.object_by_id(i) is not None for i in ids):
            # empty set (only stood in for the active surface): the first click starts a new set
            ids.clear()
            ids.append(obj.id)
            self._activate(obj)
            self.notify(tr("Zeichnet auf: {0} – weitere Flächen anklicken: dazu · B: Multi aus").format(obj.name))
            return self._changed()
        if obj.id in ids:
            ids.remove(obj.id)
            rest = [o for o in (self.scene.object_by_id(i) for i in ids) if o is not None]
            if obj is self.scene.active_surface and rest:
                self._activate(rest[-1])  # the most recently added stays active
            self.notify(tr("{0} entfernt ({1} Fläche(n)) – B: Multi aus").format(obj.name, len(rest)))
        else:
            ids.append(obj.id)
            self._activate(obj)  # last clicked = active (mouse wheel, [ ], F: color)
            self.notify(tr("{0} dazu ({1} Fläche(n)) – Klick nochmal: weg · B: Multi aus").format(obj.name, len(ids)))
        self._changed()

    def _multi_samples(self, samples):
        """For each screen point (point, surface) of the first hit under the selected
        surfaces; planes count only within their grid. (None, None) in gaps.
        Drawing happens only on visible things: an unselected mesh in front blocks
        (except in X-ray mode). Unselected planes never block."""
        surfaces = self._multi_surfaces()
        planes = [s for s in surfaces if isinstance(s, Plane)]
        endless = len(surfaces) == 1  # a single plane: infinite as in isolation
        meshes = [s for s in surfaces if isinstance(s, (MeshObject, SurfaceObject))]
        mesh_hits = self._mesh_samples(samples, meshes if self.xray else None)
        out = []
        for pos, (mesh, m_point, m_dist) in zip(samples, mesh_hits):
            if mesh is None:
                best = (None, None, None)
            elif any(mesh is m for m in meshes):
                best = (m_point, mesh, m_dist)
            else:  # unselected mesh in front: only planes in front can still hit
                best = (None, None, m_dist)
            for plane in planes:
                hit, s = self._ray_plane(pos, plane)
                if hit is not None and (endless or plane.contains(hit)) and (best[2] is None or s < best[2]):
                    best = (hit, plane, s)
            out.append(best[:2])
        return out

    @staticmethod
    def _split_runs(hits, kinds=False):
        """Split hits [(point, surface)] into lines: new line at a gap or surface change. kinds: additionally per
        line (start, end) as "hit" / "switch" / "gap" (as in multi mode when drawing freehand)."""
        runs, run, surface, out_kinds = [], [], None, []
        start, prev_miss = None, True
        for n, (point, surf) in enumerate(hits):
            if point is None or surf is not surface:
                if len(run) > 1:
                    runs.append((run, surface))
                    out_kinds.append((start, "gap" if point is None else "switch"))
                run, surface = [], surf
                start = None if point is None else ("hit" if n == 0 else "gap" if prev_miss else "switch")
            if point is not None:
                run.append(point)
            prev_miss = point is None
        if len(run) > 1:
            runs.append((run, surface))
            out_kinds.append((start, "hit"))
        return (runs, out_kinds) if kinds else runs

    # --- What lies under the cursor ---
    def _curve_under(self, pos, tolerance=PICK_PX, exclude=()):
        """(curve, 3D point) of the nearest line within tolerance pixels, otherwise (None, None).
        exclude: ids of lines that are out of the question."""
        if exclude:
            items = [(c, c.points) for c in self._visible_curves() if id(c) not in exclude]
            found = self._near_segments(pos, items, tolerance)
        else:
            found = self._near_segments(pos, None, tolerance, cache="curves")
        if not found:
            return None, None
        found = [(d, curve, self._segment_point(pos, curve.points[i], curve.points[i + 1], t))
                 for d, curve, i, t in found if i + 1 < len(curve.points)]  # (cache outdated)
        if not found:
            return None, None
        # if lines lie on top of each other on screen (e.g. front/back in an axis view),
        # the front one wins
        closest = min(f[0] for f in found)
        origin, ray_d = self.ray(pos)
        _, curve, point = min((f for f in found if f[0] <= closest + TIE_SCREEN_PX), key=lambda f: (f[2] - origin) @ ray_d)
        return curve, point

    def _near_segments(self, pos, items, tolerance, cache=None):
        """[(distance_px, key, segment_index, t)] – per line the nearest segment, only within tolerance.
        items: [(key, points (N,3))]. All lines projected at once (fast even with thousands);
        cache="curves"/"edges": all lines of the scene, cached until the next change."""
        if cache is not None:
            store = self.ws.seg_cache
            if store is None or store[0] is not self.scene:
                store = self.ws.seg_cache = (self.scene, {})
            if cache not in store[1]:
                if cache == "curves":
                    src = [(c, c.points) for c in self._visible_curves()]
                else:
                    src = self._visible_edges()
                store[1][cache] = self._segment_arrays(src)
            arrays = store[1][cache]
        else:
            arrays = self._segment_arrays(items)
        if arrays is None:
            return []
        keys, a, b, owner, first = arrays
        sa, oka = self.project(a)
        sb, okb = self.project(b)
        p = np.array([pos.x(), pos.y()])
        ab = sb - sa
        t = np.clip(np.einsum("ij,ij->i", p - sa, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0, 1)
        d = np.linalg.norm(sa + ab * t[:, None] - p, axis=1)
        d[~(oka & okb) | np.isnan(d)] = np.inf
        near = np.nonzero(d < tolerance)[0]
        best = {}
        for k in near[np.argsort(d[near])]:
            o = int(owner[k])
            if o not in best:
                best[o] = (float(d[k]), keys[o], int(k - first[o]), float(t[k]))
        return list(best.values())

    @staticmethod
    def _segment_arrays(items):
        items = [(k, np.asarray(pts)) for k, pts in items if len(pts) >= 2]
        if not items:
            return None
        counts = np.array([len(pts) - 1 for _, pts in items])
        a = np.vstack([pts[:-1] for _, pts in items])
        b = np.vstack([pts[1:] for _, pts in items])
        owner = np.repeat(np.arange(len(items)), counts)
        first = np.concatenate([[0], np.cumsum(counts)[:-1]])
        return [k for k, _ in items], a, b, owner, first

    def _segment_point(self, pos, a, b, screen_t):
        """3D point on the segment a-b under the cursor: the point nearest to the view ray.
        (Simply transferring the screen parameter to 3D is wrong in perspective –
        for lines running into the depth the point would lie far from the cursor.)"""
        origin, d = self.ray(pos)
        u = b - a
        uu, ud = u @ u, u @ d
        denom = uu - ud * ud  # d has length 1
        if denom < 1e-12 * max(uu, 1e-12):  # segment lies (almost) in the view ray
            return a + u * screen_t
        w0 = a - origin
        s = (ud * (d @ w0) - (u @ w0)) / denom
        return a + u * min(max(s, 0.0), 1.0)

    def _surface_under(self, pos):
        """(object, point) of the nearest plane, mesh or surface under the cursor."""
        plane, p_hit, p_dist = self._plane_under(pos)
        mesh, m_hit, m_dist = self._mesh_under(pos)
        # tie (surface lies exactly on the plane, e.g. a filled rectangle on the ground): the surface
        if mesh is not None and (plane is None or m_dist <= p_dist + 1e-4 * max(p_dist, 1.0)):
            return mesh, m_hit
        return plane, p_hit

    def _pick_point(self, pos):
        """3D point under the cursor: first a line, otherwise surface, plane or mesh."""
        _, point = self._curve_under(pos)
        if point is not None:
            return point
        return self._surface_under(pos)[1]

    def _front_curve(self, pos, tolerance=PICK_PX, exact_over_surface=False):
        """(curve, point, occluded_by) under the cursor: the line only if it does not lie clearly behind the
        visible surface/mesh at this place in space – otherwise (None, None, this surface). Without a line: (None,
        None, None). Lines on the surface (border, painting) stay hittable. exact_over_surface: if a surface lies under the
        cursor, the click goes to it – unless a line lies exactly beneath (LINE_EXACT_PX)."""
        surface, hit, _ = self._mesh_under(pos)  # drawing surfaces and meshes (planes do not occlude)
        if exact_over_surface and hit is not None:
            tolerance = min(tolerance, LINE_EXACT_PX)
        curve, point = self._curve_under(pos, tolerance)
        if curve is not None and surface is not None and hit is not None:
            origin, d = self.ray(pos)
            depth = (hit - self.cam.eye) @ -self.cam.back
            if (point - origin) @ d > (hit - origin) @ d + 2 * PICK_PX * float(self._world_per_px(depth)):
                return None, None, surface
        return curve, point, None

    def _object_under(self, pos):
        """Object under a click. Lines take priority – unless they lie clearly behind
        the hit surface in space (seen through tracing paper). Otherwise in dense
        drawings the surface is hardly hittable anymore, because some line runs through behind almost everywhere."""
        # lines only exactly under the cursor: in dense drawings the normal snap radius almost
        # everywhere holds one, and you would no longer reach the surface beneath
        curve, point, surface = self._front_curve(pos, LINE_EXACT_PX)
        if curve is None and surface is not None:
            return surface
        if curve is not None:
            return self.scene.owner_of(curve)
        return self._surface_under(pos)[0]
