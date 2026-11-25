"""Copy & paste: one clipboard, two ways to paste.

Whole objects (planes, stroke objects, surfaces, meshes) and single lines or
edges are copied. On pasting, the location decides:
  "objects"  (outliner)  everything becomes new objects, loose lines a new stroke object
  "elements" (viewport)  lines land in the active stroke object (duplicate: in their own; duplicated together with
                         surfaces: in a new object per source, next to it in the same group),
                         also the lines of copied stroke objects; surfaces, planes and meshes
                         become new objects, because they only exist as objects
Pasting always happens at the same place.

Relationships come along: everything pasted gets new ids, and the references of the pasted surfaces (outline,
loft cross-sections and side lines, border curves, painted lines) point to the pasted lines – the copy
behaves like the original. If an outline line is missing in the copy, the surface is cleanly decoupled (it then follows
no line, not secretly the original either).
"""
import copy
import json
import os
import uuid

from . import edges, network
from .scene import LoftSurface, MeshObject, NullObject, PatchSurface, Plane, StrokeObject, SurfaceObject, _new_id

# Between two running Mule windows: the copy also goes to the system clipboard as a small scene file
# (Scene.to_dict); the other window reads it back on pasting. Own copies are pasted from memory (INSTANCE).
MIME = "application/x-mule-objects"
INSTANCE = uuid.uuid4().hex
_LOOSE = "__mule_loose__"  # name prefix: stroke object that only carries copied single lines
_BASE = os.path.abspath(os.sep)  # mesh paths relative to the drive root – the same on both sides


class Clipboard:
    def __init__(self):
        self.objects = []  # copies of whole objects
        self.curves = []  # [(Curve copy with the id of the original, id of the stroke object it came from, or None)]
        self.tree_ids = set()  # ids of the objects that came along as part of a copied null (stay objects)

    @property
    def empty(self):
        return not self.objects and not self.curves

    def copy(self, scene, objects, edge_refs=()):
        # a null comes with everything below it (as in Cinema 4D) – as a subtree that stays whole on pasting
        objects, self.tree_ids = list(objects), set()
        for o in list(objects):
            if isinstance(o, NullObject):
                for d in scene.descendants(o):
                    self.tree_ids.add(d.id)
                    if d not in objects:
                        objects.append(d)
        # a line object comes with its surfaces (they belong to it, as when deleting) – otherwise a group copied
        # into another scene arrived as bare lines
        stroke_ids = {o.id for o in objects if isinstance(o, StrokeObject) and not isinstance(o, SurfaceObject)}
        objects += [f for f in scene.surfaces if getattr(f, "owner", None) in stroke_ids and f not in objects]
        self.objects = [copy.deepcopy(o) for o in objects]  # mesh geometry is shared in the process
        self.curves = []
        for ref in edge_refs:
            curve = edges.ref_to_curve(scene, ref)
            if curve is None:
                continue
            source = ref[1] if ref[0] == "line" else None
            self.curves.append((copy.deepcopy(curve), source))  # keeps the old id: for the mapping on pasting
        # Copy as a subgraph: a surface takes its net along (border, cross-sections, side lines, lines and painting
        # on it) – the copy is again a complete surface, internally connected like the original
        have = {c.id for c, _ in self.curves} | {c.id for o in objects if isinstance(o, StrokeObject)
                                                and not isinstance(o, SurfaceObject) for c in o.curves}
        surfaces = [o for o in objects if isinstance(o, SurfaceObject)]
        for s in surfaces:
            for c in scene.surface_lines(s):
                if c.id not in have:
                    have.add(c.id)
                    owner = scene.owner_of(c)
                    self.curves.append((copy.deepcopy(c), owner.id if owner is not None else None))


    def to_bytes(self, scene):
        """The clipboard as a small scene (json): objects as they are, single lines per source in a carrier object."""
        from .scene import Scene
        mini = Scene()
        mini.presets = scene.presets  # lines drawn on the fixed planes keep them
        mini.active_plane = mini.presets[0]
        lists = ((Plane, mini.custom), (SurfaceObject, mini.surfaces), (StrokeObject, mini.strokes),
                 (MeshObject, mini.meshes), (NullObject, mini.nulls))
        for o in self.objects:
            next(target for cls, target in lists if isinstance(o, cls)).append(o)
        by_source = {}
        for curve, source in self.curves:
            by_source.setdefault(source, []).append(curve)
        for k, (source, curves) in enumerate(by_source.items()):
            mini.strokes.append(StrokeObject(f"{_LOOSE}{k}", list(curves)))
        data = mini.to_dict(_BASE)
        data.pop("assets", None)
        return json.dumps({"instance": INSTANCE, "scene": data}).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw):
        """Clipboard of another Mule window (to_bytes); None if it is our own or unreadable."""
        from .scene import Scene
        try:
            box = json.loads(bytes(raw).decode("utf-8"))
            if box.get("instance") == INSTANCE:
                return None
            mini = Scene.from_dict(box["scene"], _BASE)
        except Exception:  # noqa: BLE001 – foreign or broken data: paste as if nothing came
            return None
        mini.assign_owners()
        clip = cls()
        nulls = [n for n in mini.nulls]
        clip.tree_ids = {d.id for n in nulls for d in mini.descendants(n)}
        for st in mini.strokes:
            if st.name.startswith(_LOOSE):
                clip.curves += [(c, None) for c in st.curves]
        clip.objects = [o for o in mini.custom + mini.nulls + mini.meshes + mini.strokes + mini.surfaces
                        if not (isinstance(o, StrokeObject) and o.name.startswith(_LOOSE))]
        return clip


def _copy_name(name, existing):
    names = {o.name for o in existing}
    if name not in names:  # e.g. pasted into another scene: keeps its name
        return name
    candidate, n = f"{name}_copy", 2
    while candidate in names:
        candidate, n = f"{name}_copy{n}", n + 1
    return candidate


def _fresh_curve(curve, id_map):
    """Copy with new id; id_map remembers old -> new (a line that is in the clipboard twice, only once)."""
    c = copy.deepcopy(curve)
    c.id = _new_id()
    id_map.setdefault(curve.id, c.id)
    return c


def _fresh_object(scene, obj, id_map, surface_map):
    """Hang a new copy with new ids (also for all lines) into the matching list of the scene."""
    o = copy.deepcopy(obj)
    o.id = _new_id()
    if isinstance(o, Plane):
        o.preset = False
        o.name = _copy_name(o.name, scene.custom)
        scene.custom.append(o)
        surface_map[obj.id] = o.id
    elif isinstance(o, SurfaceObject):
        o.curves = [_fresh_curve(c, {}) for c in o.curves]  # own shape copies, no lines of the scene
        o.name = _copy_name(o.name, scene.surfaces)
        scene.surfaces.append(o)
        surface_map[obj.id] = o.id
    elif isinstance(o, StrokeObject):
        o.curves = [_fresh_curve(c, id_map) for c in o.curves]
        o.name = _copy_name(o.name, scene.strokes)
        scene.strokes.append(o)
        surface_map[("stroke", obj.id)] = o.id  # for the owner of its surfaces
    elif isinstance(o, MeshObject):
        o.name = _copy_name(o.name, scene.meshes)
        scene.meshes.append(o)
        surface_map[obj.id] = o.id
    elif isinstance(o, NullObject):
        o.name = _copy_name(o.name, scene.nulls)
        o.params = copy.deepcopy(obj.params)
        scene.nulls.append(o)
        surface_map[obj.id] = o.id
    return o


def _remap_ids(ids, id_map):
    """Rewrite a list of line ids to the copy; None if one of them was not copied along."""
    out = [id_map.get(i) for i in ids]
    return None if any(i is None for i in out) else out


def _relink_surface(s, id_map, surface_map):
    """Rewrite the references of a pasted surface to the pasted lines or surfaces."""
    s.attached = {id_map[cid]: a for cid, a in s.attached.items() if cid in id_map}
    for a in s.attached.values():
        a["last"] = None  # remember anew at the next comparison
    if isinstance(s, LoftSurface):
        sources = []
        for src in s.sources:
            if src is None:
                sources.append(None)
            elif src[0] == "lines":
                ids = _remap_ids(src[1], id_map)
                sources.append(("lines", ids) if ids is not None else None)
            else:  # ("surface", id)
                sid = surface_map.get(src[1])
                sources.append(("surface", sid) if sid is not None else None)
        s.sources = sources
        s.rails = [(id_map[cid], col) for cid, col in s.rails if cid in id_map]
        s._synced = None
        if isinstance(s, PatchSurface):
            lines = _remap_ids(s.patch_lines, id_map)
            if s.net:  # grid of the net lines: rewrite to the copies (if one is missing, it is laid anew)
                net = {key: [[_remap_ids(ids, id_map), at] for ids, at in s.net.get(key, [])] for key in ("u", "v")}
                ok = all(ids is not None for key in ("u", "v") for ids, _ in net[key])
                s.net = dict(s.net, **net) if ok else None
            s._net_geo = None
            if lines is None:  # border curves not copied along: independent copy
                s.patch_lines, s.patch_order, s.patch_corners = [], [], []
                s.sources = [None] * len(s.curves)
                s.net = None
            else:
                s.patch_lines = lines
    elif type(s) is SurfaceObject and s.links:
        links = []
        for cid, flipped in s.links:
            if cid is None:
                links.append((None, flipped))
                continue
            ids = cid if isinstance(cid, (list, tuple)) else [cid]
            new = _remap_ids(ids, id_map)
            if new is None:
                links = None
                break
            links.append((new if isinstance(cid, (list, tuple)) else new[0], flipped))
        if links is None:  # outline not fully copied along: decouple surface, it keeps its shape
            s.links, s.counts, s.rest, s.link_cover, s.attached = [], [], None, 0.0, {}
        else:
            s.links = links
        s._fill_key = None


def paste(scene, clip, mode, duplicate=False):
    """Paste the clipboard contents. Returns (new objects, references of the new lines)."""
    id_map, surface_map = {}, {}  # old line id -> new, old surface/plane/mesh id -> new
    objects, loose = [], list(clip.curves)
    for obj in clip.objects:
        if mode == "elements" and isinstance(obj, StrokeObject) and not isinstance(obj, SurfaceObject) \
                and obj.id not in clip.tree_ids:
            loose += [(c, obj.id) for c in obj.curves]  # in the viewport: only the lines
        else:
            objects.append(obj)
    # first the lines (the surfaces need their new ids), then the objects
    refs, new_curves = [], []
    if loose and mode == "objects":
        stroke = StrokeObject(scene._unique_name("Stroke", scene.strokes), [_fresh_curve(c, id_map) for c, _ in loose])
        scene.strokes.append(stroke)
        new_curves += stroke.curves
        made = [stroke]
    elif loose and duplicate and objects:
        # Lines duplicated together with surfaces (e.g. a whole wing): the lines get their own stroke object per source object
        # next to the original (same group) – otherwise they would sit in the original object, and copy and
        # original could no longer be separated in the outliner
        made, by_source = [], {}
        for curve, source in loose:
            by_source.setdefault(source, []).append(curve)
        for source, curves in by_source.items():
            src = next((s for s in scene.strokes if s.id == source), None)
            name = _copy_name(src.name, scene.strokes) if src is not None else scene._unique_name("Stroke", scene.strokes)
            stroke = StrokeObject(name, [_fresh_curve(c, id_map) for c in curves],
                                  parent=src.parent if src is not None else None)
            scene.strokes.append(stroke)
            new_curves += stroke.curves
            refs += [("line", stroke.id, c.id) for c in stroke.curves]
            made.append(stroke)
    else:
        made = []
        for curve, source in loose:
            target = next((s for s in scene.strokes if s.id == source), None) if duplicate else None
            if target is None:
                if scene.active_stroke is None or scene.active_stroke not in scene.strokes:
                    scene.new_stroke()
                target = scene.active_stroke
            new = _fresh_curve(curve, id_map)
            target.curves.append(new)
            new_curves.append(new)
            refs.append(("line", target.id, new.id))
    created = [_fresh_object(scene, obj, id_map, surface_map) for obj in objects]
    for o in created:
        if isinstance(o, StrokeObject) and not isinstance(o, SurfaceObject):
            new_curves += o.curves
    for o in created:
        if isinstance(o, SurfaceObject):
            _relink_surface(o, id_map, surface_map)
            o.owner = surface_map.get(("stroke", o.owner))  # copied line object; otherwise assign_owners decides
    for c in new_curves:  # drawn on a surface/plane copied along: belongs to the copy
        if c.plane_id in surface_map:
            c.plane_id = surface_map[c.plane_id]
    for o in created:  # hierarchy: under a copied null -> under its copy; lines of the settings
        if getattr(o, "parent", None) in surface_map:
            o.parent = surface_map[o.parent]
        if isinstance(o, NullObject):
            for key in ("path", "up"):
                if o.params.get(key) is not None:
                    o.params[key] = id_map.get(o.params[key], o.params[key])
    scene._clean_parents()
    network.fresh_ids(new_curves)  # curve network: the copy gets its own points, not connected to the original
    return created + made, refs
