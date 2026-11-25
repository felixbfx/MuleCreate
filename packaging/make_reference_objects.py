"""Builds the reference objects (sketch_machine/assets/reference/*.usda): rough shapes in real size, only for
comparing sizes – a person 1.80 m, a car, a two-storey house. Y up, meters, standing on the ground at the origin.

    .venv/Scripts/python.exe packaging/make_reference_objects.py
"""
import math
import os

from pxr import Gf, Usd, UsdGeom, Vt

OUT = os.path.join(os.path.dirname(__file__), "..", "sketch_machine", "assets", "reference")


class Parts:
    """Collects simple solids into one mesh (one object in Mule)."""

    def __init__(self):
        self.points, self.counts, self.indices = [], [], []

    def _face(self, ids):
        self.counts.append(len(ids))
        self.indices += ids

    def box(self, lo, hi):
        (x0, y0, z0), (x1, y1, z1) = lo, hi
        b = len(self.points)
        self.points += [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)):
            self._face([b + i for i in f])

    def cylinder(self, center, radius, length, axis, n=16):
        """Cylinder around a main axis (0 = X, 2 = Z), center in the middle."""
        b = len(self.points)
        for side in (-0.5, 0.5):
            for k in range(n):
                a = 2 * math.pi * k / n
                p = [0.0, 0.0, 0.0]
                p[axis] = side * length
                u, v = [i for i in range(3) if i != axis]
                p[u], p[v] = radius * math.cos(a), radius * math.sin(a)
                self.points.append(tuple(c + o for c, o in zip(center, p)))
        self._face([b + k for k in range(n)][::-1])
        self._face([b + n + k for k in range(n)])
        for k in range(n):
            j = (k + 1) % n
            self._face([b + k, b + j, b + n + j, b + n + k])

    def sphere(self, center, radius, rings=8, segs=12):
        b = len(self.points)
        self.points.append((center[0], center[1] - radius, center[2]))
        for r in range(1, rings):
            phi = math.pi * r / rings - math.pi / 2
            for s in range(segs):
                t = 2 * math.pi * s / segs
                self.points.append((center[0] + radius * math.cos(phi) * math.cos(t), center[1] + radius * math.sin(phi),
                                    center[2] + radius * math.cos(phi) * math.sin(t)))
        top = len(self.points)
        self.points.append((center[0], center[1] + radius, center[2]))
        ring = lambda r, s: b + 1 + (r - 1) * segs + s % segs
        for s in range(segs):
            self._face([b, ring(1, s), ring(1, s + 1)])
            self._face([top, ring(rings - 1, s + 1), ring(rings - 1, s)])
        for r in range(1, rings - 1):
            for s in range(segs):
                self._face([ring(r, s), ring(r + 1, s), ring(r + 1, s + 1), ring(r, s + 1)])

    def prism(self, profile, z0, z1, along_x=False):
        """Extruded polygon (profile in XY, counterclockwise) from z0 to z1; along_x: profile in ZY, along X."""
        b, n = len(self.points), len(profile)
        pts = [(x, y, z0) for x, y in profile] + [(x, y, z1) for x, y in profile]
        self.points += [(z, y, x) for x, y, z in pts] if along_x else pts
        self._face([b + k for k in range(n)][::-1])
        self._face([b + n + k for k in range(n)])
        for k in range(n):
            j = (k + 1) % n
            self._face([b + k, b + j, b + n + j, b + n + k])


def person():
    """1.80 m, stick figure of boxes, head a ball (top exactly at 1.80)."""
    p = Parts()
    p.box((-0.17, 0.0, -0.08), (-0.03, 0.88, 0.08))  # legs
    p.box((0.03, 0.0, -0.08), (0.17, 0.88, 0.08))
    p.box((-0.21, 0.88, -0.11), (0.21, 1.48, 0.11))  # torso
    p.box((-0.29, 0.82, -0.05), (-0.21, 1.46, 0.05))  # arms
    p.box((0.21, 0.82, -0.05), (0.29, 1.46, 0.05))
    p.box((-0.05, 1.48, -0.05), (0.05, 1.56, 0.05))  # neck
    p.sphere((0.0, 1.68, 0.0), 0.12)
    return p


def car():
    """Mid-size car 4.5 × 1.8 × 1.45 m, front towards +X."""
    p = Parts()
    p.box((-2.25, 0.3, -0.9), (2.25, 0.95, 0.9))  # body
    p.prism([(-1.45, 0.95), (1.0, 0.95), (0.35, 1.45), (-1.15, 1.45)], -0.8, 0.8)  # cabin
    for x in (-1.4, 1.35):
        for z in (-0.8, 0.8):
            p.cylinder((x, 0.32, z), 0.32, 0.22, axis=2)
    return p


def house():
    """Two-storey detached house: 10 × 8 m, eaves 6 m, gable roof ridge 9.5 m, door 1 × 2.1 m."""
    p = Parts()
    p.box((-5.0, 0.0, -4.0), (5.0, 6.0, 4.0))
    p.prism([(-4.5, 5.8), (4.5, 5.8), (0.0, 9.5)], -5.4, 5.4, along_x=True)  # ridge along the long side, overhang
    p.box((-0.5, 0.0, 4.0), (0.5, 2.1, 4.08))  # door
    for x in (-3.2, 2.2):  # windows ground floor and upper floor (1 × 1.3 m)
        for y in (0.9, 3.9):
            p.box((x, y, 4.0), (x + 1.0, y + 1.3, 4.06))
    p.box((-0.5, 3.9, 4.0), (0.5, 5.2, 4.06))
    return p


OBJECTS = (("person_1m80", "Person_1m80", person), ("car", "Car_4m50", car), ("house", "House_2_storeys", house))


def write(name, prim_name, parts):
    path = os.path.join(OUT, name + ".usda")
    stage = Usd.Stage.CreateNew(path) if not os.path.exists(path) else Usd.Stage.Open(path)
    stage.GetRootLayer().Clear()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Reference")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, f"/Reference/{prim_name}")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*pt) for pt in parts.points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(parts.counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(parts.indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, prim_name, build in OBJECTS:
        print(write(name, prim_name, build()))
