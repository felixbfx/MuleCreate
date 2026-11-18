# Mule Create
# Copyright (C) 2026 Felix Barthel
# SPDX-License-Identifier: GPL-3.0-only
#
# This file is part of Mule Create.
#
# Mule Create is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, version 3 of the License.
#
# Mule Create is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Mule Create. If not, see <https://www.gnu.org/licenses/>.

"""Metal adapter constants, uniform packing and vertex layouts matching moderngl.
"""

import itertools
import re
import struct
import numpy as np

NOTHING, BLEND, DEPTH_TEST, CULL_FACE, RASTERIZER_DISCARD, PROGRAM_POINT_SIZE = (
    0,
    1,
    2,
    4,
    8,
    16,
)
POINTS, LINES, LINE_LOOP, LINE_STRIP, TRIANGLES, TRIANGLE_STRIP, TRIANGLE_FAN = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
)
ZERO, ONE = 0, 1
SRC_COLOR, ONE_MINUS_SRC_COLOR, SRC_ALPHA, ONE_MINUS_SRC_ALPHA = (
    0x0300,
    0x0301,
    0x0302,
    0x0303,
)
DST_ALPHA, ONE_MINUS_DST_ALPHA, DST_COLOR, ONE_MINUS_DST_COLOR = (
    0x0304,
    0x0305,
    0x0306,
    0x0307,
)
NEAREST, LINEAR = 0x2600, 0x2601
(
    NEAREST_MIPMAP_NEAREST,
    LINEAR_MIPMAP_NEAREST,
    NEAREST_MIPMAP_LINEAR,
    LINEAR_MIPMAP_LINEAR,
) = 0x2700, 0x2701, 0x2702, 0x2703
GL_DEPTH_BUFFER_BIT, GL_COLOR_BUFFER_BIT = 0x100, 0x4000

_GLSL_TYPES = {
    "float": ("f", 1, 1),
    "int": ("i", 1, 1),
    "uint": ("I", 1, 1),
    "bool": ("i", 1, 1),
    "vec2": ("f", 2, 1),
    "vec3": ("f", 3, 1),
    "vec4": ("f", 4, 1),
    "ivec2": ("i", 2, 1),
    "ivec3": ("i", 3, 1),
    "ivec4": ("i", 4, 1),
    "uvec2": ("I", 2, 1),
    "uvec3": ("I", 3, 1),
    "uvec4": ("I", 4, 1),
    "mat2": ("f", 2, 2),
    "mat3": ("f", 3, 3),
    "mat4": ("f", 4, 4),
}
_INPUT_COMPONENTS = {
    "float": 1,
    "vec2": 2,
    "vec3": 3,
    "vec4": 4,
    "int": 1,
    "ivec2": 2,
    "ivec3": 3,
    "ivec4": 4,
}
_IDS = itertools.count(1)


class Error(RuntimeError):
    pass


def _as_bytes(data):
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    if isinstance(data, np.ndarray):
        return np.ascontiguousarray(data).tobytes()
    return bytes(memoryview(data))


class Uniform:
    __slots__ = (
        "prog",
        "name",
        "offset",
        "base",
        "comps",
        "cols",
        "count",
        "stride",
        "mstride",
        "sampler",
    )

    def __init__(self, prog, name, member=None, sampler=False):
        self.prog, self.name, self.sampler = prog, name, sampler
        if sampler:
            return
        base, comps, cols = _GLSL_TYPES[member["type"]]
        self.base, self.comps, self.cols = base, comps, cols
        self.offset = int(member["offset"])
        dims = member.get("arrayDims") or []
        self.count = int(dims[0]) if dims else None
        self.stride = int(member["size"]) // self.count if self.count else 0
        self.mstride = int(member.get("matrixStride", 16))

    @property
    def value(self):
        if self.sampler:
            return self.prog._units.get(self.name, 0)
        n = self.comps * self.cols
        vals = struct.unpack_from(
            "<%d%s" % (n, self.base), self.prog._block, self.offset
        )
        return vals[0] if n == 1 else vals

    @value.setter
    def value(self, v):
        if self.sampler:
            self.prog._units[self.name] = int(v)
            return
        if self.count is None and self.cols == 1:
            if self.comps == 1:
                struct.pack_into("<" + self.base, self.prog._block, self.offset, v)
            else:
                struct.pack_into(
                    "<%d%s" % (self.comps, self.base), self.prog._block, self.offset, *v
                )
            return
        dtype = "<f4" if self.base == "f" else "<i4"
        self._put(np.asarray(v, dtype).reshape(-1))

    def write(self, data):
        if self.sampler:
            raise Error("write() on a sampler")
        dtype = "<f4" if self.base == "f" else "<i4"
        self._put(np.frombuffer(_as_bytes(data), dtype))

    def _put(self, flat):
        """Tightly packed values -> std140 layout (array elements and matrix columns on their strides)."""
        block = self.prog._view
        per = self.comps * self.cols
        if self.count is None:  # matrix
            for c in range(self.cols):
                part = flat[c * self.comps : (c + 1) * self.comps]
                block[
                    self.offset + c * self.mstride : self.offset
                    + c * self.mstride
                    + 4 * len(part)
                ] = part.astype(flat.dtype).view(np.uint8)
            return
        n = min(self.count, len(flat) // per)
        elems = flat[: n * per].reshape(n, self.cols, self.comps)
        stride_col = self.mstride if self.cols > 1 else 0
        for c in range(self.cols):
            cols = (
                np.ascontiguousarray(elems[:, c, :])
                .view(np.uint8)
                .reshape(n, 4 * self.comps)
            )
            start = self.offset + c * stride_col
            target = block[start : start + self.stride * (n - 1) + 4 * self.comps]
            np.lib.stride_tricks.as_strided(
                target, shape=(n, 4 * self.comps), strides=(self.stride, 1)
            )[:] = cols


_FMT = re.compile(r"^(\d*)([fiux])(\d?)$")


class VertexArray:
    def __init__(self, ctx, program, content, index_buffer=None, index_element_size=4):
        self.ctx, self.program = ctx, program
        self.index_buffer = index_buffer
        if index_buffer is not None and index_element_size != 4:
            raise Error("only 32-bit indices are supported")
        self.bindings = []  # (buffer, stride, per_instance, [(name, kind, comps, offset)])
        for entry in content:
            buf, fmt, names = entry[0], entry[1], list(entry[2:])
            tokens = fmt.split()
            per_instance = False
            if tokens and tokens[-1] in ("/v", "/i", "/r"):
                per_instance = tokens.pop() == "/i"
            attrs, offset = [], 0
            for tok in tokens:
                m = _FMT.match(tok)
                if not m:
                    raise Error("vertex format %r not supported" % tok)
                count, kind, size = (
                    int(m.group(1) or 1),
                    m.group(2),
                    int(m.group(3) or 4),
                )
                if kind == "x":
                    offset += count
                    continue
                if size != 4:
                    raise Error("only 32-bit attributes are supported (%r)" % tok)
                name = names.pop(0)
                attrs.append((name, kind, count, offset))
                offset += 4 * count
            self.bindings.append((buf, offset, per_instance, attrs))
        self.layout_key = tuple(
            (
                stride,
                inst,
                tuple(
                    (program._inputs[n][1], k, c, o)
                    for n, k, c, o in attrs
                    if n in program._inputs
                ),
            )
            for _b, stride, inst, attrs in self.bindings
        )

    def render(self, mode=TRIANGLES, vertices=-1, first=0, instances=-1):
        self.ctx._draw(self, mode, int(vertices), int(first), int(instances))

    def release(self):
        pass


# --- GL functions moderngl does not cover (the renderer gets them from Qt's QOpenGLFunctions) ---
class _GLFunctions:
    def __init__(self, ctx):
        self._ctx = ctx

    def glDepthMask(self, flag):
        self._ctx._depth_mask = bool(flag)

    def glEnable(self, cap):
        pass  # GL_MULTISAMPLE, GL_POINT_SPRITE: always on with Metal

    def glDisable(self, cap):
        pass

    def glClear(self, mask):
        fbo = self._ctx._fbo
        color = (
            (0.0, 0.0, 0.0, 0.0)
            if mask & GL_COLOR_BUFFER_BIT and fbo.color_attachments
            else None
        )
        depth = (
            1.0
            if mask & GL_DEPTH_BUFFER_BIT and fbo.depth_attachment is not None
            else None
        )
        if color is not None or depth is not None:
            self._ctx._clear(fbo, color, depth)

    def glBindVertexArray(self, vao):
        pass
