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

"""The renderer's buffer/program/draw API implemented directly with Metal 4.

Qt supplies QImage for overlays. Shaders load directly from our compiled MSL library;
program layouts and resource bindings are defined in metal/programs.json.
"""

import ctypes as C
import json
from types import SimpleNamespace

import numpy as np
from PySide6.QtGui import QImage

from .types import *
from .types import (
    _INPUT_COMPONENTS,
    _as_bytes,
    _GLFunctions,
)
from .api import API, VertexBinding, TextureBinding
from .programs import Program

# Metal's public enum values; these are ABI-stable and shared by all Metal versions.
_FORMATS = {
    (1, "f4"): (55, 4),
    (4, "f4"): (125, 16),
    (4, "f1"): (70, 4),
    (1, "f1"): (10, 1),
    (2, "f1"): (30, 2),
    (4, "f2"): (115, 8),
    (1, "f2"): (25, 2),
    (1, "u4"): (53, 4),
    (1, "i4"): (54, 4),
}
_ATTR = {"f": (28, 29, 30, 31), "i": (32, 33, 34, 35), "u": (36, 37, 38, 39)}
_BLEND = {
    ZERO: 0,
    ONE: 1,
    SRC_COLOR: 2,
    ONE_MINUS_SRC_COLOR: 3,
    SRC_ALPHA: 4,
    ONE_MINUS_SRC_ALPHA: 5,
    DST_COLOR: 6,
    ONE_MINUS_DST_COLOR: 7,
    DST_ALPHA: 8,
    ONE_MINUS_DST_ALPHA: 9,
}
_DEPTH = {"0": 0, "<": 1, "==": 2, "<=": 3, ">": 4, "!=": 5, ">=": 6, "1": 7}
_TOPOLOGY = {POINTS: 0, LINES: 1, LINE_STRIP: 2, TRIANGLES: 3, TRIANGLE_STRIP: 4}


class Resource:
    def __del__(self):
        # Command frames retain every native allocation they reference until GPU completion.
        # Dropping a Python wrapper can therefore release its ownership immediately.
        handle = getattr(self, "handle", None)
        if handle:
            self.ctx.api.release(handle)
            self.handle = None

    def release(self):
        if not self._released:
            self._released = True
            self.ctx.api.release(self.handle)
            self.handle = None


class Buffer(Resource):
    def __init__(self, ctx, data=None, reserve=0, dynamic=False):
        self.ctx = ctx
        self.dynamic = dynamic
        self._released = False
        payload = None if data is None else _as_bytes(data)
        self.size = len(payload) if payload is not None else int(reserve)
        self.handle = (
            ctx.api.call("buffer", ctx.handle, payload, max(16, self.size))
            if payload is None
            else ctx.api.call(
                "buffer",
                ctx.handle,
                payload + b"\0" * max(0, 16 - len(payload)),
                max(16, self.size),
            )
        )

    @property
    def glo(self):
        return self

    def write(self, data, offset=0):
        payload = _as_bytes(data)
        if offset < 0 or offset + len(payload) > self.size:
            raise Error("Buffer write out of bounds")
        if payload:
            self.ctx._writing()
            self.ctx.api.call(
                "write_buffer",
                self.ctx.handle,
                self.handle,
                payload,
                offset,
                len(payload),
            )

    def read(self, size=-1, offset=0):
        size = self.size - offset if size < 0 else size
        if offset < 0 or size < 0 or offset + size > self.size:
            raise Error("Buffer read out of bounds")
        out = C.create_string_buffer(size)
        self.ctx._end_pass()
        self.ctx.api.call(
            "read_buffer", self.ctx.handle, self.handle, out, offset, size
        )
        return out.raw

    def orphan(self, size=-1):
        pass


class Texture(Resource):
    def __init__(
        self, ctx, size, components=4, data=None, samples=0, dtype="f1", depth=False
    ):
        self.ctx = ctx
        self.size = tuple(map(int, size))
        self.width, self.height = self.size
        self.components, self.dtype, self.depth = components, dtype, depth
        self.samples = max(1, int(samples or 0))
        self._released = False
        self.format, self._bpp = (252, 4) if depth else _FORMATS[(components, dtype)]
        self.mipmapped = (
            data is not None and dtype == "f1" and not depth and self.samples == 1
        )
        self.filter = (LINEAR, LINEAR)
        self.repeat_x = self.repeat_y = True
        self.compare_func = ""
        self.anisotropy = 1
        self.handle = ctx.api.call(
            "texture",
            ctx.handle,
            *self.size,
            self.format,
            self.samples,
            int(self.mipmapped),
        )
        if data is not None:
            self.write(data)

    @property
    def glo(self):
        return self

    def sampler_key(self):
        mn, mg = self.filter
        mip = None
        if mn in (
            NEAREST_MIPMAP_NEAREST,
            LINEAR_MIPMAP_NEAREST,
            NEAREST_MIPMAP_LINEAR,
            LINEAR_MIPMAP_LINEAR,
        ):
            mip = (
                NEAREST
                if mn in (NEAREST_MIPMAP_NEAREST, LINEAR_MIPMAP_NEAREST)
                else LINEAR
            )
            mn = (
                NEAREST
                if mn in (NEAREST_MIPMAP_NEAREST, NEAREST_MIPMAP_LINEAR)
                else LINEAR
            )
        if not self.mipmapped:
            mip = None
        if self.samples > 1:
            mn = mg = NEAREST
        return mn, mg, mip, self.repeat_x, self.repeat_y

    def write(self, data, viewport=None, level=0, alignment=1):
        payload = _as_bytes(data)
        x, y, w, h = map(
            int,
            viewport
            or (0, 0, max(1, self.width >> level), max(1, self.height >> level)),
        )
        if len(payload) < w * h * self._bpp:
            raise Error("Texture upload is too short")
        self.ctx._writing()
        self.ctx.api.call(
            "write_texture",
            self.ctx.handle,
            self.handle,
            payload,
            x,
            y,
            w,
            h,
            level,
            self._bpp,
        )

    def build_mipmaps(self, base=0, max_level=1000):
        if self.mipmapped:
            self.ctx._writing()
            self.ctx.api.call("mipmap", self.ctx.handle, self.handle)

    def use(self, location=0):
        self.ctx._units[int(location)] = self

    def read(self, level=0, alignment=1):
        return self.ctx._read_texture(self, None)

    def release(self):
        self.ctx._units = {k: t for k, t in self.ctx._units.items() if t is not self}
        super().release()


class Framebuffer:
    def __init__(self, ctx, color_attachments=(), depth_attachment=None):
        self.ctx = ctx
        self.color_attachments = tuple(color_attachments)
        self.depth_attachment = depth_attachment
        first = (self.color_attachments or (depth_attachment,))[0]
        if first is None:
            raise Error("Framebuffer without attachments")
        self.size = first.size
        self.width, self.height = self.size
        self.samples = first.samples
        self.viewport = (0, 0, *self.size)
        self.color_mask = (True, True, True, True)
        self._released = False
        self.rp_key = (
            tuple(t.format for t in self.color_attachments),
            depth_attachment.format if depth_attachment else 0,
            self.samples,
        )

    @property
    def glo(self):
        return self

    def use(self):
        self.ctx._fbo = self
        self.ctx.viewport = self.viewport

    def clear(
        self,
        red=0.0,
        green=0.0,
        blue=0.0,
        alpha=0.0,
        depth=1.0,
        viewport=None,
        color=None,
    ):
        if color is not None:
            red, green, blue, alpha = (tuple(color) + (0.0,) * 4)[:4]
        self.ctx._clear(
            self,
            tuple(map(float, (red, green, blue, alpha)))
            if self.color_attachments
            else None,
            float(depth) if self.depth_attachment else None,
        )

    def read(
        self,
        viewport=None,
        components=3,
        attachment=0,
        alignment=1,
        dtype="f1",
        clamp=False,
    ):
        tex = self.color_attachments[attachment]
        rect = viewport or (0, 0, *self.size)
        if len(rect) == 2:
            rect = (0, 0, *rect)
        data = self.ctx._read_texture(tex, rect)
        if components != tex.components:
            kind = {"f4": "f4", "f1": "u1", "f2": "f2"}[tex.dtype]
            arr = np.frombuffer(data, kind).reshape(-1, tex.components)
            out = np.zeros((len(arr), components), kind)
            k = min(components, tex.components)
            out[:, :k] = arr[:, :k]
            data = out.tobytes()
        return data

    def release(self):
        self.ctx._clears.pop(self, None)
        if self.ctx._pass_fbo is self:
            self.ctx._end_pass()
        if self.ctx._fbo is self:
            self.ctx._fbo = self.ctx.screen
        self._released = True


class Library(Resource):
    def __init__(self, ctx, path):
        self.ctx = ctx
        self._released = False
        self.path = path
        self.version = 41
        self.binary = path.read_bytes()
        self.handle = ctx.api.call("library", ctx.handle, self.binary, len(self.binary))


class Context:
    backend = "Metal 4"
    version_code = 410
    max_samples = 8

    def __init__(self, parent=0):
        self.api = API()
        self.handle = self.api.call(
            "create", parent, str(self.api.folder / "builtin.metallib").encode()
        )
        self.gl = _GLFunctions(self)
        self.native_metal = True
        info = self.api.info(self.handle)
        self.supports_raytracing = info["metalrt"]
        self.info = {
            "GL_VENDOR": "Apple",
            "GL_RENDERER": info["device"],
            "GL_VERSION": "Native Metal 4 / MSL 4.1 / MetalKit",
            "GL_MAX_SAMPLES": 8,
            "GL_MAX_TEXTURE_SIZE": 16384,
            "METALRT": self.supports_raytracing,
        }
        self._units = {}
        self._pipelines = {}
        self._samplers = {}
        self._internal = {}
        self._shaders = []
        self._library = None
        self._dummies = {}
        self._clears = {}
        self._pass_fbo = None
        self._fbo = None
        self.screen = None
        self._overlay = None
        self._frame = False
        self.frame_count = 0
        self._flags = NOTHING
        self._blend = (SRC_ALPHA, ONE_MINUS_SRC_ALPHA)
        self.depth_func = "<"
        self._depth_mask = True
        self.line_width = 1.0
        self.scissor = None
        self._viewport = (0, 0, 1, 1)
        self._rt_cache = {}
        self._rt_scene_key = None

    def _function(self, spec):
        if self._library is None:
            self._library = Library(self, self.api.folder / "scene.metallib")
            self._shaders.append(self._library)
        return SimpleNamespace(
            entry=spec["entry"], handle=self._library.handle,
            bindings={int(k): v for k, v in spec["bindings"].items()},
        )

    def buffer(self, data=None, reserve=0, dynamic=False):
        return Buffer(self, data, reserve, dynamic)

    def external_buffer(self, glo, size):
        if not isinstance(glo, Buffer):
            raise Error("Buffer belongs to another GPU backend")
        return glo

    def texture(self, size, components, data=None, samples=0, alignment=1, dtype="f1"):
        return Texture(self, size, components, data, samples, dtype)

    def depth_texture(self, size, data=None, samples=0, alignment=4):
        return Texture(self, size, 1, data, samples, "f4", True)

    def renderbuffer(self, size, components=4, samples=0, dtype="f1"):
        return Texture(self, size, components, None, samples, dtype)

    def depth_renderbuffer(self, size, samples=0):
        return Texture(self, size, 1, None, samples, "f4", True)

    def framebuffer(self, color_attachments=(), depth_attachment=None):
        return Framebuffer(
            self,
            (color_attachments,)
            if isinstance(color_attachments, Texture)
            else color_attachments,
            depth_attachment,
        )

    def program(self, name):
        return Program(self, name)

    def vertex_array(
        self, program, content, index_buffer=None, index_element_size=4, **_
    ):
        return VertexArray(self, program, content, index_buffer, index_element_size)

    def simple_vertex_array(
        self, program, buffer, *attributes, index_buffer=None, index_element_size=4
    ):
        fmt = " ".join(
            "%df" % _INPUT_COMPONENTS[program._inputs[a][0]] for a in attributes
        )
        return self.vertex_array(
            program, [(buffer, fmt, *attributes)], index_buffer, index_element_size
        )

    def detect_framebuffer(self, glo=None):
        return self.screen

    @property
    def fbo(self):
        return self.screen

    def enable_only(self, flags):
        self._flags = int(flags)

    def enable(self, flags):
        self._flags |= int(flags)

    def disable(self, flags):
        self._flags &= ~int(flags)

    @property
    def blend_func(self):
        return self._blend

    @blend_func.setter
    def blend_func(self, value):
        self._blend = tuple(value)

    @property
    def viewport(self):
        return self._viewport

    @viewport.setter
    def viewport(self, value):
        self._viewport = (
            tuple(map(int, value)) if len(value) == 4 else (0, 0, *map(int, value))
        )

    def begin_frame(self, size):
        self._frame = True
        self.frame_count += 1
        size = tuple(map(int, size))
        if self.screen is None or self.screen.size != size:
            if self.screen:
                self.screen.release()
                self.screen.color_attachments[0].release()
            self.screen = Framebuffer(self, [Texture(self, size)])
        self._fbo = self.screen
        self.viewport = (0, 0, *size)

    def end_frame(self):
        self._flush_clears()
        self._end_pass()
        self.api.call("submit", self.handle, 0)
        self._frame = False

    def in_frame(self):
        return self._frame

    def _end_pass(self):
        if self._pass_fbo is not None:
            self.api.call("end_pass", self.handle)
            self._pass_fbo = None

    def _writing(self):
        self._end_pass()

    def _clear(self, fbo, color, depth):
        if self._pass_fbo is fbo:
            self._end_pass()
        old = self._clears.get(fbo, (None, None))
        self._clears[fbo] = (
            color if color is not None else old[0],
            depth if depth is not None else old[1],
        )

    def _flush_clears(self, keep=None):
        for fbo in list(self._clears):
            if fbo is not keep:
                self._open_pass(fbo)
                self._end_pass()

    def _open_pass(self, fbo):
        if self._pass_fbo is fbo:
            return
        color, depth = self._clears.pop(fbo, (None, None))
        self._end_pass()
        self._flush_clears(keep=fbo)
        colors = (C.c_void_p * len(fbo.color_attachments))(
            *(t.handle for t in fbo.color_attachments)
        )
        self.api.call(
            "pass",
            self.handle,
            colors,
            len(colors),
            fbo.depth_attachment.handle if fbo.depth_attachment else None,
            (C.c_double * 4)(*(color or (0, 0, 0, 0))),
            int(color is not None) + 2 * int(depth is not None),
            1.0 if depth is None else depth,
        )
        self._pass_fbo = fbo

    def _read_texture(self, tex, rect):
        self._flush_clears()
        self._end_pass()
        x, y, w, h = map(int, rect or (0, 0, *tex.size))
        out = C.create_string_buffer(w * h * tex._bpp)
        self.api.call(
            "read_texture", self.handle, tex.handle, out, x, y, w, h, tex._bpp
        )
        return out.raw

    def _sampler(self, tex):
        key = tex.sampler_key()
        if key not in self._samplers:
            mn, mg, mip, rx, ry = key
            self._samplers[key] = self.api.call(
                "sampler",
                self.handle,
                int(mn == LINEAR),
                int(mg == LINEAR),
                0 if mip is None else 1 + int(mip == LINEAR),
                int(rx),
                int(ry),
            )
        return self._samplers[key]

    def _dummy(self, typ):
        key = typ.endswith("MS")
        if key not in self._dummies:
            t = self._dummies[key] = Texture(
                self, (1, 1), 4, None, 4 if key else 0, "f1" if key else "f4"
            )
            Framebuffer(self, [t]).clear()
        return self._dummies[key]

    def _textures(self, prog, textures=None):
        if textures is None:
            textures = {}
            for name, (binding, typ) in prog.src.samplers.items():
                tex = self._units.get(prog._units.get(name, 0))
                if (
                    tex is None
                    or tex._released
                    or (tex.samples > 1) != typ.endswith("MS")
                ):
                    tex = self._dummy(typ)
                textures[binding] = tex
        bindings = []
        for stage, shader in enumerate((prog._vs, prog._fs)):
            for binding, tex in textures.items():
                if binding in shader.bindings:
                    texture, sampler = shader.bindings[binding]
                    bindings.append(
                        TextureBinding(
                            stage, texture, sampler, tex.handle, self._sampler(tex)
                        )
                    )
        return (TextureBinding * len(bindings))(*bindings)

    def _pipeline(
        self, prog, vao, mode, fbo, wide=False, internal_depth=None, color_write=True
    ):
        blend = self._blend if self._flags & BLEND and vao is not None else None
        depth = (
            (bool(self._flags & DEPTH_TEST) and fbo.depth_attachment is not None)
            if vao is not None
            else internal_depth is not None
        )
        write = depth and (self._depth_mask if vao is not None else True)
        compare = _DEPTH[self.depth_func] if depth and vao is not None else 7
        layout = vao.layout_key if vao else ()
        key = (
            prog.id,
            wide,
            layout,
            mode,
            fbo.rp_key,
            blend,
            depth,
            write,
            compare,
            fbo.color_mask,
            color_write,
        )
        if key in self._pipelines:
            return self._pipelines[key]
        attrs = []
        binds = []
        vs = prog._vs
        if wide:
            vs, locations = prog.wide()
        if vao:
            for k, (_buf, stride, inst, attributes) in enumerate(vao.bindings):
                binds.append((stride * (2 if wide else 1), bool(inst or wide)))
                for name, kind, count, offset in attributes:
                    if wide and name in locations:
                        for loc, off in zip(
                            locations[name][1:], (offset, offset + stride)
                        ):
                            attrs.append((loc, k, _ATTR[kind][count - 1], off))
                    elif not wide and name in prog._inputs:
                        attrs.append(
                            (prog._inputs[name][1], k, _ATTR[kind][count - 1], offset)
                        )
        factors = blend if blend and len(blend) == 4 else (blend * 2 if blend else ())
        description = {
            "vs": vs.entry,
            "fs": prog._fs.entry,
            "samples": fbo.samples,
            "formats": [t.format for t in fbo.color_attachments],
            "attributes": attrs,
            "bindings": binds,
            "compare": compare,
            "depth_write": write,
            "mask": sum(bit for bit, on in zip((8, 4, 2, 1), fbo.color_mask) if on)
            if color_write
            else 0,
            "blend": [_BLEND[x] for x in factors],
        }
        pipe = self.api.call(
            "pipeline",
            self.handle,
            vs.handle,
            prog._fs.handle,
            json.dumps(description).encode(),
        )
        self._pipelines[key] = pipe
        return pipe

    def _draw(self, vao, mode, vertices, first, instances):
        prog = vao.program
        prog._wait()
        indexed = vao.index_buffer is not None
        if vertices < 0:
            vertices = (
                vao.index_buffer.size // 4 - first
                if indexed
                else min(
                    (
                        b.size // stride
                        for b, stride, inst, _ in vao.bindings
                        if not inst and stride
                    ),
                    default=0,
                )
                - first
            )
        if vertices <= 0:
            return
        if mode not in _TOPOLOGY:
            raise Error("Unsupported topology: %s" % mode)
        wide = (
            mode == LINES
            and self.line_width > 1.001
            and not indexed
            and bool(prog.wide())
        )
        fbo = self._fbo
        pipe = self._pipeline(prog, vao, mode, fbo, wide)
        prog._set_line(fbo.size, self.line_width)
        textures = self._textures(prog)
        buffers = (VertexBinding * len(vao.bindings))(
            *(
                VertexBinding(b.handle, first * stride if wide else 0)
                for b, stride, _i, _a in vao.bindings
            )
        )
        self._open_pass(fbo)
        x, y, w, h = self.viewport
        H = fbo.height
        sx, sy, sw, sh = self.scissor or (0, 0, *fbo.size)
        sx = max(0, sx)
        sy = max(0, sy)
        sw = max(0, min(sw, fbo.width - sx))
        sh = max(0, min(sh, H - sy))
        if not sw or not sh:
            return
        self.api.call(
            "draw",
            self.handle,
            pipe,
            buffers,
            len(buffers),
            textures,
            len(textures),
            bytes(prog._block),
            prog._block_size,
            (C.c_double * 4)(x, H - y - h, w, h),
            (C.c_int * 4)(sx, H - sy - sh, sw, sh),
            3 if wide else _TOPOLOGY[mode],
            6 if wide else vertices,
            0 if wide else first,
            vertices // 2 if wide else max(1, instances),
            vao.index_buffer.handle if indexed else None,
        )

    def _internal_program(self, name):
        if name not in self._internal:
            self._internal[name] = Program(self, name)
        return self._internal[name]

    def copy_framebuffer(self, dst, src):
        s, d = src.color_attachments[0], dst.color_attachments[0]
        if s.samples > 1 and d.samples == 1:
            p = self._internal_program("resolve%d" % s.samples)
            textures = self._textures(p, {1: s})
            pipe = self._pipeline(p, None, TRIANGLES, dst)
            self._clear(dst, (0, 0, 0, 0), None)
            self._open_pass(dst)
            self.api.call(
                "draw",
                self.handle,
                pipe,
                None,
                0,
                textures,
                len(textures),
                bytes(p._block),
                32,
                (C.c_double * 4)(0, 0, *dst.size),
                (C.c_int * 4)(0, 0, *dst.size),
                3,
                3,
                0,
                1,
                None,
            )
        else:
            self._flush_clears()
            self._end_pass()
            self.api.call("copy_texture", self.handle, s.handle, d.handle)

    def present(self, overlay):
        self._flush_clears()
        self._end_pass()
        size = self.screen.size
        if self._overlay is None or self._overlay.size != size:
            if self._overlay:
                self._overlay.release()
            self._overlay = Texture(self, size)
        img = overlay.convertToFormat(QImage.Format_RGBA8888_Premultiplied)
        self._overlay.write(bytes(img.constBits()))
        self.api.call(
            "present",
            self.handle,
            self.screen.color_attachments[0].handle,
            self._overlay.handle,
            *size,
        )

    def ray_pick(self, meshes, origins, directions, minimum=0.0, maximum=1e30):
        """Batch MetalRT queries against cached geometry BLAS and an instanced scene TLAS.

        Returns the same position/object-id and primitive arrays as the raster picking path. IDs refer to the
        caller's mesh order, allowing surfaces to keep their existing double-precision hit refinement.
        """
        from ..renderer import _mesh_data

        origins = np.asarray(origins, dtype="f4").reshape(-1, 3)
        directions = np.asarray(directions, dtype="f4").reshape(-1, 3)
        n = len(origins)
        data = np.zeros((n, 4), "f4")
        primitives = np.zeros(n, "i4")
        valid = [
            (i, m)
            for i, m in enumerate(meshes)
            if m.geo is not None and len(m.geo.faces)
        ]
        if not valid or not n:
            return data, primitives
        self._flush_clears()
        self._end_pass()
        key = tuple((m.geo, np.asarray(m.xform).tobytes()) for _, m in valid)
        if key != self._rt_scene_key:
            structures = []
            transforms = []
            for _, mesh in valid:
                geo = mesh.geo
                cached = self._rt_cache.get(id(geo))
                if cached is None or cached[0] is not geo:
                    vbo, _, ibo, _ = _mesh_data(self, geo)
                    structure = self.api.call(
                        "rt_geometry",
                        self.handle,
                        vbo.handle,
                        ibo.handle,
                        len(geo.faces),
                    )
                    cached = self._rt_cache[id(geo)] = (geo, structure)
                structures.append(cached[1])
                transforms.append(np.asarray(mesh.xform, "f4")[:3, :].T)
            xforms = np.ascontiguousarray(transforms, "f4")
            handles = (C.c_void_p * len(structures))(*structures)
            self.api.call(
                "rt_scene", self.handle, handles, xforms.ctypes.data, len(structures)
            )
            self._rt_scene_key = key
            alive = {id(m.geo) for _, m in valid}
            for gid in list(self._rt_cache):
                if gid not in alive:
                    self.api.release(self._rt_cache.pop(gid)[1])
        rays = np.empty((n, 8), "f4")
        rays[:, :3] = origins
        rays[:, 3] = minimum
        rays[:, 4:7] = directions
        rays[:, 7] = maximum
        hits = np.empty(
            n,
            dtype=[
                ("distance", "<f4"),
                ("instance", "<u4"),
                ("primitive", "<u4"),
                ("found", "<u4"),
            ],
        )
        self.api.call("rt_trace", self.handle, rays.ctypes.data, hits.ctypes.data, n)
        mask = hits["found"] != 0
        if mask.any():
            data[mask, :3] = (
                origins[mask] + directions[mask] * hits["distance"][mask, None]
            )
            indices = np.asarray([i + 1 for i, _ in valid], "i4")
            data[mask, 3] = indices[hits["instance"][mask]]
            primitives[mask] = hits["primitive"][mask]
        return data, primitives

    def diagnostics(self):
        info = self.api.info(self.handle)
        info["libraries"] = len(self._shaders) + 1
        info["pipelines"] = len(self._pipelines)
        return info

    def grab(self):
        w, h = self.screen.size
        out = C.create_string_buffer(w * h * 4)
        tex = self.api.call("output", self.handle)
        self.api.call("read_texture", self.handle, tex, out, 0, 0, w, h, 4)
        return QImage(out.raw, w, h, w * 4, QImage.Format_RGBA8888).copy()

    def release(self):
        if self.handle:
            self._flush_clears()
            self._end_pass()
            self.api.call("submit", self.handle, 1)
            for s in self._shaders:
                s.release()
            for p in self._pipelines.values():
                self.api.release(p)
            for s in self._samplers.values():
                self.api.release(s)
            for _, structure in self._rt_cache.values():
                self.api.release(structure)
            self.api.lib.mc_destroy(self.handle)
            self.handle = None
