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

"""The narrow C ABI between the Python renderer and the Objective-C++ Metal backend."""

import ctypes as C
import json

from . import build


class VertexBinding(C.Structure):
    _fields_ = [("buffer", C.c_void_p), ("offset", C.c_uint64)]


class TextureBinding(C.Structure):
    _fields_ = [
        ("stage", C.c_int),
        ("texture", C.c_int),
        ("sampler", C.c_int),
        ("image", C.c_void_p),
        ("state", C.c_void_p),
    ]


class API:
    def __init__(self):
        self.folder = build.locate()
        self.lib = C.CDLL(str(self.folder / build.NAMES[0]))
        P, I, Z, D, S = C.c_void_p, C.c_int, C.c_size_t, C.c_double, C.c_char_p
        declarations = {
            "error": (S, []),
            "probe": (S, [S, S]),
            "create": (P, [P, S]),
            "destroy": (None, [P]),
            "release": (None, [P]),
            "info": (S, [P]),
            "submit": (I, [P, I]),
            "buffer": (P, [P, P, Z]),
            "write_buffer": (I, [P, P, P, Z, Z]),
            "read_buffer": (I, [P, P, P, Z, Z]),
            "texture": (P, [P, I, I, I, I, I]),
            "write_texture": (I, [P, P, P, I, I, I, I, I, I]),
            "read_texture": (I, [P, P, P, I, I, I, I, I]),
            "mipmap": (I, [P, P]),
            "copy_texture": (I, [P, P, P]),
            "sampler": (P, [P, I, I, I, I, I]),
            "library": (P, [P, P, Z]),
            "pipeline": (P, [P, P, P, S]),
            "pass": (I, [P, P, I, P, P, I, D]),
            "end_pass": (I, [P]),
            "draw": (I, [P, P, P, I, P, I, P, Z, P, P, I, I, I, I, P]),
            "present": (I, [P, P, P, I, I]),
            "output": (P, [P]),
            "rt_geometry": (P, [P, P, P, Z]),
            "rt_scene": (I, [P, P, P, I]),
            "rt_trace": (I, [P, P, P, I]),
        }
        for name, (result, args) in declarations.items():
            fn = getattr(self.lib, "mc_" + name)
            fn.restype = result
            fn.argtypes = args

    def call(self, name, *args):
        value = getattr(self.lib, "mc_" + name)(*args)
        if not value:
            raise RuntimeError(
                self.lib.mc_error().decode() or "Native Metal operation failed: " + name
            )
        return value

    def release(self, handle):
        if handle:
            self.lib.mc_release(handle)

    def info(self, handle):
        return json.loads(self.call("info", handle))

    def capabilities(self):
        return json.loads(self.call(
            "probe", str(self.folder / "builtin.metallib").encode(),
            str(self.folder / "scene.metallib").encode(),
        ))
