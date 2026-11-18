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

"""Native shader entry points and explicit host/MSL resource layouts.

programs.json describes the checked-in MSL sources, including uniform byte offsets
and texture/sampler slots. No shader translation or Qt reflection runs here.
"""
from functools import lru_cache
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np

from .types import Uniform, _IDS


@lru_cache(maxsize=1)
def definitions():
    return json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))


class Program:
    def __init__(self, ctx, name):
        self.ctx = ctx
        self.id = next(_IDS)
        spec = definitions()[name]
        self._vs = ctx._function(spec["vertex"])
        self._fs = ctx._function(spec["fragment"])
        self._inputs = spec["inputs"]
        self._members = {member["name"]: member for member in spec["members"]}
        self._block_size = spec["block_size"]
        self._block = bytearray(max(16, self._block_size))
        self._view = np.frombuffer(self._block, np.uint8)
        self.src = SimpleNamespace(samplers=spec["samplers"])
        self._units = {}
        self._uniforms = {}
        self._names = set(self._inputs) | set(self._members) | set(self.src.samplers)
        self._wide = spec["wide"]

    def _wait(self):
        pass

    def __contains__(self, name):
        return name in self._names

    def __getitem__(self, name):
        if name not in self._uniforms:
            if name in self.src.samplers:
                self._uniforms[name] = Uniform(self, name, sampler=True)
            elif name in self._members:
                self._uniforms[name] = Uniform(self, name, self._members[name])
            else:
                raise KeyError(name)
        return self._uniforms[name]

    def get(self, name, default=None):
        return self[name] if name in self else default

    def _set_line(self, size, width):
        member = self._members.get("line_info")
        if member is not None:
            struct.pack_into("<4f", self._block, member["offset"], *size, width, 0.0)

    def wide(self):
        if self._wide is None:
            return False
        return self.ctx._function(self._wide["vertex"]), self._wide["locations"]

    def release(self):
        pass
