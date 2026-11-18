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

"""Native Metal 4 rendering, MetalKit presentation, and MetalRT picking on macOS."""

import os
import platform
import sys
import logging
import subprocess
from functools import lru_cache


def supported_os():
    return sys.platform == "darwin" and int(platform.mac_ver()[0].split(".")[0] or 0) >= 27


@lru_cache(maxsize=1)
def capabilities():
    """Probe the launch machine and shipped libraries, without creating a viewport."""
    if not supported_os():
        return {"available": False, "reason": "Native Metal requires macOS 27 or later"}
    try:
        from .api import API
        return API().capabilities()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": str(exc)}


def wanted():
    choice = os.environ.get("MULE_GPU", "").strip().lower()
    if choice in ("gl", "opengl"):
        return False
    forced = choice in ("metal", "native")
    if sys.platform != "darwin" and not forced:
        return False
    result = capabilities()
    if forced and not result["available"]:
        raise RuntimeError(result["reason"])
    if not result["available"] and supported_os():
        logging.getLogger(__name__).info("Using OpenGL: %s", result["reason"])
    return result["available"]
