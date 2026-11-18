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

"""Build the Apple-only Metal/MetalKit bridge. No Qt headers or libraries are used."""

import hashlib
from functools import lru_cache
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

SOURCE = Path(__file__).with_name("native")
SHADERS = Path(__file__).with_name("shaders")
NAMES = ("libmule_metal.dylib", "builtin.metallib", "scene.metallib")


def _run(args):
    result = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    return result.stdout.strip()


@lru_cache(maxsize=1)
def metal_compiler():
    if sys.platform != "darwin" or not shutil.which("xcrun"):
        return None
    try:
        return _run(["xcrun", "--sdk", "macosx", "--find", "metal"])
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return None


def compile_library(source, output):
    """Compile MSL sources directly into one library, without Qt shader tools."""
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    with tempfile.TemporaryDirectory(prefix="mule_metal_") as tmp:
        def compile_one(item):
            index, path = item
            air = Path(tmp) / (str(index) + ".air")
            _run(["xcrun", "--sdk", "macosx", "metal", "-std=metal4.1", "-mmacosx-version-min=27.0",
                  "-c", path, "-o", air])
            return air
        with ThreadPoolExecutor(max_workers=8) as pool:
            objects = list(pool.map(compile_one, enumerate(sources)))
        _run(["xcrun", "--sdk", "macosx", "metallib", *objects, "-o", output])


def build(output):
    if sys.platform != "darwin":
        raise RuntimeError("The native Metal backend must be built on macOS")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "xcrun",
            "--sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-fobjc-arc",
            "-dynamiclib",
            "-O2",
            "-mmacosx-version-min=27.0",
            SOURCE / "renderer.mm",
            "-o",
            output / NAMES[0],
            "-framework",
            "Metal",
            "-framework",
            "MetalKit",
            "-framework",
            "AppKit",
            "-framework",
            "Foundation",
            "-Wl,-install_name,@rpath/" + NAMES[0],
        ]
    )
    compile_library(SOURCE / "builtin.metal", output / NAMES[1])
    compile_library(sorted(SHADERS.glob("*.metal")), output / NAMES[2])
    return output


def locate():
    bundled = Path(__file__).with_name("built")
    if all((bundled / n).is_file() for n in NAMES):
        return bundled
    if getattr(sys, "frozen", False):
        raise RuntimeError("The native Metal backend is not bundled")
    digest = hashlib.sha256()
    for p in (*sorted(SOURCE.rglob("*")), *sorted(SHADERS.rglob("*")), Path(__file__)):
        if not p.is_file():
            continue
        digest.update(p.read_bytes())
    digest.update(platform.machine().encode())
    folder = Path.home() / "Library/Caches/Mule/native-metal" / digest.hexdigest()[:24]
    if all((folder / n).is_file() for n in NAMES):
        return folder
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=folder.parent) as tmp:
        build(tmp)
        folder.mkdir(exist_ok=True)
        for name in NAMES:
            os.replace(Path(tmp) / name, folder / name)
    return folder


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("python -m sketch_machine.metal.build <output_directory>")
    print(build(sys.argv[1]))
