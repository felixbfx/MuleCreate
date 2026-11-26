# macOS setup, graphics and builds

On macOS, `./start.sh` uses `uv` to prepare Python and dependencies, builds `dist/Mule.app`, and launches the
executable inside that app. On Linux the same script uses `uv` to launch from source. Windows uses `start.bat` in
the repository root (requires Python 3.14), or `Mule\Mule.exe` from a packaged Windows release.

## Prerequisites

Follow the [README's macOS installation steps](../README.md#install-and-start) to install Homebrew, `uv`, and full
Xcode 27 or newer with its Metal toolchain. The macOS packaging spec compiles the native backend on every build,
including builds intended to run with OpenGL. The standalone Command Line Tools are insufficient.

The native backend targets macOS 27 or later, a Metal 4-capable GPU, and MSL 4.1. The selected macOS SDK must be
version 27 or newer. Check the active tools with:

```sh
xcodebuild -version
xcrun --sdk macosx --show-sdk-version
xcrun --sdk macosx metal --version
xcrun --sdk macosx --find metallib
```

If `xcode-select -p` points to `/Library/Developer/CommandLineTools`, select the full Xcode installation as shown
in the README. A missing `metal` tool or an error about `-std=metal4.1` means the selected toolchain is missing or
too old. The Metal toolchain is an additional download; see
[Apple's component installation instructions](https://developer.apple.com/documentation/xcode/downloading-and-installing-additional-xcode-components).

## Build and launch

Download and unpack the source, or clone the repository, into a folder you can write to. From that folder:

```sh
./start.sh
```

uv provides Python 3.14 and the pinned packages in `requirements.txt` and `requirements-build.txt` in an isolated,
cached environment. It downloads missing Python versions and packages, so first use needs internet. You do not
need to activate a virtual environment or install Python separately.

The script runs PyInstaller with `packaging/mule.spec`, then starts `dist/Mule.app/Contents/MacOS/Mule`. It passes
through arguments and environment variables. Each invocation builds before launching; a failed build stops the
script. To run the existing bundle directly, double-click `dist/Mule.app` in Finder.

At runtime, the app probes the Mac's default GPU and bundled shader libraries and selects Metal when available.
Automatic selection falls back to OpenGL if the native capability probe or library loading fails. These options
select a renderer for the app after building:

- `MULE_GPU=metal ./start.sh` requires the native backend.
- `MULE_GPU=metal MULE_METALRT=0 ./start.sh` requires Metal rendering with raster picking.
- `MULE_GPU=gl ./start.sh` selects OpenGL. The build still requires Xcode and its Metal toolchain.

To build and run the packaged self-test:

```sh
./start.sh --selftest /tmp/mule-selftest.json
```

The spec bundles the native Metal/MetalKit bridge, native shader libraries and program layouts, along with the
Python runtime and OpenGL dependencies. Compilation errors stop the build. The resulting app needs no Xcode,
development headers or qsb at runtime. The bundle declares a macOS 13 deployment floor for the OpenGL fallback;
the native library loads only on macOS 27+. Package self-test results include the active renderer and native
draw/ray-query counts under `gpu`.

## Native rendering

Qt provides windows, menus, input and the QPainter overlay. The scene renderer creates no QRhi resources, pipelines
or command buffers. Windows and Linux retain their existing OpenGL 3.3 renderer and never load the native Metal bridge.

`sketch_machine/metal/context.py` implements the renderer's existing buffer, texture, program, framebuffer and draw
API. The Objective-C++ bridge in `metal/native/renderer.mm` owns the Metal device, `MTL4CommandQueue`, command
allocators, render and compute encoders, argument tables, residency sets and pipelines built with `MTL4Compiler`.
Three frame slots keep resources alive until GPU completion. Uploads, multisampling, depth/blend state, synchronous
readbacks and screenshots use this backend too.

A MetalKit `MTKView` presents the viewport inside a Qt native view. Qt receives mouse, keyboard and tablet events;
MetalKit manages drawable acquisition and display. The scene and premultiplied QPainter overlay are composited on
the GPU. Normal presentation performs no CPU image readback.

Native scene shader sources live in `sketch_machine/metal/shaders/*.metal`. Edit these MSL files directly;
`metal/build.py` compiles them with `-std=metal4.1` into `scene.metallib`. The compositor and ray-query kernel in
`metal/native/builtin.metal` compile into `builtin.metallib`. Named programs, uniform byte offsets, vertex attributes,
and texture/sampler slots are defined in `metal/programs.json`. Compile-time assertions in the MSL sources check
uniform sizes and offsets against the host layouts.

The initial MSL port preserves the previous shader calculations and coordinate conventions. These checked-in
sources are now maintained independently; building and running native Metal does not invoke qsb, read QShader
packs, or import RHI. The shared renderer requests programs by name. The existing GLSL sources and OpenGL rendering
remain in `renderer.py`; OpenGL compiles those same sources through moderngl.

## MetalRT picking and drawing hits

MetalRT builds a bottom-level acceleration structure once per geometry and an instanced top-level structure when
visible objects or their transforms change. A batched ray-query compute kernel returns nearest hits, object indices
and triangle IDs. Mesh picking and drawing on meshes/surfaces use these hits. Drawing surfaces retain their existing
double-precision hit refinement. Hidden/isolated objects, instance transforms, near/far clipping and orthographic
views are respected. The sketch's lighting and appearance remain the same.

Devices without ray-tracing support, or runs with `MULE_METALRT=0`, use the existing raster picking path. Native
self-tests exercise actual acceleration-structure builds and ray dispatches and report GPU draw/query counts.

## Validation

Run these from the repository root after installing the prerequisites. uv prepares the source test environment.
The native checks also require a supported Mac and the Metal toolchain; forcing `MULE_GPU=metal` prevents a
fallback run from being mistaken for native validation.

```sh
uv run --no-project --isolated --python 3.14 --with-requirements requirements.txt python tests/run.py --core
MULE_GPU=metal MTL_DEBUG_LAYER=1 uv run --no-project --isolated --python 3.14 --with-requirements requirements.txt python tests/run.py native_metal picking paint mesh_block double_click_depth
MULE_GPU=metal MTL_DEBUG_LAYER=1 QT_SCALE_FACTOR=1 uv run --no-project --isolated --python 3.14 --with-requirements requirements.txt python tests/run.py test_render
MULE_GPU=gl dist/Mule.app/Contents/MacOS/Mule --selftest /tmp/mule-gl-selftest.json
```

Native tests cover upload/readback ordering, nearest-hit selection, misses, nonuniform instance transforms, hidden
objects and geometry replacement. The scene reference tests compare normal, lighting and X-ray output. Existing
Retina reference images use scale 1; the rest of the suite normally uses its automatic pixel-neutral scale.

References: [Metal 4 core API](https://developer.apple.com/documentation/metal/understanding-the-metal-4-core-api),
[MetalKit](https://developer.apple.com/documentation/metalkit/mtkview),
[Metal ray tracing](https://developer.apple.com/documentation/metal/ray-tracing-with-acceleration-structures),
[MSL 4.1 specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf).
