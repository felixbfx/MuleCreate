#!/usr/bin/env bash

# The app brings its own Python (like Houdini or Blender): on the first start this script uses uv to set up a
# standalone Python 3.14 inside the project folder (runtime/python) and from it the environment .venv with exactly the
# package versions of requirements.txt. The system Python (e.g. 3.9) is neither used nor changed, no admin rights are
# needed. After that the app starts without uv and without internet – also for other users who can read the folder.

set -u

SELF="${BASH_SOURCE[0]}"

while [ -L "$SELF" ]; do
    BASE="$(cd -P "$(dirname "$SELF")" && pwd)" || exit 1
    SELF="$(readlink "$SELF")" || exit 1
    case "$SELF" in /*) ;; *) SELF="$BASE/$SELF" ;; esac
done

HERE="$(cd -P "$(dirname "$SELF")" && pwd)" || exit 1

cd "$HERE" || exit 1

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

PLATFORM="$(uname -s)"
case "$PLATFORM" in
    Darwin|Linux) ;;
    *) fail "Use start.sh on macOS or Linux. On Windows, use start.bat." ;;
esac

# Prepare dependencies with uv, then build and launch Mule.app on macOS or run from source on Linux.

command -v uv >/dev/null 2>&1 \
    || fail "uv is required. On macOS, run 'brew install uv'. For Linux installation steps, see README.md."

# uv manages Python and caches the environment for these pinned requirements.
UV_ARGS=(run --no-project --isolated --managed-python --python 3.14
         --with-requirements "$HERE/requirements.txt")
export PYTHONFAULTHANDLER=1

if [ "$PLATFORM" = Darwin ]; then
    xcrun --sdk macosx metal --version >/dev/null 2>&1 \
        || fail "Install Xcode 27 or newer and its Metal toolchain, then run ./start.sh again. See README.md."
    printf '%s\n' 'Mule: building dist/Mule.app...'
    uv "${UV_ARGS[@]}" --with-requirements "$HERE/requirements-build.txt" \
        python -m PyInstaller packaging/mule.spec --noconfirm || exit "$?"
    APP="$HERE/dist/Mule.app/Contents/MacOS/Mule"
    [ -x "$APP" ] || fail "The build did not produce dist/Mule.app. Check the build output above."
    exec "$APP" "$@"
fi

exec uv "${UV_ARGS[@]}" python -m sketch_machine "$@"
