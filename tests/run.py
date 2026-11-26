"""Test runner: all tests/test_*.py, each file in its own process (its own app window).

    python tests/run.py                 all tests (levels 1 and 2)
    python tests/run.py paint layers    only files whose name contains one of the words
    python tests/run.py --core          level 1 only: core tests without a window (seconds) - after every change
    python tests/run.py --gross         level 3: check run on the large harbour scene (minutes) - after every plan step
                                        and before every package; needs the frozen scene (bench/pruefrunde.py)

Three levels (tests/README.md):
    1 Core      file carries the line  LEVEL = "core"  - no window, the tests get h = None
    2 Handling  all other files - real window, real mouse events
    3 Large     bench/pruefrunde.py and bench/hafen.py on the harbour scene
    python tests/run.py --update-baseline   recreate the reference images of the pixel comparison

Result per test:
    OK          passed
    FAIL        failed (must be fixed)
    EXPECTED    known problem (expected_failure), fails as expected
    FIXED       known problem suddenly passes - remove the marker
Exit code 1 if there is a FAIL.
"""
import glob
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_file(path, out_json):
    """In the child process: run all test_* functions of one file one after another."""
    sys.path.insert(0, ROOT)
    import importlib.util
    from tests.harness import H, QT_ERRORS
    spec = importlib.util.spec_from_file_location(os.path.splitext(os.path.basename(path))[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    core = getattr(mod, "LEVEL", "") == "core"  # level 1: without a window (the harness check modes still apply)
    h = None if core else H()
    results = []
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        fn = getattr(mod, name)
        expected = getattr(fn, "expected_failure", None)
        t = time.perf_counter()
        try:
            if h is not None:
                h.new_scene()
            QT_ERRORS.clear()
            note = fn(h)
            if h is not None:
                h.grab()
            if QT_ERRORS:  # error in paintGL or similar: coloured boxes in the viewport
                lines = QT_ERRORS[0].splitlines()
                raise RuntimeError("Error in a Qt callback: " + lines[-1] + "\n" + "\n".join(lines[-4:-1]))
            status, detail = ("FIXED", expected) if expected else ("OK", note or "")
        except Exception as exc:  # noqa: BLE001
            if expected:
                status, detail = "EXPECTED", f"{expected} – {type(exc).__name__}: {exc}"
            else:
                status, detail = "FAIL", "".join(traceback.format_exception(exc)).strip().splitlines()[-1] + \
                    "\n" + "".join(traceback.format_tb(exc.__traceback__)[-2:])
        results.append({"test": name, "status": status, "detail": str(detail), "s": round(time.perf_counter() - t, 2)})
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f)
    if h is not None:
        h.close()


def is_core(path):
    """Level 1? (recognised from the source text, without loading the file)"""
    with open(path, encoding="utf-8") as f:
        return any(line.startswith('LEVEL = "core"') for line in f)


def _neutral_scale():
    """Tests read pixels at widget coordinates (device pixel ratio 1, as on the Windows machines). On a Retina Mac the
    screen has ratio 2: then the windows run with QT_SCALE_FACTOR = 1/ratio, so one widget pixel is one image pixel
    again. Returns the factor as text, or None (ratio 1, other systems, or already set)."""
    if sys.platform != "darwin" or "QT_SCALE_FACTOR" in os.environ:
        return None
    code = ("from PySide6.QtGui import QGuiApplication; a = QGuiApplication([]); "
            "print(a.primaryScreen().devicePixelRatio())")
    try:
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout
        ratio = float(out.strip().splitlines()[-1])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None
    return None if abs(ratio - 1.0) < 1e-3 or ratio <= 0 else "%.6g" % (1.0 / ratio)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--child" in sys.argv:
        run_file(args[0], args[1])
        return 0
    env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
    scale = _neutral_scale()
    if scale:
        env["QT_SCALE_FACTOR"] = scale
    if "--update-baseline" in sys.argv:
        env["SKETCH_UPDATE_BASELINE"] = "1"
    if "--gross" in sys.argv:
        scene = os.path.join(ROOT, "Feedback", "2026-11-11_architektur", "messlatte", "hafen_messlatte.mule")
        if not os.path.exists(scene):
            print("Level 3 needs the frozen harbour scene:\n  " + scene + "\n(not in Git)")
            return 1
        return subprocess.run([sys.executable, os.path.join(ROOT, "bench", "pruefrunde.py")], cwd=ROOT, env=env).returncode
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    if args:
        files = [f for f in files if any(a in os.path.basename(f) for a in args)]
    if "--core" in sys.argv or "--kern" in sys.argv:  # --kern: old name
        files = [f for f in files if is_core(f)]
    else:
        files.sort(key=lambda f: not is_core(f))  # fast ones first: an error in the core shows within seconds
    out_dir = os.path.join(ROOT, "tests", "_out")
    os.makedirs(out_dir, exist_ok=True)
    totals = {}
    start = time.perf_counter()
    for f in files:
        name = os.path.basename(f)
        out_json = os.path.join(out_dir, name + ".json")
        if os.path.exists(out_json):
            os.remove(out_json)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", f, out_json], cwd=ROOT, env=env,
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900)
        results = json.load(open(out_json, encoding="utf-8")) if os.path.exists(out_json) else []
        print(f"\n== {name}")
        for r in results:
            totals[r["status"]] = totals.get(r["status"], 0) + 1
            line = f"  {r['status']:9} {r['test']} ({r['s']} s)"
            if r["status"] != "OK" or r["detail"]:
                line += f"\n             {r['detail'].strip()}".replace("\n", "\n             ")
            print(line)
        if proc.returncode != 0:
            totals["FAIL"] = totals.get("FAIL", 0) + 1
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-6:]
            print("  FAIL      Process aborted:\n             " + "\n             ".join(tail))
    print(f"\nTotal ({time.perf_counter() - start:.0f} s): " +
          ", ".join(f"{k} {v}" for k, v in sorted(totals.items())))
    return 1 if totals.get("FAIL") else 0


if __name__ == "__main__":
    sys.exit(main())
