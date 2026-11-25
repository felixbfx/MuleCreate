"""Mule Image: write the prompt files, the input pictures and STEPS.md for the look step.

    python image/build_prompts.py                 # in the brief folder
    python build_prompts.py --brief <brief folder> --out <folder>   # read one brief, write somewhere else

Reads  brief.json (hero cameras), image/look.txt, image/shots.md, image/parts/*.txt
Writes image/prompts/, image/inputs/, image/STEPS.md - all three are rebuilt on every run.
Never touches gen/ or image/approved/. No network, no cost.

Standard library only. The location pictures (wide view with the close camera drawn in) need PySide6;
without it they are left out and the prompts do without them.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ("pair", "decides", "scale", "note", "like", "legend")
RATIOS = [(1, 1), (5, 4), (4, 3), (3, 2), (16, 9), (2, 1), (21, 9)]

SHOTS_HEAD = """# Shots
<!--
Which hero cameras take part in the look step, and what is in each view.
One section per camera: "## <camera name as in Mule>". Cameras without a section do not take part
(unless their prompt field in Mule has text - a section here replaces that text).

Lines in a section:
  pair: <camera name>     this is a close view; the named camera is its wide view
  decides: how ...        close views: what this view settles, e.g. "how machines and galleries are built"
  like: <camera name>     further views: the close view of the same kind (default: the first close view)
  scale: ...              a real size visible in this view
  note: ...               one sentence about the structure image, e.g. "The pale arcs are not objects; do not paint them."
  everything else         what the forms in the structure image are, front to back ("In this view")

Above the first section, for all views:
  legend: yellow is ..., teal is ...    what the flat colours of the sketch stand for
  scale: ...                            real sizes of the whole scene

Order: close views are made first, then their wide view with the approved close views attached
(at most two close views per wide view), then the further views.
Run build_prompts.py again after every change.
-->

"""

LOOK_TODO = ("TODO - replace this whole text with the look paragraph: one paragraph in English, from the "
             "artist's notes only. Place and time, what the thing is for, how it is made and of what, colours, "
             "wear, how detail is spread, light and weather, the kind of picture. No shapes, no positions.\n")


def stop(msg):
    print(msg)
    sys.exit(2)


def part(name):
    with open(os.path.join(HERE, "parts", name + ".txt"), encoding="utf-8") as f:
        return f.read().strip()


def parse_fields(text):
    """Text of a section (or of Mule's prompt field) -> {"pair": .., "note": [..], "view": ..}."""
    out = {"note": [], "view": []}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(?i)^(%s)\s*:\s*(.*)$" % "|".join(KEYS), line)
        if m:
            key, val = m.group(1).lower(), m.group(2).strip()
            if key == "note":
                out["note"].append(val)
            elif val:
                out[key] = val
        else:
            out["view"].append(re.sub(r"(?i)^in this view\s*:\s*", "", line))
    out["view"] = " ".join(out["view"])
    return out


def parse_shots(text):
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    head, sections, name, buf = [], [], None, []
    for line in text.splitlines():
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if m:
            if name is not None:
                sections.append((name, "\n".join(buf)))
            name, buf = m.group(1).strip("` "), []
        elif line.startswith("# "):
            continue
        elif name is None:
            head.append(line)
        else:
            buf.append(line)
    if name is not None:
        sections.append((name, "\n".join(buf)))
    return parse_fields("\n".join(head)), [(n, parse_fields(t)) for n, t in sections]


def fmt(aspect):
    land = aspect >= 1
    a = aspect if land else 1 / aspect
    w, h = min(RATIOS, key=lambda r: abs(r[0] / r[1] - a))
    if w == h:
        return "square"
    return f"landscape, about {w}:{h}" if land else f"portrait, about {h}:{w}"


def slug(cam):
    s = re.sub(r"[^a-z0-9]+", "_", cam["id"].lower()).strip("_")
    return s[5:] if s.startswith("hero_") else s


# ---- location picture: the wide picture with the close camera as dot and wedge

def _sub(a, b): return [a[i] - b[i] for i in range(3)]
def _add(a, b): return [a[i] + b[i] for i in range(3)]
def _mul(a, s): return [x * s for x in a]
def _dot(a, b): return sum(a[i] * b[i] for i in range(3))
def _cross(a, b): return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _unit(a):
    n = math.sqrt(_dot(a, a)) or 1.0
    return [x / n for x in a]


def location_picture(brief, close, wide, up_axis, out):
    """True when written. Dot = eye of the close camera, wedge = its horizontal field of view, drawn at the
    height of its target and reaching a quarter beyond it."""
    try:
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPolygonF
    except Exception:
        return False
    img = QImage(os.path.join(brief, wide["image"]))
    if img.isNull():
        return False
    img = img.convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    f = _unit(wide["direction"])
    r = _unit(_cross(f, wide["up"]))
    u = _cross(r, f)
    eye = wide["eye"]
    t = math.tan(math.radians(wide["fov_vertical_deg"]) / 2)

    def proj(p):
        v = _sub(p, eye)
        z = _dot(v, f)
        if z <= 1e-6:
            return None
        return QPointF((0.5 + _dot(v, r) / (z * t * wide["aspect"]) / 2) * w, (0.5 - _dot(v, u) / (z * t) / 2) * h)

    k = {"x": 0, "y": 1, "z": 2}.get(str(up_axis).lower()[-1:], 1)
    ce, ct = close["eye"], close["target"]
    flat = list(close["direction"])
    flat[k] = 0.0
    if math.sqrt(_dot(flat, flat)) < 1e-6:      # looking straight down: no direction on the ground
        flat = None
    pts = [ce]
    if flat:
        flat = _unit(flat)
        upv = [0.0, 0.0, 0.0]
        upv[k] = 1.0
        side = _cross(upv, flat)
        far = math.sqrt(_dot(_sub(ct, ce), _sub(ct, ce))) * 1.25
        half = math.atan(math.tan(math.radians(close["fov_vertical_deg"]) / 2) * close["aspect"])
        base = list(ce)
        base[k] = ct[k]
        for i in range(9):
            a = -half + 2 * half * i / 8
            pts.append(_add(base, _mul(_add(_mul(flat, math.cos(a)), _mul(side, math.sin(a))), far)))
    poly = [proj(p) for p in pts]
    e = poly[0]
    if e is None or not (-0.2 * w <= e.x() <= 1.2 * w and -0.2 * h <= e.y() <= 1.2 * h):
        return False                              # the close camera is not in the wide picture
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    if len(poly) > 1 and all(poly):
        p.setPen(QPen(QColor(255, 110, 0), 3))
        p.setBrush(QBrush(QColor(255, 110, 0, 70)))
        p.drawPolygon(QPolygonF(poly))
    p.setPen(QPen(QColor(255, 255, 255), 2))
    p.setBrush(QBrush(QColor(255, 110, 0)))
    p.drawEllipse(e, 7, 7)
    p.end()
    return img.save(out)


# ---- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brief", default=os.path.dirname(HERE))
    ap.add_argument("--out", default=HERE, help="folder with look.txt and shots.md; prompts/, inputs/, STEPS.md go here")
    a = ap.parse_args()
    brief, out = os.path.abspath(a.brief), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    try:
        with open(os.path.join(brief, "brief.json"), encoding="utf-8") as f:
            d = json.load(f)
    except OSError:
        stop(f"No brief.json in {brief}. Run this in the brief folder, or pass --brief.")
    cams = [c for c in d.get("cameras", []) if not c.get("default")]
    if not cams:
        stop("This brief has no hero cameras. Set them in Mule (screenshots on the note board, tick 'hero') and export again.")

    look_path, shots_path = os.path.join(out, "look.txt"), os.path.join(out, "shots.md")
    todo = []
    if not os.path.exists(look_path):
        with open(look_path, "w", encoding="utf-8") as f:
            f.write(LOOK_TODO)
    look = re.sub(r"(?i)^look\s*:\s*", "", " ".join(open(look_path, encoding="utf-8").read().split()))
    if not look or look.startswith("TODO"):
        todo.append(f"- write the look paragraph into {look_path}")
    if not os.path.exists(shots_path):
        with open(shots_path, "w", encoding="utf-8") as f:
            f.write(SHOTS_HEAD + "legend: \nscale: \n\n<!-- Hero cameras in this brief - copy the ones that take part "
                    "out of this comment and fill them in:\n\n"
                    + "\n".join(f"## {c['name']}\n{c.get('prompt') or ''}\n" for c in cams) + "-->\n")
    scene, sections = parse_shots(open(shots_path, encoding="utf-8").read())

    by = {}
    for c in cams:
        by[c["name"].lower()] = by[c["id"].lower()] = c

    def find(name, what):
        c = by.get(name.strip("` ").lower())
        if c is None:
            stop(f"shots.md: {what} '{name}' is not a hero camera of this brief. Cameras: "
                 + ", ".join(c["name"] for c in cams))
        return c

    shots, seen = [], set()                       # (camera, fields), in the artist's order
    for name, fields in sections:
        c = find(name, "camera")
        if c["id"] not in seen:
            seen.add(c["id"])
            shots.append((c, fields))
    for c in cams:                                # prompt field in Mule, where shots.md has no section
        if c["id"] not in seen and (c.get("prompt") or "").strip():
            seen.add(c["id"])
            shots.append((c, parse_fields(c["prompt"])))
    if not shots:
        todo.append(f"- say which cameras take part: sections in {shots_path} (the format is at its top),\n"
                    "  or text in the prompt field of the hero cameras in Mule and a new export")
    if todo:
        stop("Not built yet:\n" + "\n".join(todo))

    fields = {c["id"]: f for c, f in shots}
    closes = [(c, f) for c, f in shots if f.get("pair")]
    wide_ids = []
    for c, f in closes:
        w = find(f["pair"], f"pair of {c['name']}:")
        if w["id"] == c["id"]:
            stop(f"shots.md: {c['name']} names itself as its pair.")
        if w["id"] not in fields:
            fields[w["id"]] = parse_fields(w.get("prompt") or "")
            shots.append((w, fields[w["id"]]))
        if w["id"] not in wide_ids:
            wide_ids.append(w["id"])
    rest = [c for c, f in shots if not f.get("pair") and c["id"] not in wide_ids]
    if not wide_ids and rest:                     # no pairs at all: the first camera is the wide view
        wide_ids.append(rest.pop(0)["id"])
    cam = {c["id"]: c for c in cams}
    wides = [cam[i] for i in wide_ids]
    for c, f in shots:
        if not f["view"]:
            print(f"warning: {c['name']} has no text about what is in the view - the prompt goes without it.")

    for sub in ("prompts", "inputs"):
        shutil.rmtree(os.path.join(out, sub), ignore_errors=True)
        os.makedirs(os.path.join(out, sub))
    os.makedirs(os.path.join(out, "approved"), exist_ok=True)
    inside = os.path.normcase(out).startswith(os.path.normcase(brief) + os.sep)
    rel = os.path.relpath(out, brief).replace(os.sep, "/") if inside else "."     # paths as seen from the brief folder
    gen = "gen"

    def structure(c):
        dst = f"inputs/{slug(c)}.png"
        if not os.path.exists(os.path.join(out, dst)):
            shutil.copyfile(os.path.join(brief, c["image"]), os.path.join(out, dst))
        return f"{rel}/{dst}"

    legend = scene.get("legend", "")
    legend = (": " + legend.rstrip(".")) if legend else ""

    def text(c, opening, middle):
        f = fields[c["id"]]
        bind = part("bind").replace("{format}", fmt(c.get("aspect") or 1.0)).replace("{legend}", legend)
        if f["note"]:
            bind += "\n" + "\n".join(f["note"])
        scale = " ".join(s for s in (f.get("scale"), scene.get("scale")) if s)
        view = "\n".join(s for s in (f["view"] and "In this view: " + f["view"], scale and "Scale: " + scale) if s)
        return "\n\n".join(s for s in [opening, bind] + middle + [view, "Look: " + look, part("end")] if s) + "\n"

    rows, n = [], 0

    def write(c, name, body, attach, then):
        path = f"prompts/{n:02d}_{name}.txt"
        with open(os.path.join(out, path), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        rows.append((n, f"{rel}/{path}", attach, f"{gen}/{n:02d}_{name}_a.png", then))

    if wides:
        w = wides[0]
        write(w, "mood_" + slug(w), text(w, part("open_mood"), []), [structure(w)],
              "look at colour and light only; change `look.txt` if needed. Never a target, never an input.")
    shutil.copyfile(os.path.join(HERE, "parts", "chat_context.txt"), os.path.join(out, "prompts", "00_chat_context.txt"))

    approved = lambda c: f"{rel}/approved/{slug(c)}.png"
    no_where = []
    for c, f in closes:
        n += 1
        w = by[f["pair"].strip("` ").lower()]
        attach, middle = [structure(c)], []
        wh = f"inputs/{slug(c)}_where.png"
        if location_picture(brief, c, w, d.get("up_axis", "Y"), os.path.join(out, wh)):
            attach.append(f"{rel}/{wh}")
            middle.append(part("where"))
        else:
            no_where.append(c["name"])
        middle.append(part("close").replace("{decides}", f.get("decides") or "how the things in this view are built"))
        write(c, slug(c), text(c, part("open"), middle), attach, f"artist picks one → `{approved(c)}`")
    for w in wides:
        n += 1
        mine = [(c, f) for c, f in closes if by[f["pair"].strip("` ").lower()]["id"] == w["id"]]
        if len(mine) > 2:
            print(f"warning: {w['name']} has {len(mine)} close views; only the first two are attached (three inputs at most).")
            mine = mine[:2]
        attach, middle = [structure(w)], []
        if mine:
            refs = [part("wide_ref").replace("{n}", str(i + 2)).replace("{name}", c["name"])
                    .replace("{decides}", f.get("decides") or "how things of its kind are built")
                    for i, (c, f) in enumerate(mine)]
            middle.append(" ".join(refs) + "\n" + part("wide"))
            attach += [approved(c) for c, f in mine]
        write(w, slug(w), text(w, part("open"), middle), attach, f"artist picks one → `{approved(w)}`")
    for c in rest:
        n += 1
        f = fields[c["id"]]
        attach, middle = [structure(c), approved(wides[0])], [part("further")]
        like = find(f["like"], f"'like' of {c['name']}:") if f.get("like") else (closes[0][0] if closes else None)
        if like is not None:
            lf = fields.get(like["id"]) or {}
            middle[0] += " " + (part("further_close").replace("{name}", like["name"])
                                .replace("{decides}", lf.get("decides") or "how things of its kind are built"))
            attach.append(approved(like))
        write(c, slug(c), text(c, part("open"), middle), attach, f"artist picks one → `{approved(c)}`")

    with open(os.path.join(out, "STEPS.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("# Steps\nWritten by build_prompts.py - do not edit; change `look.txt` or `shots.md` and run it again.\n\n"
                 f"By hand in a chat: paste `{rel}/prompts/00_chat_context.txt` once and wait for \"ready\". Then per step: "
                 "attach the pictures in the order given, paste the whole prompt file, save the result under the name given. "
                 "For a variant: \"Another variant of the same request.\" (save as `_b`, `_c`).\n\n"
                 "| Step | Prompt | Attach, in this order | Save as | Then |\n|---|---|---|---|---|\n")
        for num, prompt, attach, save, then in rows:
            fh.write(f"| {num} | `{prompt}` | {', '.join('`%s`' % x for x in attach)} | `{save}` | {then} |\n")
        fh.write("\nClose views do not depend on each other. A wide view starts only when its close views are approved; "
                 "further views only when the first wide view is approved.\n\n"
                 "Before you approve: lay the structure image over the result – is every large outline still where it was? "
                 "Nothing added, removed or merged? Close and wide: the same place on the same day?\n")
    print(f"{len(rows)} steps written: {os.path.join(out, 'STEPS.md')}")
    if no_where:
        print("no location picture for: " + ", ".join(no_where)
              + " (PySide6 missing, or the close camera lies outside its wide view) - their prompts do without it.")


if __name__ == "__main__":
    main()
