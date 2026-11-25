"""Brief recipes (2026-11-16): the whole working method of a build brief – rules, roles, hooks, agent types, tools, the
process part of project.md and the sentence for the chat – is a recipe that can be swapped. Mule's own method is the
built-in recipe "default" (sketch_machine/brief_package/); recipes of the user or the community live in the recipe
folder (paths.user_dir()/recipes, or MULE_RECIPES) and are chosen per export in the brief dialog.

The seam: Mule writes the content (brief.md, brief.json, scene.usd, views/, notes/) and offers its facts to the
recipe's templates; everything about how the AI works comes from the recipe. Format: docs/recipes.md.

A recipe is a folder with recipe.toml:
- [files]: recipe path → "brief/…" (brief folder) or "work/…" (work folder next to it); a trailing "/" copies a folder.
  In copied text files the tokens {{project_name}}, {{brief_dir}}, {{work_dir}}, {{bash}}, {{powershell}},
  {{recipe_id}}, {{recipe_version}} are filled in; nothing else is touched.
- [templates]: Mustache templates Mule fills with its facts (project, brief_head, start, review_start, review_intro,
  package_update); "start.md" may have language variants "start.de.md".
- [config]: a key=value file written fresh for a new brief and only extended in a running run (keys of the artist stay).
The brief folder gets a lock file (LOCK_FILE) naming recipe, version and digest; review rounds renew from the same
recipe, never from another one."""
import hashlib
import json
import os
import re
import shutil
import tempfile
import tomllib
import zipfile

from . import paths

BRIEF_INTERFACE = 1  # version of what Mule offers recipes (facts, tokens, folders) – docs/recipes.md
DEFAULT_ID = "default"
BUILTIN_DIR = os.path.join(paths.ROOT, "sketch_machine", "brief_package")
SHIPPED_DIR = os.path.join(paths.ROOT, "sketch_machine", "brief_recipes")  # further recipes that come with Mule
RECIPE_FILE = "recipe.toml"
LOCK_FILE = ".mule_recipe.json"
TEMPLATES = ("project", "brief_head", "start", "review_start", "review_intro", "package_update", "reading", "look",
             "open_points", "context_head")
TOKENS = ("project_name", "brief_dir", "work_dir", "bash", "powershell", "recipe_id", "recipe_version")
HARNESSES = ("claude-code", "any")
# what Mule writes into the brief folder itself – a recipe may not put files there (compared without case: Windows)
MULE_FILES = {"brief.md", "brief.json", "project.md", "context.md", "core.md", "scene.usd", "review.md",
              "clarified_history.md", ".mcp.json", ".codex", "views", "notes", "shots", "videos", "gen",
              "review_rounds", ".mule_recipe.json"}
_SKIP = {"__pycache__", ".git", ".DS_Store", "Thumbs.db"}
_TEXT = (".md", ".json", ".sh", ".ps1", ".psm1", ".conf", ".txt", ".toml", ".py", ".yml", ".yaml", ".cmd", ".bat",
         ".js", ".ts", ".ini", ".cfg")
_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
MAX_ZIP = 50 * 1024 * 1024  # unpacked size of an imported recipe


class RecipeError(Exception):
    pass


class Recipe:
    """A loaded recipe; problems lists what makes it unusable (then it is shown, but cannot be chosen). Loading never
    raises: a recipe being edited (by hand or by an AI) may be half done, and must not block the brief dialog."""

    def __init__(self, path, builtin=False):
        self.path = os.path.abspath(path)
        self.builtin = builtin
        self.shipped = False  # comes with Mule, but is not the default (brief_recipes/)
        self.problems = []
        self.meta = {}
        self.files = {}
        self.templates = {}
        self.config_file = None
        self.config_values = {}
        self.keep = []
        try:
            with open(os.path.join(self.path, RECIPE_FILE), "rb") as fh:
                self.meta = tomllib.load(fh)
        except FileNotFoundError:
            self.problems.append(f"no {RECIPE_FILE}")
        except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
            self.problems.append(f"{RECIPE_FILE} cannot be read: {exc}")
        m = self.meta
        text = lambda k, d="": str(m.get(k) or d)  # noqa: E731
        self.id = text("id", os.path.basename(self.path))
        self.name = text("name", self.id)
        self.version = text("version", "v0")
        self.author = text("author")
        self.description = text("description")
        mf = m.get("made_for")
        self.made_for = [str(x) for x in mf] if isinstance(mf, list) else ([str(mf)] if mf else [])
        self.size = text("size")
        self.harness = text("harness", "claude-code")
        self.interface = m.get("brief_interface", BRIEF_INTERFACE)
        self.forked_from = text("forked_from")
        self.running_marker = m.get("running_marker") or ""
        self.changes = m.get("changes") or ""
        if not self.meta:
            return
        try:
            self._validate(m)
        except Exception as exc:  # noqa: BLE001 – whatever is wrong, the recipe is shown as unusable, never a crash
            self.problems.append(f"{RECIPE_FILE} cannot be used: {type(exc).__name__}: {exc}")

    def _validate(self, m):
        if not valid_id(self.id):
            self.problems.append(f"id „{self.id}“: lower-case letters, digits, '-', '_', '.' (not at the end), "
                                 "no Windows device name")
        if version_key(self.version) is None:
            self.problems.append(f"version „{self.version}“: write it as v<major>.<minor>, e.g. v1.0")
        if not isinstance(self.interface, int) or self.interface > BRIEF_INTERFACE:
            self.problems.append(f"needs brief interface {self.interface}, this Mule offers {BRIEF_INTERFACE} – "
                                 "update Mule")
        if self.harness not in HARNESSES:
            self.problems.append(f"harness „{self.harness}“ unknown (known: {', '.join(HARNESSES)})")
        for key in ("files", "templates", "config"):
            if not isinstance(m.get(key, {}), dict):
                self.problems.append(f"[{key}] must be a table")
                return
        for src, dst in m.get("files", {}).items():
            target = clean_target(dst)
            if not (self._inside(src) and target):
                self.problems.append(f"files: „{src}“ → „{dst}“ leaves its folder (targets start with brief/ or work/, "
                                     "forward slashes, no '.', '..', ':' or names ending in a dot or space)")
                continue
            full = os.path.join(self.path, src)
            if src.endswith("/") != target.endswith("/"):
                self.problems.append(f"files: „{src}“ → „{dst}“: folder to folder, file to file")
            elif not (os.path.isdir(full) if src.endswith("/") else os.path.isfile(full)):
                self.problems.append(f"files: „{src}“ is missing")
            else:
                self.files[src] = target
        for key, rel in m.get("templates", {}).items():
            if key not in TEMPLATES:
                self.problems.append(f"templates: „{key}“ unknown (known: {', '.join(TEMPLATES)})")
            elif not (self._inside(rel) and os.path.isfile(os.path.join(self.path, rel))):
                self.problems.append(f"templates: „{rel}“ is missing")
            else:
                self.templates[key] = rel
        if "start" not in self.templates:
            self.problems.append("templates: „start“ (the sentence for the chat) is required")
        conf = m.get("config", {})
        if conf:
            target = clean_target(conf.get("file", ""))
            if not target or target.endswith("/"):
                self.problems.append("config: „file“ must be a file under brief/ or work/")
            else:
                self.config_file = target
                values = conf.get("values", {})
                if not isinstance(values, dict):
                    self.problems.append("[config.values] must be a table")
                else:
                    self.config_values = {str(k): str(v) for k, v in values.items()}
        keep = [*(conf.get("keep") or ()), *(m.get("keep") or ())]
        for k in keep if isinstance(keep, list) else ():
            target = clean_target(k)
            if target:
                self.keep.append(target)
            else:
                self.problems.append(f"keep: „{k}“ must be under brief/ or work/")
        if self.running_marker:
            target = clean_target(self.running_marker)
            if not target:
                self.problems.append("running_marker must be under brief/ or work/")
            self.running_marker = target or ""
        taken = sorted({t.split("/")[1] for t in [*(t for _s, t in plan(self)), self.config_file or ""]
                        if t.startswith("brief/") and t.split("/")[1].lower() in MULE_FILES})
        if taken:
            self.problems.append("files: " + ", ".join(taken) + " in the brief folder belong to Mule's content")
        if self.changes and not (self._inside(self.changes) and os.path.isfile(os.path.join(self.path, self.changes))):
            self.problems.append(f"changes: „{self.changes}“ is missing")
        check_templates(self)

    def _inside(self, rel):
        if not isinstance(rel, str) or not rel or "\\" in rel or ":" in rel or rel.startswith("/"):
            return False
        full = os.path.normpath(os.path.join(self.path, rel))
        return full == self.path or full.startswith(self.path + os.sep)

    @property
    def ok(self):
        return not self.problems

    def label(self):
        return f"{self.name} ({self.version})"

    def template(self, key, lang=None):
        """Text of a template (language variant name.<lang>.ext first) or None. UTF-8, with or without BOM."""
        rel = self.templates.get(key)
        if not rel:
            return None
        cands = []
        if lang:
            stem, ext = os.path.splitext(rel)
            cands.append(f"{stem}.{lang}{ext}")
        cands.append(rel)
        for c in cands:
            p = os.path.join(self.path, c)
            if os.path.isfile(p):
                with open(p, encoding="utf-8-sig") as fh:
                    return fh.read()
        return None

    def render(self, key, ctx, lang=None):
        """Filled template (None if the recipe has none)."""
        t = self.template(key, lang)
        return None if t is None else render(t, ctx)

    def changes_text(self):
        if not self.changes:
            return ""
        try:
            with open(os.path.join(self.path, self.changes), encoding="utf-8-sig") as fh:
                return fh.read()
        except (OSError, UnicodeDecodeError):
            return ""


class Missing:
    """The recipe a brief was made with, when it is no longer installed (or a brief without one): no templates – Mule
    writes its short neutral texts and never another recipe's process words."""
    builtin = False
    templates = {}

    def __init__(self, lock=None):
        lock = lock or {}
        self.id = lock.get("id") or ""
        self.name = lock.get("name") or self.id or "no recipe"
        self.version = lock.get("version") or ""

    def label(self):
        return f"{self.name} ({self.version})" if self.version else self.name

    def render(self, key, ctx, lang=None):
        return None

    def changes_text(self):
        return ""


_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
_PART = re.compile(r"^[^<>:\"|?*\\/\x00-\x1f]+$")


def _part_ok(part):
    """One path component that means the same on every system: no '.', '..', no trailing dot or space (Windows drops
    them), no device names, no characters Windows forbids."""
    return (bool(_PART.match(part)) and part not in (".", "..") and not part.endswith((".", " "))
            and part.split(".")[0].lower() not in _RESERVED
            and not re.search(r"~\d", part))  # 8.3 short names ("BRIEF~1.JSO") would name another file


def clean_target(t):
    """A recipe target as Mule uses it ("brief/…" or "work/…", forward slashes, a trailing "/" for a folder), or None
    when it could point anywhere else – checked part by part, so that "brief/\\Users", "brief/./brief.md",
    "brief/brief.md." and "brief/x/../.." are all refused."""
    if not isinstance(t, str) or not t.startswith(("brief/", "work/")) or "\\" in t:
        return None
    root, rest = t.split("/", 1)
    folder = rest.endswith("/")
    parts = rest.rstrip("/").split("/") if rest.rstrip("/") else []
    if not parts or not all(_part_ok(p) for p in parts):
        return None
    return root + "/" + "/".join(parts) + ("/" if folder else "")


def valid_id(rid):
    return bool(_ID.match(rid)) and _part_ok(rid) and not rid.endswith((".mule_new", ".mule_old"))


def version_key(v):
    """v1.2.3 → (1, 2, 3), padded so that v1 == v1.0; a heading may go on after the number ("## v1.1 (2026-11-16)").
    None if there is no version number."""
    mt = re.match(r"\s*v?(\d+(?:\.\d+)*)", str(v or ""))
    if not mt:
        return None
    nums = [int(x) for x in mt.group(1).split(".")]
    return tuple(nums + [0] * (4 - len(nums)))[:max(4, len(nums))]


# ---- folders and lookup

def user_dir(create=True):
    """Recipe folder of the user (MULE_RECIPES overrides – tests); created on demand, not by just listing."""
    path = os.environ.get("MULE_RECIPES") or os.path.join(paths.user_dir(create), "recipes")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def default():
    return Recipe(BUILTIN_DIR, builtin=True)


def available():
    """Built-in recipe first, then the user's (by name); recipes whose id is taken by an earlier one are marked."""
    out = [default()]
    try:  # recipes that come with Mule (not the default): chosen like the user's, an own recipe cannot take their id
        shipped = sorted(os.listdir(SHIPPED_DIR))
    except OSError:
        shipped = []
    for n in shipped:
        p = os.path.join(SHIPPED_DIR, n)
        if os.path.isfile(os.path.join(p, RECIPE_FILE)):
            r = Recipe(p)
            r.shipped = True
            out.append(r)
    try:
        root = user_dir(create=False)
        names = sorted(os.listdir(root))
    except OSError:
        names = []
    for n in names:
        p = os.path.join(root, n)
        if os.path.isdir(p) and n not in _SKIP and not n.startswith(".") and not n.endswith((".mule_new", ".mule_old")):
            r = Recipe(p)
            if any(o.id == r.id for o in out):
                r.problems.append(f"id „{r.id}“ is already taken by another recipe")
            out.append(r)
    return [out[0]] + sorted(out[1:], key=lambda r: r.name.lower())


def find(rid):
    """Usable recipe with this id or None."""
    rid = rid or DEFAULT_ID
    if rid == DEFAULT_ID:
        return default()
    return next((r for r in available() if r.id == rid and r.ok), None)


def get(rid):
    """Recipe with this id, the default if it is missing or broken."""
    return find(rid) or default()


# ---- templates (Mustache subset)

_TAG = re.compile(r"\{\{\s*([#^/]?)\s*([\w.]+)\s*\}\}")
_STANDALONE = re.compile(r"(?m)^[ \t]*(\{\{\s*[#^/]\s*[\w.]+\s*\}\})[ \t]*\r?\n")


def render(template, ctx):
    """Mustache subset: {{name}}, {{a.b}}, {{.}}, sections {{#x}}…{{/x}} (list: once per item, item fields visible,
    plus _first/_last; other truthy value: once) and inverted sections {{^x}}…{{/x}}. No HTML escaping (Markdown).
    A line holding only a section tag disappears with its line break. Unknown names give an empty text."""
    tree, rest = _parse(_STANDALONE.sub(r"\1", template), 0, None)
    return _emit(tree, [ctx])


def _parse(text, pos, closing):
    out = []
    while True:
        m = _TAG.search(text, pos)
        if not m:
            if closing:
                raise RecipeError(f"template: {{{{/{closing}}}}} is missing")
            out.append(text[pos:])
            return out, len(text)
        out.append(text[pos:m.start()])
        kind, name = m.group(1), m.group(2)
        if kind in ("#", "^"):
            body, pos = _parse(text, m.end(), name)
            out.append((kind, name, body))
        elif kind == "/":
            if name != closing:
                raise RecipeError(f"template: {{{{/{name}}}}} without its opening tag")
            return out, m.end()
        else:
            out.append(("", name, None))
            pos = m.end()


def _lookup(stack, name):
    if name == ".":
        return stack[-1]
    first, *more = name.split(".")
    for frame in reversed(stack):
        if isinstance(frame, dict) and first in frame:
            v = frame[first]
            break
    else:
        return None
    for part in more:
        v = v.get(part) if isinstance(v, dict) else None
    return v


def _falsy(v):
    return v is None or v is False or v == "" or v == [] or v == {} or (type(v) is int and v == 0)


def _emit(tree, stack):
    out = []
    for node in tree:
        if isinstance(node, str):
            out.append(node)
            continue
        kind, name, body = node
        v = _lookup(stack, name)
        if kind == "":
            out.append("" if v is None or v is False else str(v))
        elif kind == "^":
            if _falsy(v):
                out.append(_emit(body, stack))
        elif not _falsy(v):
            if isinstance(v, list):
                for i, item in enumerate(v):
                    out.append(_emit(body, stack + [{"_first": i == 0, "_last": i == len(v) - 1}, item]))
            else:
                out.append(_emit(body, stack + [v]))
    return "".join(out)


def check_templates(recipe):
    """Template syntax errors as problems (so a broken recipe cannot be chosen)."""
    for key in recipe.templates:
        for lang in (None, "de", "en"):
            t = recipe.template(key, lang)
            if t is None:
                continue
            try:
                _parse(_STANDALONE.sub(r"\1", t), 0, None)
            except RecipeError as exc:
                recipe.problems.append(f"{recipe.templates[key]}: {exc}")
                break
    return recipe


# ---- installing into a brief

def _resolve(target, brief, work):
    """Path of a clean target – checked once more on the real path (links, junctions): it must lie inside the brief or
    the work folder, otherwise RecipeError and nothing is written."""
    target = clean_target(target)
    if not target:
        raise RecipeError("a recipe target leaves the brief and work folders")
    root, rest = target.split("/", 1)
    base = brief if root == "brief" else work
    dst = os.path.join(base, *rest.rstrip("/").split("/"))
    real, real_base = os.path.realpath(dst), os.path.realpath(base)
    if os.path.normcase(real) != os.path.normcase(real_base) and \
            not os.path.normcase(real).startswith(os.path.normcase(real_base) + os.sep):
        raise RecipeError(f"„{target}“ would end up outside the {root} folder")
    return dst


def _fill_bytes(raw, tokens):
    if b"{{" not in raw:
        return raw
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw
    for k, v in tokens.items():
        text = text.replace("{{" + k + "}}", v)
    return text.encode("utf-8")


def _walk(src):
    """Files of a folder, relative, without caches and version-control folders."""
    for base, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP)
        for f in sorted(files):
            if f in _SKIP or f.endswith(".pyc"):
                continue
            yield os.path.relpath(os.path.join(base, f), src)


def plan(recipe):
    """[(source file, target)] the recipe installs – folders expanded."""
    out = []
    for src, dst in recipe.files.items():
        if src.endswith("/"):
            for rel in _walk(os.path.join(recipe.path, src)):
                out.append((os.path.join(recipe.path, src, rel), dst + rel.replace(os.sep, "/")))
        else:
            out.append((os.path.join(recipe.path, src), dst))
    return out


def _targets(recipe):
    """Every target the recipe writes (files and its config file)."""
    return [t for _s, t in plan(recipe)] + ([recipe.config_file] if recipe.config_file else [])


def _protected(dst, brief, keep_real):
    """Never removed by a recipe switch, whatever a lock file says (it lies in the brief folder, where an AI or a
    download could change it): Mule's content – judged by the real name, so 8.3 short names and other spellings do not
    get past –, keep files, links."""
    real = os.path.realpath(dst)
    rel = os.path.relpath(real, os.path.realpath(brief))
    first = rel.split(os.sep)[0].casefold()
    return (rel.startswith("..") or first in MULE_FILES or os.path.normcase(real) in keep_real
            or os.path.islink(dst) or not os.path.isfile(dst))


def _remove_previous(brief, work, recipe, keep_targets):
    """What the recipe that was in this brief before installed and `recipe` does not write – removed so that a brief
    never holds two ways of working (another recipe exported into the same folder, or a newer version that drops a
    file). Deliberately narrow: only in the brief folder (the work folder holds the run's results and may be shared by
    two briefs), never while a run is in progress, never Mule's content, keep files, the run marker or links. Taken
    from the lock's file list; locks without one: from that recipe's plan. -> removed targets."""
    lock = read_lock(brief) or {}
    prev = find(lock.get("id")) if lock.get("id") else None
    for r in (prev, recipe):
        try:
            if r is not None and is_running(r, brief, work):
                return []
        except RecipeError:
            return []
    old = lock.get("files")
    if not isinstance(old, list):
        old = _targets(prev) if prev else []
    keep_real = set()
    for t in [*keep_targets, *(prev.keep if prev else []),
              *([prev.running_marker] if prev and prev.running_marker else []),
              *([recipe.running_marker] if recipe.running_marker else [])]:
        try:
            keep_real.add(os.path.normcase(os.path.realpath(_resolve(t, brief, work))))
        except RecipeError:
            pass
    new = {t.casefold() for t in keep_targets}
    removed = []
    for t in old:
        t = clean_target(t) if isinstance(t, str) else None
        if not t or not t.startswith("brief/") or t.endswith("/") or t.casefold() in new:
            continue
        try:
            dst = _resolve(t, brief, work)
        except RecipeError:
            continue
        if _protected(dst, brief, keep_real):
            continue
        os.remove(dst)
        removed.append(t)
        base = os.path.realpath(brief)
        d = os.path.dirname(dst)
        while os.path.realpath(d) != base and os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
            d = os.path.dirname(d)
    return removed


def install(recipe, brief, work, tokens, fresh_conf=True):
    """Copy the recipe into brief folder and work folder, write the config file and the lock file. In a running run
    (fresh_conf False) the config keeps its values and only gets new keys; files under keep are never overwritten.
    What an earlier recipe of this brief installed and this one does not is removed first. Every target is checked
    against the real path before it is written. Returns the written files relative to the brief folder (work-folder
    files are not listed)."""
    targets = _targets(recipe)
    resolved = {t: _resolve(t, brief, work) for t in targets}  # every target checked before anything is removed or written
    _remove_previous(brief, work, recipe, set(targets) | set(recipe.keep))
    written = []
    skip = {recipe.config_file} if recipe.config_file else set()
    for src, target in plan(recipe):
        if target in skip:
            continue
        dst = resolved[target]
        if target in recipe.keep and os.path.exists(dst):
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as fh:
            raw = fh.read()
        if os.path.splitext(src)[1].lower() in _TEXT:
            raw = _fill_bytes(raw, tokens)
        with open(dst, "wb") as fh:
            fh.write(raw)
        if target.startswith("brief/"):
            written.append(target[6:])
    if recipe.config_file:
        conf_path = resolved[recipe.config_file]
        values = {k: _fill_bytes(v.encode("utf-8"), tokens).decode("utf-8") for k, v in recipe.config_values.items()}
        if fresh_conf or not os.path.isfile(conf_path):
            os.makedirs(os.path.dirname(conf_path), exist_ok=True)
            with open(conf_path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write("".join(f"{k}={v}\n" for k, v in values.items()))
            if recipe.config_file.startswith("brief/"):
                written.append(recipe.config_file[6:])
        else:
            with open(conf_path, encoding="utf-8-sig") as fh:
                old = fh.read()
            have = {line.split("=", 1)[0].strip() for line in old.splitlines() if "=" in line}
            missing = [k for k in values if k not in have]
            if missing:
                with open(conf_path, "a", encoding="utf-8", newline="\n") as fh:
                    # a file without a final line break would glue the new key onto the run's last value
                    fh.write(("\n" if old and not old.endswith("\n") else "")
                             + "".join(f"{k}={values[k]}\n" for k in missing))
                if recipe.config_file.startswith("brief/"):
                    written.append(recipe.config_file[6:])
    write_lock(brief, recipe, targets)
    written.append(LOCK_FILE)
    return written


def is_running(recipe, brief, work):
    return bool(recipe.running_marker) and os.path.exists(_resolve(recipe.running_marker, brief, work))


def digest(recipe):
    """sha256 over what makes the recipe (recipe.toml, the files it installs, its templates in every language, its
    changes file; path and content) – names exactly which recipe a result came from. Other files in its folder
    (tests, their runs, notes) do not count, so the same recipe has the same digest on every machine."""
    rels = {RECIPE_FILE, *(os.path.relpath(src, recipe.path) for src, _t in plan(recipe))}
    if recipe.changes:
        rels.add(recipe.changes)
    for rel in recipe.templates.values():
        stem, ext = os.path.splitext(rel)
        folder = os.path.dirname(os.path.join(recipe.path, rel))
        rels.add(rel)
        if os.path.isdir(folder):
            base = os.path.basename(stem)
            rels.update(os.path.relpath(os.path.join(folder, f), recipe.path) for f in os.listdir(folder)
                        if f.startswith(base + ".") and f.endswith(ext))
    h = hashlib.sha256()
    for rel in sorted(r.replace(os.sep, "/") for r in rels):
        p = os.path.join(recipe.path, *rel.split("/"))
        if not os.path.isfile(p):
            continue
        h.update(rel.encode("utf-8") + b"\0")
        with open(p, "rb") as fh:
            h.update(fh.read())
        h.update(b"\0")
    return h.hexdigest()


def write_lock(brief, recipe, targets=None):
    """Lock of a brief: which recipe, which version, its digest – and the files it installed, so that another recipe
    or a newer version can take exactly these away again."""
    with open(os.path.join(brief, LOCK_FILE), "w", encoding="utf-8") as fh:
        json.dump({"id": recipe.id, "name": recipe.name, "version": recipe.version, "author": recipe.author,
                   "builtin": recipe.builtin, "digest": digest(recipe),
                   "files": sorted(targets if targets is not None else _targets(recipe))},
                  fh, indent=1, ensure_ascii=False)
        fh.write("\n")


def read_lock(brief):
    """Lock of a brief folder; briefs from before recipes (only .claude/package_version) count as the default."""
    try:
        with open(os.path.join(brief, LOCK_FILE), encoding="utf-8") as fh:
            lock = json.load(fh)
        return lock if isinstance(lock, dict) and lock.get("id") else None
    except (OSError, ValueError):
        pass
    pv = os.path.join(brief, ".claude", "package_version")
    if os.path.isfile(pv):
        with open(pv, encoding="utf-8") as fh:
            return {"id": DEFAULT_ID, "version": fh.read().strip() or "unknown", "builtin": True}
    return None


# ---- making a recipe: an empty one to start from, or a copy of an existing one

FORMAT_FILE = "FORMAT.md"  # docs/recipes.md, put into every new recipe so the AI that writes it has the format
START_FILE = "START_HERE.md"
# files of Mule's own development that a copy does not need (hook tests and their runs, German notes)
_COPY_SKIP = ("tests", "_runs", "LIESMICH.md", "AENDERUNGEN_HOOKS.md", "PILOT_README_de.md")

_EMPTY = {
    "recipe.toml": '''# Brief recipe for Mule – the format is in FORMAT.md, how to start in START_HERE.md
id = "{id}"
name = "{name}"
version = "v0.1"
author = ""
description = ""
made_for = []
size = ""
harness = "claude-code"
brief_interface = 1
forked_from = ""
changes = "CHANGES.md"

[templates]
start = "templates/start.md"
project = "templates/project.md"

[files]
"CLAUDE.md" = "brief/CLAUDE.md"
''',
    "CLAUDE.md": '''# Sketch Brief – {{{{project_name}}}}

This folder is a Sketch Brief from Mule, a 3D sketching tool. `project.md` holds the task and the round, `brief.md`
the artist's notes and the scene, `brief.json` the same as data, `views/` and `notes/` the pictures.

(How the AI works with it – this recipe's way of working – goes here or into files this one includes with `@file`.)
''',
    "templates/start.md": "Check and discuss the Sketch Brief „{{{{project_name}}}}“ ({{{{brief_dir}}}}): read CLAUDE.md, "
                          "project.md and brief.md and open the pictures they name, then tell me what you understood "
                          "and ask what is open. Build nothing until I confirm.{{{{#look}}}} Before building: agree an "
                          "image prompt per hero camera with me, have images generated, I pick.{{{{/look}}}}\n",
    "templates/start.de.md": "Prüfe und besprich den Sketch Brief „{{{{project_name}}}}“ ({{{{brief_dir}}}}): Lies "
                             "CLAUDE.md, project.md und brief.md und öffne die Bilder, die sie nennen; sag mir dann, "
                             "was du verstanden hast, und frag, was offen ist. Gebaut wird erst nach meiner "
                             "Bestätigung.{{{{#look}}}} Vor dem Bauen: je Hero-Kamera einen Bild-Prompt mit mir "
                             "abstimmen, Bilder erzeugen lassen, ich wähle aus.{{{{/look}}}}\n",
    "templates/project.md": '''# Project – {{{{project_name}}}}

## Task
{{{{#task_sentence}}}}
**In one sentence:** {{{{task_sentence}}}}
{{{{/task_sentence}}}}
{{{{^task_sentence}}}}
The artist set no task sentence – work out the goal from the notes and confirm it with the artist.
{{{{/task_sentence}}}}
{{{{#task_notes_quoted}}}}

**Artist's notes for this export** (they win over a note in the scene):
{{{{task_notes_quoted}}}}
{{{{/task_notes_quoted}}}}

## Parts this round is about
{{{{#round_map}}}}
- {{{{word}}}} ({{{{ids}}}}{{{{#group}}}} and everything in it{{{{/group}}}}){{{{#binding}}}} – {{{{binding}}}}{{{{/binding}}}}
{{{{/round_map}}}}
{{{{^round_map}}}}
No focus and no binding set in Mule – the whole scene.
{{{{/round_map}}}}

## Pictures
- hero cameras: {{{{#cameras}}}}{{{{name}}}} `{{{{image}}}}`{{{{^_last}}}}, {{{{/_last}}}}{{{{/cameras}}}}
- {{{{boards}}}} whole-board pictures and {{{{board_sections}}}} section pictures in `notes/`
{{{{^live}}}}
- work folder: `{{{{work_dir}}}}`
{{{{/live}}}}
''',
    "CHANGES.md": "# Changes\n## v0.1\n- Started.\n",
    START_FILE: '''# {name} – a brief recipe for Mule

This folder is a **brief recipe**: the way an AI works with a Sketch Brief from Mule. It starts nearly empty – it
already works (the AI reads the brief and discusses it), everything else is up to you.

**Read first:** `FORMAT.md` – what `recipe.toml` may say, where files go (`brief/…`, `work/…`), the tokens Mule
fills in, and the facts every template can use.

**What is here:**
- `recipe.toml` – id, name, version, which files go into the brief, which templates Mule fills.
- `CLAUDE.md` – copied into every brief folder; Claude Code reads it first. The way of working starts here.
- `templates/start.md` (+ `start.de.md`) – the sentence the artist pastes into the chat.
- `templates/project.md` – the project document Mule writes on every export.
- `CHANGES.md` – one section per version; review rounds quote what is new.

**Working on it:** change files here; Mule reads them on the next export (in the brief dialog: Brief recipe →
… → Reload). After every change raise `version` in `recipe.toml` and add a section to `CHANGES.md`. Add rules,
roles, agent types (`.claude/agents/`), hooks (`.claude/settings.json` + scripts) or tools by putting the files
here and listing them under `[files]`. Try it by exporting a brief into a test folder and starting a session there.

**brief.md is content only with this recipe:** the scene, the notes, the pictures. Mule's own reading aid (how to
read notes and marks), its look-image procedure and its "raise with the artist" line come only with recipes that have
the templates `reading`, `look`, `open_points` and `context_head` (see FORMAT.md) – Mule Default has all four, a
good example (Brief recipe → … → Copy as my own recipe).

This file and `FORMAT.md` stay here – they are not copied into briefs.
''',
}


def slug(name):
    """Id from a name: lower case, letters/digits/_ only, never "default"; unique in the recipe folder."""
    base = re.sub(r"[^a-z0-9]+", "_", name.strip().lower().replace("ä", "ae").replace("ö", "oe")
                  .replace("ü", "ue").replace("ß", "ss")).strip("_")[:48] or "recipe"
    if base == DEFAULT_ID:
        base = "my_default"
    rid, k = base, 2
    while os.path.exists(os.path.join(user_dir(), rid)):
        rid, k = f"{base}_{k}", k + 1
    return rid


def _doc(name):
    """A file of docs/ – next to the program (installed) or in the repo (source); None if missing."""
    for p in (paths.doc_path("docs", name), os.path.join(paths.ROOT, "docs", name)):
        if os.path.isfile(p):
            return p
    return None


def _format_doc():
    return _doc("recipes.md")


ABOUT_DEFAULT = (("mule_default_recipe.md", "ABOUT_MULE_DEFAULT.md"),
                 ("mule_default_recipe_de.md", "ABOUT_MULE_DEFAULT_de.md"))  # the German one only where it exists


START_COPY = '''# {name} – a brief recipe for Mule

This folder is a **brief recipe**: the way an AI works with a Sketch Brief from Mule. It is a copy of **{source}** –
complete and working; change it step by step.

**Read first:**
- `FORMAT.md` – what `recipe.toml` may say, where files go (`brief/…`, `work/…`), the tokens Mule fills in, and the
  facts every template can use.
{about}
**Working on it:** change files here; Mule reads them on the next export (in the brief dialog: Brief recipe → … →
Reload). Change one thing at a time, so a comparison with the original tells you what that one thing did. After every
change raise `version` in `recipe.toml` and add a section to the changes file named there. Hook tests, if the
original had them, were left behind – test hooks by exporting a brief into a test folder and starting a session there.

This file, `FORMAT.md` and the `ABOUT_*` files stay here – they are not copied into briefs.
'''


def _add_guides(path, name, source=None):
    """FORMAT.md and START_HERE.md into a new recipe (a copy keeps a START_HERE.md it already has)."""
    doc = _format_doc()
    if doc:
        shutil.copyfile(doc, os.path.join(path, FORMAT_FILE))
    if not os.path.exists(os.path.join(path, START_FILE)):
        if source is None:
            text = _EMPTY[START_FILE].format(name=name)
        else:
            about = [f for _s, f in ABOUT_DEFAULT if os.path.isfile(os.path.join(path, f))]
            text = START_COPY.format(name=name, source=source, about="- " + " / ".join(f"`{f}`" for f in about)
                                     + " – how the original works, rule by rule, and why.\n" if about else "")
        with open(os.path.join(path, START_FILE), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)


def create_empty(name):
    """A new, nearly empty recipe in the recipe folder – usable at once (the AI reads and discusses the brief).
    -> Recipe."""
    rid = slug(name)
    path = os.path.join(user_dir(), rid)
    for rel, text in _EMPTY.items():
        if rel == START_FILE:
            continue
        full = os.path.join(path, *rel.split("/"))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text.format(id=rid, name=name.replace('"', "'")) if rel == "recipe.toml" else
                     text.replace("{{{{", "{{").replace("}}}}", "}}"))
    _add_guides(path, name)
    return Recipe(path)


def _set_key(toml, key, value):
    """Set a top-level key of recipe.toml (the line before the first [section]); added if missing."""
    line = f"{key} = {json.dumps(value, ensure_ascii=False)}"
    head, sep, rest = toml.partition("\n[")
    pat = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    head = pat.sub(lambda _m: line, head, count=1) if pat.search(head) else head.rstrip("\n") + "\n" + line
    return head + sep + rest


def copy_recipe(recipe, name):
    """A recipe of one's own from an existing one: same files, own id and name, version v1.0, forked_from names
    the original. Mule's development files (hook tests, German notes) stay behind. -> Recipe."""
    rid = slug(name)
    path = os.path.join(user_dir(), rid)

    def skip(folder, names):
        rel = os.path.relpath(folder, recipe.path)
        return [n for n in names if n in _SKIP or n.endswith(".pyc") or n in _COPY_SKIP and
                (n != "tests" or rel.replace(os.sep, "/") == "_claude")]
    shutil.copytree(recipe.path, path, ignore=skip)
    p = os.path.join(path, RECIPE_FILE)
    with open(p, encoding="utf-8") as fh:
        toml = fh.read()
    for k, v in (("id", rid), ("name", name), ("version", "v1.0"), ("author", ""),
                 ("forked_from", f"{recipe.id} {recipe.version}")):
        toml = _set_key(toml, k, v)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(toml)
    if recipe.changes:  # the original's history has versions above v1.0 – review rounds would quote it as new
        with open(os.path.join(path, recipe.changes), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"# Changes\n## v1.0\n- Copy of {recipe.name} {recipe.version}.\n")
    if recipe.builtin:  # how Mule Default works and why – the starting point of the copy, explained
        for src, dst in ABOUT_DEFAULT:
            doc = _doc(src)
            if doc:
                shutil.copyfile(doc, os.path.join(path, dst))
    _add_guides(path, name, recipe.label())
    return Recipe(path)


def unpack_folder(folder):
    """A recipe folder from somewhere (a download, a colleague's share) – copied into a temporary folder and loaded,
    like an unpacked zip; install_unpacked puts it into the recipe folder."""
    if not os.path.isfile(os.path.join(folder, RECIPE_FILE)):
        raise RecipeError(f"no {RECIPE_FILE} in this folder")
    tmp = tempfile.mkdtemp(prefix="mule_recipe_")
    dst = os.path.join(tmp, "recipe")
    total = 0
    for rel in _walk(folder):
        full = os.path.join(folder, rel)
        if os.path.islink(full):
            shutil.rmtree(tmp, ignore_errors=True)
            raise RecipeError(f"„{rel}“ is a link – not allowed in a recipe")
        total += os.path.getsize(full)
        if total > MAX_ZIP:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RecipeError("the recipe is larger than 50 MB")
    shutil.copytree(folder, dst, ignore=shutil.ignore_patterns(*_SKIP, "*.pyc"))
    r = Recipe(dst)
    r._tmp = tmp
    return r


# ---- what a recipe does on the machine (shown before import and in the dialog)

_NET = re.compile(r"\b(curl|wget|Invoke-WebRequest|Invoke-RestMethod|iwr|irm|Start-BitsTransfer|scp|ssh|nc|ncat|"
                  r"FromBase64String|base64\s+-d|DownloadString|DownloadFile)\b|https?://", re.I)


_KNOWN_SETTINGS = {"hooks", "statusLine", "permissions", "mcpServers", "$schema"}


def _hook_command(hk):
    """One hook as text: a command with its args, or its type and URL/prompt for hooks that run no command."""
    if not isinstance(hk, dict):
        return str(hk)
    if hk.get("command") or hk.get("args"):
        return " ".join([str(hk.get("command", "")), *map(str, hk.get("args") or ())]).strip()
    return f"[{hk.get('type', '?')}] " + str(hk.get("url") or hk.get("prompt") or "")[:120]


def summary(recipe):
    """{"files", "hooks": [(event, command)], "allow", "mode", "network": [(file, line, text)], "mcp", "other":
    [(file, key, value)], "unreadable": [file]} – read from the recipe's Claude settings and scripts. A help for
    looking, not a guarantee: a settings file it cannot read is named as not reviewed, never left out quietly."""
    out = {"files": 0, "hooks": [], "allow": [], "mode": "", "network": [], "mcp": [], "brief": [], "work": [],
           "other": [], "unreadable": []}
    for src, target in plan(recipe):
        out["files"] += 1
        rel = target.split("/", 1)[1]
        out["brief" if target.startswith("brief/") else "work"].append(rel)
        name = os.path.basename(target)
        if name in ("settings.json", "settings.local.json", ".mcp.json"):
            try:
                with open(src, encoding="utf-8-sig") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    raise ValueError("not an object")
                for event, groups in (data.get("hooks") or {}).items():
                    for g in groups if isinstance(groups, list) else [groups]:
                        hooks = g.get("hooks") if isinstance(g, dict) else None
                        for hk in hooks if isinstance(hooks, list) else [g]:
                            out["hooks"].append((str(event), _hook_command(hk)))
                if data.get("statusLine"):
                    out["hooks"].append(("statusLine", _hook_command(data["statusLine"])))
                perm = data.get("permissions") or {}
                if isinstance(perm, dict):
                    out["allow"] += [str(x) for x in perm.get("allow") or ()]
                    out["mode"] = out["mode"] or str(perm.get("defaultMode") or "")
                    if perm.get("additionalDirectories"):
                        out["other"].append((rel, "permissions.additionalDirectories",
                                             str(perm["additionalDirectories"])[:160]))
                servers = data.get("mcpServers") or {}
                out["mcp"] += list(servers.keys()) if isinstance(servers, dict) else [str(servers)]
                # everything else a settings file can do (env, apiKeyHelper, model, …) is shown as it is
                for key in sorted(set(data) - _KNOWN_SETTINGS):
                    out["other"].append((rel, key, json.dumps(data[key], ensure_ascii=False)[:160]))
            except Exception:  # noqa: BLE001 – unreadable or of an unknown shape: say so, never stay silent
                out["unreadable"].append(rel)
        if os.path.splitext(src)[1].lower() in _TEXT and not src.endswith(".md"):
            try:
                with open(src, encoding="utf-8", errors="replace") as fh:
                    for k, line in enumerate(fh, 1):
                        if _NET.search(line):
                            out["network"].append((rel, k, line.strip()[:120]))
            except OSError:
                out["unreadable"].append(rel)
    return out


# ---- import

def unpack(zip_path):
    """Unpack a recipe zip into a fresh temporary folder (checked: no paths out of it, size limit) and load it.
    -> Recipe (its folder is temporary; install_unpacked moves it into the recipe folder)."""
    tmp = tempfile.mkdtemp(prefix="mule_recipe_")
    try:
        with zipfile.ZipFile(zip_path) as z:
            total = 0
            for info in z.infolist():
                name = info.filename.replace("\\", "/")
                if name.startswith("/") or ".." in name.split("/") or ":" in name:
                    raise RecipeError(f"„{info.filename}“ points outside the recipe")
                total += info.file_size
                if total > MAX_ZIP:
                    raise RecipeError("the recipe is larger than 50 MB unpacked")
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise RecipeError(f"„{info.filename}“ is a link – not allowed in a recipe")
            z.extractall(tmp)
    except zipfile.BadZipFile as exc:
        shutil.rmtree(tmp, ignore_errors=True)
        raise RecipeError(f"not a zip file: {exc}") from exc
    except RecipeError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    root = tmp
    if not os.path.isfile(os.path.join(root, RECIPE_FILE)):  # zip of the folder: one level down
        subs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d not in _SKIP
                and d != "__MACOSX"]
        if len(subs) == 1 and os.path.isfile(os.path.join(root, subs[0], RECIPE_FILE)):
            root = os.path.join(root, subs[0])
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            raise RecipeError(f"no {RECIPE_FILE} in the zip")
    r = Recipe(root)
    r._tmp = tmp
    return r


def target_of(recipe):
    """Folder the recipe goes to in the recipe folder."""
    return os.path.join(user_dir(), recipe.id)


def install_unpacked(recipe, replace=False):
    """Copy an unpacked recipe into the recipe folder. Replacing goes through a fresh copy first and swaps the folders
    only when it is complete – an open window or session in the old folder can stop it, but never leave half a recipe.
    -> the installed Recipe."""
    if recipe.id == DEFAULT_ID:
        raise RecipeError("the id „default“ belongs to Mule's own recipe – give the recipe its own id")
    if not valid_id(recipe.id):
        raise RecipeError(f"id „{recipe.id}“ is not usable as a folder name")
    dst = target_of(recipe)
    if os.path.islink(dst) or getattr(os.path, "isjunction", lambda _p: False)(dst):
        raise RecipeError(f"„{dst}“ is a link to another folder – Mule does not replace it; change it there or "
                          "remove the link first")
    if os.path.exists(dst) and not replace:
        raise RecipeError(f"a recipe „{recipe.id}“ is already installed")
    fresh, old = dst + ".mule_new", dst + ".mule_old"
    try:
        for p in (fresh, old):
            shutil.rmtree(p, ignore_errors=True)
        shutil.copytree(recipe.path, fresh, ignore=shutil.ignore_patterns(*_SKIP, "*.pyc"))
        if os.path.exists(dst):
            os.rename(dst, old)
        try:
            os.rename(fresh, dst)
        except OSError:
            if os.path.exists(old) and not os.path.exists(dst):
                os.rename(old, dst)  # put the old one back
            raise
        shutil.rmtree(old, ignore_errors=True)
    except OSError as exc:
        shutil.rmtree(fresh, ignore_errors=True)
        raise RecipeError(f"could not install „{recipe.id}“ – is its folder open in Explorer or a session? ({exc})")
    discard(recipe)
    return Recipe(dst)


def discard(recipe):
    """Remove the temporary folder of an unpacked recipe (never anything else)."""
    tmp = getattr(recipe, "_tmp", None)
    if tmp and os.path.basename(tmp).startswith("mule_recipe_"):
        shutil.rmtree(tmp, ignore_errors=True)
