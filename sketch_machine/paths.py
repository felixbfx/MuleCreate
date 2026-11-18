"""Where files live: in the repo (development) or as an installed program (PyInstaller).

When installed, the program folder is often write-protected: the bible and log then live in the user folder. There one can
also put one's own MCP entries (.mcp.json, .codex/config.toml) – every export carries them into the brief folder.
"""
import faulthandler
import os
import shutil
import sys

FROZEN = bool(getattr(sys, "frozen", False))
ROOT = sys._MEIPASS if FROZEN else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIBLE_NAME = os.path.join("bible", "sketch_machine_bible.md")
LOG_NAME = "mule.log"


def doc_path(*parts):
    """Documentation next to the program: in the repo its root, installed the folder of Mule.exe (docs/,
    AI_HANDBOOK.md – the release copies them there). In the macOS app bundle the spec puts them into
    Contents/Resources (reached through sys._MEIPASS), as nothing but code may sit next to the executable;
    the third-party licenses (THIRD_PARTY_NOTICES.md, LICENSE.txt, licenses/) live in the same folder."""
    if FROZEN and sys.platform == "darwin":
        base = ROOT
    else:
        base = os.path.dirname(sys.executable) if FROZEN else ROOT
    return os.path.join(base, *parts)


def user_dir(create=True):
    """Folder for the user's data (is created unless create is False)."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    path = os.path.join(base, "Mule")
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def bible_path():
    """Workflow bible: in the repo; when installed, a writable copy that updates do not overwrite."""
    source = os.path.join(ROOT, BIBLE_NAME)
    if not FROZEN:
        return source
    path = os.path.join(user_dir(), BIBLE_NAME)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.copyfile(source, path)
    return path


def tool_config_dir():
    """Folder with the MCP entries for the brief folder."""
    return user_dir() if FROZEN else ROOT


def start_log():
    """When installed there is no console: output, errors and crashes go to the log (new each start)."""
    if not FROZEN:
        return None
    path = os.path.join(user_dir(), LOG_NAME)
    log = open(path, "w", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    faulthandler.enable(log)
    return path
