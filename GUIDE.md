<img src="sketch_machine/assets/mule_icon.png" alt="Mule Create logo" width="72" align="right">

# Mule Create – Tester Guide (Alpha 1)

Mule Create is a program for sketching in 3D space with lines. You draw on planes that you set up in space, add
notes and reference pictures, and export a **Sketch Brief**: a folder with pictures and text from which an AI
(e.g. Claude Code or Codex) rebuilds the object in Houdini or Blender.

This is an early public alpha. There will be bugs – please report them (see below).

## Install and start

**Windows (recommended):** **[download `Mule_Windows.zip`](https://github.com/felixbfx/MuleCreate/releases/latest/download/Mule_Windows.zip)** (always the newest version; all versions:
[releases](https://github.com/felixbfx/MuleCreate/releases)), unzip it into a folder of its own and start `Mule\Mule.exe`.
No Python and no installation needed.

**Windows warning on first start:** the program is not signed yet, so Windows SmartScreen shows
“Windows protected your PC”. Click **“More info”**, then **“Run anyway”**. (On a German Windows:
„Weitere Informationen“ → „Trotzdem ausführen“.) If your company does not allow this, cancel and ask your IT.

**From source:** unzip `Mule_Source.zip`. The launchers are in the extracted repository root:

- macOS: install Homebrew, uv and Xcode with its Metal toolchain as described in
  [the README](README.md#install-and-start), then run `./start.sh`. It uses uv to prepare dependencies, builds
  `dist/Mule.app`, and launches that app. You can also double-click the built app in Finder.
- Linux: install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run `./start.sh`. uv provides
  Python 3.14 and the pinned packages, then launches Mule from source. First use needs internet for downloads.
- Windows: double-click `start.bat` (the first start sets everything up itself, needs Python 3.14 and internet once), or by hand `py -3.14 -m venv .venv`, `.venv\Scripts\pip install -r requirements.txt`,
  `.venv\Scripts\python -m sketch_machine`.

Linux has hardly been tested so far.

Mule needs a graphics card with OpenGL 3.3. Traces outside its folder: settings in the registry
(`HKCU\Software\SketchMachine`), log and recordings in `%APPDATA%\Mule` (on Linux in the home folder).

**Language:** menu View → Sprache / Language, then restart.

**Example scene:** [Example_Alpaca_Monowheel.zip](https://github.com/felixbfx/MuleCreate/releases/download/v0.99.255/Example_Alpaca_Monowheel.zip) – a finished project to look at: the sketch and
the model an AI built from it (USD, with and without cargo). Unzip, keep the folder together, open
`Alpaca_Monowheel.mule`. Free to use under CC BY 4.0 (credit: "Alpaca Monowheel by Felix Barthel (Mule Create)").

## The main workflows

The manual inside the program (**F1**) explains everything in more detail; every function and shortcut is in
`AI_HANDBOOK.md` – written for AI assistants: give it to your AI and ask your question.

**Drawing.** Draw on the active plane with the left mouse button. Camera: Alt (or Space) + left / middle / right
mouse button. Holding still at the end of a stroke makes a straight line or a recognised shape (circle,
rectangle …). **X / Y / Z** picks the ground plane, **Space + N** creates a new plane from the camera view, **Q**
draws freely without a plane. In **Edit** mode move, rotate and scale (**T / R / E**). Undo: Ctrl + Z.

**Notes and reference pictures.** In Notes mode attach notes to parts of the drawing. Every note has a note board:
drag in pictures and videos, write by hand or with the keyboard, draw arrows onto picture details, take
screenshots of the scene. Ticking “hero camera” on a screenshot turns it into a camera the AI will later look
through.

**Sketch Brief.** File → Export Sketch Brief: first the task in one sentence, then choose a folder. Mule puts
pictures, notes and working rules for the AI there. Open that folder in Claude Code or Codex and let the AI work
from it. In the first round the AI asks about whatever the notes leave open.

**Review mode.** Walk through the scene and talk your feedback, like in a review session.
- **Review → Start review** starts the recording (Stop review or Send Review ends it). Mule records the microphone
  and remembers camera, cursor and selection (no video).
- During the review the **Mark** tool is on (button next to “Review”): every click in the view sets a marker at
  exactly that spot, a click on a picture on a board remembers that picture (“the tank should look like this”) –
  the AI then puts it on the note of that part. Navigate with Alt or Space as always; **Ctrl + Z** removes the last
  marker. Button off: keep working normally, the recording goes on. Saying “here” or “there” while the cursor rests
  also counts as pointing.
- **Send Review** (button next to “Review”) stops the recording, writes the transcript and puts a sentence on the
  clipboard. Tell the AI in your chat “the review is ready”: it creates draft notes in the open scene (dashed).
  Every draft shows your original sentence, **▶ Listen** plays the moment, **✓ Accept** turns it into a normal
  note. Nothing is saved until you save.
- The transcript needs Whisper (see below). Without Whisper the review is recorded but not transcribed.

**AI connection (MCP).** While Mule is running it offers an MCP server on this computer:
`http://127.0.0.1:8770/mcp` (local only, no login). Through it the AI sees the open scene, can render pictures
from the hero cameras, take dictated notes, export the brief and work through reviews.

- **Claude Code:** create a file `.mcp.json` in your project folder:
  ```json
  {"mcpServers": {"sketch_machine": {"type": "http", "url": "http://127.0.0.1:8770/mcp"}}}
  ```
  or run once `claude mcp add --transport http sketch_machine http://127.0.0.1:8770/mcp`.
- **Codex:** in `~/.codex/config.toml` (or `.codex/config.toml` in the project):
  ```toml
  [mcp_servers.sketch_machine]
  url = "http://127.0.0.1:8770/mcp"
  ```
- Mule has to be open before the AI session starts. A second Mule does not start a second server.

## Whisper (optional, for review transcripts)

Speech recognition is not part of the package: it is large and runs in a Python of its own. Mule sets it up
itself: on your first review Mule asks (**Set Up …** / Later / Don't ask again), or at any time via the Review menu →
**“Set Up Transcription …”**.

- This needs, once, a **Python 3.10–3.13** from [python.org](https://www.python.org/downloads/). In the installer
  tick **“Add python.exe to PATH”** at the bottom. If Mule finds no Python, the dialog explains this.
- Mule does the rest: its own environment in `%APPDATA%\Mule\whisper_venv`, faster-whisper (with an NVIDIA
  graphics card also the CUDA libraries) and the speech model `large-v3-turbo`. About **2–4 GB** of downloads in
  total, a few minutes depending on your internet.
- With an NVIDIA graphics card transcribing is fast, otherwise the processor does the work (much slower, but it
  runs in the background).

For advanced users: your own Python with faster-whisper works too. Mule finds it in `~/mcp/whisper_venv` or via
the environment variable `SM_WHISPER_PYTHON`.

## Where Mule keeps its files

Mule has no installer and writes nothing into its own program folder. Help → **Where Mule keeps its files …** shows
all of it on your machine, with sizes, and opens the data folder.

| What | Where (Windows) | Notes |
|---|---|---|
| The program | the folder you unzipped | delete it to remove Mule |
| Your data | `%APPDATA%\Mule` (`C:\Users\<you>\AppData\Roaming\Mule`) | kept across updates |
| – `recipes\` | | brief recipes you made or imported |
| – `Reviews\` | | spoken feedback |
| – `Aufnahmen\` | | click logs of the recorder, only when you switch it on |
| – `whisper_venv\` | | speech recognition, only when you set it up (2–4 GB) |
| – `bible\`, `mule.log` | | your copy of the workflow bible, log of the last start |
| Settings | registry `HKEY_CURRENT_USER\Software\SketchMachine` | window, colours, language, pen pressure |
| Backups | `backups\` next to each saved scene; unsaved scenes in `Documents\Mule\backups` | |
| Briefs | where you export them, with `work_<name>` next to each | |

On macOS the data folder is `~/Library/Application Support/Mule`, on Linux `~/.local/share/Mule`; the settings are a
file there (the overview names it).

**Removing Mule completely:** delete the program folder and the data folder, and the settings with **Delete settings
…** in that overview. Your scenes, backups and briefs stay where you put them.

## Known limits

- Alpha: scene files may change between versions; older scenes are not always supported further.
  Copy important scenes first.
- Only Windows is tested in the package. Linux runs from source and is hardly tested; macOS not at all.
- The program is not signed (SmartScreen warning, see above).
- Whisper is not in the package (see above). For the AI connection you need your own Claude or Codex account.
- The interface and the manual (F1) exist in English and German (View → Sprache / Language).
- Internally, package, settings, scene format and MCP server are still called “Sketch Machine”.
- Large scenes can get slow while drawing.
- Videos on the note board play without sound.
- The MCP server has no password: it only listens on this computer (127.0.0.1), but any program running on this
  computer could read the open scene and write notes while Mule is open.

## Reporting problems

There is no built-in “Report a problem” function yet. Please open an issue on GitHub (see `CONTRIBUTING.md`) with:

1. what you did, what you expected and what happened,
2. the version (in the release name, e.g. “Mule Create Alpha 1 (v0.99.229)”),
3. the file `mule.log` from `%APPDATA%\Mule` (Windows: `Win + R`, type `%APPDATA%\Mule`),
4. if possible the scene (the `.mule` file; if it has videos, also the `videos/` folder next to it) and a screenshot.

Ideas, things that annoy you, and what worked well or badly with the AI are just as welcome as bug reports.

## License

Mule Create – Copyright (C) 2026 Felix Barthel. Free software under the GNU General Public License v3.0
(`LICENSE`). Third-party software in the package with its licenses: `THIRD_PARTY_NOTICES.md` and the folder
`licenses/` next to `Mule.exe`. The name “Mule Create” and the Mule logo are not covered by the license.
