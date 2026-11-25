"""English manual: same structure as manual.py (line for line, test_language compares the lines per tab)."""
from .manual import _section

START = "<h1>Mule Create</h1>" + """<p class="lead">Sketch in space as if on paper. The drawing is the lines; planes are
tracing paper you draw on. Lines become surfaces, and notes are prompts for later processing in Houdini.</p>""" + \
    _section("Three modes", "Each mode has its own bar on the left.", [
        ("<b>Draw</b>: pen, brush, eraser", "[A] [D] · [Esc]"),
        ("<b>Edit</b>: select, move, build surfaces", "[Tab]"),
        ("<b>Notes</b>: headings and boards in space", "[N]"),
    ]) + \
    _section("Camera", "", [
        ("Orbit · pan · zoom", "[Alt]/[Space] + [LMB] · [MMB] · [RMB]"),
        ("Perspective · Top · Front · Right", "[Space]+[1] … [4]"),
        ("Focus selection · Home", "[Space]+[G] · [Space]+[H]"),
        ("Maximize viewport · split", "[Space]+[F] · right-click → Split"),
    ]) + \
    _section("File", "", [
        ("Save · open · new", "[Ctrl]+[S] · [O] · [N]"),
        ("Import geometry (USD: hierarchy as in Solaris) · export Sketch Brief", "[Ctrl]+[I] · [Ctrl]+[E]"),
        ("Insert reference object (to compare sizes): person 1.80 m, car, detached house", "File menu"),
        ("Brief assignment on export: assignment in one sentence, notes, discuss only, look pre-stage (the AI clarifies "
         "the rest from the notes)", "[Ctrl]+[E]"),
        ("Screenshot as hero camera for the brief (picture + USD camera), name below it", "check box at the picture's top right"),
        ("Caption on the board (what the picture stands for, what matters – appears after the picture in the brief)",
         "right-click the picture"),
        ("Discuss the brief with the AI: autoprompt on the clipboard after export, Claude sees the "
         "scene (MCP)", "check box in the brief dialog"),
        ("Build brief: way of working with rules, roles and hooks, project.md, work folder work_<name> next to it – "
         "start the Claude session in the brief folder; change the level live", "SET: level=scout|study|final"),
        ("Review round: feedback for the running chat – goes to <brief>/review_rounds/ without picking a folder, only "
         "what changed in the notes, no new rules; paste the sentence into the same chat", "“Builds on” in the brief dialog"),
        ("Choose the build brief's way of working: Mule Default or a recipe of your own (rules, roles, hooks, texts); "
         "make a new or copied recipe, import one as .zip or folder, look at its contents", "“Brief recipe” in the brief dialog · …"),
        ("Visibility map (detail only where it is seen) in Blender, one command; check the tool",
         "tools/vis_map/vis_blender.py · --selftest"),
        ("Export scene as USD (surfaces as polymeshes, meshes referenced)", "[Ctrl]+[Shift]+[E]"),
        ("Startup animation when the program starts (click or key skips it)", "View menu"),
        ("Reset workspace: outliner, note and toolbars back in their place", "View menu"),
        ("Recording: log every input with time and processing time, optionally starts OBS too", "View → Recording"),
        ("Review: walk through the scene and talk your feedback – voice, camera and cursor are recorded, Claude turns "
         "it into draft notes", "View → Review"),
        ("Mark during the review: a click in the view sets a marker, a click on a picture on a board remembers the "
         "picture for the note of that part", "Mark tool · [Ctrl]+[Z] undo marker"),
        ("Accept a review draft or listen to its spot (dashed note)", "Board: ✓ Accept · ▶ Listen"),
        ("Claude's arrows follow their text and picture when you move or scale them on the board", "Board"),
        ("Export curves only: USD for 3D programs, JSON for scripts and AI chats, width per point", "File → Export → Curves"),
        ("Undo · redo (including selection)", "[Ctrl]+[Z] · [Ctrl]+[Shift]+[Z]"),
    ])

DRAW = _section("Tools", "", [
    ("Pen (graphite) ↔ brush (color, oil texture)", "[A]"),
    ("Eraser – [Ctrl] lines only, [Ctrl]+[Shift] whole lines", "[D]"),
    ("Paint mode: color flat on the surface, no editable lines", "[S]"),
    ("Color the active surface · reset", "[F] · [Shift]+[F]"),
]) + \
    _section("Drawing target", "You draw on the active plane or surface (tinted, strong outline).", [
        ("Activate a surface", "click"),
        ("Base plane by view axis (side · ground · front) · last own plane", "[X] [Y] [Z] · [4]"),
        ("New plane from the view", "[Space]+[N]"),
        ("New stroke object (in the group you are working in)", "[Space]+[B]"),
        ("Move plane", "wheel · &#91; &#93; · [Shift]+[RMB]"),
        ("Multi: on several surfaces at once", "[B]"),
        ("Free: no plane, start on lines/surfaces", "[Q]"),
        ("Picture on a helper plane/surface: Load Picture, Project from the camera, Lock prevents it", "arrow in front of it in the Outliner"),
    ]) + \
    _section("Strokes", "", [
        ("Hold still at the end: straight line, circle, rectangle, arc", "hold ½ s"),
        ("Straight line in 10° steps", "[Ctrl]"),
        ("Snap to points (rings) and lines of the drawing surface (off: magnet button)", "automatic"),
        ("Pen size · width of all lines", "[Ctrl]+wheel · [Ctrl]+[Shift]+wheel"),
        ("Largest stroke width (default 50 m, up to 5 km)", "Right-click → Maximum stroke width …"),
    ])

EDIT = _section("Select", "", [
    ("Points · edges · surfaces", "[1] [2] [3]"),
    ("Show the active object's points: select the object", "click a line"),
    ("Replace · add · remove", "click · [Shift] · [Ctrl]"),
    ("Rectangle (hidden parts only with X-Ray)", "drag"),
    ("Jump to another object (edit it)", "double-click it"),
    ("Loop · surface with its lines (on the edited object)", "double-click"),
    ("Connected: in the object · everywhere · loop along the network", "[L] · [Shift]+[L] · [Ctrl]+[L]"),
    ("Drag points: auto · soft · hard", "bar on the left"),
    ("Hide · isolate · show all", "[H] · [Shift]+[H] · [Alt]+[H]"),
    ("Color selected surfaces and brush lines (pencil stays) · reset the surfaces' color",
     "color picker → Apply · [Shift]+click"),
    ("Convert pencil ↔ brush stroke", "right-click"),
]) + \
    _section("Move", "", [
        ("Move · rotate · scale", "[T] · [R] · [E]"),
        ("Snap (fine with [Shift])", "[Ctrl]"),
        ("Snap to a point and connect", "hold [V]"),
        ("Selection only, neighbors stay", "[Alt] while dragging"),
        ("Move pivot · axes: world → local → camera", "[Alt]+[A] · [Space]+[M]"),
    ]) + \
    _section("Points and edges", "Lines are connected when they share a point (filled point); an "
             "open ring is a free end.", [
                 ("Insert point", "[Ctrl]+click"),
                 ("Slide points along their line", "hold [S] + drag"),
                 ("<b>Hardness</b> (on/off): shape a corner – point: all lines kink, edge: the cross lines kink (crease)", "[K]"),
                 ("Fully hard · fully soft", "[Shift]+[K] · [Alt]+[K]"),
                 ("<b>Smooth</b> (on/off): click, drag – fully smoothed, the edge is straight", "[J]"),
                 ("Dissolve (shape stays) · remove (smooth again) · delete (straight bridge)",
                  "[Del] · [Shift]+[Del] · [Ctrl]+[Del]"),
                 ("Round corners ([F]: round ↔ chamfer)", "[Ctrl]+[B]"),
                 ("Width of the selected lines", "[Ctrl]+wheel"),
             ]) + \
    _section("Organize", "", [
        ("Copy · paste · duplicate · repeat", "[Ctrl]+[C] · [V] · [D] · [Shift]+[D]"),
        ("Copy into another scene or a second Mule window: paste over the Outliner (board elements with their images)",
         "[Ctrl]+[C] · [V]"),
        ("Selection to new object – surfaces whose lines all move go along", "[Ctrl]+[G]"),
        ("Merge several line objects (their surfaces then sit under the one)", "select objects, [Ctrl]+[G]"),
        ("Clean up: weld, connect, duplicates", "Cleanup at the top"),
    ]) + \
    _section("Groups", "As in Cinema 4D: a null collects objects in the Outliner and moves them along.", [
        ("Null from the selection", "[Alt]+[G]"),
        ("Ungroup (objects stay)", "[Shift]+[G]"),
        ("Reparent: drag onto a null · back: onto a category", "Outliner"),
        ("Lines + selected surface/objects into a new group (at the lines' location)", "[Ctrl]+[G]"),
        ("Color of a null (applies below it down to the next colored null)", "right-click → Color"),
        ("Expand/collapse everything below", "[Shift]+click on the arrow"),
        ("Search · drag the name column wider", "field above the Outliner · header"),
        ("Binding level for the brief (null: everything below)", "“Brief” column in the Outliner"),
        ("Move only the null", "[Alt]+[A]"),
    ]) + \
    _section("Symmetry and array", "Generators: you edit the original, mirror image and copies follow live "
             "(display only).", [
        ("Symmetry · array from the selection (without one: a new stroke inside; takes the selected null's position)", "Objects menu"),
        ("On/off · mirror plane · kept side · mode, count, random", "Outliner, under the generator"),
        ("Stroke ends snap to the mirror plane", "magnet on"),
        ("Clicking a copy selects its generator", "Edit"),
        ("Path · up line (along line): select a line, take it", "Outliner"),
        ("Convert: real lines and surfaces (cut, welded)", "[Alt]+[C]"),
    ])

SURFACES = _section("Build surfaces", "Surfaces come from lines and follow them; the lines on a surface "
                    "shape it.", [
                        ("Surface from selected lines", "[F]"),
                        ("Loft · ring · force curve surface", "[Shift]+[F] · [Ctrl]+[Shift]+[F] · [Alt]+[F]"),
                        ("Split loft", "[Alt]+[D]"),
                        ("Line from border to border: network line, the surface passes exactly and smoothly through it", "automatic"),
                        ("Extrude (axis: [X] [Y] [Z] [N]) – a line at every point kinked 5°+, separate walls from 15°", "[X]"),
                        ("Extrude a surface: it moves up", "surface mode, [X]"),
                        ("Move a surface: all its lines and its paint move with it", "select the surface, gizmo"),
                    ]) + \
    _section("Appearance", "Surfaces occlude like real surfaces; their material is display only.", [
        ("Material", "dropdown in the Outliner"),
        ("Helper surface (frozen, only the gizmo moves it) · transparency · resolution", "arrow in front of the surface"),
        ("Material scale", "right-click X-Ray"),
        ("Color · reset", "Draw: [F] · [Shift]+[F]"),
    ])

NOTES = _section("Notes", "Short headings in space – prompts for later. Always upright.", [
    ("New note: on a line or in empty space", "double-click"),
    ("Rename · rotate · delete", "[F2] · [R] · [Del]"),
    ("Move ([Shift]: vertical)", "drag"),
    ("Another leader line: drag from the ring at the bottom left", "drag"),
    ("Size, rotate with camera, detach", "right-click"),
    ("Octopus: click an anchor point with an octopus (magenta with ring, on the selected note) – its arms appear (again: off)", "click"),
    ("Octopus: drag from the clicked anchor point onto parts of the same kind", "[Shift] + drag"),
    ("Octopus: add/remove a part in the viewport (note or anchor point selected)", "[Shift] + click"),
    ("Octopus: add the whole group (null) of the clicked part", "[Ctrl] + [Shift] + click"),
    ("Octopus: attach selected objects/lines to the anchor · remove arm · clear", "right-click · [Del] on the ring"),
    ("Selected note: connected parts glow magenta · only those of one anchor ([Esc]: all)", "click an anchor point"),
    ("Parts hidden by surfaces are missing (Notes mode: faint, X-Ray: everything)", "–"),
]) + \
    _section("Note board", "Text and pictures for the note (Local) or the scene (Global), panel below the Outliner.", [
        ("Open large", "⤢"),
        ("Pin: the board stays visible in every mode (reference while drawing, painting, Edit)", "📌 · View menu"),
        ("Insert view as a picture (at the note or on the board) · fly back", "camera · double-click the picture"),
        ("Paint: pen · eraser · off (color: dots at the top of the board)", "[A] · [D] · [Esc] / ✎"),
        ("Brush width on the board (the same at every zoom; Ctrl+wheel on a picture/text when not painting: scale it)", "[Ctrl] + wheel / right-drag"),
        ("Width of the strokes on the board (selected, else all)", "[Ctrl] + [Shift] + wheel / right-drag"),
        ("Arrow from a label to a picture: the connection lights up briefly after you pause (selected stroke: permanently)", "pen"),
        ("Backdrop: frame with a title around the selection – move it by its tab (content moves along), a section in the brief", "[B] · ▭"),
        ("Auto backdrops: frames around everything close together", "[Shift]+[B] · right-click ▭"),
        ("Straight line / circle / rectangle / arc: hold still at the stroke end", "hold"),
        ("Erase whole strokes", "[Ctrl]+[Shift] + eraser"),
        ("Paste picture/text (also a picture from the browser) · drag files in", "[Ctrl]+[V] · drag & drop"),
        ("Selected items from board to board: copy · cut · paste (global ↔ local)",
         "[Ctrl]+[C] · [X] · [V]"),
        ("Drag a video in (arrows and captions as with pictures): button/[Space] plays, drag the timeline · right-click: still", "drag & drop"),
        ("Scale picture/text (pictures always lie below the strokes)", "drag a corner · [Ctrl]+wheel"),
        ("Retake screenshots (missing ones, otherwise all; same camera and format, display as currently set)", "↻ (top of the board)"),
        ("Retake one screenshot (Shift: from the current view)", "↻ at the picture's top right"),
        ("Undo · redo (shared with the 3D view)", "[Ctrl]+[Z] · [Ctrl]+[Y]"),
        ("Zoom · pan", "wheel / [Alt]+[RMB] · [MMB]"),
    ])

TIMING = _section("Shots (prototype)", "Show what happens when, without animating: a drawing holds until the next one, cameras jump from position to position. Shot notes work like notes – with their own Shots panel (Local/Global).", [
    ("Create a flipbook layer (only in Shots; only paintable here)", "[Space]+[N]"),
    ("Move · rotate · scale a flipbook layer with all its drawings (gizmo; no mouse wheel) · gizmo off", "[T] [R] [E] · [Esc]"),
    ("Frame · first/last · next/previous drawing", "[←] [→] · [Shift]+[←] [→] · [↑] [↓]"),
    ("Painting/erasing at a frame: in its drawing – if there is none, the one shown is copied (keep painting)", "pen / brush / eraser"),
    ("Empty drawing · delete drawing", "timeline"),
    ("Move a drawing (change timing) · delete it", "drag the diamond · right-click"),
    ("Onion skin (previous drawing 30 %, next one hinted) · play (every frame)", "[O] · [Alt]+[A]"),
    ("Create a shot note (on a drawing: attached to it) · click it: its board in the Shots panel", "double-click · click"),
    ("Move a shot note · leader to more layers · delete", "drag · drag the ring · [Del]"),
    ("Global shot note (for all shots)", "Shots panel → Global"),
    ("New camera from the view (its picture goes on the board)", "🎥 on the board / at the note"),
    ("Pick a camera · look through · leave", "click the symbol · double-click (symbol, picture) / 👁 · [Esc]"),
    ("Every position is a picture on the board, its settings below it: colour, name, frame, focal length", "bar under the picture"),
    ("Inside the camera: change position (no move) · new position from here (move, new picture)", "⟲ · ➜"),
    ("Focal length of the position (keeps the framing; [Alt]: plain zoom) · delete a position", "slider · delete the picture"),
    ("Frame of the position · symbol size · retake the picture", "right-click “from F…” · right-click the picture · ↻"),
    ("Export: cameras stepped or linear (brief and USD dialog), the brief gets a Shots part with pictures and boards", "[Ctrl]+[E] · [Ctrl]+[Shift]+[E]"),
])

VIEW = _section("Top right", "", [
    ("Drawing aids (off: raw result, without planes and helper surfaces)", "drawing aids icon"),
    ("X-Ray – meshes with all layers (opacity: right-click)", "X-Ray icon"),
    ("Add wireframe in X-Ray – all edges (opacity: right-click)", "sphere icon"),
    ("Hide lines, color lines, paint, surfaces separately", "show/hide icon"),
    ("Color view ↔ black and white · Color Compress", "icon · View → Color Compress"),
    ("Resolution of all surfaces (viewport quality)", "slider at the top (mesh icon)"),
    ("Chalkboard ↔ paper · pencil and chalk color", "moon icon · View menu"),
    ("Bottom of the viewport: active tool and switches", "–"),
    ("Interface language (Deutsch/English, applies after a restart)", "View → Sprache / Language"),
    ("Where Mule keeps data, settings and backups – with sizes; removing Mule completely", "Help → Where Mule keeps its files …"),
]) + \
    _section("Light", "One sun, display only – lines, surfaces and imported meshes cast and receive shadows.", [
        ("Light on/off", "button below the axis gizmo"),
        ("Place the sun: sky disc at the bottom right", "button below it"),
        ("Light steps (3 … smooth; the fewer steps, the more small ripples are smoothed)", "right-click the light icon"),
    ])

KEYS = _section("Everywhere", "", [
    ("Draw · Edit · Notes · manual", "[A] [D] · [Tab] · [N] · [F1]"),
    ("Camera · views", "[Alt]/[Space] + mouse · [Space]+[1] … [4]"),
    ("Undo · save · brief", "[Ctrl]+[Z] · [Ctrl]+[S] · [Ctrl]+[E]"),
]) + _section("Draw", "", [
    ("Brush · eraser · paint mode", "[A] · [D] · [S]"),
    ("Planes · new plane · multi · free", "[X] [Y] [Z] [4] · [Space]+[N] · [B] · [Q]"),
    ("Color", "[F]"),
]) + _section("Edit", "", [
    ("Points · edges · surfaces", "[1] [2] [3]"),
    ("Move · rotate · scale", "[T] [R] [E]"),
    ("Surface · loft · curve surface · extrude", "[F] · [Shift]+[F] · [Alt]+[F] · [X]"),
    ("Hardness · smooth · slide", "[K] · [J] · hold [S]"),
    ("Round · split loft · new object · connected", "[Ctrl]+[B] · [Alt]+[D] · [Ctrl]+[G] · [L]"),
    ("Hide · isolate · show all", "[H] · [Shift]+[H] · [Alt]+[H]"),
])

TABS = (("Start", START), ("Draw", DRAW), ("Edit", EDIT), ("Surfaces", SURFACES), ("Notes", NOTES),
        ("Shots", TIMING), ("View", VIEW), ("Keys", KEYS))
