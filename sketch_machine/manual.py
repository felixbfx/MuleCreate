"""In-app manual (Help → Manual, F1): a separate window with tabs – Start, Draw, Edit, Surfaces,
Notes, View, Keys. Opens at the tab of the mode the user is currently in. Keep it short: one line per topic.

Content as terse HTML; keys are written [Ctrl]+[Z] – they are shown as small key caps."""
import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTabWidget, QTextBrowser, QVBoxLayout, QWidget

from . import i18n, ui
from .i18n import tr


def _keys(text):
    """[Key] -> key cap."""
    return re.sub(r"\[([^\]]+)\]", r'<span class="key">&nbsp;\1&nbsp;</span>', text)


def _table(rows):
    body = "".join(f'<tr><td class="what">{_keys(a)}</td><td class="how">{_keys(b)}</td></tr>' for a, b in rows)
    return f'<table width="100%" cellspacing="0" cellpadding="6">{body}</table>'


def _section(title, intro="", rows=()):
    out = f"<h2>{title}</h2>"
    if intro:
        out += f"<p>{_keys(intro)}</p>"
    if rows:
        out += _table(rows)
    return out


CSS = f"""
body {{ color: {ui.TEXT}; font-size: 13px; }}
h1 {{ color: {ui.ACCENT}; font-size: 20px; font-weight: 600; margin-bottom: 2px; }}
h2 {{ color: {ui.ACCENT}; font-size: 15px; font-weight: 600; margin-top: 18px; margin-bottom: 4px; }}
p {{ color: {ui.TEXT}; margin-top: 2px; margin-bottom: 6px; }}
p.lead {{ color: {ui.TEXT_DIM}; }}
td.what {{ color: {ui.TEXT}; border-bottom: 1px solid {ui.LINE}; }}
td.how {{ color: {ui.TEXT_DIM}; border-bottom: 1px solid {ui.LINE}; width: 34%; }}
.key {{ background-color: {ui.RAISED}; color: {ui.TEXT}; font-family: Consolas, monospace; font-size: 12px; }}
"""

START = "<h1>Mule Create</h1>" + """<p class="lead">Im Raum skizzieren wie auf Papier. Die Zeichnung sind die
Linien; Ebenen sind Transparentpapier, auf das man zeichnet. Aus Linien werden Flächen, Notizen sind Prompts für die
spätere Weiterverarbeitung in Houdini.</p>""" + \
    _section("Drei Modi", "Jeder Modus hat links seine eigene Leiste.", [
        ("<b>Zeichnen</b>: Stift, Pinsel, Radierer", "[A] [D] · [Esc]"),
        ("<b>Edit</b>: wählen, bewegen, Flächen bauen", "[Tab]"),
        ("<b>Notizen</b>: Überschriften und Bretter im Raum", "[N]"),
    ]) + \
    _section("Kamera", "", [
        ("Drehen · Schieben · Zoomen", "[Alt]/[Space] + [LMB] · [MMB] · [RMB]"),
        ("Perspektive · Top · Front · Right", "[Space]+[1] … [4]"),
        ("Auswahl fokussieren · Home", "[Space]+[G] · [Space]+[H]"),
        ("Viewport maximieren · teilen", "[Space]+[F] · Rechtsklick → Split"),
    ]) + \
    _section("Datei", "", [
        ("Speichern · Öffnen · Neu", "[Strg]+[S] · [O] · [N]"),
        ("Geometrie importieren (USD: Hierarchie wie in Solaris) · Sketch Brief exportieren", "[Strg]+[I] · [Strg]+[E]"),
        ("Referenzobjekt einfügen (Größenvergleich): Mensch 1,80 m, Auto, Einfamilienhaus", "Datei-Menü"),
        ("Brief-Auftrag beim Export: Auftrag in einem Satz, Hinweise, nur besprechen, Look-Vorstufe (den Rest klärt "
         "die KI aus den Notizen)", "[Strg]+[E]"),
        ("Screenshot als Hero-Kamera fürs Brief (Bild + USD-Kamera), Name darunter", "Haken oben rechts am Bild"),
        ("Bildunterschrift im Brett (wofür das Bild steht, worauf es ankommt – steht im Brief hinter dem Bild)",
         "Rechtsklick aufs Bild"),
        ("Brief mit der KI besprechen: Autoprompt nach dem Export in der Zwischenablage, Claude sieht die "
         "Szene (MCP)", "Haken im Brief-Dialog"),
        ("Bau-Brief: Arbeitsweise mit Regeln, Rollen und Hooks, project.md, Arbeitsordner work_<Name> daneben – "
         "Claude-Sitzung im Brief-Ordner starten; Stufe live ändern", "SET: level=scout|study|final"),
        ("Review-Runde: Feedback für den laufenden Chat – ohne Ordnerwahl nach <Brief>/review_rounds/, nur was sich an "
         "den Notizen geändert hat, keine neuen Regeln; den Satz im selben Chat einfügen", "„Baut auf“ im Brief-Dialog"),
        ("Arbeitsweise des Bau-Briefs wählen: Mule Default oder ein eigenes Recipe (Regeln, Rollen, Hooks, Texte); "
         "neues oder kopiertes Recipe anlegen, als .zip oder Ordner importieren, Inhalt ansehen", "„Brief-Recipe“ im Brief-Dialog · …"),
        ("Sichtkarte (Detail nur, wo man es sieht) in Blender, ein Befehl; Werkzeug prüfen",
         "tools/vis_map/vis_blender.py · --selftest"),
        ("Szene als USD exportieren (Flächen als Polymeshes, Meshes referenziert)", "[Strg]+[Shift]+[E]"),
        ("Startanimation beim Programmstart (Klick oder Taste überspringt)", "Menü Ansicht"),
        ("Arbeitsbereich zurücksetzen: Outliner, Notiz und Leisten wieder an ihrem Platz", "Menü Ansicht"),
        ("Aufnahme: jede Eingabe mit Zeit und Rechendauer mitschreiben, OBS startet auf Wunsch mit", "Ansicht → Aufnahme"),
        ("Review: durch die Szene gehen und Feedback einsprechen – Stimme, Kamera und Cursor werden aufgezeichnet, "
         "Claude macht daraus Entwurfs-Notizen", "Ansicht → Review"),
        ("Während des Reviews markieren: Klick in die Ansicht setzt eine Marke, Klick auf ein Bild im Brett merkt sich "
         "das Bild für die Notiz zum Teil", "Werkzeug Markieren · [Strg]+[Z] Marke zurück"),
        ("Review-Entwurf übernehmen bzw. die Stelle anhören (gestrichelte Notiz)", "Brett: ✓ Übernehmen · ▶ Anhören"),
        ("Claudes Pfeile folgen Text und Bild, wenn du sie im Brett verschiebst oder skalierst", "Brett"),
        ("Nur Kurven exportieren: USD für 3D-Programme, JSON für Skripte und KI-Chats, Dicke je Punkt", "Datei → Exportieren → Kurven"),
        ("Rückgängig · Wiederholen (auch Auswahl)", "[Strg]+[Z] · [Strg]+[Shift]+[Z]"),
    ])

DRAW = _section("Werkzeuge", "", [
    ("Stift (Grafit) ↔ Pinsel (Farbe, Öltextur)", "[A]"),
    ("Radierer – [Strg] nur Linien, [Strg]+[Shift] ganze Linien", "[D]"),
    ("Malmodus: Farbe flach auf die Fläche, keine editierbaren Linien", "[S]"),
    ("Aktive Fläche einfärben · zurück", "[F] · [Shift]+[F]"),
]) + \
    _section("Worauf", "Gezeichnet wird auf die aktive Ebene oder Fläche (getönt, kräftiger Rand).", [
        ("Fläche aktivieren", "Klick"),
        ("Grundebene nach Blickachse (Seite · Boden · Front) · letzte eigene", "[X] [Y] [Z] · [4]"),
        ("Neue Ebene aus der Ansicht", "[Space]+[N]"),
        ("Neues Stroke-Objekt (in der Gruppe, in der du arbeitest)", "[Space]+[B]"),
        ("Ebene verschieben", "Mausrad · &#91; &#93; · [Shift]+[RMB]"),
        ("Multi: auf mehrere Flächen zugleich", "[B]"),
        ("Frei: ohne Ebene, an Linien/Flächen ansetzen", "[Q]"),
        ("Bild auf Hilfsebene/-fläche: Bild laden, Project aus der Kamera, Lock sperrt", "Pfeil davor im Outliner"),
    ]) + \
    _section("Striche", "", [
        ("Am Ende stillhalten: Gerade, Kreis, Rechteck, Bogen", "½ s halten"),
        ("Gerade in 10°-Schritten", "[Strg]"),
        ("Einrasten an Punkten (Ringe) und Linien der Zeichenfläche (aus: Magnet-Knopf)", "automatisch"),
        ("Stiftgröße · Dicke aller Linien", "[Strg]+Mausrad · [Strg]+[Shift]+Mausrad"),
        ("Größte Strichbreite (Standard 50 m, bis 5 km)", "Rechtsklick → Maximale Strichbreite …"),
    ])

EDIT = _section("Wählen", "", [
    ("Punkte · Kanten · Flächen", "[1] [2] [3]"),
    ("Punkte des aktiven Objekts zeigen: Objekt wählen", "Klick auf Linie"),
    ("Ersetzen · dazu · weg", "Klick · [Shift] · [Strg]"),
    ("Rechteck (Verdecktes nur mit X-Ray)", "Ziehen"),
    ("Zu einem anderen Objekt springen (bearbeiten)", "Doppelklick darauf"),
    ("Loop · Fläche samt Linien (am bearbeiteten Objekt)", "Doppelklick"),
    ("Verbunden: im Objekt · überall · Loop nach Netzfluss", "[L] · [Shift]+[L] · [Strg]+[L]"),
    ("Punkte ziehen: Auto · Weich · Hart", "Leiste links"),
    ("Ausblenden · Isolieren · alles zeigen", "[H] · [Shift]+[H] · [Alt]+[H]"),
    ("Gewählte Flächen und Pinsellinien färben (Bleistift bleibt) · Färbung der Flächen zurücksetzen",
     "Farbwahl → Anwenden · [Shift]+Klick"),
    ("Bleistift ↔ Pinselstrich umwandeln", "Rechtsklick"),
]) + \
    _section("Bewegen", "", [
        ("Verschieben · Drehen · Skalieren", "[T] · [R] · [E]"),
        ("Einrasten (fein mit [Shift])", "[Strg]"),
        ("Auf Punkt einrasten und verbinden", "[V] halten"),
        ("Nur die Auswahl, Nachbarn bleiben", "[Alt] beim Ziehen"),
        ("Pivot verschieben · Achsen: Welt → Lokal → Kamera", "[Alt]+[A] · [Space]+[M]"),
    ]) + \
    _section("Punkte und Kanten", "Linien sind verbunden, wenn sie sich einen Punkt teilen (gefüllter Punkt); ein "
             "offener Ring ist ein freies Ende.", [
                 ("Punkt einfügen", "[Strg]+Klick"),
                 ("Punkte auf ihrer Linie gleiten lassen", "[S] halten + ziehen"),
                 ("<b>Härte</b> (an/aus): Ecke formen – Punkt: alle Linien knicken, Kante: die Querlinien knicken (Falte)", "[K]"),
                 ("Ganz hart · ganz weich", "[Shift]+[K] · [Alt]+[K]"),
                 ("<b>Glätten</b> (an/aus): anklicken, ziehen – voll geglättet ist die Kante gerade", "[J]"),
                 ("Auflösen (Form bleibt) · entfernen (glatt neu) · löschen (gerade Brücke)",
                  "[Entf] · [Shift]+[Entf] · [Strg]+[Entf]"),
                 ("Ecken abrunden ([F]: rund ↔ Fase)", "[Strg]+[B]"),
                 ("Dicke der gewählten Linien", "[Strg]+Mausrad"),
             ]) + \
    _section("Ordnen", "", [
        ("Kopieren · Einfügen · Duplizieren · wiederholen", "[Strg]+[C] · [V] · [D] · [Shift]+[D]"),
        ("In eine andere Szene oder ein zweites Mule-Fenster kopieren: über dem Outliner einfügen (Brett-Elemente samt Bildern)",
         "[Strg]+[C] · [V]"),
        ("Auswahl in neues Objekt – Flächen, deren Linien alle umziehen, ziehen mit um", "[Strg]+[G]"),
        ("Mehrere Linienobjekte zusammenlegen (ihre Flächen stehen danach unter dem einen)", "Objekte wählen, [Strg]+[G]"),
        ("Aufräumen: verschweißen, verbinden, Doppeltes", "Cleanup oben"),
    ]) + \
    _section("Gruppen", "Wie in Cinema 4D: ein Null sammelt Objekte im Outliner und bewegt sie mit.", [
        ("Null aus der Auswahl", "[Alt]+[G]"),
        ("Gruppe auflösen (Objekte bleiben)", "[Shift]+[G]"),
        ("Umhängen: auf ein Null ziehen · zurück: auf eine Rubrik", "Outliner"),
        ("Linien + gewählte Fläche/Objekte in eine neue Gruppe (am Ort der Linien)", "[Strg]+[G]"),
        ("Farbe eines Nulls (gilt darunter bis zum nächsten Null mit Farbe)", "Rechtsklick → Farbe"),
        ("Alles darunter auf-/zuklappen", "[Shift]+Klick auf den Pfeil"),
        ("Suchen · Namensspalte breiter ziehen", "Feld über dem Outliner · Kopfzeile"),
        ("Verbindlichkeit fürs Brief (Null: alles darunter)", "Spalte „Brief“ im Outliner"),
        ("Nur das Null bewegen", "[Alt]+[A]"),
    ]) + \
    _section("Symmetrie und Array", "Generatoren: bearbeitet wird das Original, Spiegelung und Kopien folgen live "
             "(nur Anzeige).", [
        ("Symmetrie · Array aus der Auswahl (ohne: neues Stroke darin; übernimmt Lage des gewählten Nulls)", "Menü Objekte"),
        ("An/aus · Spiegelebene · behaltene Seite · Modus, Anzahl, Zufall", "Outliner, unter dem Generator"),
        ("Strich-Enden rasten an der Spiegelebene ein", "Magnet an"),
        ("Klick auf eine Kopie wählt ihren Generator", "Edit"),
        ("Pfad · Up-Linie (Entlang Linie): Linie wählen, übernehmen", "Outliner"),
        ("Konvertieren: echte Linien und Flächen (geschnitten, verschweißt)", "[Alt]+[C]"),
    ])

SURFACES = _section("Flächen bauen", "Flächen entstehen aus Linien und folgen ihnen; die Linien auf einer Fläche "
                    "formen sie.", [
                        ("Fläche aus gewählten Linien", "[F]"),
                        ("Loft · ringförmig · Kurvenfläche erzwingen", "[Shift]+[F] · [Strg]+[Shift]+[F] · [Alt]+[F]"),
                        ("Loft teilen", "[Alt]+[D]"),
                        ("Linie von Rand zu Rand: Netzlinie, die Fläche geht genau und rund durch sie", "automatisch"),
                        ("Extrudieren (Achse: [X] [Y] [Z] [N]) – Linie an jedem Punkt ab 5° Knick, ab 15° eigene Wände", "[X]"),
                        ("Fläche extrudieren: sie wandert nach oben", "Flächenmodus, [X]"),
                        ("Fläche bewegen: alle ihre Linien und ihre Malerei wandern mit", "Fläche wählen, Gizmo"),
                    ]) + \
    _section("Aussehen", "Flächen decken wie echte Flächen; ihr Material ist nur Darstellung.", [
        ("Material", "Klick in die Material-Spalte im Outliner"),
        ("Hilfsfläche (eingefroren, nur Gizmo bewegt sie) · Transparenz · Auflösung", "Pfeil vor der Fläche"),
        ("Material-Maßstab", "Rechtsklick auf X-Ray"),
        ("Einfärben · zurück", "Zeichnen: [F] · [Shift]+[F]"),
    ])

NOTES = _section("Notizen", "Kurze Überschriften im Raum – Prompts für später. Immer aufrecht.", [
    ("Neue Notiz: an einer Linie oder im Leeren", "Doppelklick"),
    ("Umbenennen · drehen · löschen", "[F2] · [R] · [Entf]"),
    ("Verschieben ([Shift]: senkrecht)", "Ziehen"),
    ("Weitere Hinweislinie: vom Ring unten links ziehen", "Ziehen"),
    ("Größe, mit Kamera drehen, lösen", "Rechtsklick"),
    ("Krake: Ankerpunkt mit Krake (magenta mit Ring, an der gewählten Notiz) anklicken – seine Arme erscheinen (nochmal: aus)", "Klick"),
    ("Krake: vom angeklickten Ankerpunkt auf gleichartige Teile ziehen", "[Shift] + Ziehen"),
    ("Krake: Teil im Viewport dazu / wieder heraus (Notiz bzw. Ankerpunkt gewählt)", "[Shift] + Klick"),
    ("Krake: die ganze Gruppe (Null) des angeklickten Teils dazu", "[Strg] + [Shift] + Klick"),
    ("Krake: gewählte Objekte/Linien an den Anker · Arm entfernen · leeren", "Rechtsklick · [Entf] am Ring"),
    ("Gewählte Notiz: verbundene Teile leuchten magenta · nur die eines Ankers ([Esc]: alle)", "Klick auf Ankerpunkt"),
    ("Von Flächen Verdecktes fehlt (Modus Notizen: blass, X-Ray: alles)", "–"),
]) + \
    _section("Notizbrett", "Text und Bilder zur Notiz (Lokal) oder zur Szene (Global), Panel unter dem Outliner.", [
        ("Groß öffnen", "⤢"),
        ("Anheften: Brett bleibt in jedem Modus sichtbar (Referenz beim Zeichnen, Malen, Edit)", "📌 · Menü Ansicht"),
        ("Ansicht als Bild einfügen (an der Notiz oder im Brett) · zurückfliegen", "Kamera · Doppelklick aufs Bild"),
        ("Malen: Stift · Radierer · aus (Farbe: Punkte oben im Brett)", "[A] · [D] · [Esc] / ✎"),
        ("Pinselbreite im Brett (gleich bei jedem Zoom; Strg+Mausrad auf Bild/Text ohne Malen: skalieren)", "[Strg] + Mausrad / rechte Maustaste ziehen"),
        ("Dicke der Striche im Brett (gewählte, sonst alle)", "[Strg] + [Shift] + Mausrad / rechte Maustaste ziehen"),
        ("Pfeil von Beschriftung auf Bild: Verbindung leuchtet kurz nach dem Absetzen auf (gewählter Strich: dauerhaft)", "Stift"),
        ("Backdrop: Rahmen mit Titel um die Auswahl – am Reiter verschieben (Inhalt wandert mit), im Brief ein Abschnitt", "[B] · ▭"),
        ("Auto-Backdrops: Rahmen um alles, was nah beieinander liegt", "[Shift]+[B] · Rechtsklick ▭"),
        ("Gerade / Kreis / Rechteck / Bogen: am Strichende stillhalten", "halten"),
        ("Ganze Striche radieren", "[Strg]+[Shift] + Radierer"),
        ("Bild/Text einfügen (auch Bild aus dem Browser) · Dateien hineinziehen", "[Strg]+[V] · Drag & Drop"),
        ("Gewählte Elemente von Brett zu Brett: kopieren · ausschneiden · einfügen (Global ↔ Lokal)",
         "[Strg]+[C] · [X] · [V]"),
        ("Video hineinziehen (Pfeile und Unterschriften wie bei Bildern): Knopf/[Leertaste] spielt, Zeitleiste ziehen · Rechtsklick: Standbild", "Drag & Drop"),
        ("Bild/Text skalieren (Bilder liegen immer unter den Strichen)", "Ecke ziehen · [Strg]+Mausrad"),
        ("Screenshots neu aufnehmen (fehlende, sonst alle; gleiche Kamera und Format, Darstellung wie eingestellt)", "↻ (oben im Brett)"),
        ("Einen Screenshot neu aufnehmen (Shift: aus der jetzigen Ansicht)", "↻ oben rechts am Bild"),
        ("Rückgängig · Wiederholen (gemeinsam mit dem 3D-View)", "[Strg]+[Z] · [Strg]+[Y]"),
        ("Zoomen · verschieben", "Mausrad / [Alt]+[RMB] · [MMB]"),
    ])

TIMING = _section("Shots (Prototyp)", "Zeitliche Abläufe zeigen, ohne zu animieren: Zeichnungen gelten, bis die nächste kommt, Kameras springen von Stellung zu Stellung. Shot-Notizen funktionieren wie Notizen – mit eigenem Panel Shots (Lokal/Global).", [
    ("Flipbook-Ebene anlegen (nur in Shots; nur hier bemalbar)", "[Space]+[N]"),
    ("Flipbook-Ebene samt allen Zeichnungen bewegen · drehen · skalieren (Gizmo; kein Mausrad) · Gizmo weg", "[T] [R] [E] · [Esc]"),
    ("Frame · erster/letzter · nächste/vorige Zeichnung", "[←] [→] · [Shift]+[←] [→] · [↑] [↓]"),
    ("Malen/Radieren bei einem Frame: in seiner Zeichnung – gibt es keine, wird die geltende kopiert (weitermalen)", "Stift / Pinsel / Radierer"),
    ("Leere Zeichnung · Zeichnung löschen", "Timeline"),
    ("Zeichnung verschieben (Timing ändern) · löschen", "Raute ziehen · Rechtsklick"),
    ("Onion-Skin (vorige Zeichnung 30 %, nächste angedeutet) · abspielen (jeder Frame)", "[O] · [Alt]+[A]"),
    ("Shot-Notiz anlegen (an einer Zeichnung: daran angeheftet) · anklicken: ihr Brett im Panel Shots", "Doppelklick · Klick"),
    ("Shot-Notiz verschieben · Hinweislinie zu weiteren Ebenen · löschen", "ziehen · am Ring ziehen · [Entf]"),
    ("Globale Shot-Notiz (für alle Shots)", "Panel Shots → Global"),
    ("Neue Kamera aus der Ansicht (ihr Bild kommt ins Brett)", "🎥 im Brett / an der Notiz"),
    ("Kamera wählen · durchschauen · verlassen", "Klick aufs Symbol · Doppelklick (Symbol, Bild) / 👁 · [Esc]"),
    ("Jede Stellung ist ein Bild im Brett, darunter ihre Einstellungen: Farbe, Name, Frame, Brennweite", "Leiste unterm Bild"),
    ("In der Kamera: Stellung ändern (keine Fahrt) · neue Stellung ab hier (Fahrt, neues Bild)", "⟲ · ➜"),
    ("Brennweite der Stellung (hält den Ausschnitt; [Alt]: reiner Zoom) · Stellung löschen", "Regler · Bild löschen"),
    ("Frame der Stellung · Symbolgröße · Bild neu aufnehmen", "Rechtsklick auf „ab F…“ · Rechtsklick aufs Bild · ↻"),
    ("Export: Kameras gestuft oder linear (Brief- und USD-Dialog), im Brief der Teil Shots mit Bildern und Brettern", "[Strg]+[E] · [Strg]+[Shift]+[E]"),
])

VIEW = _section("Oben rechts", "", [
    ("Zeichenhilfen (aus: Ergebnis roh, ohne Ebenen und Hilfsflächen)", "Zeichenhilfen-Icon"),
    ("X-Ray – Meshes mit allen Schichten (Deckkraft: Rechtsklick)", "X-Ray-Icon"),
    ("Drahtgitter im X-Ray dazuschalten – alle Kanten (Deckkraft: Rechtsklick)", "Kugel-Icon"),
    ("Linien, Farblinien, Malerei, Flächen einzeln ausblenden", "Ein-/Ausblenden-Icon"),
    ("Farbansicht ↔ Schwarz-Weiß · Color Compress", "Icon · Ansicht → Color Compress"),
    ("Auflösung aller Flächen (Viewport-Qualität)", "Regler oben (Netz-Icon)"),
    ("Kreidetafel ↔ Papier · Bleistift- und Kreidefarbe", "Mond-Icon · Ansicht-Menü"),
    ("Unten im Viewport: aktives Werkzeug und Schalter", "–"),
    ("Sprache der Oberfläche (Deutsch/English, gilt nach dem Neustart)", "Ansicht → Sprache / Language"),
    ("Wo Mule Daten, Einstellungen und Sicherungen ablegt – mit Größe; Mule ganz entfernen", "Hilfe → Wo Mule Dateien ablegt …"),
]) + \
    _section("Licht", "Eine Sonne, reine Darstellung – Linien, Flächen und importierte Meshes werfen und empfangen Schatten.", [
        ("Licht an/aus", "Knopf unter dem Achsen-Gizmo"),
        ("Sonne setzen: Himmelsscheibe unten rechts", "Knopf darunter"),
        ("Licht-Stufen (3 … fließend; je weniger Stufen, desto stärker werden kleine Wellen geglättet)", "Rechtsklick aufs Licht-Icon"),
    ])

KEYS = _section("Überall", "", [
    ("Zeichnen · Edit · Notizen · Handbuch", "[A] [D] · [Tab] · [N] · [F1]"),
    ("Kamera · Ansichten", "[Alt]/[Space] + Maus · [Space]+[1] … [4]"),
    ("Rückgängig · Speichern · Brief", "[Strg]+[Z] · [Strg]+[S] · [Strg]+[E]"),
]) + _section("Zeichnen", "", [
    ("Pinsel · Radierer · Malmodus", "[A] · [D] · [S]"),
    ("Ebenen · neue Ebene · Multi · Frei", "[X] [Y] [Z] [4] · [Space]+[N] · [B] · [Q]"),
    ("Einfärben", "[F]"),
]) + _section("Edit", "", [
    ("Punkte · Kanten · Flächen", "[1] [2] [3]"),
    ("Verschieben · Drehen · Skalieren", "[T] [R] [E]"),
    ("Fläche · Loft · Kurvenfläche · Extrudieren", "[F] · [Shift]+[F] · [Alt]+[F] · [X]"),
    ("Härte · Glätten · Gleiten", "[K] · [J] · [S] halten"),
    ("Abrunden · Loft teilen · neues Objekt · verbunden", "[Strg]+[B] · [Alt]+[D] · [Strg]+[G] · [L]"),
    ("Ausblenden · Isolieren · alles zeigen", "[H] · [Shift]+[H] · [Alt]+[H]"),
])

TABS = (("Start", START), ("Zeichnen", DRAW), ("Edit", EDIT), ("Flächen", SURFACES), ("Notizen", NOTES),
        ("Shots", TIMING), ("Ansicht", VIEW), ("Tasten", KEYS))

STYLE = f"""
QTabWidget::pane {{ border: none; border-top: 1px solid {ui.LINE}; }}
QTabBar::tab {{ background: transparent; color: {ui.TEXT_DIM}; padding: 7px 16px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {ui.TEXT}; }}
QTabBar::tab:selected {{ color: {ui.ACCENT}; border-bottom: 2px solid {ui.ACCENT}; }}
QTextBrowser {{ background: {ui.PANEL}; border: none; padding: 14px 22px; }}
"""


def _tabs():
    """Tabs in the UI language."""
    if i18n.LANG == "en":
        from .manual_en import TABS as english
        return english
    return TABS


class ManualWindow(QWidget):
    """Own window next to the drawing (not modal: one can keep working)."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle(tr("Mule Create – Handbuch"))
        self.setStyleSheet(STYLE)
        self.resize(720, 620)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        # first thing in the manual: the introduction in pictures and the way to the AI handbook
        top = QHBoxLayout()
        top.setContentsMargins(10, 8, 10, 4)
        self.intro_button = QPushButton(tr("▶  Einführung in Bildern"))
        self.intro_button.setStyleSheet(f"QPushButton {{ background: {ui.ACCENT}; color: {ui.BG}; font-weight: 600; "
                                        f"border: 0; border-radius: 5px; padding: 7px 16px; }}")
        self.intro_button.clicked.connect(self._open_intro)  # bound methods, no lambda holding self: no cycle
        self.ai_button = QPushButton(tr("Mit der eigenen KI fragen (AI_HANDBOOK.md)"))
        self.ai_button.clicked.connect(self._open_ai_handbook)
        top.addWidget(self.intro_button)
        top.addWidget(self.ai_button)
        top.addStretch(1)
        lay.addLayout(top)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for title, html in _tabs():
            page = QTextBrowser()
            page.document().setDefaultStyleSheet(CSS)
            page.setHtml(f"<body>{html}</body>")
            self.tabs.addTab(page, title.replace("&", "&&"))  # & would otherwise be swallowed as a shortcut marker
        lay.addWidget(self.tabs)

    def _open_intro(self):
        from . import intro
        intro.show_intro(self.parent() or self)

    def _open_ai_handbook(self):
        from . import intro
        intro.show_ai_handbook(self)

    def open_at(self, title):
        """Show and jump to tab title (e.g. that of the current mode)."""
        names = [t for t, _ in _tabs()]
        if title in names:
            self.tabs.setCurrentIndex(names.index(title))
        self.show()
        self.raise_()
        self.activateWindow()
