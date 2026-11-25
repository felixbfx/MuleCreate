# Image brief – {{project_name}}
Recipe {{recipe.name}} {{recipe.version}}. Target pictures for the look of this scene, made with an image model before
the detailed 3D work. Nothing is built in 3D in this step.

{{#task_notes}}
The artist about this round:
{{task_notes_quoted}}

{{/task_notes}}
## Who answers what
The sketch decides where everything is and how big it is: position, size, outline, layout.
The pictures decide only how it looks: materials, how things are built, how dense the detail is, wear, light, atmosphere.
A picture that moves, adds, removes or resizes a form is wrong, however good it looks.

## What matters in this folder
| | |
|---|---|
| `views/hero_*.png` | the structure images: one per hero camera, binding for everything spatial |
| `image/shots.md` | per camera: what is in the view, real sizes, which wide view a close view belongs to |
| `image/look.txt` | the look paragraph; goes word for word into every prompt |
| `image/build_prompts.py` | writes `image/prompts/`, `image/inputs/` and `image/STEPS.md` from the three above |
| `image/STEPS.md` | the steps: which prompt, which pictures attached in which order, where to save |
| `gen/` | results · `image/approved/` – what the artist picked |
| `brief.md` | only the artist's notes: the source for the look paragraph and the lines per camera |

Not needed in this step: `scene.usd`, the part lists and sizes in `brief.md`, the front, side and top views,
the id pictures. `brief.json` is read by the script only.

## Hero cameras
{{#cameras}}
- `{{name}}` – `{{image}}`
{{/cameras}}
{{^cameras_set}}
The artist set no hero cameras. Ask for them before anything else: this step works from hero cameras only.
{{/cameras_set}}

## Order
1. Look paragraph and lines per camera (`image/look.txt`, `image/shots.md`) – the artist confirms them.
2. `python image/build_prompts.py` – then follow `image/STEPS.md`: mood probe, close views, wide views, further views.
3. Two or three variants per step, never more before the artist has looked. The artist picks; the pick goes to
   `image/approved/` under the name in `STEPS.md`.

## After approval
The approved pictures stand for appearance only. Position, size and outline stay with the sketch.
