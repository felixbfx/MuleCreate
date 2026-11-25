# Mule Image – how to work with this brief

This brief is for look pictures from the hero cameras. You build nothing in 3D and you change nothing in the scene.
Read `project.md` first. Talk to the artist in the artist's language; prompts and files stay English.

## What you read, and what you leave closed
- Read: `project.md`, the artist's notes in `brief.md` (project note, notes on parts, captions), the pictures
  `views/hero_*.png` of the cameras that take part, the artist's reference pictures in `notes/`.
- Leave closed: `scene.usd`, `brief.json` (the script reads it), part lists, id pictures, front / side / top views.
  They cost context and add nothing to a picture prompt.

## Step 1 – the texts, with the artist
1. `image/look.txt`: one paragraph, built only from the artist's notes: place and time, what the thing is for,
   how it is made and of what, colours, wear, how detail is spread, light and weather, the kind of picture.
   No shapes and no positions – the structure image has those. Where the notes leave light or weather open,
   say so and propose one; do not decide silently.
2. `image/shots.md`: which cameras take part, which close view belongs to which wide view, and per camera what the
   forms in its structure image are, front to back. The format is at the top of that file. Name only what the
   artist drew or wrote. Where you are not sure what a form is, ask with the picture; do not guess in the prompt.
   Two or three real sizes help more than anything else – ask for them if the notes have none.
3. Show both to the artist and wait for an OK.

## Step 2 – build
Run `python image/build_prompts.py` in the brief folder (it needs Python; with PySide6 it also draws the
location pictures). It writes `image/prompts/`, `image/inputs/` and `image/STEPS.md`. Run it again after every
change to `look.txt` or `shots.md`; never edit a prompt file by hand.

## Step 3 – pictures
Follow `image/STEPS.md` from top to bottom.
- With an image tool: `python image/fal_image.py --prompt-file <prompt> --image <first> --image <second> --out <save as>`
  (needs `FAL_KEY` in the environment; the artist sets it – never ask for it in the chat, never print it).
  It costs money per picture: say the estimate before the first call and stay inside the budget the artist named.
- Without one: the artist does the steps by hand in a chat; you wait for the pictures in `gen/`.
- The structure image is always the first image. At most three input images. Hero views only.
- Two or three variants per step, then stop and let the artist pick. Copy the pick to `image/approved/` under the
  name `STEPS.md` gives. A wide view starts only when its close views are approved.

## Before you show a picture
Lay the structure image over it (or compare side by side): every large outline still where it was? Nothing added,
removed or merged? Say plainly what moved. A picture that changed the forms is not offered as a candidate.

## What you never do
Build or change 3D. Call an approved picture a target for form. Send pictures or texts anywhere but the image tool
the artist agreed to.
