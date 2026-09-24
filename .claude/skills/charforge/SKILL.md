---
name: charforge
description: Build, check and repair game-ready 3D characters with charforge.py (text or image in; rigged, animated glTF/FBX out). Use when making a character, rebuilding one after a code change, judging a build's QA outputs, or diagnosing a bad build - which stage to rerun and what to look at.
---

# CharForge runbook

`charforge.py make` turns a prompt or an image into a character a game engine can use: a mesh with
modelled hands, a Mixamo skeleton with fingers, a face rig (jaw, mouth, blinks, smile, brows,
pucker), spring chains for what hangs free, 18 locomotion and gesture clips, PBR textures, LODs
and a manifest. This file is for whoever drives it - a person or a model. It says what the
program decides by itself, where judgement is still needed, what each check should look like,
and what to do when it does not.

Everything below was learned by building characters by hand in Blender and then turning each
decision into a rule the program applies. When a rule fails on a new character, fix the rule
(and write down the measurement that shows it), not the character.

## Run it

```bash
.venv/bin/python charforge.py make --name mara --style realistic --height 1.68 \
    --prompt "a young woman field researcher with a long dark brown braided ponytail, ..."
.venv/bin/python charforge.py make --name vex --image concept.png --style stylized --height 1.72
.venv/bin/python charforge.py make --name mara --from rig        # rerun from a stage onward
.venv/bin/python charforge.py stages                            # every stage, what it writes
```

- `--style` is `realistic`, `anime` or `stylized`. It changes the prompts, the face finder's
  expectations (drawn faces have bigger eyes and faint mouths) and the shading the manifest asks
  for. Style, height and prompt are recorded in `work/<name>/style.json` on the first run; later
  runs reuse them, so `--from <stage>` never needs them again.
- Every stage writes into `work/<name>/` and is skipped when its output exists. After changing a
  script, rerun from the first stage that uses it (table below). Logs: `work/<name>/logs/<stage>.log`.
- One heavy GPU job at a time. A TRELLIS pass, a ComfyUI job and Blender's GPU renders sharing
  the GPU get the TRELLIS pass killed by macOS (`Impacting Interactivity`); rerun with `--from`.
- About 10-15 minutes per character on an M-series Mac after generation (generation: ~25).

| stage | script | rerun from here after changing |
|---|---|---|
| reference, generate, multiview | `pipeline/comfy.py`, `vendor/trellis2mlx` | the prompt or image |
| views, parts | `blender/render_views.py`, `pipeline/parts.py` | part labels |
| skeleton | `pipeline/skeleton.py` | joint estimation |
| solidify | `blender/sdf_io.py`, `pipeline/solidify.py` | the solid |
| joints | `pipeline/refine_joints.py` | limb tracing, wrists, elbows |
| hands | `pipeline/cut_hands.py`, `blender/hands.py`, `blender/hand_model.py` | hands |
| retopo, labels, texclean | `blender/retopo.py`, `pipeline/transfer_labels.py`, `pipeline/texture_cleanup.py` | mesh, bakes |
| texture | `blender/uv_maps.py`, `pipeline/face_detail.py`, `pipeline/project_texture.py` | texture |
| weights, rig | `pipeline/geodesic_weights.py`, `blender/rig_build.py` | skinning |
| springs, frame, tpose | `blender/springs.py`, `blender/normalize_frame.py`, `blender/tpose.py` | chains, rest pose |
| face | `blender/face_render.py`, `pipeline/face_landmarks.py`, `blender/face_rig.py` | face rig |
| animate, package, web | `blender/retarget.py`, `blender/package.py`, `tools/optimize_glb.py` | clips, export |

## What the program decides by itself

These were judgement calls in the hand-built characters. Each is now a rule with a measurement
behind it (the code comment next to each rule has the numbers).

- **Inside and outside** of a leaky generated mesh: a voxel is outside if it sees out along at
  least 3 of 26 directions (OpenVDB's sign is unreliable on these meshes).
- **Hair**: closed into a solid only where hair is the nearest part, so bangs never veil the
  eyes; hair the parser calls "bag" or "hat" is relabelled by colour when it is joined to the
  scalp's hair.
- **Joints**: every limb is traced through the solid to its tip; a joint's depth comes from the
  solid, not the side view; the wrist is where the palm rounds into the wrist or the hand enters
  its sleeve; the elbow is re-placed by proportion when the upper arm and forearm disagree; the
  body's midline comes from the legs. Both hands get one length.
- **Skinning**: weights are measured through the body (geodesic), never straight-line; the head
  is rigid down to the jaw line (measured from the face's profile); shoes are rigid below the
  ankle; each leg below mid-thigh binds to its own bones; accessories below the neck ride the
  torso.
- **Springs**: a hanging part gets a chain only if it hangs free. A part fused to the body along
  its length stays rigid, because a chain on it tears the surface.
- **Face**: landmarks are found chin first, then nose, then mouth; the mouth is cut open along
  the lip seam with a mouth pouch behind it; shapes rest at zero. On drawn styles an eye is the
  largest blob that is not the hair's colour. A landmark that fails its order-and-size checks
  switches its shapes off instead of putting them in the wrong place; on a drawn face with no
  eye found, the mouth is switched off too.
- **Texture**: source views are projected only if their silhouette overlaps the mesh's by
  at least 0.75 IoU; the face is read from a detailed close-up made by image-to-image at
  strength 0.28 (higher changes the face); the finer face flow is used only when it is a small
  correction (under 2.5% of the figure's height - realistic faces), never on a face the
  generator drew differently from the reference.
- **Clips**: heading from travel, feet planted with IK on contacts read from the sole, 60 fps,
  in place with the ground speed in the manifest.

## Where judgement is still needed - check these, in this order

Open each image; read each log line. Each check says what good looks like and what to do if not.

1. **`work/<name>/reference.png`** - one full-body figure, A-pose, arms clear of the body, feet
   visible, plain background. *If not:* regenerate with another `--seed`, or give an image. A
   busy background is survivable (`reference_mask.png` cuts it out) but expect softer texture.

2. **`work/<name>/qa/skeleton_front.png`, `skeleton_side.png`** - the yellow traces run down the
   middle of each arm to the fingertips and each leg toward the sole; orange dots: elbow
   mid-arm, wrist at the base of the hand, knee mid-leg, ankle just above the shoe, neck and
   head on the midline. Blue dots are the pose model's raw guesses (often wrong on stylised
   bodies - that is expected). *If a trace wanders* (into a ponytail, into a torso), look at
   `logs/joints.log`: "put inside its limb", "moved where the palm rounds", "trace ended ... the
   estimate kept" say which rule acted. Fix the rule in `refine_joints.py` and rerun from
   `joints`.

3. **`logs/hands.log`** - "hand length X cm (measured a / b)": one length for both, and the two
   measurements should roughly agree. A wrist wider than ~8 cm (at 1.75 m) means the cut went
   through a palm or a puffy cuff: check the wrist in step 2.

4. **`logs/texture.log`** - each view's "silhouette IoU" at or above 0.75 (below, the view is
   skipped and that side keeps the softer bake); "face: N texels ... read from the detailed
   face"; "face flow left out" on drawn characters is expected. *To judge the face*, render the head (the playground zooms in) - the detailed face
   should look like the reference, not a different person.

5. **`logs/rig.log`** - "head rigid above the jaw line (X cm below the ears ..., chin measured
   from the face's profile)": 8-10 cm on the characters so far. "from the ear-to-crown distance"
   means the profile could not be read (hair over the face, a mask) - check the jaw by nodding
   the head in the playground.

6. **`work/<name>/face_qa.png`** - green ellipses on the eyes, red dots on the mouth corners and
   the line between the lips, blue on the nose, yellow on the chin. `logs/face_landmarks.log`
   says which parts were not trusted ("mouth not trusted ... no mouth cut or jaw", "no blinks").
   A gated part is safe - the character just lacks it. *If a mark is wrong but trusted*, fix
   `face_landmarks.py` and rerun from `face`; never let a wrong mouth cut ship.

7. **`logs/springs.log`** - chains built, or "left rigid" with the reason. A braid or bag fused
   to the body is expected to stay rigid.

8. **The playground** (`python tools/build_site.py`, then open `docs/index.html` through a local
   server) - walk, run, strafe, crouch, jump, turn, wave; hold T to talk; watch the head turn
   with the face, the blink, the feet. In the browser console, `await __cf.footAudit()` measures
   foot slip per gait (around 2-5% is the current standard).

## Motion from a video

`pipeline/video_motion.py --video clip.mp4 --library work/motion_library.npz` finds the library
clip whose 2D projection follows the person in a video (retrieval, not reconstruction: it
returns clean motion capture the pipeline already retargets). The library is built once with
`blender/motion_library.py`. `--selftest 40` measures it (see README). To give a character the
motion: `--add-to <name> --as <clip>` records it in `work/<name>/extra_clips.json`, then
`charforge.py make --name <name> --from animate` retargets it with the rest. Check the match
first: the top few lines of its output are near-ties when the library holds duplicates, and a
cost above ~0.3 means nothing in the library moves like the video. Generating the video itself
needs a video model, which is not installed.

## Rules for changing the pipeline

- Measure before and after, on more than one character. Every number in the README comes from
  a script in this repository; keep it that way.
- A gate beats a guess: when a rule cannot be sure, make it switch a feature off and say so in
  the log, rather than ship it wrong.
- Write the measurement that motivated a rule into the comment above it, with the character it
  was found on.
