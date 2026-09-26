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
.venv/bin/python charforge.py studio                            # the same in a local web app, :8830
```

The Studio (`studio/`) queues these same commands one at a time, shows each stage with its time
left (learned from its own logs), keeps every log in `work/_studio/`, and serves a playground with
every local character at `/play/`. It can be restarted while a job runs: it finds the job's process
again and waits for it. Drive either; do not run a pipeline job beside one the Studio is running.

- `--style` is `realistic`, `anime` or `stylized`. It changes the prompts, the face finder's
  expectations (drawn faces have bigger eyes and faint mouths) and the shading the manifest asks
  for. Style, height and prompt are recorded in `work/<name>/style.json` on the first run; later
  runs reuse them, so `--from <stage>` never needs them again.
- Every stage writes into `work/<name>/` and is skipped when its output exists. After changing a
  script, rerun from the first stage that uses it (table below). Logs: `work/<name>/logs/<stage>.log`.
- One heavy GPU job at a time. A TRELLIS pass, a ComfyUI job and Blender's GPU renders sharing
  the GPU get the TRELLIS pass killed by macOS (`Impacting Interactivity`). The pipeline now
  evaluates TRELLIS two transformer blocks at a time and retries a killed pass once in smaller
  pieces; if it still dies, rerun with `--from`.
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
| animate | `blender/retarget.py` | clips |
| refine | `blender/export_skin.py`, `pipeline/refine_pose.py`, `blender/apply_pose_corrections.py` | clips made from video |
| package, web | `blender/package.py`, `tools/optimize_glb.py` | export |

## What the program decides by itself

These were judgement calls in the hand-built characters. Each is now a rule with a measurement
behind it (the code comment next to each rule has the numbers).

- **The reference's arms**: a new reference is read by a pose model (`pipeline/pose_gate.py`) and
  drawn again on a new seed, up to three draws, while its hands are under 0.45 torso lengths from
  the hips - they would fuse to the body (Bo's fists on his hips read 0.33; good references
  0.60-0.79). A picture you give is only warned about.
- **Inside and outside** of a leaky generated mesh: a voxel is outside if it sees out along at
  least 3 of 26 directions (OpenVDB's sign is unreliable on these meshes).
- **Hair**: closed into a solid only where hair is the nearest part, so bangs never veil the
  eyes; hair the parser calls "bag" or "hat" is relabelled by colour when it is joined to the
  scalp's hair.
- **Joints**: every limb is traced through the solid to its tip; a joint's depth comes from the
  solid, not the side view; the wrist is where the palm rounds into the wrist or the hand enters
  its sleeve; the elbow is re-placed by proportion when the upper arm and forearm disagree; the
  body's midline comes from the legs. Both hands get one length.
- **Arms glued to the body**: cut free below the armpit (`free_arms.py`) - first the thin webs the
  generator stretches between arm and body (solid under ~9 mm, on the body's side of the arm) - then
  checked slab by slab straight across the arm, and cut again, wider and further out, where the arm
  still reaches the body; pieces the cuts split off go. `logs/arms.log` says how much web went, how
  many stretches were re-cut and how many stay joined.
- **Loose pieces** (a sole the generator left detached, a hair spike) ride rigidly with the body part
  nearest each; the modelled hands keep their own rule.
- **UVs**: islands the unwrap folded onto themselves are unwrapped again, finer, and every island
  brought to one texel density (`retopo.py`): overlapping faces go from hundreds or thousands to a
  few dozen at most.
- **Skinning**: weights are measured through the body (geodesic), never straight-line; below the
  armpit the arm's weight stays on the arm's own surface (flooded from 0.30 of the upper arm, below
  where the cut opens); where the colours tell a vest's armhole from its sleeve it moves with the
  collarbone, and where they tell a jacket from the trousers its hem follows the pelvis, not the
  thighs; the head
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
  generator drew differently from the reference. A generated side or back view paints nothing
  right beside the outline of a nearer surface (its hands stand off the mesh's by 10-20 px); the
  modelled hands are only what stands outside the body, in the face's skin tone; surfaces the arm
  cut opened take the colour of the nearest old surface along the mesh.
- **Clips**: heading from travel, feet planted with IK on contacts read from the sole, 60 fps,
  in place with the ground speed in the manifest.
- **Moves from a video**: the nearest capture is bent to follow the video only where it misses
  by more than 1.2x the pose model's jitter (on the self-test, captures the library holds leave
  0.66-0.84 jitters unexplained, captures it lacks mostly 1.5-6). A bent bone takes its image
  from the video and its depth from its length - the video's own length for it, its longest in the
  clip (90th percentile), with the capture's stretch from frame to frame; the camera angle is
  refined to the degree first. The head turns to the video's nose, eyes and ears when the
  performer is calibrated (`tools/calibrate_joints.py`: its joints and face as the pose model
  sees them, read off front and side renders of its rest pose - never trust its arm readings on a
  T-pose: it took Aoi's cuffs for wrists). The retarget aims limbs, spine and head along the fit,
  and the refine stage fits the character's own mesh to the video's silhouette.

## Where judgement is still needed - check these, in this order

Open each image; read each log line. Each check says what good looks like and what to do if not.

0. **"described (<model>): <age>, <height> m - <sentence>"** (reference stage, a short prompt): the
   prompt written out by `pipeline/describe.py`. It should keep everything the prompt said, give
   the character clothes with a colour each and an age that fits (a boy, a girl, a scout: a
   child). *If not:* rerun with `--description "..."` (your own sentence), `--literal` (the prompt
   as typed), or another model (`CF_DESCRIBE_MODEL`; the Setup page shows which one is used).
   The sentence is recorded in `work/<name>/style.json` and reused by reruns.

1. **`work/<name>/reference.png`** - one full-body figure, A-pose, arms clear of the body, feet
   visible, plain background. The reference stage prints "hands N torso lengths from the hips" for
   each draw; a redraw means the first picture had the hands against the body. Made by Qwen-Image 2.1 by default: `reference_rgba.png` is its own
   cut-out (what TRELLIS is given) and `reference_mask.png` comes from its alpha - check the mask
   has no holes and no halo. *If not:* regenerate with another `--seed`, or give an image, or
   `--image-model krea2` (and for any commercial character: Qwen's licence is research-only). A
   busy background is survivable (`reference_mask.png` cuts it out) but expect softer texture.

1b. **"front silhouette on the reference: IoU X (route, seed S)"** (generate, and once more after
   multiview): 0.90-0.94 is a sound model. Under 0.8 TRELLIS built the picture's backdrop in as a
   board (Gray, Knight, Kaito: 0.35-0.44), and the stage makes it again by itself - through
   TRELLIS's background remover, then on a new seed. "No try matches ... keeping the closest"
   means all three failed: look at `work/<name>/gate/*/view_00.png` and try another `--seed`, or
   redraw the reference. `work/<name>/generate.json` records the seed and route that were kept.

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
   through a palm or a puffy cuff: check the wrist in step 2. **`logs/arms.log`** - where each arm
   was glued, and "still joined" if a join survived the re-cut (then the rig keeps that side of the
   jacket with the arm). **`logs/retopo.log`** - "UVs: N faces overlapping after the unwrap, M after
   unwrapping those again": M in the tens; hundreds means shared texels - patches of one garment's
   paint on another.

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

8. **The poses that break things** - render the extremes and look, front-left and back-right:
   `blender -b -noaudio --python blender/render_poses.py -- --blend work/<name>/final.blend --shots
   "jump@1.2,land@0.5,crouch_walk@0.4,wave@0.5,fall@1.0" --az 30,210 --res 360 --out audit/<name>`.
   A vest or jacket side lifting with the arm, a hem wrapped round the thighs, a pocket standing off
   the leg, paint of one garment on another: each has a rule above; find which one did not act.
   The Studio's Poses button renders the same set into `work/<name>/qa/poses/`, and
   `python tools/make_reel.py --name <name>` (the Reel button) films the clips back to back.

9. **The playground** (`python tools/build_site.py`, then open `docs/index.html` through a local
   server, or the Studio's `/play/`) - walk, run, strafe, crouch, jump, turn, wave; hold T to talk; watch the head turn
   with the face, the blink, the feet. In the browser console, `await __cf.footAudit()` measures
   foot slip per gait (around 2-5% is the current standard).

## A new move, from a video

```bash
.venv/bin/python charforge.py move --name wren --move punch_combo      # generate, read, rig, preview
.venv/bin/python charforge.py move --name wren --move punch_combo,spell_cast   # several; rigs once
.venv/bin/python charforge.py move --name wren --all                   # every move in motion_prompts.json
.venv/bin/python charforge.py move --name wren --move punch_combo --preview-only   # re-render the preview
.venv/bin/python charforge.py move --name aoi --move punch_combo --performer wren  # Wren's video, on Aoi
.venv/bin/python charforge.py move --name juno3 --move punch_combo@mara,spell_cast@aoi  # each from its own video
.venv/bin/python charforge.py move --name wren --move dance --video my_clip.mp4   # your own video
.venv/bin/python charforge.py move --name wren --move punch_combo --refit   # fit again (keeps the video)
.venv/bin/python pipeline/motion_video.py --moves                      # the moves it can ask for
```

The fitted clip is a Mixamo skeleton, so one video serves every character: `--performer` reuses
another character's video and fit (made first if missing). What runs, in order (the video and
fit are in `work/<performer>/motion_videos/`, the preview in `work/<name>/motion_videos/`):

1. **video** (`pipeline/motion_video.py`): the character's `reference.png` and the move's prompt
   from `animations/motion_prompts.json` go to MiniMax H3 (reference-to-video) in ComfyUI: 5 s,
   512 x 768, 4 steps with the turbo LoRA. The prompt is MiniMax's six-section format, with the
   move as timed beats (ready, the move, recovery) and the camera locked off on the whole body
   in a plain studio. About 25-35 min on an M5 with 24 GB; the text encoder runs on the CPU in a
   first job, ComfyUI unloads it, and the video model runs in a second.
2. **motion** (`pipeline/video_motion.py --fit`): the figure is cut out of every frame
   (`pipeline/foreground.py --video`, rembg - the pose model's boxes and the fidelity score's
   silhouettes; a colour threshold takes half of a studio backdrop for the figure); ViTPose reads
   17 points at 24 fps; the nearest of the 2,453 library captures is found (camera angle, mirror,
   timing); then, where that capture misses the video by more than the pose model's jitter, each
   bone is turned to follow the video **on the video's own body** - each bone as long as the video
   shows it at its longest (a fit on the capture's body bent Aoi's short torso into a 28 deg lean) -
   its image from the video, its depth from its length - up to a sign, which an arm's upper arm and
   forearm choose together: no elbow folded past 150-160 deg, no wrist behind the torso within the
   chest's width and height (Aoi's spell cast came out with an elbow folded to 176 deg on the
   capture's side; `CF_ARM_SIGNS=0` restores the old choice for comparison). The head is turned to the video's face
   (nose, eyes, ears) with the performer's own face points (`tools/calibrate_joints.py`, made on
   first use into `work/<name>/joint_calib.json`). `blender/fit_clip.py` writes a Mixamo FBX.
   DWPose (`tools/dwpose.py`, when `models/dwpose/dw-ll_ucoco_384.onnx` is there) reads the hands -
   21 points each, which ViTPose's 17 do not reach - into `<move>_dw.npz`, for the refine.
3. **rig**: `make --from animate` retargets the new clips with the rest; for a clip from a video
   `blender/retarget.py` then aims the character's own limbs, spine, neck and head along the fitted
   points (a rotation copy carries the two skeletons' rest-pose differences - thighs 4-12 deg,
   collarbones 16-27 - into every frame), before the feet are planted.
4. **refine** (stage `refine`, `pipeline/refine_pose.py`): each clip made from the character's own
   video is re-posed frame by frame - its skinned mesh (`blender/export_skin.py`) posed in PyTorch,
   projected by the video's camera, twenty bones turned a little until the silhouette lies on the
   video's figure, the joints on the pose model's and the face as the video's - and written back
   (`blender/apply_pose_corrections.py`). With DWPose's reading, the hands too: the wrists, the
   forearms about their length and the thirty finger joints (each only as a finger bends, within
   its range) turn until each hand's points about its wrist lie on DWPose's. The hands may not move
   the arm - that is the body's terms' to place; a hand pulling on it lifted Aoi's hands to her
   chin. No elbow may bend past 150 deg (`--w-elbows`): where the video shows an upper arm at full
   length the fit has no depth side left to choose, and the refine turns the arm instead. The rig's wrist point is first moved to where DWPose sees a wrist on this character
   (`tools/calibrate_hands.py`, once per rig, `work/<name>/hand_calib.json`) - on Pip's cartoon hand
   his sleeve's cuff, 4 cm up - as far as that holds across poses. A clip from another character's
   video is not refined (a different body cannot lie on that
   figure); a correction that does not help is not applied. `work/<name>/refine.json` has the
   silhouette overlap and the hands' error before and after.
5. **preview**: `<move>_compare.mp4`, the video and the character side by side, seen from the
   video's camera angle relative to the body's average heading (retarget.py turns a stationary
   clip to face its mean hip line). And from every side:
   `python tools/make_angles.py --name <n> --clip <move> --video` writes
   `work/<n>/qa/angles_<move>_video.mp4` - the video, its camera, the left side and above, all
   orthographic on a 10 cm grid, with each elbow's and knee's bend and each shoe's height read out
   under them.

A move is written once, in `animations/motion_prompts.json`: a one-line summary and timed beats
(ready, the move, recovery, the end state - MiniMax's prompt guides ask for action chains, and a
one-word action drifts). Each move costs about 35 minutes of video on the M5; the rest of the
chain about 4.

Check, in this order:

- **the angles video** (`make_angles.py --video`, or **Angles** in the Studio): the side and top
  panels show what the video's camera cannot - an arm folded back through itself, a foot off the
  floor, a hand behind the body. The readout goes red past 155 deg at an elbow, 165 at a knee, and
  for a shoe below the floor. "[apply] feet: the lowest sole was A..B cm off ... moved it to the
  height the video's figure stands above its floor line" (refine) says how far the feet were put
  back.
- **`<move>.mp4`**: one person, whole body in every frame, the camera still, the move as written.
  A cut, a zoom or a second person breaks what follows - regenerate with another `--seed`.
- **the match lines** (`<move>_match.json`, printed): a cost under ~0.3 is a capture that moves
  like the video; the top lines are often near-ties (Mixamo holds duplicates).
- **"turned to follow the video: image error A -> B"**: B should be near the jitter. "retimed
  only" means the capture already matched. "bones X deg from the fit's" is the Blender round trip:
  under 1 deg is normal; more means the solve or the FBX axes went wrong. Its joints sit 3-5 cm
  off - the Mixamo skeleton keeps the capture's bone lengths; the directions are what carry over.
- **"silhouette IoU (points, same placement): clip A -> refined B"** (refine): B above A by
  0.05-0.15 is normal. Much less means the fit is far off somewhere a small turn cannot reach.
- **"[hands] <name>: ... the wrists X cm off, Y cm from clip to clip -> trusted Z%"** (hand_calib,
  once per rig): X over ~2 cm with a small Y is a hand whose wrist the pose model reads elsewhere
  (a cuff, a glove); a large Y is a hand it reads unsteadily - little of either is trusted.
- **"elbows: clip bends to A deg (N frames past 150) -> refined B deg (M)"** (refine): B at or under
  about 151. A far above 150 in the clip means the fit put the forearm back through the upper arm -
  its jacket sleeve then sticks out past the elbow; look at that stretch of the compare video.
- **"hands (points about the wrist): clip A -> refined B palm lengths"** (refine): B near 0.1-0.2.
  "DWPose sure of N of M hand-frames" says how much of the clip it could read; hands blurred by a
  fast move, under hair or at the frame's edge are left as the capture has them.
- **`<move>_compare.mp4`**: the character should do what the video shows. Where it does not, look
  at the match first (a wrong capture bends into odd depths), then the video.
- **how close, in numbers**: `tools/motion_fidelity.py --name <n> --move <m>` renders the clip at
  the video's timing and camera and scores it against the video (pose model on both: joint error
  in torso lengths, silhouette IoU, face; and with DWPose on both, the hands - their points about
  the wrist in palm lengths, how often the palm faces the way the video's does, how far off they
  point - with close-ups in `<move>_hands.png`). `tools/motion_stages.py` scores each stage (fit, FBX,
  rig, render) to find which one loses the video. Before blaming the rig for a dip in the score,
  read its line "the pose model misreads the render in N frames (a-b); the render scores X without
  them": the pose model fails on poses it rarely saw, on the render as on the video - at the top of
  Mara's kick it loses her raised leg on the render for 7 frames though her skeleton kicks as the
  video does (0.069; 0.049 without them). The side and top panels of the angles video settle it.
  Hold a rig's skeleton against the video only through the render: the pose model reads a stylised
  character's wrists at its cuffs on both, so skeleton-against-video overstates arm error.
- **a video beside a render must be in time**: a 24 fps video against 30 fps renders repeats a frame
  now and then. If a character seems to lag its video more and more, and the video stands still at
  the end, the comparison is resampling wrongly, not the rig.

Model files and licence: see `pipeline/motion_video.py`'s docstring. On a Mac the fine-tune is
prepared once: `tools/f8_scales_to_f32.py` (PyTorch's MPS backend has no float8), then
`tools/merge_lora_quantized.py --device cpu` to bake in the turbo LoRA (never load the LoRA
through ComfyUI here: it keeps a copy of every patched weight and a 24 GB Mac swaps to a halt). MiniMax H3's licence excludes the US, EU, UK and South Korea; do not publish its videos
or moves made from them without the user's say-so.

## Rules for changing the pipeline

- Measure before and after, on more than one character. Every number in the README comes from
  a script in this repository; keep it that way.
- A gate beats a guess: when a rule cannot be sure, make it switch a feature off and say so in
  the log, rather than ship it wrong.
- Write the measurement that motivated a rule into the comment above it, with the character it
  was found on.
