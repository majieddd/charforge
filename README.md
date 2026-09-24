# CharForge

A text prompt or an image goes in; a rigged, animated character comes out that you can drop into
a game engine - glTF and FBX, a Mixamo skeleton with fingers and a jaw, a face that blinks and
talks, PBR textures, level-of-detail meshes and a manifest of everything a character controller
needs. It works in three art styles - realistic, anime and stylised - with one pipeline, and
everything runs locally on an Apple Silicon laptop: image generation, 3D generation, segmentation,
the solid, retopology, hands, rigging, skinning, the face rig, texture projection, retargeting
and packaging.

**[▶ Open the live playground](https://majieddd.github.io/charforge/)** - a small game level with
a character controller driving the generated characters. <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd>
moves relative to the camera (strafes and back-pedals included), <kbd>Shift</kbd> sprints,
<kbd>X</kbd> walks, <kbd>C</kbd> crouches, <kbd>Space</kbd> jumps (fall off the platform for the
hard landing), hold <kbd>Q</kbd> or the right mouse button to aim - the character faces the camera
and strafes - <kbd>E</kbd> waves, hold <kbd>T</kbd> to talk. Swing the camera and the character
turns in place. Each character downloads only when selected; **Download** fetches its package.

<!--ROSTER-->
| | style | made from | height | triangles (LOD0 / 1 / 2) | face | package |
|---|---|---|---|---|---|---|
| **Aoi** | anime | *"a young anime adventurer girl with a long high ponytail of silver-blue hair, a short fitted navy jacket with gold trim over a white shirt, fitted black trousers and knee-high brown leather boots"* | 1.62 m | 75,821 / 30,327 / 11,373 | jaw, blink, brows_up, pucker, smile | [aoi.zip](https://github.com/majieddd/charforge/releases/download/v3.0.0/aoi.zip) |
| **Mara** | realistic | *"a young woman field researcher with a long dark brown braided ponytail, a fitted olive green field jacket with the sleeves rolled to the forearm, a grey t-shirt, tan cargo trousers and brown leather hiking boots"* | 1.68 m | 67,101 / 26,839 / 10,065 | jaw, blink, brows_up, pucker, smile | [mara.zip](https://github.com/majieddd/charforge/releases/download/v3.0.0/mara.zip) |
| **Pip** | stylized | *"a cheerful cartoon courier with messy orange hair and freckles, a puffy yellow vest over a blue hoodie, baggy khaki cargo shorts, striped socks and chunky red sneakers"* | 1.55 m | 78,867 / 31,545 / 11,830 | jaw, blink, brows_up, pucker, smile | [pip.zip](https://github.com/majieddd/charforge/releases/download/v3.0.0/pip.zip) |
| **Juno** | realistic | *"a woman in a red hooded windbreaker, grey cargo trousers and black hiking boots, with a short black bob haircut"* | 1.70 m | 67,157 / 26,862 / 10,072 | jaw, blink, brows_up, pucker, smile | [juno3.zip](https://github.com/majieddd/charforge/releases/download/v3.0.0/juno3.zip) |
| **Rowan** | realistic | *"a man in a green bomber jacket with medium-length wavy hair"* | 1.78 m | 68,448 / 27,378 / 10,267 | jaw, blink, brows_up, pucker, smile | [rowan.zip](https://github.com/majieddd/charforge/releases/download/v3.0.0/rowan.zip) |
| **Wren** | realistic | *"a woman in a brown leather jacket with a satchel and a braid"* | 1.68 m | 69,587 / 27,834 / 10,438 | jaw, blink, brows_up, pucker, smile | [wren.zip](https://github.com/majieddd/charforge/releases/download/v3.0.0/wren.zip) |
| **Vex** | stylized | made from a painted concept image, no prompt | 1.72 m | 74,271 / 29,707 / 11,139 | jaw, blink, brows_up, pucker, smile | [vex.zip](https://github.com/majieddd/charforge/releases/download/v3.0.0/vex.zip) |
<!--/ROSTER-->

## Three styles, one pipeline

<p align="center">
  <img src="results/v3/styles.png" width="820" alt="Aoi (anime), Mara (realistic) and Pip (stylised), made by the same pipeline">
</p>

Aoi, Mara and Pip were made to test one claim: nothing in the pipeline is tuned to a look. Each
came from a single prompt and `--style`, which decides how the reference image is asked for, how
the side and back views are repainted, what the face finder expects (drawn eyes are bigger, drawn
mouths are faint lines) and how the manifest asks an engine to shade it (the playground cel-shades
the anime character with an ink outline). Every other stage is the same code.

They also broke things no realistic character had, which is the point of making them. The pose
model read Aoi's jacket cuffs as her wrists and put her mouth on her eyelids; Pip's cartoon hands
have separate fingers, so "the wrist is the thinnest section" found a finger; his cargo shorts'
hem stopped the leg tracer. Each of those is now a rule that holds for all seven characters -
[REVIEW.md](REVIEW.md), third pass.

## What you get

`python charforge.py make --name mara --style realistic --prompt "..."` writes `out/mara/`:

| file | what it is |
|---|---|
| `mara.glb` | the character: mesh, 64-bone Mixamo skeleton (fingers and a jaw), face morph targets, baked PBR material, 18 clips - for Unreal, Godot, three.js and anything that reads glTF |
| `mara.fbx` | the same for Unity, with `_LOD0`/`_LOD1`/`_LOD2` meshes that Unity turns into an LOD Group on import |
| `textures/` | 4K `albedo`, `normal` (OpenGL) and `normal_directx` (Unreal), `orm` (occlusion/roughness/metallic), and Unity's `metallic_smoothness` + `occlusion` |
| `mara.json` | the manifest: height, collision capsule, art style and how to shade it, the face (jaw bone, morphs, how to drive them), spring chains, and per clip its length, looping, ground speed, travel direction, turn angle, and where the left foot strikes |
| `reference.png` | the image the character was generated from |

**Conventions**, as an engine expects them: metres, Y-up in glTF, facing +Z; origin on the floor
under the pelvis; T-pose rest; bones named `mixamorig:*` (Unity's Humanoid mapper and every
Mixamo preset recognise them, fingers included); clips baked at 60 fps **in place**, with the
speed and direction to move the character recorded in the manifest.

**The clips** (18): idle; walk, jog, run and sprint on one stride clock, so a controller blends
them by speed; walk and jog backwards; strafe left and right; turn left, right and 180 in place;
crouch idle and crouch walk; jump, fall loop and hard landing; wave. Mixamo captures, chosen by
measurement and retargeted with the feet planted.

**The face**: a `jaw` bone, and morph targets `blink_L`, `blink_R`, `smile`, `brows_up` and
`pucker`, all at rest by default. The mouth is cut open along the lips with a mouth behind it,
so the jaw opens it instead of stretching the lips. Clips never key the face; a game drives it on
top of any clip (the manifest says how - the playground blinks every few seconds and talks while
<kbd>T</kbd> is held).

**Hands** are modelled, not generated: the generator's fused paddles are cut off at the wrist and
replaced by a hand with fingers and 15 finger bones, posed and sized to the arm.

**Engine setup** is in the manifest (`materials.per_engine`); in short:

- **Unreal** - import the `.glb` or `.fbx`. Base Color = `albedo`, Normal = `normal_directx`,
  and `orm` plugs straight into Unreal's own occlusion/roughness/metallic packing.
- **Unity (URP or Built-in)** - import the `.fbx`, set Rig to *Humanoid* (the names auto-map),
  Base Map = `albedo`, Normal Map = `normal`, Metallic = `metallic_smoothness`, Occlusion =
  `occlusion`. The face morphs arrive as blend shapes.
- **Godot 4** - import the `.glb`; materials, morphs and animations are wired, and Godot builds LODs.

## Make your own

```bash
python charforge.py make --name mara --height 1.68 --style realistic --prompt "a young woman field researcher"
python charforge.py make --name vex  --height 1.72 --style stylized  --image concept_art.png
python charforge.py make --name mara --from rig        # rerun from a stage; style, height, prompt are remembered
python charforge.py stages                            # what each stage does and writes
```

A prompt is first turned into a reference image (Krea 2 in ComfyUI); an image is used as given.
Generation takes about 25 minutes on an M-series laptop (two TRELLIS passes and the image model),
the rest 10-15. Every stage writes to `work/<name>/` and is skipped next time if that output
exists; `--from <stage>` reruns from any point and `--until <stage>` stops early.

**Driving it, or checking a build**: [`.claude/skills/charforge/SKILL.md`](.claude/skills/charforge/SKILL.md)
is the runbook - what the program decides by itself, the eight things to look at after a build
(each with what good looks like and what to do if not), and which stage to rerun after which
change. It is written so a person or a model can follow it; Claude Code loads it automatically
in this repository.

| stage | what happens | script |
|---|---|---|
| reference | prompt → full-body reference image, or your image as is | `pipeline/comfy.py` |
| generate, multiview | TRELLIS.2 via MLX; side and back views repainted and fed back as stochastic multi-view conditioning | `vendor/trellis2mlx`, [patch](patches/) |
| views, parts | eight orbit renders; a human parser per view, back-projected to body / clothing / hair / accessory labels; hair the parser calls "hat" or "bag" relabelled by colour | `pipeline/parts.py` |
| skeleton | joints from a pose model on the front and side views | `pipeline/skeleton.py` |
| solidify | the generated shell becomes one solid: inside is what cannot see out along 3 of 26 directions; hair flakes closed where hair is nearest | `pipeline/solidify.py` |
| joints | every limb traced through the solid to its tip; wrists where the palm rounds or enters a sleeve; elbows by proportion; midline from the legs | `pipeline/refine_joints.py` |
| hands | the generated hands cut off at the wrist, modelled hands with fingers joined on, one length for both | `pipeline/cut_hands.py`, `blender/hands.py` |
| retopo, labels, texclean | 60k-triangle mesh; albedo, normal, roughness/metallic and AO baked; skin painted onto clothing removed | `blender/retopo.py`, `pipeline/texture_cleanup.py` |
| texture | the source images projected back onto the mesh; the face read from a detailed close-up | `pipeline/project_texture.py`, `pipeline/face_detail.py` |
| weights, rig | weights measured through the body; finger bones at the modelled knuckles; head rigid to the jaw line; rigid shoes; one leg each | `pipeline/geodesic_weights.py`, `blender/rig_build.py` |
| springs | bone chains for what hangs free | `blender/springs.py` |
| frame, tpose | metres, soles on the floor, origin under the pelvis; T rest pose, hands squared | `blender/normalize_frame.py`, `blender/tpose.py` |
| face | landmarks (chin, nose, mouth, eyes, checked), mouth cut open, jaw, five shapes | `pipeline/face_landmarks.py`, `blender/face_rig.py` |
| animate | 18 clips retargeted: heading and travel direction, feet planted with leg IK | `blender/retarget.py` |
| package, web | Mixamo names, glTF + FBX with LODs, engine texture variants, manifest; 2K WebP build | `blender/package.py`, `tools/optimize_glb.py` |

## Motion from a video

The long-term chain is text → image → a video of the character moving → motion → the rig.
`pipeline/video_motion.py` is the video → motion link: a pose model reads the person in each
frame, and the answer is the clip in the Mixamo library (2,453 clips, `blender/motion_library.py`)
whose 2D projection follows the video best, at the right camera angle and speed. It is retrieval,
not reconstruction - it can only return a motion the library has - but what it returns is clean
motion capture the pipeline already retargets, where lifting a video to 3D directly gives jittery
poses and sliding feet.

Measured two ways (`results/v3/motion_selftest.json`):

| test | right clip first | first, or an exact tie | in the top 5 | camera angle |
|---|---|---|---|---|
| 40 library clips seen from a random angle, some mirrored, with pose noise, a 3 s stretch and a speed change | 68% | 100% | 100% | 5° off (median) |
| Juno rendered doing three of her clips from 30°, through the real pose model | 2 of 3 (jump, wave) | 2 of 3 | 3 of 3 (crouch-walk second) | 22° |

"An exact tie" is a library clip that is the same motion as the right one - Mixamo carries
duplicates - within 10% of its cost. Every miss in the first test was one.

Generating the video itself needs a video model (Wan 2.x or LTX-Video in ComfyUI), which is not
installed.

## Measured, not asserted

Every number here comes from a script in this repository. The adversarial reviews that drove
each rebuild are in [REVIEW.md](REVIEW.md); the running task list, including what was tried and
rejected, is in [CONTRIBUTING.md](CONTRIBUTING.md).

**Feet stay where they are put.** Measured on the deformed shoe - its lowest vertex while it
touches the floor - with every in-place clip carried along its own direction at the speed its
manifest states (`blender/foot_audit.py --blend work/<name>/final.blend --report
work/<name>/retarget.json`; in the playground, `await __cf.footAudit()` measures the same through
its own controller and agrees within a percent or two):

<!--FEET-->
| contact slip, share of ground speed (worse foot) | Aoi | Mara | Pip | Juno | Rowan | Wren | Vex |
|---|---|---|---|---|---|---|---|
| walk | 1.0% | 2.9% | 4.8% | 3.7% | 2.2% | 2.1% | 4.7% |
| jog | 1.7% | 2.6% | 3.3% | 3.6% | 1.9% | 2.4% | 2.6% |
| run | 0.7% | 2.8% | 3.0% | 2.5% | 3.3% | 2.8% | 2.4% |
| sprint | 1.1% | 5.1% | 5.2% | 6.9% | 3.2% | 4.6% | 9.2% |
| walk back | 4.2% | 7.4% | 10.0% | 6.0% | 8.0% | 6.8% | 5.4% |
| jog back | 1.2% | 4.6% | 5.9% | 4.0% | 4.3% | 2.6% | 3.2% |
| strafe left | 2.2% | 4.9% | 14.9% | 6.7% | 6.0% | 4.3% | 3.6% |
| strafe right | 1.9% | 5.2% | 7.2% | 3.3% | 5.3% | 4.2% | 3.3% |
| crouch walk | 1.5% | 2.4% | 5.8% | 6.3% | 1.9% | 3.7% | 26.0% |
| soles through the floor, worst | 0.1 cm | 0.2 cm | 0.8 cm | 0.5 cm | 0.2 cm | 0.2 cm | 0.3 cm |
<!--/FEET-->

Two cells are open problems, not noise: Vex's right shoe rides ~1.4 cm high while planted in the
crouch-walk, so the audit mostly sees its heel-to-toe roll; and Pip's short cartoon legs take the
capture's side-step with the planted foot sliding. Every other gait on every character is under
10%, most under 5%.

**The face follows the head.** The head joint is where a pose model puts the ears, and the face
hangs below it; skinned by distance through the body, Juno's nose was 52% neck, her mouth 61%,
her chin 73% - every head turn moved her hair and left her face behind. The head is now one
rigid body down to a jaw line measured from the face's profile (8-9 cm below the ears on every
character so far); nose, mouth and chin are 100% head.

<p align="center">
  <img src="results/v3/head_turn.png" width="720" alt="The same head turn before and after: the face used to stay behind while the hair turned">
</p>

The mouth is cut open along the seam between the lips, with a mouth behind it:

<p align="center">
  <img src="results/v3/mouth.png" width="720" alt="Juno at rest, with the jaw opened 9 and 18 degrees, and talking">
</p>

**The face is as sharp as the texture allows.** A full-body reference gives a face about a
hundred pixels; the texture has room for ~270 texels there. The face region is enlarged and
detailed by image-to-image at low strength and read through the front view's own alignment:

<p align="center">
  <img src="results/v3/face_detail.png" width="720" alt="Juno's face before and after the face detail pass">
</p>

**Dual quaternion skinning beats any weighting scheme we tried** - same mesh, same weights, in
the playground's shader (Rowan's earlier build): jacket faces collapsing under half their rest area
in a run, 11.49% with linear blending, **5.96%** with dual quaternions; skin 0.79% → **0.11%**.

## Running it

Requires macOS on Apple Silicon, Blender 5.x (on `PATH`, or `$BLENDER`), and Python 3.11+.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python tools/check_env.py
```

`check_env.py` reports what is present and what is missing across the three environments this
spans - the host Python, Blender's own interpreter, and the vendored generator. Generation needs
[trellis2mlx](https://github.com/lyonsno/trellis2mlx) in `vendor/`, with the multi-view patch
applied, and its weights; a text prompt (and the face detail pass) needs ComfyUI with Krea 2 on
`localhost:8188` (without it, give `--image`; the face is then painted from the reference alone).
The animation stage reads Mixamo clips from the `jasongzy/Mixamo` mirror on Hugging Face, by the
hashes in `animations/default_clips.json`.

```bash
git clone https://github.com/lyonsno/trellis2mlx vendor/trellis2mlx
git -C vendor/trellis2mlx apply ../../patches/trellis2mlx-stochastic-multiview.patch
```

The site and release packages are generated: `python tools/build_site.py` rebuilds `docs/`
(GitHub Pages), `web/dist/` (a single-file build) and `dist/*.zip` from `out/` and `web/roster.json`.

## Honest limitations

- **Hair and bags generated against the body stay rigid.** Spring chains swing what hangs free,
  but the generator lays braids, ponytails and satchels against the back or hip as one surface
  with it, and a chain on a fused part tears it (a 10° swing stretched Wren's satchel 19x). Aoi's
  long hair, Mara's braid and Wren's braid and bag were all generated that way. The fix is in how
  they are generated, not in the rig.
- **Drawn faces are partly estimated.** A drawn eye is not a clean blob to the face finder, so
  Aoi's eyes are placed from the outline of her head, and a drawn mouth's width is guessed. She
  blinks and talks on those estimates; a landmark model for illustrations would do better.
- **Hair is a solid.** It is closed into clean clumps instead of flakes, which is how anime and
  stylised game hair is built; realistic hair wants cards over it, and alpha-tested cards were
  tried and looked worse without temporal anti-aliasing (`blender/hair_cards.py`, experimental).
- **Motion comes from a library.** 18 clips by default; video → motion returns library clips;
  generating the video, and text → motion, need models that are not installed.
- **Clothing is skinned, not simulated.** A baked cloth cache ties a garment to one clip.
- **The glTF carries one LOD.** The FBX carries three.

## Assets and licensing

The **code** is MIT (see [LICENSE](LICENSE)). The **character meshes and textures** are generated
output and carry no third-party rights beyond the generating models' own terms (TRELLIS.2, MIT;
Krea 2 for the reference images and the face detail).

The **animation clips are derived from Mixamo captures** (via the `jasongzy/Mixamo` mirror) and
retargeted onto each skeleton. Mixamo content is Adobe's; its licence permits use in projects
but not redistribution as stock animation. The clips here ship as part of a demo - for anything
commercial, download your own clips from [mixamo.com](https://www.mixamo.com) and point
`animations/default_clips.json` at them. Open an issue if you are the rights holder and want this
removed.

## Repository layout

```
charforge.py  the one entry point: text or image in, game-ready package out
pipeline/     image generation client, parts, skeleton, the solid, joints, hands, weights,
              texture projection and face detail, face landmarks, video -> motion
blender/      Blender stages (retopology, hands, rig, springs, face rig, framing, retargeting,
              packaging), the motion library, and audits
tools/        GLB optimisation, site and release builder, asset audit, environment check
animations/   the default clip set, by Mixamo file hash, with how each was chosen
web/          the playground template and roster; web/dist is the single-file build
docs/         GitHub Pages build of the playground
.claude/      the runbook (skills/charforge/SKILL.md)
patches/      the stochastic multi-view conditioning patch for trellis2mlx
results/      rendered comparisons and the raw audit JSON behind the numbers above
attic/        the superseded first pipeline, kept for the measurements made with it
```
