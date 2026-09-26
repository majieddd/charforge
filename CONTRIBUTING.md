# Contributing — the running task list

This is the master list: what is being worked on, what is done, and what was tried and rejected.
It is the first place to look before starting anything, because several obvious-looking ideas
have already been measured and are worse than what is shipped.

**The rule of this project: a change lands with a number.** Every claim below is backed by a
script in this repository that you can re-run. "Looks better" is not a result; the deformation
audit and a full-resolution render are.

---

## How to evaluate a change

Run these two before and after. Both take a built `.blend`:

```bash
# per-bone deformation over every frame of every clip
blender -b -noaudio --python blender/deform_audit.py -- --blend work/rowan/final.blend --out audit.json

# linear blend vs dual quaternion, same weights
blender -b -noaudio --python blender/dqs_test.py -- --blend work/rowan/final.blend
```

**Look at every character in the poses that break things** - the standard clips' extremes and the
moves from video, front-left and back-right - before and after:

```bash
blender -b -noaudio --python blender/render_poses.py -- --blend work/pip/final.blend \
    --shots "idle@0.5,jump@1.2,land@0.5,crouch_walk@0.4,wave@0.5,fall@1.0,victory_cheer@2.2" \
    --az 30,210 --res 360 --out audit/pip
# a texture fix, seen on the posed character without a rebuild
blender -b -noaudio --python blender/render_poses.py -- --blend work/pip/final.blend \
    --shots "idle@0.5" --az 210,270 --frame upper --albedo candidate_albedo.png --out try/pip
```

and watch it move - a reel of the clips back to back from one camera (the Studio's Reel button runs
the same):

```bash
python tools/make_reel.py --name pip            # work/pip/qa/reel.mp4
```

Then **render at full resolution and look at it**. Thumbnails actively mislead — a build that
looked cleaner at thumbnail scale turned out to have ballooned sleeves and a shredded hand at
1100px. Use `blender/render_anim.py --res 1100` and crop, or `blender/collapse_map.py` to paint
the error onto the surface.

Metrics that matter, in order:
1. **fraction of externally-visible faces outside [0.5, 2.0]× rest area** — squash *and* stretch,
   hidden geometry excluded
2. median per-bone p99 edge stretch
3. face-normal swing past the bone's own rotation

Metrics that do **not** matter on their own: `spread>=3` (vertices blending bones 3+ joints
apart), cross-midline binding count, unweighted-vertex count. These are proxies. Driving
`spread>=3` from 8.69% to 0% took visual quality sharply *down*.

**For anything touching animation**, run the foot audit too - it plays every clip on the deformed
mesh at the ground speed the manifest states, the way a controller moves the character:

```bash
blender -b -noaudio --python blender/foot_audit.py -- --blend work/rowan/final.blend \
    --report work/rowan/retarget.json
```

and in the playground's console, `await __cf.footAudit()` measures the same in the browser, with
the real controller, blending and interpolation in the loop. The browser number is the one that
counts: a 30 fps bake measured 3% at the keys and 13% in the browser.

---

## Open — ranked by expected impact

### 1. Hair, bags and layered garments generated apart from the body
Chains swing what hangs free (`blender/springs.py`), but the generator lays braids, ponytails and
satchels against the body as one surface, and a chain on a fused part tears it (REVIEW 33). A
garment over another is the same problem, now half solved: Pip's vest moves with his torso and
collarbones (REVIEW, the polish pass), but with both arms overhead the surface that joined vest and
sleeve stretches at the armpit. The fix is upstream: condition the reference on hair held clear of
the back, or generate hair and outer garments as their own meshes. Aoi's hair is the largest part
of what her refined clip still misses against its video.

Smaller, from the same pass: a join between arm and body survives in the first slab below the
cut's start on Mara, Juno, Rowan and Vex's left arm (`free_arms.py` reports `still_joined`); the
reference check (`pipeline/pose_gate.py`) reads only the arms, not a held prop or a cape.

### 2. Video -> motion: what the pose models cannot read, depth, and a licence-clean video model
`charforge.py move` now runs the whole chain: MiniMax H3 (a community fine-tune, in ComfyUI on the
Mac's GPU) makes the video, `pipeline/video_motion.py --fit` reads it and bends the nearest
capture to follow it on the video's own body, `blender/retarget.py` aims the character's limbs and
head along the fit, and the `refine` stage poses the character's own mesh against the video's
silhouette and turns the hands to DWPose's reading (0.054-0.127 torso lengths, IoU 0.77-0.85, hands
0.15-0.25 palm lengths against the video, `tools/motion_fidelity.py`).
Open:
- **Hands DWPose cannot read.** A hand blurred by a fast move, under hair or overhead at the
  frame's edge (Pip's, through his hop) keeps the capture's; so, in part, do cartoon fingers,
  which DWPose fans the wrong way now and then. A hand model trained on stylised hands, or the
  hand's silhouette in the refine's silhouette terms at a finer scale, would reach them.
- **Poses the pose model rarely saw.** The rig now kicks as the video does (the angles video shows
  her leg at head height from the side and above), but reading the render of Mara at the top of the
  kick ViTPose loses the raised leg for 7 frames (61-67), which is most of that move's score (0.069;
  0.049 without them). A score that skips frames where the pose model's reading of the render
  disagrees with the rig's own skeleton would say so by itself. DWPose folds the leg too, at
  confidence 0.16-0.21 (REVIEW, Tried and dropped). ViTPose++ huge is untried - ask before downloading.
  The rig also lowers the leg a little early (knee bent 70° at 2.8 s, the video's leg still up).
- **Depth.** The fit's remaining error on moves the library lacks is mostly depth (0.143 of 0.178
  torso lengths), which comes from the nearest capture. Taking it from the best of several
  captures did not help (REVIEW, Tried and dropped). A monocular 3D pose model, or a second
  generated view of the same move, would.
- **The licence.** MiniMax H3's licence excludes the US, EU, UK and South Korea. Wan 2.2
  (Apache-2.0, `wan2.2_ti2v_5B`, 18 GB with its text encoder and VAE) would run through the same
  script with a different workflow - ask before downloading.
- **Stylised performers.** The pose model reads realistic bodies best; for an anime or cartoon
  character, a realistic performer's video (`--performer`) may give a cleaner move.

### 3. Faces for drawn styles
Drawn eyes read as "not a clean blob" and fall back to positions estimated from the head's
outline; a drawn mouth's width is guessed (`pipeline/face_landmarks.py`). Aoi blinks and talks on
those estimates. A face-landmark model trained on illustrations would replace the guesses.

### 4. Text-to-motion
HY-Motion is vendored but its weight download failed. With it, "a character who limps" is a
prompt. Retarget its output through the same `ground_and_plant`.

### 5. Runtime foot IK for slopes and stairs
The clips are planted on a flat floor; a two-bone IK per leg at runtime, reusing the manifest's
contact phases, would put each foot on its own step.

### 6. Raise generative fidelity
Faces are now detailed from a close-up (REVIEW 34); the geometry under them is still the
generator's. More conditioning views were measured against an H3 turntable of Mara
(`tools/turntable_fidelity.py`): the reference alone 0.778 IoU over the turn, with the repainted side
and back 0.781, with three turntable frames 0.786 - more views barely move the shape, and views
that disagree get averaged (her braid). A finer TRELLIS grid, or fitting the mesh to the reference's
silhouette directly, are what is left.

### 7. Pose-space correctives for shoulder and elbow
The playground drives its own shader and could apply them; the shoulder carries the largest
per-bone deformation.

### 8. A learned rig (UniRig)
CUDA-only dependencies. The geodesic weights it would replace are measured and adequate.

### 9. Reproducible from a clean checkout
Needs a bootstrap script and a pinned `requirements.txt`.

---

## Done

Newest first. Each of these has a script and a measurement.

- **Hands and faces of the characters from a few words** (REVIEW). `pipeline/cut_hands.py` reads
  the part labels at the wrist: a bare forearm (0.92-1.00 "arms" against 0.13 or less under a
  sleeve) makes the modelled hand broader and thicker to meet it, up to 1.6x (`blender/hands.py`,
  `blender/rig_build.py` follow the same bulk). `pipeline/project_texture.py` gives the hands the
  face's colour when no warm skin exists (Gray's grey, not a fallback peach), and with
  `--keep-base-face` (realistic characters) leaves the face to the generator's own texture when the
  face flow is refused - Knight's doubled face, and the face rig it then found.
- **Checking the checks** (REVIEW, "seeing a move from every side"). The compare and angles videos
  played a 24 fps video 25% fast beside 30 fps renders (each frame taken once, never twice) - the
  rig looked 0.2 s late and more; `tools/side_by_side.py`, `tools/make_angles.py` and
  `pipeline/video_motion.py` now repeat frames when the output runs faster. `tools/motion_stages.py`
  re-reads a rig when its blend is newer than its cache, and lists the frames where the pose model
  misreads the render against the rig's own skeleton (Mara's kick: 7 frames, 0.069 -> 0.049
  without them). `blender/face_render.py` writes positions near 1, not at +4, where EEVEE's
  half-float film has 0.5 mm steps instead of 3.9 mm (face landmarks were 2.5 mm off on average).
- **Seeing a move from every side** (REVIEW). `tools/make_angles.py` / `blender/render_angles.py`:
  any clip from several orthographic cameras at one scale on a 10 cm grid, side by side in one
  video (with `--video`, the move's video first), and each elbow's and knee's bend and each shoe's
  height read out under them; **Angles** in the Studio. It found the feet of the moves from video
  standing 2-7 cm off the floor after the refine: `blender/apply_pose_corrections.py` now puts
  each key's lowest shoe as far above the floor as the video's figure stands above its own floor
  line (frames where the figure runs off the picture keep the retarget's planting).
- **Characters from a few words** (REVIEW). A short prompt is written out by a local language model
  (`pipeline/describe.py`: fields filled by Ollama, the sentence composed here, props, poses and
  heights dropped, the age group from the prompt's own words); the height follows the description;
  a name already taken is refused by the Studio and by `make`. TRELLIS's first pass is checked from
  the front against the picture, and a board built in with the figure (Gray, Knight, Kaito: IoU
  0.35-0.44 against 0.90-0.94) is made again from the picture through TRELLIS's own background
  remover (Knight 0.40 -> 0.92), then on a new seed.
- **Elbows that fold back through the arm** (REVIEW, the polish pass). One camera fixes a bone's
  depth only up to its sign; `video_motion.py`'s fit now chooses an upper arm's and its forearm's
  together - no elbow past 150-160 deg (the capture library's pass 150 in 0.05% of frames), no
  wrist behind the torso within the chest's width and height (1% of the library's are) - and
  `refine_pose.py` holds every elbow under 150 deg (`--w-elbows`). Aoi's spell cast: 176 -> 150 deg,
  silhouette and hands unchanged; self-test on 40 moves the library lacks 0.212 -> 0.211 torso
  lengths. `tools/motion_fidelity.py` reports the hands' median error beside the mean, since a few
  misread frames can move a mean (Aoi's sleeve ends read as hands: mean 0.33, median 0.17).
- **The polish pass: the surface when it moves** (REVIEW has the numbers). Arms cut free of the
  body are checked slab by slab and cut again where a join survives (`free_arms.py`); the rig keeps
  the arm's weight on the arm from 0.30 of the upper arm, a vest's armhole with the collarbone and a
  jacket's hem with the pelvis where colours tell the garments apart (`rig_build.py`); UV islands
  folded onto themselves are unwrapped again (`retopo.py`, 224-2,602 overlapping faces per character
  -> 6-31, and 97-158 on the layered Rowan, Kaito and Knight); the texture no longer paints a generated view's outline onto what lies behind it, finds
  the modelled hand only outside the body, takes the hands' tone from the face and colours surfaces
  the arm cut opened from the nearest old surface along the mesh (`project_texture.py`); a reference
  whose hands touch the body is drawn again (`pose_gate.py`); a TRELLIS stage macOS stops on the GPU
  is retried in smaller pieces. Seen in `blender/render_poses.py`, before and after, on every
  character.
- **CharForge Studio** (`charforge.py studio`, `studio/`). A local web app: make a character from a
  description or an image, watch the job queue with each stage's time left, turn characters round
  and play their clips, add moves from video, rerun from a stage, and walk every local character
  round the playground at `/play/`. It learns each stage's time from its own logs and survives a
  restart under a running job.

- **Hands that follow the video.** DWPose (`tools/dwpose.py`; 134 MB, Apache-2.0) reads 21 points a
  hand on the video, and the refine stage turns the wrists, the forearms about their length and
  thirty finger joints - each within a finger's range, never the arm - until the hands lie on it,
  from a wrist point calibrated to where DWPose sees one on the character (`tools/calibrate_hands.py`).
  Scored with DWPose on both sides (`tools/motion_fidelity.py`): Aoi's spell 0.64 -> 0.25 palm
  lengths, palms the video's way 62% -> 99%; Mara 0.24 -> 0.21 and 0.15; Pip 0.23 -> 0.18; the
  body's scores within 0.006. What did not work is in REVIEW, Tried and dropped.
- **Built by hand, then turned into rules (REVIEW, third pass).** A solid instead of a shell
  (26-direction visibility), joints traced through it (`refine_joints.py`), modelled hands with
  15 finger bones each (`cut_hands.py`, `hands.py`, `hand_model.py`), geodesic weights
  (`geodesic_weights.py`), the head rigid to the jaw line, a face rig with the mouth cut open
  (`face_rig.py`), spring chains for free-hanging parts (`springs.py`), texture projected back
  from the source views with a detailed face (`project_texture.py`, `face_detail.py`), 18 clips
  and a controller that uses them (strafes, back-pedals, turns in place, crouch, fall and land),
  and three new characters in three styles (Mara, Aoi, Pip). Runbook: `.claude/skills/charforge/SKILL.md`.

- **Image input, tested on a painting.** Vex was made from a digital painting with a street behind
  her and no prompt - background removal, both TRELLIS passes and every later stage held. She
  exposed two defects, fixed for every character: legs welded by the voxel remesh (cut along the
  skeleton in `blender/retopo.py`, then each leg piece bound to its own bones in
  `blender/rig_retopo.py` - nearest-bone assignment pulled blades out of the inner knee), and
  coloured hair taken for skin by the face-rescue rule (`pipeline/transfer_labels.py` now takes
  its skin reference from real skin). The texture cleanup also learned to judge a patch by its
  mean colour, after growth turned Vex's zipper teal.
- **Skin painted onto clothing, removed.** TRELLIS's multi-view pass can decode the body's colour
  into a garment where its conditioning views disagree (Juno's trouser legs). `pipeline/texture_cleanup.py`
  flags clothing texels near the character's own skin tone and far from the fabric around them,
  grows each into the whole patch, and fills only patches whose faces are at least ~4 cm from any
  body vertex *on the mesh* - the UV-space version painted jacket onto every cuff. Juno: 10
  patches filled; Rowan: none; Wren: 16 specks. Skips entirely if >4% of the clothing matches the
  skin (a tan coat).
- **Origin, capsule, LODs.** The origin was the median of the vertices near the floor - the
  median of a two-lump distribution, one lump per foot - and landed at the inner edge of the
  denser foot: 9.3 cm off-centre on Rowan, 12.0 cm on Wren, so every turn in place orbited a
  point beside the body. It now sits under the pelvis (`blender/normalize_frame.py`). The
  manifest carries a collision capsule measured on the idle pose, and the FBX carries LOD1 (40%)
  and LOD2 (15%) named for Unity's automatic LOD Group (`blender/package.py`).
- **Locomotion that plants its feet.** Browser-measured contact slip went from 30% (walk) and 79%
  (run) to 2.2% / 2.5% / 1.6% for walk / jog / run, with soles within 3 mm of the floor.
  Each step was a separate defect, found by measuring the previous fix:
  - *Clip choice.* The shipped run travelled at 45 degrees to its own body and the walk at 13.
    `blender/scan_locomotion.py` graded 2,147 clips on body/travel agreement, straightness,
    source slip and cycle closure; finalists judged on filmstrips. New walk, jog, run and jump.
  - *Heading from travel* (not the hip line), and *in place without pinning the pelvis* - only
    the mean velocity is removed, the stride's surge and sway stay.
  - *Height from the floor, key by key*: the pelvis follows the higher planted sole down to it.
  - *Contact from the capture, planted with leg IK*: each contact pins the sole point the foot
    rolls on - lowest at heel strike, lowest at toe-off - read off the mesh, because the rig's
    "ball" joint sits 9 cm up inside the shoe. Soles never cross the floor; swing feet keep a
    clearance that eases in from contact. Ground speed comes from the planted soles.
  - *Rigid shoes*: below the ankle the foot's bones own the shoe (`blender/rig_retopo.py`); it had
    been skinned half to the shin and bent like rubber.
  - *60 fps*: at 30 a run's contact is 4-6 keys and interpolating between them slid the foot.
  - *Playground*: three gaits blended by speed on one stride clock; below walking speed the walk
    slows instead of mixing in the idle (46% slip at 0.9 m/s); the jump enters at the manifest's
    `entry_s`, 0.25 s before takeoff instead of 0.67 s.
- **One entry point, text or image.** `charforge.py make --prompt | --image` runs fifteen cached,
  resumable stages; `run_pipeline.sh` (which ran the rejected procedural pipeline) is in `attic/`.
- **Engine conventions.** Real height in metres, origin on the floor, Mixamo bone names (with the
  animation curves rewritten - Blender 5's layered actions do not follow a bone rename, and the
  first build shipped every clip bound to nothing), FBX beside glTF, Unity and Unreal texture
  variants, a manifest. Verified by importing both files cold.
- **Real PBR, twice the texel density.** Roughness and metallic baked from the generator's own
  material instead of a flat 0.5; UVs smoothed, unwrapped and concave-packed (median texel
  density +60%, 4.7x fewer islands); TRELLIS's inconsistently wound shells fixed face by face with
  a ray test, which restored the missing normal detail (46% flat -> 6%); the AO bake that Metal
  silently corrupts is validated and redone on the CPU.
- **A playground that behaves like a game.** Gravity and airtime solved from the jump clip,
  sliding collision, stairs, a speed-blended gait, environment lighting, per-character capsule.
- **Build tooling.** `tools/build_site.py` generates the Pages site, the single-file artifact and
  the release zips from `out/`; nothing is assembled by hand.

- **Targeted weight-gradient smoothing.** The steepness of the weight field across a face
  carries a lift of 2.48x on collapse, so it is softened — but only where it is steep, and by a
  *continuously varying* amount (a smoothstep over the gradient percentile, not a selection).
  That distinction is the whole point: every previous weight change on this project used a binary
  per-vertex decision and tore the surface. This one does not regress anything.
  Clothing median per-bone p99 stretch 63.5% → 62.0%, worst 134.0% → 131.2%, collapse during the
  walk 8.83% → 8.60%; the skin is untouched and identical. `blender/smooth_hotspots.py`.
  A strength sweep found mild is best — pushing harder buys 0.05% of collapse and costs 5 points
  of median stretch and 37 of worst:

  | strength | collapse (walk) | cloth median p99 | cloth worst |
  |---|---|---|---|
  | none | 8.83% | 63.5% | 134.0% |
  | **mild (shipped)** | **8.60%** | **62.0%** | **131.2%** |
  | medium | 8.57% | 62.2% | 137.6% |
  | strong | 8.55% | 67.1% | 167.9% |

- **A second character, and a roster to pick from.** Wren — a woman in a brown leather jacket
  with a satchel — was rebuilt from her raw generation through the current pipeline with no
  parameter changes, and lands in the same place as Rowan (clothing median p99 54.7% vs 62.0%,
  worst 128.8% vs 131.2%). The playground now carries both, switchable without a reload, each
  downloaded only when selected, plus a panel explaining how to run the generator locally.
- **Web build 3.4x smaller at no visible cost.** 23.45 MB → 6.88 MB. Textures were 86% of the
  file; the 4K albedo turned out to be upsampled from TRELLIS's native 1024 and held nothing over
  2K. Albedo 2048 WebP q95, normal map 2048 WebP **lossless** (a normal is a direction, not a
  colour — lossy cost it 5 dB against the albedo's 2). Verified by re-rendering a matched
  close-up: **44.9 dB PSNR**, mean difference 0.64/255. `tools/optimize_glb.py`.
- **Found out what actually collapses.** `blender/collapse_why.py` — see the rejected table for
  what this ruled out.
- **Dual quaternion skinning in the playground.** Jacket faces under half rest area 8.80% → 6.18%
  (walk), 11.49% → 5.96% (run); skin 0.79% → 0.11%. No measurable frame cost. Toggleable.
  `web/playground.html`, verified by `blender/dqs_test.py`.
- **Per-bone deformation audit.** Turns "the cloth stretches" into a number attached to a joint,
  over every frame of every clip. `blender/deform_audit.py`.
- **Deformation error painted onto the surface.** `blender/collapse_map.py` — makes the numbers
  addressable instead of aggregate.
- **Hidden-face correction to the collapse metric.** Only 56% of collapsed faces are externally
  visible against an 86.5% baseline; the honest visible figure is 4.93%, not 8.80%.
- **Re-rig on the retopologised mesh.** The pipeline had been rigging the raw marching-cubes
  surface (2.30% open edges, 0.48% non-manifold, 0.29% zero-area) instead of the clean one (0.00%
  on all three). This one decision explained bone-heat failure, QuadriFlow no-ops, cloth
  divergence and face tearing.
- **Face reclassification in Lab colour.** Segmentation labelled ~13% of the hair group as hair
  when its albedo was plainly skin — cheek, jaw, forehead. Those fragments were bound separately
  and tore through the face under motion. Sub-second numpy pass, seeded from the mesh's own
  extremes so nothing is hard-coded to one character. `pipeline/transfer_labels.py`.
- **Stochastic multi-view conditioning.** Varying which view conditions each denoising step, which
  is what TRELLIS was trained for, rather than concatenating view tokens. Open edges 2.30% → 1.10%.
  Token concatenation was tried first and destroyed the mesh (10.58% open edges).
- **T-pose before rigging.** Arms baked to 90° so the rest pose is a modelling-standard T.
- **Gait-driven procedural clips replaced by Mixamo retargeting.** World-space rest-delta
  retargeting, scale from world-space leg length. `blender/retarget.py`.
- **Normal-map denoise.** Speckle 13.63% → 1.60% at 2K; at 4K with corrected bake settings (cage
  extrusion 0.10, ray distance 0.14) the raw bake is already 1.81% → 0.90%. Unbaked texels
  needing repair fell 31.8% → 4.1% — that was a bake-settings problem, not a resolution one.
- **PNG instead of JPEG for the atlas.** JPEG bled across the tiny-island UV layout, producing
  rust speckles and dark gashes.

---

## Tried and rejected — do not redo these without new evidence

| idea | result | why |
|---|---|---|
| Ray-visibility masking (bone must be unoccluded from the vertex) | skin median p99 stretch 16.4% → **141.3%** | binary per-vertex mask; neighbours disagree on a grazing ray and the surface tears between them |
| Bilateral left/right prior | no help on top of the above | same reason; geodesic distance gets laterality for free anyway |
| Skeleton-graph spread bound (≤2 joints between any two influences) | 16.4% → **44.3%** | ditto, plus it is a hard mask keyed on a discrete label, so it jumps wherever the dominant bone changes |
| Geodesic distance along the surface | numerically a wash (21.4% / 73.0%), visually worse | on a *baggy* sleeve the surface distance far exceeds the anatomical distance, so it inflates sleeves; bridging the separately-islanded hands to the cuff shreds them |
| A-pose rest instead of T-pose | collapse down (8.8% → 5.1%) but ballooning up sharply (area p99 1.69 → 3.73), visibly worse | ballooning reads more than collapse |
| Cloth simulation on the garment | diverged at 8,438%, blew the legs apart by walk frame 14 | one connected garment pinned only at the top 18%; removed entirely |
| Blender bone heat (`ARMATURE_AUTO`) | failed at 100% unweighted, twice | needs a closed manifold; a jacket is an open shell |
| QuadriFlow remesh | silent no-op | same reason; decimation preserves the shell |
| Joint-aligned edge loops as the fix for cloth collapse | faces near a joint are **less** likely to collapse (lift 0.49x, only 4.9% of collapses) | this was item #1 on the roadmap for a week. It is not candy-wrapper: collapsing faces are *further* from joints than average (29.9 vs 26.0 face widths) and *larger* than average. Measure before building |
| Aggressive weight smoothing (even when targeted) | strong setting: cloth median p99 62.0% → 67.1%, worst 131% → 168% | the mild setting is the whole win. Past it you trade real stretch for a rounding error of collapse |
| Choosing locomotion clips by speed and flight time alone | a run at 45 degrees to its own facing, a walk at 13 | the features find clips that move like a run, not ones a controller can drive; grade body/travel agreement |
| Removing a clip's mean hip-line heading | turned a dead-straight walk 13 degrees off course | a capture's pelvis need not face its travel; the travel direction is what the controller and the planted feet depend on |
| Pinning the pelvis to make clips in place | 20-30% foot slip | discards the stride's surge and sway, which the planted feet then inherit |
| A height reference from the capture (frame 0, or its rest pose) | run floated 4.8 cm / every clip sank 4-5 cm | neither stands on the capture's floor; measure the floor where the feet are |
| Pinning the skeleton's ball joint | pelvis dragged down 6-11 cm | on a generated rig it sits at the instep, 9 cm up inside the shoe |
| Holding the rolling pivot at floor height | sole 3-4 cm through the floor on 40% of walk keys | the rest of the sole rotates through the floor; put the lowest sole point on it instead |
| Blending the idle into the walk below walking speed | 46% slip at 0.9 m/s | play the walk slower; fade to idle only near a stop |
| A 30 fps bake | run slid 13% in the browser, 3% at the keys | a run's contact is 4-6 keys at 30 fps; interpolation between them moves the pinned foot |
| `recalc_face_normals` on TRELLIS shells | normal-map repairs 27% -> 42%, AO black | the shells are double-walled with inconsistent winding; flip faces by a per-face ray test instead |
| Smoothing the source surface before baking | high-frequency normal energy -9%, dead texels up | no benefit to faceting |
| Choosing each leg's vertices by the nearer leg bone | blades pulled out of the inner knee | the estimated knee sits off-centre in the leg; use which connected piece of the surface the vertex is on |
| Seeding the face-rescue rule's skin colour from the hair group | 89% of pink hair moved into the face | take skin from the body; move nothing if most of a group would move |
| The median of floor vertices as the origin | 9-12 cm off-centre | a two-foot distribution's median lands at the inner edge of the denser foot |
| A vision model as a quality gate | misdescribes A-pose and side lighting | see `attic/pipeline/gates.py`; caps anything built on it |
| Hair cards over the solid hair | speckle inside the hair, cards off it, shimmer without TAA | the solid read better at game distance; `blender/hair_cards.py` kept as an experiment |
| Carving a gap between a braid and the body in the voxel solid | hollow pockets inside the hood, no separation | the braid was generated inside the hood's volume; there is no surface to cut along |
| Spring chains on fused parts | a 10-degree swing stretched 163 edges past 2x (worst 19x) | a chain needs a part that meets the body only at its root |
| A close-up face camera aligned on its own | doubled eyes, a dot grid | two alignments disagree by a few pixels; read the detail through the front view's alignment |
| Image-to-image on the face at strength 0.35 | the eyes changed shape | 0.28 details the face without changing it |
| The thinnest cross-section as the wrist | a finger, on hands modelled with separate fingers | find where the palm rounds into the wrist or enters a sleeve |
| The pose model's wrists and face marks on drawn characters | cuffs as wrists, a mouth on the eyelids | trained on photographs; take joints from the geometry and landmarks bottom-up with checks |

---

## Measured but inconclusive

- **Layer poke-through.** `blender/pokethrough.py` asks how much geometry is hidden at rest but
  surfaces during animation — a t-shirt patch pushing through the jacket would read as a grey
  flicker. It reports 3.40% of clothing faces and 4.85% of skin faces. That number is *ambiguous*
  and was not acted on: the inside of a jacket legitimately becomes visible when the arm swings,
  and this test cannot tell that apart from a layer actually crossing. Separating the two needs
  the part labels carried into the test so it can ask specifically whether a *t-shirt* face
  surfaced through a *jacket* face. Worth finishing before anyone trusts the figure.

---

## House style

- Scripts are standalone and take `--blend in --out out`. They print what they measured, in
  units, as they go.
- A script that cannot verify its own output fails loudly rather than writing a file. Blender's
  operators return success while doing nothing — `parent_set(ARMATURE_AUTO)`,
  `quadriflow_remesh` and `Surface Deform` bind have all done exactly that here. Assert on the
  result, never on the return value.
- Comments explain *why*, especially where a previous approach failed. The failures are the
  expensive knowledge in this repository.
- No new dependency without a note in the PR on what it buys.
