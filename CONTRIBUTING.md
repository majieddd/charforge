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

### 1. Hands
The generator fuses the fingers, so the rig has one stub past each wrist. A character that holds
a weapon or a phone needs at least a hand socket and ideally a thumb and a finger chain. Two
routes: split the fingers geometrically in the retopology (the hand is a small, well-lit region
of every orbit render, so a hand-pose estimator can place finger joints), or condition the
reference image on an open hand so the generator separates them itself.

### 2. A face rig
The head moves as one piece. A jaw bone and eye bones from face landmarks on the frontal render
are the minimum for dialogue; ARKit-style blendshapes are the AAA bar.

### 3. The rest of a locomotion set
Six clips move a character around a level. A shipped third-person character also has turn in
place, start and stop transitions, strafes and a crouch. `blender/scan_locomotion.py` already
grades travel direction against body facing, so a strafe is a clip that travels at ±90 degrees
with the body facing forward - finding them is a query over `work/loco_scan.jsonl`, then a look
at the filmstrips.

### 4. Text-to-motion
Everything animated comes from the Mixamo library. HY-Motion (text to motion) is vendored but its
weight download failed at 63 MB of several GB. With it, "a character who limps" is a prompt, not
a search. Re-fetch the weights, then retarget its output through the same `ground_and_plant`.

### 5. Runtime foot IK for slopes and stairs
The clips are planted on a flat floor. On stairs the playground steps the whole capsule; a
two-bone IK on each leg at runtime, reusing the manifest's contact phases, would put each foot
on its own step.

### 6. Raise generative fidelity
Hair reads as a solid mass; faces are smooth rather than sculpted. More conditioning views into
the multi-view pass, or a finer TRELLIS sampling grid. This is the generation stage, not the
pipeline.

### 7. A different body type
Three characters, all adult humans of similar proportion in jackets and trousers. The constants
that would actually break - the weld tolerance, the rigid-foot band, the capsule's height band -
need a subject that stresses them: a child's proportions, a long skirt or coat that breaks the
two-leg assumption, a non-human silhouette.

### 8. Pose-space correctives for shoulder and elbow
glTF morph targets are applied before skinning, so a standard viewer cannot do pose-space
deformation, but the playground drives its own shader and can. The shoulder is where the two
largest per-bone deformation figures sit.

### 9. A learned rig (UniRig)
8.2 GB of UniRig weights are on disk, but it depends on `flash_attn` and sparse convolutions,
both CUDA-only. A port to Apple Silicon is a research project; the proximity-plus-smoothing
weights it would replace are measured and adequate.

### 10. Reproducible from a clean checkout
`charforge.py` assumes a populated `vendor/`, downloaded weights and a running ComfyUI. Needs a
bootstrap script and a pinned `requirements.txt`.

---

## Done

Newest first. Each of these has a script and a measurement.

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
