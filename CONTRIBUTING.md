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
blender -b -noaudio --python blender/deform_audit.py -- --blend work/char01/final.blend --out audit.json

# linear blend vs dual quaternion, same weights
blender -b -noaudio --python blender/dqs_test.py -- --blend work/char01/final.blend
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

---

## Open — ranked by expected impact

### 1. Improve face shape in the retopology
Aspect ratio is the single strongest factor, but the mesh is not pathological: p50 aspect is
1.66, p90 is 2.20, and only 2.75% of faces exceed 3. So this is not about fixing slivers, it is
about pushing an already-decent mesh toward more uniform quads. A relax-and-reproject pass
(tangential smoothing, then shrinkwrap back onto the original high-resolution surface) is the
standard move and does not change the silhouette.

### 2. Pose-space correctives for shoulder and elbow
What AAA does. glTF morph targets are applied *before* skinning so a standard viewer cannot do
pose-space deformation, but the playground drives its own shader and can. Drive a corrective
shape from joint angle to restore volume on the inside of a deep bend. Still worth doing even
though the collapse turned out not to be joint-centred — the shoulder is where the two largest
per-bone figures sit.

### 3. Strip occluded interior geometry
44% of collapsing faces are never seen — the t-shirt sleeve under the jacket, the jacket's inner
lining. Note this is now a *metric hygiene* task, not a size one: geometry and animation are only
3.17 MB of the GLB, so culling saves well under a megabyte. Do it to stop the audit reporting
damage nobody can see, not to shrink the download.

### 4. Raise generative fidelity
Hair reads as a solid mass; the face is smooth rather than sculpted. Levers: more conditioning
views into the stochastic multi-view pass, or a higher TRELLIS sampling grid. This is the
generation stage, not the pipeline.

### 5. Make the pipeline reproducible end-to-end from a clean checkout
`run_pipeline.sh` assumes a populated `vendor/` and downloaded weights. Needs a bootstrap script
and a `requirements.txt` that is actually pinned.

### 6. Second character
Every measurement in this repository comes from one character. Several constants are suspiciously
well-suited to it. A second subject with different proportions and a different garment is the
cheapest way to find what is genuinely general.

---

## Done

Newest first. Each of these has a script and a measurement.

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
| A vision model as a quality gate | misdescribes A-pose and side lighting | see `pipeline/gates.py`; caps anything built on it |

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
