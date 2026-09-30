# E102 visibility follow-on

This is the living handoff for the unfinished E102 audit. Keep prior findings below as history; the newest continuation notes are at the top.

## Continuation status - 2026-09-30 (runtime validation done; see E102 and E128 in the tracker)

- **Controls exist and pass.** `tools/aberration_controls.py` builds a synthetic character with known aberrations (`blender/aberration_controls.py`) and audits it at step 1 / bvh-step 1 under DQS and LBS: 22/22 checks pass (results kept in `research/data/e102_controls.json`). Run it after any change to the audit or `blender/mesh_contact.py`.
- **BVH overlap is a triangle test.** Checked directly: two triangles whose boxes overlap without touching are not returned; a piercing pair is. The pairs are real intersections, now `intersecting_pairs`; contact vs penetration is told apart by a measured depth (`penetration_*`, `deep_*_frames_pct`, and `_body` variants that leave out the head region because it carries the hair).
- **Three errors the controls caught, fixed.** (1) Blender re-splits quads by pose, so the loop-triangle guard tripped on the controls at 55 degrees of elbow bend - it was right; triangles are now split once at rest and reused. (2) A fixed ring search capped depth at 3.1 cm for a 5 cm push; replaced by a flood through buried vertices. (3) Open region rims and layered garments gave false "inside" readings (Aoi's forearm 15.9 cm, Vex's elbow 15.7 cm at 105 deg and 0 at 115); inside now needs a straight projection onto the face and a ray that leaves through the far wall.
- **Cadence.** `--step 1` is now the default (about 20% more time than step 2; a one-frame pop on a skipped frame is otherwise missed).
- **Shared code.** Regions, rest pose, intersections and penetration live in `blender/mesh_contact.py`, used by the audit and by `blender/joint_limits.py` (E128). The refactor was checked to give field-for-field identical output.
- **Normal-ray exposure** is unchanged and still a geometric proxy, not camera visibility; floor flags are still a fixed plane. Human calibration (blinded ratings) is the open part of E102.
- Runs, logs and renders: `charforge_notes/polish/e102_runs/2026-09-30/` (outside the repo). Pre-clearance `final.blend` copies are in `pre_clearance/`.

## Continuation status - 2026-09-26

- The raw crushed/stretched/sheared area metrics remain separate from the sparse normal-ray exposure subset. The audit now forms explicit surface and BVH frame lists, so uneven cadences such as `--step 4 --bvh-step 6` are represented by their actual frame numbers (`--through-step` remains a compatibility alias).
- A final code review found that `BVHTree.overlap()` returns candidate index pairs; it does not prove triangle intersection. The old `through` name and interpretation are retired. The field is now `bvh_overlap_candidates`, and its 3 cm filter is explicitly **rest-centroid separation**, not minimum surface distance. Treat it as a broad-phase triage count only. Blender's [BVHTree API](https://docs.blender.org/api/current/mathutils.bvhtree.html) describes overlap indices; a narrow-phase triangle test and calibrated surface-distance filter remain required before claiming penetration.
- The audit now fails if evaluated loop-triangle connectivity or ordering differs from rest, before applying rest masks or exposure ids. This is a guard, not proof that every modifier configuration is supported.
- The pop heuristic now scores the middle local-shape sample against the chord through its neighbors:
  `||L_mid - (L_prev + L_next)/2|| / mean_rest_edge_length`.
  With the default `--pop 0.5`, the spatial threshold is half a mean rest edge. Its eligible windows are the `n-2` three-sample centers; the report records the center frame and window duration `2 * step / fps`. It is cadence-specific local-shape displacement, **not physical jerk** and not comparable across different schedules without recalibration.
- Pop renders and each raw-deformation kind's normal-ray-exposure-worst render are now included. A normal ray remains a single-direction geometric heuristic—not camera/pixel visibility—and its result must be visually reviewed before interpretation.
- Existing JSON/renders are not overwritten by default. Pass `--force` only for an intentional replacement; JSON publication is staged and atomic. The character assets themselves were not changed.
- No Blender audit or render was run. The Studio was active and owns the project's GPU lock during this continuation. Runtime validation, DQS/LBS comparison, injected controls, and image review are pending; do not mark E102 complete or alter skinning weights from these code changes.

## Checkpoint

- The repo is on `measure-aberrations` at `25e8e19` (`origin/main`).
- `blender/aberration_audit.py` and `tools/aberrations.py` are both untracked active work. Keep them intact while continuing.
- Claude's last stated step was to use the posed BVH already built every six frames to report visible subsets of crushed, stretched, and sheared faces.
- The render review in that conversation identified Bo's idle armpit crush as hidden by the arm, while Bo's crouch-walk shorts and belly fold looked real. Aoi's idle `through` flags looked mostly like ordinary arm/body contact; her crouch-walk shorts stretch looked genuine.

## Indexing constraint

`blender/aberration_audit.py:evaluated()` creates `F` from evaluated **loop triangles**, concatenates every mesh, and offsets each mesh's vertex indices. Its per-frame BVH is built from that same `V, F` pair with `all_triangles=True`.

`blender/pokethrough.py` instead visits each evaluated mesh's original `m.polygons`, builds a polygon BVH, and stores face indices separately per mesh. Its `visible()` ray test is a useful starting idea, but its face IDs cannot be applied to the audit's `F` masks. Recompute triangle centers and use the audit's triangle normals and BVH at the same sampled pose.

## Low-risk integration shape

1. Preserve the existing raw `crushed`, `stretched`, and `sheared` counts and percentages. Add exposed counts alongside them so the established raw measurements and DQS/LBS comparisons remain available.
2. On the existing sparse BVH frames, compute triangle centers from `V[F].mean(axis=1)`. For flagged live triangles, cast from `center + N * epsilon` along that triangle's evaluated normal `N`; classify a ray with no hit before the chosen reach as **normal-ray exposed**. Use the current concatenated triangle order, and intersect the exposure mask with each original flag mask before counting.
3. Report the exposure sample frames and denominator. With defaults (`--step 2`, `--bvh-step 6`), BVH frames are six frames apart; raw area measures still cover every second frame. Keep those coverage rates explicit rather than implying the exposed subset was measured on every raw sample.
4. Leave `through` pair counts separate until their own visibility rule is calibrated. A face exposure check and an intersecting-pair check answer different questions.

At the original handoff, the loop selected BVH samples with `k % max(1, through_step // step) == 0`, so an uneven pair such as `--step 4 --through-step 6` silently used the wrong cadence. That was replaced with explicit frame-number schedules; the report now lists the exact sampled frames.

## Interpretation and review

The normal-ray test is a geometry exposure heuristic, not camera-specific pixel visibility. Keep that wording in JSON fields and summary text. Triangle winding, concave surfaces, and the single ray direction can all affect its result. Do not let an exposed subset replace the raw score or trigger a rig/weight change by itself.

When resuming, inspect the exposed masks and rendered examples for Bo idle (the suspected hidden armpit) and Aoi idle (the suspected ordinary contact), then Bo `crouch_walk` (the suspected genuine shorts/belly collapse) and Aoi `crouch_walk` (the suspected genuine shorts stretch). Add Knight2's worst crouch-walk frame as a scale check if its existing audit output is available. If those examples agree with the exposed counts, run the roster comparison with the same frame schedule and skinning modes.
