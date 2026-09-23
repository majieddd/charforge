# Attic

Code that is no longer on the path from a prompt or image to a shipped character, kept because
the measurements in [CONTRIBUTING.md](../CONTRIBUTING.md) and [REVIEW.md](../REVIEW.md) were
made with it. Nothing in `charforge.py` imports or runs anything here. Scripts moved here may
have broken relative paths; they are a record, not a toolkit.

| File | What it was | Replaced by |
|---|---|---|
| `run_pipeline.sh` | The first entry point. Chained the procedural-gait pipeline below; never ran retopology, the T-pose, retargeting or packaging. | `charforge.py` |
| `blender/build_rig.py`, `split_parts.py`, `fix_weights.py`, `reskin*.py`, `reclassify_face.py` | Rigging the raw marching-cubes surface, and three rounds of re-skinning it | `blender/rig_retopo.py` on the retopologised mesh |
| `blender/animate.py`, `gait.py`, `probe_axes.py` | A procedural walk sampled from published gait curves | Mixamo captures via `blender/retarget.py` |
| `blender/cloth_sim.py`, `bake_cloth.py`, `retopo_garment.py` | Cloth simulation baked per clip | Dropped: a baked cloth cache ties the garment to one clip and cannot follow a game's blended locomotion |
| `blender/verify_deform.py`, `anim_qa.py` | Per-part deformation checks on the old rig | `blender/deform_audit.py`, `tools/asset_audit.py` |
| `blender/web_export.py` | glTF export of the old rig | `blender/package.py` + `tools/optimize_glb.py` |
| `blender/render_hero.py`, `render_exported.py` | One-off renders for the old report | `blender/thumbnail.py`, `blender/render_clip_seq.py` |
| `blender/make_stub_character.py` | A hand-built stand-in character used before the generators were installed | Real generated characters |
| `pipeline/gates.py` | A local-VLM gate on the reference image | Dropped: the VLM cannot report pose or lighting reliably, so the gate passed images that failed downstream |
| `pipeline/mv_condition.py`, `backview.py` | The A/B/C experiment that justified multi-view conditioning (raw renders vs. diffusion-enhanced views) | The `multiview` stage in `charforge.py` |
| `pipeline/run_hunyuan.py`, `probe_pixal3d.py` | Evaluations of other image-to-3D models | TRELLIS.2 |
| `build_report.py` | The HTML report for the first character | The playground and the manifest |
