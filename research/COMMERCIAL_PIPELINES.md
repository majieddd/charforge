# How Meshy and Tripo deliver HD, quad, rigged, animated characters - and what CharForge changes (E162, 2026-10-08)

Read from primary or near-primary sources on 2026-10-08. A broader sweep by four research agents also reached open tools
and papers; any claim of theirs not checked against a source is left out of this note.

## Meshy (API reference)
- **Image to 3D.** `ai_model` meshy-6-lite | meshy-6 | meshy-7.1 | latest, plus `meshy-t2` for "smart topology".
  - `topology`: `quad` (quad-dominant) or `triangle` (the default).
  - `target_polycount`: 100-300,000, default 30,000. Smart topology: 100-15,000, default 4,000.
  - `should_remesh`: off by default for Meshy 6/7. Unremeshed "provides a mesh for the highest-quality model".
  - `texture_resolution`: 2k | 4k | 8k, default 2k. `enable_pbr` adds metallic, roughness and normal maps.
  - `pose_mode`: a-pose | t-pose. `remove_lighting` strips highlights and shadows from the base colour (default true).
  - https://docs.meshy.ai/en/api/image-to-3d
- **Remesh.** `topology` quad | triangle; `target_polycount` default 30,000, which "may deviate"; or `decimation_mode`
  adaptive levels 1-4. https://docs.meshy.ai/en/api/remesh
- **Rigging.** Takes a textured humanoid facing +Z with at most 300,000 faces; `height_meters` is optional. Returns a
  rigged FBX/GLB plus "walking" and "running" clips. Bone naming, fingers and face bones are not documented.
  https://docs.meshy.ai/en/api/rigging

## Tripo (API, as documented for Comfy's Tripo nodes and Tripo's developer docs)
- `texture_quality`: standard | detailed (HD) | extreme (8K).
- `face_limit`: -1 lets Tripo choose, about 1.4M faces on v3 standard and 2M on detailed.
- `quad`: quad output, delivered as FBX, up to 150,000 faces.
- `smart_low_poly`: "hand-crafted style" topology at 500-20,000 faces (quad 500-10,000). Its docs say it may fail on
  complex subjects.
- `auto_size`: scales the model to metres.
- Sources: https://docs.comfy.org/built-in-nodes/TripoImageToModelNodeV2.md and
  https://developers.tripo3d.com/zh/docs/generation-image-to-model/standard.md

## What this means for CharForge
| capability | Meshy / Tripo | CharForge (v0.17) | lane |
|---|---|---|---|
| surface | dense generation (Tripo ~1.4-2M faces), optional quad remesh at a chosen count | TRELLIS.2 at 1M faces, decimated to ~60k triangles, maps baked at 4096 | topology (E163) |
| texture | generated at 2K / 4K / 8K, PBR, delit | 4096 albedo projected from an 832x1216 reference and 1024x1024 side/back views: about 1 MP of real detail | texture (E164) |
| fidelity to the picture | proprietary | TRELLIS.2; Pixal3D (pixel-aligned) under test | generation (E155) |
| rig | auto-rig (biped/quadruped), walk and run | 64-bone Mixamo skeleton with fingers, geodesic weights chosen at extreme poses | skinning (E161) |
| face | ARKit 52 shapes (Meshy's marketing; not in the API docs) | jaw bone and 6 shapes | face (E166) |
| motion | mocap presets retargeted by bone name | Mixamo captures retargeted, then joint limits and a per-frame contact solve | motion (E165) |
| viewer | studio lighting, smooth crossfades | three.js playground, DQS skinning | playground (E167) |

The two clearest root causes of "not HD" and "not quad" are the texture's source resolution and the triangle decimation.
For "not fluid", the question is how much roughness our corrections add on top of the captures (E165 measures it first).
