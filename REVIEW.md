# Adversarial review

The goal was never "a pipeline that measures well". It was: **text or an image goes in, and a
working animated character comes out that you can drop into a game.** This review judges the
project against that goal, the way a game developer would judge a file someone handed them —
not against the metrics the project invented for itself.

Every finding below was reproduced by importing the shipped `.glb` into an empty Blender scene,
or by reading the code path that produced it. Two findings that looked damning turned out to be
wrong on closer inspection; they are listed at the end rather than quietly dropped.

## Verdict

The **generation is good** and the **measurement discipline is good**. Almost everything between
the two and the user is not. There is no command that produces what the README describes; the
file that comes out would be rejected on import by any engine; and the playground demonstrates a
character standing in a scene rather than a character in a game.

---

## Broken — the product does not do what the README says

**1. The documented entry point runs the pipeline everyone rejected.** `run_pipeline.sh` calls
`build_rig.py` (skinning the raw marching-cubes surface), `animate.py` (the procedural gait
described in review as "the worst animation I've seen") and the old verifier. It never runs
retopology, the T-pose, Mixamo retargeting, the weight smoothing or the web export. Every
character shipped in the last week was built by chaining scripts by hand. A new user following
the README gets the character that was thrown out.

**2. Image input is not first-class.** The goal says text *or* an image. The generator is
image-conditioned underneath, but the only way in is a prompt that is first turned into an
image by ComfyUI. Bringing your own concept art is not possible without editing the script.

## Wrong by convention — an engine would reject it on import

**3. The character is 2.00 m tall.** Every character is normalised so its bounding box spans
exactly two units. Engines assume one unit is one metre, so both characters arrive as giants,
and anything scaled against them — doors, props, other characters — is wrong.

**4. The origin is at the hips, 1.00 m above the feet.** Engines place an asset with its origin
on the floor. Dropped in at y = 0, this character is buried to the waist. The playground only
looks right because it measures the bounding box at load and shoves the model up — a patch over
the asset, not a property of it.

**5. The skeleton follows no standard.** 25 bones with SMPL-style joint names —
`left_shoulder` is the upper arm, `left_collar` is the clavicle, `left_elbow` is the forearm.
Engine humanoid mappers match by name, and these names mean something else there: Unity's
auto-mapper would read `left_shoulder` as the clavicle. There is no root bone on the ground for
root motion, no finger bones, and no Mixamo-compatible naming, so the thousands of animations
people actually use cannot be applied without writing a retargeting table.

**6. Material data the generator produced is thrown away.** TRELLIS.2 emits a
metallic-roughness texture. The bake keeps colour and normal only and ships a constant roughness
of 0.5 on everything. For Rowan the generator asked for 0.92 (nearly matte cloth) and got 0.5
— visibly plasticky. For Wren it had real distinctions (leather around 0.55, denim and skin near
0.9, metal buckles up to 0.68 metallic) and all of them were flattened.

**7. 64% of the texture is empty.** The two meshes share one atlas; skin covers 8% of it and
clothing 27.5%. The layout is 1,071 UV islands, 804 of them smaller than ten faces — the
signature of `smart_project` on a bumpy voxel surface with no repacking afterwards. Tiny islands
bleed into each other at every mip level, which is where the "rust speckles" that forced PNG
over JPEG came from. Packing properly roughly doubles texel density at the same file size.

**8. GLB only.** Unreal and Godot import glTF well; Unity does not natively. No FBX, no
per-engine notes, no packaged folder — a `.glb` and a promise.

## Does not behave like a game

**9. Jump is an animation, not a jump.** Space plays the clip in place. There is no vertical
velocity and no gravity; the character cannot leave the ground, cannot land on the low boxes the
scene is full of, and cannot fall.

**10. Collision stops dead.** Touching an obstacle zeroes movement entirely. Every game since
the 1990s slides the player along the surface instead.

**11. Walk and run are two states with a hard switch at 2.4 m/s.** Games blend locomotion by
speed, so a character accelerating through a jog looks like a jog.

**12. The run cycle pops every half second.** At the loop point the knee jumps 20 cm, twice a
normal frame step, and no earlier end frame closes it: the source clip is not a whole cycle.
Walk and idle loop cleanly (seams 0.4× a frame step).

**13. The lighting is not PBR lighting.** A hemisphere, a sun and a rim light with no
environment map. Physically based materials reflect their surroundings; with nothing to reflect,
leather, skin and cloth converge on the same dull response — which, with finding 6, is why
everything looked like the same plastic.

---

## Checked and struck

- **"The character faces backwards."** A crude foot-vertex heuristic said so. The foot *bones*
  point to glTF +Z, which is the correct convention. Wrong.
- **"A stray Icosphere ships in the file."** Blender's glTF importer creates a bone-display helper
  in a collection called `glTF_not_exported`. It is not in the file. Wrong.

## Good — keep all of it

- **Generation.** TRELLIS.2 with stochastic multi-view conditioning produces genuinely good base
  geometry and colour. This is the part competitors charge for, and it runs locally.
- **Retopology to a watertight mesh, then baking** from the high-resolution surface. The concept
  is right; finding 7 is about the UV step, not the approach.
- **Mixamo retargeting.** Walk and idle loop cleanly and read as human.
- **Dual quaternion skinning**, the **targeted weight smoothing** and the **texture compression**.
  Each was measured and each made things better.
- **The measurement discipline.** The deformation audit, the collapse map and the ablation habit
  are why this review could be written with numbers in it.

## Out of reach for now

- **UniRig** (learned rigging and skinning; 8.2 GB of weights are on disk). It depends on
  `flash_attn` and sparse convolutions, both CUDA-only. A port to Apple Silicon is a research
  project, not a fix.
- **HY-Motion** (text-to-motion). Its weight download failed — 63 MB of a multi-gigabyte model is
  present. Text-driven animation is therefore not available until that is re-fetched.

## The rebuild, in order

1. One command, text **or** image in, a game-ready package out — replacing `run_pipeline.sh`.
2. The package follows convention: real height, origin at the feet, Mixamo bone names, FBX
   alongside glTF, roughness and metallic restored, UVs packed.
3. Animation that loops: cycle extraction for clips that are not whole cycles.
4. A playground that behaves like a game: gravity and airtime, sliding collision, speed-blended
   locomotion, environment lighting, and a way to download what you are looking at.
5. Dead code moved out of the main path, so the repository describes one pipeline.
