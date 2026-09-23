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
*Correction, found while fixing it: the diagnosis was wrong. The source run closes with a 0.00×
seam. The pop was the export — clips keyed from Blender frame 1 are written starting at
t = 1/30 s, so every loop held its first pose for a frame. Fixed by sliding each clip to t = 0.*

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

---

# Second pass — after the rebuild

The rebuild above was measured the way this project measures everything, and measuring it found
a second layer of problems, almost all in animation. They were invisible in a turntable and
obvious the moment a controller moved the character: feet that skated, a run that went sideways.

## Status of the first review

| # | finding | status |
|---|---|---|
| 1 | entry point runs the rejected pipeline | fixed — `charforge.py`; the old path is in `attic/` |
| 2 | image input not first-class | fixed — `--image`; TRELLIS removes the background |
| 3 | 2.00 m tall | fixed — real height in metres (`normalize_frame.py`) |
| 4 | origin at the hips | fixed — on the floor; then fixed again, see 19 |
| 5 | no standard skeleton | fixed — Mixamo names, curves rewritten to follow |
| 6 | roughness and metallic thrown away | fixed — baked from the generator's own material |
| 7 | 64% of the atlas empty | improved — texel density +60%, islands 4.7x fewer |
| 8 | GLB only | fixed — FBX, per-engine textures, manifest |
| 9 | jump is cosmetic | fixed — gravity and airtime solved from the clip |
| 10 | collision stops dead | fixed — push-out sliding, stairs |
| 11 | hard walk/run switch | fixed — speed blend on a shared stride clock; now three gaits |
| 12 | run pops at the loop | fixed — it was the export offset (corrected above) |
| 13 | no environment lighting | fixed — image-based lighting |

## New findings

**14. The run ran diagonally.** It travelled at exactly 45 degrees to the direction its body
faced; the walk crabbed at 13. The clips had been picked by how they moved — speed, flight time,
bob — which finds clips that move like a run, not clips a controller can drive. A controller
moves the character the way it faces, so a clip whose travel disagrees with its body slides
sideways at speed × sin(angle), whatever else is done. Replaced by grading all 2,147 clips on
body/travel agreement (`blender/scan_locomotion.py`) and choosing a new walk, jog, run and jump.

**15. The character floated and sank.** Every clip was 4–5 cm off the floor: the vertical
reference was taken from the capture (first frame, then its rest pose), and neither stands on the
capture's own floor. Now measured where the feet are, key by key.

**16. Planted feet skated.** 30% of ground speed in the walk and 79% in the run, in the browser.
Two causes stacked: making clips in place by pinning the pelvis threw away the stride's surge and
sway, and copying joint rotations onto legs of different proportions drifts a planted foot
through the stance even when its average speed is right. Fixed by reading contact from the
capture and planting it with leg IK on the mesh's own sole; now 1–3%.

**17. The manifest's run speed was wrong.** 5.84 m/s, measured from hip travel — which ran along
the diagonal. The character's stride supported about 3.5. Speeds now come from the planted soles.

**18. The shoes were rubber.** The skinning diffused the shin's weight down into the shoe, so it
bent at every ankle flex, and gave foot planting no fixed point to hold. Below the ankle the
foot's bones now own the shoe.

**19. The origin was 9–12 cm off-centre.** The median of the vertices near the floor is the
median of a two-lump distribution and lands at the inner edge of the denser foot; every turn in
place orbited a point beside the body. It now sits under the pelvis.

**20. A standing jump waited two thirds of a second.** The capture's anticipation, played in
full. The manifest now gives an entry point a quarter second before takeoff.

**21. Slow walking skated.** Below walking speed the playground mixed the idle into the walk,
dragging the planted foot toward a standing pose: 46% slip at 0.9 m/s. It now slows the walk.

**22. No LODs, no capsule.** A game sizes its collider from the character and swaps meshes with
distance. The manifest carries a capsule measured on the idle pose; the FBX carries three LODs
named for Unity's automatic LOD Group.

**23. The generator paints skin onto clothing.** Juno's second, multi-view pass - and only the
second - put hand-sized patches of skin tone on the outside and back of both trouser legs, where
the conditioning views disagreed. Found by looking at a close-up, not by any metric. A new stage
(`pipeline/texture_cleanup.py`) finds islands of the character's own skin tone inside a garment
and fills them from the surrounding fabric. Its first version also "fixed" every cuff and
neckline, where the part labels are a few texels off the skin's true edge; whether a patch
touches skin is now asked on the 3D mesh, not in the atlas. On Rowan it changes nothing.

**24. Legs that touch become webbed.** Vex stands with her knees almost together. The voxel
remesh works at about 1 cm, fused the two legs where they were closer than that, and the rig
split the weld between left and right - every stride stretched it into a web between her shins.
Cutting the weld in the retopology (faces below mid-thigh with vertices nearer each leg's axis)
was not enough on its own: distance-based weights still gave inner-knee vertices half of each
leg, and choosing the leg by the nearer bone handed strips of one leg to the other, because the
estimated knee sits off-centre. Each leg below mid-thigh is now bound to its own bones, chosen by
which connected piece of the surface a vertex is on, and the cut's scar is smoothed back into
the leg. Rowan, Wren and Juno had no weld; the rule changes nothing for them.

**25. Pink hair was taken for a face.** The rule that rescues face fragments from the hair group
took its "skin" reference from the brightest, reddest part of the hair itself - fine for dark hair,
wrong for pink, and it moved 89% of Vex's hair into the face. The reference is now the character's
own skin, a vertex must be genuinely close to it, and a rule that would move most of a group
declines to move any of it. It also stops moving half of Rowan's hair, which it had been doing
harmlessly since hair and face share one skinned mesh.

## Still out of reach

Hands are fused, so there are no finger bones; there is no face rig; six clips are a working set,
not a full locomotion set; text-to-motion waits on a weight download. See CONTRIBUTING.md.
