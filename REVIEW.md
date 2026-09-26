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

# Third pass — built by hand, then turned into rules

The second pass fixed animation. This one started from the other end: a character was built by
hand in Blender - mesh, rig, weights, face, texture - by someone looking at every step at close
range, and each decision that made it better was turned into a rule the pipeline applies. Then
every character was rebuilt on those rules, and three new ones were made to prove they hold
across art styles: Mara (realistic), Aoi (anime) and Pip (stylised cartoon). The new characters
found most of what follows; none of it showed in a turntable.

## Status of the second review's "out of reach"

| | status |
|---|---|
| fused hands, no finger bones | fixed — hands cut out of the solid and modelled, 15 finger bones per hand in the Mixamo layout |
| no face rig | fixed — jaw, mouth cut open, blink L/R, smile, brows, pucker |
| six clips | fixed — 18: idle, walk, jog, run, sprint, back-pedals, strafes, turns, crouch idle/walk, jump, fall, land, wave |
| hair a solid mass | improved — closed only where hair is nearest; spring chains where it hangs free (see 33) |

## New findings

**26. The side view's depth was mirrored.** `skeleton.py` flipped the sign of the 90-degree
view's horizontal axis, so every joint's depth came out mirrored front to back. On slim
characters the refinement pulled joints inside their limbs within its 7 cm search and hid it; on
Aoi the legs were 11 cm behind themselves and no limb could be traced at all.

**27. The pose model reads drawn bodies wrong.** Trained on photographs, it put Aoi's wrists at
her jacket cuffs, 20+ cm from her fingertips, and from the side put both wrists on her ponytail.
Joints are now found from the geometry: each limb traced through the solid to its tip, depth from
the solid, the wrist where the palm rounds into the wrist or the hand enters its sleeve (Pip's
cartoon hands have separate fingers, so "the thinnest section" was a finger), the elbow by
proportion when upper arm and forearm disagree, the midline from the legs (the deepest point
across Aoi's neck slid 9 cm into her ponytail).

**28. The face was skinned to the neck.** The head joint is the ear midpoint - the pivot a nod
turns about - and the face hangs below it. Geodesic weights gave Juno's nose 52% neck, her mouth
61%, her chin 73%, and even her crown 22%: every head turn in every clip moved the hair and left
the face behind. The head is now rigid above a line from under the chin to the nape, the chin
measured from the face's profile (a crease between the lips dips back for a centimetre and
comes forward again; the end of the chin does not).

**29. Wren's hair followed her spine.** The parser labelled her dark scalp hair "hat", the
accessory rule bound accessories to the torso, and 76% of her crown was neck and spine. The
rigid head fixes the binding; the labels are fixed too - hair the parser calls "bag" or "hat" is
relabelled by colour when it joins the scalp's hair (her braid read as satchel strap).

**30. Every face shipped with its eyes shut.** Blender 5 creates shape keys at 1.0, glTF writes
each key's value as the mesh's default morph weight, and every face rig loaded in an engine
with all five shapes applied at once. The playground drove the morphs each frame and hid it.

**31. Opening the jaw stretched the lower lip into a band.** A generated face is one closed
surface with lips painted on it. The mesh is now cut along the seam between the lips (a new
landmark: the darkest row across the mouth) and a mouth pouch is built behind the cut.

**32. The face finder put Aoi's mouth on her eyelids.** On a drawn face the cheeks stand further
forward than the nose, so "the most forward point below the eyes" was a cheek at eye level, and
the mouth was searched below that. Landmarks are now found from the bottom up - chin (a 2-3 cm
step back to the neck), nose, mouth - with the midline from the ears (the model's eye marks sat
2 cm to one side of her face), and a mouth or eye that fails its order-and-size checks switches
its shapes off instead of shipping them in the wrong place.

**33. A chain on a fused part tears it.** Wren's satchel swung on a spring chain, but it is one
surface with her hip: a 10-degree swing stretched 163 edges past twice their length (worst 19x).
Chains are now built only for parts that meet the body at their root alone. Aoi's hair, Mara's
braid and Wren's braid and bag were generated lying against the body and stay rigid. Carving a
gap between a braid and a hood it sits in was tried and made hollow pockets, not a separation.

**34. The face texture was as soft as the reference's face was small.** A full-body reference
gives a face about a hundred pixels; the texture has room for ~270 texels there. The face region
is now enlarged and detailed by image-to-image (strength 0.28 - at 0.35 the eyes changed shape)
and read through the front view's own alignment; a separate close-up camera aligned on its own
landed a few pixels off and doubled the eyes where the two were blended.

**35. One hand could be a quarter bigger than the other.** Each side's measurement fails in its
own way and each was clamped on its own (Aoi: 17.5 and 21.9 cm). Both hands now get one length,
the median of the two measurements and the adult norm.

**36. A rebuild from a late stage resized the character.** `--height` defaulted to 1.75 m and
was not recorded, so rerunning Juno from the rig stage made her 1.75 m. Style, height and prompt
are now recorded with the character.

**37. A face flow gave Pip a second pair of eyes.** The finer flow that lands a photograph's
features on the generator's geometry (it stopped Juno's lips being dragged down her chin) moves
them 1-2% of the figure's height on realistic characters. On Pip and Aoi the generated face
disagrees with the reference by 3.4% and 4.7%, and warping across that painted Pip's eyes twice
and tore his face at the nose, where its crop ended. It is now left out past 2.5%.

**38. Drawn eyes are not blobs the way photographed ones are.** A drawn eye is lashes, iris and
highlight, often under a fringe of the same colour family, and big enough to run off a window
sized for a realistic eye. On drawn styles the eye is the largest blob that is not the hair's
colour, a blob may touch its window, and the window reaches further down (the pose model marked
Pip's brows). Eyes it cannot find are not guessed at: no blinks, and no mouth - on a drawn face
the mouth is placed from the eyes and nose, and a guessed eye once put Aoi's mouth on her chin.

## Tried and dropped

- **Hair cards.** 637 alpha-tested strands over Juno's bob softened the silhouette but speckled
  inside the hair, ran off it across an eye, and shimmer without temporal anti-aliasing (the
  playground has none). The solid hair looked better at game distance. `blender/hair_cards.py` is
  kept as an experiment.
- **A video's joints taken as they are.** Lifting the pose model's 2D joints to 3D from bone
  lengths alone made clips the library already matched 4x worse (0.022 -> 0.088 torso lengths,
  12-clip self-test): near the image plane a bone's depth, sqrt(L^2 - l^2), swings with the pose
  model's noise. The fit now keeps the capture wherever it agrees with the video to within that
  noise, and only then (the gate) bends it at all.
- **ComfyUI applying H3's turbo LoRA itself (on a Mac).** On Apple GPUs ComfyUI loads a model whole
  and patches a LoRA into every weight at load, keeping a backup of each: 15 GB of weights plus the
  copies swapped for 20 minutes without finishing. Its patch also requantizes each weight with new
  scales - 4.5% error per layer to carry a 0.02% change. Baked in offline instead, on the layer's
  own grid (`tools/merge_lora_quantized.py`): 0.5% noise, loads in a minute.
- **Telling the vest from the sleeve by colour and tying it to the chest.** Clothing on the upper
  body split into garments by k-means on the albedo (lightness half-weighted), and a garment that
  covers the chest but not an arm (under 2% of it past 30% of the upper arm and within 35% of its
  length of the arm's axis) sent its collarbone and arm weight to spine3. The colours find the vest
  exactly (Pip: Lab 69/10/63, 2,691 vertices, 1.7% on an arm; nothing on the six others once a
  garment must be 8% of the clothing and a fifth of it on the chest - Aoi's silver hair, labelled
  clothing, was caught before that). But a puffy cartoon vest covers the top of the upper arm in
  the A-pose: moved outright its panels tore from the sleeve into streaks (silhouette IoU 0.782 ->
  0.795 on victory_cheer, and plainly worse to look at); faded in from its edge, or with the part
  over the arm left alone, the wings stayed. On one fused surface something has to stretch - the
  fix is garments made or cut as separate layers.
- **Letting clothing on the body's side of the shoulder go of the upper arm.** Pip's puffy vest
  lifts into wings when the arms go overhead (victory_cheer, a move from a video). Fading the
  upper-arm weight of clothing medial to the shoulder joint (1,835 vertices) changed nothing
  visible: the panels that lift ride the collarbone, which a capture raises ~30 deg with arms
  overhead, and the fade had handed them more of it. The fix needs the vest told apart from the
  sleeve (they are one fused surface, both "clothing") and tied to the chest.
- **Depth from the best of several captures.** Letting the next three matches lend a bone its
  depth where they follow the video more closely in the image was worse than the first match
  alone (0.186 vs 0.178 torso lengths, 40 moves the library lacks): a capture that agrees in the
  image does not agree in depth any more often.

- **Fitting the video on the capture's body.** The fit reconstructs a bone's depth from its length,
  and the capture's lengths bent the video's body to fit them: Aoi (legs 1.03 torso lengths to
  Mixamo's 0.80) came out leaning 28 deg back, her short torso read as a lean. Replaced by the
  video's own lengths (each bone at its longest in the clip): her fit's image error 0.061 -> 0.028.
- **Reading a character's arms off a T-pose render.** For calibrating where the pose model sees
  joints, the T-pose is the wrong pose: on Aoi it read the jacket cuffs as wrists and the elbows
  mid-sleeve (0.40-0.52 torso lengths off her rig's joints), while her A-pose and her video read
  fine. The arms are left to the bones' heads; shoulders, hips, knees, ankles and the face calibrate.
- **Leaving the feet unplanted on a clip from a video.** The fitted legs follow the video, so the
  plant looked unnecessary; without it Pip's cheer scored worse (0.114 -> 0.128 torso lengths,
  IoU 0.714 -> 0.694). Planting stays.
- **A whole-body pose model for the top of Mara's kick.** DWPose (`tools/dwpose.py`: 133 points,
  134 MB, Apache-2.0) was run on the kick for the raised leg ViTPose loses. It reads it worse: at
  the apex (frames 62-64) it puts the kicking ankle 0.9 torso lengths from the hip - a folded leg,
  at her fists - with confidence 0.16-0.21, where ViTPose follows the leg out to the foot at head
  height (1.8 torso lengths from the hip) with 0.58-0.68; on the frames either side both fold it.
  What changed instead is how the fit treats those three good frames: a joint's unsure frames are
  filled from its parent's motion (up to 4 frames), and the fit's corrections are smoothed with
  each frame weighted by its confidence, so a sure reading among unsure ones is followed rather
  than averaged away; the refine reads the same filled points. Close to neutral on the scores -
  the kick 0.083 -> 0.079 torso lengths, IoU 0.821 either way, the other moves within 0.001 -
  because the frames around the apex are still misread. DWPose is kept for the hands, which it
  reads at median confidence 0.75-0.88.
- **Letting the hands move the arm.** The refine's hand term (each hand's points about its wrist
  against DWPose's) first reached every bone above the hand. On Aoi's spell it swung the forearms
  to point the hands: hands 0.64 -> 0.24 palm lengths, but her silhouette lost 0.005 IoU, her right
  elbow moved off the pose model's (0.177 -> 0.191 torso lengths) and at frame 48 her hands rose to
  her chin where the video holds them at her chest. Now the hands turn only the wrists, the
  forearms about their length and the fingers - the arm is the body's terms' to place - and,
  weighted ten times higher since it can no longer bend the body, the hands reach the same 0.24
  with the silhouette unchanged (0.769) and the joints closer (0.134 -> 0.126).
- **Calibrating every hand point to where DWPose sees it.** On the character's own renders
  (`tools/calibrate_hands.py`: a wave, the idle and a jump from three angles) DWPose puts the hand
  points 1-3 cm from the rig's - 4 cm up Pip's wrist, at his sleeve's cuff, and his cartoon fingers
  fanned the wrong way now and then. Moving all 21 points there made the hands worse where the
  offsets were small (Aoi 0.24 -> 0.29 palm lengths, Mara's punch 0.20 -> 0.22) and did not help
  Pip's (0.21): a finger point's offset moves with the pose, and the video's poses are not the
  wave's. The wrist alone, trusted as far as its offset holds from clip to clip and stands clear
  of the ~1.5 cm DWPose's points wander between poses (Pip's 78%, Aoi's 3-36%, Mara's 13-18%), is
  kept: Pip's hands 0.21 -> 0.18, his silhouette 0.789 -> 0.792.

- **Letting the vest's side go below the armpit wherever its colour says vest,** even where the
  flood had crossed a join onto it. The join itself then stretched into sheets from the vest to the
  raised elbow, worse than the flap. The joins are cut in the solid instead.
- **Colouring an arm's unseen surface with the arm's colour at its height.** Written for Pip's
  sleeves before the real cause (folded UV islands) was found; once the UVs were repaired it
  changed nothing visible, and it was taken out.
- **Checking the cut in slabs of the nearest section.** Torso voxels went to the armpit's sections,
  whose centres are nearest the body, and Vex's joins went unseen; slabs straight across the joints'
  line find them.
- **Telling a jacket from the trousers by the human parser's classes** ("top" over "pants") where
  their colours are too close (Mara's olive jacket and tan trousers are 5-6 apart in Lab). The parser
  gave the lower back of Mara's, Juno's and Rowan's jackets to their trousers; the hem rule then split
  each jacket across the seat, and a crouch stretched the split into a pale band - worse than the hem
  following the thighs. The classes are still carried onto the mesh (`labels.json`), unused.
- **Cutting more directions where a join survives.** On Vex's right arm the ring ran inside a
  bulging sleeve, and the sleeve's outer skin lay on the jacket beyond it - more directions changed
  nothing; the ring moves outward instead.

## Found and fixed: a clip from a video against its video

Scored frame by frame against the video (`tools/motion_fidelity.py`, the same pose model on the
render and the video), the four moves sat at IoU 0.61-0.75 and 0.09-0.21 torso lengths; the models
at rest lie on their reference pictures at 0.90-0.93. Stage by stage (`tools/motion_stages.py`):
fit 0.035-0.070, FBX the same, the character's rig 0.07-0.20 - the retarget lost it. Fixed by the
fit on the video's own body (above), aiming the character's limbs, spine and head along the fitted
points after the rotation copy (thighs differ 4-12 deg at rest from Mixamo's, collarbones 16-27),
turning the head to the video's nose, eyes and ears, and a new stage, `refine`, that poses the
character's own skinned mesh against the video's silhouette, joints and face. Now 0.054-0.127 torso
lengths, IoU 0.77-0.85. Pip's elbow was also 4 cm low (63% of the way down the arm, measured along
a centre line that wandered through his sleeve) - now 55%, joint to joint.

The hands kept the capture's: ViTPose's 17 points stop at the wrist, so Aoi's palms faced down where
the video's face the camera and Pip's hands hung half open where the video makes fists. DWPose
(`tools/dwpose.py`) reads 21 points a hand, and the refine stage now turns the wrists, the forearms
about their length and the thirty finger joints - each only as a finger bends, within its range -
until each hand's points about its wrist lie on DWPose's; the arm is left to the body's terms, and
the rig's wrist point sits where DWPose sees a wrist on that character (`tools/calibrate_hands.py`).
Scored with DWPose on the video and the render: hands 0.64 -> 0.25 palm lengths on Aoi's spell (the
palm the video's way 62% -> 99% of frames, pointing 30 -> 8 deg off), 0.24 -> 0.21 and 0.24 -> 0.15
on Mara's punch and kick, 0.23 -> 0.18 on Pip's cheer; joints, silhouette and face within 0.006 of
before.

## Found and fixed: the surface when it moves (the polish pass)

Every character rendered in the standard clips and the moves from video, front-left and back-right
(`blender/render_poses.py`, with `--albedo` to try a texture on the posed character without a
rebuild), and each fault traced to its cause before it was touched.

- **Pip's vest lifted into wings with his arms.** Three causes, in order. The generated solid glued
  his arms to the vest's sides: `free_arms.py` cuts them free below the armpit (T-pose faces
  stretched past 3x: 1,150 -> 390). The rig's rule that keeps the arm's weight on the arm floods the
  arm's own surface from the sleeve's outside - and started the flood at 0.25 of the upper arm,
  exactly where the cut opens and sleeve and vest are still one surface: it walked onto the vest on
  both arms (the left let go of nothing, the right of 5 vertices). From 0.30 it stays on the arm and
  555 and 1,280 vertices of the vest let go. Above the cut the vest's armhole *is* the arm's surface
  and takes half its weight, as any shoulder does; where the colours tell the torso's garment from
  the sleeve (Pip's are 100-104 apart in Lab) it now moves with the collarbone - 812 and 1,603
  vertices. With both arms overhead the vest stays down; the armpit between vest and sleeve stretches.
- **The cut left joins.** Checked slab by slab (5% of the upper arm, straight across the joints'
  line), five of seven characters still had the arm reaching the body somewhere in the cut: Vex's
  left arm at 0.70, her right at 0.40 and 0.55, Juno's left in four places. The rays that decide
  where an arm is glued go round a section 10 degrees apart and a web can survive between them, or
  where a bulging sleeve put the cut inside the sleeve. `free_arms.py` now checks its own result and
  cuts again where a join survives - every direction facing the body, the ring 12% further out a
  round (at most 1.6 arm radii). On the final rebuilds: 3-9 joined slabs an arm after the first cut,
  1-3 after the re-cut (Juno 9 -> 2 and 3 -> 1, Vex 6 -> 2 and 6 -> 1, Kaito 9 -> 3 on both arms);
  free are Wren's left arm, Pip's left and Mara's right. Where a join is left, the rig keeps that
  side of the garment with the arm, as before (open).
- **Every texture had islands folded onto themselves.** Smart UV Project, run on a smoothed copy of
  the mesh, laid islands over themselves wherever that copy curls: 224 (Juno) to 2,602 (Vex) faces
  per character shared texels with another surface, 1,529 on Pip - the backs of his sleeves showed
  his vest's paint. Repacking cannot part an island from itself (1,529 before and after); the
  overlapping faces are unwrapped again, finer, as islands of their own, and every island rescaled
  to one texel density before packing (without that, the new islands took 42% of Pip's atlas):
  6-31 faces left on the mesh that ships (Pip 8, Mara 6, Wren 8, Juno 24, Vex 30, Aoi 31), 97-158 on
  Rowan, Kaito and Knight, whose layered cloth and plates fold the most; the median texel density
  unchanged (0.77 -> 0.75).
- **Aoi's trousers had light streaks.** Not the old generated hands (masking where the images show
  them changed 12,513 texels and left the streaks) - her side view's hands stand 10-20 px off the
  mesh's, and the trouser texels just past the mesh's hand read the picture's. A generated view now
  fades out beside the outline of anything nearer (a depth jump between neighbouring pixels; a
  surface merely seen at a slant has none). Not in the reference: the mesh was made from it, and
  half of Aoi's face lies behind a lock of her hair.
- **The modelled hands' region took in a hip.** The hand is found as a region round the wrist cut;
  it took in the side of Aoi's hip (36,482 texels, painted skin) and 161,553 texels of Pip. A texel
  near a wrist is the hand's now only if it stands outside the solid the hand was joined to.
- **Pip's hands were brown.** Their tone was the median of every skin-like colour on the body -
  orange hair, khaki shorts and the vest's shading pass for skin: (137, 95, 71) against a peach
  face. The face's own texels decide it now: (222, 152, 122).
- **A vest's hem followed the thighs.** Its lowest part carried half its weight on them (median
  0.50, up to 0.70), and spreading the legs in a jump pulled the hem into a W; Juno's jacket wrapped
  round her thighs in a squat. Where the torso's colours and the upper thighs' differ, the upper
  garment near the hips follows the pelvis: Pip 816 vertices, Juno 1,118, Vex 2,154; one colour top
  to bottom (Mara, Rowan, Wren, Knight, Kaito, Aoi), nothing changes.
- **A reference in the wrong pose went all the way through.** Bo, a cartoon chef, came from
  Qwen-Image with his fists on his hips despite the A-pose spelled out, and the 3D model fused them:
  his hands came out in shreds at his belt, 32 minutes later. `pipeline/pose_gate.py` reads the
  arms of every new reference (the pipeline's own references: 0.60-0.79 torso lengths from wrist to
  hip; Bo's 0.33) and the stage draws it again, on a new seed, below 0.45.
- **Webs between arm and body.** Knight's generated armour filled the space between forearm and
  hip with membranes a few voxels thick, and waving stretched them into grey sheets hanging from his
  arms. `free_arms.py` now opens the solid by 3 voxels between each arm and the body before cutting:
  8,286 and 10,349 voxels of web on Knight, 1,200-12,000 on the others; pieces the cuts split off go.
  Thicker fins survive on Knight (open).
- **Loose pieces stayed behind.** Kaito's mesh came as 40 pieces - the body, two modelled hands and
  37 bits the generator left detached (the inner toes of his split-toe boots, hair spikes). With no
  path through the body the weights never reached them, and a floating spike hung by his head in
  every clip. Each now rides rigidly with the body part nearest it.
- **The web build refused Knight.** His armour's roughness and metal change texel by texel: the
  encoding alone held 28.5 dB at WebP q95 against a 32 dB gate, and the gate had also been counting
  the 2K downscale the build asks for. It now judges the encoding against the image it encoded,
  and steps q95 -> q100 -> lossless until one passes.
- **Ten characters did not fit the published playground.** Seven at 1K came to 56 MB of a 64 MB
  artifact. For that preview copy only (`tools/build_site.py`): skin weights as bytes, UVs and
  indices as 16-bit (core glTF, no extension), rotation keys interpolation reproduces to a quarter
  of a degree dropped (captures key every frame at 60 fps), 768 px textures, lossy normal maps - Pip
  6.9 -> 4.3 MB, and his clips play the same in three.js.
- **A rerun from the reference forgot the prompt.** `--from reference` read only the command line's
  `--prompt`; the one recorded with the character is used now (Bo's redraw failed on it).
- **An elbow folded back through the arm.** In Aoi's spell cast the hands press together in front
  of her chest, elbows out; for 16 frames her left elbow was bent to 176 deg, the forearm back through
  the upper arm, and the jacket sleeve riding the upper arm stuck out past the fold like a tube - the
  hand score caught it, DWPose reading the tube's end as her hand (mean 0.25 -> 0.33 palm lengths
  after the rebuild, median 0.17). The fit takes a bone's depth from its image length, which fixes it
  only up to a sign, and it took the capture's side for her upper arm: tipped back. The capture
  library's elbows pass 150 deg in 0.05% of frames (159 at most), and 1% of its wrists within the
  chest's width and height lie behind the torso, so an arm's two bones now choose their sides
  together against both (`video_motion.py`; the self-test on 40 moves the library lacks, 0.212 ->
  0.211 torso lengths). That reached 162 deg: where the video shows the upper arm at its full length
  no side is left to choose - the pose model puts her elbow at the outside of a puffy sleeve. The
  refine stage now holds every elbow under 150 deg (`relu(bend - 150)^2`): 150 deg, 4 frames at the
  limit, the silhouette (0.822 -> 0.823) and the hands (0.125 -> 0.127) where they were. Tried first
  and dropped: requiring a wrist over the torso in the image to be in front of it - from random
  cameras half of such wrists are behind the body, seen from its back.
- **Two characters died on the GPU.** Knight's and Kaito's 1024-resolution TRELLIS stage (~8k
  tokens) stopped with "Impacting Interactivity": macOS kills a GPU command buffer that holds up the
  display too long, and six transformer blocks went in one. Two at a time now, and a retry one at a
  time with fewer operations per buffer if it happens anyway.

## Found and fixed: characters made from a few words

Three characters were made from the Studio with short prompts, and they came out poor, each for its
own reason. Traced:

- **The prompt went to the image model as it was typed.** "A boy scout" came back a grown man in a
  scout uniform, and "gray alien that is extremely muscular and looks like a gigachad" a grey clay
  sculpture with no clothes and no colour. Nothing asked for an age, a build, clothing or colours. A
  local language model now writes a short prompt out first (`pipeline/describe.py`, Ollama; it is
  unloaded as soon as it answers). It fills in fields - who, build, skin, hair, face, top, bottom,
  shoes, up to two close-fitting accessories - and the sentence is put together from them. Asked
  for free prose under rules instead, llama3.2 3B and qwen2.5vl 7B broke them one prompt in two
  ("holding a wooden badge", "a small shield", "standing at 1.5 meters tall", a gray alien with a
  beard). So a field that names a prop, a pose or a height is dropped, and the age group comes from
  the prompt's own words: asked to fill in "a witch" and "a young knight", the 7B model made both
  children. Prompts of 20 words or more are used as written, and `--literal` keeps any prompt as it
  is. The Studio shows the written-out text before anything is drawn, to edit.
- **The height was the slider's.** The Studio sent 1.72 m for every character, so the boy scout was
  1.72 m. The height now follows the description (a child 1.2-1.5 m, an adult about 1.75) unless
  one is set.
- **A name already taken gave back the old character.** "A young knight in steel armor", named
  knight, ran every stage cached in 0.0 min. It also recorded its prompt, style (stylized) and
  height (1.72) over the existing Knight's, which the next rebuild of Knight would have used. The
  Studio now refuses a taken name as it is typed and offers a free one. `charforge.py make` will not
  write a new prompt, style or height over an existing character: it needs `--from` for that.
- **TRELLIS built a board into the model.** Gray's first pass was a flat grey board with the figure
  behind it. Knight's had a board behind him on all three seeds tried, and Kaito's one through his
  legs. From the front their silhouettes lie on their pictures at 0.35-0.44, against 0.90-0.94 for a
  sound model, and at TRELLIS's first stage they fill 1,604-3,209 cells of 32^3 against 906-1,401.
  The board is the picture's backdrop made solid, and the later stages had cut it into Knight's
  membranes and Kaito's loose pieces. It comes from the input route, not the seed. Through the image
  model's own cut-out (Qwen-Image's alpha), Knight made a board on seeds 7, 1007 and 2007. Through
  TRELLIS's own background remover on the same picture, his model is clean (0.92). A 20% margin
  round the figure made it a board and nothing else. Both passes are now checked from the front:
  below 0.8, the first pass is made again from the picture through the remover, then on a new seed,
  and the second pass follows the route that worked.
- **Two of them came out with no face rig.** The face camera frames the head from its joint to the
  top of the hair, and knight2's head joint sat 12 cm under the top: the frame stopped above his
  chin, and the chin, placed by a fixed distance below the eyes, fell off the surface. On the boy
  scout, the walk down the face's midline left the surface before it found the step to the neck.
  The frame now takes at least 11% of the height (a head is 12-13% of an adult), and the chin is
  the last surface point of that walk, or the nearest surface to the guess. Both have their jaw and
  five shapes. Checking the boy scout's also showed the face read in stairs: the render that gives
  the world position under each pixel wrote position + 4, and EEVEE's film keeps half floats, whose
  step between 4 and 8 is 3.9 mm - his eyelids and lips were placed 2.5 mm off on average, 5.9 at
  most. It now writes the position around the frame's centre (values near 1), where the step is
  0.5 mm; the same pixels come out, four times finer. New characters get it; the ones made before
  keep their shapes.
- **knight2's hands floated off his forearms.** The arm's traced centre line reached his gauntlet
  and doubled back up the vambrace for 36 cm, so the traced "fingertips" came out beside the elbow,
  the hand joint a forearm's length behind the wrist (every other character: 0.6-0.9 beyond it),
  and the forearm's direction at the wrist pointed back up the arm. The hand cut then carved the
  forearm away and kept the old hand, and the retopology closed the gap with dark fins. The trace
  now ends at its farthest point down the forearm, and the cut falls back to the elbow-to-wrist
  direction wherever the traced one turns back. No other character's joints change.
- **Gray had one human hand.** The modelled hands take the body's skin tone from texels that are
  skin-coloured - warm, mid-light - and a grey alien has none, so they fell back to a fixed peach
  (204, 160, 135): one hand came out peach, the other grey only where a side view painted over it.
  With no warm skin anywhere, the hands now take the face's own colour, whatever its hue. Of the
  thirteen characters only Gray took the fallback; the others read their faces as before.
- **Gray's hands were a person's on a hulk's forearms.** The modelled hand has a person's
  proportions and a wrist clipped to a person's (at most 0.17 of the hand's length across); Gray's
  forearms measure 11-13 cm at the wrist, so each ended in a flat step like a cuff, and the waving
  hand read as a small hook. A sleeve measures as wide (Pip's cuffs 0.41, knight2's gauntlets 0.38),
  so the width cannot tell them apart - the part labels can: the vertices just above the cut are
  "arms" on a bare forearm and "top" on a sleeve (Gray, Mara and the boy scout 0.92-1.00 skin, every
  sleeved or armoured character 0.13 or less). On a bare forearm the hand is now made broader and
  thicker to meet it, fingers included, up to 1.6x (Gray 1.6, Mara would take 1.16); the rig's hand
  region follows. Gray's wave is an open hand now. A faint ring still marks the join.
- **Knight's face was painted twice.** The picture's face is laid on the model's with a fine warp
  (the face flow), refused past 2.5% of the figure's height because on drawn faces a bigger warp
  painted Pip a second pair of eyes. Knight's generated face sat 2.5% from his picture's: refused,
  the picture went on unaligned - his fringe across one eye, every feature doubled - and the face
  stage, finding no clean eyes, gave him no blinks or jaw. Forcing the warp smeared his eyes into
  bands. The generator's own face is softer but lies where its geometry is, so for a realistic
  character whose face the warp refuses, the front view now leaves the face and fringe to it (the
  drawn styles keep the projection, which suits their large features). Knight's face reads as one
  face, and the face stage found both eyes and the mouth: blinks, jaw, smile, brows and pucker.
  Of the realistic characters only Knight hits the refusal.

## Found and fixed: seeing a move from every side

One camera cannot see depth, and each move from video had only been compared with its video from
the video's own camera. `tools/make_angles.py` (and **Angles** on each character's page in the
Studio) renders any clip from several cameras at once, side by side in one video. With `--video` the
move's video comes first. Every panel is orthographic, at one scale, on a 10 cm grid on the floor and
the wall behind. Each elbow's and knee's bend and each shoe's lowest point above the floor are read out
under the panels, red past what the motion library ever does.

It found:

- **Aoi's elbow folded back through her arm** (above).
- **The feet of every move from video stood off the floor.** The retarget plants a foot only where
  it judges the foot planted. Aoi's spell cast begins with neither foot judged so, and she stood
  5-7 cm up. The refine then turned hips, knees and ankles toward the video, moving the lowest sole
  by up to 3-4 cm either way, and nothing put it back. The step that applies the refine now reads
  the video's own floor line: the lowest pixel of its figure frame by frame, the median over the
  frames where the figure does not run off the bottom of the picture (Aoi's feet leave it in the
  middle of her cast). It moves the pelvis so that each key's lowest shoe stands as far above the
  floor as the video's figure does - on it while standing, as high as a hop - converted at the
  refine's own scale. Aoi's lowest shoe: 3.1 cm on average and up to 5.5 cm before; now 0.6 cm on average and 1.9 at most, in the stretch where her feet leave the video's frame and the retarget's planting is kept.
- **The video beside the rig ran 25% fast.** Mara's jab, in the angles video, came 0.2 s after the
  video's, and more as the clip went on; at the end the video stood still while she kept punching.
  The rig was not late: the fidelity score samples it at the video's own 24 fps and matches it frame
  for frame. The comparison was. `tools/side_by_side.py` (every move's `_compare.mp4`) and
  `make_angles.py` resampled the 24 fps video to the renders' 30 fps by taking a video frame only when
  its time came up, never twice, so 124 frames played at 30 fps and the last one held for the rest.
  They now repeat a frame when the output runs faster (the fit's own reader, `video_motion.py`, had
  the same loop but reads at the video's own rate, where it takes each frame once - fixed anyway). All
  four moves' compare and angles videos were remade; the jab, cross and hook now land on the same
  frames.
- **Mara's kick scores worse than it moves.** Its joint error jumps from about 0.1 to 0.46 at the top
  of the kick, which read as the rig folding the leg. The side and top views say otherwise: her leg is
  up at head height as in the video. Held joint by joint, her skeleton is 0.12-0.14 torso lengths off
  the video in those frames, as through the rest of the kick; the pose model reading the *render* puts
  her right ankle 2.9 torso lengths from the skeleton's, in frames 61-67. Without those 7 frames the
  kick scores 0.049 instead of 0.069. What the rig does get wrong there: it lowers the leg a little
  early (knee bent 70° at 2.8 s, the video's leg still up). `tools/motion_stages.py`, which splits an
  error into fit, retarget, rig and render, kept its skeleton dumps forever and was scoring a rig two
  rebuilds old; it now reads them again when the blend or FBX is newer.

## Found and fixed: jobs sharing the GPU

Two refines run at the same time corrupted each other. Aoi's spell cast went from 0.672 to 0.647 in
silhouette overlap and Mara's kick from 0.733 to 0.485, where each alone reaches 0.82-0.86: every
loss rose in the first 50 steps and never came back, and the refine stage rightly refused both
results. A TRELLIS pass beside another was stopped twice by macOS ("Impacting Interactivity"), its
retry included. And a refine beside another job's part labeller or skeleton reader (PyTorch too)
began from a silhouette term of 2.33 instead of 0.0013 and ended in NaN - run alone, Pip's was normal
with the elbow term and without it. Every GPU step - TRELLIS, the image model's calls, the prompt
model, every Blender render and bake, the scripts that run a model (the pose check, the part labeller,
the skeleton reader, the face landmarks, the refine), a move's video, masks and readings, the angles
renders - now takes one lock on this Mac (`work/.gpu.lock`, `charforge.gpu()`), and a second job waits
for it and says so.

## Still out of reach

- **A sleeve that ends past the elbow.** Bent far, an elbow folds the forearm back, but the part of a
  jacket sleeve beyond the joint keeps pointing along the upper arm, as linear skinning does. In Aoi's
  spell cast (elbows 101-150 deg) her sleeve cuffs stick out sideways, and DWPose reads their light
  lining as her hands in 21 frames: 53% of her hand error's mean (0.33, median 0.17). Her own hands lie
  where the video's do. Corrective shapes at the elbow, or sleeves generated apart from the arm, would
  reach it.


Hair and bags generated lying against the body cannot swing without tearing - the fix is
upstream, in how they are generated. A garment fused to the one under it (Pip's vest) now moves
with the body it belongs to, but the surface between them has to stretch with the arms overhead.
A move from a video borrows its depth from the nearest capture, so a move unlike anything in the
library comes out right in the image and approximate in depth. The pose model fails on poses it
rarely saw - at the top of Mara's kick it loses the raised leg on the render of her (7 frames),
which inflates that move's score though the rig kicks as the video does. The hands follow DWPose
where it is sure of them; blurred by a fast move, under hair or overhead at the frame's edge (Pip's)
they keep the capture's, and cartoon fingers it reads unsteadily. See CONTRIBUTING.md.
