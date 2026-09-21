"""Second-pass scan: is the character actually standing up?

The first scan separated clips by motion - speed, flight phase, cadence, hand height - and that
was enough to find a run and a jump. It was not enough to find an idle: a *seated* idle has zero
root speed and low joint motion, scores exactly like a standing one, and retargets onto the rig
as a character crouching in mid-air. Measuring posture directly is what separates them.

  thigh_down   mean cosine between the thigh bone and world-down. 1.0 is standing, ~0 is sitting
  torso_up     mean cosine between the spine and world-up. Catches crouches and sneaks
  both are averaged over the clip, so a squat mid-jump does not disqualify a standing clip
"""
import argparse, glob, json, math, os, sys
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--files", required=True, help="JSON list of filenames to check")
ap.add_argument("--out", required=True)
a = ap.parse_args(argv)

names = json.load(open(a.files))
out = {}
for i, name in enumerate(names):
    path = os.path.join(a.dir, name)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.fbx(filepath=path, ignore_leaf_bones=True,
                                 automatic_bone_orientation=False)
    except Exception as e:
        out[name] = {"error": str(e)[:100]}
        continue
    rig = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    act = rig.animation_data.action if rig and rig.animation_data else None
    if not act:
        out[name] = {"error": "no action"}
        continue
    f0, f1 = (int(x) for x in act.frame_range)
    SW = rig.matrix_world
    thigh = next((rig.pose.bones[n] for n in ("mixamorig:LeftUpLeg", "LeftUpLeg")
                  if n in rig.pose.bones), None)
    spine = next((rig.pose.bones[n] for n in ("mixamorig:Spine1", "mixamorig:Spine", "Spine1")
                  if n in rig.pose.bones), None)
    if not (thigh and spine):
        out[name] = {"error": "missing bones"}
        continue
    td, tu = [], []
    for f in range(f0, f1 + 1, max(1, (f1 - f0) // 12)):
        bpy.context.scene.frame_set(f)
        ty = ((SW @ thigh.matrix).to_3x3() @ Vector((0, 1, 0))).normalized()
        sy = ((SW @ spine.matrix).to_3x3() @ Vector((0, 1, 0))).normalized()
        td.append(ty.dot(Vector((0, 0, -1))))
        tu.append(sy.dot(Vector((0, 0, 1))))
    out[name] = {"thigh_down": round(sum(td) / len(td), 3),
                 "torso_up": round(sum(tu) / len(tu), 3)}
    if i % 20 == 0:
        print(f"[posture] {i}/{len(names)}", flush=True)
json.dump(out, open(a.out, "w"), indent=2)
print(f"[posture] {len(out)} -> {a.out}", flush=True)
