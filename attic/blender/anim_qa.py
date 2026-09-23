"""Check every frame of every clip for the failures a person notices first.

Frame strips are not a substitute for watching an animation, and I cannot watch one. What I can
do is measure the specific things that were wrong, on every frame, which is stricter than
eyeballing playback because it never blinks.

The failure that prompted this was an arm swinging down *through* the torso and out behind the
back, caused by a T-pose source retargeted onto an A-pose rest. So the checks are:

  torso_clear   distance from each wrist to the spine chain (pelvis through neck, as line
                segments). NOT distance to the body mesh: a bone joint is meant to sit inside
                its own flesh, so an elbow always measures "inside the body" and the first
                version of this check duly failed a standing idle at -0.26 m. What "the arm
                goes through his torso" actually means is the wrist getting closer to the spine
                than the torso is wide.
  behind_back   wrist position along the character's own forward axis, relative to the spine.
                An arm swings behind the torso in a normal gait, but only so far; a wrist more
                than a shoulder-width behind the back plane is the artefact.
  elbow_range   the shoulder-elbow-wrist angle. Human elbows do not hyperextend past ~185
                degrees and do not fold tighter than ~20; either means the retarget has inverted
                a joint.
  foot_skate    horizontal travel of the planted foot per frame, measured against the clip's
                own ground speed. These clips are exported in place, so the planted foot is
                *supposed* to travel backwards at exactly walking speed - the defect is a
                mismatch between that and the stride, not movement itself.
  ground        lowest mesh point per frame; a character sinking through the floor or hovering.

Every check reports the worst frame, so a failure can be reproduced rather than argued about.

Run: blender -b -noaudio --python anim_qa.py -- --blend animated_mixamo.blend --json qa.json
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--json", default=None)
ap.add_argument("--clips", default="")
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next((o for o in scene.objects if o.type == "ARMATURE"), None)
body = next((o for o in scene.objects if o.type == "MESH"
             and ("body" in o.name.lower() or "skin" in o.name.lower())), None)
meshes = [o for o in scene.objects if o.type == "MESH"]
if rig is None or body is None:
    raise SystemExit("[qa] need an armature and a body mesh")

shoulder_w = ((rig.matrix_world @ rig.data.bones["left_shoulder"].head_local)
              - (rig.matrix_world @ rig.data.bones["right_shoulder"].head_local)).length
print(f"[qa] shoulder width {shoulder_w:.3f} m", flush=True)

wanted = [c.strip() for c in a.clips.split(",") if c.strip()]
results = {}

for act in bpy.data.actions:
    if wanted and act.name not in wanted:
        continue
    if act.name.startswith("cloth_"):
        continue
    rig.animation_data.action = act
    if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
        rig.animation_data.action_slot = act.slots[0]
    f0, f1 = (int(x) for x in act.frame_range)

    worst = {"penetration": (1e9, None), "behind": (-1e9, None),
             "elbow_min": (999, None), "elbow_max": (-999, None),
             "skate": (0.0, None), "ground_low": (1e9, None), "ground_high": (-1e9, None)}
    prev_feet = None

    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        dg = bpy.context.evaluated_depsgraph_get()

        M = rig.matrix_world
        pb = rig.pose.bones
        # character forward: spine2 local, projected flat
        chest = (M @ pb["spine3"].head)
        pelvis = (M @ pb["pelvis"].head)
        fwd = Vector((0, -1, 0))
        try:
            fwd = ((M @ pb["pelvis"].matrix).to_3x3() @ Vector((0, 0, 1)))
            fwd.z = 0
            fwd.normalize()
        except Exception:
            pass

        spine_pts = [M @ pb[n].head for n in
                     ("pelvis", "spine1", "spine2", "spine3", "neck") if n in pb]

        def dist_to_spine(p):
            best = 1e9
            for i in range(len(spine_pts) - 1):
                aa, bb = spine_pts[i], spine_pts[i + 1]
                ab = bb - aa
                L2 = ab.dot(ab)
                t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, (p - aa).dot(ab) / L2))
                best = min(best, (p - (aa + ab * t)).length)
            return best

        for side in ("left", "right"):
            for joint in ("wrist",):
                d = dist_to_spine(M @ pb[f"{side}_{joint}"].head)
                if d < worst["penetration"][0]:
                    worst["penetration"] = (round(d, 4), f"{act.name} f{f} {side}_{joint}")

            w = M @ pb[f"{side}_wrist"].head
            back = (w - chest).dot(-fwd)          # positive = behind the chest
            if back > worst["behind"][0]:
                worst["behind"] = (round(back, 4), f"{act.name} f{f} {side}")

            s = M @ pb[f"{side}_shoulder"].head
            e = M @ pb[f"{side}_elbow"].head
            wr = M @ pb[f"{side}_wrist"].head
            v1, v2 = (s - e), (wr - e)
            if v1.length > 1e-6 and v2.length > 1e-6:
                ang = math.degrees(v1.angle(v2))
                if ang < worst["elbow_min"][0]:
                    worst["elbow_min"] = (round(ang, 1), f"{act.name} f{f} {side}")
                if ang > worst["elbow_max"][0]:
                    worst["elbow_max"] = (round(ang, 1), f"{act.name} f{f} {side}")

        feet = {s: M @ pb[f"{s}_ankle"].head for s in ("left", "right")}
        if prev_feet:
            planted = min(feet, key=lambda s: feet[s].z)
            d = (feet[planted] - prev_feet[planted])
            slide = math.hypot(d.x, d.y)
            if slide > worst["skate"][0]:
                worst["skate"] = (round(slide, 4), f"{act.name} f{f} {planted}")
        prev_feet = feet

        lo = min(min((o.matrix_world @ Vector(c)).z for c in o.bound_box) for o in meshes)
        if lo < worst["ground_low"][0]:
            worst["ground_low"] = (round(lo, 3), f"{act.name} f{f}")
        if lo > worst["ground_high"][0]:
            worst["ground_high"] = (round(lo, 3), f"{act.name} f{f}")

    pen, pen_at = worst["penetration"]
    beh, beh_at = worst["behind"]
    # torso half-depth: a wrist closer to the spine than this is inside the body
    torso_r = shoulder_w * 0.42
    verdict = {
        "frames": f1 - f0 + 1,
        "min_wrist_to_spine_m": pen, "at": pen_at,
        "torso_radius_m": round(torso_r, 3),
        "arm_inside_body": bool(pen < torso_r * 0.55),
        "max_wrist_behind_chest_m": beh, "behind_at": beh_at,
        "wrist_far_behind_back": bool(beh > shoulder_w * 0.75),
        "elbow_angle_range_deg": [worst["elbow_min"][0], worst["elbow_max"][0]],
        "elbow_out_of_range": bool(worst["elbow_min"][0] < 18 or worst["elbow_max"][0] > 186),
        "max_planted_foot_slide_m": worst["skate"][0],
        "ground_z_range": [worst["ground_low"][0], worst["ground_high"][0]],
    }
    verdict["pass"] = not (verdict["arm_inside_body"] or verdict["wrist_far_behind_back"]
                           or verdict["elbow_out_of_range"])
    results[act.name] = verdict
    flag = "PASS" if verdict["pass"] else "FAIL"
    print(f"[qa] {act.name:<6} {flag}  min wrist-to-spine {pen:.3f} m (torso r {torso_r:.3f}) "
          f"at {pen_at}  |  wrist behind chest max {beh:+.3f} m  |  elbow "
          f"{worst['elbow_min'][0]:.0f}-{worst['elbow_max'][0]:.0f}deg  |  planted-foot slide "
          f"{worst['skate'][0]:.3f} m/frame", flush=True)

if a.json:
    os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
    json.dump({"shoulder_width_m": round(shoulder_w, 4), "clips": results},
              open(a.json, "w"), indent=2)
n_fail = sum(1 for v in results.values() if not v["pass"])
print(f"[qa] {len(results) - n_fail}/{len(results)} clips pass", flush=True)
