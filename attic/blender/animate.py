"""Author a suite of game animations on a CharForge rig, and export them in one GLB.

These are procedural cycles, not motion capture, but they are sampled from published gait curves
rather than eyeballed - see gait.py, which also records why the first version of this file was
not fit to judge the rig with (it never moved the arms). Bone names follow the SMPL-H body
layout that skeleton.py emits, which is also what Mixamo and the text-to-motion models map onto,
so retarget.py can drop a captured clip in beside these without touching the rig.

Run: blender -b -noaudio --python animate.py -- --rig rigged.glb --out animated.glb
"""
import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Euler, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--rig", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=30)
a = ap.parse_args(argv)

R = math.radians


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gait import ARM_DOWN, RUN, WALK, locomotion_pose, sample  # noqa: E402


def _locomotion(nframes, curves, lean=0.0, bob=3.0, sway=2.0, stride_per_frame=None):
    """Dense per-frame tracks for a looping locomotion cycle."""
    tracks, root = {}, []
    for f in range(nframes):
        phase = f / nframes
        pose, r = locomotion_pose(curves, phase, lean=lean, bob_cm=bob, sway_cm=sway)
        for bone, ang in pose.items():
            tracks.setdefault(bone, []).append((f + 1, ang))
        root.append((f + 1, r))
    # close the loop exactly on the last frame
    pose, r = locomotion_pose(curves, 0.0, lean=lean, bob_cm=bob, sway_cm=sway)
    for bone, ang in pose.items():
        tracks[bone].append((nframes + 1, ang))
    root.append((nframes + 1, r))
    return tracks, root


def _idle(nframes=90):
    """Breathing plus a slow weight shift. Stillness is what makes a rig look dead."""
    tracks, root = {}, []
    for f in range(nframes + 1):
        t = f / nframes
        breath = math.sin(2 * math.pi * t * 2)          # two breaths per loop
        shift = math.sin(2 * math.pi * t)
        put = lambda b, v: tracks.setdefault(b, []).append((f + 1, v))
        put("spine2", (-1.2 * breath, 0, 0))
        put("spine3", (0.8 * breath, 0, 0))
        put("neck", (0.6 * breath, 1.5 * shift, 0))
        put("pelvis", (0, 0.8 * shift, 0))
        put("left_shoulder", (1.5 * breath, 0, ARM_DOWN))
        put("right_shoulder", (1.5 * breath, 0, -ARM_DOWN))
        put("left_elbow", (-(10 + 2 * breath), 0, 0))
        put("right_elbow", (-(10 + 2 * breath), 0, 0))
        put("left_hip", (0, 0, -1.5 * shift))
        put("right_hip", (0, 0, -1.5 * shift))
        root.append((f + 1, (0.004 * shift, 0.0, 0.004 * breath)))
    return tracks, root


def _jump(nframes=48):
    """Crouch, launch, airborne tuck, land, absorb. The old version had no airtime at all."""
    tracks, root = {}, []
    # phase boundaries as fractions of the clip
    for f in range(nframes + 1):
        t = f / nframes
        if t < 0.22:                     # crouch
            k = t / 0.22
            crouch, lift, air = k, 0.0, 0.0
        elif t < 0.34:                   # extension / launch
            k = (t - 0.22) / 0.12
            crouch, lift, air = 1 - k, k, 0.0
        elif t < 0.66:                   # airborne: a real ballistic arc
            k = (t - 0.34) / 0.32
            crouch, lift, air = 0.0, 1.0, math.sin(math.pi * k)
        elif t < 0.80:                   # landing absorb
            k = (t - 0.66) / 0.14
            crouch, lift, air = k, 1 - k, 0.0
        else:                            # recover to stand
            k = (t - 0.80) / 0.20
            crouch, lift, air = 1 - k, 0.0, 0.0
        tuck = air
        put = lambda b, v: tracks.setdefault(b, []).append((f + 1, v))
        put("left_hip", (55 * crouch + 40 * tuck, 0, 0))
        put("right_hip", (55 * crouch + 40 * tuck, 0, 0))
        put("left_knee", (-(75 * crouch + 70 * tuck), 0, 0))
        put("right_knee", (-(75 * crouch + 70 * tuck), 0, 0))
        put("left_ankle", (-25 * crouch + 30 * lift - 10 * tuck, 0, 0))
        put("right_ankle", (-25 * crouch + 30 * lift - 10 * tuck, 0, 0))
        put("spine2", (18 * crouch - 6 * lift, 0, 0))
        put("spine3", (8 * crouch - 3 * lift, 0, 0))
        # arms drive the jump: swing back in the crouch, up through launch
        swing = 40 * crouch - 95 * lift - 70 * tuck        # back in the crouch, up on launch
        put("left_shoulder", (swing, 0, ARM_DOWN * (1 - 0.5 * (lift + tuck))))
        put("right_shoulder", (swing, 0, -ARM_DOWN * (1 - 0.5 * (lift + tuck))))
        put("left_elbow", (-(20 + 30 * crouch), 0, 0))
        put("right_elbow", (-(20 + 30 * crouch), 0, 0))
        root.append((f + 1, (0.0, 0.0, -0.16 * crouch + 0.62 * air)))
    return tracks, root


def _turn(nframes=60):
    """A stepping 180, not a turntable. The old clip rotated the root rigidly."""
    tracks, root = {}, []
    for f in range(nframes + 1):
        t = f / nframes
        yaw = 180.0 * (0.5 - 0.5 * math.cos(math.pi * min(t / 0.85, 1.0)))
        step = math.sin(2 * math.pi * min(t / 0.85, 1.0) * 2)
        put = lambda b, v: tracks.setdefault(b, []).append((f + 1, v))
        put("pelvis", (0, yaw, 0))
        put("spine2", (0, -8 * math.sin(math.pi * min(t / 0.85, 1.0)), 0))
        put("left_hip", (18 * step, 0, 0))
        put("right_hip", (-18 * step, 0, 0))
        put("left_knee", (-max(0.0, 35 * step), 0, 0))
        put("right_knee", (-max(0.0, -35 * step), 0, 0))
        put("left_ankle", (-8 * step, 0, 0))
        put("right_ankle", (8 * step, 0, 0))
        put("left_shoulder", (12 * step, 0, ARM_DOWN))
        put("right_shoulder", (-12 * step, 0, -ARM_DOWN))
        put("left_elbow", (-18, 0, 0))
        put("right_elbow", (-18, 0, 0))
        root.append((f + 1, (0.0, 0.0, 0.012 * abs(step))))
    return tracks, root


def _wave(nframes=60):
    """Raise the right arm and actually wave it, with the torso engaged."""
    tracks, root = {}, []
    for f in range(nframes + 1):
        t = f / nframes
        raise_k = min(1.0, t / 0.25) if t < 0.8 else max(0.0, (1.0 - t) / 0.2)
        wag = math.sin(2 * math.pi * t * 3) * raise_k
        put = lambda b, v: tracks.setdefault(b, []).append((f + 1, v))
        # waving arm goes up via abduction (local Z), away from the hanging rest pose
        put("right_shoulder", (-10 * raise_k, 0, -ARM_DOWN + (ARM_DOWN + 55) * raise_k))
        put("right_elbow", (-(55 * raise_k + 18 * wag), 0, 0))
        put("right_wrist", (0, 0, 25 * wag))
        put("left_shoulder", (0, 0, ARM_DOWN))
        put("left_elbow", (-14, 0, 0))
        put("spine2", (0, 6 * raise_k, -4 * raise_k))
        put("neck", (0, -5 * raise_k, 0))
        put("pelvis", (0, 3 * raise_k, 0))
        root.append((f + 1, (0.0, 0.0, 0.0)))
    return tracks, root


def clips():
    """name -> (frames, loop, tracks, root) with dense per-frame samples."""
    walk_t, walk_r = _locomotion(40, WALK, lean=2.0, bob=3.0, sway=2.0)
    run_t, run_r = _locomotion(26, RUN, lean=9.0, bob=5.5, sway=1.4)
    idle_t, idle_r = _idle()
    jump_t, jump_r = _jump()
    turn_t, turn_r = _turn()
    wave_t, wave_r = _wave()
    return {
        "idle": (90, True, idle_t, idle_r),
        "walk": (40, True, walk_t, walk_r),
        "run": (26, True, run_t, run_r),
        "jump": (48, False, jump_t, jump_r),
        "wave": (60, False, wave_t, wave_r),
        "turn": (60, False, turn_t, turn_r),
    }


def action_fcurves(act):
    """Blender 4.4+ moved fcurves into action layers/slots/channelbags."""
    if hasattr(act, "fcurves"):
        return list(act.fcurves)
    out = []
    for layer in act.layers:
        for strip in layer.strips:
            for slot in act.slots:
                cb = strip.channelbag(slot)
                if cb:
                    out.extend(cb.fcurves)
    return out


def main():
    if a.rig.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=a.rig)   # keeps part_id attribute + vertex groups
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=a.rig)
    rig = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if rig is None:
        raise SystemExit("no armature in " + a.rig)
    bpy.context.view_layer.objects.active = rig
    bpy.context.scene.render.fps = a.fps

    made = []
    for name, (nframes, loop, tracks, root) in clips().items():
        act = bpy.data.actions.new(name)
        act.use_fake_user = True
        if rig.animation_data is None:
            rig.animation_data_create()
        rig.animation_data.action = act
        bpy.ops.object.mode_set(mode="POSE")
        for pb in rig.pose.bones:
            pb.rotation_mode = "XYZ"
            pb.rotation_euler = Euler((0, 0, 0))
            pb.location = Vector((0, 0, 0))
        # Every clip keyframes every bone, even the ones it does not move. An action only
        # drives the channels it contains, so a clip that animates just the pelvis would
        # otherwise inherit whatever pose the previously played clip left behind - in Blender
        # and in the engine.
        for pb in rig.pose.bones:
            if pb.name not in tracks:
                pb.rotation_euler = Euler((0, 0, 0), "XYZ")
                pb.keyframe_insert("rotation_euler", frame=1)
        # Root translation: bob, sway and the jump arc live on the pelvis location channel.
        pelvis = rig.pose.bones.get("pelvis")
        if pelvis is not None and root:
            for f, (x, y, z) in root:
                pelvis.location = Vector((x, y, z))
                pelvis.keyframe_insert("location", frame=f)

        for bone, keys in tracks.items():
            pb = rig.pose.bones.get(bone)
            if pb is None:
                print(f"  [animate] {name}: no bone {bone!r}, skipped")
                continue
            for f, (rx, ry, rz) in keys:
                pb.rotation_euler = Euler((R(rx), R(ry), R(rz)), "XYZ")
                pb.keyframe_insert("rotation_euler", frame=max(1, f))
        bpy.ops.object.mode_set(mode="OBJECT")
        curves = action_fcurves(act)
        for fc in curves:
            for kp in fc.keyframe_points:
                kp.interpolation = "BEZIER"
                kp.easing = "EASE_IN_OUT"
        made.append({"name": name, "frames": nframes, "loop": loop,
                     "bones": sorted(set(tracks) & {b.name for b in rig.pose.bones})})
        print(f"  [animate] {name}: {nframes} frames, {len(curves)} curves")

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = max(c["frames"] for c in made)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=a.out, export_format="GLB", use_selection=True,
                              export_animations=True, export_animation_mode="ACTIONS",
                              export_nla_strips=False, export_bake_animation=True,
                              export_attributes=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.splitext(a.out)[0] + ".blend")
    json.dump({"clips": made, "fps": a.fps}, open(os.path.splitext(a.out)[0] + "_clips.json", "w"), indent=2)
    print(f"[animate] {len(made)} clips -> {a.out}")


main()
