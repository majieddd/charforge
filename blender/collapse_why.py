"""What do the collapsing faces have in common?

The deformation audit says 6-9% of jacket faces fall below half their rest area. The roadmap
assumes the cause is missing edge loops at the joints, but that was a guess, and guesses have
been expensive on this project. Four candidate explanations, all cheap to test, each implying a
different fix:

  small at rest     a face that is tiny to begin with swings a large *relative* area for a small
                    absolute motion. Fix: uniform remeshing, not joint loops.
  near a joint      the classic candy-wrapper. Fix: edge loops and weight gradient resolution
                    at the joint.
  weight gradient   the face spans a steep change in the weight field, so its corners follow
                    different bones. Fix: smoother weights, or more geometry to spread the
                    gradient over.
  already thin      a sliver face (bad aspect ratio) has little area to lose. Fix: retopology
                    quality.

Reports each factor as a lift: how much more likely a face is to collapse given the property,
versus the base rate. A lift near 1 means the property explains nothing.

Run: blender -b -noaudio --python collapse_why.py -- --blend final.blend --clip walk
"""
import argparse
import sys
from collections import defaultdict

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clip", default="walk")
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--thresh", type=float, default=0.5)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type == "ARMATURE")
meshes = [o for o in scene.objects if o.type == "MESH" and o.find_armature() == rig]

act = bpy.data.actions.get(a.clip)
rig.animation_data.action = act
if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
    rig.animation_data.action_slot = act.slots[0]
f0, f1 = (int(x) for x in act.frame_range)

M = rig.matrix_world
joints = np.array([(M @ b.head_local)[:] for b in rig.data.bones]
                  + [(M @ b.tail_local)[:] for b in rig.data.bones])

for ob in meshes:
    me = ob.data
    nf = len(me.polygons)
    names = [vg.name for vg in ob.vertex_groups]
    faces = [list(p.vertices) for p in me.polygons]

    rest_area = np.empty(nf)
    me.polygons.foreach_get("area", rest_area)
    ctr = np.empty(nf * 3)
    me.polygons.foreach_get("center", ctr)
    ctr = ctr.reshape(-1, 3)

    W = np.zeros((len(me.vertices), len(names)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)

    # --- candidate properties -------------------------------------------------------------
    # distance from the face to the nearest joint, in units of the local face size
    d_joint = np.min(np.linalg.norm(ctr[:, None, :] - joints[None, :, :], axis=2), axis=1)
    face_len = np.sqrt(np.maximum(rest_area, 1e-12))

    # weight gradient across the face: how far apart its corners' weight vectors are
    wgrad = np.zeros(nf)
    for i, f in enumerate(faces):
        w = W[f]
        wgrad[i] = np.abs(w - w.mean(0)).sum(1).max()

    # aspect ratio: longest edge over shortest, in the rest pose
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3)
    aspect = np.ones(nf)
    for i, f in enumerate(faces):
        p = V[f]
        e = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
        aspect[i] = e.max() / max(e.min(), 1e-9)

    # --- which faces actually collapse, over the clip --------------------------------------
    worst = np.ones(nf)
    for fr in range(f0, f1 + 1, a.step):
        scene.frame_set(fr)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(dg)
        m2 = ev.to_mesh()
        A = np.empty(len(m2.polygons))
        m2.polygons.foreach_get("area", A)
        worst = np.minimum(worst, A / np.maximum(rest_area, 1e-12))
        ev.to_mesh_clear()
    bad = worst < a.thresh
    base = bad.mean()
    if base == 0:
        print(f"[why] {ob.name}: nothing collapses")
        continue

    print(f"\n[why] {ob.name}: {int(bad.sum()):,}/{nf:,} faces collapse below "
          f"{a.thresh:.0%} of rest area ({100*base:.2f}%)", flush=True)

    def lift(name, prop, hi=True):
        """P(collapse | top decile of prop) / P(collapse)"""
        q = np.percentile(prop, 90 if hi else 10)
        sel = prop >= q if hi else prop <= q
        if sel.sum() == 0:
            return
        r = bad[sel].mean() / base
        share = bad[sel].sum() / bad.sum()
        print(f"[why]    {name:<34s} lift {r:5.2f}x   holds {100*share:5.1f}% of all collapses",
              flush=True)

    lift("smallest decile by rest area", rest_area, hi=False)
    lift("nearest decile to a joint", d_joint, hi=False)
    lift("steepest decile of weight gradient", wgrad, hi=True)
    lift("worst decile of aspect ratio", aspect, hi=True)

    # how concentrated is it? if a few bones own it, the fix is local
    dom = np.array([np.bincount(np.argmax(W[f], axis=1), minlength=len(names)).argmax()
                    for f in faces])
    cnt = defaultdict(int)
    for b in dom[bad]:
        cnt[names[b]] += 1
    top = sorted(cnt.items(), key=lambda kv: -kv[1])[:6]
    tot = sum(cnt.values())
    print(f"[why]    collapses by dominant bone: "
          + ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in top), flush=True)

    # and what does the weight field look like there vs everywhere?
    print(f"[why]    weight gradient: collapsing {wgrad[bad].mean():.3f} vs "
          f"rest of mesh {wgrad[~bad].mean():.3f}", flush=True)
    print(f"[why]    rest area:       collapsing {rest_area[bad].mean():.6f} vs "
          f"rest of mesh {rest_area[~bad].mean():.6f}", flush=True)
    print(f"[why]    distance to joint (in face widths): collapsing "
          f"{(d_joint/face_len)[bad].mean():.1f} vs {(d_joint/face_len)[~bad].mean():.1f}",
          flush=True)
