"""Is the remaining collapse the weights, or linear blend skinning itself?

Every weighting scheme tried so far bottoms out around the same place: clothing faces collapsing
to a few percent of their rest area at bent elbows and shoulders, with normals swinging past
150 degrees. That is the signature of linear blend skinning, not of bad weights - LBS averages
the bone *matrices*, and the average of two rotations 90 degrees apart is a matrix that shrinks
what it touches. The tighter the bend, the more volume disappears. It is the classic
candy-wrapper, and no amount of re-weighting removes it.

Dual quaternion skinning blends the rotations as rotations instead, so a vertex between two
bones travels along an arc rather than a chord and the volume survives. It is what game engines
use for exactly this case.

This applies both to the *same* weights, frame by frame, and reports the same measurements the
deformation audit uses. If DQS moves area and swing sharply while LBS does not, the remaining
problem is the skinning algorithm and belongs in the viewer, not in another weight solve.

Run: blender -b -noaudio --python dqs_test.py -- --blend animated.blend
"""
import argparse
import json
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--blend", required=True)
ap.add_argument("--clips", default="walk,run,idle")
ap.add_argument("--step", type=int, default=3)
ap.add_argument("--json", default=None)
a = ap.parse_args(argv)

bpy.ops.wm.open_mainfile(filepath=a.blend)
scene = bpy.context.scene
rig = next(o for o in scene.objects if o.type == "ARMATURE")
meshes = [o for o in scene.objects if o.type == "MESH" and o.find_armature() == rig]


def mat_to_dq(T):
    """4x4 rigid transform -> (real, dual) quaternion pair, both as (w,x,y,z)."""
    R, t = T[:3, :3], T[:3, 3]
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / np.linalg.norm(q)
    tq = np.array([0.0, t[0], t[1], t[2]])
    # dual = 0.5 * t_quat * q
    w0, x0, y0, z0 = tq
    w1, x1, y1, z1 = q
    d = 0.5 * np.array([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1])
    return q, d


def face_stats(P, faces, rest_area, rest_nrm):
    A = np.zeros(len(faces))
    N = np.zeros((len(faces), 3))
    for k, f in enumerate(faces):
        p = P[f]
        n = np.zeros(3)
        for i in range(len(f)):                     # Newell's method, any polygon
            c, d = p[i], p[(i + 1) % len(f)]
            n += np.cross(c, d)
        A[k] = 0.5 * np.linalg.norm(n)
        L = np.linalg.norm(n)
        N[k] = n / L if L > 1e-12 else (0, 0, 1)
    ratio = A / np.maximum(rest_area, 1e-12)
    swing = np.degrees(np.arccos(np.clip((N * rest_nrm).sum(1), -1, 1)))
    return ratio, swing


out = {}
for ob in meshes:
    me = ob.data
    n = len(me.vertices)
    names = [vg.name for vg in ob.vertex_groups]
    R = np.empty(n * 3)
    me.vertices.foreach_get("co", R)
    R = R.reshape(-1, 3)
    W = np.zeros((n, len(names)))
    for v in me.vertices:
        for g in v.groups:
            W[v.index, g.group] = g.weight
    W /= np.maximum(W.sum(1, keepdims=True), 1e-12)
    faces = [list(p.vertices) for p in me.polygons]
    ra = np.empty(len(faces))
    me.polygons.foreach_get("area", ra)
    rn = np.empty(len(faces) * 3)
    me.polygons.foreach_get("normal", rn)
    rn = rn.reshape(-1, 3)
    Rh = np.concatenate([R, np.ones((n, 1))], 1)

    res = {}
    for clip in [c.strip() for c in a.clips.split(",")]:
        act = bpy.data.actions.get(clip)
        if act is None:
            continue
        rig.animation_data.action = act
        if hasattr(rig.animation_data, "action_slot") and getattr(act, "slots", None):
            rig.animation_data.action_slot = act.slots[0]
        f0, f1 = (int(x) for x in act.frame_range)
        acc = {"lbs": ([], []), "dqs": ([], [])}
        for fr in range(f0, f1 + 1, a.step):
            scene.frame_set(fr)
            T = np.zeros((len(names), 4, 4))
            for bi, nm in enumerate(names):
                pb = rig.pose.bones.get(nm)
                if pb is None:
                    T[bi] = np.eye(4)
                    continue
                T[bi] = np.array(pb.matrix @ pb.bone.matrix_local.inverted())

            P_lbs = np.einsum('nb,bij,nj->ni', W, T, Rh)[:, :3]

            q = np.zeros((len(names), 4))
            dq = np.zeros((len(names), 4))
            for bi in range(len(names)):
                q[bi], dq[bi] = mat_to_dq(T[bi])
            # antipodal fix: every bone's quaternion must lie in the same hemisphere as the
            # pivot, or blending two equivalent rotations cancels them out
            piv = q[np.argmax(W.sum(0))]
            sgn = np.where((q @ piv) < 0, -1.0, 1.0)[:, None]
            qb = W @ (q * sgn)
            db = W @ (dq * sgn)
            L = np.linalg.norm(qb, axis=1, keepdims=True)
            qb, db = qb / np.maximum(L, 1e-12), db / np.maximum(L, 1e-12)
            w, x, y, z = qb[:, 0], qb[:, 1], qb[:, 2], qb[:, 3]
            # rotate
            rot = np.stack([
                1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1
            ).reshape(-1, 3, 3)
            t = 2.0 * np.stack([
                -db[:, 0] * x + db[:, 1] * w - db[:, 2] * z + db[:, 3] * y,
                -db[:, 0] * y + db[:, 1] * z + db[:, 2] * w - db[:, 3] * x,
                -db[:, 0] * z - db[:, 1] * y + db[:, 2] * x + db[:, 3] * w], 1)
            P_dqs = np.einsum('nij,nj->ni', rot, R) + t

            for tag, P in (("lbs", P_lbs), ("dqs", P_dqs)):
                ratio, swing = face_stats(P, faces, ra, rn)
                acc[tag][0].append(ratio)
                acc[tag][1].append(swing)

        row = {}
        for tag in ("lbs", "dqs"):
            r = np.concatenate(acc[tag][0])
            s = np.concatenate(acc[tag][1])
            row[tag] = {"area_p01": round(float(np.percentile(r, 1)), 3),
                        "area_p50": round(float(np.percentile(r, 50)), 3),
                        "area_p99": round(float(np.percentile(r, 99)), 3),
                        "swing_p99": round(float(np.percentile(s, 99)), 1),
                        "faces_below_half_area": round(float((r < 0.5).mean() * 100), 3)}
        res[clip] = row
        print(f"[dqs] {ob.name} / {clip}", flush=True)
        for tag in ("lbs", "dqs"):
            d = row[tag]
            print(f"[dqs]    {tag.upper()}  area p01 {d['area_p01']:.3f}  p50 {d['area_p50']:.3f} "
                  f" p99 {d['area_p99']:6.2f}   swing p99 {d['swing_p99']:5.1f}deg   "
                  f"faces under half area {d['faces_below_half_area']:.2f}%", flush=True)
    out[ob.name] = res

if a.json:
    json.dump(out, open(a.json, "w"), indent=2)
