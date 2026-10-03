"""Points of a skinned mesh, posed by their own skin weights without evaluating the mesh.

Used where a solve needs to know where the deformed surface is, many times a key: retarget.py's
foot planting and clearance.py's floor cap both pose the shoe below the ankle this way. Dual
quaternion by default, because that is how the playground (and Unity/Unreal with the switch on)
deforms the character; linear blending pulled Cadet's blended boot points in by a few centimetres
where the ankle flexes hard, so the planted boot measured on the floor sat under it on screen.

    cloud = skin_points.build(meshes, rig, keep=lambda world_co, weights: ...)
    P = skin_points.pose(cloud, rig)            # (n, 3) world positions at the current pose
"""
import numpy as np


def build(meshes, rig, keep, max_points=None):
    """Points of `meshes` for which keep(world_co, {bone: weight}) is true, with up to four
    normalised bone weights each - every such vertex, or thinned to about `max_points`.

    Callers that take the lowest point should keep every vertex: thinned, the lowest point jumps
    from one sample to the next as a foot rolls (see retarget.sole_points)."""
    pts, wts, vid = [], [], []
    for o in meshes:
        names = {g.index: g.name for g in o.vertex_groups}
        mw = o.matrix_world
        for v in o.data.vertices:
            w = {names[g.group]: g.weight for g in v.groups
                 if names.get(g.group) in rig.data.bones and g.weight > 0}
            q = mw @ v.co
            if w and keep(q, w):
                pts.append(tuple(q) + (1.0,))
                vid.append((o.name, v.index))
                wts.append(sorted(((x, nm) for nm, x in w.items()), reverse=True)[:4])
    if not pts:
        return None
    step = max(1, len(pts) // max_points) if max_points else 1
    pts, wts, vid = pts[::step], wts[::step], vid[::step]
    bones = sorted({nm for w in wts for _, nm in w})
    bi = {nm: k for k, nm in enumerate(bones)}
    idx, wgt = np.zeros((len(pts), 4), int), np.zeros((len(pts), 4))
    for r, w in enumerate(wts):
        tot = sum(x for x, _ in w) or 1.0
        for c, (x, nm) in enumerate(w):
            idx[r, c], wgt[r, c] = bi[nm], x / tot
    MW = np.array(rig.matrix_world)
    rest_inv = np.array([np.linalg.inv(MW @ np.array(rig.data.bones[nm].matrix_local)) for nm in bones])
    return {"rest": np.array(pts), "bones": bones, "idx": idx, "w": wgt, "rest_inv": rest_inv, "n": len(pts), "vid": vid}


def _qmul(a, b):
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack([a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
                     a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
                     a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
                     a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0], -1)


def _mat_to_quat(R):
    """Rotation matrices (k, 3, 3) -> unit quaternions (k, 4), w first."""
    from mathutils import Matrix
    return np.array([tuple(Matrix(r.tolist()).to_quaternion()) for r in R])


def pose(cloud, rig, dqs=True):
    """World positions of the cloud's points at the rig's current pose."""
    MW = np.array(rig.matrix_world)
    pb = rig.pose.bones
    skin = np.array([MW @ np.array(pb[nm].matrix) @ cloud["rest_inv"][k] for k, nm in enumerate(cloud["bones"])])
    rest = cloud["rest"][:, :3]
    if not dqs:
        per = np.einsum("bij,nj->nbi", skin, cloud["rest"])
        p = np.einsum("nc,nci->ni", cloud["w"], per[np.arange(cloud["n"])[:, None], cloud["idx"]])
        return p[:, :3]
    qr = _mat_to_quat(skin[:, :3, :3])                                  # (b, 4)
    t = np.concatenate([np.zeros((len(skin), 1)), skin[:, :3, 3]], 1)
    qd = 0.5 * _qmul(t, qr)
    R = qr[cloud["idx"]]                                                # (n, 4, 4)
    D = qd[cloud["idx"]]
    sign = np.sign(np.einsum("nci,ni->nc", R, R[:, 0]))                 # the same hemisphere as the first bone
    sign[sign == 0] = 1.0
    w = cloud["w"] * sign
    Qr = np.einsum("nc,nci->ni", w, R)
    Qd = np.einsum("nc,nci->ni", w, D)
    norm = np.linalg.norm(Qr, axis=1, keepdims=True)
    Qr, Qd = Qr / norm, Qd / norm
    conj = Qr * np.array([1.0, -1.0, -1.0, -1.0])
    trans = 2.0 * _qmul(Qd, conj)[:, 1:]
    v = np.concatenate([np.zeros((len(rest), 1)), rest], 1)
    rot = _qmul(_qmul(Qr, v), conj)[:, 1:]
    return rot + trans
