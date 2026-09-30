"""Where a posed character's body regions meet and how deep one has gone into another.

Shared by aberration_audit.py (counting it over clips) and joint_limits.py (finding the angle at which
a joint starts to push one limb into another). Checked against aberration_controls.py's known depths
(0.5, 1, 2 and 5 cm measured exactly; tools/aberration_controls.py).

    C = Contact(rig, meshes, apart=0.03)       # rest pose read once
    V, A, N = C.posed(depsgraph)               # posed vertices, triangle areas and normals
    tree, pairs, by_region = C.intersections(V)
    depth, vdepth = C.penetration(V, by_region)  # {(region_a, region_b): metres}, per-vertex metres

Regions are named from the dominant bone of each vertex (left_shoulder is the upper arm, left_elbow the
forearm, left_wrist and the fingers the hand, left_hip the thigh, left_knee the shin; Mixamo names too).
Triangles are split once, at rest: Blender splits a quad along its shorter diagonal in the current shape,
so a bending quad changes its loop triangles mid-clip and a comparison with rest would compare different
triangles.
"""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

import bpy

REGIONS = [("hand.L", ("LeftHand",)), ("hand.R", ("RightHand",)), ("forearm.L", ("LeftForeArm",)),
           ("forearm.R", ("RightForeArm",)), ("upperarm.L", ("LeftArm",)), ("upperarm.R", ("RightArm",)),
           ("shoulder.L", ("LeftShoulder",)), ("shoulder.R", ("RightShoulder",)), ("head", ("Head", "Neck", "jaw", "Jaw")),
           ("foot.L", ("LeftFoot", "LeftToe")), ("foot.R", ("RightFoot", "RightToe")),
           ("shin.L", ("LeftLeg",)), ("shin.R", ("RightLeg",)), ("thigh.L", ("LeftUpLeg",)), ("thigh.R", ("RightUpLeg",)),
           ("torso", ("Spine", "Hips"))]
INTERNAL = {"collar": "shoulder", "shoulder": "upperarm", "elbow": "forearm", "wrist": "hand", "thumb": "hand",
            "index": "hand", "middle": "hand", "ring": "hand", "pinky": "hand", "hip": "thigh", "knee": "shin",
            "ankle": "foot", "foot": "foot"}


def region_of(bone):
    b = bone.replace("mixamorig:", "")
    for side, s_ in (("left_", ".L"), ("right_", ".R")):
        if b.startswith(side):
            part = b[len(side):].rstrip("0123456789")
            return INTERNAL.get(part, "other") + s_
    if b in ("pelvis", "spine1", "spine2", "spine3"):
        return "torso"
    if b in ("neck", "head", "head_top", "jaw"):
        return "head"
    for name, keys in REGIONS:
        if any(b.startswith(k) for k in keys):
            return name
    return "other"                                   # spring chains, accessories


class Contact:
    def __init__(self, rig, meshes, apart=0.03):
        self.rig, self.meshes, self.apart = rig, meshes, apart
        pose_position = rig.data.pose_position
        rig.data.pose_position = "REST"
        bpy.context.view_layer.update()
        self.V0, self.F, self.A0, self.N0 = self._evaluated(bpy.context.evaluated_depsgraph_get(), None)
        rig.data.pose_position = pose_position
        bpy.context.view_layer.update()
        F, V0 = self.F, self.V0

        dom_bone = []
        for ob in meshes:
            names = [g.name for g in ob.vertex_groups]
            for v in ob.data.vertices:
                best, bw = None, 0.0
                for g in v.groups:
                    if g.weight > bw and names[g.group] in rig.data.bones:
                        best, bw = names[g.group], g.weight
                dom_bone.append(best)
        self.bones = sorted({b for b in dom_bone if b})
        self.bidx = {b: i for i, b in enumerate(self.bones)}
        self.vdom = np.array([self.bidx.get(b, -1) for b in dom_bone])
        vdom = self.vdom
        self.fdom = np.array([np.bincount(vdom[f][vdom[f] >= 0], minlength=len(self.bones)).argmax()
                              if (vdom[f] >= 0).any() else -1 for f in F])
        self.region_names = sorted({region_of(b) for b in self.bones} | {"other"})
        self.ridx = {r: i for i, r in enumerate(self.region_names)}
        self.breg = np.array([self.ridx[region_of(b)] for b in self.bones])
        self.freg = np.where(self.fdom >= 0, self.breg[np.maximum(self.fdom, 0)], self.ridx["other"])
        self.vreg = np.where(vdom >= 0, self.breg[np.maximum(vdom, 0)], self.ridx["other"])
        self.live = self.A0 > np.percentile(self.A0, 2) * 0.5          # degenerate slivers say nothing
        self.cent0 = V0[F].mean(1)

        # vertex neighbours (CSR over the triangles' edges) and each vertex's mean rest edge length
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        e = np.unique(np.sort(e, 1), axis=0)
        pairs = np.vstack([e, e[:, ::-1]])
        pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
        self.nbr_idx = pairs[:, 1]
        self.deg = np.bincount(pairs[:, 0], minlength=len(V0))
        self.nbr_ok = bool((self.deg > 0).all())
        self.nbr_start = np.concatenate([[0], np.cumsum(self.deg)[:-1]])
        ln = np.linalg.norm(V0[e[:, 0]] - V0[e[:, 1]], axis=1)
        self.edge_len = np.maximum(np.bincount(e[:, 0], ln, len(V0)) + np.bincount(e[:, 1], ln, len(V0)), 1e-9) \
            / np.maximum(self.deg, 1)
        self._rest_trees = {}

    # ---- posing ---------------------------------------------------------------------------------
    def _evaluated(self, dg, F_rest):
        V, F = [], []
        off = 0
        for ob in self.meshes:
            ev = ob.evaluated_get(dg)
            me = ev.to_mesh()
            v = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", v)
            v = v.reshape(-1, 3) @ np.array(ob.matrix_world)[:3, :3].T + np.array(ob.matrix_world)[:3, 3]
            if F_rest is None:
                me.calc_loop_triangles()
                t = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get("vertices", t)
                F.append(t.reshape(-1, 3) + off)
            V.append(v); off += len(v)
            ev.to_mesh_clear()
        V = np.vstack(V)
        F = np.vstack(F) if F_rest is None else F_rest
        e1, e2 = V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]
        cr = np.cross(e1, e2)
        area = 0.5 * np.linalg.norm(cr, axis=1)
        return V, F, area, cr / np.maximum(2 * area[:, None], 1e-12)

    def posed(self, dg, label=""):
        """Posed world vertices, the rest triangles' areas and normals."""
        V, _, A, N = self._evaluated(dg, self.F)
        if len(V) != len(self.V0):
            raise SystemExit(f"[contact] evaluated vertex count differs from rest {label} "
                             f"({len(V)} vs {len(self.V0)}); rest triangles cannot be reused")
        return V, A, N

    # ---- contact --------------------------------------------------------------------------------
    def _region_tree(self, Vx, r, cache):
        if r not in cache:
            idx = np.flatnonzero(self.freg == r)
            cache[r] = BVHTree.FromPolygons(Vx.tolist(), self.F[idx].tolist(), all_triangles=True) if len(idx) else None
        return cache[r]

    def intersections(self, V):
        """The posed tree and the intersecting live triangle pairs whose centroids were more than
        `apart` apart at rest, also grouped by region pair. BVHTree.overlap() is a triangle-triangle
        test after the box test (checked with two triangles whose boxes overlap without touching)."""
        F, live, cent0, freg, names = self.F, self.live, self.cent0, self.freg, self.region_names
        tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
        pairs, by_region = [], {}
        for i, j in tree.overlap(tree):
            if i >= j or not (live[i] and live[j]):
                continue
            if set(F[i]) & set(F[j]):
                continue
            if np.linalg.norm(cent0[i] - cent0[j]) < self.apart:
                continue
            ri, rj = freg[i], freg[j]
            (ra, ta), (rb, tb) = sorted(((ri, i), (rj, j)), key=lambda t: names[t[0]])
            s = by_region.setdefault((ra, rb), (set(), set()))
            s[0].add(ta); s[1].add(tb)
            pairs.append((i, j))
        return tree, pairs, by_region

    def penetration(self, V, by_region, only=None):
        """For each pair of regions that intersect: how far (m) either has gone inside the other.

        Every crossing triangle has a vertex on each side of the other surface; from those, flood through
        the region's own vertices for as long as they stay inside the other region (signed distance to its
        nearest posed face, along that face's normal, below zero), so a limb sunk 5 cm is measured to its
        deepest vertex however fine the mesh (a fixed number of rings stopped at 3 cm on the controls).
        Inside only if the vertex projects straight onto the face: a region is an open patch (the upper arm
        is a tube open at the shoulder and elbow), past its open end the nearest point is on the rim, the
        offset is oblique to the rim face's normal and the sign means nothing - it read Aoi's bent forearm
        as 15.9 cm inside her upper arm. From inside a closed surface the nearest point is always a straight
        projection. Vertices within `apart` of the other region at rest are seams and are skipped.
        `only`: a set of region-name pairs to measure (both orders). Returns ({pair: depth_m}, vdepth)."""
        V0, F, vreg = self.V0, self.F, self.vreg
        nbr_idx, nbr_start, deg = self.nbr_idx, self.nbr_start, self.deg
        trees, vdepth, out = {}, np.zeros(len(V)), {}
        for (ra, rb), (ta, tb) in by_region.items():
            if ra == rb:
                continue
            if only is not None and (self.region_names[ra], self.region_names[rb]) not in only \
                    and (self.region_names[rb], self.region_names[ra]) not in only:
                continue
            best = 0.0
            for src, dst, tris in ((ra, rb, ta), (rb, ra, tb)):
                tree, tree0 = self._region_tree(V, dst, trees), self._region_tree(V0, dst, self._rest_trees)
                if tree is None:
                    continue
                stack = [int(v) for v in np.unique(F[list(tris)].ravel()) if vreg[v] == src]
                seen = set(stack)
                while stack and len(seen) < 50000:
                    vi = stack.pop()
                    if tree0.find_nearest(Vector(V0[vi]), self.apart)[0] is not None:
                        continue
                    loc, nrm, _, dist = tree.find_nearest(Vector(V[vi]), 0.25)
                    if loc is None:
                        continue
                    sd = float((Vector(V[vi]) - loc).dot(nrm))
                    if sd >= 0 or -sd < 0.9 * dist:
                        continue
                    # Enclosed, not just behind one face: from the vertex, deeper along -normal, the ray must leave
                    # the region through its far wall (a face met from behind). Past an open sleeve end it escapes;
                    # between two layers (sleeve over skin) it meets the next layer's front. The nearest-face test
                    # alone read Vex's elbow as 15.7 cm deep at 105 degrees and 0.0 at 115.
                    hit = tree.ray_cast(Vector(V[vi]) - nrm * 1e-4, -nrm, 0.6)
                    if hit[0] is None or hit[1].dot(-nrm) <= 0:
                        continue
                    best = max(best, -sd)
                    vdepth[vi] = max(vdepth[vi], -sd)
                    for nb in nbr_idx[nbr_start[vi]:nbr_start[vi] + deg[vi]]:
                        nb = int(nb)
                        if nb not in seen and vreg[nb] == src:
                            seen.add(nb)
                            stack.append(nb)
            out[(ra, rb)] = best
        return out, vdepth

    def depth_between(self, V, pairs_named):
        """The deepest penetration (m) among the named region pairs, e.g. {("forearm.L", "upperarm.L")}."""
        _, _, by_region = self.intersections(V)
        d, _ = self.penetration(V, by_region, only=set(pairs_named))
        return max(d.values(), default=0.0)
