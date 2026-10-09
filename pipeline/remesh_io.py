"""Shared pieces of the E168 remeshing experiments: a minimal glTF reader, frame conversions, angle-weighted
vertex normals and nearest-point queries on a dense triangle mesh. Numpy, scipy and PIL only (no Blender).

Frames: glTF is Y-up, Blender is Z-up. A glTF point (x, y, z) is the Blender point (x, -z, y); the character's
height runs along glTF Y (Blender Z) and its front faces Blender -Y (the render cameras sit there), which is glTF +Z.
Joints in joints_refined.json are in the skeleton's frame, two units tall and centred: a Blender point is
J * H / 2 + centre, with H and centre taken from the generator's mesh (retopo.py's head_weights does the same).
"""
import json
import struct
import io

import numpy as np
from scipy.spatial import cKDTree

COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def to_blender(P):
    """glTF (Y-up) points to Blender (Z-up) points."""
    P = np.asarray(P, dtype=np.float64)
    return np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)


def to_gltf(B):
    """Blender (Z-up) points to glTF (Y-up) points."""
    B = np.asarray(B, dtype=np.float64)
    return np.stack([B[..., 0], B[..., 2], -B[..., 1]], -1)


def read_glb(path):
    with open(path, "rb") as f:
        magic, _ver, length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise SystemExit(f"not a GLB: {path}")
        chunks = []
        while f.tell() < length:
            clen, ctype = struct.unpack("<II", f.read(8))
            chunks.append((ctype, f.read(clen)))
    J = json.loads(chunks[0][1].decode("utf8"))
    BIN = chunks[1][1] if len(chunks) > 1 else b""
    return J, BIN


def accessor(J, BIN, idx):
    a = J["accessors"][idx]
    dt = np.dtype(COMP[a["componentType"]]).newbyteorder("<")
    nc = NCOMP[a["type"]]
    count = a["count"]
    if "bufferView" not in a:
        return np.zeros((count, nc), dt)
    bv = J["bufferViews"][a["bufferView"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", dt.itemsize * nc)
    return np.ndarray(shape=(count, nc), dtype=dt, buffer=BIN, offset=off,
                      strides=(stride, dt.itemsize)).copy()


def node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    M = np.eye(4)
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    S = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    M[:3, :3] = R * S[None, :]
    M[:3, 3] = t
    return M


def normal_texture(J, BIN, mat_index):
    """The normal map embedded in a material, as an (H, W, 3) uint8 array (None when there is none)."""
    if mat_index is None or "materials" not in J:
        return None
    nt = J["materials"][mat_index].get("normalTexture")
    if nt is None:
        return None
    img = J["images"][J["textures"][nt["index"]]["source"]]
    from PIL import Image
    bv = J["bufferViews"][img["bufferView"]]
    off = bv.get("byteOffset", 0)
    raw = BIN[off:off + bv["byteLength"]]
    return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def read_mesh(path, want_tex=False):
    """Every mesh primitive of a GLB, with node transforms applied (glTF frame).
    Returns dict: V (n,3), F (m,3) int64, N (n,3) or None, UV (n,2) or None, tex (H,W,3) uint8 or None."""
    J, BIN = read_glb(path)
    parts = []
    for node in J.get("nodes", []):
        if "mesh" not in node:
            continue
        M = node_matrix(node)
        for prim in J["meshes"][node["mesh"]]["primitives"]:
            at = prim["attributes"]
            P = accessor(J, BIN, at["POSITION"]).astype(np.float64)
            P = (M[:3, :3] @ P.T).T + M[:3, 3]
            N = accessor(J, BIN, at["NORMAL"]).astype(np.float64) if "NORMAL" in at else None
            if N is not None:
                N = (M[:3, :3] @ N.T).T
            UV = accessor(J, BIN, at["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in at else None
            if "indices" in prim:
                F = accessor(J, BIN, prim["indices"]).astype(np.int64).reshape(-1)
            else:
                F = np.arange(len(P), dtype=np.int64)
            F = F.reshape(-1, 3)
            tex = normal_texture(J, BIN, prim.get("material")) if want_tex else None
            parts.append((P, F, N, UV, tex))
    if not parts:
        raise SystemExit(f"no mesh in {path}")
    off, Vs, Fs, Ns, UVs = 0, [], [], [], []
    tex = None
    for P, F, N, UV, t in parts:
        Vs.append(P)
        Fs.append(F + off)
        Ns.append(N)
        UVs.append(UV)
        tex = tex if tex is not None else t
        off += len(P)
    V = np.concatenate(Vs)
    F = np.concatenate(Fs)
    N = np.concatenate(Ns) if all(n is not None for n in Ns) else None
    UV = np.concatenate(UVs) if all(u is not None for u in UVs) else None
    return {"V": V, "F": F, "N": N, "UV": UV, "tex": tex}


def write_glb(path, V, F):
    """Plain triangle mesh as GLB, in glTF coordinates (what retopo.py's import_one reads back as Blender Z-up)."""
    import trimesh
    m = trimesh.Trimesh(vertices=np.asarray(V, dtype=np.float64), faces=np.asarray(F, dtype=np.int64), process=False)
    m.export(path)


def face_normals_areas(V, F):
    P = V[F]
    cr = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    a2 = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(a2, 1e-300)[:, None]
    return n, a2 / 2


def vertex_normals(V, F):
    """Smooth vertex normals, weighted by the corner angles (what Blender's shade-smooth gives)."""
    P = V[F]
    n_f, _ = face_normals_areas(V, F)
    corner = []
    for k in range(3):
        a = P[:, (k + 1) % 3] - P[:, k]
        b = P[:, (k + 2) % 3] - P[:, k]
        cosv = np.sum(a * b, 1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
        corner.append(np.arccos(np.clip(cosv, -1, 1)))
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], n_f * corner[k][:, None])
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-300)


def closest_on_triangles(P, A, B, C):
    """Closest point to each P on each triangle ABC (arrays (..., 3)). Ericson's region tests, vectorised.
    Returns (Q, bary) with bary (..., 3) the weights of A, B, C."""
    ab, ac, ap = B - A, C - A, P - A
    d1 = np.sum(ab * ap, -1)
    d2 = np.sum(ac * ap, -1)
    bp = P - B
    d3 = np.sum(ab * bp, -1)
    d4 = np.sum(ac * bp, -1)
    cp = P - C
    d5 = np.sum(ab * cp, -1)
    d6 = np.sum(ac * cp, -1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        vab = np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0)
        wac = np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0)
        s_bc = (d4 - d3) + (d5 - d6)
        wbc = np.where(s_bc != 0, (d4 - d3) / s_bc, 0.0)
        den = va + vb + vc
        den = np.where(den != 0, den, 1.0)
        vf = vb / den
        wf = vc / den
    z = np.zeros_like(d1)
    o = np.ones_like(d1)
    conds = [(d1 <= 0) & (d2 <= 0),
             (d3 >= 0) & (d4 <= d3),
             (vc <= 0) & (d1 >= 0) & (d3 <= 0),
             (d6 >= 0) & (d5 <= d6),
             (vb <= 0) & (d2 >= 0) & (d6 <= 0),
             (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)]
    ch = [(o, z, z), (z, o, z), (1 - vab, vab, z), (z, z, o), (1 - wac, z, wac), (z, 1 - wbc, wbc)]
    default = (1 - vf - wf, vf, wf)
    bary = np.stack([np.select(conds, [c[k] for c in ch], default=default[k]) for k in range(3)], -1)
    Q = bary[..., 0:1] * A + bary[..., 1:2] * B + bary[..., 2:3] * C
    return Q, bary


class TriangleIndex:
    """Nearest point on a triangle mesh: candidate triangles from a KD-tree on their centroids, then the exact
    closest point on each candidate. With k candidates this is exact unless the true nearest triangle's centroid
    lies beyond the k-th nearest centroid, which k around 24-64 avoids on the meshes used here."""

    def __init__(self, V, F, k=32):
        self.V = np.asarray(V, dtype=np.float64)
        self.F = np.asarray(F, dtype=np.int64)
        self.k = min(k, len(self.F))
        self.tree = cKDTree(self.V[self.F].mean(axis=1))

    def query(self, P, chunk=20000):
        P = np.asarray(P, dtype=np.float64)
        n = len(P)
        Q = np.empty((n, 3))
        face = np.empty(n, np.int64)
        bary = np.empty((n, 3))
        dist = np.empty(n)
        for s in range(0, n, chunk):
            p = P[s:s + chunk]
            _, idx = self.tree.query(p, k=self.k)
            idx = idx.reshape(len(p), self.k)
            T = self.F[idx]                                 # (c, k, 3)
            q, b = closest_on_triangles(p[:, None, :], self.V[T[..., 0]], self.V[T[..., 1]], self.V[T[..., 2]])
            d = np.linalg.norm(q - p[:, None, :], axis=-1)
            best = np.argmin(d, axis=1)
            r = np.arange(len(p))
            Q[s:s + chunk] = q[r, best]
            bary[s:s + chunk] = b[r, best]
            face[s:s + chunk] = idx[r, best]
            dist[s:s + chunk] = d[r, best]
        return Q, face, bary, dist


def bbox_blender(mesh_V):
    """The generator mesh's bounding box in Blender frame: (lo, hi)."""
    B = to_blender(mesh_V)
    return B.min(0), B.max(0)


def joints_world(J_skel, lo, hi):
    """joints_refined.json's skeleton-frame points as Blender world points (retopo.py's mapping)."""
    H = float(hi[2] - lo[2])
    c = (lo + hi) / 2
    return {k: np.array(v, dtype=np.float64) * (H / 2) + c for k, v in J_skel.items()}, H, c


def load_joints(path):
    d = json.load(open(path))
    return d["joints"] if "joints" in d else d


def boundary_and_nonmanifold(F):
    """Count edges used by one face (boundary) and by three or more (non-manifold)."""
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, counts = np.unique(E, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts >= 3).sum())


def weld(V, F):
    """Merge duplicate positions (glTF splits vertices at UV seams and hard edges); drop collapsed faces."""
    uniq, inv = np.unique(np.round(V, 9), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    Fw = inv[F]
    ok = (Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])
    Fw = Fw[ok]
    used = np.zeros(len(uniq), bool)
    used[Fw.ravel()] = True
    newidx = np.cumsum(used) - 1
    return uniq[used], newidx[Fw]


def piece_mask(V, F, min_faces):
    """Faces of the connected pieces with at least min_faces faces: retopo.py's drop_debris rule (a piece under
    0.25% of the low's faces is debris; the garments and strands kept apart by design are far larger). Welds the
    seams first, so a UV seam is not a break."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    uniq, inv = np.unique(np.round(V, 9), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    Fw = inv[F]
    n = len(uniq)
    E = np.concatenate([Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]])
    g = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    fl = lab[Fw[:, 0]]
    cnt = np.bincount(fl, minlength=lab.max() + 1)
    return (cnt >= min_faces)[fl]


def largest_component(V, F):
    """Faces of the largest connected piece (loose flakes and debris left out)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(V)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    counts = np.bincount(lab[F[:, 0]], minlength=lab.max() + 1)
    return lab[F[:, 0]] == np.argmax(counts)
