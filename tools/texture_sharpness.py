"""How sharp an albedo is at full resolution, where it seams, and how it compares with the picture it came from.

    ../.venv/bin/python tools/texture_sharpness.py --name mara --albedo work/mara/texwork/base/albedo.png \
        --out work/mara/texwork/base/sharpness.json [--front] [--front-res 2048] [--no-audit]

blender/model_quality.py's texture measures pool the atlas to 1024 px, which cannot see the 4096-texel detail
that the projection changes; these are read at full resolution. All numbers use the surface texels (the ones
a mesh point lies on, from work/<name>/texproj/uv_position.npy) and leave out the 3 texels nearest an island's
edge, so gutters and seams are not counted as detail.

  texels_per_height  texels across the figure's height (Z extent of the surface), a unit-free density
  detail_L           RMS of the lightness high-pass (Gaussian, sigma 2 texels) in L*: the fine detail
  grad_L_p90         90th percentile of the lightness gradient (Sobel), in L* per texel
  detail_front/back/side  detail_L split by normal: front (-Y), back (+Y), the rest is side, as model_quality's
                     garment split; back_over_front and side_over_front say whether the generated views are
                     as sharp as the picture's own side
  seam_de            CIEDE2000 across UV seams: a surface texel next to another island's texel in 3D (within
                     1.5 texels) and more than 6 texels away on the atlas; seam_ratio = seam p50 / interior p50
  interior_de        CIEDE2000 between neighbouring texels inside an island (the normal step)
  audit_texture      blender/model_quality.py's texture measures on the same albedo (seam_de, back_sharp,
                     front_back_de, light_r2, dark/bright), for comparison with the audit
  front              with --front: the unlit albedo rendered from the texture stage's own front camera
                     (blender/render_front.py), laid on the picture by the similarity that overlaps the two
                     silhouettes (pipeline/align.py), at the picture's own scale. front_ratio is the lightness
                     high-pass RMS inside the figure, render over picture: 1.0 keeps the picture's detail, below
                     1.0 is softer than the picture, above is added grain or ringing.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "pipeline")]
from model_quality import ciede2000, srgb_to_lab, texture as audit_texture  # noqa: E402
from align import image_mask, similarity  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
BLENDER = shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"


def high_pass(L, valid, sigma):
    """L minus its Gaussian blur over the valid texels only (a normalised convolution), zero off the surface."""
    num = ndimage.gaussian_filter(np.where(valid, L, 0.0).astype(np.float32), sigma)
    den = ndimage.gaussian_filter(valid.astype(np.float32), sigma)
    return np.where(valid, L - num / np.maximum(den, 1e-6), 0.0).astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else None


def albedo_measures(alb_path, texproj):
    alb = np.asarray(Image.open(alb_path).convert("RGB"), np.float32) / 255.0
    pos = np.load(texproj / "uv_position.npy").astype(np.float32)
    nrm = np.load(texproj / "uv_normal.npy").astype(np.float32)
    valid = np.isfinite(pos).all(-1) & (np.abs(pos).sum(-1) > 0)
    inner = cv2.erode(valid.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0       # 3 texels off every edge
    inner1 = ndimage.binary_erosion(valid)                                        # 1 texel, 4-neighbours (as model_quality)
    lab = srgb_to_lab(alb)
    L = lab[..., 0]
    out = {"size": int(alb.shape[0]), "surface_texels": int(valid.sum())}

    # density: texels across the figure's height (Blender Z is up) and across a texel on the body
    z = pos[valid][:, 2]
    height = float(z.max() - z.min())
    ok = valid[:, 1:] & valid[:, :-1]
    texel = float(np.median(np.linalg.norm(pos[:, 1:] - pos[:, :-1], axis=-1)[ok]))
    out["texels_per_height"] = round(height / texel, 1)

    hp = high_pass(L, valid, 2.0)
    out["detail_L"] = round(rms(hp[inner]), 4)
    Lz = np.where(valid, L, 0.0).astype(np.float32)
    g = np.hypot(cv2.Sobel(Lz, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(Lz, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    out["grad_L_p90"] = round(float(np.quantile(g[inner], 0.9)), 4)
    front = inner & (nrm[..., 1] < -0.5)
    back = inner & (nrm[..., 1] > 0.5)
    side = inner & ~front & ~back
    out["detail_front"], out["detail_back"], out["detail_side"] = (round(rms(hp[m]), 4) for m in (front, back, side))
    out["back_over_front"] = round(rms(hp[back]) / rms(hp[front]), 3)
    out["side_over_front"] = round(rms(hp[side]) / rms(hp[front]), 3)

    # seams: texels on an island's edge, next (in 3D) to another island's edge texel that is far on the atlas
    ey, ex = np.nonzero(valid & ~inner1)
    P = pos[ey, ex]
    d, j = cKDTree(P).query(P, k=8, distance_upper_bound=1.5 * texel)
    seam = []
    for k in range(1, 8):
        hit = np.isfinite(d[:, k]) & (j[:, k] < len(ey))
        jj = np.where(hit, j[:, k], 0)
        far = hit & (np.hypot(ey - ey[jj], ex - ex[jj]) > 6)
        if far.any():
            s_ = np.flatnonzero(far)
            seam.append(ciede2000(lab[ey[s_], ex[s_]], lab[ey[jj[s_]], ex[jj[s_]]]))
    seam = np.concatenate(seam) if seam else np.array([])
    iy, ix = np.nonzero(inner[:, :-1] & inner[:, 1:])
    pick = RNG.choice(len(iy), min(100_000, len(iy)), replace=False)
    interior = ciede2000(lab[iy[pick], ix[pick]], lab[iy[pick], ix[pick] + 1])
    s50, i50 = float(np.median(seam)), float(np.median(interior))
    out["seam_de"] = {"p50": round(s50, 3), "p95": round(float(np.quantile(seam, 0.95)), 3), "pairs": int(len(seam))}
    out["interior_de"] = {"p50": round(i50, 3), "p95": round(float(np.quantile(interior, 0.95)), 3)}
    out["seam_ratio"] = round(s50 / max(i50, 1e-6), 3)
    return out, alb, valid


def front_measures(work, albedo, out_dir, res, texproj):
    """The unlit albedo from the texture stage's front camera, laid on the picture the projection used."""
    sys.path.insert(0, str(ROOT))
    import charforge
    out_dir.mkdir(parents=True, exist_ok=True)
    render = out_dir / f"front_unlit_{res}.png"
    cmd = [BLENDER, "-b", "-noaudio", "--python", str(ROOT / "blender" / "render_front.py"), "--",
           "--mesh", str(work / "retopo.glb"), "--albedo", str(albedo), "--views", str(texproj / "views.json"),
           "--out", str(render), "--res", str(res)]
    with charforge.gpu("texture sharpness: front render"):
        p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not render.exists():
        raise SystemExit("render_front failed:\n" + (p.stdout + p.stderr)[-1500:])
    ref_path = work / "reference_delit.png" if (work / "reference_delit.png").exists() else work / "reference.png"
    rgba = np.asarray(Image.open(render).convert("RGBA"))
    rm, rgb = rgba[..., 3] > 127, rgba[..., :3]
    ref = np.asarray(Image.open(ref_path).convert("RGB"))
    mask_path = work / "reference_mask.png"
    imask = (np.asarray(Image.open(mask_path).convert("L").resize(ref.shape[1::-1])) > 127) if mask_path.exists() \
        else image_mask(ref)
    (s, tx, ty), iou = similarity(rm, imask)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    r1 = cv2.resize(rgb, None, fx=s, fy=s, interpolation=interp)
    m1 = cv2.resize(rm.astype(np.uint8) * 255, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) > 127
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    H, W = ref.shape[:2]
    warped = cv2.warpAffine(r1, M, (W, H), flags=cv2.INTER_LINEAR)
    wm = cv2.warpAffine(m1.astype(np.uint8), M, (W, H), flags=cv2.INTER_NEAREST) > 0
    both = cv2.erode((wm & imask).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    Lw = srgb_to_lab(warped.astype(np.float32) / 255.0)[..., 0]
    Lr = srgb_to_lab(ref.astype(np.float32) / 255.0)[..., 0]
    hw = high_pass(Lw, both, 1.0)
    hr = high_pass(Lr, both, 1.0)
    ratio = rms(hw[both]) / rms(hr[both])
    # the two side by side at the picture's scale, for a look (the render outside its silhouette in black)
    shown = np.where(both[..., None], warped, 0).astype(np.uint8)
    Image.fromarray(np.concatenate([shown, ref], 1)).save(out_dir / "front_vs_picture.png")
    return {"render": str(render), "reference": str(ref_path), "scale_s": round(float(s), 4), "silhouette_iou": round(float(iou), 4),
            "front_ratio": round(float(ratio), 4), "detail_render": round(rms(hw[both]), 4),
            "detail_picture": round(rms(hr[both]), 4), "pixels": int(both.sum())}


RNG = np.random.default_rng(7)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--albedo", required=True, help="the albedo to measure (a path)")
    ap.add_argument("--out", required=True, help="the JSON to write")
    ap.add_argument("--front", action="store_true", help="also render the front camera and compare with the picture")
    ap.add_argument("--front-res", type=int, default=2048)
    ap.add_argument("--no-audit", action="store_true", help="leave out model_quality's texture measures")
    a = ap.parse_args()
    work = ROOT / "work" / a.name
    texproj = work / "texproj"
    albedo = Path(a.albedo).resolve()
    res, alb, valid = albedo_measures(albedo, texproj)
    del alb
    res = {"name": a.name, "albedo": str(albedo), **res}
    if not a.no_audit:
        audit = audit_texture(work, 1.0, albedo=str(albedo))
        audit.pop("texel_mm", None)                       # its scale is not this run's; texels_per_height stands
        res["audit_texture"] = audit
    if a.front:
        res["front"] = front_measures(work, albedo, Path(a.out).parent, a.front_res, texproj)
    Path(a.out).write_text(json.dumps(res, indent=1))
    keys = ("texels_per_height", "detail_L", "grad_L_p90", "detail_front", "detail_back", "detail_side",
            "back_over_front", "seam_ratio")
    print("[sharp] " + "  ".join(f"{k} {res[k]}" for k in keys), flush=True)
    if "front" in res:
        print(f"[sharp] front: ratio {res['front']['front_ratio']} (render {res['front']['detail_render']} over "
              f"picture {res['front']['detail_picture']}), silhouette IoU {res['front']['silhouette_iou']}", flush=True)


if __name__ == "__main__":
    main()
