#!/usr/bin/env python3
"""Pixal3D's single view as the generate stage's first pass (E155): a pass1.glb made from the reference picture.

    ../.venv/bin/python pipeline/pixal3d_generate.py --picture work/<n>/reference.png --mask work/<n>/reference_mask.png \
        --raw work/<n>/pixal3d.glb --out work/<n>/pixal3d_pass1.glb [--seed 7] [--force]

The picture goes to Pixal3D with its own cut-out as alpha (the matte the front silhouette is judged on), single-view,
with the sv flow weights (vendor/pixal3d-cpp/build/trellis-cli --sv-image: the CLI crops the picture as the reference
preprocess does and synthesises the front camera). Its mesh then goes through blender/fix_pixal3d.py into pass1's
convention: turned about the vertical so that its face is on the camera the gates use (azimuth 0), the floor sheet
under the feet removed, a million faces, pass1's height and centre. Which way it faces is measured, not assumed: DWPose's
face score on a front and a back render of the fixed model (the read the back-face gate uses) decides whether the turn
was needed, and the choice is kept in pixal3d_pass1.json beside the mesh. The run takes 6-16 minutes and holds the GPU
lock (charforge.gpu).

charforge.py's s_generate calls generate(r) when the make command gets --generator pixal3d: the generate stage's own
gates decide (the front silhouette on the reference, charforge.front_iou; a face on the back of the head,
charforge.back_face), and a model that fails them is made again on the next seed, as TRELLIS's is.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]              # the worktree this file lives in (its vendor/ and models/ link the checkout)
CLI = ROOT / "vendor" / "pixal3d-cpp" / "build" / "trellis-cli"
MODELS = ROOT / "vendor" / "pixal3d-cpp" / "models" / "pixal3d-sv"
FIX = ROOT / "blender" / "fix_pixal3d.py"
RENDER_VIEWS = ROOT / "blender" / "render_views.py"
BACK_FACE = ROOT / "pipeline" / "back_face.py"
TPY = ROOT / "vendor" / "trellis2mlx" / ".venv" / "bin" / "python"   # DWPose runs here (onnxruntime), as charforge.back_face does
TIME = "/usr/bin/time"
HEIGHT = 0.99                                           # pass1's height: 0.978-1.000 over the six TRELLIS.2 pass1 meshes (mean 0.989)
FACES = 1_000_000                                       # pass1's size: TRELLIS.2 is run with --target-faces 1,000,000
TURN = 180.0                                            # the raw output faces away from azimuth 0: on the six E155 characters the turned face reads 0.89-1.03 on the front and 0.17-0.51 on the back


def _charforge():
    """charforge itself: the module already loaded (charforge.py make calls generate() from its own process, so the GPU
    lock is the one the stages share), else imported from this worktree."""
    for name in ("charforge", "__main__"):
        m = sys.modules.get(name)
        if m is not None and hasattr(m, "gpu") and hasattr(m, "front_iou"):
            return m
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import charforge
    return charforge


def sv_input(picture: Path, mask: Path, out: Path) -> Path:
    """The picture with the mask as its alpha: the cut-out the front silhouette is judged on."""
    from PIL import Image
    rgb = Image.open(picture).convert("RGB")
    al = Image.open(mask).convert("L")
    if al.size != rgb.size:
        al = al.resize(rgb.size)
    rgba = rgb.convert("RGBA")
    rgba.putalpha(al)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out)
    return out


def run_cli(rgba: Path, raw: Path, seed: int) -> dict:
    """Pixal3D single view on one matted picture. Unsimplified (--decim 0): blender/fix_pixal3d.py sets the face count."""
    cf = _charforge()
    raw.parent.mkdir(parents=True, exist_ok=True)
    cmd = [TIME, "-l", str(CLI), "-m", str(MODELS), "--sv-image", str(rgba), "--pixal3d-weights", "sv", "--res", "1024",
           "--decim", "0", "--atlas", "2048", "--seed", str(seed), "-o", str(raw)]
    t0 = time.time()
    with cf.gpu("Pixal3D"):
        p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    (raw.parent / (raw.stem + ".log")).write_text(p.stdout + "\n" + p.stderr)
    m = re.search(r"(\d+)\s+maximum resident set size", p.stderr)
    return {"seconds": round(dt, 1), "peak_gb": round(int(m.group(1)) / 1e9, 2) if m else None,
            "ok": raw.exists() and p.returncode == 0, "seed": seed}


def fix(raw: Path, out: Path, turn: float = TURN, faces: int = FACES, height: float = HEIGHT) -> str:
    """blender/fix_pixal3d.py: turned, floor sheet removed, decimated, scaled and centred. Returns its [fix] line."""
    cf = _charforge()
    cmd = [cf.blender_bin(), "-b", "-noaudio", "--python", str(FIX), "--", "--in", str(raw), "--out", str(out),
           "--turn", str(turn), "--faces", str(faces), "--height", str(height)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    log = out.parent / (out.stem + ".fix.log")
    log.write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"fix_pixal3d failed (exit {p.returncode}); see {log}")
    return next((ln for ln in p.stdout.splitlines() if ln.startswith("[fix]")), "[fix] done")


def facing(glb: Path, work: Path) -> dict:
    """DWPose's face score on a front (azimuth 0, the camera the gates use) and a back (azimuth 180) render of glb: the
    side with the face is the front. Renders with blender/render_views.py as front_iou does; the score is back_face.py's."""
    cf = _charforge()
    d = work / "orient"
    shutil.rmtree(d, ignore_errors=True)
    cmd = [cf.blender_bin(), "-b", "-noaudio", "--python", str(RENDER_VIEWS), "--", "--mesh", str(glb), "--out", str(d),
           "--views", "2", "--res", "512", "--elev", "0", "--ortho", "2.15"]
    with cf.gpu("Pixal3D orientation"):
        p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not (d / "view_01.png").exists():
        raise RuntimeError(f"orientation renders failed for {glb}; see {d}")
    score = {}
    for side, png in (("front", d / "view_00.png"), ("back", d / "view_01.png")):
        js = d / f"{side}_face.json"
        q = subprocess.run([str(TPY), str(BACK_FACE), "--render", str(png), "--out", str(js)], capture_output=True, text=True)
        if q.returncode != 0 or not js.exists():
            raise RuntimeError(f"DWPose failed on {png}: {q.stderr[-400:]}")
        score[side] = json.load(open(js))["face"]
    return score


def produce(picture: Path, mask: Path, raw: Path, out: Path, seed: int, force: bool = False, refix: bool = False) -> dict:
    """The raw Pixal3D mesh (one run, kept at raw) and the fixed pass1-style mesh at out, turned to face the front.
    What was done is written beside out (pixal3d_pass1.json); a fixed mesh made from this raw with that record is reused."""
    info = {"seed": seed}
    if force or not raw.exists():
        info["run"] = run_cli(sv_input(picture, mask, raw.with_name(raw.stem + "_input.png")), raw, seed)
        if not info["run"]["ok"]:
            raise RuntimeError(f"Pixal3D failed for seed {seed}; see {raw.with_name(raw.stem + '.log')}")
        print(f"[pixal3d] seed {seed}: {info['run']['seconds'] / 60:.1f} min, peak {info['run']['peak_gb']} GB", flush=True)
    rec = out.with_suffix(".json")
    fresh = (not force and not refix and out.exists() and rec.exists() and out.stat().st_mtime >= raw.stat().st_mtime
             and json.load(open(rec)).get("turn") is not None)
    if not fresh:
        turn = TURN
        line = fix(raw, out, turn)
        sc = facing(out, out.parent)
        if sc["back"] > sc["front"]:
            # the face is already on the front: the raw output was not turned
            turn = 0.0
            line = fix(raw, out, turn)
            sc = facing(out, out.parent)
        info.update(turn=turn, fix=line, face_front=round(sc["front"], 3), face_back=round(sc["back"], 3), made=True)
        print(line, flush=True)
        print(f"      face on the front {sc['front']:.2f}, on the back {sc['back']:.2f}: turned {turn:g} deg", flush=True)
        rec.write_text(json.dumps(info, indent=1))
    else:
        info.update(json.load(open(rec)))
    return info


def generate(r) -> None:
    """charforge.s_generate's Pixal3D route (make --generator pixal3d): the generate stage's gates on the fixed model.
    Writes r.work/pass1.glb and generate.json, like the TRELLIS route does."""
    cf = _charforge()
    out = r.path("pass1.glb")
    (r.work / "generate.json").unlink(missing_ok=True)
    # the cut-out: Qwen-made references come with one; a picture given with --image or drawn by Krea gets it from the
    # generation stage's background remover, as the texture stage does (charforge.reference_mask)
    picture, mask = r.path("reference.png"), cf.reference_mask(r)
    if not mask.exists():
        raise SystemExit("Pixal3D needs the reference's cut-out (reference_mask.png) and the background remover is not set up")
    best = None
    tries = [r.a.seed, r.a.seed + 1000]
    for k, seed in enumerate(tries):
        tag = f"pass1_pixal3d_s{seed}"            # gate/<tag>: the folder the head swap reads (generate.json's input)
        raw, fixed = r.work / f"pixal3d_s{seed}.glb", r.work / f"{tag}.glb"
        produce(picture, mask, raw, fixed, seed)
        iou = cf.front_iou(r, fixed, tag)
        print(f"      front silhouette on the reference: IoU {iou:.2f} (Pixal3D single view, seed {seed})", flush=True)
        bf = cf.back_face(r, fixed, tag) if iou >= cf.FRONT_IOU_MIN else 0.0
        two_faced = bf > cf.BACK_FACE_MAX
        rank = (not two_faced, iou)
        if best is None or rank > best[3]:
            best = (fixed, iou, seed, rank)
        if iou >= cf.FRONT_IOU_MIN and not two_faced:
            break
        if k < len(tries) - 1:
            print("      " + ("the model has a face on the back of its head too - generating it again" if two_faced else
                         "the model does not match the reference from the front - generating it again"), flush=True)
    fixed, iou, seed, rank = best
    if not rank[0]:
        print("      every try had a face on the back of its head: keeping the closest from the front", flush=True)
    if iou < cf.FRONT_IOU_MIN:
        print(f"      no try matches from the front: keeping the closest (IoU {iou:.2f}, seed {seed})", flush=True)
    shutil.copy(fixed, out)
    json.dump({"seed": seed, "front_iou": round(iou, 3), "input": "pixal3d", "generator": "pixal3d"},
              open(r.work / "generate.json", "w"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--picture", required=True)
    ap.add_argument("--mask", required=True, help="the cut-out: its alpha is the picture's matte")
    ap.add_argument("--raw", required=True, help="the raw Pixal3D mesh (kept; reused unless --force)")
    ap.add_argument("--out", required=True, help="the fixed mesh, in pass1's convention")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    info = produce(Path(a.picture), Path(a.mask), Path(a.raw), Path(a.out), a.seed, force=a.force)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
