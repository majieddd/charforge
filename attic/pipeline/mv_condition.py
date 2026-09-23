"""Multi-view conditioning: give the 3D model views of the character it was never shown.

TRELLIS.2 is conditioned on DINOv3 features of the reference image, and the MLX port accepts
several images, concatenating their feature tokens along the context axis. The first CharForge
pass used exactly one. Everything behind the character is therefore invented - measurably so,
though less badly than assumed: see backview.py, which puts the pass-1 back at 0.83 of the
front's crease density on char01.

The subtlety that decides whether this stage is worth anything: re-conditioning on renders of
the pass-1 mesh adds **no information**. Those renders are the model's own guess played back to
it, so at best they sharpen the existing guess and at worst they entrench it. New information
has to come from somewhere else, and here it comes from the image model: Krea 2 img2img at
moderate denoise keeps the render's silhouette and colours while repainting the surface with
what a diffusion prior knows about the back of a garment - a yoke seam, a centre vent, a collar
roll, the fall of hair on a neck.

So three arms, and the middle one is the control that keeps the result honest:

  pass1    reference image only                      (what the pipeline already produced)
  pass2    reference + raw renders of side and back  (control: multi-view with no new signal)
  pass3    reference + *enhanced* side and back      (treatment: multi-view plus an image prior)

Comparing pass3 to pass1 alone would credit multi-view conditioning for what the diffusion
prior did. Comparing pass2 to pass1 separates them.

    python mv_condition.py --char char01 --stage render
    python mv_condition.py --char char01 --stage enhance     # needs ComfyUI on :8188
    python mv_condition.py --char char01 --stage generate --arm pass2
    python mv_condition.py --char char01 --stage compare
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"
TRELLIS = ROOT / "vendor" / "trellis2mlx"

# Views the reference never shows. Index into the orbit render: 0=front, 1=side, 2=back, 3=side.
NEW_VIEWS = {1: "side", 2: "back", 3: "side_opposite"}

ENHANCE_NEG = ("blurry, low detail, flat shading, melted geometry, extra limbs, "
               "text, watermark, duplicated figure")


def sh(cmd, **kw):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode})")
    return r


def stage_render(work: Path, mesh: Path, res: int, ortho: float):
    """Orbit renders of the pass-1 mesh: the geometric scaffold the enhancement paints onto."""
    out = work / "mvsrc"
    sh([BLENDER, "-b", "-noaudio", "--python", ROOT / "blender" / "render_views.py", "--",
        "--mesh", mesh, "--out", out, "--views", 4, "--res", res, "--elev", 0, "--ortho", ortho])
    return out


def enhance_prompt(view: str, spec_text: str) -> str:
    """A prompt that tells the image model which side it is looking at.

    Without this the model happily repaints a back view as a front view, complete with a face,
    which is the single worst thing that can happen to a conditioning image.
    """
    where = {
        "side": "seen from the side, in strict profile, no face toward the camera",
        "back": "seen from directly behind, the back of the head and shoulders toward the camera, "
                "no face visible at all",
        "side_opposite": "seen from the opposite side, in strict profile, no face toward the camera",
    }[view]
    return (f"full body photograph of {spec_text}, {where}, standing straight, arms slightly out, "
            "plain white studio background, even soft lighting, sharp fabric detail, visible seams "
            "and stitching, photographic, high detail")


def stage_enhance(work: Path, spec_text: str, denoise: float, steps: int, seed: int):
    """Repaint the non-front renders through Krea 2 img2img so they carry a real image prior."""
    sys.path.insert(0, str(ROOT / "pipeline"))
    import comfy

    src = work / "mvsrc"
    out = work / "mvenh"
    out.mkdir(parents=True, exist_ok=True)
    made = {}
    for idx, view in NEW_VIEWS.items():
        img = src / f"view_{idx:02d}.png"
        if not img.exists():
            print(f"  [skip] {img} missing", flush=True)
            continue
        name = f"charforge_mv_{work.name}_{view}.png"
        comfy.upload_image(img, name=name)
        wf = comfy.krea2_i2i(name, enhance_prompt(view, spec_text), negative=ENHANCE_NEG,
                             denoise=denoise, steps=steps, seed=seed + idx)
        paths, secs = comfy.run(wf, out, prefix=view)
        if paths:
            final = out / f"{view}.png"
            shutil.move(str(paths[0]), final)
            made[view] = str(final)
            print(f"  [enhance] {view}: {secs:.1f}s -> {final}", flush=True)
    json.dump(made, open(out / "index.json", "w"), indent=2)
    return out


def stage_generate(work: Path, ref: Path, arm: str, extra: list[str]):
    """Run the MLX TRELLIS port with one, or several, conditioning images."""
    if arm == "pass1":
        images = [ref]
    elif arm == "pass2":
        images = [ref] + [work / "mvsrc" / f"view_{i:02d}.png" for i in (1, 2)]
    elif arm == "pass3":
        enh = work / "mvenh"
        images = [ref] + [enh / f"{v}.png" for v in ("side", "back")]
    else:
        raise SystemExit(f"unknown arm {arm}")
    missing = [p for p in images if not Path(p).exists()]
    if missing:
        raise SystemExit(f"missing conditioning images: {missing}")

    out = work / f"mesh_{arm}.glb"
    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env["PYTHONPATH"] = "."
    dino = ROOT / "models" / "dinov3-vitl16-hf"
    if dino.exists():
        env["TRELLIS_DINOV3_PATH"] = str(dino)
    # Identical to the settings pass 1 was generated with (run_pipeline.sh step 3). The whole
    # comparison rests on conditioning being the only thing that differs, so resolution, step
    # count, face budget and seed are pinned rather than left to argparse defaults.
    cmd = [str(TRELLIS / ".venv" / "bin" / "python"), "-u", "generate.py",
           "--image", *[str(p) for p in images], "--output", str(out),
           "--resolution", "1024", "--steps", "12", "--target-faces", "200000",
           "--seed", "7", *extra]
    print(f"  [{arm}] {len(images)} conditioning view(s)", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(TRELLIS), env=env)
    if r.returncode != 0:
        raise SystemExit(f"trellis failed ({r.returncode})")
    print(f"  [{arm}] {time.time()-t0:.0f}s -> {out}", flush=True)
    return out


def stage_compare(work: Path, arms: list[str]):
    """Back-vs-front detail for every arm that produced a mesh, plus a rendered contact sheet."""
    sys.path.insert(0, str(ROOT / "pipeline"))
    meshes, labels = [], []
    p1 = work / "trellis_mesh.glb"
    if p1.exists():
        meshes.append(p1)
        labels.append("pass1 (reference only)")
    for arm in arms:
        m = work / f"mesh_{arm}.glb"
        if m.exists():
            meshes.append(m)
            labels.append({"pass2": "pass2 (raw multi-view)",
                           "pass3": "pass3 (enhanced multi-view)"}.get(arm, arm))
    if not meshes:
        raise SystemExit("no meshes to compare")
    sh([BASE / ".venv" / "bin" / "python", ROOT / "pipeline" / "backview.py",
        *[str(m) for m in meshes], "--labels", *labels,
        "--out", work / "backview_multiview.json"])
    for m, lab in zip(meshes, labels):
        tag = lab.split()[0]
        sh([BLENDER, "-b", "-noaudio", "--python", ROOT / "blender" / "render_views.py", "--",
            "--mesh", m, "--out", work / "mvcmp" / tag, "--views", 4, "--res", 768,
            "--elev", 0, "--ortho", 2.15])
    print(f"[mv] comparison renders under {work / 'mvcmp'}", flush=True)


def main(a):
    work = ROOT / "work" / a.char
    ref = Path(a.ref) if a.ref else next(iter(sorted(work.glob("ref_*.png"))), None)
    if a.stage in ("render", "all"):
        stage_render(work, Path(a.mesh) if a.mesh else work / "trellis_mesh.glb", a.res, a.ortho)
    if a.stage in ("enhance", "all"):
        stage_enhance(work, a.spec, a.denoise, a.steps, a.seed)
    if a.stage in ("generate", "all"):
        for arm in (a.arm.split(",") if a.arm else ["pass2", "pass3"]):
            stage_generate(work, ref, arm, a.trellis_args)
    if a.stage in ("compare", "all"):
        stage_compare(work, (a.arm.split(",") if a.arm else ["pass2", "pass3"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default="char01")
    ap.add_argument("--ref", default=None)
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--stage", default="all",
                    choices=["render", "enhance", "generate", "compare", "all"])
    ap.add_argument("--arm", default=None, help="comma-separated: pass2,pass3")
    ap.add_argument("--spec", default="a man wearing a dark green bomber jacket over a grey "
                                      "t-shirt, black jeans and brown lace-up boots, with "
                                      "medium-length wavy hair")
    ap.add_argument("--denoise", type=float, default=0.45)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--res", type=int, default=1024)
    ap.add_argument("--ortho", type=float, default=2.15)
    ap.add_argument("--trellis-args", nargs=argparse.REMAINDER, default=[])
    main(ap.parse_args())
