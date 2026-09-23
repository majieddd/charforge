#!/usr/bin/env python3
"""CharForge: a text prompt or an image in, a game-ready animated character out.

    python charforge.py make --prompt "a knight in weathered plate armour" --name knight
    python charforge.py make --image concept.png --name knight --height 1.85
    python charforge.py make --name wren --from rig          # rerun from a stage onward
    python charforge.py stages                               # list the stages

This replaces run_pipeline.sh, which ran a pipeline that had been thrown out - skinning the raw
generated surface and animating it with procedural gait curves - while every character that
actually shipped was built by chaining scripts by hand. There is now one path, and it is the
path the published characters were built on.

Every stage writes into work/<name>/ and is skipped when its output already exists, so a run
that fails at stage 12 resumes at stage 12. The finished character lands in out/<name>/: glTF,
FBX, loose textures and a manifest, plus a compressed build for the web.
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

ROOT = Path(__file__).resolve().parent
TRELLIS = ROOT / "vendor" / "trellis2mlx"
MIXAMO_HF = Path.home() / ".cache/huggingface/hub/datasets--jasongzy--Mixamo/snapshots"


def blender_bin() -> str:
    for cand in (os.environ.get("BLENDER"), shutil.which("blender"),
                 "/Applications/Blender.app/Contents/MacOS/Blender"):
        if cand and Path(cand).exists():
            return cand
    raise SystemExit("Blender not found: install Blender 5.x or set BLENDER=/path/to/blender")


# ---- the stages ------------------------------------------------------------------------------
# name, what it produces, one line of what it is for. Order is execution order.
STAGES = [
    ("reference", "reference.png", "the image everything is generated from"),
    ("generate",  "pass1.glb",     "TRELLIS.2: geometry and PBR texture from the reference"),
    ("multiview", "mesh.glb",      "second pass conditioned on enhanced side and back views"),
    ("views",     "views/projection.json", "orbit renders for segmentation and pose"),
    ("parts",     "parts.json",    "per-vertex body / clothing / hair / accessory labels"),
    ("skeleton",  "joints.json",   "3D joint positions from the orbit renders"),
    ("retopo",    "retopo.glb",    "clean quad mesh; colour, normal, roughness/metal, AO baked"),
    ("labels",    "labels.json",   "part labels carried onto the clean mesh"),
    ("texclean",  "albedo_clean.png", "skin the generator painted onto clothing, removed"),
    ("rig",       "rig.blend",     "skeleton fitted, skin weights solved"),
    ("frame",     "rig_m.blend",   "metres, soles on the floor, origin under the pelvis"),
    ("tpose",     "rig_t.blend",   "rest pose baked to a T"),
    ("animate",   "animated.blend", "Mixamo captures retargeted onto the rig"),
    ("smooth",    "final.blend",   "weight field softened where it is steepest"),
    ("package",   "PACKAGE",       "glTF + FBX + textures + manifest, Mixamo bone names"),
    ("web",       "WEB",           "compressed build for the browser"),
]
NAMES = [s[0] for s in STAGES]


class Run:
    def __init__(self, a):
        self.a = a
        self.work = ROOT / "work" / a.name
        self.out = ROOT / "out" / a.name
        self.logs = self.work / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.blender = None
        self.t0 = time.time()

    def path(self, rel):
        if rel == "PACKAGE":
            return self.out / f"{self.a.name}.glb"
        if rel == "WEB":
            return self.out / f"{self.a.name}_web.glb"
        return self.work / rel

    def sh(self, stage, cmd, cwd=None, env=None, keep=None):
        """Run one command, logging everything, echoing only the lines that carry a result."""
        log = self.logs / f"{stage}.log"
        with open(log, "w") as fh:
            p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=fh,
                               stderr=subprocess.STDOUT)
        lines = log.read_text(errors="replace").splitlines()
        if keep:
            for ln in lines:
                if any(k in ln for k in keep):
                    print(f"      {ln.strip()[:150]}", flush=True)
        if p.returncode != 0:
            tail = "\n".join("      " + x for x in lines[-15:])
            raise SystemExit(f"\n[{stage}] failed (exit {p.returncode}). Last lines of {log}:\n{tail}")

    def bl(self, stage, script, *args, keep):
        self.blender = self.blender or blender_bin()
        self.sh(stage, [self.blender, "-b", "-noaudio", "--python", ROOT / "blender" / script,
                        "--", *args], keep=keep)

    def py(self, stage, script, *args, keep):
        self.sh(stage, [sys.executable, ROOT / "pipeline" / script, *args], keep=keep)


# ---- stage bodies ----------------------------------------------------------------------------
def s_reference(r: Run):
    out = r.path("reference.png")
    if r.a.image:
        src = Path(r.a.image).expanduser()
        if not src.exists():
            raise SystemExit(f"--image {src} does not exist")
        shutil.copy(src, out)
        print(f"      using your image {src.name}", flush=True)
        return
    if not r.a.prompt:
        raise SystemExit("give --prompt or --image")
    sys.path.insert(0, str(ROOT / "pipeline"))
    import comfy  # noqa: E402  (needs ComfyUI with Krea 2 on :8188)
    neg = "cropped, close-up, multiple people, text, watermark, dramatic shadows, blurry, sitting"
    full = (f"full body character reference sheet of {r.a.prompt}, standing in a relaxed A-pose, "
            "arms slightly away from the body, facing the camera directly, neutral expression, "
            "photorealistic game character, even neutral studio lighting, no shadows, plain flat "
            "light grey background, full figure visible head to toe, centered, orthographic view")
    wf = comfy.krea2_t2i(full, neg, width=832, height=1216, steps=8, cfg=1.0, seed=r.a.seed, gguf=True)
    paths, dt = comfy.run(wf, r.work, prefix="ref", timeout=2400)
    shutil.copy(paths[0], out)
    print(f"      prompt -> image in {dt:.0f}s", flush=True)


def free_comfy():
    """Ask ComfyUI to drop its cached models before TRELLIS runs. On Apple Silicon the GPU shares
    system memory, and ComfyUI keeps the image model resident after a job: the next TRELLIS pass
    then runs against it and the machine swaps (13.8 GB of swap in use on a 24 GB laptop, a
    reference image that took 259 s). The model is the one this pipeline just used, and ComfyUI
    reloads it on the next request; --keep-comfy-loaded turns this off. Best effort - no ComfyUI,
    nothing to free."""
    import urllib.request
    try:
        req = urllib.request.Request("http://127.0.0.1:8188/free", method="POST",
                                     data=b'{"unload_models": true, "free_memory": true}',
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10).read()
    except Exception:                                        # noqa: BLE001
        pass


def trellis(r: Run, stage, images, out):
    if not r.a.keep_comfy_loaded:
        free_comfy()
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONPATH=".")
    dino = ROOT / "models" / "dinov3-vitl16-hf"
    if dino.exists():
        env["TRELLIS_DINOV3_PATH"] = str(dino)
    py = TRELLIS / ".venv" / "bin" / "python"
    if not py.exists():
        raise SystemExit(f"TRELLIS environment missing at {py} - see README, 'Running it'")
    r.sh(stage, [py, "-u", "generate.py", "--image", *images, "--output", out,
                 "--resolution", "1024", "--steps", "12", "--target-faces", "200000",
                 "--seed", str(r.a.seed)], cwd=TRELLIS, env=env, keep=("Total:", "Saved:"))


def s_generate(r: Run):
    trellis(r, "generate", [r.path("reference.png")], r.path("pass1.glb"))


def s_multiview(r: Run):
    """Stochastic multi-view: the second pass sees side and back views that a diffusion prior has
    repainted from renders of the first. Needs ComfyUI; without it the first pass is used."""
    out = r.path("mesh.glb")
    if r.a.quality == "fast":
        shutil.copy(r.path("pass1.glb"), out)
        print("      --quality fast: single-view mesh kept", flush=True)
        return
    try:
        sys.path.insert(0, str(ROOT / "pipeline"))
        import comfy  # noqa: E402
        comfy._get("/system_stats")          # importing the client proves nothing; ask the server
    except Exception as e:                                   # noqa: BLE001
        shutil.copy(r.path("pass1.glb"), out)
        print(f"      ComfyUI unavailable ({e}); single-view mesh kept", flush=True)
        return
    mv = r.work / "mv"
    mv.mkdir(exist_ok=True)
    r.bl("multiview", "render_views.py", "--mesh", r.path("pass1.glb"), "--out", mv / "src",
         "--views", 4, "--res", 1024, "--elev", 0, "--ortho", 2.15, keep=("[render_views]",))
    spec = r.a.prompt or "the character in the reference image"
    neg = ("blurry, low detail, flat shading, melted geometry, extra limbs, text, watermark, "
           "duplicated figure")
    enhanced = []
    for idx, view in ((1, "side"), (2, "back")):
        src = mv / "src" / f"view_{idx:02d}.png"
        name = f"{r.a.name}_{view}.png"
        comfy.upload_image(src, name=name)
        prompt = (f"{view} view of {spec}, full body, same outfit and colours, detailed fabric "
                  "and seams, even studio lighting, plain light grey background")
        wf = comfy.krea2_i2i(name, prompt, negative=neg, denoise=0.42, steps=10, seed=r.a.seed)
        paths, secs = comfy.run(wf, mv, prefix=view)
        enhanced.append(paths[0])
        print(f"      enhanced {view} view in {secs:.0f}s", flush=True)
    trellis(r, "multiview", [r.path("reference.png"), *enhanced], out)


def s_views(r: Run):
    r.bl("views", "render_views.py", "--mesh", r.path("mesh.glb"), "--out", r.work / "views",
         "--views", 8, "--res", 640, keep=("[render_views]",))


def s_parts(r: Run):
    r.py("parts", "parts.py", "--views", r.work / "views", "--out", r.path("parts.json"),
         keep=("[parts]",))


def s_skeleton(r: Run):
    r.py("skeleton", "skeleton.py", "--views", r.work / "views", "--out", r.path("joints.json"),
         keep=("[skeleton]",))


def s_retopo(r: Run):
    r.bl("retopo", "retopo.py", "--mesh", r.path("mesh.glb"), "--out", r.path("retopo.glb"),
         "--bake-res", 4096, "--cage-extrusion", 0.10, "--ray-distance", 0.14,
         keep=("source winding", "UVs:", "AO mean", "AO bake", "normal map:", "normal median", "ORM:"))


def s_labels(r: Run):
    r.py("labels", "transfer_labels.py", "--retopo", r.path("retopo.glb"), "--source",
         r.path("mesh.glb"), "--parts", r.path("parts.json"), "--out", r.path("labels.json"),
         keep=("reclassified", "final:"))


def s_texclean(r: Run):
    r.py("texclean", "texture_cleanup.py", "--retopo", r.path("retopo.glb"), "--labels",
         r.path("labels.json"), "--albedo", r.work / "baked_base_color.png", "--out",
         r.path("albedo_clean.png"), "--mask", r.work / "texclean_mask.png",
         keep=("[texclean]",))


def s_rig(r: Run):
    r.bl("rig", "rig_retopo.py", "--mesh", r.path("retopo.glb"), "--labels", r.path("labels.json"),
         "--joints", r.path("joints.json"), "--out", r.path("rig.blend"),
         "--albedo", r.path("albedo_clean.png"),
         "--json", r.work / "rig.json", keep=("unweighted", "skeleton", "base colour replaced"))


def s_frame(r: Run):
    r.bl("frame", "normalize_frame.py", "--blend", r.path("rig.blend"), "--out",
         r.path("rig_m.blend"), "--height", r.a.height, keep=("[frame] now", "[frame] origin"))


def s_tpose(r: Run):
    r.bl("tpose", "tpose.py", "--blend", r.path("rig_m.blend"), "--out", r.path("rig_t.blend"),
         keep=("rest after",))


def clips_manifest(r: Run) -> Path:
    """The default clip set, resolved against the local Mixamo mirror."""
    spec = json.load(open(ROOT / "animations" / "default_clips.json"))
    snaps = sorted(MIXAMO_HF.glob("*/animation")) if MIXAMO_HF.exists() else []
    if not snaps:
        raise SystemExit("animation clips not found: this stage retargets Mixamo captures from a "
                         "local mirror. Download clips from mixamo.com (FBX, 'without skin') and "
                         "point animations/default_clips.json at them.")
    resolved = {}
    for clip, fname in spec["clips"].items():
        p = snaps[-1] / fname
        if not p.exists():
            raise SystemExit(f"clip {clip!r}: {fname} not in {snaps[-1]}")
        resolved[clip] = str(p)
    out = r.work / "clips.json"
    json.dump(resolved, open(out, "w"), indent=1)
    return out


def s_animate(r: Run):
    r.bl("animate", "retarget.py", "--rig", r.path("rig_t.blend"), "--clips", clips_manifest(r),
         "--out", r.work / "animated.glb", "--blend-out", r.path("animated.blend"),
         "--json", r.work / "retarget.json", keep=("heading", "feet:", "leg-lengths/s", "WARNING", "clips ->"))


def s_smooth(r: Run):
    r.bl("smooth", "smooth_hotspots.py", "--blend", r.path("animated.blend"), "--out",
         r.path("final.blend"), "--json", r.work / "smooth.json", keep=("gradient p99",))


def s_package(r: Run):
    args = ["--blend", r.path("final.blend"), "--out-dir", r.out, "--name", r.a.name,
            "--retarget", r.work / "retarget.json"]
    if r.a.prompt:
        args += ["--prompt", r.a.prompt]
    args += ["--image", r.path("reference.png")]
    r.bl("package", "package.py", *args,
         keep=("renamed", "glTF ->", "FBX  ->", "FBX LODs", "manifest:", "capsule:", "[pkg]    "))


def s_web(r: Run):
    r.sh("web", [sys.executable, ROOT / "tools" / "optimize_glb.py", "--in", r.path("PACKAGE"),
                 "--out", r.path("WEB"), "--res", 2048], keep=("->",))
    r.bl("thumb", "thumbnail.py", "--blend", r.path("final.blend"),
         "--out", r.out / f"{r.a.name}_thumb.png", keep=("[thumb]",))


BODY = {n: globals()[f"s_{n}"] for n in NAMES}


def make(a):
    r = Run(a)
    start = NAMES.index(a.from_stage) if a.from_stage else 0
    stop = NAMES.index(a.until) if a.until else len(NAMES) - 1
    print(f"CharForge  {a.name}  ->  {r.out}", flush=True)
    for i, (name, produces, why) in enumerate(STAGES):
        if i > stop:
            break
        target = r.path(produces)
        if i < start and target.exists():
            continue
        if i >= start and target.exists() and not a.force and a.from_stage is None:
            print(f"  [{i+1:2d}/{len(STAGES)}] {name:<9s} cached", flush=True)
            continue
        t = time.time()
        print(f"  [{i+1:2d}/{len(STAGES)}] {name:<9s} {why}", flush=True)
        BODY[name](r)
        if not target.exists():
            raise SystemExit(f"[{name}] finished without producing {target}")
        print(f"               {time.time()-t:5.0f}s", flush=True)
    print(f"\ndone in {(time.time()-r.t0)/60:.1f} min -> {r.out}", flush=True)
    m = r.out / f"{a.name}.json"
    if m.exists():
        man = json.load(open(m))
        print(f"  {man['height_m']:.2f} m, {man['triangles']:,} triangles, "
              f"{man['skeleton']['bones']} bones ({man['skeleton']['convention']}), "
              f"{len(man['clips'])} clips", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="build a character")
    m.add_argument("--name", required=True, help="folder name under work/ and out/")
    src = m.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="describe the character")
    src.add_argument("--image", help="or give a full-body image of the character")
    m.add_argument("--height", type=float, default=1.75, help="standing height in metres")
    m.add_argument("--quality", choices=("best", "fast"), default="best",
                   help="best adds the multi-view second pass (needs ComfyUI)")
    m.add_argument("--seed", type=int, default=7)
    m.add_argument("--keep-comfy-loaded", action="store_true",
                   help="do not ask ComfyUI to unload its cached models before a TRELLIS pass "
                        "(use this if you are working in ComfyUI while CharForge runs)")
    m.add_argument("--from", dest="from_stage", choices=NAMES, help="rerun from this stage onward")
    m.add_argument("--until", choices=NAMES, help="stop after this stage")
    m.add_argument("--force", action="store_true", help="ignore cached stage outputs")
    sub.add_parser("stages", help="list the stages")
    a = ap.parse_args()
    if a.cmd == "stages":
        for i, (n, p, w) in enumerate(STAGES):
            print(f"{i+1:2d}  {n:<9s} {p:<22s} {w}")
        return
    make(a)


if __name__ == "__main__":
    main()
