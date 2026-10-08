"""Replay the texture stage's CPU steps on a built character, without the stage's GPU steps.

charforge.py's texture stage runs project_texture.py, then skin_tone.py and albedo_range.py (not for anime). Each
command is logged as the first line of work/<name>/logs/<stage>.log, so this replays those exact arguments, with the
main checkout's paths turned into this worktree's, and writes into work/<name>/texwork/<tag>/. A projection change
is then measured in a minute, without re-running the stage's Hunyuan3D-Paint or delighting steps (which a
`charforge.py make --from texclean` would repeat for Pip).

    OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 ../.venv/bin/python pipeline/tex_replay.py --name mara --tag base
    ... --proj-args "--sampler linear --base-ramp none --detail mix"   # the projection's options, for an A/B
    ... --projector <script>                     # another projection script, same arguments (default project_texture.py)
    ... --skip-post                              # projection only (for measuring the projection alone)

The first run on a character also regenerates the per-view position maps that the disk clean-up of 2026-10-08 removed
(view_<az>_position.npy), with the texture_maps stage's own command, under the GPU lock.
"""
import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]               # this worktree
MAIN = "/Users/ascent/Documents/TestingLaya/charforge"   # the paths the logs were written with
PY = ROOT.parent / ".venv" / "bin" / "python"


def logged(work: Path, stage: str):
    """The argument list a stage logged (after the python and script), with paths made this worktree's."""
    p = work / "logs" / f"{stage}.log"
    if not p.exists():
        return None
    first = p.read_text(errors="replace").splitlines()[0]
    if not first.startswith("$ "):
        raise SystemExit(f"{p}: first line is not a logged command")
    toks = [t.replace(MAIN, str(ROOT)) for t in shlex.split(first[2:])]
    return toks[2:]                                       # drop the python and the script


def set_arg(args, flag, value):
    args = list(args)
    i = args.index(flag)
    args[i + 1] = str(value)
    return args


def ensure_maps(work: Path):
    """The per-view position maps (view_<az>_position.npy) that the disk clean-up of 2026-10-08 removed as scratch.
    Regenerated with the texture_maps stage's own logged command into texwork/texproj_regen/, under the GPU lock
    (Blender's bakes), and copied in where missing. uv_position.npy is compared, to show the regeneration agrees."""
    import numpy as np
    texproj = work / "texproj"
    import json
    tags = [v["tag"] for v in json.loads((texproj / "views.json").read_text())["views"]]   # only the views in use: a
    missing = [t for t in tags if not (texproj / f"view_{t}_position.npy").exists()]       # stale extra albedo is not one
    if not missing:
        return
    first = (work / "logs" / "texture_maps.log").read_text(errors="replace").splitlines()[0]
    toks = [t.replace(MAIN, str(ROOT)) for t in shlex.split(first[2:])]
    regen = work / "texwork" / "texproj_regen"
    regen.mkdir(parents=True, exist_ok=True)
    i = toks.index("--out-dir")
    toks[i + 1] = str(regen)
    sys.path.insert(0, str(ROOT))
    import charforge
    print(f"[replay] regenerating {len(missing)} view maps with the texture_maps command", flush=True)
    t = time.time()
    with charforge.gpu("texture maps (Blender)"):
        p = subprocess.run(toks, cwd=str(ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("texture_maps failed:\n" + (p.stdout + p.stderr)[-2000:])
    for f in regen.iterdir():
        dst = texproj / f.name
        if not dst.exists():
            dst.write_bytes(f.read_bytes())
    a_ = np.load(regen / "uv_position.npy").astype(np.float32)
    b_ = np.load(texproj / "uv_position.npy").astype(np.float32)
    same = np.array_equal(np.nan_to_num(a_, nan=-9), np.nan_to_num(b_, nan=-9))
    print(f"    maps: {time.time() - t:.0f}s; uv_position.npy regenerated {'identically' if same else 'DIFFERENTLY'} "
          f"(max |diff| {float(np.nanmax(np.abs(a_ - b_))):.2e})", flush=True)


def run(script, args, stage):
    t = time.time()
    p = subprocess.run([str(PY), str(script), *args], cwd=str(ROOT), capture_output=True, text=True)
    out = (p.stdout + p.stderr).splitlines()
    for ln in out:
        if ln.startswith("[") or "Error" in ln or "Traceback" in ln:
            print(f"    {ln[:200]}", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"[{stage}] failed ({p.returncode}): " + "\n".join(out[-12:]))
    print(f"    {stage}: {time.time() - t:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tag", required=True, help="work/<name>/texwork/<tag>/ receives the outputs")
    ap.add_argument("--projector", default="pipeline/project_texture.py")
    ap.add_argument("--skip-post", action="store_true", help="stop after the projection")
    ap.add_argument("--proj-args", default="", help="extra arguments for the projection, e.g. '--sampler lanczos'")
    ap.add_argument("--range-args", default="", help="extra arguments for albedo_range.py, e.g. '--curve soft'")
    a = ap.parse_args()
    work = ROOT / "work" / a.name
    out_dir = work / "texwork" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_maps(work)
    proj_args = logged(work, "texture")
    if proj_args is None:
        raise SystemExit(f"no logs/texture.log for {a.name}: build it with charforge.py first")
    projected = out_dir / "projected.png"
    final = out_dir / "albedo.png"
    proj_args = set_arg(proj_args, "--out", projected) + shlex.split(a.proj_args)
    print(f"[replay] {a.name} / {a.tag}: projection with {a.projector}", flush=True)
    run(ROOT / a.projector, proj_args, "texture")
    if a.skip_post:
        return
    src = projected
    skin_args = logged(work, "texture_skin")
    if skin_args is not None:
        skinned = out_dir / "skinned.png"
        skin_args = set_arg(set_arg(skin_args, "--albedo", src), "--out", skinned)
        run(ROOT / "pipeline" / "skin_tone.py", skin_args, "texture_skin")
        src = skinned
    range_args = logged(work, "texture_range")
    if range_args is not None:
        range_args = set_arg(set_arg(range_args, "--albedo", src), "--out", final) + shlex.split(a.range_args)
        run(ROOT / "pipeline" / "albedo_range.py", range_args, "texture_range")
    else:
        final.write_bytes(src.read_bytes())
    print(f"[replay] wrote {final}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
