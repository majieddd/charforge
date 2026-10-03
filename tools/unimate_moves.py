"""Moves from words, generated on a character's own skeleton by UniMate (experiment E129).

    python tools/unimate_moves.py --name cadet --move "salute=An object stands at attention and salutes." \
        [--move "march=An object marches in place."] [--reps 3] [--seed 7]     # generate; previews per move
    python tools/unimate_moves.py --name cadet --pick salute=1 --pick march=0 --rebuild   # keep those samples

Prompts work best in the wording of UniMate's training captions: "An object <does something>.", short.

UniMate (Mou et al., SIGGRAPH Asia 2026; vendor/unimate, weights in vendor/unimate/weights) is a flow-
matching model conditioned on a skeleton's topology and T-pose as well as the text, so it animates a
skeleton it never saw. Here the skeleton is the character's own:

  1. preprocess  UniMate's own preprocess_char.py (it targets Blender 4.0, which runs as a Python module in
                 vendor/unimate/.venv-bpy) on the packaged out/<name>/<name>.glb without its morph targets
                 (blender/unimate_input.py: they tore the canonical mesh apart) -> work/<name>/unimate/char/
                 {cond.npy, motions/, <name>_canonical.glb}: the canonical T-pose, the topology conditioning
                 and the character's own clips as motion features. Redone when the package is newer.
  2. sample      tools/unimate_sample.py (UniMate's sampler, a fixed-step ODE on Apple's GPU) on a run
                 folder whose only dataset is that character -> 60 frames (2 s at 30 fps) per repetition
  3. export      UniMate's animate_motion.py drives the canonical character -> one FBX per move
  4. preview     each sample rendered at five moments, one contact sheet per move: work/<name>/qa/words/<move>.png
  5. --pick      the chosen sample goes into work/<name>/extra_clips.json like a clip made from a video, so
                 `charforge.py make --name <name> --from animate` retargets it, plants the feet and holds it
                 inside the joint limits like every other clip (--rebuild runs that)

Nothing in the character's own build is changed until --pick and the rerun from animate.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
U = ROOT / "vendor" / "unimate"
PY, BPY = U / ".venv" / "bin" / "python", U / ".venv-bpy" / "bin" / "python"
EXP_SRC = U / "weights" / "unimate_uniml3d_f60_v2"
ENV = dict(os.environ, HF_HOME=str(U / "hf_home"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           PYTHONPATH=str(U))


def run(cmd, what, cwd=U, env=ENV):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[unimate] {what} failed:\n" + "\n".join((r.stdout + r.stderr).splitlines()[-15:]))
    return r.stdout + r.stderr


def preprocess(name, glb, char_dir, prefix):
    cond = char_dir / "cond.npy"
    if cond.exists() and cond.stat().st_mtime > glb.stat().st_mtime and (char_dir / f"{glb.stem}_canonical.glb").exists():
        return
    shutil.rmtree(char_dir, ignore_errors=True)
    run([BPY, "data_process/mesh_animation/preprocess_char.py", "--", "--char_path", glb, "--output_dir", char_dir,
         "--face_r", f"{prefix}RightUpLeg", "--face_l", f"{prefix}LeftUpLeg", "--formats", "glb"], "preprocess")
    print(f"[unimate] {name}: skeleton and clips read by UniMate -> {char_dir}", flush=True)


def exp_dir(char_dir, exp):
    c = json.load(open(EXP_SRC / "config.json"))
    c["dataset"]["dataset_list"] = ["objaverse"]
    c["objaverse"] = {"type": "objaverse", "path": str(char_dir), "objects_num": -1, "filter_object": False}
    (exp / "checkpoints").mkdir(parents=True, exist_ok=True)
    json.dump(c, open(exp / "config.json", "w"), indent=2)
    shutil.copy(EXP_SRC / "dataset_stats.npy", exp / "dataset_stats.npy")
    ck = exp / "checkpoints" / "checkpoint_step_100000.pt"
    if not ck.exists():
        ck.symlink_to(EXP_SRC / "checkpoints" / "checkpoint_step_100000.pt")


def blender_bin():
    return subprocess.run([sys.executable, "-c", "import charforge; print(charforge.blender_bin())"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip().splitlines()[-1]


def contact_sheet(move, files, text, out_png):
    """Every sample of a move at five moments, one row each - what to choose from."""
    from PIL import Image, ImageDraw
    rows = []
    for r, f in enumerate(files):
        pre = Path(f).with_suffix("")
        frames = sorted(glob.glob(f"{pre}_f_*.png"))
        if not frames:
            run([blender_bin(), "-b", "-noaudio", "--python", ROOT / "blender" / "render_frames.py", "--",
                 "--file", Path(f).with_suffix(".glb") if Path(f).with_suffix(".glb").exists() else f,
                 "--out", f"{pre}_f", "--frames", "0,12,24,36,47", "--res", "300", "400"], f"preview {move} {r}",
                cwd=ROOT, env=os.environ)
            frames = sorted(glob.glob(f"{pre}_f_*.png"))
        ims = [Image.open(x).convert("RGB") for x in frames]
        w, h = ims[0].size
        row = Image.new("RGB", (w * len(ims), h + 22), "white")
        ImageDraw.Draw(row).text((6, 5), f"sample {r}", fill="black")
        for i, im in enumerate(ims):
            row.paste(im, (i * w, 22))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(x.height for x in rows) + 26), "white")
    ImageDraw.Draw(sheet).text((6, 6), f"{move}: \"{text}\" - 0, 0.4, 0.8, 1.2, 1.6 s", fill="black")
    y = 26
    for x in rows:
        sheet.paste(x, (0, y))
        y += x.height
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png)


def add_moves(name, record, picks):
    """Write the chosen samples into the character's extra clips."""
    extra = ROOT / "work" / name / "extra_clips.json"
    have = json.load(open(extra)) if extra.exists() else {}
    for move, rep in picks.items():
        files = record["moves"][move]["fbx"]
        if not 0 <= rep < len(files):
            raise SystemExit(f"[unimate] {move} has samples 0-{len(files) - 1}, not {rep}")
        have[move] = {"file": files[rep], "mirror": False, "from_text": record["moves"][move]["text"], "engine": "unimate",
                      "sample": rep, "seed": record["moves"][move]["seed"]}
    json.dump(have, open(extra, "w"), indent=1)
    print(f"[unimate] added to the character's clips: " + ", ".join(f"{m} (sample {r})" for m, r in picks.items()), flush=True)


def main(a):
    name = a.name
    w = ROOT / "work" / name / "unimate"
    rec_path = w / "moves.json"
    record = json.load(open(rec_path)) if rec_path.exists() else {"moves": {}}
    record.setdefault("moves", {})
    if any(not isinstance(v, dict) for v in record["moves"].values()):     # an older run's layout
        record = {"moves": {}}
    if a.move:
        generate(a, name, w, record)
        json.dump(record, open(rec_path, "w"), indent=1)
    picks = dict((m.split("=")[0], int(m.split("=")[1])) for m in (a.pick or []))
    if a.add and a.move:
        picks.update({m.split("=", 1)[0]: 0 for m in a.move})
    if picks:
        add_moves(name, record, picks)
        if a.rebuild:
            print(f"[unimate] {name}: retargeting, planting and clearing every clip again (charforge.py make --from animate)",
                  flush=True)
            r = subprocess.run([sys.executable, "-u", ROOT / "charforge.py", "make", "--name", name, "--from", "animate"], cwd=ROOT)
            if r.returncode != 0:
                raise SystemExit("[unimate] the rebuild failed")


def generate(a, name, w, record):
    pkg = ROOT / "out" / name
    glb = pkg / f"{name}.glb"
    if not glb.exists():
        raise SystemExit(f"no package for {name}: run charforge.py make first")
    man = json.load(open(pkg / f"{name}.json"))
    prefix = man.get("skeleton", {}).get("prefix", "mixamorig:")
    char_dir, exp = w / "char", w / "exp"
    # without morph targets: UniMate's canonical bake moves the mesh but not its shape keys (Cadet tore apart)
    src = w / f"{name}.glb"
    if not src.exists() or src.stat().st_mtime < glb.stat().st_mtime:
        w.mkdir(parents=True, exist_ok=True)
        run([blender_bin(), "-b", "-noaudio", "--python", ROOT / "blender" / "unimate_input.py", "--", "--glb", glb,
             "--out", src], "the input without morph targets", cwd=ROOT, env=os.environ)
    preprocess(name, src, char_dir, prefix)
    exp_dir(char_dir, exp)
    import numpy as np
    obj = list(np.load(char_dir / "cond.npy", allow_pickle=True).item())[0]
    moves = dict(m.split("=", 1) for m in a.move)
    bad = [m for m in moves if not re.match(r"^[a-z][a-z0-9_]{0,31}$", m)]
    if bad:
        raise SystemExit(f"[unimate] a move name is lowercase letters, digits and _: {bad}")
    cases_path = w / "cases.json"
    json.dump({f"{obj}-{k}": v for k, v in moves.items()}, open(cases_path, "w"), indent=1)
    out = w / "samples"
    shutil.rmtree(out, ignore_errors=True)
    sys.path.insert(0, str(ROOT))
    sys.argv, argv0 = [sys.argv[0]], sys.argv
    import charforge                                       # the GPU lock shared with every other CharForge job
    sys.argv = argv0
    with charforge.gpu("UniMate sampling") if a.device != "cpu" else contextlib.nullcontext():
        run([PY, ROOT / "tools" / "unimate_sample.py", "--steps", a.steps, "--method", a.method, "--device", a.device,
             "--exp_dir", exp, "--test_cases_json", cases_path, "--num_repetitions", a.reps, "--only_save_motion",
             "--seed", a.seed, "--output_dir", out], "sampling")
    npys = sorted(glob.glob(str(out / "**" / "*.npy"), recursive=True))
    print(f"[unimate] {len(npys)} motions for {len(moves)} moves x {a.reps}", flush=True)
    canon = char_dir / f"{src.stem}_canonical.glb"
    made = {}
    for npy in npys:
        stem = Path(npy).stem                                  # <obj>-<move>-rep_<r>-<i>
        move = stem[len(obj) + 1:].split("-rep_")[0]
        rep = int(stem.split("-rep_")[1].split("-")[0]) if "-rep_" in stem else 0
        dst = w / "fbx" / f"{move}_s{a.seed}_r{rep}"
        shutil.rmtree(dst, ignore_errors=True)
        run([BPY, "data_process/mesh_animation/animate_motion.py", "--", "--dataset_type", "objaverse",
             "--anim_path", npy, "--char_path", canon, "--cond_path", char_dir / "cond.npy", "--output_dir", dst],
            f"export {stem}")
        fbx = sorted(dst.glob("*.fbx"))
        if fbx:
            made.setdefault(move, {})[rep] = str(fbx[0])
    for move, reps in made.items():
        files = [reps[r] for r in sorted(reps)]
        record["moves"][move] = {"text": moves[move], "seed": a.seed, "steps": a.steps, "fbx": files}
        if not a.no_preview:
            contact_sheet(move, files, moves[move], ROOT / "work" / name / "qa" / "words" / f"{move}.png")
    print(f"[unimate] exported: " + ", ".join(f"{m} ({len(v)} samples)" for m, v in made.items())
          + ("" if a.no_preview else f"; previews in work/{name}/qa/words/"), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--move", action="append", default=[], help="name=text: generate (repeatable)")
    ap.add_argument("--pick", action="append", default=[], help="name=sample: add that sample of a generated move (repeatable)")
    ap.add_argument("--rebuild", action="store_true", help="after adding, rebuild the character from the animate stage")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--method", default="euler")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--add", action="store_true", help="add sample 0 of each generated move to the character's clips")
    main(ap.parse_args())
