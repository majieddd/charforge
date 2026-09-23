"""Report what a checkout is missing before anything slow starts.

The pipeline spans three environments that fail in different ways: the host Python (this one),
Blender's bundled interpreter, and the vendored generation stage with its model weights. A
missing piece in any of them currently surfaces halfway through a long run, as an import error
or - worse - as a Blender operator that returns success while doing nothing.

    python tools/check_env.py
"""
from __future__ import annotations

import importlib.metadata as md
import os
import shutil
import subprocess
import sys

OK, BAD, WARN = "  ok  ", " MISS ", " warn "
rows, missing = [], 0


def note(state, what, detail=""):
    global missing
    rows.append((state, what, detail))
    if state == BAD:
        missing += 1


# ---- host python -------------------------------------------------------------------------
req = os.path.join(os.path.dirname(__file__), os.pardir, "requirements.txt")
want = []
if os.path.exists(req):
    for line in open(req):
        line = line.split("#")[0].strip()
        if line:
            want.append(line.split("==")[0])
for pkg in want:
    try:
        note(OK, f"python: {pkg}", md.version(pkg))
    except md.PackageNotFoundError:
        note(BAD, f"python: {pkg}", "pip install -r requirements.txt")

if sys.version_info < (3, 11):
    note(WARN, "python version", f"{sys.version.split()[0]} - built against 3.11+")
else:
    note(OK, "python version", sys.version.split()[0])

# ---- blender -----------------------------------------------------------------------------
blender = os.environ.get("BLENDER") or shutil.which("blender") \
    or "/Applications/Blender.app/Contents/MacOS/Blender"
if os.path.exists(blender) or shutil.which(blender):
    try:
        v = subprocess.run([blender, "--version"], capture_output=True, text=True,
                           timeout=60).stdout.strip().splitlines()[0]
        major = int(v.split()[1].split(".")[0])
        note(OK if major >= 5 else WARN, "blender", f"{v} at {blender}")
    except Exception as e:                                    # noqa: BLE001
        note(WARN, "blender", f"found at {blender} but would not report a version: {e}")
else:
    note(BAD, "blender", "set BLENDER=/path/to/blender, or put it on PATH")

# ---- generation stage --------------------------------------------------------------------
root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
trellis = os.path.join(root, "vendor", "trellis2mlx")
if os.path.isdir(trellis):
    note(OK, "vendor/trellis2mlx", trellis)
    patched = os.path.join(trellis, "trellmlx", "samplers.py")
    if os.path.exists(patched) and "MultiViewConditioning" in open(patched, encoding="utf-8",
                                                                  errors="ignore").read():
        note(OK, "stochastic multi-view patch", "applied")
    else:
        note(BAD, "stochastic multi-view patch",
             "git -C vendor/trellis2mlx apply ../../patches/trellis2mlx-stochastic-multiview.patch")
else:
    note(BAD, "vendor/trellis2mlx",
         "git clone https://github.com/lyonsno/trellis2mlx vendor/trellis2mlx")

try:
    note(OK, "mlx", md.version("mlx"))
except md.PackageNotFoundError:
    note(WARN, "mlx", "needed only for the generation stage; lives with trellis2mlx")

# ---- the characters the playground ships ------------------------------------------------------
import json as _json
roster = os.path.join(root, "web", "roster.json")
ids = [c["id"] for c in _json.load(open(roster))["characters"]] if os.path.exists(roster) else []
for cid in ids:
    glb = os.path.join(root, "docs", f"{cid}.glb")
    if os.path.exists(glb):
        note(OK, f"docs/{cid}.glb", f"{os.path.getsize(glb)/1e6:.2f} MB")
    else:
        note(WARN, f"docs/{cid}.glb", "absent; run tools/build_site.py after building it")
if not ids:
    note(WARN, "web/roster.json", "no roster; the playground has nothing to load")

w = max(len(r[1]) for r in rows)
print()
for state, what, detail in rows:
    print(f"[{state}] {what.ljust(w)}  {detail}")
print()
if missing:
    print(f"{missing} missing. The Blender stages and the playground do not need the "
          f"generation stage - a checkout can run everything downstream of `work/` without it.")
    sys.exit(1)
print("Everything the pipeline needs is present.")
