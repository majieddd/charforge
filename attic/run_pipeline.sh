#!/bin/zsh
# CharForge: prompt -> rigged, animated, verified game character. Entirely local.
#
#   ./run_pipeline.sh "a man in a green bomber jacket with medium-length wavy hair" char02
#
# Stages, each independently runnable and each leaving inspectable output in work/<name>/:
#   1 reference image   ComfyUI + Krea 2 Turbo
#   2 prompt gate       local VLM description -> typed pass/fail
#   3 geometry+texture  TRELLIS.2 via MLX
#   4 multi-view render Blender, with exact per-vertex visibility
#   5 part segmentation human parser back-projected onto the mesh
#   6 skeleton          ViTPose on the front/side renders -> 3D joints
#   7 rig               armature + per-part skinning
#   8 animation         six self-contained clips
#   9 verification      per-part deformation measurements
set -euo pipefail

PROMPT="${1:?usage: run_pipeline.sh \"prompt\" [name]}"
NAME="${2:-char_$(date +%H%M%S)}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
BASE="$(cd "$ROOT/.." && pwd)"
VENV="$BASE/.venv/bin/python"
BLENDER="${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}"
[ -x "$BLENDER" ] || BLENDER="$(command -v blender || true)"
[ -x "$BLENDER" ] || { print -u2 "Blender not found: set BLENDER=/path/to/blender"; exit 1; }
export CHARFORGE_ROOT="$ROOT"
OUT="$ROOT/work/$NAME"
mkdir -p "$OUT"

step() { print -P "\n%F{blue}==> [$1/9] $2%f"; }

step 1 "reference image"
"$VENV" - "$PROMPT" "$OUT" <<'PY'
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, os.path.join(os.environ["CHARFORGE_ROOT"], "pipeline"))
import comfy
prompt, out = sys.argv[1], pathlib.Path(sys.argv[2])
NEG = "cropped, close-up, multiple people, text, watermark, dramatic shadows, blurry, sitting"
full = (f"full body character reference sheet of {prompt}, standing in a relaxed A-pose, arms slightly "
        "away from the body, facing the camera directly, neutral expression, photorealistic game "
        "character, even neutral studio lighting, no shadows, plain flat light grey background, "
        "full figure visible head to toe, centered, orthographic view")
wf = comfy.krea2_t2i(full, NEG, width=832, height=1216, steps=8, cfg=1.0, seed=7, gguf=True)
paths, dt = comfy.run(wf, out, prefix="ref", timeout=2400)
print(f"    {paths[0]}  ({dt:.0f}s)")
PY
REF=$(ls "$OUT"/ref_*.png | head -1)

step 2 "prompt-match gate"
"$VENV" "$ROOT/pipeline/gates.py" --image "$REF" --out "$OUT/gate_reference.json" || true

step 3 "geometry + PBR texture (TRELLIS.2 / MLX)"
( cd "$ROOT/vendor/trellis2mlx" && source .venv/bin/activate \
  && TRELLIS_DINOV3_PATH="$ROOT/models/dinov3-vitl16-hf" PYTHONPATH=. \
     python -u generate.py --image "$REF" --output "$OUT/trellis_mesh.glb" \
       --resolution 1024 --steps 12 --target-faces 200000 --seed 7 ) \
  | grep -vE "it/s\]|B/s\]" | tail -6

step 4 "multi-view render"
"$BLENDER" -b -noaudio --python "$ROOT/blender/render_views.py" -- \
  --mesh "$OUT/trellis_mesh.glb" --out "$OUT/views" --views 8 --res 640 2>&1 | grep -E "render_views\]"

step 5 "part segmentation"
"$VENV" "$ROOT/pipeline/parts.py" --views "$OUT/views" 2>&1 | grep -E "\[parts\]"

step 6 "skeleton"
"$VENV" "$ROOT/pipeline/skeleton.py" --views "$OUT/views" 2>&1 | grep -E "\[skeleton\]"

step 7 "rig"
"$BLENDER" -b -noaudio --python "$ROOT/blender/build_rig.py" -- \
  --mesh "$OUT/trellis_mesh.glb" --joints "$OUT/joints.json" --parts "$OUT/parts.json" \
  --out "$OUT/rigged.glb" 2>&1 | grep -E "\[rig\]"

step 8 "animation"
"$BLENDER" -b -noaudio --python "$ROOT/blender/animate.py" -- \
  --rig "$OUT/rigged.blend" --out "$OUT/animated.glb" 2>&1 | grep -E "\[animate\]"

step 9 "deformation verification"
"$BLENDER" -b -noaudio --python "$ROOT/blender/verify_deform.py" -- \
  --blend "$OUT/animated.blend" --out "$OUT/verify.json" 2>&1 | grep -E "\[verify\]"

cp "$OUT/animated.glb" "$ROOT/viewer/character.glb"
cp "$OUT/verify.json"  "$ROOT/viewer/verify.json"
print -P "\n%F{green}done%f  $OUT/animated.glb"
print "view it:  python3 -m http.server -d $ROOT/viewer 8900   then open http://127.0.0.1:8900"
