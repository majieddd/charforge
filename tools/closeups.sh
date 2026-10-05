#!/bin/zsh
# Eight lit close-ups of a character - face, three-quarter, a hand from above and both hanging in idle, a shoulder,
# the upper back, a foot - and one sheet of them (E138): where the faults a person sees first show, before any average.
#   zsh tools/closeups.sh <name> [<final.blend>]        (from charforge/; renders on the CPU, beside a running build)
# Writes work/<name>/qa/closeups/*.png and work/<name>/qa/closeups.png.
n=$1; blend=${2:-work/$n/final.blend}
O=work/$n/qa/closeups; mkdir -p $O
BL=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
J() { $BL -b -noaudio --python blender/render_joint.py -- --blend $blend --res 480 --cpu "$@" > /dev/null 2>&1; }
J --out $O/face --bone head --span 0.15 --az 0 --elev 0
J --out $O/face34 --bone head --span 0.15 --az 50 --elev 0
J --out $O/lhand_top --bone left_wrist --span 0.15 --az 0 --elev 75
J --out $O/lhand_idle --bone left_wrist --span 0.17 --az 70 --elev 0 --clip idle --at 0.3
J --out $O/rhand_idle --bone right_wrist --span 0.17 --az 70 --elev 0 --clip idle --at 0.3
J --out $O/lshoulder --bone left_shoulder --span 0.3 --az 35 --elev -12
J --out $O/back --bone spine2 --span 0.4 --az 180 --elev 0
J --out $O/lfoot --bone left_ankle --span 0.22 --az 30 --elev 10
python3 - $O $n <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw
d, n = Path(sys.argv[1]), sys.argv[2]
names = ["face_rest", "face34_rest", "lhand_top_rest", "lhand_idle_idle_030", "rhand_idle_idle_030", "lshoulder_rest", "back_rest", "lfoot_rest"]
W = 300
out = Image.new("RGB", (4 * (W + 4), 2 * (W + 4) + 20), (255, 255, 255))
ImageDraw.Draw(out).text((4, 3), f"{n}: face, three-quarter, left hand from above, hands hanging (idle), shoulder, upper back, foot", fill=(0, 0, 0))
for i, k in enumerate(names):
    p = d / f"{k}.png"
    im = Image.open(p).convert("RGB").resize((W, W)) if p.exists() else Image.new("RGB", (W, W), (90, 0, 0))
    out.paste(im, ((i % 4) * (W + 4), 20 + (i // 4) * (W + 4)))
out.save(d.parent / "closeups.png")
print(f"[closeups] -> {d.parent / 'closeups.png'}")
PY
