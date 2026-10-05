#!/bin/zsh
# A character's face beside its picture: the reference's face, then the face rendered in soft studio light from the
# front, at three-quarter and in profile - each part (eyes, brows, nose, mouth, skin, hairline) large enough to judge
# (E140). Framed from the eye distance the face stage measured, so every face fills the frame alike.
#   zsh tools/face_review.sh <name> [<final.blend>] [<out.png>]     (from charforge/; CPU)
n=$1; blend=${2:-work/$n/final.blend}; out=${3:-work/$n/qa/face_review.png}
O=$(dirname $out)/face_review_parts/$n; mkdir -p $O
BL=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
span=$(../.venv/bin/python -c "
import json
try:
    ied = json.load(open('work/$n/face.json'))['ied_m']; h = json.load(open('out/$n/$n.json'))['height_m']
    print(round(min(0.2, max(0.07, 3.6 * ied / h)), 4))
except Exception:
    print(0.12)")
for az in 0 40 90; do
  $BL -b -noaudio --python blender/render_joint.py -- --blend $blend --out $O/az$az --bone head --span $span --az $az --elev 0 --res 640 --cpu --soft > /dev/null 2>&1
done
../.venv/bin/python - $n $O $out <<'PY'
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
n, O, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
W = 400
cells = []
ref = Path(f"work/{n}/texproj/face_src.png")
if ref.exists():
    cells.append(("picture", Image.open(ref).convert("RGB")))
for az, lab in (("0", "front"), ("40", "three-quarter"), ("90", "profile")):
    p = O / f"az{az}_rest.png"
    if p.exists():
        cells.append((lab, Image.open(p).convert("RGB")))
im = Image.new("RGB", (len(cells) * (W + 4), W + 22), (255, 255, 255))
d = ImageDraw.Draw(im)
for i, (lab, c) in enumerate(cells):
    w, h = c.size
    s = min(w, h)
    c = c.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((W, W))
    im.paste(c, (i * (W + 4), 20))
    d.text((i * (W + 4) + 4, 3), f"{n}: {lab}", fill=(0, 0, 0))
im.save(out)
print(f"[face] -> {out}")
PY
# and a number: the front render's face points against the picture's (tools/face_score.py)
pic=work/$n/reference_delit.png; [ -f $pic ] || pic=work/$n/reference.png
TPY=vendor/trellis2mlx/.venv/bin/python
[ -x $TPY ] && $TPY tools/face_score.py --render $O/az0_rest.png --picture $pic --mask work/$n/reference_mask.png --out ${out%.png}.json 2>&1 | tail -1
[ -f ${out%.png}.json ] && ../.venv/bin/python tools/face_likeness.py --score ${out%.png}.json 2>&1 | tail -1
