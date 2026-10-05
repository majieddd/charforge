#!/bin/zsh
# Install Hunyuan3D 2.1 for texture views (charforge.py's texture stage, --texture-views hunyuan) - about 30 GB.
#
#   tools/setup_hunyuan.sh          (from charforge/)
#
# What it downloads, and from where:
#   code      github.com/VladimirTalyzin/hunyuan3d-2.1-mac-rocm (MIT wrappers; runs Tencent's Hunyuan3D 2.1 on Apple
#             Silicon), pinned below, into vendor/hunyuan3d-2.1-mac-rocm; its install.sh then clones
#             github.com/Tencent-Hunyuan/Hunyuan3D-2.1, makes a Python venv (PyTorch with MPS, its requirements),
#             builds one native extension (no CUDA) and downloads the weights: tencent/Hunyuan3D-2.1 shape and
#             PBR paint models (~14 GB), facebook/dinov2-giant (~4.5 GB) and RealESRGAN_x4plus (64 MB)
#   delight   tencent/Hunyuan3D-2, folder hunyuan3d-delight-v2-0 (~4.3 GB), for tools/delight.py
# Licences: the wrappers are MIT; the Hunyuan3D weights are under the Tencent Hunyuan 3D community licences, which
# exclude the EU, the UK and South Korea and need a separate licence above 100 million monthly users - read them.
# install.sh installs python@3.11 and git-lfs with Homebrew if they are missing, and runs Apple's command line
# tools installer if those are missing.
set -e
cd "$(dirname "$0")/.."
H=vendor/hunyuan3d-2.1-mac-rocm
PIN=6f4b63b
if [ ! -d $H ]; then
  git clone -q https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac-rocm.git $H
  git -C $H checkout -q $PIN
fi
[ -x $H/venv/bin/python ] && [ -d $H/weights/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1 ] || (cd $H && ./install.sh)
$H/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print("delight model:", snapshot_download("tencent/Hunyuan3D-2", allow_patterns=["hunyuan3d-delight-v2-0/*"]))
PY
$H/venv/bin/python $H/scripts/doctor.py | tail -3
echo "Hunyuan3D 2.1 is installed in $H"
