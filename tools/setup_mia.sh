#!/bin/zsh
# Make-It-Animatable v2 (CVPR 2025, Guo et al.): learned skin weights for the weights stage (experiment E157). Optional; the
# pipeline falls back to its geodesic weights when this folder is absent.
#   zsh tools/setup_mia.sh        (about 4 GB of weights, ~1 GB environment; CPU only, nothing needs CUDA)
# Code: github.com/jasongzy/Make-It-Animatable (MIT), branch v2, with its Hunyuan3D-2.1 submodule (Tencent's community licence,
# which excludes the EU, UK and South Korea). Weights: huggingface.co/jasongzy/Make-It-Animatable output/best/v2 (Apache-2.0
# per the model card). The model was trained on Mixamo characters; read the terms of each before any use beyond research.
set -e
cd "$(dirname "$0")/.."
D=vendor/Make-It-Animatable
mkdir -p vendor
[[ -d $D/.git ]] || git clone --depth 1 https://github.com/jasongzy/Make-It-Animatable $D
cd $D
git fetch --depth 1 origin v2
git checkout -q -B v2 FETCH_HEAD
git submodule update --init --depth 1 util/Hunyuan3D_21          # the ShapeVAE the v2 models are built on
[[ -d .venv ]] || uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch torchvision "numpy<2" einops timm trimesh scipy potpourri3d PyMCubes plyfile \
    shapely scikit-image omegaconf pyyaml tqdm transformers diffusers accelerate matplotlib huggingface_hub
# MIA imports PyTorch3D only for its (pure Python) transforms: the package folder without its C extension is enough
if [[ ! -d pyshim/pytorch3d ]]; then
  T=$(mktemp -d)
  git clone --depth 1 --branch V0.7.8 https://github.com/facebookresearch/pytorch3d $T/p3d
  mkdir -p pyshim && cp -R $T/p3d/pytorch3d pyshim/pytorch3d && rm -rf $T
fi
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("jasongzy/Make-It-Animatable", allow_patterns=["output/best/v2/*"], local_dir=".", max_workers=2)
PY
echo "Make-It-Animatable ready: python charforge.py make --name <n> --skin-weights mia   (or let 'auto' try it)"
