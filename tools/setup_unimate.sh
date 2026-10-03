#!/bin/zsh
# Install UniMate for moves from words (tools/unimate_moves.py) into vendor/unimate - about 3.2 GB.
#
#   tools/setup_unimate.sh          (from charforge/; needs uv: https://docs.astral.sh/uv/)
#
# What it downloads, and from where:
#   code         github.com/Friedrich-M/UniMate (MIT), without its README images
#   weights      huggingface.co/Linzhan/UniMate  unimate_uniml3d_f60_v2: checkpoint_step_100000.pt (1.19 GB, MIT),
#                config.json, dataset_stats.npy
#   text encoder huggingface.co/google/flan-t5-base  model.safetensors (990 MB, Apache-2.0) + tokenizer
#   python       two environments: vendor/unimate/.venv (Python 3.11: torch, transformers, torchdiffeq, spacy ...)
#                for sampling, and vendor/unimate/.venv-bpy (Python 3.10: Blender 4.0 as a module, from
#                download.blender.org/pypi) for UniMate's own preprocessing and export, written for Blender 4.0
#   helpers      github.com/inbar-2344/Motion (Animation/Quaternions, pure Python), which UniMate imports
# The checkpoint is a PyTorch pickle; tools/unimate_sample.py loads it with torch's default weights_only=True.
set -e
cd "$(dirname "$0")/.."
U=vendor/unimate
H=https://huggingface.co
command -v uv >/dev/null || { echo "needs uv (https://docs.astral.sh/uv/)"; exit 1; }

if [ ! -d $U/unimate ]; then
  git clone -q --filter=blob:none --sparse https://github.com/Friedrich-M/UniMate.git $U
  git -C $U sparse-checkout set --no-cone '/*' '!/assets/'
fi

W=$U/weights/unimate_uniml3d_f60_v2
mkdir -p $W/checkpoints $U/weights/flan-t5-base
for f in config.json dataset_stats.npy; do
  [ -s $W/$f ] || curl -fsSL -o $W/$f $H/Linzhan/UniMate/resolve/main/unimate_uniml3d_f60_v2/$f
done
[ -s $W/checkpoints/checkpoint_step_100000.pt ] || curl -fL -o $W/checkpoints/checkpoint_step_100000.pt \
  $H/Linzhan/UniMate/resolve/main/unimate_uniml3d_f60_v2/checkpoints/checkpoint_step_100000.pt
for f in config.json generation_config.json special_tokens_map.json spiece.model tokenizer.json tokenizer_config.json model.safetensors; do
  [ -s $U/weights/flan-t5-base/$f ] || curl -fL -o $U/weights/flan-t5-base/$f $H/google/flan-t5-base/resolve/main/$f
done
# the text encoder as an offline Hugging Face cache entry, so nothing is fetched at run time
C=$U/hf_home/hub/models--google--flan-t5-base
mkdir -p $C/refs $C/snapshots/local
printf local > $C/refs/main
for f in $U/weights/flan-t5-base/*; do ln -sf "$PWD/$f" $C/snapshots/local/$(basename $f); done

if [ ! -x $U/.venv/bin/python ]; then
  uv venv -q -p 3.11 $U/.venv
  VIRTUAL_ENV=$U/.venv uv pip install -q torch torchdiffeq==0.2.5 einops==0.8.2 "transformers>=4.57" sentencepiece \
    "spacy>=3.8,<3.9" https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
    num2words==0.5.14 torch-geometric tyro loguru rich tqdm "numpy<2.3" huggingface_hub safetensors accelerate matplotlib \
    imageio pillow "setuptools<81" wheel
  VIRTUAL_ENV=$U/.venv uv pip install -q --no-build-isolation "git+https://github.com/inbar-2344/Motion.git"
fi
if [ ! -x $U/.venv-bpy/bin/python ]; then
  uv venv -q -p 3.10 $U/.venv-bpy
  VIRTUAL_ENV=$U/.venv-bpy uv pip install -q --extra-index-url https://download.blender.org/pypi/ \
    --index-strategy unsafe-best-match bpy==4.0.0 "numpy<2" loguru tqdm matplotlib imageio imageio-ffmpeg "setuptools<81" wheel
  VIRTUAL_ENV=$U/.venv-bpy uv pip install -q --no-build-isolation "git+https://github.com/inbar-2344/Motion.git"
fi
$U/.venv/bin/python -c "import torch, transformers, torchdiffeq; print('UniMate sampling env ok, torch', torch.__version__)"
$U/.venv-bpy/bin/python -c "import bpy; print('UniMate Blender', bpy.app.version_string)"
echo "UniMate is installed in $U"
