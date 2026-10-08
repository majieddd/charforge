"""Headless Make-It-Animatable v2 on one mesh (experiment E157): joints and skin weights, no gradio, no CUDA.

    vendor/Make-It-Animatable/.venv/bin/python pipeline/mia_infer.py --mesh m.glb --out m_mia.npz [--device cpu|mps]
        [--mia-dir vendor/Make-It-Animatable]       (tools/setup_mia.sh makes that folder)

Writes bw (the mesh's vertices as trimesh loads it, by 52 Mixamo joints), verts (the same vertices, in the input's coordinates),
joint heads and tails and the joint names; pipeline/mia_weights.py maps them to the pipeline's own vertices by position."""
import argparse
import os
import sys
import time
import types
from unittest import mock

SELF = os.path.dirname(os.path.abspath(__file__))
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--mia-dir", default=os.path.join(SELF, "..", "vendor", "Make-It-Animatable"))
HERE = os.path.abspath(_pre.parse_known_args()[0].mia_dir)
sys.path.insert(0, os.path.join(HERE, "pyshim"))        # PyTorch3D's pure-Python transforms (no C extension needed)
sys.path.insert(0, os.path.join(SELF, "mia_shims"))     # torch_cluster.fps in plain PyTorch
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "util", "Hunyuan3D_21", "hy3dshape"))
sys.path.insert(0, os.path.join(HERE, "util", "Hunyuan3D_21"))
sp = types.ModuleType("spaces")
sp.GPU = lambda f=None, **k: f if callable(f) else (lambda g: g)
sys.modules["spaces"] = sp
for m in ("gradio", "gradio.helpers", "bpy", "bpy.types", "bpy.props", "bpy_extras", "bpy_extras.object_utils", "mathutils", "mathutils.geometry", "bmesh", "fake_bpy_module", "pymeshlab", "cv2"):
    sys.modules[m] = mock.MagicMock()

import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mia-dir", default=HERE)
ap.add_argument("--mesh", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--device", default="cpu")
ap.add_argument("--threads", type=int, default=3)
a = ap.parse_args()
torch.set_num_threads(a.threads)
mesh_path, out_path = os.path.abspath(a.mesh), os.path.abspath(a.out)

# The kinematic tree is read from a Mixamo template FBX with Blender in the original; the hierarchy is the standard one.
os.makedirs(os.path.join(HERE, "data", "Mixamo"), exist_ok=True)
open(os.path.join(HERE, "data", "Mixamo", "bones.fbx"), "a").close()
from util import blender_utils  # noqa: E402


class _Bone:
    def __init__(self, name, parent=None):
        self.name, self.parent, self.children = name, parent, []
        if parent is not None:
            parent.children.append(self)


def _mixamo_armature():
    P = "mixamorig:"
    bones = {}

    def add(n, parent=None):
        bones[P + n] = _Bone(P + n, bones[P + parent] if parent else None)

    add("Hips")
    for n, par in (("Spine", "Hips"), ("Spine1", "Spine"), ("Spine2", "Spine1"), ("Neck", "Spine2"), ("Head", "Neck")):
        add(n, par)
    for s in ("Left", "Right"):
        add(f"{s}Shoulder", "Spine2")
        add(f"{s}Arm", f"{s}Shoulder")
        add(f"{s}ForeArm", f"{s}Arm")
        add(f"{s}Hand", f"{s}ForeArm")
        for f in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            add(f"{s}Hand{f}1", f"{s}Hand")
            add(f"{s}Hand{f}2", f"{s}Hand{f}1")
            add(f"{s}Hand{f}3", f"{s}Hand{f}2")
        add(f"{s}UpLeg", "Hips")
        add(f"{s}Leg", f"{s}UpLeg")
        add(f"{s}Foot", f"{s}Leg")
        add(f"{s}ToeBase", f"{s}Foot")
    arm = types.SimpleNamespace(data=types.SimpleNamespace(bones=bones))
    return arm


blender_utils.reset = lambda *a, **k: None
blender_utils.load_file = lambda p, *a, **k: p
blender_utils.get_armature_obj = lambda *a, **k: _mixamo_armature()
import app_v2 as A  # noqa: E402

t0 = time.time()
A.init_models()                       # chdir to HERE, loads the three checkpoints to the CPU
for m in (A.model_bw, A.model_coarse, A.model_pose):
    m.float()                         # the checkpoints are half precision; autocast was CUDA's job
if a.device != "cpu":
    dev = torch.device(a.device)
    for m in (A.model_bw, A.model_coarse, A.model_pose):
        m.to(dev)
    A.device = dev
print(f"[mia] models loaded {time.time() - t0:.0f}s on {A.device}", flush=True)

for _n in ("state", "output_joints_coarse", "output_normed_input", "output_sample", "output_joints", "output_bw",
           "output_rest_lbs", "output_rest_vis", "output_anim_vis", "output_anim"):      # gradio components the steps return
    setattr(A, _n, _n)
db = A.DB()
A.prepare_input(mesh_path, False, 0.0, db, True)
t1 = time.time()
A.preprocess(db)
A.infer(True, db)
print(f"[mia] inference {time.time() - t1:.0f}s", flush=True)
A.vis(True, "LeftArm", False, True, db)               # post-process the weights, back to the input's coordinates
names = [n.replace(A.MIXAMO_PREFIX, "") for n in A.BONES_IDX_DICT]
np.savez_compressed(out_path, bw=db.bw.astype(np.float32), verts=db.verts.astype(np.float32), joints=db.joints, joints_tail=db.joints_tail,
                    names=np.array(names), faces=np.asarray(db.faces))
print(f"[mia] {len(db.verts)} vertices, {db.bw.shape[1]} joints -> {out_path}  ({time.time() - t0:.0f}s)", flush=True)
