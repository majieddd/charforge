"""Probe how far Pixal3D gets on Apple Silicon, and validate the sparse-conv backend written for it.

Two questions:
  1. is the pure-PyTorch submanifold sparse convolution correct? (checked against a dense
     conv3d reference on a fully occupied grid, where the two must agree exactly)
  2. do Pixal3D's sparse conv + attention paths execute on MPS with the pure-PyTorch backends?
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_CONV_BACKEND", "none")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

PIX = Path(__file__).resolve().parents[1] / "vendor" / "Pixal3D"
sys.path.insert(0, str(PIX))

import torch  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
results = {}


def check_conv_correctness():
    """On a fully occupied grid, submanifold sparse conv == dense conv3d with padding=k//2."""
    import pixal3d.modules.sparse as sp
    torch.manual_seed(0)
    D = H = W = 6
    Ci, Co, K = 5, 7, 3
    dense = torch.randn(1, Ci, D, H, W)

    zz, yy, xx = torch.meshgrid(torch.arange(D), torch.arange(H), torch.arange(W), indexing="ij")
    coords = torch.stack([torch.zeros_like(zz).reshape(-1), zz.reshape(-1),
                          yy.reshape(-1), xx.reshape(-1)], 1).int()
    feats = dense[0].permute(1, 2, 3, 0).reshape(-1, Ci).contiguous()
    st = sp.SparseTensor(feats=feats, coords=coords)

    conv = sp.SparseConv3d(Ci, Co, K, bias=True)
    ref = torch.nn.Conv3d(Ci, Co, K, padding=K // 2, bias=True)
    # conv.weight is (Co, Kd, Kh, Kw, Ci); Conv3d wants (Co, Ci, Kd, Kh, Kw)
    with torch.no_grad():
        ref.weight.copy_(conv.weight.permute(0, 4, 1, 2, 3).contiguous())
        ref.bias.copy_(conv.bias)

    got = conv(st).feats
    want = ref(dense)[0].permute(1, 2, 3, 0).reshape(-1, Co)
    err = (got - want).abs().max().item()
    results["conv_max_abs_error_vs_dense"] = round(err, 6)
    results["conv_correct"] = err < 2e-4
    print(f"[probe] sparse conv vs dense conv3d: max abs error {err:.2e} "
          f"({'match' if err < 2e-4 else 'MISMATCH'})", flush=True)


def check_mps_forward(n_vox=20000, ci=64, co=64):
    """A realistic-size sparse layer on the GPU."""
    import pixal3d.modules.sparse as sp
    torch.manual_seed(0)
    res = 64
    coords = torch.randint(0, res, (n_vox, 3))
    coords = torch.unique(coords, dim=0)
    coords = torch.cat([torch.zeros(len(coords), 1, dtype=torch.long), coords], 1).int()
    feats = torch.randn(len(coords), ci)
    st = sp.SparseTensor(feats=feats.to(DEV), coords=coords.to(DEV))
    conv = sp.SparseConv3d(ci, co, 3, bias=True).to(DEV)
    t0 = time.perf_counter()
    out = conv(st)
    if DEV == "mps":
        torch.mps.synchronize()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = conv(st)                       # neighbour map now cached
    if DEV == "mps":
        torch.mps.synchronize()
    cached = time.perf_counter() - t0
    results["mps_voxels"] = int(len(coords))
    results["mps_first_forward_s"] = round(first, 3)
    results["mps_cached_forward_s"] = round(cached, 3)
    results["mps_output_shape"] = list(out.feats.shape)
    print(f"[probe] {len(coords):,} voxels on {DEV}: first forward {first:.2f}s "
          f"(builds neighbour map), cached forward {cached:.3f}s -> {tuple(out.feats.shape)}",
          flush=True)


def check_attention():
    from pixal3d.modules.attention import scaled_dot_product_attention
    q = torch.randn(2, 128, 8, 32, device=DEV)
    k = torch.randn(2, 128, 8, 32, device=DEV)
    v = torch.randn(2, 128, 8, 32, device=DEV)
    o = scaled_dot_product_attention(q, k, v)
    results["attention_backend"] = os.environ["ATTN_BACKEND"]
    results["attention_output_shape"] = list(o.shape)
    print(f"[probe] attention ({os.environ['ATTN_BACKEND']}) on {DEV}: {tuple(o.shape)}", flush=True)


def check_imports():
    """Which Pixal3D modules import cleanly without CUDA-only packages."""
    mods = ["pixal3d.modules.sparse", "pixal3d.modules.attention",
            "pixal3d.modules.sparse.transformer", "pixal3d.models.sparse_structure_flow",
            "pixal3d.models.structured_latent_flow", "pixal3d.pipelines"]
    ok, bad = [], {}
    for m in mods:
        try:
            __import__(m)
            ok.append(m)
        except Exception as e:
            bad[m] = f"{type(e).__name__}: {e}"[:160]
    results["imports_ok"] = ok
    results["imports_failed"] = bad
    for m in ok:
        print(f"[probe] import OK   {m}", flush=True)
    for m, e in bad.items():
        print(f"[probe] import FAIL {m}  {e}", flush=True)


if __name__ == "__main__":
    print(f"[probe] device={DEV} torch={torch.__version__}", flush=True)
    check_imports()
    check_conv_correctness()
    check_mps_forward()
    check_attention()
    import json
    out = Path(__file__).resolve().parents[1] / "work" / "pixal3d_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    print("[probe] " + json.dumps({k: v for k, v in results.items() if k != "imports_ok"}), flush=True)
