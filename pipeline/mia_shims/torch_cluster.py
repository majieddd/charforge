"""Pure-PyTorch stand-in for torch_cluster.fps (farthest point sampling), which needs a CUDA/C++ build. Same call as
torch_cluster's: fps(src (N, D), batch (N,), ratio) -> indices into src, concatenated over the batch."""
import math

import torch


def fps(src, batch=None, ratio=0.5, random_start=True, batch_size=None, ptr=None):
    if batch is None:
        batch = torch.zeros(src.shape[0], dtype=torch.long, device=src.device)
    out = []
    for b in torch.unique(batch):
        idx = (batch == b).nonzero().squeeze(1)
        x = src[idx]
        n = x.shape[0]
        k = max(1, int(math.ceil(float(ratio) * n)))
        sel = torch.zeros(k, dtype=torch.long, device=src.device)
        dist = torch.full((n,), float("inf"), device=src.device, dtype=x.dtype)
        far = int(torch.randint(n, (1,)).item()) if random_start else 0
        for i in range(k):
            sel[i] = far
            dist = torch.minimum(dist, ((x - x[far]) ** 2).sum(1))
            far = int(torch.argmax(dist).item())
        out.append(idx[sel])
    return torch.cat(out)
