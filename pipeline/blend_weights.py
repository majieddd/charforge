"""Skin weights that are softer where softness helps and default where it does not (experiment E154).

    python pipeline/blend_weights.py --sharp weights_sharp.npz --soft weights_soft.npz --out weights_hybrid.npz

A gentler falloff and more smoothing (geodesic_weights.py --power 2 --smooth 30) take the stretch and crush out of a loose
jacket's armpit, and cost a boot's ankle: on Cadet they make a tiptoe 30% more stretched and a heel strike twice as sheared, and
his run pops at the foot. So each vertex takes the soft weights in proportion to how little it follows the legs: its share of
weight on the hips, knees and ankles under the default weights, L, gives L * default + (1 - L) * soft - the legs and feet as
before, the arms and torso soft, a smooth change at the hips. The head and neck keep the default too: soft there, Juno's and
Knight's heads stretched and sheared more in the clips (head turn in the battery 1.01 -> 1.10 for Juno; E160). The strongest
four influences are kept, as geodesic_weights.py does."""
import argparse

import numpy as np


def dense(Z, nb):
    n = Z["index"].shape[0]
    D = np.zeros((n, nb))
    np.add.at(D, (np.repeat(np.arange(n)[:, None], Z["index"].shape[1], 1), Z["index"].astype(int)), Z["weight"].astype(float))
    return D


KEEP = ("_hip", "_knee", "_ankle", "head", "neck")


def blend(sharp, soft, k=4, keep=KEEP):
    bones = [str(b) for b in sharp["bones"]]
    assert bones == [str(b) for b in soft["bones"]], "the two weight sets must be over the same bones"
    leg = np.array([b.endswith(keep) or b in keep for b in bones])
    Ds, Df = dense(sharp, len(bones)), dense(soft, len(bones))
    L = Ds[:, leg].sum(1, keepdims=True)
    H = L * Ds + (1 - L) * Df
    idx = np.argsort(-H, axis=1)[:, :k]
    w = np.take_along_axis(H, idx, 1)
    w /= np.maximum(w.sum(1, keepdims=True), 1e-9)
    return sharp["bones"], idx.astype(np.int16), w.astype(np.float32), L[:, 0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sharp", required=True)
    ap.add_argument("--soft", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", default=",".join(KEEP), help="bones (or name endings) whose vertices keep the default weights")
    a = ap.parse_args()
    bones, idx, w, L = blend(np.load(a.sharp, allow_pickle=True), np.load(a.soft, allow_pickle=True),
                             keep=tuple(x for x in a.keep.split(",") if x))
    np.savez_compressed(a.out, bones=bones, index=idx, weight=w)
    print(f"[weights] hybrid: {int((L > 0.5).sum()):,} of {len(L):,} vertices follow the legs; {a.out}", flush=True)
