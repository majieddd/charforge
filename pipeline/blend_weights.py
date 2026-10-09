"""Skin weights that are softer where softness helps and default where it does not (experiment E154).

    python pipeline/blend_weights.py --sharp weights_sharp.npz --soft weights_soft.npz --out weights_hybrid.npz

A gentler falloff and more smoothing (geodesic_weights.py --power 2 --smooth 30) take the stretch and crush out of a loose
jacket's armpit, and cost a boot's ankle: on Cadet they make a tiptoe 30% more stretched and a heel strike twice as sheared, and
his run pops at the foot. So each vertex takes the soft weights in proportion to how little it follows the legs: its share of
weight on the hips, knees and ankles under the default weights, L, gives L * default + (1 - L) * soft - the legs and feet as
before, the arms and torso soft, a smooth change at the hips. The head and neck keep the default too: soft there, Juno's and
Knight's heads stretched and sheared more in the clips (head turn in the battery 1.01 -> 1.10 for Juno; E160). The strongest
four influences are kept, as geodesic_weights.py does.

weights2 added two opt-in variants on the armpit band, both judged by tools/weights_trial.py on Pip and neither adopted:
--arm-band lo,hi keeps the default weights wherever the default binds a vertex to the arm (sleeve and armhole stay with the
arm); --soft-arm lo,hi is its inverse: only the arm-bound vertices take the soft weights and the trunk keeps the default.
The inverse removes the land and victory_cheer regressions of the shipped hybrid but keeps sprint's three extra deep frames
(see tools/weights_trial.py output in the lane report). The defaults are unchanged: the shipped hybrid is reproduced exactly."""
import argparse

import numpy as np


def dense(Z, nb):
    n = Z["index"].shape[0]
    D = np.zeros((n, nb))
    np.add.at(D, (np.repeat(np.arange(n)[:, None], Z["index"].shape[1], 1), Z["index"].astype(int)), Z["weight"].astype(float))
    return D


KEEP = ("_hip", "_knee", "_ankle", "head", "neck")


ARM = ("shoulder", "elbow", "wrist")       # the arm bones of a Mixamo-like body: upper arm, forearm, hand (E161, weights2)


def blend(sharp, soft, k=4, keep=KEEP, arm_band=None, soft_arm=None):
    """The hybrid. arm_band=(lo, hi): a vertex whose default weights bind it to the arm by lo to hi (the sum over the arm
    bones, ramped) keeps the default weights as the legs do. Without it the sleeve and the armhole of a loose garment take
    the soft weights' lighter arm share and lag the arm, and the arm passes through them (weights2)."""
    bones = [str(b) for b in sharp["bones"]]
    assert bones == [str(b) for b in soft["bones"]], "the two weight sets must be over the same bones"
    leg = np.array([b.endswith(keep) or b in keep for b in bones])
    Ds, Df = dense(sharp, len(bones)), dense(soft, len(bones))
    L = Ds[:, leg].sum(1, keepdims=True)
    if arm_band is not None:
        lo, hi = arm_band
        arm = np.array([any(t in b for t in ARM) for b in bones])
        A = Ds[:, arm].sum(1, keepdims=True)
        L = np.maximum(L, np.clip((A - lo) / max(hi - lo, 1e-9), 0, 1))
    H = L * Ds + (1 - L) * Df
    if soft_arm is not None:
        # the inverse (weights2): the trunk and the legs keep the default weights; only a vertex the default binds to an
        # arm (ramped lo..hi of its arm share) takes the soft weights, and only where it does not follow the legs
        lo, hi = soft_arm
        arm = np.array([any(t in b for t in ARM) for b in bones])
        A = Ds[:, arm].sum(1, keepdims=True)
        S = np.clip((A - lo) / max(hi - lo, 1e-9), 0, 1) * (1 - np.minimum(L, 1.0))
        H = (1 - S) * Ds + S * Df
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
    ap.add_argument("--arm-band", default="", help="lo,hi: vertices whose default arm share is lo..hi keep the default weights "
                                                   "(ramped); empty: off (the shipped hybrid)")
    ap.add_argument("--soft-arm", default="", help="lo,hi: the inverse of --arm-band - only vertices the default binds to an "
                                                   "arm (ramped) take the soft weights; the trunk keeps the default (weights2)")
    a = ap.parse_args()
    band = tuple(float(x) for x in a.arm_band.split(",")) if a.arm_band else None
    soft_arm = tuple(float(x) for x in a.soft_arm.split(",")) if a.soft_arm else None
    bones, idx, w, L = blend(np.load(a.sharp, allow_pickle=True), np.load(a.soft, allow_pickle=True),
                             keep=tuple(x for x in a.keep.split(",") if x), arm_band=band, soft_arm=soft_arm)
    np.savez_compressed(a.out, bones=bones, index=idx, weight=w)
    print(f"[weights] hybrid: {int((L > 0.5).sum()):,} of {len(L):,} vertices follow the legs; {a.out}", flush=True)
