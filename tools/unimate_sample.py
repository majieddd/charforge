"""UniMate's sampler (vendor/unimate) with a fixed-step ODE, so it runs on Apple's GPU.

    vendor/unimate/.venv/bin/python tools/unimate_sample.py --exp_dir <dir> --test_cases_json <json> \
        [--steps 50] [--method euler|midpoint] [--device mps] [UniMate's own sample.py options]

UniMate integrates its flow with dopri5 (adaptive, atol 1e-6): on MPS torchdiffeq builds its tolerances
in float64, which MPS has no support for, and on the CPU three 2 s motions took over ten minutes. A fixed
grid needs no tolerance and costs exactly `--steps` model calls (x2 with classifier-free guidance).
Everything else is UniMate's own `unimate.inference.sample.main`. Run with HF_HUB_OFFLINE=1 and HF_HOME
pointing at the local flan-t5-base (tools/unimate_moves.py does).
"""
import os
import sys

UNIMATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vendor", "unimate")
sys.path.insert(0, UNIMATE)

steps, method, device = 50, "euler", None
argv = []
it = iter(sys.argv[1:])
for x in it:
    if x == "--steps":
        steps = int(next(it))
    elif x == "--method":
        method = next(it)
    elif x == "--device":
        device = next(it)
    else:
        argv.append(x)
sys.argv = [sys.argv[0]] + argv

import torch  # noqa: E402
import tyro  # noqa: E402
from unimate.models.flow import transport as T  # noqa: E402
from unimate.inference import sample as S  # noqa: E402

_orig = T.Sampler.sample_ode


def sample_ode(self, *a, **k):
    k.update(sampling_method=method, num_steps=steps)
    return _orig(self, *a, **k)


T.Sampler.sample_ode = sample_ode

if device:
    # the saved config says cuda; the device is read from it after loading
    _from_json = S.MainConfig.from_json

    def from_json(p):
        c = _from_json(p)
        c.sampling.device = device
        return c
    S.MainConfig.from_json = staticmethod(from_json)

if __name__ == "__main__":
    print(f"[unimate] ODE: {method}, {steps} steps, device {device or 'from config'}", flush=True)
    S.main(tyro.cli(S.InferenceArgs))
