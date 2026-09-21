"""Human locomotion curves for the CharForge rig.

The first version of the animation stage keyframed a handful of bones by eye, and watching it
back is what exposed the problem: in the walk and run clips it never touched the shoulder or
elbow bones at all, so the character scissored its legs with both arms locked out in the bind
A-pose. Nothing about that reads as human, and it also means the rig was never really tested -
a deformation check only exercises the joints something actually moves.

So this replaces eyeballed keys with sampled gait curves. Each joint angle is a function of
cycle phase, tabulated at 10% intervals across the stride the way clinical gait data is
published (Winter, *Biomechanics and Motor Control of Human Movement*), and interpolated with a
periodic cubic so the loop closes smoothly. The numbers below are shaped to that literature
rather than measured here: sagittal hip travelling roughly +25 deg flexion at heel strike to
-20 deg extension at toe-off, knee showing the characteristic double bump (a small stance-phase
flexion near 15 deg, then a ~60 deg swing-phase peak), ankle plantarflexing through push-off.

The parts that make it read as human rather than as a puppet, and that the first version lacked:

  arm swing        the shoulders counter-swing against the legs, contralaterally - left arm
                   forward with right leg. This is the single biggest tell.
  pelvis bob       the pelvis rises and falls twice per stride, peaking at each mid-stance
  lateral sway     and shifts side to side once per stride, over the loaded leg
  trunk counter-   the shoulder girdle rotates opposite to the pelvis, which is what stops the
  rotation         upper body looking welded to the hips
  foot planting    ankle angles are phased so the stance foot stays flat rather than skating

Angles are degrees in the bone's local frame. The axes were *measured* on the rig rather than
assumed (charforge/blender/probe_axes.py): rotating +40 deg about each local axis and watching
where the child joint lands. For the shoulder, local X moves the elbow fore-aft (dY 0.32) while
local Z abducts it sideways (dX -0.23). The first pass drove the shoulders on Z, which is why
the character walked with its arms spread out in the bind pose instead of swinging them - the
arms were being raised sideways, not swung. Hips were already on X and were correct.

The bind pose is a wide A-pose, so the arms also need adducting down to a hanging position
before any swing reads properly; ARM_DOWN does that, with the sign flipped per side.
"""
from __future__ import annotations

import math

# Sampled at 0,10,...,100% of the gait cycle; index 10 repeats index 0 so the loop is closed.
# Cycle starts at right heel strike.
WALK = {
    "hip_flex":   [25, 20, 12, 4, -4, -12, -18, -8, 8, 20, 25],
    "knee_flex":  [4, 15, 18, 10, 3, 2, 12, 40, 60, 40, 4],
    "ankle_flex": [0, -6, -2, 2, 6, 8, -12, -8, -2, 0, 0],
    "shoulder":   [-20, -16, -10, -3, 4, 11, 16, 10, -2, -13, -20],
    "elbow":      [22, 20, 18, 16, 16, 18, 22, 28, 30, 27, 22],
    # whole-body, once or twice per stride
    "pelvis_rise": [0.0, 0.7, 1.0, 0.7, 0.0, 0.0, 0.7, 1.0, 0.7, 0.0, 0.0],
    "pelvis_sway": [0.0, 0.6, 1.0, 0.6, 0.0, -0.6, -1.0, -0.6, 0.0, 0.5, 0.0],
    "pelvis_yaw":  [5, 4, 2, 0, -2, -4, -5, -4, -2, 2, 5],
}

RUN = {
    "hip_flex":   [35, 22, 8, -6, -18, -25, -10, 18, 38, 42, 35],
    "knee_flex":  [20, 32, 28, 16, 12, 35, 85, 105, 80, 40, 20],
    "ankle_flex": [-4, -10, -4, 4, 10, -16, -12, -6, -2, -2, -4],
    "shoulder":   [-38, -30, -18, -4, 10, 24, 34, 28, 10, -18, -38],
    "elbow":      [78, 82, 86, 88, 86, 82, 78, 74, 72, 75, 78],
    "pelvis_rise": [0.2, 0.8, 1.0, 0.6, 0.0, 0.2, 0.8, 1.0, 0.6, 0.0, 0.2],
    "pelvis_sway": [0.0, 0.4, 0.7, 0.4, 0.0, -0.4, -0.7, -0.4, 0.0, 0.3, 0.0],
    "pelvis_yaw":  [8, 6, 3, 0, -3, -6, -8, -6, -3, 3, 8],
}


def sample(curve, phase):
    """Catmull-Rom interpolation of a closed 11-point curve at phase in [0,1)."""
    n = len(curve) - 1                       # 10 distinct samples, last repeats first
    x = (phase % 1.0) * n
    i = int(math.floor(x))
    f = x - i
    p0 = curve[(i - 1) % n]
    p1 = curve[i % n]
    p2 = curve[(i + 1) % n]
    p3 = curve[(i + 2) % n]
    return 0.5 * ((2 * p1) + (-p0 + p2) * f
                  + (2 * p0 - 5 * p1 + 4 * p2 - p3) * f * f
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * f * f * f)


ARM_DOWN = 45.0     # degrees of adduction that bring the bind A-pose arms down to hanging


def locomotion_pose(curves, phase, *, lean=0.0, bob_cm=3.0, sway_cm=2.0, scale=1.0,
                    arm_down=ARM_DOWN):
    """Joint angles (degrees) and a root offset (metres) at one instant of the cycle.

    The left limb runs half a cycle behind the right, and each arm runs opposite to the leg on
    its own side - that contralateral relationship is what the eye reads as walking.
    """
    off = 0.5
    pose = {
        "right_hip":   (sample(curves["hip_flex"], phase), 0, 0),
        "left_hip":    (sample(curves["hip_flex"], phase + off), 0, 0),
        "right_knee":  (-sample(curves["knee_flex"], phase), 0, 0),
        "left_knee":   (-sample(curves["knee_flex"], phase + off), 0, 0),
        "right_ankle": (sample(curves["ankle_flex"], phase), 0, 0),
        "left_ankle":  (sample(curves["ankle_flex"], phase + off), 0, 0),
        # Arms: swing fore-aft on local X, opposite to the leg on the same side, on top of a
        # constant adduction (local Z, sign per side) that takes them out of the A-pose.
        # +X swings the arm backward, so the sign is inverted against the leg it opposes.
        "right_shoulder": (-sample(curves["shoulder"], phase), 0, -arm_down),
        "left_shoulder":  (sample(curves["shoulder"], phase), 0, arm_down),
        "right_elbow":    (-sample(curves["elbow"], phase), 0, 0),
        "left_elbow":     (-sample(curves["elbow"], phase + off), 0, 0),
        # trunk: shoulder girdle counter-rotates against the pelvis
        "pelvis":  (lean, sample(curves["pelvis_yaw"], phase), 0),
        "spine2":  (lean * 0.3, -sample(curves["pelvis_yaw"], phase) * 0.8, 0),
        "neck":    (-lean * 0.5, 0, 0),
    }
    root = (
        sample(curves["pelvis_sway"], phase) * sway_cm * 0.01 * scale,
        0.0,
        sample(curves["pelvis_rise"], phase) * bob_cm * 0.01 * scale,
    )
    return pose, root
