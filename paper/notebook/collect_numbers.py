"""The lab book's numbers, read from the characters' logs after the last rebuild -> numbers.json."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CF = HERE.parents[1]                        # the repository
N = json.load(open(HERE / "numbers.json"))
# the overlap counts before the repair, measured on the shipped meshes the same way (Blender's select_overlap)
BEFORE = {"pip": 1529, "aoi": 1012, "mara": 1059, "juno3": 224, "wren": 1025, "vex": 2602}


def log(name, stage):
    p = CF / "work" / name / "logs" / f"{stage}.log"
    return p.read_text(errors="replace") if p.exists() else ""


after = {}
for n in ("pip", "aoi", "mara", "juno3", "rowan", "wren", "vex", "knight", "kaito", "bo"):
    m = re.search(r"UVs: (\d+) faces overlapping after the unwrap, (\d+) after unwrapping those again(?: \((\d+) on the mesh that ships\))?",
                  log(n, "retopo"))
    if m:
        after[n] = int(m.group(3) or m.group(2))
print("overlaps after:", after)
old = {k: v for k, v in BEFORE.items() if k in after}
N["uv"] = {"pip": [BEFORE["pip"], after.get("pip", 25)], "vex": [BEFORE["vex"], after.get("vex", 46)],
           "min_before": min(old.values()), "max_before": max(old.values()),
           "min_after": min(after[k] for k in old), "max_after": max(after[k] for k in old)}

rig = log("pip", "rig")
lg = re.findall(r"\[rig\] (left|right) arm: its own surface below the armpit [\d,]+ vertices; ([\d,]+) beyond it let go", rig)
d = {s: int(v.replace(",", "")) for s, v in lg}
if d:
    N["pip_rules"] = {"left_let_go": d.get("left", 0), "right_let_go": d.get("right", 0)}
for n in ("juno3", "pip"):
    m = re.search(r"\[rig\] hem: .*?: ([\d,]+) vertices near the hips let go", log(n, "rig"))
    if m:
        N.setdefault("hem", {})[n] = int(m.group(1).replace(",", ""))

joined = []
for n in ("pip", "aoi", "mara", "juno3", "rowan", "wren", "vex", "knight", "kaito", "bo"):
    p = CF / "work" / n / "free_arms.json"
    if p.exists():
        sides = json.load(open(p)).get("sides", {})
        k = sum(int(v.get("still_joined", 0) or 0) for v in sides.values())
        if k:
            joined.append(n)
names = {"juno3": "Juno", "knight": "Knight", "kaito": "Kaito", "bo": "Bo"}
if joined:
    who = ", ".join(names.get(n, n.capitalize()) for n in joined)
    N["joins_left"] = (f"The cut starts a quarter of the way down the upper arm and is checked slab by slab; a join still "
                       f"survives somewhere on {who}, and there the rig keeps that side of the garment with the arm as before.")
else:
    N["joins_left"] = "None survived the last rebuild's check - kept here while it is proved on more characters."

# the four moves from video as they score now (tools/motion_fidelity.py): the mean and, since a few frames the
# pose model misreads can swing it, the median of the hand error
import numpy as np
moves = {}
for n, mv in (("mara", "punch_combo"), ("mara", "roundhouse_kick"), ("aoi", "spell_cast"), ("pip", "victory_cheer")):
    f = CF / "work" / n / "motion_videos" / f"{mv}_fidelity.json"
    if not f.exists():
        continue
    d = json.load(open(f))
    e = np.array([[np.nan if x is None else x for x in r] for r in d.get("per_frame_hands", {}).get("error", [])], float)
    moves[f"{n}/{mv}"] = {"joints": d["joint_error_torso"], "iou": d["silhouette_iou"], "face": d["face_error_torso"],
                         "hands": d.get("hand_error_palm"), "facing": d.get("palm_facing_agree"),
                         "pointing": d.get("hand_direction_deg"),
                         "hands_median": d.get("hand_error_palm_median",
                                               round(float(np.nanmedian(e)), 4) if e.size and np.isfinite(e).any() else None)}
N["moves_now"] = moves
# Aoi's elbow in her spell cast: 176 deg on the rig that shipped before (measured with the fit then), and what the
# refine stage reports now for the clip it was given (the new fit) and for its own result
rj = CF / "work" / "aoi" / "refine" / "spell_cast_corr.json"
if rj.exists():
    r = json.load(open(rj))
    if "elbow_max_deg_refined" in r:
        N["elbow"] = {"before": 176, "fit": round(r["elbow_max_deg_clip"]), "after": round(r["elbow_max_deg_refined"]),
                      "frames_fit": r["elbow_frames_past_150_clip"], "frames_after": r["elbow_frames_past_150_refined"]}
print("moves now:", json.dumps(moves))

for c in N["cast"]:
    n = c["name"].lower()
    man = CF / "out" / n / f"{n}.json"
    if man.exists():
        m = json.load(open(man))
        c["height"] = f"{m['height_m']:.2f} m"
        c["tris"] = f"{m['triangles']:,}"
json.dump(N, open(HERE / "numbers.json", "w"), indent=1)
print(json.dumps({k: N[k] for k in ("uv", "pip_rules", "hem", "joins_left")}, indent=1))
