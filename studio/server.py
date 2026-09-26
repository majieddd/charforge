"""CharForge Studio: make characters from a browser, on this Mac.

    python charforge.py studio                 # then open http://localhost:8830
    python charforge.py studio --port 8840 --no-browser

Everything CharForge does from the command line, with a face on it: a character from a prompt or
an image, moves from video, a stage run again, the result turned round in 3D and played through its
clips. Nothing leaves the machine - the server listens on 127.0.0.1 only and runs the same
`charforge.py make` / `move` commands you would type, one at a time (two would fight over the GPU),
each with its log kept under work/_studio/. Jobs survive a restart of the studio as a list; a job
that was running when it stopped is marked interrupted - start it again and the pipeline resumes
from the last stage it finished.

Only the standard library: no new install.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
HOME = ROOT / "work" / "_studio"
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,31}$")
STAGE_RE = re.compile(r"^\s*\[\s*(\d+)/(\d+)\]\s+(\w+)\s*(.*)$")
DUR_RE = re.compile(r"^\s{6,}(\d+)s\s*$")                        # the time a stage took, printed as it ends
# first guesses, in seconds, until jobs on this Mac have timed each stage themselves
FIRST_GUESS = {"reference": 90, "generate": 480, "multiview": 540, "views": 40, "parts": 60, "skeleton": 45,
               "solidify": 60, "joints": 20, "hands": 30, "retopo": 70, "labels": 2, "texclean": 4, "texture": 60,
               "weights": 10, "rig": 20, "springs": 5, "frame": 5, "tpose": 10, "face": 20, "animate": 150,
               "refine": 60, "package": 40, "web": 20}
sys.path.insert(0, str(ROOT))
import charforge  # noqa: E402  (the stage list and styles; importing runs nothing)

STAGES = [{"name": n, "produces": p, "why": w} for n, p, w in charforge.STAGES]
STYLES = sorted(charforge.STYLES)


def stage_path(name, produces):
    if produces == "PACKAGE":
        return ROOT / "out" / name / f"{name}.glb"
    if produces == "WEB":
        return ROOT / "out" / name / f"{name}_web.glb"
    return ROOT / "work" / name / produces


def read_json(p, default=None):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return default


def url(p: Path):
    """A file under the repository, as the studio serves it."""
    return "/files/" + str(p.relative_to(ROOT)) + f"?v={int(p.stat().st_mtime)}"


# ---- characters --------------------------------------------------------------------------------
def characters():
    out = []
    for d in sorted((ROOT / "work").iterdir()):
        if not d.is_dir() or d.name.startswith("_") or not NAME_RE.match(d.name):
            continue
        if not ((d / "style.json").exists() or (d / "reference.png").exists()):
            continue
        out.append(summary(d.name))
    return sorted(out, key=lambda c: -c["updated"])


def summary(name):
    w, o = ROOT / "work" / name, ROOT / "out" / name
    st = read_json(w / "style.json", {}) or {}
    done = [s["name"] for s in STAGES if stage_path(name, s["produces"]).exists()]
    thumb = o / f"{name}_thumb.png"
    thumb = thumb if thumb.exists() else (w / "reference.png" if (w / "reference.png").exists() else None)
    moves = list((read_json(w / "extra_clips.json", {}) or {}).keys())
    mtimes = [p.stat().st_mtime for p in (w, o) if p.exists()]
    return {"name": name, "style": st.get("style", "realistic"), "prompt": st.get("prompt"),
            # the height given, else the one the description implied (charforge.height_of's order)
            "height": st.get("height") or st.get("height_auto"), "image_model": st.get("image_model"),
            "stages_done": len(done), "stages": len(STAGES), "done": done,
            "complete": (o / f"{name}_web.glb").exists(), "thumb": url(thumb) if thumb else None,
            "moves": moves, "updated": max(mtimes) if mtimes else 0}


def detail(name):
    w, o = ROOT / "work" / name, ROOT / "out" / name
    c = summary(name)
    c["manifest"] = read_json(o / f"{name}.json")
    for key, p in (("web_glb", o / f"{name}_web.glb"), ("glb", o / f"{name}.glb"), ("fbx", o / f"{name}.fbx"),
                   ("reference", w / "reference.png")):
        c[key] = url(p) if p.exists() else None
    extra = read_json(w / "extra_clips.json", {}) or {}
    mv = []
    for m, spec in extra.items():
        src = Path(spec.get("from_video") or "")
        performer = src.parent.parent.name if src.suffix == ".mp4" else name
        vids = ROOT / "work" / performer / "motion_videos"
        mine = w / "motion_videos"
        fid = read_json(mine / f"{m}_fidelity.json") or {}
        item = {"name": m, "performer": performer,
                "compare": url(mine / f"{m}_compare.mp4") if (mine / f"{m}_compare.mp4").exists() else None,
                "video": url(vids / f"{m}.mp4") if (vids / f"{m}.mp4").exists() else None,
                "hands_sheet": url(mine / f"{m}_hands.png") if (mine / f"{m}_hands.png").exists() else None,
                "fidelity": {k: fid.get(k) for k in ("joint_error_torso", "silhouette_iou", "face_error_torso",
                                                     "hand_error_palm", "palm_facing_agree", "hand_direction_deg")}
                if fid else None}
        mv.append(item)
    c["move_details"] = mv
    pz = w / "qa" / "poses"
    c["poses"] = [url(p) for p in sorted(pz.glob("*.png"), key=lambda p: (p.name.rsplit("_", 1)[0], p.name))] \
        if pz.exists() else []
    c["poses_at"] = max((p.stat().st_mtime for p in pz.glob("*.png")), default=None) if pz.exists() else None
    reel = w / "qa" / "reel.mp4"
    c["reel"] = url(reel) if reel.exists() else None
    # the same clip from several cameras at once (tools/make_angles.py), newest first
    ang = []
    for f in sorted((w / "qa").glob("angles_*.mp4"), key=lambda p: -p.stat().st_mtime) if (w / "qa").exists() else []:
        meta = read_json(f.with_suffix(".json"), {}) or {}
        ang.append({"url": url(f), "clip": meta.get("clip") or f.stem[7:], "video": bool(meta.get("video")),
                    "worst_elbow": meta.get("worst_elbow_deg"), "worst_knee": meta.get("worst_knee_deg"),
                    "at": f.stat().st_mtime})
    c["angles"] = ang
    man_ = read_json(ROOT / "out" / name / f"{name}.json", {}) or {}
    c["clip_names"] = [x.get("name") for x in man_.get("clips", []) if x.get("name")]
    c["video_moves"] = [m for m, spec in extra.items() if str(spec.get("from_video", "")).endswith(".mp4")]
    c["logs"] = sorted(p.name for p in (w / "logs").glob("*.log")) if (w / "logs").exists() else []
    c["stage_list"] = [{**s, "done": s["name"] in c["done"]} for s in STAGES]
    return c


# ---- jobs --------------------------------------------------------------------------------------
def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True


def find_pid(cmd):
    """The process running this command line, if one is (a job started before the Studio last restarted)."""
    want = " ".join(str(c) for c in cmd)
    try:
        ps = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in ps.splitlines():
        pid_, _, command = line.strip().partition(" ")
        if command.strip() == want:
            return int(pid_)
    return None


def read_log_stages(path):
    """The stages a job log shows, in order, each with the seconds it took (None while it runs)."""
    out = []
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = STAGE_RE.match(line)
        if m:
            out.append([m.group(3), int(m.group(1)), int(m.group(2)), None])
            continue
        d = DUR_RE.match(line)
        if d and out and out[-1][3] is None:
            out[-1][3] = int(d.group(1))
    return out


class Clock:
    """How long each stage takes on this Mac: the median of what the job logs recorded - the character's
    own where it has been through that stage before (Pip's refine stage takes four to eight minutes, the
    characters with no moves from video none at all), everyone's otherwise."""

    def __init__(self):
        self.seen = {}
        self.lock = threading.Lock()

    def medians(self, name=None):
        with self.lock:
            for f in HOME.glob("*.log"):
                try:
                    mt = f.stat().st_mtime
                except OSError:
                    continue
                if self.seen.get(f.name, (None,))[0] != mt:
                    who = None
                    try:
                        with open(f, errors="replace") as fh:
                            m = re.search(r"--name (\S+)", fh.readline())
                            who = m.group(1) if m else None
                    except OSError:
                        pass
                    self.seen[f.name] = (mt, who, [(st, sec) for st, _, _, sec in read_log_stages(f) if sec is not None])
            per, mine = {}, {}
            for _, who, durs in self.seen.values():
                for st, sec in durs:
                    per.setdefault(st, []).append(sec)
                    if name and who == name:
                        mine.setdefault(st, []).append(sec)
        med = dict(FIRST_GUESS)
        med.update({st: float(statistics.median(v)) for st, v in per.items()})
        med.update({st: float(statistics.median(v)) for st, v in mine.items()})
        return med


CLOCK = Clock()


class Jobs:
    """One job at a time, in order; the list kept in work/_studio/jobs.json."""

    def __init__(self):
        HOME.mkdir(parents=True, exist_ok=True)
        self.path = HOME / "jobs.json"
        self.lock = threading.Lock()
        self.jobs = read_json(self.path, []) or []
        self.procs = {}
        self.adopted = {}                    # jobs still running from before a restart: id -> pid
        for j in self.jobs:
            if j["status"] == "running":
                # the Studio restarted under a job: its process was started in a session of its own and goes
                # on; wait for it rather than start the next job beside it
                pid = j.get("pid") or find_pid(j["cmd"])
                if pid and alive(pid):
                    j["pid"] = pid
                    self.adopted[j["id"]] = pid
                else:
                    j["status"] = "interrupted"
        self.wake = threading.Event()
        self.save()
        threading.Thread(target=self.worker, daemon=True).start()

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        json.dump(self.jobs, open(tmp, "w"), indent=1)
        tmp.replace(self.path)

    def add(self, kind, name, cmd, title):
        j = {"id": uuid.uuid4().hex[:10], "kind": kind, "name": name, "title": title, "cmd": [str(c) for c in cmd],
             "status": "queued", "created": time.time(), "started": None, "ended": None, "code": None}
        j["log"] = str(HOME / f"{j['id']}.log")
        with self.lock:
            self.jobs.append(j)
            self.save()
        self.wake.set()
        return j

    def get(self, jid):
        return next((j for j in self.jobs if j["id"] == jid), None)

    def worker(self):
        while True:
            for jid, pid in list(self.adopted.items()):
                if alive(pid):
                    continue
                with self.lock:                                   # it ended; its log says how
                    j = self.get(jid)
                    self.adopted.pop(jid, None)
                    if j and j["status"] == "running":
                        tail = Path(j["log"]).read_text(errors="replace")[-3000:] if Path(j["log"]).exists() else ""
                        j["status"] = "done" if "\ndone in " in tail else "failed"
                        j["ended"] = time.time()
                        self.save()
            if self.adopted:
                time.sleep(2)
                continue
            with self.lock:
                j = next((j for j in self.jobs if j["status"] == "queued"), None)
                if j:
                    j["status"], j["started"] = "running", time.time()
                    self.save()
            if not j:
                self.wake.wait(2)
                self.wake.clear()
                continue
            with open(j["log"], "ab") as log:
                log.write(f"$ {' '.join(j['cmd'])}\n".encode())
                log.flush()
                try:
                    p = subprocess.Popen(j["cmd"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
                except OSError as e:
                    log.write(f"could not start: {e}\n".encode())
                    p = None
                if p:
                    self.procs[j["id"]] = p
                    with self.lock:
                        j["pid"] = p.pid
                        self.save()
                    code = p.wait()
                    self.procs.pop(j["id"], None)
                else:
                    code = -1
            with self.lock:
                if j["status"] == "running":
                    j["status"] = "done" if code == 0 else "failed"
                j["code"], j["ended"] = code, time.time()
                self.save()

    def clear(self):
        """Take the finished jobs off the list; their logs stay in work/_studio/."""
        with self.lock:
            keep = [j for j in self.jobs if j["status"] in ("running", "queued")]
            n = len(self.jobs) - len(keep)
            self.jobs = keep
            self.save()
        return n

    def cancel(self, jid):
        with self.lock:
            j = self.get(jid)
            if not j:
                return False
            if j["status"] == "queued":
                j["status"] = "cancelled"
                self.save()
                return True
            if j["status"] != "running":
                return False
            j["status"] = "cancelled"
            self.save()
        p = self.procs.get(jid)
        if p is None and jid in self.adopted:                     # a job from before a restart
            try:
                os.killpg(self.adopted[jid], signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        if p:
            try:
                os.killpg(p.pid, signal.SIGTERM)                  # Blender and friends go with it
            except ProcessLookupError:
                pass

            def hard():
                time.sleep(10)
                if p.poll() is None:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            threading.Thread(target=hard, daemon=True).start()
        return True

    def stages_of(self, j):
        """The stages a job will run, or None where it is not a run of the pipeline (moves)."""
        names = [st["name"] for st in STAGES]
        cmd = j.get("cmd") or []
        if j["kind"] == "rerun" and "--from" in cmd:
            st = cmd[cmd.index("--from") + 1]
            return names[names.index(st):] if st in names else names
        if j["kind"] == "make":
            return names
        return None

    def typical(self, kind):
        """How long a finished job of this kind took, the median of the ones on record."""
        ds = [j["ended"] - j["started"] for j in self.jobs
              if j["kind"] == kind and j["status"] == "done" and j.get("started") and j.get("ended")]
        return float(statistics.median(ds)) if ds else 360.0

    def view(self, j, med=None):
        """The job for the page: where it has got to, from its log, and how long it should take."""
        v = {k: j[k] for k in ("id", "kind", "name", "title", "status", "created", "started", "ended", "code")}
        med = CLOCK.medians(j["name"])
        seen = read_log_stages(j["log"]) if j.get("started") else []
        last = ""
        try:
            with open(j["log"], "rb") as f:
                f.seek(max(0, os.path.getsize(j["log"]) - 4000))
                for line in f.read().decode("utf-8", "replace").splitlines():
                    if line.strip():
                        last = line.strip()
        except OSError:
            pass
        stage, i, n = (seen[-1][0], seen[-1][1], seen[-1][2]) if seen else (None, 0, len(STAGES))
        v.update(stage=stage, stage_i=i, stage_n=n, last=last[-200:])
        now = j["ended"] or time.time()
        v["elapsed"] = (now - j["started"]) if j["started"] else 0
        plan = self.stages_of(j)
        if plan is None:
            total = self.typical(j["kind"])
            left = max(0.0, total - v["elapsed"]) if j["status"] == "running" else total
        else:
            total = sum(med.get(st, 30.0) for st in plan)
            left = total
            if j["status"] == "running" and seen:
                done_s = sum(sec for _, _, _, sec in seen if sec is not None)
                cur = seen[-1]
                in_stage = max(0.0, v["elapsed"] - done_s) if cur[3] is None else 0.0
                after = plan[plan.index(cur[0]) + 1:] if cur[0] in plan else []
                left = max(0.0, med.get(cur[0], 30.0) - in_stage) * (cur[3] is None) + sum(med.get(st, 30.0) for st in after)
        if j["status"] not in ("running", "queued"):
            left = 0.0
        v["est_total"], v["est_left"] = round(total), round(left)
        return v

    def views(self):
        """Every job for the page, and when each queued one should start."""
        med = CLOCK.medians()
        vs = [self.view(j, med) for j in self.jobs]
        t = sum(v["est_left"] for v in vs if v["status"] == "running")
        for v in sorted((v for v in vs if v["status"] == "queued"), key=lambda v: v["created"]):
            v["starts_in"] = round(t)
            t += v["est_total"]
        return list(reversed(vs))

_PLAY = {"key": None, "html": None}


def playground_page():
    """The walkable playground with every character on this Mac that has a web build - not only the
    published roster - each streamed from its package. Built here from tools/build_site.py's own pieces,
    again only when a package or the template changes."""
    tpl = ROOT / "web" / "playground.html"
    pkgs = sorted(p for p in (ROOT / "out").iterdir() if p.is_dir()) if (ROOT / "out").exists() else []
    ready = [p for p in pkgs if all((p / f"{p.name}{x}").exists() for x in (".json", "_web.glb", "_thumb.png"))]
    key = (tpl.stat().st_mtime, tuple((p.name, (p / f"{p.name}_web.glb").stat().st_mtime) for p in ready))
    if _PLAY["key"] == key:
        return _PLAY["html"]
    sys.path.insert(0, str(ROOT / "tools"))
    import build_site as bs                                           # thumbnails, manifest subset, injection
    cfg = read_json(ROOT / "web" / "roster.json", {}) or {}
    named = {c["id"]: c for c in cfg.get("characters", [])}
    entries, src = [], {}
    for p in ready:
        cid = p.name
        man = read_json(p / f"{cid}.json", {}) or {}
        if not man.get("clips"):
            continue
        c = named.get(cid, {})
        style = (man.get("style") or {}).get("name", "") if isinstance(man.get("style"), dict) else str(man.get("style") or "")
        entries.append({"id": cid, "name": c.get("name", cid.capitalize()), "blurb": c.get("blurb", style),
                        "prompt": (man.get("source") or {}).get("prompt") or "", "thumb": bs.thumb_uri(p / f"{cid}_thumb.png"),
                        "download": f"/files/out/{cid}/{cid}.glb", "manifest": bs.page_manifest(man)})
        src[cid] = f"(p) => fetchGLB('/files/out/{cid}/{cid}_web.glb', p)"
    html = bs.inject(tpl.read_text(), bs.roster_block(entries, src)) if entries else \
        "<p style='font-family:sans-serif;padding:2em'>No character has a web build yet - make one first.</p>"
    _PLAY.update(key=key, html=html)
    return html


JOBS = None


# ---- the web side ------------------------------------------------------------------------------
SERVE_DIRS = ("out", "work", "docs", "web/dist", "results")
SERVE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".glb", ".gltf", ".fbx", ".mp4", ".wav", ".json", ".zip", ".html",
             ".js", ".css", ".bin", ".log", ".txt"}


class Handler(BaseHTTPRequestHandler):
    server_version = "CharForgeStudio/1"

    def log_message(self, fmt, *args):                               # quiet
        pass

    # helpers
    def send(self, code, body, ctype="application/json", headers=None):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def jbody(self):
        try:
            return json.loads(self.body() or b"{}")
        except ValueError:
            return {}

    def file(self, p: Path, download=False):
        if not p.exists() or not p.is_file():
            return self.send(404, {"error": "not found"})
        size = p.stat().st_size
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        if p.suffix == ".glb":
            ctype = "model/gltf-binary"
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng and rng.startswith("bytes="):                          # video seeking
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else 0
            end = int(b) if b else size - 1
        length = end - start + 1
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{p.name}"')
        self.end_headers()
        with open(p, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    # routes
    def do_GET(self):
        u = urlparse(self.path)
        path, q = unquote(u.path), parse_qs(u.query)
        if path == "/" or path == "/index.html":
            return self.file(STATIC / "index.html")
        if path in ("/play", "/play/"):
            try:
                return self.send(200, playground_page(), ctype="text/html; charset=utf-8")
            except Exception as e:                                    # a broken package should not take the Studio down
                return self.send(500, f"<pre>the playground could not be built: {e}</pre>", ctype="text/html")
        if path.startswith("/static/"):
            p = (STATIC / path[len("/static/"):]).resolve()
            return self.file(p) if str(p).startswith(str(STATIC)) else self.send(403, {"error": "no"})
        if path.startswith("/files/"):
            rel = path[len("/files/"):]
            p = (ROOT / rel).resolve()
            ok = any(str(p).startswith(str((ROOT / d).resolve()) + os.sep) for d in SERVE_DIRS) and p.suffix.lower() in SERVE_EXT
            if not ok or ".env" in p.parts:
                return self.send(403, {"error": "not served"})
            return self.file(p, download="download" in q)
        if path == "/api/characters":
            return self.send(200, characters())
        m = re.match(r"^/api/character/([a-z0-9_]+)$", path)
        if m:
            if not (ROOT / "work" / m.group(1)).is_dir():
                return self.send(404, {"error": "no such character"})
            return self.send(200, detail(m.group(1)))
        m = re.match(r"^/api/character/([a-z0-9_]+)/log/([\w.-]+\.log)$", path)
        if m:
            p = ROOT / "work" / m.group(1) / "logs" / m.group(2)
            return self.send(200, p.read_text(errors="replace")[-40000:] if p.exists() else "", "text/plain; charset=utf-8")
        if path == "/api/meta":
            moves = read_json(ROOT / "animations" / "motion_prompts.json", {}) or {}
            return self.send(200, {"stages": STAGES, "styles": STYLES, "moves": moves.get("moves", {}),
                                   "root": str(ROOT)})
        if path == "/api/jobs":
            return self.send(200, JOBS.views())
        m = re.match(r"^/api/jobs/(\w+)/log$", path)
        if m:
            j = JOBS.get(m.group(1))
            if not j:
                return self.send(404, {"error": "no such job"})
            off = int((q.get("offset") or ["0"])[0])
            try:
                with open(j["log"], "rb") as f:
                    size = os.path.getsize(j["log"])
                    if off > size or off < 0:
                        off = 0
                    f.seek(max(off, size - 200000))
                    data = f.read()
                return self.send(200, {"offset": size, "text": data.decode("utf-8", "replace")})
            except OSError:
                return self.send(200, {"offset": 0, "text": ""})
        if path == "/api/env":
            return self.send(200, env_check(refresh="refresh" in q))
        if path == "/api/settings":
            sys.path.insert(0, str(ROOT / "pipeline"))
            import describe  # noqa: E402
            models, url = describe.config()
            st = read_json(ROOT / "settings.json", {}) or {}
            return self.send(200, {"describe_model": st.get("describe_model", ""), "describe_url": st.get("describe_url", ""),
                                   "using": models, "using_url": url, "default": describe.DEFAULT_MODELS,
                                   "installed": describe._installed(url), "env": bool(os.environ.get("CF_DESCRIBE_MODEL"))})
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/settings":
            # which local model writes prompts out (pipeline/describe.py reads this at each call)
            b = self.jbody()
            st = read_json(ROOT / "settings.json", {}) or {}
            for k in ("describe_model", "describe_url"):
                if k in b:
                    v = " ".join(str(b[k] or "").split())[:200]
                    if k == "describe_url" and v and not re.match(r"^https?://(127\.0\.0\.1|localhost)(:\d+)?(/.*)?$", v):
                        return self.send(400, {"error": "a server on this Mac: http://127.0.0.1:<port>/v1"})
                    if v:
                        st[k] = v
                    else:
                        st.pop(k, None)
            json.dump(st, open(ROOT / "settings.json", "w"), indent=1)
            return self.send(200, st)
        if path == "/api/describe":
            # a short prompt written out - age and build, skin and hair, each garment with its colour - for
            # the page to show and let the person edit before anything is drawn (pipeline/describe.py)
            b = self.jbody()
            prompt = " ".join((b.get("prompt") or "").split())[:600]
            if not prompt:
                return self.send(400, {"error": "describe the character first"})
            sys.path.insert(0, str(ROOT / "pipeline"))
            from describe import describe  # noqa: E402
            try:
                d = describe(prompt, b.get("style") if b.get("style") in STYLES else "realistic")
            except Exception as e:                       # noqa: BLE001
                return self.send(500, {"error": f"could not write it out: {e}"})
            return self.send(200, {k: d[k] for k in ("description", "age", "height_m", "engine")})
        if path == "/api/upload":
            data = self.body()
            if len(data) > 40 << 20:
                return self.send(413, {"error": "image over 40 MB"})
            ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}.get(self.headers.get("Content-Type", ""), ".png")
            up = HOME / "uploads"
            up.mkdir(parents=True, exist_ok=True)
            p = up / f"{uuid.uuid4().hex[:10]}{ext}"
            p.write_bytes(data)
            return self.send(200, {"upload": p.name})
        if path == "/api/jobs":
            b = self.jbody()
            try:
                j = make_job(b)
            except ValueError as e:
                return self.send(400, {"error": str(e)})
            return self.send(200, JOBS.view(j))
        m = re.match(r"^/api/jobs/(\w+)/cancel$", path)
        if m:
            return self.send(200, {"ok": JOBS.cancel(m.group(1))})
        if path == "/api/jobs/clear":
            return self.send(200, {"cleared": JOBS.clear()})
        m = re.match(r"^/api/character/([a-z0-9_]+)/reveal$", path)
        if m and sys.platform == "darwin":
            p = ROOT / "out" / m.group(1)
            p = p if p.exists() else ROOT / "work" / m.group(1)
            subprocess.Popen(["open", str(p)])
            return self.send(200, {"ok": True})
        return self.send(404, {"error": "not found"})


def make_job(b):
    kind = b.get("kind")
    py, cf = sys.executable, ROOT / "charforge.py"
    name = (b.get("name") or "").strip().lower()
    if not NAME_RE.match(name):
        raise ValueError("a name is lowercase letters, digits and _, up to 32, starting with a letter or digit")
    if kind == "make":
        cmd = [py, "-u", cf, "make", "--name", name]
        exists = (ROOT / "work" / name).exists()
        if exists and (ROOT / "work" / name / "reference.png").exists() and (b.get("image") or (b.get("prompt") or "").strip()):
            # every stage of a taken name is cached: the new prompt would be ignored (and it overwrote the
            # recorded one) - "a young knight in steel armor" named knight gave back the existing Knight
            raise ValueError(f"there is already a character named {name} - pick another name "
                             f"(or run {name} again from a stage on its own page)")
        if b.get("image"):
            img = HOME / "uploads" / Path(b["image"]).name
            if not img.exists():
                raise ValueError("the uploaded image is gone - upload it again")
            cmd += ["--image", img]
        elif (b.get("prompt") or "").strip():
            cmd += ["--prompt", b["prompt"].strip()[:600]]
            if b.get("literal"):
                cmd += ["--literal"]
            elif (b.get("description") or "").strip():
                cmd += ["--description", " ".join(b["description"].split())[:1200]]
        elif not exists:
            raise ValueError("give a prompt or an image")
        if b.get("style") in STYLES:
            cmd += ["--style", b["style"]]
        if b.get("height"):                   # none: the height the description implies
            h = float(b["height"])
            if not 0.5 <= h <= 3.0:
                raise ValueError("height is in metres, 0.5 to 3")
            cmd += ["--height", f"{h:.2f}"]
        if b.get("image_model") in ("qwen21", "krea2"):
            cmd += ["--image-model", b["image_model"]]
        if b.get("quality") in ("best", "fast"):
            cmd += ["--quality", b["quality"]]
        if b.get("seed") not in (None, ""):
            cmd += ["--seed", str(int(b["seed"]))]
        title = f"make {name}" + (f": {b['prompt'].strip()[:60]}" if b.get("prompt") else "")
    elif kind == "rerun":
        st = b.get("from")
        if st not in [s["name"] for s in STAGES]:
            raise ValueError("pick a stage to run from")
        cmd = [py, "-u", cf, "make", "--name", name, "--from", st]
        title = f"{name}: run again from {st}"
    elif kind == "poses":
        # the poses that break things - the standard clips' extremes and the character's own moves from
        # video at a third and two thirds - front-left and back-right, into work/<name>/qa/poses
        blend = ROOT / "work" / name / "final.blend"
        if not blend.exists():
            raise ValueError("no finished character to pose yet")
        man = read_json(ROOT / "out" / name / f"{name}.json", {}) or {}
        have = {c.get("name"): c.get("seconds", 0) for c in man.get("clips", [])}
        shots = [f"{c}@{t}" for c, t in (("idle", 0.5), ("walk", 0.3), ("run", 0.2), ("jump", 0.9), ("jump", 1.2),
                                         ("land", 0.5), ("crouch_walk", 0.4), ("wave", 0.5), ("fall", 1.0))
                 if c in have]
        std = {"idle", "walk", "jog", "run", "sprint", "walk_back", "jog_back", "strafe_left", "strafe_right",
               "turn_left", "turn_right", "turn_180", "crouch_idle", "crouch_walk", "jump", "fall", "land", "wave"}
        for c, sec in have.items():
            if c not in std and sec:
                shots += [f"{c}@{sec / 3:.2f}", f"{c}@{2 * sec / 3:.2f}"]
        out = ROOT / "work" / name / "qa" / "poses"
        if out.exists():
            for old in out.glob("*.png"):
                old.unlink()
        cmd = [charforge.blender_bin(), "-b", "-noaudio", "--python", ROOT / "blender" / "render_poses.py", "--",
               "--blend", blend, "--shots", ",".join(shots), "--az", "30,210", "--res", "420", "--out", out]
        title = f"{name}: render the poses"
    elif kind == "angles":
        if not (ROOT / "work" / name / "final.blend").exists():
            raise ValueError("no finished character to film yet")
        man = read_json(ROOT / "out" / name / f"{name}.json", {}) or {}
        clips = [x.get("name") for x in man.get("clips", [])]
        clip = b.get("clip")
        if clip not in clips:
            raise ValueError("pick one of the character's clips")
        views = b.get("views") or "front,left,top"
        if not re.match(r"^(front|back|left|right|top)(,(front|back|left|right|top))*$", views):
            raise ValueError("views are front, back, left, right and top")
        cmd = [py, "-u", ROOT / "tools" / "make_angles.py", "--name", name, "--clip", clip, "--views", views]
        extra = read_json(ROOT / "work" / name / "extra_clips.json", {}) or {}
        if b.get("video") and str((extra.get(clip) or {}).get("from_video", "")).endswith(".mp4"):
            cmd.append("--video")
        title = f"{name}: {clip.replace('_', ' ')} from several angles"
    elif kind == "reel":
        if not (ROOT / "work" / name / "final.blend").exists():
            raise ValueError("no finished character to film yet")
        cmd = [py, "-u", ROOT / "tools" / "make_reel.py", "--name", name]
        title = f"{name}: record a reel"
    elif kind == "move":
        moves = [m for m in (b.get("moves") or []) if re.match(r"^[a-z0-9_]+(@[a-z0-9_]+)?$", m)]
        if not moves:
            raise ValueError("pick at least one move")
        cmd = [py, "-u", cf, "move", "--name", name, "--move", ",".join(moves)]
        if b.get("performer") and NAME_RE.match(b["performer"]):
            cmd += ["--performer", b["performer"]]
        title = f"{name}: moves {', '.join(moves)}"
    else:
        raise ValueError("unknown job")
    return JOBS.add(kind, name, cmd, title)


_ENV = {"at": 0, "rows": None}


def env_check(refresh=False):
    if _ENV["rows"] is None or refresh or time.time() - _ENV["at"] > 600:
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "check_env.py")], cwd=ROOT,
                           capture_output=True, text=True, timeout=300)
        rows = []
        for line in (r.stdout + r.stderr).splitlines():
            m = re.match(r"^\[\s*(ok|MISS|warn)\s*\]\s+(.*?)(?:\s{2,}(.*))?$", line.strip())
            if m:
                rows.append({"state": m.group(1), "what": m.group(2), "detail": m.group(3) or ""})
        _ENV.update(at=time.time(), rows=rows, text=r.stdout[-8000:])
    return {"rows": _ENV["rows"], "text": _ENV.get("text", ""), "at": _ENV["at"]}


def main(port=8830, open_browser=True):
    global JOBS
    # One Studio at a time runs the queue: a second one (the launcher double-clicked while one runs) would
    # read the same list and start the next job beside the first. It opens the one that is running instead.
    HOME.mkdir(parents=True, exist_ok=True)
    lock = HOME / "server.lock"
    other = read_json(lock, {}) or {}
    if other.get("pid") and other.get("pid") != os.getpid() and alive(other["pid"]):
        url_ = f"http://localhost:{other.get('port', port)}"
        print(f"CharForge Studio is already running at {url_} - opening that one", flush=True)
        if open_browser and sys.platform == "darwin":
            subprocess.Popen(["open", url_])
        return
    json.dump({"pid": os.getpid(), "port": port}, open(lock, "w"))
    JOBS = Jobs()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CharForge Studio on http://localhost:{port}  (jobs and logs in {HOME})", flush=True)
    if open_browser and sys.platform == "darwin":
        subprocess.Popen(["open", f"http://localhost:{port}"])
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    main(a.port, not a.no_browser)
