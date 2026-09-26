"""Rebuild the lab book: what CharForge does now first, the methods it replaced in an archive at the end.

    python paper/notebook/build_book.py   (reads index.before_polish.html and numbers.json, writes docs/paper/notebook.html)

The old page's sections are cut out by id and reused where they still describe the pipeline as it is;
the rest goes into the archive, collapsed. New sections: the Studio, how a character is made now,
the polish pass, the new characters, what is left.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # the repository
MEDIA = HERE.parent / "media"              # paper/media, shared with the paper
OUT = ROOT / "docs" / "paper" / "notebook.html"
old = (HERE / "index.before_polish.html").read_text()
N = json.load(open(HERE / "numbers.json"))


def block(tag_id, kind="section"):
    """The whole <section id=...>...</section> (or header) block."""
    m = re.search(rf'<{kind}[^>]*id="{tag_id}"[^>]*>', old)
    start = m.start()
    depth, i = 0, start
    pat = re.compile(rf"<(/?){kind}\b[^>]*>")
    for mm in pat.finditer(old, start):
        depth += -1 if mm.group(1) else 1
        if depth == 0:
            return old[start:mm.end()]
    raise ValueError(tag_id)


style = old[old.index("<style>"):old.index("</style>") + len("</style>")]
extra_css = """
  .facts{grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}
  /* archive */
  details.arch{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin-top:14px}
  details.arch>summary{cursor:pointer;list-style:none;padding:14px 18px;display:flex;gap:12px;align-items:baseline;flex-wrap:wrap}
  details.arch>summary::-webkit-details-marker{display:none}
  details.arch>summary::before{content:"+";font:600 16px/1 var(--mono);color:var(--accent);width:14px}
  details.arch[open]>summary::before{content:"\\2212"}
  details.arch>summary b{font:600 16px/1.3 var(--cond)}
  details.arch>summary span{font-size:13px;color:var(--muted)}
  details.arch>.inner{padding:0 18px 18px;border-top:1px solid var(--line)}
  details.arch>.inner section{padding-top:22px}
  /* studio shots and figures */
  .shots{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,460px),1fr));gap:16px}
  .shot{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:grid}
  .shot img{display:block;width:100%;height:auto;max-width:100%}
  .shot figcaption{padding:10px 14px 12px;font-size:13px;color:var(--ink2)}
  .shot figcaption b{color:var(--ink)}
  figure.wide{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px}
  figure.wide img,figure.wide video{display:block;width:100%;height:auto;border-radius:8px;max-width:100%}
  figure.wide figcaption{margin-top:10px;font-size:13px;color:var(--ink2)}
  /* polish cards */
  .fix{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:18px;align-items:start;background:var(--panel);
       border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin-top:16px}
  @media (max-width:820px){.fix{grid-template-columns:1fr}}
  .fix img{display:block;width:100%;height:auto;border-radius:8px;border:1px solid var(--line);max-width:100%}
  .fix .txt{display:grid;gap:8px;align-content:start}
  .fix .txt p{font-size:14px;color:var(--ink2)}
  .fix .num{font:700 20px/1.1 var(--cond);color:var(--ink)}
  .fix .num small{font:500 12px var(--mono);color:var(--muted)}
  .tag{font:500 10.5px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--accent)}
  /* new characters */
  .cast{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,320px),1fr));gap:16px}
  .who{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:grid;grid-template-rows:auto 1fr}
  .who video,.who img.hero{display:block;width:100%;height:auto;aspect-ratio:1/1;object-fit:cover;background:#1b2027;max-width:100%}
  .who .body{padding:12px 14px 14px;display:grid;gap:8px;align-content:start}
  .who .title{display:flex;justify-content:space-between;align-items:baseline;gap:8px}
  .who .prompt{font-size:13px;color:var(--ink2);font-style:italic}
  .who .refrow{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
  .who .refrow img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line);background:var(--panel2)}
  .who .refrow figcaption{font:11px/1.3 var(--mono);color:var(--muted);margin-top:3px}
  .who figure{margin:0}
  .fixlist{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);gap:14px;align-items:start}
  .fixlist > :first-child{grid-row:span 2}
  @media (max-width:760px){.fixlist{grid-template-columns:1fr}.fixlist > :first-child{grid-row:auto}}
  .angles{display:grid;grid-template-columns:1fr;gap:16px}
  .angles figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px;overflow:hidden}
  .angles video{display:block;width:100%;height:auto;border-radius:8px;background:#0a0c10;max-width:100%}
  .angles figcaption{margin-top:8px;font:600 14px/1.3 var(--cond);color:var(--ink);display:flex;justify-content:space-between;gap:12px}
  .angles figcaption .hint{font:400 12px/1.3 var(--sans);color:var(--muted)}
  .angles video{cursor:pointer}
  ul.left{margin:0;padding-left:18px;display:grid;gap:8px;max-width:78ch}
  ul.left li{font-size:14px;color:var(--ink2)}
  ul.left li b{color:var(--ink)}
"""
style = style.replace("</style>", extra_css + "</style>")

nav = """<nav class="top" aria-label="Sections">
  <div class="in">
    <a class="brand" href="./" style="color:var(--ink);text-decoration:none;border:0">CharForge Lab Notebook</a>
    <a href="#studio">Studio</a>
    <a href="#prompts">Few-word prompts</a>
    <a href="#pipeline">How it is made</a>
    <a href="#polish">Polish</a>
    <a href="#cast">New characters</a>
    <a href="#moves">Moves from video</a>
    <a href="#angles">Four sides</a>
    <a href="#fidelity">How close</a>
    <a href="#hands">Hands</a>
    <a href="#mac">On the Mac</a>
    <a href="#left">What is left</a>
    <a href="#archive">Archive</a>
  </div>
</nav>"""

u = N["uv"]
facts = f"""
    <div class="facts">
      <div class="fact"><div class="v">{u['pip'][0]:,} → {u['pip'][1]} <small>faces</small></div><div class="k">of Pip's texture sharing texels with another surface - his vest's paint on his sleeves. Every character had {u['min_before']}–{u['max_before']:,}; now {u['min_after']}–{u['max_after']}</div></div>
      <div class="fact"><div class="v">wings → none</div><div class="k">Pip's vest stays on his torso with his arms overhead; the rig's rule for it had never acted - its flood walked onto the vest</div></div>
      <div class="fact"><div class="v">Studio <small>:8830</small></div><div class="k">make, watch, turn round and walk your own characters in a local app - <span class="mono">python charforge.py studio</span></div></div>
      <div class="fact"><div class="v">{N['cast_count']} new</div><div class="k">characters made with the pipeline as it is now - {N['cast_names']}</div></div>
      <div class="fact"><div class="v">0.77–0.84 <small>IoU</small></div><div class="k">the rig's silhouette on the video's figure, four moves from video in three styles, every frame</div></div>
      <div class="fact"><div class="v">30° → 8°</div><div class="k">how far Aoi's hands point from the video's, turned to DWPose's reading - palms the right way 62% → 99% of frames</div></div>
    </div>"""

opener = f"""<header class="opener" id="top">
    <div class="eyebrow">24-26 September 2026 · M5 MacBook, 24 GB shared memory</div>
    <h1>What CharForge makes now, and what fixing it took</h1>
    <p class="sub">A prompt or a picture becomes a rigged, animated game character on this laptop, and a video becomes a
      move on it. This round went through every character in the poses that break things and traced each fault to
      its cause before touching it: Pip's vest no longer lifts into wings when he raises his arms, no texture carries
      another garment's paint, a hem stays with the hips instead of the thighs, and a reference drawn in the wrong pose
      is drawn again before half an hour goes into it. Then the characters made from a few words in the Studio came out
      poor, and that was traced too: <a href="#prompts">a short prompt is written out</a> before anything is drawn, and a
      model with the picture's backdrop built into it as a board is made again. Every move can now be watched from
      <a href="#angles">four sides at once</a>, which found an elbow folded through an arm and feet standing off the
      floor. CharForge Studio does all of it without a terminal. The page leads with how things work now; what they
      replaced is in the <a href="#archive">archive</a> at the end.</p>
{facts}
    <div class="licence"><span aria-hidden="true">⚠</span><div><b>Licences.</b> The videos of moves on this page come from MiniMax H3 (the Singularity community fine-tune). Its licence excludes the US, EU, UK and South Korea and covers what it generates - this page is private; check before sharing it. Qwen-Image 2.1, which draws the references, is under a research-only licence: commercial use needs a licence from Qwen.</div></div>
  </header>"""

studio = """<section id="studio">
    <div class="sechead">
      <div class="eyebrow">Make your own · a local web app, nothing uploaded anywhere</div>
      <h2>CharForge Studio</h2>
      <p>Everything this page describes, run from the browser: describe a character in a few words or drop in a picture,
        pick a style, press Make. <b>Write it out</b> shows what the image model will be asked for - the few words
        completed with an age and a build, skin and hair, and each garment in its own colour, by a local model - and lets
        you edit it; the height follows from it unless you set one. A name already taken is refused on the spot (a new
        prompt under an old name used to give back the old character). Jobs run one at a time in a queue that says what each is doing and how long it has
        left - learned from the stages' own logs on this Mac. Each character gets a page: turn it round in 3D, play its
        clips, compare its moves with the videos they came from, watch any clip from the front, the side and above at once
        (<b>Angles</b>), download the <span class="mono">.glb</span> and
        <span class="mono">.fbx</span>, add moves from video, rerun from any stage. <b>Walk around</b> opens the
        playground with every character on this Mac, not only the published ones. The Studio can be closed and opened
        again while a job runs; it finds the job and waits for it.</p>
    </div>
    <div class="cmd">cd charforge
python charforge.py studio            # opens http://localhost:8830
# or double-click "CharForge Studio.command" in Finder</div>
    <div class="shots" style="margin-top:16px">
      <figure class="shot"><img src="media/studio_grid.webp" alt="The Studio's character grid: the fourteen characters on this Mac with their styles, heights and moves" width="1440" height="900" loading="lazy">
        <figcaption><b>Characters</b> - everything in <span class="mono">work/</span> and <span class="mono">out/</span>, newest first.</figcaption></figure>
      <figure class="shot"><img src="media/studio_char.webp" alt="Aoi's page in the Studio: the 3D model on a grid, clip controls, her prompt and numbers, her move from video and the stages" width="1440" height="900" loading="lazy">
        <figcaption><b>A character</b> - the model playing its clips, its moves beside their videos with their scores, every stage.</figcaption></figure>
      <figure class="shot"><img src="media/studio_new.webp" alt="The New character form: 'a boy scout' written out as a ten-year-old scout in a striped shirt, khaki shorts and hiking boots, with the height set from it" width="1440" height="900" loading="lazy">
        <figcaption><b>New character</b> - a few words, <b>Write it out</b>, edit what comes back; style, and a height that follows the description.</figcaption></figure>
      <figure class="shot"><img src="media/studio_jobs.webp" alt="The Jobs page: the running job's stage and progress bar with the time left, and the queued jobs with when each starts and how long it takes" width="1440" height="900" loading="lazy">
        <figcaption><b>Jobs</b> - the stage each job is in, its time left, when the next ones start; logs one click away.</figcaption></figure>
      <figure class="shot" style="grid-column:1/-1"><img src="media/studio_angles.webp" alt="The Angles panel on Mara's page, paused at the top of her roundhouse kick: the video, its camera, the left side and above, with the joint readout under them and play, frame-step and scrub controls below" width="1440" height="900" loading="lazy">
        <figcaption><b>Angles</b> - any clip from four sides at once, the joint readout under it, and a frame at a time
        (<span class="mono">,</span> and <span class="mono">.</span>) to find the worst one. Paused here at the top of Mara's kick.</figcaption></figure>
    </div>
  </section>"""

PR = N.get("prompts", {})
prompts_sec = f"""<section id="prompts">
    <div class="sechead">
      <div class="eyebrow">Few-word prompts · three characters made in the Studio by a user</div>
      <h2>From a few words to a character</h2>
      <p>Made from the Studio with short prompts, they came out poor, and each for its own reason. "A boy scout" came back a
        grown man, at the height slider's default adult 1.72 m, and his model lost the picture's face and hair. "Gray alien
        that is extremely muscular" came back a grey clay sculpture - no clothes, no colour - and its 3D model a flat board
        with a body behind it. "A young knight in steel armor" gave back the Knight already made, in 0.0 minutes: the name
        was taken, every stage was cached, and the new prompt was quietly written over the old character's.</p>
    </div>
    <div class="fixlist">
      <div class="note-card"><div class="eyebrow">A prompt is written out first</div>
        <p>A local language model (Ollama, on this Mac; unloaded as soon as it has answered) fills in what a few words leave
        open - an age and a build, skin and hair, the face, and the top, the bottom and the shoes each with its own colour -
        and the sentence is put together from its fields. Asked for free prose under rules, 3B and 7B models broke them one
        prompt in two ("holding a wooden badge", "a small shield", "standing at 1.5 meters tall"), so a field that names a
        prop, a pose or a height is dropped, and the age comes from your own words. "A boy scout" now reads: <i>{PR.get('scout', 'a boy scout, light skin, short brown hair ... wearing a blue and white striped shirt, khaki shorts and brown hiking boots, with a yellow neckerchief')}</i>. The Studio shows it before anything is drawn, to edit; a prompt of 20 words or more is used as written.</p></div>
      <div class="note-card"><div class="eyebrow">The height follows the description</div>
        <p>A child 1.2-1.5 m, a teenager about 1.6, an adult about 1.75 - unless you set it. The slider no longer sends 1.72
        for everyone.</p></div>
      <div class="note-card"><div class="eyebrow">A name already taken is refused</div>
        <p>In the Studio as you type (with a free name offered), and by <span class="mono">charforge.py make</span>, which will
        not write a new prompt, style or height over a character that exists - rerun it from a stage for that.</p></div>
    </div>
    <figure class="wide" style="margin-top:18px"><img src="media/prompts_board.webp" alt="TRELLIS's first pass seen from the front: Gray's is a flat grey board hiding the figure, Knight's has a board behind him, Kaito's a board through his legs; Bo's is clean" width="1218" height="300" loading="lazy">
      <figcaption><b>A board built into the model.</b> TRELLIS's first pass from the front, with how its silhouette lies on the
      picture's: sound models 0.90-0.94, these 0.35-0.44 - the picture's plain backdrop made solid, in front of the figure,
      behind it or through it. Knight's membranes and Kaito's loose pieces were what was left of it after the later stages.
      Both passes are now checked from the front, and a model that does not match is made again on a new seed.
      {PR.get('board_note', '')}</figcaption></figure>
    {'<figure class="wide" style="margin-top:18px"><img src="media/prompts_before_after.webp" alt="The boy scout and the grey alien: the picture and the model as they first came out, and as they come out now" loading="lazy"><figcaption><b>Two characters from a user’s few-word prompts, before and now.</b> ' + PR.get('after_note', 'Made again with the prompt written out - the new pictures and models replace these as they finish.') + '</figcaption></figure>' if (MEDIA / 'prompts_before_after.webp').exists() else ''}
  </section>"""

pipeline = """<section id="pipeline">
    <div class="sechead">
      <div class="eyebrow">How a character is made now · steps marked new changed this round</div>
      <h2>Two lanes, one skeleton</h2>
      <p>A character is built once from a prompt or an image. Moves are then added to it from videos - one video per move
        serves every character, because the clip it becomes is a standard Mixamo skeleton.</p>
    </div>
    <div class="lanes">
      <div class="lane">
        <h3>Making a character <span class="eyebrow">charforge.py make · ~35 min</span></h3>
        <ol class="steps">
          <li class="new"><b>The prompt, written out</b><span>a local model completes a few words into a character - age and build, skin and hair, each garment in its own colour, nothing held in the hands - and the height follows from it</span><code>describe.py</code></li>
          <li class="new"><b>Reference image, checked</b><span>Qwen-Image 2.1 draws it in an A-pose with its own alpha; a pose model reads the arms and it is drawn again, up to three times, while the hands touch the body</span><code>reference · pose_gate.py</code></li>
          <li class="new"><b>3D model, checked from the front</b><span>TRELLIS.2, a second pass seeing repainted side and back views; a model whose front silhouette does not lie on the picture's (a board built in with the figure) is made again on a new seed</span><code>generate · multiview</code></li>
          <li><b>Solid, joints, hands</b><span>closed into a solid, limbs traced to their tips, modelled hands with 15 finger bones each</span><code>solidify · joints · hands</code></li>
          <li class="new"><b>Arms cut free</b><span>of whatever they were generated against below the armpit, then checked slab by slab and cut again - wider and further out - where an arm still reaches the body</span><code>free_arms.py</code></li>
          <li class="new"><b>Mesh, UVs, texture</b><span>game topology; UV islands folded onto themselves unwrapped again; the source images projected back on - not beside a nearer surface's outline - the hands in the face's skin tone</span><code>retopo · texture</code></li>
          <li class="new"><b>Rig</b><span>weights measured through the body; the arm's weight on the arm alone below the armpit; a vest's armhole with the collarbone and a jacket's hem with the pelvis, told apart by colour</span><code>weights · rig</code></li>
          <li><b>Face, clips, package</b><span>jaw and five shapes; 18 Mixamo clips with planted feet; glTF + FBX with LODs and a manifest</span><code>face · animate · package</code></li>
        </ol>
      </div>
      <div class="lane">
        <h3>Adding a move from video <span class="eyebrow">charforge.py move · ~40 min each</span></h3>
        <ol class="steps">
          <li><b>Write the move</b><span>timed beats - ready, the move, recovery - in MiniMax's six-section prompt format</span><code>motion_prompts.json</code></li>
          <li><b>Generate the video</b><span>MiniMax H3 in ComfyUI: the reference image in, 5 s of the character doing it out</span><code>motion_video.py</code></li>
          <li><b>Read the pose</b><span>ViTPose's 17 points on every frame; DWPose's 21 per hand</span><code>video_motion.py · dwpose.py</code></li>
          <li><b>Nearest capture, fitted</b><span>the closest of 2,453 Mixamo clips, bent toward the video only where it misses by more than the pose model's jitter (-34% error on moves the library lacks)</span><code>match · --fit</code></li>
          <li><b>Clip onto the rig</b><span>retargeted, the character's own limbs, spine and head aimed along the fit, feet planted</span><code>retarget.py</code></li>
          <li><b>Refine against the video</b><span>the character's mesh re-posed frame by frame until its silhouette, joints, face and hands lie on the video</span><code>refine_pose.py</code></li>
        </ol>
        <div class="cmd">python charforge.py move --name mara --move punch_combo,roundhouse_kick
python charforge.py move --name juno3 --move punch_combo@mara,spell_cast@aoi   # other characters' videos</div>
      </div>
    </div>
  </section>"""


def dims(img, w, h):
    """The image's own size when it exists, so the page reserves the right space."""
    try:
        from PIL import Image
        return Image.open(MEDIA / img).size
    except Exception:
        return w, h


def fix(tag, title, img, alt, num, body, w=1200, h=600):
    w, h = dims(img, w, h)
    return f"""<article class="fix">
      <img src="media/{img}" alt="{alt}" width="{w}" height="{h}" loading="lazy">
      <div class="txt">
        <div class="tag">{tag}</div>
        <h3>{title}</h3>
        <div class="num">{num}</div>
        {body}
      </div>
    </article>"""


P = N["pip_rules"]
E = N.get("elbow")
elbow_card = "" if not E else fix("the move", "Elbows no longer fold back through the arm", "polish_elbow.webp",
    "Aoi's spell cast at 1.5 and 2 seconds: the video; before, at 1.5 seconds her left jacket sleeve sticks far out sideways past an elbow folded back on itself; after, both elbows out as in the video with her hands in front of her chest. At 2 seconds the elbow was already within range and nothing changes",
    f"{E['before']}° → {E['after']}° <small>her most folded elbow in the spell; the motion library's reach 159° at most</small>",
    f'''<p>In her spell cast Aoi presses her hands together in front of her chest, elbows out. One camera cannot say how far a
    bone reaches toward it, so the fit takes each bone's depth from its length in the image - up to a sign - and it took the
    matched capture's side for her upper arm: tipped back. The elbow came out folded to {E['before']}°, the forearm back through
    the upper arm, and her jacket sleeves, which move with the upper arm, stuck straight out sideways past the fold. The hand
    score caught it: the pose model read those sleeve ends as her hands.</p>
    <p>The fit now chooses an upper arm's side and its forearm's together. An elbow folded past 150-160° costs - the motion
    library's pass 150° in 0.05% of frames - and so does a wrist within the chest's width and height put behind the torso,
    as 1% of the library's such wrists are. That alone brought this elbow to {E['fit']}°: where the video shows an upper arm at
    its full length there is no side left to choose (the pose model puts her elbow at the outside of a puffy sleeve). So the
    refine stage also holds every elbow under 150°: {E['after']}°, with the silhouette and the hands where they were. On 40
    library moves it had never seen, the new fit is as close as the old (0.212 → 0.211 torso lengths).</p>''', 912, 606)
FT = N.get("feet")
feet_card = "" if not FT or not (MEDIA / "polish_feet.webp").exists() else fix(
    "the move", "Feet back on the floor after the refine", "polish_feet.webp",
    "Aoi's spell cast from the side at 0, 1 and 2 seconds: before, both boots hover above the grid's floor line; now they stand on it",
    f"{FT['before_cm']} → {FT['after_cm']} cm <small>Aoi's lowest shoe above the floor, most of her spell cast</small>",
    f'''<p>Seen from the side for the first time, Aoi's spell cast stood 2-6 cm in the air. The retarget had put her planted
    soles on the floor (0.02 cm off it on average); the refine then turned her hips, knees and ankles toward the video,
    which lifted them, and nothing put them back. The correction step now measures each key's lowest shoe before the turn
    and after, and moves the pelvis by the difference - the lowest foot goes back to where the retarget had it, on the floor
    while planted, as high as it was in a jump. {FT.get('note', '')}</p>''', 720, 240)
polish = f"""<section id="polish">
    <div class="sechead">
      <div class="eyebrow">Polish · every character in the poses that break things, front-left and back-right</div>
      <h2>What broke when the characters moved, and what fixed it</h2>
      <p>Each character was rendered at the extremes of its clips and moves - the jump's crouch, the landing, the wave, the
        fall, arms overhead - and every fault traced to its cause before anything was changed. Twice the obvious cause
        was the wrong one, and the fix that looked right changed nothing; those are in the <a href="#archive">archive</a>.
        Left of each pair: this morning. Right: now.</p>
    </div>
    {fix("the rig", "Pip's vest no longer lifts into wings", "polish_wings.webp",
         "Pip from behind in five poses, before and after: before, the yellow vest flares out like wings and a pocket juts from his thigh; after, the vest stays on his torso",
         f"{P['left_let_go']:,} + {P['right_let_go']:,} <small>vest vertices now let go of the arms (before: 0 and 5)</small>",
         '''<p>Three causes, stacked. The 3D model glued his arms to the vest's sides - now cut free below the armpit. The rig's
         rule that keeps an arm's weight on the arm floods the arm's own surface from the sleeve's outside, and it started
         at the very slab where the cut opens, where sleeve and vest are still one surface: it walked onto the vest on both
         arms and changed nothing. It starts a little lower now. Above the cut, the vest's armhole <em>is</em> the arm's
         surface and took half its weight, as any shoulder does; the vest's yellow and the sleeve's blue tell them apart, and
         the armhole now moves with the collarbone.</p>
         <p>The cargo pocket that stood off his thigh like a plank: the part labeller calls it an accessory, and accessories
         rode the torso. Below the hip joints one now stays on its leg (off-thigh vertices in the landing 122 → 25).</p>''', 1200, 1200)}
    {fix("the rig", "Arms overhead", "polish_cheer.webp",
         "Pip's victory cheer with both arms straight up, before and after: before, the vest's sides rise with the arms; after, the vest stays down and the armpits stretch",
         "vest down <small>armpit stretches</small>",
         '''<p>With both arms straight up the vest stays on his torso. What is left is the stretch between vest and sleeve at the
         armpit: they are one surface, and a surface joining a garment that stays to a sleeve that rises has to stretch
         somewhere. The fix for that is the vest as its own layer.</p>''', 1200, 600)}
    {fix("the texture", "No garment carries another's paint", "polish_uv.webp",
         "The back of Pip's upper arm: before, patches of the vest's yellow on his blue sleeve; after, clean blue",
         f"{u['min_before']}–{u['max_before']:,} → {u['min_after']}–{u['max_after']} <small>faces sharing texels, per character</small>",
         f'''<p>The patches looked like a projection error, and three rounds of fixes to the projection changed nothing. The cause
         was the UV layout: unwrapped on a smoothed copy of the mesh, islands folded onto themselves wherever it curls, and
         two surfaces painted the same texels - {u['pip'][0]:,} faces on Pip, {u['vex'][0]:,} on Vex. Repacking cannot part an island
         from itself; the overlapping faces are unwrapped again, finer each time, and every island brought back to one texel
         density (unwrapped alone, they took 42% of Pip's atlas).</p>''', 1200, 600)}
    {fix("the texture", "Aoi's trousers lost their light streaks", "polish_aoi.webp",
         "Aoi's trouser texture before and after: before, pale slashes across the dark fabric; after, clean",
         "10–20 px <small>her side view's hands stood off the mesh's</small>",
         '''<p>Not the generated hands the modelled ones replaced (masking them changed 12,513 texels and left the streaks): in
         the generated side view her hands hang a little lower than the mesh's, and the trouser texels just past the mesh's
         hand read the picture's hand. A generated view now fades out beside the outline of anything nearer - a depth jump
         between neighbouring pixels. Not in the reference, which the mesh was made from: half of Aoi's face lies behind a
         lock of her hair.</p>
         <p>The same pass found the modelled hands' region reaching into her hip (36,482 texels painted skin) and 161,553 texels
         of Pip: a texel is the hand's now only if it stands outside the body the hand was joined to.</p>''', 1200, 900)}
    {fix("the texture", "Hands the colour of the face", "polish_hands.webp",
         "Pip's hands before and after: brown and grey before, peach like his face after",
         "(137, 95, 71) → (222, 152, 122)",
         '''<p>The modelled hands took the median of every skin-like colour on the body - on Pip that is his orange hair, khaki
         shorts and the vest's shading as much as skin. The face's own texels decide it now.</p>''', 1200, 600)}
    {fix("the rig", "A jacket's hem follows the hips, not the thighs", "polish_hem.webp",
         "Juno crouching, before and after: before, the red jacket's hem is pulled down between her thighs; after, it hangs level over her hips",
         f"{N['hem']['juno3']:,} <small>vertices of Juno's jacket moved from the thighs to the pelvis</small>",
         '''<p>The lowest part of Pip's vest carried half its weight on his thighs (median 0.50), and spreading the legs in a
         jump pulled its hem into a W; Juno's windbreaker wrapped round her thighs in a squat. Where the torso's colours and
         the upper thighs' differ, the upper garment near the hips follows the pelvis, and a long jacket still rides on the
         seat below it; one colour top to bottom - Mara's olive and tan are 5 apart - nothing changes, and her kick still
         drags her jacket along the raised thigh.</p>''', 1200, 540)}
    {fix("the solid", "Webs between an arm and the body - most of them a board", "polish_webs.webp",
         "Knight waving and jumping, before and after: before, grey sheets hang from his arms to his hips; after, none",
         "8,286 + 10,349 → 1,877 + 2,326 <small>voxels of web between Knight's arms and body</small>",
         '''<p>Waving stretched grey sheets from Knight's arms to his hips. Cutting thin webs out of the solid before the arm
         is freed took some (anything under about 9 mm goes, and pieces the cuts split off are dropped), but fins and a strap
         remained - because most of it was not a web: TRELLIS had built the picture's backdrop into his model as a board
         behind him (see <a href="#prompts">Few-word prompts</a>), and the later stages cut it into sheets. Made again through
         TRELLIS's own background remover, his model has no board, the web cut finds a fifth as much, and the sheets, fins
         and strap are gone.</p>''', 1200, 300)}
    {fix("the rig", "Loose pieces ride with the body", "polish_loose.webp",
         "Kaito crouching: before, a hair spike floats beside his head, left where the rest pose had it; after, it moves with his head",
         "37 <small>loose pieces on Kaito, now each riding with the body part nearest it</small>",
         '''<p>Kaito's mesh came as 40 pieces: the body, the two modelled hands, and 37 bits the generator left detached - the
         inner toes of his split-toe boots, hair spikes. The weights follow paths through the body and never reached them,
         so they stayed where the rest pose left them. Each now rides rigidly with the body part nearest it.</p>''', 720, 360)}
    {elbow_card}
    {feet_card}
    {fix("the reference", "A picture in the wrong pose is drawn again", "polish_bo.webp",
         "Bo the chef: the first reference with his fists on his hips, the model it made with shredded hands at his belt, and the reference drawn again with his arms clear",
         "0.33 <small>torso lengths from Bo's wrists to his hips; good references 0.60–0.79</small>",
         '''<p>Asked for an A-pose in so many words, the image model drew a chef's habitual stance - fists on hips - and 32 minutes
         later the fused hands came out in shreds at his belt. A pose model now reads the arms of every new reference, and
         below 0.45 it is drawn again on a new seed. A picture you give is only warned about.</p>''', 1200, 600)}
  </section>"""

cast_cards = []
for c in N["cast"]:
    refs = "".join(f'<figure><img src="media/{r[0]}" alt="{r[1]}" width="400" height="400" loading="lazy"><figcaption>{r[2]}</figcaption></figure>' for r in c["refs"])
    cast_cards.append(f"""<article class="who">
        <video src="media/{c['video']}" autoplay muted loop playsinline controls preload="metadata" aria-label="{c['name']} walking, waving and turning"></video>
        <div class="body">
          <div class="title"><h3>{c['name']}</h3><span class="chip">{c['style']}</span></div>
          <p class="prompt">"{c['prompt']}"</p>
          <dl class="kv"><dt>height</dt><dd>{c['height']}</dd><dt>triangles</dt><dd>{c['tris']}</dd><dt>made in</dt><dd>{c['time']}</dd><dt>notes</dt><dd style="font-family:var(--sans)">{c['notes']}</dd></dl>
          <div class="refrow">{refs}</div>
        </div>
      </article>""")
cast = f"""<section id="cast">
    <div class="sechead">
      <div class="eyebrow">New characters · made start to finish by the pipeline as it is now</div>
      <h2>{N['cast_title']}</h2>
      <p>{N['cast_intro']}</p>
    </div>
    <div class="cast">{''.join(cast_cards)}</div>
  </section>"""

# --- reused sections -----------------------------------------------------------------------------
moves = block("moves").replace(
    "His elbows were rigged 4 cm too far down the arm - fixed. Arms overhead still lift his puffy vest into wings (open problems).",
    "His elbows were rigged 4 cm too far down the arm - fixed. With his arms overhead his vest now stays on his torso (<a href=\"#polish\">Polish</a>).").replace(
    "A head-height kick with the torso leaning 30°. At the top of the kick the pose model reads the leg as folded, and the rig follows it - the one failure left here (below).",
    "A head-height kick with the torso leaning 30°. At its top the rig's leg is up at head height as in the video (<a href=\"#angles\">four sides</a>); "
    "the pose model misreads the render of that leg for 7 frames, most of this score - without them, joints 0.049.")

AN = [(n, mv, t) for n, mv, t in (("aoi", "spell_cast", "Aoi · spell cast"), ("mara", "punch_combo", "Mara · punch combo"),
                                    ("mara", "roundhouse_kick", "Mara · roundhouse kick"), ("pip", "victory_cheer", "Pip · victory cheer"))
      if (MEDIA / f"angles_{n}_{mv}.mp4").exists()]
angles_sec = "" if not AN else f"""<section id="angles">
    <div class="sechead">
      <div class="eyebrow">Every move from four sides at once · tools/make_angles.py</div>
      <h2>The same frame from the video's camera, the side and above</h2>
      <p>One camera cannot see depth: from the video's angle a clip can look right while an arm folds back through itself or
        the feet float. Each move is now rendered from several cameras at once, side by side in one video with the video
        itself - every panel orthographic and at one scale, on a 10 cm grid, so what lies in front of what, and how far,
        reads straight off it - and each elbow's and knee's bend and each shoe's height above the floor are read out under
        the panels, red where they go further than the motion library ever does. The Studio has it for any clip of any
        character (<b>Angles</b>).</p>
    </div>
    <div class="angles">{''.join(f'<figure><video src="media/angles_{n}_{mv}.mp4" poster="media/angles_{n}_{mv}.webp" autoplay muted loop playsinline preload="metadata" onclick="this.paused ? this.play() : this.pause()" title="Click to pause or play" aria-label="{t} from the video, its camera, the side and above"></video><figcaption>{t} <span class=\"hint\">click to pause</span></figcaption></figure>' for n, mv, t in AN)}</div>
    <p class="ink2" style="margin-top:12px">It found what the front view hid - Aoi's elbow folded back through her arm, and
      the feet of every move from video standing off the floor after the refine (<a href="#polish">Polish</a>). And two
      things about the checking itself. The video beside the rig ran 25% fast: the 24 fps video was stretched onto the
      renders' 30 fps by taking each frame once, never twice, so Mara's jab seemed to land 0.2 s late and her last guard
      held while she kept punching. Every compare video here has been remade in time. And the dip in her kick's score at
      its top is the pose model misreading the render, not the rig: from the side and above her leg is up at head height
      as in the video.</p>
  </section>"""

fid = block("fidelity")
a = fid.index('<h3 style="margin:30px 0 10px">Where the error came from</h3>')
b = fid.index('<h3 style="margin:30px 0 10px">What fixed it</h3>')
error_source = fid[a:b]
fid = fid[:a] + fid[b:]
fid = fid.replace('<h3 style="margin:30px 0 10px">What fixed it</h3>', '<h3 style="margin:30px 0 10px">How the rig follows the video</h3>')
fid = fid.replace("Now: facing forward, arms up as in the video. The vest still lifts into wings.",
                  "Now: facing forward, arms up as in the video. (The vest's wings in the before picture are gone too - see Polish.)")
fid = fid.replace("""<p class="ink2" style="font-size:14px"><b>The pose model's blind spots.</b> At the top of the kick it loses the leg either side of the apex - ankle confidence 0.24-0.40, the points bunched at the hip - and the rig follows it: frames 56-70 overlap the video at IoU 0.54, the rest of the kick at 0.86. A correction a few degrees at a time cannot swing a leg 90°. And its 17 points stop at the wrists, so hands keep the capture's angle.</p>""",
                  """<p class="ink2" style="font-size:14px"><b>The pose model's blind spots.</b> This picture is from before the refine, when the rig followed the pose model's folded reading of the kick. Now the rig's leg is up at head height as in the video, seen from the side and above (<a href="#angles">four sides</a>), and the skeleton is 0.12-0.14 torso lengths off the video at the top, as through the rest of the kick. What dips is the score: reading the <i>render</i>, the pose model puts her right ankle 2.9 torso lengths from where her skeleton has it, in frames 61-67. Without those frames the kick scores 0.049 instead of 0.069. What the rig still does wrong there: it lowers the leg a little early - at 2.8 s her knee is bent 70° while the video's leg is still up.</p>""")
fid = fid.replace("""The top of Mara's kick: in the video the leg points up at head height while the pose model's leg points are bunched near her hip; the rig's leg is folded""",
                  """The top of Mara's kick before the refine: in the video the leg points up at head height while the pose model's leg points are bunched near her hip; the rig's leg is folded""")
fid = fid.replace("""<p class="ink2" style="font-size:14px"><b>Parts fused to the body.</b> Aoi's hair and Pip's vest are one surface with what they lie on, so they cannot move apart from it (tried for the vest today - see below).</p>""",
                  """<p class="ink2" style="font-size:14px"><b>Parts fused to the body.</b> Aoi's hair is one surface with the back it lies on, so it cannot swing apart from it. Pip's vest now moves with his torso (<a href="#polish">Polish</a>).</p>""")

hands = block("hands")

# the four moves as they score now (collect_numbers.py -> moves_now): the quoted scores, the chart's "now" bars
# (290 px to an IoU of 1) and the hands table, each hand error with its median beside it
MV = N.get("moves_now", {})
SCORED = [("mara/punch_combo", "Mara · punch", 51, "0.054 · IoU 0.845", "0.054 · 0.845</td><td class=\"num\">0.011"),
          ("mara/roundhouse_kick", "Mara · kick", 97, "0.076 · IoU 0.823", "0.076 · 0.823</td><td class=\"num\">0.013"),
          ("aoi/spell_cast", "Aoi · spell", 143, "0.127 · IoU 0.769", "0.127 · 0.769</td><td class=\"num\">0.030"),
          ("pip/victory_cheer", "Pip · cheer", 189, "0.082 · IoU 0.791", "0.082 · 0.791</td><td class=\"num\">0.020")]
for key, label, y, old_mv, old_tab in SCORED:
    m = MV.get(key)
    if not m:
        continue
    assert old_mv in moves and old_tab in fid, key
    moves = moves.replace(old_mv, f"{m['joints']:.3f} · IoU {m['iou']:.3f}")
    fid = fid.replace(old_tab, f"{m['joints']:.3f} · {m['iou']:.3f}</td><td class=\"num\">{m['face']:.3f}")
    w = m["iou"] * 290
    fid, k = re.subn(rf'<rect x="118" y="{y}" width="[\d.]+" height="14" rx="2" fill="var\(--bar-b\)"/><text class="val" x="[\d.]+" y="{y + 11}">[\d.]+</text>',
                     f'<rect x="118" y="{y}" width="{w:.1f}" height="14" rx="2" fill="var(--bar-b)"/><text class="val" x="{118 + w + 6:.1f}" y="{y + 11}">{m["iou"]:.2f}</text>', fid)
    assert k == 1, key
    row = re.search(rf"<tr><td>{re.escape(label)}</td>(.*?)</tr>", hands)
    if row and m.get("hands") is not None:
        (h0, _), (f0, _), (p0, _) = re.findall(r'<td class="num">([^<]*?) → <b class="good">([^<]*?)</b></td>', row.group(1))
        hands = hands.replace(row.group(0),
                              f'<tr><td>{label}</td><td class="num">{h0} → <b class="good">{m["hands"]:.2f}</b> <small class="ink2">median {m["hands_median"]:.2f}</small></td>'
                              f'<td class="num">{f0} → <b class="good">{m["facing"]:.0%}</b></td><td class="num">{p0} → <b class="good">{m["pointing"]:.0f}°</b></td></tr>')
hands = hands.replace("""(lower is better), how often the palm faces the video's way, how far off the hand points.""",
                      f"""(lower is better), how often the palm faces the video's way, how far off the hand points. {N.get('hands_note', '')}""")
mac = block("mac")
watchdog = """
      <div class="note-card">
        <div class="eyebrow">macOS stops a GPU job that holds the display up</div>
        <div class="big">6 blocks → 2 a buffer</div>
        <p>Knight's and Kaito's 3D models died at TRELLIS's 1024 stage (~8,000 tokens) with "Impacting Interactivity": macOS kills a GPU command buffer that runs too long while the screen needs the GPU. The flow model now evaluates two transformer blocks at a time, and a killed pass is retried once a block at a time with fewer operations per buffer. Both then built.</p>
      </div>
    </div>"""
mac = mac.replace("""    </div>
  </section>""", watchdog + """
  </section>""", 1)

left = f"""<section id="left">
    <div class="sechead">
      <div class="eyebrow">What is left · measured, ranked by what it costs a character</div>
      <h2>Open problems</h2>
    </div>
    <ul class="left">
      <li><b>Layered garments and hair are one surface with what they lie on.</b> Pip's vest now moves with his torso, but
        with both arms overhead the surface between vest and sleeve stretches at the armpit; Aoi's hair and Mara's and Wren's
        braids stay rigid because a chain on a fused part tears it. The fix is upstream: generate them as their own layers.</li>
      <li><b>A join survives where the arm cut begins.</b> {N['joins_left']}</li>
      <li><b>Armour is the hardest case.</b> Most of what looked like membranes between Knight's plates was a board the 3D
        model had built behind him; made from the cut-out picture, he is clean. Gloves and gauntlets are not: the hand step
        models a hand and paints it with the face's skin, so knight2's hands are bare. Remade the same way, Kaito's
        loose boot toes are whole bare toes in open sandals, as drawn - painted with smudges of the sandal's black.</li>
      <li><b>A sleeve that ends past the elbow.</b> Bent far, an elbow folds the forearm back but the part of a jacket
        sleeve beyond the joint keeps pointing along the upper arm, as linear skinning does: in Aoi's spell cast her sleeve
        cuffs stick out sideways past her elbows, and the pose model reads their light lining as her hands (her hand score's
        mean 0.33 against a median of 0.17). Corrective shapes at the elbow, or sleeves generated apart from the arm, would
        reach it.</li>
      <li><b>The prompt model is small.</b> Gemma 4 (e4b), the default now, kept 19 of 20 rules in the test set; the
        7B model before it dressed the grey alien all in black against "each garment its own colour", and he keeps that
        outfit. What any of them adds is a guess. A better local model plugs in on the Studio's Setup page, and the
        written-out text can always be edited before drawing.</li>
      <li><b>A face the rig will not guess.</b> The face stage places blinks, a jaw and a mouth only on eyes and a
        mouth it can find, and refuses rather than guess: Bo's eyes are behind round glasses and his mouth under a
        moustache, so he plays every clip but neither blinks nor talks. (Knight was refused too, until his face was
        painted where his geometry has it - then it found both, and he has all five shapes.)</li>
      <li><b>The reference check reads the arms only.</b> A held prop, a cape or a hat the 3D model will fuse is still for a person to catch.</li>
      <li><b>Poses the pose models rarely saw.</b> Reading the render of Mara at the top of her high kick, ViTPose loses the raised leg for 7 frames, which is most of the kick's score (0.069; 0.049 without them) - the score, not the rig, which kicks as the video does but lowers the leg a little early. Hands blurred by a fast move, under hair or overhead at the frame's edge keep the capture's angle.</li>
      <li><b>Depth in moves from video</b> is borrowed from the nearest library capture; a move unlike anything in the library is right in the image and approximate in depth.</li>
      <li><b>Licences.</b> H3's for anything public; Qwen-Image 2.1 is research-only (Krea 2 remains a flag away); a licence-clean video model (Wan 2.2, Apache-2.0) is the alternative to try.</li>
    </ul>
  </section>"""

# --- the archive -----------------------------------------------------------------------------------
def arch(title, note, inner):
    return f"""<details class="arch"><summary><b>{title}</b><span>{note}</span></summary><div class="inner">{inner}</div></details>"""

views = block("views")
fit = block("fit")
images = block("images")
open_old = block("open")
# the dead-ends table and the old wings card from the old "open" section
open_old = open_old.replace("<h2>What did not work, and what is left</h2>", "<h2>What did not work (until 25 September)</h2>")
new_dead = """<tr><td>Let a vest's side go below the armpit wherever its colour says vest</td><td class="ink2">joins between arm and vest stretched into sheets up to the raised elbow - worse than the flap</td><td><span class="pill dropped">dropped</span></td></tr>
          <tr><td>Colour an arm's unseen surface with the arm's colour at its height</td><td class="ink2">no visible change once the folded UV islands were found and repaired</td><td><span class="pill dropped">removed</span></td></tr>
          <tr><td>Mask where the source images show the old generated hands</td><td class="ink2">12,513 texels changed; Aoi's trouser streaks stayed - the side view's hands were the cause</td><td><span class="pill dropped">not wired in</span></td></tr>
          <tr><td>Check the arm cut in slabs of the nearest section</td><td class="ink2">torso voxels went to the armpit's sections; Vex's joins went unseen</td><td><span class="pill dropped">replaced</span></td></tr>
          <tr><td>Cut more directions where a join survives</td><td class="ink2">the ring ran inside a bulging sleeve (Vex, right arm at 0.41) - nothing changed; the ring moves outward instead</td><td><span class="pill dropped">replaced</span></td></tr>
          <tr><td>Repair folded UV islands on the shipped mesh, down to an island a face</td><td class="ink2">atlas coverage 0.58 → 0.42 - a third of the texture wasted</td><td><span class="pill dropped">dropped</span></td></tr>
          <tr><td>Tell a jacket from the trousers by the human parser's classes where their colours are too close</td><td class="ink2">the parser gave the lower back of Mara's, Juno's and Rowan's jackets to their trousers; the hem rule split each jacket across the seat, and a crouch showed a pale band</td><td><span class="pill dropped">dropped</span></td></tr>
          <tr><td>Fill the arm cut's new surfaces from the nearest texel facing the same way</td><td class="ink2">through the air, the back of a sleeve and the back of the vest beside it both face backwards</td><td><span class="pill dropped">replaced</span></td></tr>
        </tbody>"""
open_old = open_old.replace("        </tbody>", "          " + new_dead, 1)
open_old = open_old.replace("<h3>Open: Pip's vest lifts into wings</h3>", "<h3>Was open: Pip's vest lifts into wings (fixed 25 September - see Polish)</h3>")
open_old = open_old.replace('<h3 style="margin-top:6px">Also open</h3>', '<h3 style="margin-top:6px">Also open then</h3>')

def demote(sec):
    """Archived sections keep their content; their ids change so the nav's anchors stay unique."""
    return re.sub(r'<section id="([a-z]+)"', r'<section id="old-\1"', sec, count=1)

archive = f"""<section id="archive">
    <div class="sechead">
      <div class="eyebrow">Archive · methods and experiments this page used to lead with</div>
      <h2>What was replaced, and what was tried</h2>
      <p>Kept for the numbers and the reasoning: each of these was measured, and several obvious-looking ideas in them are
        worse than what ships. Open one to read it.</p>
    </div>
    {arch("Dead ends, and the problems open before this round", "the table of what was tried and dropped, now with this round's", demote(open_old))}
    {arch("Where the error in moves from video came from", "the first measurement of the old pipeline, stage by stage - replaced by the fit on the video's own body and the refine stage",
          '<section id="old-errsrc">' + error_source + '</section>')}
    {arch("Bending the nearest capture toward the video: how the fit was built", "the self-test, the steps that made it, and solving it into bones - the fit itself is still a step", demote(fit))}
    {arch("Does a turntable video make the model more accurate?", "an experiment: not worth 30 minutes of H3 per character", demote(views))}
    {arch("Krea 2 against Qwen-Image 2.1", "the comparison that made Qwen-Image 2.1 the default image model", demote(images))}
  </section>"""

footer = """<footer>
    <div>Built from <code>charforge/</code> on 24-25 September 2026. Numbers from each character's logs in <code>work/&lt;name&gt;/logs/</code> (<code>arms</code>, <code>retopo</code>, <code>texture</code>, <code>rig</code>), <code>work/&lt;name&gt;/motion_videos/*_fidelity.json</code> and <code>refine.json</code>, <code>results/v3/</code>, and the renders from <code>blender/render_poses.py</code>; the review of this round is in <code>REVIEW.md</code>, "the polish pass".</div>
    <div>Every figure on this page was measured by a script in the repository; nothing is estimated except the time per clip, which varies with what else the Mac is doing.</div>
  </footer>"""

head = old[:old.index("<style>")].replace("<title>CharForge Lab Book</title>", "<title>CharForge Lab Notebook</title>")
head = head.replace('<meta charset="utf-8">\n', "")
# a whole page for GitHub Pages (the artifact host used to add the document shell itself)
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n' + head)
banner = ("""<div class="nbbanner"><div class="in">This is the <b>lab notebook</b>: the chronological record of every
  experiment, kept as it was written. The curated account - method, evaluation, references - is the
  <a href="./">CharForge paper</a>; open work is in the <a href="https://github.com/majieddd/charforge/blob/main/research/TRACKER.md">experiment tracker</a>.</div></div>""")
style = style.replace("</style>", """  .nbbanner{background:var(--panel2);border-bottom:1px solid var(--line);font-size:13.5px;color:var(--ink2)}
  .nbbanner .in{max-width:1100px;margin:0 auto;padding:9px 20px}
</style>""")
page = (head + style + "\n</head>\n<body>\n" + nav + "\n" + banner + '\n\n<div class="wrap">\n  ' + opener + "\n\n  " + studio + "\n\n  " + prompts_sec + "\n\n  " + pipeline
        + "\n\n  " + polish + "\n\n  " + cast + "\n\n  " + moves + "\n\n  " + angles_sec + "\n\n  " + fid + "\n\n  " + hands + "\n\n  " + mac
        + "\n\n  " + left + "\n\n  " + archive + "\n\n  " + footer + "\n</div>\n</body>\n</html>\n")
page = page.replace('<meta name="description" content="Workflow, measurements and side-by-side comparisons from the video-to-motion and image-model tests.">',
                    '<meta name="description" content="How CharForge makes characters now: the Studio, the polish pass, new characters, moves from video - with the replaced methods archived.">')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(page)
print("wrote", OUT, len(page), "bytes")
