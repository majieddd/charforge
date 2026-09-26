// CharForge Studio - the page. Talks to studio/server.py; no build step.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const $ = (sel, el = document) => el.querySelector(sel);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const view = $('#view');
let META = null;

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const ct = r.headers.get('content-type') || '';
  const body = ct.includes('json') ? await r.json() : await r.text();
  if (!r.ok) throw new Error(body.error || body || r.statusText);
  return body;
}
const post = (path, data) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });
// a rough duration, the way a person would say it
const about = (s) => { const m = Math.round(s / 60); return s < 90 ? 'about a minute' : m < 60 ? `about ${m} min` : `about ${Math.floor(m / 60)} h${m % 60 ? ` ${m % 60} min` : ''}`; };
const mins = (s) => s < 60 ? `${Math.round(s)} s` : s < 3600 ? `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}` : `${(s / 3600).toFixed(1)} h`;
const STYLE_TXT = { realistic: 'Photoreal skin, hair and fabric', anime: 'Cel-shaded, drawn eyes', stylized: 'Chunky cartoon proportions' };

// ---- routing ---------------------------------------------------------------------------------
const routes = [
  [/^#?\/?$/, chars, 'chars'], [/^#\/new$/, newChar, 'new'], [/^#\/jobs$/, jobs, 'jobs'],
  [/^#\/setup$/, setup, 'setup'], [/^#\/c\/([a-z0-9_]+)$/, charPage, 'chars'],
];
let leave = null;
async function route() {
  if (leave) { leave(); leave = null; }
  const h = location.hash || '#/';
  for (const [re, fn, nav] of routes) {
    const m = h.match(re);
    if (m) {
      document.querySelectorAll('[data-nav]').forEach((a) => a.toggleAttribute('aria-current', a.dataset.nav === nav) || a.removeAttribute('aria-current'));
      document.querySelectorAll('[data-nav]').forEach((a) => { if (a.dataset.nav === nav) a.setAttribute('aria-current', 'page'); });
      try { await fn(...m.slice(1)); } catch (e) { view.innerHTML = `<div class="empty">Something went wrong: ${esc(e.message)}</div>`; }
      return;
    }
  }
  view.innerHTML = '<div class="empty">Not found</div>';
}
addEventListener('hashchange', route);

// ---- the live job pill -----------------------------------------------------------------------
async function pill() {
  try {
    const js = await api('/api/jobs');
    const run = js.find((j) => j.status === 'running');
    const queued = js.filter((j) => j.status === 'queued');
    const q = queued.length;
    const el = $('#live');
    if (!run) { el.hidden = true; return js; }
    el.hidden = false;
    const all = (run.est_left || 0) + queued.reduce((t, j) => t + (j.est_total || 0), 0);
    el.innerHTML = `<i class="dot"></i><span>${esc(run.title)} · ${run.stage ? `${run.stage_i}/${run.stage_n} ${esc(run.stage)}` : 'starting'} · ${mins(run.elapsed)}${q ? ` · ${q} queued · all done in ${about(all)}` : ` · ${about(run.est_left || 0)} left`}</span>`;
    return js;
  } catch { return []; }
}
setInterval(pill, 3000);

// ---- characters ------------------------------------------------------------------------------
async function chars() {
  view.innerHTML = '<div class="empty">Loading…</div>';
  const list = await api('/api/characters');
  const cards = list.map((c) => {
    const pct = Math.round(100 * c.stages_done / c.stages);
    return `<a class="card" href="#/c/${c.name}">
      <div class="img">${c.thumb ? `<img src="${c.thumb}" alt="" loading="lazy">` : '<span class="ph">no picture yet</span>'}</div>
      <div class="meta">
        <div class="name"><span>${esc(c.name)}</span><span class="chip">${esc(c.style)}</span></div>
        <div class="facts">${c.height ? `<span>${Number(c.height).toFixed(2)} m</span>` : ''}${c.moves.length ? `<span>${c.moves.length} move${c.moves.length > 1 ? 's' : ''} from video</span>` : ''}${c.complete ? '<span>ready</span>' : `<span>stage ${c.stages_done}/${c.stages}</span>`}</div>
        <div class="bar ${c.complete ? 'good' : ''}"><i style="width:${c.complete ? 100 : pct}%"></i></div>
      </div></a>`;
  }).join('');
  view.innerHTML = `<div class="head"><div><h1>Characters</h1><p class="sub">${list.length} on this Mac, newest first. Each one is a folder in work/ and a package in out/.</p></div>
      <a class="btn primary" href="#/new">＋ New character</a></div>
    ${list.length ? `<div class="grid">${cards}</div>` : '<div class="empty">No characters yet. Make the first one.</div>'}`;
}

// ---- one character ---------------------------------------------------------------------------
// The angles video's own controls, under it: a browser's controls sit over the bottom of the picture,
// which is where the joint readout is, and they show exactly when the video is paused to be read.
// One frame at a time (, and .) is how an elbow's worst frame is found.
function angleControls() {
  const v = $('#a-vid'), play = $('#a-play'), seek = $('#a-seek'), time = $('#a-time');
  if (!v) return;
  const FPS = 30;
  const show = () => {
    play.textContent = v.paused ? 'Play' : 'Pause';
    if (v.duration) seek.value = String(Math.round(1000 * v.currentTime / v.duration));
    time.textContent = `${v.currentTime.toFixed(2)} s / ${(v.duration || 0).toFixed(2)} s`;
  };
  const toggle = () => { v.paused ? v.play() : v.pause(); };
  const stepBy = (n) => { v.pause(); v.currentTime = Math.max(0, Math.min((v.duration || 0) - 1e-3, v.currentTime + n / FPS)); };
  play.onclick = toggle; v.onclick = toggle;
  $('#a-prev').onclick = () => stepBy(-1); $('#a-next').onclick = () => stepBy(1);
  seek.oninput = () => { v.pause(); if (v.duration) v.currentTime = v.duration * seek.value / 1000; };
  for (const ev of ['timeupdate', 'play', 'pause', 'loadedmetadata', 'seeked']) v.addEventListener(ev, show);
  const onKey = (e) => {
    if (!document.body.contains(v)) { removeEventListener('keydown', onKey); return; }   // left the page
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName) && e.target !== seek) return;
    if (e.key === ',') stepBy(-1); else if (e.key === '.') stepBy(1);
  };
  addEventListener('keydown', onKey);
  show();
}

async function charPage(name) {
  view.innerHTML = '<div class="empty">Loading…</div>';
  const c = await api(`/api/character/${name}`);
  const man = c.manifest || {};
  const clips = (man.clips || []).length;
  view.innerHTML = `
    <div class="head"><div><h1>${esc(c.name)} <span class="chip">${esc(c.style)}</span></h1>
      <p class="sub">${c.complete ? 'Ready: packaged in out/' + esc(c.name) : `Stage ${c.stages_done} of ${c.stages} done`}</p></div>
      <div class="actions">
        ${c.web_glb ? `<a class="btn primary" href="/play/#${esc(c.name)}" target="_blank" rel="noopener">Walk around ↗</a>` : ''}
        <button class="btn" id="b-moves">Add moves from video</button>
        <button class="btn" id="b-rerun">Run again from a stage…</button>
        <button class="btn" id="b-open">Open folder</button>
        ${c.glb ? `<a class="btn" href="${c.glb}&download=1">Download .glb</a>` : ''}
        ${c.fbx ? `<a class="btn" href="${c.fbx}&download=1">Download .fbx</a>` : ''}
      </div></div>
    <div class="charpage">
      <div class="viewer" id="viewer"><div class="msg">${c.web_glb ? 'Loading the model…' : 'No model yet - it appears when the package stage has run.'}</div></div>
      <div class="side">
        <div class="panel">
          ${c.prompt ? `<p class="prompt">“${esc(c.prompt)}”</p>` : ''}
          <div class="facts2">
            <div><b>${man.height_m ? man.height_m.toFixed(2) + ' m' : '–'}</b><span>height</span></div>
            <div><b>${man.triangles ? (man.triangles / 1000).toFixed(0) + 'k' : '–'}</b><span>triangles</span></div>
            <div><b>${man.skeleton ? man.skeleton.bones : '–'}</b><span>bones</span></div>
            <div><b>${clips || '–'}</b><span>clips</span></div>
          </div>
        </div>
        ${c.move_details.length ? `<div class="panel"><h2>Moves from video</h2>${c.move_details.map(moveCard).join('')}</div>` : ''}
        <div class="panel"><div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px"><h2>Reel</h2>
            <button class="btn" id="b-reel">${c.reel ? 'Record again' : 'Record a reel'}</button></div>
          <p class="sub" style="margin:4px 0 10px">A short video of the character: idle, a wave, walking, jogging, a jump, a turn and each of its moves from video, from one camera.</p>
          ${c.reel ? `<video src="${c.reel}" muted loop playsinline autoplay controls preload="metadata" style="width:100%;border-radius:8px;display:block"></video>` : ''}
        </div>
        <div class="panel"><div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px"><h2>Poses</h2>
            <button class="btn" id="b-poses">${c.poses.length ? 'Render again' : 'Render the poses'}</button></div>
          <p class="sub" style="margin:4px 0 10px">The poses that break things - the jump's crouch, the landing, the wave, the fall, and each move from video - from the front and from behind. Look for a garment lifting with an arm, a hem wrapped round the thighs, paint of one garment on another.</p>
          ${c.poses.length ? `<div class="poses">${c.poses.map((u) => `<a href="${u}" target="_blank" rel="noopener"><img src="${u}" alt="${esc(decodeURIComponent(u.split('/').pop().split('?')[0]).replace('.png', '').replace(/_/g, ' '))}" loading="lazy"></a>`).join('')}</div>
            <p class="sub" style="margin-top:8px">Rendered ${new Date(c.poses_at * 1000).toLocaleString()}.</p>` : ''}
        </div>
        <div class="panel"><h2>Stages</h2><div class="stages">${c.stage_list.map((s) => `<div class="stage ${s.done ? 'done' : ''}" title="${esc(s.why)}">${esc(s.name)}</div>`).join('')}</div>
          ${c.logs.length ? `<p class="sub" style="margin-top:10px">Logs: ${c.logs.slice(0, 40).map((l) => `<a href="#" data-log="${esc(l)}">${esc(l.replace('.log', ''))}</a>`).join(', ')}</p>` : ''}</div>
        ${c.reference ? `<div class="panel"><h2>Reference image</h2><img src="${c.reference}" alt="The image ${esc(c.name)} was generated from" style="width:100%;border-radius:8px;display:block"></div>` : ''}
      </div>
    </div>
        <div class="panel angles-wide"><h2>Angles</h2>
          <p class="sub" style="margin:4px 0 10px">One clip from the front, the side and above at once, at one scale on a 10 cm grid, with each elbow's and knee's bend and each shoe's height read out under it - what one camera hides (an elbow folded back through the arm, a foot off the floor) shows here. A move from video can have its video beside it.</p>
          <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px">
            <select id="a-clip">${(c.clip_names || []).map((n) => `<option value="${esc(n)}" ${c.video_moves.includes(n) ? 'selected' : ''}>${esc(n.replace(/_/g, ' '))}${c.video_moves.includes(n) ? ' (from video)' : ''}</option>`).join('')}</select>
            <label class="check" style="display:flex;gap:6px;align-items:center;font-size:13px"><input type="checkbox" id="a-video" checked> with its video</label>
            <button class="btn" id="b-angles">Render</button></div>
          ${c.angles.length ? `<select id="a-pick" style="margin-bottom:8px">${c.angles.map((x, i) => `<option value="${i}">${esc(x.clip.replace(/_/g, ' '))}${x.video ? ', with its video' : ''} - ${new Date(x.at * 1000).toLocaleString()}</option>`).join('')}</select>
            <video id="a-vid" src="${c.angles[0].url}" muted loop playsinline autoplay preload="metadata" style="width:100%;border-radius:8px;display:block;cursor:pointer" title="Click to pause or play"></video>
            <div class="vctl"><button class="btn" id="a-play">Pause</button>
              <button class="btn" id="a-prev" title="One frame back (,)" aria-label="One frame back">‹ frame</button>
              <button class="btn" id="a-next" title="One frame on (.)" aria-label="One frame on">frame ›</button>
              <input type="range" id="a-seek" min="0" max="1000" value="0" aria-label="Position in the clip">
              <span class="mono" id="a-time">0.00 s</span></div>
            <p class="sub" id="a-note" style="margin-top:6px"></p>` : ''}
        </div>`;
  $('#b-moves').onclick = () => movesDialog(c);
  $('#b-rerun').onclick = () => rerunDialog(c);
  $('#b-reel').onclick = async () => {
    try { await post('/api/jobs', { kind: 'reel', name: c.name }); location.hash = '#/jobs'; }
    catch (e) { alert(e.message); }
  };
  const aNote = (x) => { const n = $('#a-note'); if (n && x) n.textContent = `Most bent: elbow ${x.worst_elbow != null ? Math.round(x.worst_elbow) + '°' : '-'}, knee ${x.worst_knee != null ? Math.round(x.worst_knee) + '°' : '-'} (the motion library's elbows reach 159° at most).`; };
  if (c.angles.length) { aNote(c.angles[0]); $('#a-pick').onchange = (e) => { const x = c.angles[+e.target.value]; $('#a-vid').src = x.url; aNote(x); }; angleControls(); }
  const aSync = () => { const v = $('#a-clip'); if (v) { const isv = c.video_moves.includes(v.value); $('#a-video').disabled = !isv; if (!isv) $('#a-video').checked = false; } };
  if ($('#a-clip')) { $('#a-clip').onchange = aSync; aSync(); }
  $('#b-angles').onclick = async () => {
    try { await post('/api/jobs', { kind: 'angles', name: c.name, clip: $('#a-clip').value, video: $('#a-video').checked }); location.hash = '#/jobs'; }
    catch (e) { alert(e.message); }
  };
  $('#b-poses').onclick = async () => {
    try { await post('/api/jobs', { kind: 'poses', name: c.name }); location.hash = '#/jobs'; }
    catch (e) { alert(e.message); }
  };
  $('#b-open').onclick = () => post(`/api/character/${c.name}/reveal`, {});
  view.querySelectorAll('[data-log]').forEach((a) => a.onclick = async (e) => {
    e.preventDefault();
    const t = await api(`/api/character/${c.name}/log/${a.dataset.log}`);
    modal(`<h2>${esc(a.dataset.log)}</h2><pre class="log">${esc(t) || '(empty)'}</pre><div class="actions" style="margin-top:12px"><button class="btn" data-close>Close</button></div>`, true);
  });
  if (c.web_glb) leave = viewer($('#viewer'), c.web_glb, man);
}

function moveCard(m) {
  const f = m.fidelity;
  const nums = f ? [
    f.joint_error_torso != null && `joints ${f.joint_error_torso.toFixed(3)}`,
    f.silhouette_iou != null && `IoU ${f.silhouette_iou.toFixed(3)}`,
    f.hand_error_palm != null && `hands ${f.hand_error_palm.toFixed(2)}`,
    f.palm_facing_agree != null && `palms ${Math.round(f.palm_facing_agree * 100)}%`,
  ].filter(Boolean) : [];
  return `<div class="move"><div style="display:flex;justify-content:space-between;gap:8px;align-items:baseline">
      <b>${esc(m.name.replace(/_/g, ' '))}</b><span class="muted" style="font-size:12px">${m.performer !== undefined ? 'video of ' + esc(m.performer) : ''}</span></div>
    ${m.compare || m.video ? `<video src="${m.compare || m.video}" muted loop playsinline autoplay controls preload="auto"></video>` : ''}
    ${nums.length ? `<div class="nums">${nums.map(esc).join('<span class="muted">·</span>')}</div>` : ''}</div>`;
}

// the 3D viewer: the web build, every clip
function viewer(el, src, man) {
  el.innerHTML = '';
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  const pm = new THREE.PMREMGenerator(renderer);
  scene.environment = pm.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.add(new THREE.HemisphereLight(0xdfe8f5, 0x3a3226, 0.6));
  const key = new THREE.DirectionalLight(0xfff3e0, 2.2); key.position.set(2, 4, 3); scene.add(key);
  const grid = new THREE.GridHelper(6, 24, 0x5b6b82, 0x33404f); grid.material.transparent = true; grid.material.opacity = 0.35; scene.add(grid);
  const cam = new THREE.PerspectiveCamera(35, 1, 0.05, 100);
  const ctl = new OrbitControls(cam, renderer.domElement);
  ctl.enableDamping = true; ctl.target.set(0, 0.9, 0); cam.position.set(1.4, 1.35, 3.2);
  const bar = document.createElement('div'); bar.className = 'ctl';
  bar.innerHTML = `<select aria-label="Clip"></select><button class="btn" data-a="play">Pause</button>
    <select aria-label="Speed" class="speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1" selected>1×</option></select>
    <button class="btn" data-a="spin">Turn</button><button class="btn" data-a="reset">Reset view</button><span class="muted" style="font-size:12px;margin-left:auto" id="vt"></span>`;
  el.appendChild(bar);
  const [clipSel, speedSel] = bar.querySelectorAll('select');
  let mixer = null, action = null, root = null, playing = true, spin = false, stop = false;
  const H = man.height_m || 1.75;
  const frame = () => { ctl.target.set(0, H * 0.44, 0); cam.position.set(H * 0.95, H * 0.72, H * 2.3); ctl.update(); };
  const clock = new THREE.Clock();
  new GLTFLoader().load(src, (g) => {
    root = g.scene; scene.add(root);
    root.traverse((o) => { if (o.isMesh) { o.frustumCulled = false; } });
    mixer = new THREE.AnimationMixer(root);
    const names = g.animations.map((a) => a.name).sort((a, b) => (a === 'idle' ? -1 : b === 'idle' ? 1 : a.localeCompare(b)));
    const extra = new Set((man.clips || []).filter((c) => !['idle', 'walk', 'jog', 'run', 'sprint', 'walk_back', 'jog_back', 'strafe_left', 'strafe_right', 'turn_left', 'turn_right', 'turn_180', 'crouch_idle', 'crouch_walk', 'jump', 'fall', 'land', 'wave'].includes(c.name)).map((c) => c.name));
    clipSel.innerHTML = names.map((n) => `<option value="${esc(n)}">${esc(n.replace(/_/g, ' '))}${extra.has(n) ? ' ★' : ''}</option>`).join('');
    const play = (n) => {
      const clip = g.animations.find((a) => a.name === n);
      if (!clip) return;
      if (action) action.stop();
      action = mixer.clipAction(clip); action.reset().play(); playing = true; bar.querySelector('[data-a=play]').textContent = 'Pause';
    };
    clipSel.onchange = () => play(clipSel.value);
    play(names.includes('idle') ? 'idle' : names[0]);
    // the manifest's height: a compressed web build's bounds do not survive into three's skinned-mesh box
    frame();
    el.querySelector('.msg')?.remove();
  }, undefined, (e) => { el.insertAdjacentHTML('beforeend', `<div class="msg">Could not load the model: ${esc(e.message || e)}</div>`); });
  bar.onclick = (e) => {
    const a = e.target.closest('[data-a]')?.dataset.a;
    if (a === 'play') { playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; }
    if (a === 'spin') { spin = !spin; e.target.setAttribute('aria-pressed', spin); }
    if (a === 'reset' && root) frame();
  };
  const vt = bar.querySelector('#vt');
  const fit = () => { const r = el.getBoundingClientRect(); renderer.setSize(r.width, r.height, false); cam.aspect = r.width / Math.max(r.height, 1); cam.updateProjectionMatrix(); };
  const ro = new ResizeObserver(fit); ro.observe(el); fit();
  const loop = () => {
    if (stop) return;
    requestAnimationFrame(loop);
    const dt = Math.min(clock.getDelta(), 0.05);
    if (mixer && playing) mixer.update(dt * Number(speedSel.value));
    if (spin && root) root.rotation.y += dt * 0.6;
    if (action) vt.textContent = `${action.time.toFixed(2)} / ${action.getClip().duration.toFixed(2)} s`;
    ctl.update(); renderer.render(scene, cam);
  };
  loop();
  return () => { stop = true; ro.disconnect(); renderer.dispose(); };
}

// ---- dialogs ---------------------------------------------------------------------------------
function modal(html, wide = false) {
  const m = $('#modal'); const box = $('.box', m);
  box.className = 'box' + (wide ? ' wide' : ''); box.innerHTML = html; m.hidden = false;
  const close = () => { m.hidden = true; box.innerHTML = ''; };
  m.onclick = (e) => { if (e.target === m || e.target.closest('[data-close]')) close(); };
  addEventListener('keydown', function k(e) { if (e.key === 'Escape') { close(); removeEventListener('keydown', k); } });
  return { box, close };
}

async function movesDialog(c) {
  META = META || await api('/api/meta');
  const all = await api('/api/characters');
  const moves = Object.entries(META.moves);
  const { box, close } = modal(`<h2>Add moves from video</h2>
    <p class="sub">Each move is a 5-second video of ${esc(c.name)} doing it, made by MiniMax H3 in ComfyUI (about 35 minutes each on an M5), then read, fitted, rigged and refined against the video. Or borrow another character's video of the move: minutes, not half an hour.</p>
    <div class="checks">${moves.map(([k, m]) => `<label><input type="checkbox" value="${esc(k)}" ${c.moves.includes(k) ? 'disabled' : ''}><div><b>${esc(k.replace(/_/g, ' '))}${c.moves.includes(k) ? ' <span class="chip">has it</span>' : ''}</b><br><span>${esc(m.summary || m.prompt || '')}</span></div></label>`).join('')}</div>
    <div class="field"><label for="perf">Whose video</label><select id="perf"><option value="">${esc(c.name)}'s own (new video)</option>${all.filter((x) => x.name !== c.name).map((x) => `<option value="${x.name}">${esc(x.name)}'s (reused if it has one)</option>`).join('')}</select></div>
    <div class="actions"><button class="btn primary" id="go">Start</button><button class="btn" data-close>Cancel</button></div><div class="err" id="err"></div>`);
  $('#go', box).onclick = async () => {
    const picked = [...box.querySelectorAll('input:checked')].map((i) => i.value);
    try {
      await post('/api/jobs', { kind: 'move', name: c.name, moves: picked, performer: $('#perf', box).value || undefined });
      close(); location.hash = '#/jobs';
    } catch (e) { $('#err', box).textContent = e.message; }
  };
}

async function rerunDialog(c) {
  META = META || await api('/api/meta');
  const { box, close } = modal(`<h2>Run ${esc(c.name)} again from a stage</h2>
    <p class="sub">Every stage from the one you pick onward runs again; the ones before it are kept. Use it after changing the pipeline, or to retry a stage that failed.</p>
    <div class="field"><select id="st">${META.stages.map((s, i) => `<option value="${s.name}">${i + 1}. ${esc(s.name)} - ${esc(s.why)}</option>`).join('')}</select></div>
    <div class="actions"><button class="btn primary" id="go">Run</button><button class="btn" data-close>Cancel</button></div><div class="err" id="err"></div>`);
  $('#go', box).onclick = async () => {
    try { await post('/api/jobs', { kind: 'rerun', name: c.name, from: $('#st', box).value }); close(); location.hash = '#/jobs'; }
    catch (e) { $('#err', box).textContent = e.message; }
  };
}

// ---- a new character -------------------------------------------------------------------------
async function newChar() {
  META = META || await api('/api/meta');
  view.innerHTML = `<div class="head"><div><h1>New character</h1><p class="sub">Describe one, or give a picture of one - full body, standing. It runs on this Mac; nothing is uploaded anywhere.</p></div></div>
  <div class="form">
    <div class="panel">
      <div class="tabs" role="tablist"><button role="tab" aria-selected="true" data-t="prompt">From a description</button><button role="tab" aria-selected="false" data-t="image">From an image</button></div>
      <div data-p="prompt"><div class="field"><label for="prompt">Describe the character</label>
        <textarea id="prompt" placeholder="a boy scout, a muscular grey alien, a pirate captain in a red coat…"></textarea>
        <span class="hint">A few words is enough. The pipeline asks for a full-body A-pose itself.</span></div>
        <div class="field"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
          <button class="btn" id="b-desc" type="button">Write it out</button>
          <label class="check" style="display:flex;gap:6px;align-items:center;font-size:13px"><input type="checkbox" id="literal"> use my words exactly as written</label></div>
          <span class="hint">Writing it out settles what a short prompt leaves open - age and build, skin and hair, each garment with its own colour - so the picture comes out dressed, coloured and the right size. It runs on this Mac (a local model through Ollama); you can edit the result. Skipped, it is done anyway when the job starts.</span></div>
        <div class="field" id="desc-f" hidden><label for="desc">What the image model is asked for <span class="muted" id="desc-src"></span></label>
          <textarea id="desc" rows="4"></textarea></div></div>
      <div data-p="image" hidden><div class="field"><label>Image</label>
        <div class="drop" id="drop" tabindex="0">Drop a full-body picture here, or click to choose one<br><span class="hint">PNG or JPEG; one character, whole body, plain background is best</span></div>
        <input type="file" id="file" accept="image/png,image/jpeg,image/webp" hidden></div></div>
      <div class="field"><label for="name">Name</label><input type="text" id="name" placeholder="scout" maxlength="32" autocomplete="off">
        <span class="hint" id="name-hint">Lowercase letters, digits and _. It names the folders work/&lt;name&gt; and out/&lt;name&gt;.</span></div>
      <div class="field"><label>Style</label><div class="seg">${META.styles.map((s, i) => `<label><input type="radio" name="style" value="${s}" ${i === 1 ? 'checked' : ''}><b>${s}</b><span>${STYLE_TXT[s] || ''}</span></label>`).join('')}</div></div>
      <div class="field"><label for="height">Height</label>
        <label class="check" style="display:flex;gap:6px;align-items:center;font-size:13px;margin-bottom:6px"><input type="checkbox" id="hset"> set it myself</label>
        <div class="range"><input type="range" id="height" min="1.0" max="2.3" step="0.01" value="1.75" disabled><output id="hout">from the description</output></div>
        <span class="hint">Left alone, the height follows the description: a child 1.2-1.5 m, an adult about 1.75.</span></div>
      <details class="adv"><summary>More options</summary>
        <div class="field"><label for="imodel">Image model (for a description)</label><select id="imodel"><option value="qwen21">Qwen-Image 2.1 - best cut-outs; research-only licence</option><option value="krea2">Krea 2 - commercial use</option></select></div>
        <div class="field"><label for="quality">Quality</label><select id="quality"><option value="best">Best - a second 3D pass from repainted side and back views</option><option value="fast">Fast - one pass</option></select></div>
        <div class="field"><label for="seed">Seed</label><input type="number" id="seed" value="7"></div>
      </details>
    </div>
    <div class="panel">
      <h2>What happens</h2>
      <ol class="steps">
        <li>A reference image is drawn (or yours is used), then turned into a textured 3D model by TRELLIS.2.</li>
        <li>The model is closed into a solid, its arms freed from the body, modelled hands with finger bones put on, retopologised and textured.</li>
        <li>It is rigged with a Mixamo skeleton, skinned, given a face rig, and 18 clips are retargeted onto it.</li>
        <li>Packaged for engines: glTF, FBX with LODs, textures and a manifest - and a web build you can turn round here.</li>
      </ol>
      <p class="sub" style="margin:14px 0">About 40 minutes on an M5 with 24 GB. One job runs at a time; more wait in the queue. You can close this page - the job keeps going.</p>
      <button class="btn primary" id="start" style="width:100%;justify-content:center;padding:11px">Make it</button>
      <div class="err" id="err"></div>
    </div>
  </div>`;
  let tab = 'prompt', upload = null;
  view.querySelectorAll('[data-t]').forEach((b) => b.onclick = () => {
    tab = b.dataset.t;
    view.querySelectorAll('[data-t]').forEach((x) => x.setAttribute('aria-selected', x === b));
    view.querySelectorAll('[data-p]').forEach((p) => p.hidden = p.dataset.p !== tab);
  });
  const h = $('#height'), hset = $('#hset');
  let suggested = null;
  const hshow = () => { $('#hout').textContent = hset.checked ? `${Number(h.value).toFixed(2)} m` : (suggested ? `${suggested.toFixed(2)} m, from the description` : 'from the description'); };
  h.oninput = hshow;
  hset.onchange = () => { h.disabled = !hset.checked; hshow(); };
  // names already taken: a new prompt under one of them used to give back the old character
  const taken = new Set((await api('/api/characters')).map((c) => c.name));
  const free = (n) => { if (!taken.has(n)) return n; for (let k = 2; ; k++) if (!taken.has(`${n}${k}`)) return `${n}${k}`; };
  const checkName = () => {
    const n = $('#name').value.trim().toLowerCase(), bad = taken.has(n);
    $('#name-hint').innerHTML = bad ? `<b style="color:var(--bad,#e5484d)">There is already a character named ${esc(n)}.</b> Pick another name - ${esc(free(n))} is free - or run ${esc(n)} again from its own page.`
      : 'Lowercase letters, digits and _. It names the folders work/&lt;name&gt; and out/&lt;name&gt;.';
    $('#start').disabled = bad;
  };
  $('#prompt').oninput = () => {
    $('#desc-f').hidden = true; $('#desc').value = '';                       // written out from an older prompt
    if ($('#name').dataset.touched) return;
    const w = $('#prompt').value.toLowerCase().match(/[a-z]+/g) || [];
    const skip = new Set(['a', 'an', 'the', 'young', 'old', 'with', 'in', 'and', 'of', 'cheerful', 'tall', 'short', 'little', 'big', 'that', 'is']);
    $('#name').value = free((w.find((x) => !skip.has(x) && x.length > 2) || '').slice(0, 20));
    checkName();
  };
  $('#name').oninput = () => { $('#name').dataset.touched = 1; checkName(); };
  $('#literal').onchange = () => { $('#b-desc').disabled = $('#literal').checked; if ($('#literal').checked) $('#desc-f').hidden = true; };
  $('#b-desc').onclick = async () => {
    const prompt = $('#prompt').value.trim();
    if (!prompt) { $('#err').textContent = 'Describe the character first.'; return; }
    const b = $('#b-desc'); b.disabled = true; b.textContent = 'Writing it out…'; $('#err').textContent = '';
    try {
      const d = await post('/api/describe', { prompt, style: view.querySelector('input[name=style]:checked').value });
      $('#desc').value = d.description; $('#desc-f').hidden = false;
      $('#desc-src').textContent = `(${d.engine}; ${d.age}) - edit it as you like`;
      suggested = d.height_m; if (!hset.checked) h.value = d.height_m; hshow();
    } catch (e) { $('#err').textContent = e.message; }
    b.disabled = false; b.textContent = 'Write it out';
  };
  const drop = $('#drop'), file = $('#file');
  const take = async (f) => {
    if (!f) return;
    drop.innerHTML = 'Uploading…';
    const r = await fetch('/api/upload', { method: 'POST', headers: { 'Content-Type': f.type || 'image/png' }, body: f });
    const j = await r.json(); upload = j.upload;
    drop.innerHTML = `<img src="${URL.createObjectURL(f)}" alt="">${esc(f.name)} - click to choose another`;
    if (!$('#name').value) $('#name').value = f.name.toLowerCase().replace(/\.[a-z]+$/, '').replace(/[^a-z0-9_]/g, '_').slice(0, 20);
  };
  drop.onclick = () => file.click(); drop.onkeydown = (e) => { if (e.key === 'Enter') file.click(); };
  file.onchange = () => take(file.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove('over'); take(e.dataTransfer.files[0]); };
  $('#start').onclick = async () => {
    const body = { kind: 'make', name: $('#name').value.trim().toLowerCase(), style: view.querySelector('input[name=style]:checked').value,
      image_model: $('#imodel').value, quality: $('#quality').value, seed: $('#seed').value };
    if (hset.checked) body.height = Number(h.value);
    else if (suggested) body.height = suggested;
    if (tab === 'prompt') {
      body.prompt = $('#prompt').value;
      if ($('#literal').checked) body.literal = true;
      else if (!$('#desc-f').hidden && $('#desc').value.trim()) body.description = $('#desc').value.trim();
    } else body.image = upload;
    $('#err').textContent = '';
    try { await post('/api/jobs', body); location.hash = '#/jobs'; } catch (e) { $('#err').textContent = e.message; }
  };
}

// ---- jobs ------------------------------------------------------------------------------------
async function jobs() {
  view.innerHTML = `<div class="head"><div><h1>Jobs</h1><p class="sub">One runs at a time; the rest wait. Logs are kept in work/_studio/.</p></div>
    <div class="actions"><button class="btn" id="b-clear">Clear finished</button></div></div><div class="jobs" id="jl"></div>`;
  $('#b-clear').onclick = async () => { await post('/api/jobs/clear', {}); $('#jl').innerHTML = ''; draw(); };
  const open = new Set(), offs = {};
  const draw = async () => {
    const all = await pill();
    const jl = $('#jl'); if (!jl) return;
    // in the order they run: the running one, then the queue oldest first, then the finished newest first
    const rank = { running: 0, queued: 1 };
    const js = [...all].sort((a, b) => (rank[a.status] ?? 2) - (rank[b.status] ?? 2)
      || (a.status === 'queued' ? a.created - b.created : (b.ended || b.created) - (a.ended || a.created)));
    jl.replaceChildren(...js.map((j) => document.getElementById('j' + j.id)).filter(Boolean));
    if (!js.length) { jl.innerHTML = '<div class="empty">No jobs yet. <a href="#/new">Make a character</a>.</div>'; return; }
    for (const j of js) {
      let el = document.getElementById('j' + j.id);
      if (!el) {
        el = document.createElement('div'); el.className = 'job'; el.id = 'j' + j.id;
        el.innerHTML = `<div class="row"><span class="title"></span><span class="badge"></span><span class="muted el"></span>
          <button class="btn" data-a="log">Log</button><button class="btn danger" data-a="cancel">Stop</button><a class="btn" data-a="see" href="#/c/${j.name}">Open</a></div>
          <div class="stageline"><span class="st"></span><span class="muted pc"></span></div><div class="bar"><i></i></div><div class="last"></div><pre class="log" hidden></pre>`;
        el.onclick = async (e) => {
          const a = e.target.closest('[data-a]')?.dataset.a;
          if (a === 'log') { open.has(j.id) ? open.delete(j.id) : open.add(j.id); el.querySelector('pre').hidden = !open.has(j.id); offs[j.id] = 0; el.querySelector('pre').textContent = ''; draw(); }
          if (a === 'cancel') { await post(`/api/jobs/${j.id}/cancel`, {}); draw(); }
        };
        jl.appendChild(el);
      }
      el.querySelector('.title').textContent = j.title;
      const b = el.querySelector('.badge'); b.className = 'badge ' + j.status; b.textContent = j.status;
      el.querySelector('.el').textContent = j.started ? mins(j.elapsed) : '';
      el.querySelector('[data-a=cancel]').hidden = !['running', 'queued'].includes(j.status);
      const frac = j.status === 'done' ? 1 : j.stage_n ? (j.stage_i - (j.status === 'running' ? 0.5 : 0)) / j.stage_n : 0;
      el.querySelector('.bar i').style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
      el.querySelector('.bar').className = 'bar' + (j.status === 'done' ? ' good' : '');
      el.querySelector('.st').textContent = j.stage && j.status !== 'queued' ? `stage ${j.stage_i} of ${j.stage_n}: ${j.stage}`
        : (j.status === 'queued' ? `starts in ${about(j.starts_in || 0)} · takes ${about(j.est_total || 0)}` : '');
      el.querySelector('.pc').textContent = j.stage_n && j.status === 'running' ? `${Math.round(frac * 100)}% · ${about(j.est_left || 0)} left` : '';
      el.querySelector('.last').textContent = j.last || '';
      if (open.has(j.id)) {
        const r = await api(`/api/jobs/${j.id}/log?offset=${offs[j.id] || 0}`);
        const pre = el.querySelector('pre');
        if (r.text) { pre.textContent = (offs[j.id] ? pre.textContent : '') + r.text; pre.scrollTop = pre.scrollHeight; }
        offs[j.id] = r.offset;
      }
    }
  };
  await draw();
  const t = setInterval(draw, 2500);
  leave = () => clearInterval(t);
}

// ---- setup -----------------------------------------------------------------------------------
async function setup() {
  view.innerHTML = `<div class="head"><div><h1>Setup</h1><p class="sub">What this Mac has for each part of the pipeline (tools/check_env.py). Anything missing shows up here before a long run fails on it.</p></div>
    <button class="btn" id="re">Check again</button></div>
    <div class="panel" id="pm" style="margin-bottom:14px"><h2>Writing prompts out</h2>
      <p class="sub" style="margin:4px 0 10px">A short prompt is written out by a local language model before the picture is drawn - age and build, skin and hair, each garment with its colour. Pick which one (installed in Ollama), or give a server with the OpenAI chat API on this Mac (LM Studio: <code>http://127.0.0.1:1234/v1</code>), then try it on a prompt.</p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <select id="pm-model"></select>
        <input type="text" id="pm-url" placeholder="optional: http://127.0.0.1:1234/v1" style="min-width:260px">
        <button class="btn" id="pm-save">Save</button></div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px">
        <input type="text" id="pm-try" value="a boy scout" style="min-width:260px"><button class="btn" id="pm-go">Try it</button></div>
      <p class="sub" id="pm-out" style="margin-top:8px"></p></div>
    <div class="panel" id="env"><div class="empty">Checking… (up to a minute)</div></div>
    <div class="panel" style="margin-top:14px"><h2>Running it yourself</h2><ol class="steps">
      <li>Start the studio: <code>python charforge.py studio</code> - or double-click <b>CharForge Studio.command</b> in the charforge folder.</li>
      <li>It opens <code>http://localhost:8830</code>. Only this Mac can reach it.</li>
      <li>Make characters from <a href="#/new">New character</a>; watch them in <a href="#/jobs">Jobs</a>; turn them round and play their clips on each character's page.</li>
      <li>Close the terminal window to stop it. A job that was running resumes from its last finished stage when started again.</li></ol></div>`;
  const load = async (refresh) => {
    const r = await api('/api/env' + (refresh ? '?refresh=1' : ''));
    $('#env').innerHTML = r.rows && r.rows.length ? `<table class="env">${r.rows.map((x) => `<tr><td><span class="st ${x.state}">${x.state}</span></td><td>${esc(x.what)}</td><td class="muted">${esc(x.detail)}</td></tr>`).join('')}</table>`
      : `<pre class="log">${esc(r.text)}</pre>`;
  };
  $('#re').onclick = () => { $('#env').innerHTML = '<div class="empty">Checking…</div>'; load(true); };
  load(false);
  const pm = async () => {
    const st = await api('/api/settings');
    const cur = (st.describe_model || '').split(',')[0];
    $('#pm-model').innerHTML = `<option value="">default (${esc(st.default)})</option>` +
      st.installed.map((m) => `<option value="${esc(m)}" ${m === cur ? 'selected' : ''}>${esc(m)}</option>`).join('');
    $('#pm-url').value = st.describe_url || '';
    $('#pm-out').textContent = `Using ${st.using.join(', ')}${st.using_url ? ' at ' + st.using_url : ' (Ollama)'}${st.env ? ' - set by CF_DESCRIBE_MODEL, which wins over this' : ''}.`;
  };
  $('#pm-save').onclick = async () => {
    try { await post('/api/settings', { describe_model: $('#pm-model').value, describe_url: $('#pm-url').value.trim() }); await pm(); }
    catch (e) { $('#pm-out').textContent = e.message; }
  };
  $('#pm-go').onclick = async () => {
    const b = $('#pm-go'); b.disabled = true; $('#pm-out').textContent = 'Writing it out…';
    try { const d = await post('/api/describe', { prompt: $('#pm-try').value, style: 'stylized' });
      $('#pm-out').textContent = `${d.engine} · ${d.age}, ${d.height_m.toFixed(2)} m - ${d.description}`; }
    catch (e) { $('#pm-out').textContent = e.message; }
    b.disabled = false;
  };
  pm();
}

pill();
route();
