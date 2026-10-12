'use strict';

const $ = (sel) => document.querySelector(sel);
const stage = $('#stage');
const frame = $('#frame');
const grid = $('#grid');
const strip = $('#strip');
const enc = encodeURIComponent;
const dpr = () => window.devicePixelRatio || 1;

const FILTERS = ['all', 'unmarked', 'keep', 'reject'];
const FILTER_LABEL = { all: 'All photos', unmarked: 'Unmarked', keep: 'Keepers', reject: 'Rejects' };
const AHEAD = 5;
const BEHIND = 2;
const AHEAD_ZOOMED = 2; // full-size images are ~100 MB each once decoded
const BEHIND_ZOOMED = 1;

const S = {
  folder: '',
  version: '',
  photos: [], // all names, in shooting order
  order: new Map(), // name -> index in photos
  marks: {}, // name -> 'keep' | 'reject'
  list: [], // names in the current filter, frozen until the filter changes
  pos: -1, // position in list
  lastName: null,
  filter: 'all',
  auto: true,
  strip: true,
  view: 'loupe',
  zoom: false,
  zx: 0.5, // zoom centre, as a fraction of the image
  zy: 0.5,
  clip: false, // raw clipping overlay
  mode: 'cull', // or 'edit' (editor.js)
};
const current = () => (S.pos >= 0 ? S.list[S.pos] : null);

// --- Image loading ---------------------------------------------------------
// The photos you're about to see are fetched and *decoded* ahead of time, and
// the decoded <img> element itself is moved into the stage, so stepping to the
// next photo never waits on the network or on JPEG decoding.

const cache = new Map(); // url -> { img, ready, failed, promise }

function fitSide() {
  // Long side of a ~3:2 photo fitted to the stage, in device pixels. The server
  // sends the smallest embedded JPEG that covers it (1620px or full-size on a D3500).
  const w = stage.clientWidth;
  const h = stage.clientHeight;
  const side = Math.max(Math.min(w, h * 1.5), Math.min(h, w * 1.5)) * dpr();
  return Math.ceil(side / 256) * 256;
}

const imgUrl = (name, full) => `/img/${enc(name)}?w=${full ? 0 : fitSide()}&v=${S.version}`;
const thumbUrl = (name) => `/thumb/${enc(name)}?v=${S.version}`;

function load(url, priority) {
  let e = cache.get(url);
  if (e) return e;
  const img = new Image();
  img.fetchPriority = priority;
  img.draggable = false;
  img.src = url;
  e = { img, ready: false, failed: false };
  e.promise = img.decode().then(
    () => { e.ready = true; },
    () => {
      // Chrome also rejects decode() when its decoded-image cache is full, though the
      // image loaded fine. It's still displayable; it just decodes on first paint.
      if (img.naturalWidth > 0) e.ready = true;
      else e.failed = true;
    },
  );
  cache.set(url, e);
  return e;
}

function prefetch() {
  const keep = new Set();
  const want = (pos, full, priority) => {
    if (pos < 0 || pos >= S.list.length) return;
    const url = imgUrl(S.list[pos], full);
    keep.add(url);
    load(url, priority);
    meta(S.list[pos]);
  };
  want(S.pos, false, 'high');
  if (S.view === 'loupe') {
    if (S.zoom) want(S.pos, true, 'high');
    for (let d = 1; d <= AHEAD; d++) want(S.pos + d, false, 'low');
    for (let d = 1; d <= BEHIND; d++) want(S.pos - d, false, 'low');
    if (S.zoom) {
      for (let d = 1; d <= AHEAD_ZOOMED; d++) want(S.pos + d, true, 'low');
      for (let d = 1; d <= BEHIND_ZOOMED; d++) want(S.pos - d, true, 'low');
    }
  }
  for (const [url, e] of cache) {
    if (keep.has(url)) continue;
    if (!e.ready) e.img.src = ''; // cancels the download
    cache.delete(url);
  }
}

const metas = new Map(); // name -> exif summary + full-size dimensions
const metaPending = new Set();

function meta(name) {
  if (!metas.has(name) && !metaPending.has(name)) {
    metaPending.add(name);
    fetch(`/meta/${enc(name)}?v=${S.version}`)
      .then((r) => r.json())
      .then((m) => {
        metas.set(name, m);
        if (S.mode === 'edit') {
          editorMetaArrived(name);
        } else if (name === current()) {
          updateHeader();
          layout();
        }
      })
      .catch(() => {})
      .finally(() => metaPending.delete(name));
  }
  return metas.get(name);
}

const placeholders = new Map();

function placeholder(name) {
  // The thumbnail blown up, so something is on screen the instant you land on a photo.
  let img = placeholders.get(name);
  if (!img) {
    if (placeholders.size > 100) placeholders.clear();
    img = new Image();
    img.draggable = false;
    img.onload = () => { if (shown === img) layout(); };
    img.src = thumbUrl(name);
    placeholders.set(name, img);
  }
  return img;
}

// --- Raw clipping overlay ---------------------------------------------------------
// A magenta-on-transparent PNG drawn exactly over the photo. Masks need the whole raw
// decoded (~0.7 s each), so the server works on the current photo, then prepares its
// neighbours. While an arrow key is held we wait for browsing to pause, so we don't
// queue up work for every photo flown past.

const clipLayer = $('#cliplayer');
let clipShown = null; // overlay <img> in the layer
let clipWant = null; // url of the overlay the current view needs
let clipPending = null; // overlay <img> being fetched
let clipTimer;
let keyHeld = false; // last navigation came from an auto-repeating key

const clipUrl = (name) => `/clip/${enc(name)}?w=${S.zoom ? 0 : fitSide()}&v=${S.version}`;

function updateClip() {
  const name = current();
  const url = S.clip && S.view === 'loupe' && name ? clipUrl(name) : null;
  if (url === clipWant) return;
  clipWant = url;
  clearTimeout(clipTimer);
  if (clipPending) clipPending.src = '';
  clipPending = null;
  // Never leave the previous photo's mask on screen; a different size of this one is fine.
  if (clipShown && clipShown.dataset.name !== name) {
    clipLayer.replaceChildren();
    clipShown = null;
  }
  if (!url) {
    clipLayer.replaceChildren();
    clipShown = null;
    return;
  }
  clipTimer = setTimeout(() => {
    hintClips();
    const img = new Image();
    img.draggable = false;
    img.dataset.name = name;
    img.src = url;
    clipPending = img;
    img.decode().catch(() => {}).then(() => {
      if (clipWant !== url) return;
      clipPending = null;
      if (!img.naturalWidth) {
        toast(`No raw clipping data for ${name}`, true);
        return;
      }
      clipLayer.replaceChildren(img);
      clipShown = img;
      layout();
    });
  }, keyHeld ? 150 : 0);
}

function hintClips() {
  const names = [];
  for (const d of [0, 1, 2, 3, 4, 5, -1, -2]) {
    const n = S.list[S.pos + d];
    if (n) names.push(n);
  }
  fetch('/api/clip-hint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names }),
  }).catch(() => {});
}

function toggleClip() {
  S.clip = !S.clip;
  toast(S.clip ? 'Raw clipping on' : 'Raw clipping off');
  render();
  save();
}

// --- Loupe -------------------------------------------------------------------

let shown = null;

function render() {
  if (S.mode === 'edit') return;
  updateHeader();
  updateCurrentCell();
  updateClip();
  if (S.view !== 'loupe') {
    prefetch();
    return;
  }
  $('#empty').hidden = S.pos >= 0;
  const name = current();
  if (!name) {
    stage.replaceChildren();
    shown = null;
    frame.className = '';
    return;
  }
  const fit = load(imgUrl(name, false), 'high');
  const full = S.zoom ? load(imgUrl(name, true), 'high') : null;
  const best = [full, fit].find((e) => e && e.ready);
  show(best ? best.img : placeholder(name));
  // Upgrade to a sharper version as soon as it's decoded.
  for (const e of [full, fit]) {
    if (e === best) break;
    if (e && !e.ready && !e.failed) {
      e.promise.then(() => { if (current() === name && S.view === 'loupe') render(); });
    }
  }
  const mark = S.marks[name];
  stage.classList.toggle('zoomed', S.zoom);
  stage.classList.toggle('reject', mark === 'reject');
  frame.className = mark || '';
  prefetch();
}

function show(img) {
  if (shown !== img) {
    stage.replaceChildren(img);
    shown = img;
  }
  layout();
}

function zoomedSize(img) {
  // 100% = one pixel of the full-size image per screen pixel.
  const m = metas.get(current()) || {};
  return [(m.width || img.naturalWidth || 3) / dpr(), (m.height || img.naturalHeight || 2) / dpr()];
}

function clampCentre(c, view, size) {
  if (size <= view) return 0.5;
  const half = view / 2 / size;
  return Math.min(1 - half, Math.max(half, c));
}

function layout() {
  const img = shown;
  if (!img) return;
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  let w, h, x, y;
  if (S.zoom) {
    [w, h] = zoomedSize(img);
    S.zx = clampCentre(S.zx, W, w);
    S.zy = clampCentre(S.zy, H, h);
    x = W / 2 - S.zx * w;
    y = H / 2 - S.zy * h;
  } else {
    const nw = img.naturalWidth || 3;
    const nh = img.naturalHeight || 2;
    const s = Math.min(W / nw, H / nh);
    w = nw * s;
    h = nh * s;
    x = (W - w) / 2;
    y = (H - h) / 2;
  }
  for (const el of clipShown ? [img, clipShown] : [img]) {
    el.style.width = `${w}px`;
    el.style.height = `${h}px`;
    el.style.transform = `translate(${x}px, ${y}px)`;
  }
}

function toggleZoom(px, py) {
  if (S.zoom || !shown) {
    S.zoom = false;
  } else {
    S.zoom = true;
    S.zx = S.zy = 0.5;
    const r = shown.getBoundingClientRect();
    if (px != null && px >= r.left && px <= r.right && py >= r.top && py <= r.bottom) {
      // Keep the point under the cursor where it is.
      const sr = stage.getBoundingClientRect();
      const [w, h] = zoomedSize(shown);
      S.zx = (px - r.left) / r.width + (sr.width / 2 - (px - sr.left)) / w;
      S.zy = (py - r.top) / r.height + (sr.height / 2 - (py - sr.top)) / h;
    }
  }
  render();
}

let drag = null;
let pointer = null;

stage.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  drag = { x: e.clientX, y: e.clientY, moved: false };
  stage.setPointerCapture(e.pointerId);
});

stage.addEventListener('pointermove', (e) => {
  pointer = { x: e.clientX, y: e.clientY };
  if (!drag) return;
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  if (!drag.moved && Math.hypot(dx, dy) < 4) return;
  drag.moved = true;
  if (S.zoom && shown) {
    S.zx -= dx / parseFloat(shown.style.width);
    S.zy -= dy / parseFloat(shown.style.height);
    drag.x = e.clientX;
    drag.y = e.clientY;
    layout();
  }
});

stage.addEventListener('pointerup', (e) => {
  const d = drag;
  drag = null;
  if (d && !d.moved) toggleZoom(e.clientX, e.clientY);
});

stage.addEventListener('pointerleave', () => { if (!drag) pointer = null; });

// --- Filmstrip and grid ---------------------------------------------------------

let gridDirty = true;
let curCells = [];

function markCell(cell, name) {
  const m = S.marks[name];
  cell.classList.toggle('keep', m === 'keep');
  cell.classList.toggle('reject', m === 'reject');
}

function buildThumbs(box) {
  const frag = document.createDocumentFragment();
  S.list.forEach((name, pos) => {
    const cell = document.createElement('div');
    cell.className = 'cell';
    cell.dataset.pos = pos;
    cell.title = name;
    const img = document.createElement('img');
    img.loading = 'lazy';
    img.decoding = 'async';
    img.draggable = false;
    img.alt = '';
    img.src = thumbUrl(name);
    cell.append(img);
    markCell(cell, name);
    frag.append(cell);
  });
  box.replaceChildren(frag);
  curCells = [];
}

function updateCurrentCell() {
  for (const c of curCells) c.classList.remove('cur');
  curCells = [];
  if (S.pos < 0) return;
  for (const [box, opts] of [
    [strip, { inline: 'center', block: 'nearest' }],
    [grid, { inline: 'nearest', block: 'nearest' }],
  ]) {
    const c = box.children[S.pos];
    if (!c) continue;
    c.classList.add('cur');
    curCells.push(c);
    if (box.offsetParent !== null) c.scrollIntoView(opts);
  }
}

function gridCols() {
  return getComputedStyle(grid).gridTemplateColumns.split(' ').length;
}

function onCellClick(e) {
  const cell = e.target.closest('.cell');
  if (cell) go(Number(cell.dataset.pos));
}
strip.addEventListener('click', onCellClick);
grid.addEventListener('click', onCellClick);
grid.addEventListener('dblclick', (e) => { if (e.target.closest('.cell')) setView('loupe'); });

function setView(view) {
  S.view = view;
  grid.hidden = view !== 'grid';
  document.body.classList.toggle('gridview', view === 'grid');
  if (view === 'grid' && gridDirty) {
    buildThumbs(grid);
    gridDirty = false;
  }
  render();
}

// --- State -------------------------------------------------------------------------

function matches(name) {
  const m = S.marks[name];
  return S.filter === 'all' || (S.filter === 'unmarked' ? !m : m === S.filter);
}

function rebuildList(keepName) {
  S.list = S.photos.filter(matches);
  // Stay on the same photo, or the first one after it that's in the filter.
  const at = S.order.get(keepName) ?? 0;
  let pos = S.list.findIndex((n) => S.order.get(n) >= at);
  if (pos < 0) pos = S.list.length - 1;
  S.pos = pos;
  if (pos >= 0) S.lastName = S.list[pos];
  buildThumbs(strip);
  grid.replaceChildren();
  gridDirty = true;
  if (S.view === 'grid') setView('grid');
}

function go(pos) {
  if (pos < 0 || pos >= S.list.length || pos === S.pos) return;
  S.pos = pos;
  S.lastName = S.list[pos];
  render();
  save();
}

async function mark(m) {
  const name = current();
  if (!name) return;
  const prev = S.marks[name] ?? null;
  setMark(name, m);
  if (m && S.auto && S.view === 'loupe') {
    if (S.pos < S.list.length - 1) go(S.pos + 1);
    else {
      render();
      toast('Last photo');
    }
  } else {
    render();
  }
  try {
    const r = await fetch('/api/mark', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, mark: m }),
    });
    if (!r.ok) throw new Error(r.statusText);
  } catch {
    setMark(name, prev);
    render();
    toast(`Couldn't save the mark on ${name}. Is the server still running?`, true);
  }
}

function setMark(name, m) {
  if (m) S.marks[name] = m;
  else delete S.marks[name];
  const pos = S.list.indexOf(name);
  for (const box of [strip, grid]) {
    const c = box.children[pos];
    if (c) markCell(c, name);
  }
}

function cycleFilter(dir) {
  const i = FILTERS.indexOf(S.filter);
  S.filter = FILTERS[(i + dir + FILTERS.length) % FILTERS.length];
  rebuildList(current() ?? S.lastName);
  render();
  save();
  toast(`${FILTER_LABEL[S.filter]}: ${S.list.length}`);
}

function save() {
  try {
    localStorage.setItem(`cull:${S.folder}`, JSON.stringify({
      name: S.lastName, filter: S.filter, auto: S.auto, strip: S.strip, clip: S.clip, mode: S.mode,
    }));
  } catch {}
}

function restore() {
  try {
    return JSON.parse(localStorage.getItem(`cull:${S.folder}`)) || {};
  } catch {
    return {};
  }
}

// --- Header, toast, help ---------------------------------------------------------

function updateHeader() {
  if (S.mode === 'edit') return;
  const name = current();
  $('#name').textContent = name || '—';
  const m = name && S.marks[name];
  const badge = $('#badge');
  badge.hidden = !m;
  badge.className = `badge cull-only ${m || ''}`;
  badge.textContent = m === 'keep' ? 'KEEP' : 'REJECT';
  $('#pos').textContent = `${S.pos + 1} / ${S.list.length}`;

  const x = (name && metas.get(name)) || {};
  const taken = x.taken && x.taken.replace(/^(\d{4}):(\d\d):(\d\d) (\d\d:\d\d).*/, '$1-$2-$3 $4');
  $('#exif').textContent = [x.shutter, x.aperture, x.iso && `ISO ${x.iso}`, x.focal, x.ev, taken]
    .filter(Boolean)
    .join('  ');

  let keep = 0;
  let reject = 0;
  for (const n of S.photos) {
    const v = S.marks[n];
    if (v === 'keep') keep++;
    else if (v === 'reject') reject++;
  }
  $('#counts').innerHTML =
    `<span class="k">✓ ${keep}</span>  <span class="r">✗ ${reject}</span>  ${S.photos.length - keep - reject} unmarked`;
  $('#filter').textContent = FILTER_LABEL[S.filter];
  $('#filter').classList.toggle('on', S.filter !== 'all');
  $('#auto').classList.toggle('on', S.auto);
  $('#zoomchip').hidden = !S.zoom || S.view !== 'loupe';
  $('#clipchip').hidden = !S.clip || S.view !== 'loupe';
  document.title = name ? `${name} · Cull` : 'Cull';
}

let toastTimer;
function toast(msg, err = false) {
  const t = $('#toast');
  t.textContent = msg;
  t.className = err ? 'show err' : 'show';
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = ''; }, err ? 5000 : 1200);
}

function toggleAuto() {
  S.auto = !S.auto;
  toast(S.auto ? 'Auto-advance on' : 'Auto-advance off');
  updateHeader();
  save();
}

function toggleHelp() {
  $('#help').hidden = !$('#help').hidden;
}

$('#filter').addEventListener('click', () => cycleFilter(1));
$('#auto').addEventListener('click', toggleAuto);
$('#helpchip').addEventListener('click', toggleHelp);
$('#zoomchip').addEventListener('click', () => toggleZoom());
$('#clipchip').addEventListener('click', toggleClip);
$('#help').addEventListener('click', toggleHelp);
for (const b of document.querySelectorAll('button')) b.addEventListener('mousedown', (e) => e.preventDefault());

// --- Keyboard ------------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  keyHeld = e.repeat;
  if (!$('#help').hidden) {
    if (k === 'Escape' || k === '?' || k === 'h') toggleHelp();
    e.preventDefault();
    return;
  }
  if (S.mode === 'edit') {
    if (k === '?' || k === 'h') toggleHelp();
    else if (!editorKey(e, k)) return;
    e.preventDefault();
    return;
  }
  const inGrid = S.view === 'grid';
  switch (k) {
    case 'ArrowRight':
    case ' ':
      go(S.pos + 1);
      break;
    case 'ArrowLeft':
      go(S.pos - 1);
      break;
    case 'ArrowUp':
      if (inGrid) go(S.pos - gridCols());
      else if (!e.repeat) mark('keep');
      break;
    case 'ArrowDown':
      if (inGrid) go(Math.min(S.pos + gridCols(), S.list.length - 1));
      else if (!e.repeat) mark('reject');
      break;
    case 'p':
      if (!e.repeat) mark('keep');
      break;
    case 'x':
      if (!e.repeat) mark('reject');
      break;
    case 'u':
      if (!e.repeat) mark(null);
      break;
    case 'z':
      if (!inGrid) toggleZoom(pointer?.x, pointer?.y);
      break;
    case 'j':
      toggleClip();
      break;
    case 'e':
      setMode('edit');
      break;
    case 'g':
      setView(inGrid ? 'loupe' : 'grid');
      break;
    case 'Enter':
      if (inGrid) setView('loupe');
      break;
    case 'Escape':
      if (inGrid) setView('loupe');
      else if (S.zoom) toggleZoom();
      break;
    case 'f':
      cycleFilter(e.shiftKey ? -1 : 1);
      break;
    case 'a':
      toggleAuto();
      break;
    case 's':
      S.strip = !S.strip;
      document.body.classList.toggle('nostrip', !S.strip);
      save();
      break;
    case 'Home':
      go(0);
      break;
    case 'End':
      go(S.list.length - 1);
      break;
    case '?':
    case 'h':
      toggleHelp();
      break;
    default:
      return;
  }
  e.preventDefault();
});

// --- Modes ------------------------------------------------------------------------------

function setMode(mode) {
  if (mode === S.mode) return;
  const from = S.mode === 'edit' ? editorLeave() : current() ?? S.lastName;
  S.mode = mode;
  document.body.classList.toggle('mode-edit', mode === 'edit');
  for (const b of document.querySelectorAll('#modes button')) b.classList.toggle('on', b.dataset.mode === mode);
  history.replaceState(null, '', `#${mode}`);
  if (mode === 'edit') {
    editorEnter(from).catch((err) => toast(`Couldn't open the editor: ${err.message}`, true));
  } else {
    // Back on the photo you were editing, if the filter shows it.
    const pos = from ? S.list.indexOf(from) : -1;
    if (pos >= 0) {
      S.pos = pos;
      S.lastName = from;
    }
    render();
  }
  save();
}

for (const b of document.querySelectorAll('#modes button')) {
  b.addEventListener('click', () => setMode(b.dataset.mode));
}

// --- Start -----------------------------------------------------------------------------

let resizeTimer;
new ResizeObserver(() => {
  if (!S.photos.length) return;
  layout();
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(render, 150); // may need a bigger embedded JPEG now
}).observe(stage);

async function init() {
  const st = await (await fetch('/api/state')).json();
  S.folder = st.folder;
  S.version = st.version;
  S.photos = st.photos;
  S.order = new Map(S.photos.map((n, i) => [n, i]));
  S.marks = st.marks;
  $('#folder').textContent = `${st.folder} · marks are saved to .photos.json in that folder`;

  const saved = restore();
  if (FILTERS.includes(saved.filter)) S.filter = saved.filter;
  if (typeof saved.auto === 'boolean') S.auto = saved.auto;
  if (typeof saved.strip === 'boolean') S.strip = saved.strip;
  if (typeof saved.clip === 'boolean') S.clip = saved.clip;
  document.body.classList.toggle('nostrip', !S.strip);
  rebuildList(saved.name ?? S.photos[0]);
  render();
  const hash = location.hash.slice(1);
  const mode = hash === 'edit' || hash === 'cull' ? hash : saved.mode;
  for (const b of document.querySelectorAll('#modes button')) b.classList.toggle('on', b.dataset.mode === 'cull');
  if (mode === 'edit') setMode('edit');
}

init().catch((err) => toast(`Couldn't load the photo list: ${err}`, true));
