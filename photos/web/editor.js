'use strict';

// The editor: develops the kept photos with the WebGL renderer in develop.js.
// How settings spread through a series is decided by the server (edits.py), which
// sends the outcome as runs: consecutive photos sharing settings.

const ED = {
  loaded: false,
  groups: [], // series number of every photo, parallel to S.photos
  taken: [], // capture time of every photo, 'YYYY:MM:DD HH:MM:SS'
  runs: [], // run number of every photo; a run shares settings
  runSettings: [], // settings of every run
  runEdited: [], // whether a run's settings differ from the defaults
  raw: [], // whether each photo is a raw file
  defaults: null,
  limits: null,
  tempStep: null,
  kept: [], // kept raw photos, in shooting order
  pos: -1,
  live: null, // the current photo's settings, including a drag not saved yet
  before: false, // showing the photo without edits
  zoom: false, // 100% view: the full-size photo as an export will develop it, noise reduction included
  zx: 0.5, // its centre, as a fraction of the photo
  zy: 0.5,
  view: null, // where the full-size photo sits on the canvas, while zoomed (see Developer.draw)
  full: { name: null, loaded: new Map(), pending: new Map(), queue: [] }, // full-size tiles of a photo
  dev: null,
  shown: null, // name of the photo whose data is on the GPU
  shownDenoised: false, // whether that's the noise-reduced preview yet
  previews: new Map(), // name -> { promise, abort, info }: quick previews
  denoised: new Map(), // the same, noise-reduced
  cells: [],
  saving: Promise.resolve(),
};

const editView = $('#editview');
const editCanvas = $('#editcanvas');
const editPlaceholder = $('#editplaceholder');
const editStrip = $('#editstrip');
const SLIDERS = [...document.querySelectorAll('#panel .slider')];
const editCurrent = () => ED.kept[ED.pos] ?? null;

// --- Series ---------------------------------------------------------------------------

const settingsOf = (i) => ED.runSettings[ED.runs[i]];

function sameRange(ids, i) {
  // [first, last] index of the consecutive photos with the same id as photo i.
  let a = i;
  let b = i;
  while (a > 0 && ids[a - 1] === ids[i]) a--;
  while (b + 1 < ids.length && ids[b + 1] === ids[i]) b++;
  return [a, b];
}

const seriesOf = (i) => sameRange(ED.groups, i);
// The photos sharing photo i's settings: a change at the first of them reaches all of them.
const runOf = (i) => sameRange(ED.runs, i);

function applyRuns(st) {
  ED.runs = st.runs;
  ED.runSettings = st.runSettings;
  ED.runEdited = st.runEdited;
}

function keptBetween(a, b) {
  let n = 0;
  for (let i = a; i <= b; i++) if (S.marks[S.photos[i]] === 'keep') n++;
  return n;
}

// --- Sliders -------------------------------------------------------------------------------

// The temperature slider is linear in mired (1e6 / K), which is closer to how different
// temperatures look, and lands on the server's grid of temperatures.
const miredOf = (t) => 1e6 / t;

function tempToSlider(t) {
  const [lo, hi] = ED.limits.temp;
  return Math.round(((miredOf(lo) - miredOf(t)) / (miredOf(lo) - miredOf(hi))) * 1000);
}

function sliderToTemp(v) {
  const [lo, hi] = ED.limits.temp;
  const t = 1e6 / (miredOf(lo) - (v / 1000) * (miredOf(lo) - miredOf(hi)));
  return Math.min(hi, Math.max(lo, Math.round(t / ED.tempStep) * ED.tempStep));
}

function asShotTemp() {
  return ED.dev && ED.shown === editCurrent() ? ED.dev.info.asShotTemp : null;
}

function formatValue(key, v) {
  if (key === 'exposure') return `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`;
  if (key === 'saturation' || key === 'contrast') return `${v > 0 ? '+' : ''}${v}`;
  if (key === 'temp') {
    const t = v ?? asShotTemp();
    return v == null ? (t ? `${t} K, as shot` : 'as shot') : `${v} K`;
  }
  return String(v);
}

function updateSliders() {
  for (const el of SLIDERS) {
    const key = el.dataset.key;
    const input = el.querySelector('input');
    const v = ED.live?.[key] ?? ED.defaults?.[key];
    if (key === 'temp') {
      const t = v ?? asShotTemp();
      input.value = t ? tempToSlider(t) : 500;
    } else {
      input.value = v;
    }
    const label = el.querySelector('.sval');
    label.textContent = ED.live ? formatValue(key, v) : '';
    label.classList.toggle('default', v === ED.defaults?.[key]);
  }
}

for (const el of SLIDERS) {
  const key = el.dataset.key;
  const input = el.querySelector('input');
  input.addEventListener('input', () => {
    if (!ED.live) return;
    const v = Number(input.value);
    ED.live = { ...ED.live, [key]: key === 'temp' ? sliderToTemp(v) : v };
    ED.before = false;
    updateSliders();
    requestDraw();
  });
  input.addEventListener('change', commit);
  // Hand the arrow keys back to photo navigation once the mouse lets go.
  input.addEventListener('pointerup', () => setTimeout(() => input.blur(), 0));
  el.addEventListener('dblclick', () => {
    if (!ED.live) return;
    ED.live = { ...ED.live, [key]: ED.defaults[key] };
    updateSliders();
    requestDraw();
    commit();
  });
}

$('#resetPhoto').addEventListener('click', () => {
  if (!ED.live) return;
  ED.live = { ...ED.defaults };
  updateSliders();
  requestDraw();
  commit();
});

function commit() {
  const name = editCurrent();
  if (!name || !ED.live) return;
  const settings = { ...ED.live };
  // One at a time, so the server sees edits in the order they were made.
  ED.saving = ED.saving
    .then(() => fetch('/api/edit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, settings }),
    }))
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(r.statusText))))
    .then((res) => {
      applyRuns(res);
      buildEditStrip();
      updateSeries();
    })
    .catch((err) => toast(`Couldn't save the edit: ${err.message}`, true));
}

// --- Rendering --------------------------------------------------------------------------------

function previewSide() {
  // Long side of a ~3:2 photo fitted to the view, in device pixels.
  const w = editView.clientWidth;
  const h = editView.clientHeight;
  return Math.ceil(Math.max(Math.min(w, h * 1.5), Math.min(h, w * 1.5)) * dpr());
}

// Each photo comes in two steps: a quick preview without noise reduction (~0.5 s), then
// the export's own input shrunk, noise reduction included (develop.denoised_preview; ~3 s,
// unless the server prepared it while you were on the photo before).
function preview(name, denoised = false) {
  const cache = denoised ? ED.denoised : ED.previews;
  let p = cache.get(name);
  if (!p) {
    const abort = new AbortController();
    const promise = fetch(`/develop/${enc(name)}?w=${previewSide()}${denoised ? '&nr=1' : ''}&v=${S.version}`, { signal: abort.signal })
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.statusText))))
      .then(parsePreview);
    p = { promise, abort, info: null };
    promise.then(
      (info) => { p.info = info; },
      () => { if (cache.get(name) === p) cache.delete(name); },
    );
    cache.set(name, p);
  }
  return p.promise;
}

let drawQueued = false;
function requestDraw() {
  if (!drawQueued) {
    drawQueued = true;
    requestAnimationFrame(() => {
      drawQueued = false;
      if (!(ED.dev && ED.shown && ED.shown === editCurrent())) return;
      const view = ED.view && { ...ED.view, tiles: visibleTiles().filter((t) => t.loaded).map((t) => t.rect) };
      ED.dev.draw(ED.before ? ED.defaults : ED.live, view);
    });
  }
  const loading = ED.view && visibleTiles().some((t) => !t.loaded);
  $('#editnote').textContent = ED.before
    ? 'Before'
    : ED.shown !== editCurrent() ? 'Developing…'
    : loading ? 'Loading full size…'
    : !ED.shownDenoised ? 'Reducing noise…' : '';
}

function layoutEdit() {
  const W = editView.clientWidth;
  const H = editView.clientHeight;
  const onGpu = ED.dev && ED.shown && ED.shown === editCurrent();
  const el = onGpu ? editCanvas : editPlaceholder;
  editCanvas.hidden = !onGpu;
  editPlaceholder.hidden = onGpu || !editCurrent();
  editView.classList.toggle('zoomed', ED.zoom && !!onGpu);
  $('#editzoom').hidden = !ED.zoom;
  if (onGpu && ED.zoom) {
    layoutZoomed(W, H);
    return;
  }
  ED.view = null;
  const iw = onGpu ? ED.dev.info.width : editPlaceholder.naturalWidth || 3;
  const ih = onGpu ? ED.dev.info.height : editPlaceholder.naturalHeight || 2;
  const s = Math.min(W / iw, H / ih);
  const w = iw * s;
  const h = ih * s;
  Object.assign(el.style, { left: `${(W - w) / 2}px`, top: `${(H - h) / 2}px`, width: `${w}px`, height: `${h}px` });
  if (onGpu) {
    sizeCanvas(Math.round(w * dpr()), Math.round(h * dpr()));
    requestDraw();
  }
}
editPlaceholder.addEventListener('load', layoutEdit);

function sizeCanvas(cw, ch) {
  if (editCanvas.width !== cw || editCanvas.height !== ch) {
    editCanvas.width = cw;
    editCanvas.height = ch;
  }
}

// --- 100% view ---------------------------------------------------------------------------------
// One pixel of the full-size photo per screen pixel. The server prepares what an export
// feeds the develop shader (develop.Full: camera RGB after noise reduction, and the grain
// given back) and sends it in tiles; the same shader develops them here, so sliders stay
// instant and you see the export's noise and detail.

const TILE_FETCHES = 6;

function layoutZoomed(W, H) {
  Object.assign(editCanvas.style, { left: '0px', top: '0px', width: `${W}px`, height: `${H}px` });
  sizeCanvas(Math.round(W * dpr()), Math.round(H * dpr()));
  const { fullWidth: fw, fullHeight: fh } = ED.dev.info;
  const cw = editCanvas.width;
  const ch = editCanvas.height;
  ED.zx = clampCentre(ED.zx, cw, fw);
  ED.zy = clampCentre(ED.zy, ch, fh);
  ED.view = { x0: Math.round(ED.zx * fw - cw / 2), y0: Math.round(ED.zy * fh - ch / 2), width: fw, height: fh };
  wantTiles();
  requestDraw();
}

function tileRect(i, j) {
  const { fullWidth: fw, fullHeight: fh, fullTile: T } = ED.dev.info;
  return [i * T, j * T, Math.min(T, fw - i * T), Math.min(T, fh - j * T)];
}

function visibleTiles(margin = 0) {
  // The tiles on the canvas (and `margin` more around it), nearest its centre first.
  if (!ED.view || ED.full.name !== editCurrent()) return [];
  const { fullWidth: fw, fullHeight: fh, fullTile: T } = ED.dev.info;
  const { x0, y0 } = ED.view;
  const cw = editCanvas.width;
  const ch = editCanvas.height;
  const range = (a, size, n) => [Math.max(0, Math.floor(a / T) - margin), Math.min(n - 1, Math.floor((a + size - 1) / T) + margin)];
  const [i0, i1] = range(x0, cw, Math.ceil(fw / T));
  const [j0, j1] = range(y0, ch, Math.ceil(fh / T));
  const tiles = [];
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const rect = tileRect(i, j);
      const d = Math.hypot(rect[0] + rect[2] / 2 - x0 - cw / 2, rect[1] + rect[3] / 2 - y0 - ch / 2);
      tiles.push({ key: `${i},${j}`, rect, d, loaded: ED.full.loaded.has(`${i},${j}`) });
    }
  }
  return tiles.sort((a, b) => a.d - b.d);
}

function wantTiles() {
  const name = editCurrent();
  if (ED.full.name !== name) {
    for (const abort of ED.full.pending.values()) abort.abort();
    ED.full = { name, loaded: new Map(), pending: new Map(), queue: [] };
    ED.dev.allocFull(ED.dev.info.fullWidth, ED.dev.info.fullHeight);
  }
  ED.full.queue = visibleTiles(1).filter((t) => !t.loaded && !ED.full.pending.has(t.key));
  pumpTiles();
}

function pumpTiles() {
  const f = ED.full;
  while (f.pending.size < TILE_FETCHES && f.queue.length) {
    const { key, rect } = f.queue.shift();
    const abort = new AbortController();
    f.pending.set(key, abort);
    fetch(`/full/${enc(f.name)}?t=${key}&v=${S.version}`, { signal: abort.signal })
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.statusText))))
      .then((buffer) => {
        if (ED.full !== f || ED.shown !== f.name) return;
        ED.dev.putTile(...rect, new Uint16Array(buffer));
        f.loaded.set(key, rect);
        requestDraw();
      })
      .catch((err) => {
        if (err.name !== 'AbortError' && ED.full === f && !f.failed) {
          f.failed = true;
          toast(`Couldn't load ${f.name} at full size: ${err.message}`, true);
        }
      })
      .finally(() => {
        f.pending.delete(key);
        if (ED.full === f) pumpTiles();
      });
  }
}

function toggleEditZoom(px, py) {
  if (ED.zoom) {
    ED.zoom = false;
    layoutEdit();
    return;
  }
  if (!(ED.dev && ED.shown && ED.shown === editCurrent())) return;
  ED.zoom = true;
  ED.zx = ED.zy = 0.5;
  const r = editCanvas.getBoundingClientRect();
  if (px != null && px >= r.left && px <= r.right && py >= r.top && py <= r.bottom) {
    // Keep the point under the cursor where it is.
    const sr = editView.getBoundingClientRect();
    ED.zx = (px - r.left) / r.width + ((sr.width / 2 - (px - sr.left)) * dpr()) / ED.dev.info.fullWidth;
    ED.zy = (py - r.top) / r.height + ((sr.height / 2 - (py - sr.top)) * dpr()) / ED.dev.info.fullHeight;
  }
  layoutEdit();
}

$('#editzoom').addEventListener('click', () => toggleEditZoom());

let editDrag = null;
let editPointer = null;

editView.addEventListener('pointerdown', (e) => {
  if (e.button !== 0) return;
  editDrag = { x: e.clientX, y: e.clientY, moved: false };
  editView.setPointerCapture(e.pointerId);
});

editView.addEventListener('pointermove', (e) => {
  editPointer = { x: e.clientX, y: e.clientY };
  if (!editDrag) return;
  const dx = e.clientX - editDrag.x;
  const dy = e.clientY - editDrag.y;
  if (!editDrag.moved && Math.hypot(dx, dy) < 4) return;
  editDrag.moved = true;
  if (ED.view) {
    ED.zx -= (dx * dpr()) / ED.view.width;
    ED.zy -= (dy * dpr()) / ED.view.height;
    editDrag.x = e.clientX;
    editDrag.y = e.clientY;
    layoutEdit();
  }
});

editView.addEventListener('pointerup', (e) => {
  const d = editDrag;
  editDrag = null;
  if (d && !d.moved) toggleEditZoom(e.clientX, e.clientY);
});

editView.addEventListener('pointerleave', () => { if (!editDrag) editPointer = null; });

let editTimer;
function showEdit() {
  const name = editCurrent();
  $('#editempty').hidden = !!name;
  updateEditHeader();
  updateStripMarks();
  if (!name) {
    ED.live = null;
    ED.shown = null;
    updateSliders();
    updateSeries();
    layoutEdit();
    return;
  }
  ED.live = { ...settingsOf(S.order.get(name)) };
  updateSliders();
  updateSeries();
  if (ED.shown !== name) {
    // The camera's JPEG stands in while the raw is decoded (~1 s the first time).
    editPlaceholder.src = `/img/${enc(name)}?w=${fitSideEdit()}&v=${S.version}`;
  }
  layoutEdit();
  requestDraw();
  clearTimeout(editTimer);
  // While an arrow key is held, wait for browsing to pause before decoding anything.
  editTimer = setTimeout(() => loadEdit(name), keyHeld ? 150 : 0);
}

function fitSideEdit() {
  return Math.ceil(previewSide() / 256) * 256;
}

function loadEdit(name) {
  if (!ED.dev) return; // no WebGL: the camera JPEG stays up
  const near = [0, 1, 2, -1].map((d) => ED.kept[ED.pos + d]).filter(Boolean);
  fetch('/api/develop-hint', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ names: near, w: previewSide() }),
  }).catch(() => {});
  // The server prepares noise-reduced data for this photo and the next only (see develop-hint).
  for (const [cache, keep] of [[ED.previews, near], [ED.denoised, near.slice(0, 2)]]) {
    for (const [n, p] of cache) {
      if (!keep.includes(n)) {
        p.abort.abort();
        cache.delete(n);
      }
    }
  }
  const show = (info) => {
    if (editCurrent() !== name || S.mode !== 'edit') return;
    ED.dev.load(info);
    ED.shown = name;
    ED.shownDenoised = info.grain;
    updateSliders(); // the as-shot temperature is known now
    layoutEdit();
    // Then the neighbours, so the next photo is ready when you get there.
    for (const n of near.slice(1)) preview(n).catch(() => {});
  };
  const failed = (err) => {
    if (err.name !== 'AbortError' && editCurrent() === name) toast(`Couldn't develop ${name}: ${err.message}`, true);
  };
  const ready = ED.denoised.get(name)?.info;
  if (ready) {
    if (!(ED.shown === name && ED.shownDenoised)) show(ready);
  } else {
    preview(name).then((info) => { if (ED.shown !== name) show(info); }).catch(failed);
  }
  preview(name, true)
    .then((info) => {
      if (!(ED.shown === name && ED.shownDenoised)) show(info);
      if (editCurrent() === name && near[1]) preview(near[1], true).catch(() => {});
    })
    .catch(failed);
}

// --- Panel and strip -------------------------------------------------------------------------

function updateSeries() {
  const name = editCurrent();
  const title = $('#seriestitle');
  const info = $('#seriesinfo');
  if (!name) {
    title.textContent = '';
    info.textContent = '';
    return;
  }
  const i = S.order.get(name);
  const [a, b] = seriesOf(i);
  const [ra, rb] = runOf(i);
  const clock = (k) => ED.taken[k]?.slice(11) ?? '?';
  title.textContent = b > a ? `Series of ${b - a + 1} · ${clock(a)} – ${clock(b)}` : `Single photo · ${clock(i)}`;
  const lines = [];
  if (b > a) lines.push(`${keptBetween(a, b)} of the ${b - a + 1} are kept.`);
  const following = keptBetween(i + 1, rb);
  const reach = rb - i;
  lines.push(
    following || reach
      ? `A change applies to this photo and the <b>${following}</b> following kept photo${following === 1 ? '' : 's'}` +
        (reach > following ? ` (${reach} following in all, counting unpicked ones).` : '.')
      : 'A change applies to this photo only.',
  );
  if (i > ra) {
    const earlier = i - a === 1 ? 'the earlier photo keeps its settings' : `the ${i - a} earlier photos keep their settings`;
    lines.push(`It splits the series before this photo; ${earlier}.`);
  }
  else if (i > a) lines.push('This photo already starts its own part of the series.');
  info.innerHTML = lines.map((l) => `<div>${l}</div>`).join('');
}

function buildEditStrip() {
  const frag = document.createDocumentFragment();
  ED.cells = [];
  ED.kept.forEach((name, pos) => {
    const i = S.order.get(name);
    if (pos > 0) {
      const j = S.order.get(ED.kept[pos - 1]);
      const sep = document.createElement('div');
      if (ED.groups[i] !== ED.groups[j]) sep.className = 'gap';
      else if (ED.runs[i] !== ED.runs[j]) sep.className = 'split';
      if (sep.className) frag.append(sep);
    }
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
    frag.append(cell);
    ED.cells.push(cell);
  });
  editStrip.replaceChildren(frag);
  updateStripMarks();
}

function updateStripMarks() {
  const name = editCurrent();
  const i = name ? S.order.get(name) : -1;
  const [, rb] = name ? runOf(i) : [0, -1];
  ED.kept.forEach((n, pos) => {
    const cell = ED.cells[pos];
    if (!cell) return;
    const k = S.order.get(n);
    cell.classList.toggle('cur', pos === ED.pos);
    cell.classList.toggle('affected', k >= i && k <= rb);
    cell.classList.toggle('edited', ED.runEdited[ED.runs[k]]);
  });
  const cur = ED.cells[ED.pos];
  if (cur && S.mode === 'edit') cur.scrollIntoView({ inline: 'center', block: 'nearest' });
}

editStrip.addEventListener('click', (e) => {
  const cell = e.target.closest('.cell');
  if (cell) editGo(Number(cell.dataset.pos));
});

function updateEditHeader() {
  const name = editCurrent();
  $('#name').textContent = name || '—';
  $('#pos').textContent = ED.kept.length ? `${ED.pos + 1} / ${ED.kept.length} kept` : '0 kept';
  const x = (name && (metas.get(name) || meta(name))) || {};
  const taken = x.taken && x.taken.replace(/^(\d{4}):(\d\d):(\d\d) (\d\d:\d\d).*/, '$1-$2-$3 $4');
  $('#exif').textContent = [x.shutter, x.aperture, x.iso && `ISO ${x.iso}`, x.focal, x.ev, taken].filter(Boolean).join('  ');
  document.title = name ? `${name} · Edit` : 'Edit';
}

function editorMetaArrived(name) {
  if (name !== editCurrent()) return;
  updateEditHeader();
  updateSeries();
}

// --- Navigation and modes -------------------------------------------------------------------

function editGo(pos) {
  if (pos < 0 || pos >= ED.kept.length || pos === ED.pos) return;
  ED.pos = pos;
  ED.before = false;
  showEdit();
}

async function editorEnter(name) {
  if (!ED.loaded) {
    const st = await (await fetch('/api/edits')).json();
    Object.assign(ED, {
      groups: st.groups,
      taken: st.taken,
      raw: st.raw,
      defaults: st.defaults,
      limits: st.limits,
      tempStep: st.tempStep,
    });
    applyRuns(st);
    for (const el of SLIDERS) {
      const input = el.querySelector('input');
      if (el.dataset.key !== 'temp') [input.min, input.max] = ED.limits[el.dataset.key];
    }
    ED.loaded = true;
    try {
      ED.dev = new Developer(editCanvas, await loadShaders());
    } catch (err) {
      toast(`Can't develop in this browser: ${err.message}`, true);
    }
  }
  ED.kept = S.photos.filter((n, i) => S.marks[n] === 'keep' && ED.raw[i]);
  // Open on the photo picked last, or the nearest kept one after it.
  const at = S.order.get(name) ?? 0;
  let pos = ED.kept.findIndex((n) => S.order.get(n) >= at);
  if (pos < 0) pos = ED.kept.length - 1;
  ED.pos = pos;
  buildEditStrip();
  showEdit();
}

function editorLeave() {
  clearTimeout(editTimer);
  return editCurrent();
}

new ResizeObserver(() => { if (S.mode === 'edit') layoutEdit(); }).observe(editView);

function editorKey(e, k) {
  if (e.target instanceof HTMLInputElement) {
    if (k === 'Escape') e.target.blur();
    else if (k.startsWith('Arrow') || k === 'Home' || k === 'End' || k === 'PageUp' || k === 'PageDown') return false;
  }
  switch (k) {
    case 'ArrowRight':
    case ' ':
      editGo(ED.pos + 1);
      return true;
    case 'ArrowLeft':
      editGo(ED.pos - 1);
      return true;
    case 'Home':
      editGo(0);
      return true;
    case 'End':
      editGo(ED.kept.length - 1);
      return true;
    case '\\':
      ED.before = !ED.before;
      requestDraw();
      return true;
    case 'z':
      toggleEditZoom(editPointer?.x, editPointer?.y);
      return true;
    case 'Escape':
      if (ED.before) {
        ED.before = false;
        requestDraw();
      } else if (ED.zoom) {
        toggleEditZoom();
      }
      return true;
    case 'e':
      setMode('cull');
      return true;
    default:
      return false;
  }
}

// --- Export ---------------------------------------------------------------------------------------

$('#exportBtn').addEventListener('click', async () => {
  await ED.saving;
  try {
    const st = await (await fetch('/api/export', { method: 'POST' })).json();
    showExport(st);
  } catch (err) {
    toast(`Couldn't start the export: ${err.message}`, true);
  }
});

function showExport(st) {
  const out = $('#exportStatus');
  if (st.running) {
    out.textContent = `Exporting ${st.done} / ${st.total}…`;
    setTimeout(() => fetch('/api/export').then((r) => r.json()).then(showExport).catch(() => {}), 700);
  } else {
    const ok = st.done - st.failed.length;
    out.textContent = st.failed.length ? `Exported ${ok}, ${st.failed.length} failed` : `Exported ${ok} to ${st.outDir}`;
    if (st.failed.length) toast(`Export failed for ${st.failed.join(', ')}`, true);
  }
}
