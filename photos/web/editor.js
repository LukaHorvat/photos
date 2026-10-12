'use strict';

// The editor: develops the kept photos with the WebGL renderer in develop.js.
// How settings spread through a series is decided by the server (edits.py); `resolveAll`
// here is its edits.resolve, for display. Keep the two in sync.

const ED = {
  loaded: false,
  groups: [], // series number of every photo, parallel to S.photos
  taken: [], // capture time of every photo, 'YYYY:MM:DD HH:MM:SS' 
  edits: {}, // name -> settings, on the photo where a run of settings starts
  raw: [], // whether each photo is a raw file
  defaults: null,
  limits: null,
  pipeline: null,
  settings: new Map(), // name -> settings it gets
  kept: [], // kept raw photos, in shooting order
  pos: -1,
  live: null, // the current photo's settings, including a drag not saved yet
  before: false, // showing the photo without edits
  dev: null,
  shown: null, // name of the photo whose data is on the GPU
  previews: new Map(), // name -> { promise, abort }
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

function resolveAll() {
  ED.settings.clear();
  let current = ED.defaults;
  S.photos.forEach((name, i) => {
    if (i === 0 || ED.groups[i] !== ED.groups[i - 1]) current = ED.defaults;
    if (ED.edits[name]) current = ED.edits[name];
    ED.settings.set(name, current);
  });
}

function seriesOf(i) {
  // [first, last] photo index of the series photo i is in.
  let a = i;
  let b = i;
  while (a > 0 && ED.groups[a - 1] === ED.groups[i]) a--;
  while (b + 1 < S.photos.length && ED.groups[b + 1] === ED.groups[i]) b++;
  return [a, b];
}

function runOf(i) {
  // [first, last] index of the photos sharing photo i's settings: a change at the first
  // of them would reach all of them.
  let a = i;
  let b = i;
  while (a > 0 && ED.groups[a - 1] === ED.groups[i] && !ED.edits[S.photos[a]]) a--;
  while (b + 1 < S.photos.length && ED.groups[b + 1] === ED.groups[i] && !ED.edits[S.photos[b + 1]]) b++;
  return [a, b];
}

function keptBetween(a, b) {
  let n = 0;
  for (let i = a; i <= b; i++) if (S.marks[S.photos[i]] === 'keep') n++;
  return n;
}

const isDefault = (s) => Object.keys(ED.defaults).every((k) => s[k] === ED.defaults[k]);

// --- Sliders -------------------------------------------------------------------------------

const [TEMP_MIN, TEMP_MAX] = [2000, 15000];
const miredOf = (t) => 1e6 / t;
// The temperature slider is linear in mired, which is closer to how different temperatures look.
const tempToSlider = (t) => Math.round(((miredOf(TEMP_MIN) - miredOf(t)) / (miredOf(TEMP_MIN) - miredOf(TEMP_MAX))) * 1000);
const sliderToTemp = (v) => Math.round(1e6 / (miredOf(TEMP_MIN) - (v / 1000) * (miredOf(TEMP_MIN) - miredOf(TEMP_MAX))) / 50) * 50;

function asShotTemp() {
  return ED.dev && ED.shown === editCurrent() ? ED.dev.info.asShotTemp : null;
}

function formatValue(key, v) {
  if (key === 'exposure') return `${v > 0 ? '+' : ''}${v.toFixed(2)} EV`;
  if (key === 'saturation') return `${v > 0 ? '+' : ''}${v}`;
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
      ED.edits = res.edits;
      resolveAll();
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

function preview(name) {
  let p = ED.previews.get(name);
  if (!p) {
    const abort = new AbortController();
    const promise = fetch(`/develop/${enc(name)}?w=${previewSide()}&v=${S.version}`, { signal: abort.signal })
      .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.statusText))))
      .then(parsePreview);
    promise.catch(() => { if (ED.previews.get(name)?.promise === promise) ED.previews.delete(name); });
    p = { promise, abort };
    ED.previews.set(name, p);
  }
  return p.promise;
}

let drawQueued = false;
function requestDraw() {
  if (drawQueued) return;
  drawQueued = true;
  requestAnimationFrame(() => {
    drawQueued = false;
    if (ED.dev && ED.shown && ED.shown === editCurrent()) ED.dev.draw(ED.before ? ED.defaults : ED.live);
  });
  $('#editnote').textContent = ED.before ? 'Before' : (ED.shown === editCurrent() ? '' : 'Developing…');
}

function layoutEdit() {
  const W = editView.clientWidth;
  const H = editView.clientHeight;
  const onGpu = ED.dev && ED.shown && ED.shown === editCurrent();
  const el = onGpu ? editCanvas : editPlaceholder;
  editCanvas.hidden = !onGpu;
  editPlaceholder.hidden = onGpu || !editCurrent();
  const iw = onGpu ? ED.dev.info.width : editPlaceholder.naturalWidth || 3;
  const ih = onGpu ? ED.dev.info.height : editPlaceholder.naturalHeight || 2;
  const s = Math.min(W / iw, H / ih);
  const w = iw * s;
  const h = ih * s;
  Object.assign(el.style, { left: `${(W - w) / 2}px`, top: `${(H - h) / 2}px`, width: `${w}px`, height: `${h}px` });
  if (onGpu) {
    const cw = Math.round(w * dpr());
    const ch = Math.round(h * dpr());
    if (editCanvas.width !== cw || editCanvas.height !== ch) {
      editCanvas.width = cw;
      editCanvas.height = ch;
    }
    requestDraw();
  }
}
editPlaceholder.addEventListener('load', layoutEdit);

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
  ED.live = { ...ED.settings.get(name) };
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
  for (const [n, p] of ED.previews) {
    if (!near.includes(n)) {
      p.abort.abort();
      ED.previews.delete(n);
    }
  }
  preview(name)
    .then((info) => {
      if (editCurrent() !== name || S.mode !== 'edit') return;
      if (ED.shown !== name) {
        ED.dev.load(info);
        ED.shown = name;
      }
      updateSliders(); // the as-shot temperature is known now
      layoutEdit();
      // Then the neighbours, so the next photo is ready when you get there.
      for (const n of near.slice(1)) preview(n).catch(() => {});
    })
    .catch((err) => {
      if (err.name !== 'AbortError' && editCurrent() === name) toast(`Couldn't develop ${name}: ${err.message}`, true);
    });
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
      else if (runOf(i)[0] > j) sep.className = 'split';
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
    cell.classList.toggle('edited', !isDefault(ED.settings.get(n)));
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
      groups: st.groups, taken: st.taken, edits: st.edits, raw: st.raw, defaults: st.defaults, limits: st.limits, pipeline: st.pipeline,
    });
    ED.loaded = true;
    try {
      ED.dev = new Developer(editCanvas, ED.pipeline);
    } catch (err) {
      toast(`Can't develop in this browser: ${err.message}`, true);
    }
  }
  resolveAll();
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
    case 'Escape':
      if (ED.before) {
        ED.before = false;
        requestDraw();
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
