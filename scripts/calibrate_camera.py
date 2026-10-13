"""Fit a camera's colour profile to its own JPEGs: photos/profiles/<model>.npz.

    uv run python scripts/calibrate_camera.py /mnt/d/Radno [more folders...]

Every sampled raw is decoded at half size and compared with the JPEG embedded in it
(look.samples): averages of evenly coloured blocks, and near-clipped pixels. From those:
  - each photo's tone curve (look.fit_tone, as the app does it),
  - a camera-wide colour table: output colour (at luminance 1) by scene chromaticity and
    output brightness, the median over blocks in each cell; its brightest level comes
    from near-clipped pixels, so bright colours fade to white like the camera's,
  - at high ISO, how much colour the camera takes out of dark tones.
Prints the colour difference to the camera on photos not used for fitting.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import numpy as np  # noqa: E402

from photos import develop, look  # noqa: E402
from photos.raw import RAW_EXTS, Photo, read_file  # noqa: E402

N, LEVELS = look.TABLE_SIZE, look.TABLE_LEVELS
HIGH_ISO = 1600  # above this the camera starts taking colour out of shadows


def load(path: Path):
    photo = Photo(path)
    data = read_file(path)
    jpeg = photo.embedded_jpeg(data)
    if jpeg is None:
        return None
    rgb, color = develop.decode(data, half_size=True)
    s = look.samples(rgb, color, jpeg, max_blocks=6000, max_pixels=8000)
    tone = look.fit_tone(s) if s is not None else None
    if tone is None or len(s.S) < 200:
        return None
    return path.name, photo.iso or 0, photo.camera, s, tone


def lab(lin):
    xyz = np.clip(lin, 0, None) @ develop.SRGB_TO_XYZ.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def cell(c):
    return np.clip(np.round((c + 1) / 2 * (N - 1)).astype(int), 0, N - 1)


def level(y):
    return np.clip((np.clip(y, 0, 1) ** (1 / 2.4) * LEVELS).astype(int), 0, LEVELS - 1)


def binned_median(keys, values, shape, minimum=10):
    """Median of values (n x 3) per bin; NaN where there are fewer than `minimum`."""
    flat = np.ravel_multi_index(keys, shape)
    order = np.argsort(flat, kind="stable")
    flat, values = flat[order], values[order]
    out = np.full((int(np.prod(shape)), 3), np.nan)
    count = np.zeros(int(np.prod(shape)))
    starts = np.flatnonzero(np.r_[True, flat[1:] != flat[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], flat.size]):
        count[flat[a]] = b - a
        if b - a >= minimum:
            out[flat[a]] = np.median(values[a:b], 0)
    return out.reshape(*shape, 3), count.reshape(shape)


def smooth_fill(T, count, fallback):
    """Smooth the difference from `fallback` across neighbouring chromaticities, fading to
    `fallback` where there is little data."""
    w = np.minimum(1, count / 100)
    D = np.where(np.isnan(T), 0, T - fallback) * w[..., None]
    k = np.exp(-0.5 * (np.arange(-3, 4) / 1.5) ** 2)
    k /= k.sum()
    for ax in (0, 1):
        D = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 3, mode="edge"), k, mode="valid"), ax, D)
        w = np.apply_along_axis(lambda v: np.convolve(np.pad(v, 3, mode="edge"), k, mode="valid"), ax, w)
    return fallback + D / np.maximum(w, 1e-3)[..., None] * np.minimum(1, w * 3)[..., None]


def fit_table(train):
    identity = look.identity_table()[:, :, 0]
    keys2, keys3, targets = [], [], []
    for name, iso, camera, s, tone in train:
        lc = s.C @ look.LUMA
        ok = lc > 1e-4
        c = cell(look.chromaticity(s.S[ok]))
        keys2.append(c)
        keys3.append(np.c_[c, level(look.tone_at(tone, s.S[ok] @ look.LUMA))])
        targets.append(s.C[ok] / lc[ok, None])
    k2, k3, t = np.concatenate(keys2), np.concatenate(keys3), np.concatenate(targets)
    T2, n2 = binned_median((k2[:, 0], k2[:, 1]), t, (N, N))
    T2 = smooth_fill(T2, n2, identity)
    T3, n3 = binned_median((k3[:, 0], k3[:, 1], k3[:, 2]), t, (N, N, LEVELS))
    T = np.stack([smooth_fill(T3[:, :, j], n3[:, :, j], T2) for j in range(LEVELS)], 2)
    # Near-clipped pixels: what the camera does with colours at the top of its tone curve.
    acc, cnt = np.zeros(T.shape), np.zeros(T.shape[:3])
    for name, iso, camera, s, tone in train:
        lc = s.hC @ look.LUMA
        ok = lc > 0.05
        if not ok.any():
            continue
        c = cell(look.chromaticity(s.hS[ok]))
        j = level(look.tone_at(tone, s.hS[ok] @ look.LUMA))
        np.add.at(acc, (c[:, 0], c[:, 1], j), s.hC[ok] / lc[ok, None])
        np.add.at(cnt, (c[:, 0], c[:, 1], j), 1)
    w = np.minimum(1, cnt / 50)[..., None]
    T = T * (1 - w) + acc / np.maximum(cnt, 1)[..., None] * w
    return T.astype(np.float32)


def colour_error(samples, table, shadow=lambda iso: 0.0, dark_only=False):
    e, ec = [], []
    for name, iso, camera, s, tone in samples:
        pred = look.model(s.S, tone, table, shadow(iso))
        a, b = lab(pred), lab(s.C)
        sel = b[:, 0] < 40 if dark_only else slice(None)
        d = (a - b)[sel]
        e.append(np.sqrt((d**2).sum(-1)))
        ec.append(np.sqrt((d[:, 1:] ** 2).sum(-1)))
    return np.concatenate(e).mean(), np.concatenate(ec).mean()


def fit_shadow(high, table):
    """Share of colour taken out of the darkest tones, rising with ISO: slope * log2(ISO / iso0)."""
    if len(high) < 3:
        return 0.0, 1e9
    best = None
    for iso0 in np.geomspace(400, 3200, 13):
        for slope in np.arange(0, 1.01, 0.05):
            rule = lambda iso: float(np.clip(slope * np.log2(iso / iso0), 0, 1))
            err = colour_error(high, table, rule, dark_only=True)[1]
            if best is None or err < best[0]:
                best = (err, slope, iso0)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--every", type=int, default=4, help="fit on every n-th low-ISO photo")
    args = ap.parse_args()

    files = sorted(p for f in args.folders for p in f.iterdir() if p.suffix.lower() in RAW_EXTS)
    with ThreadPoolExecutor(8) as ex:
        isos = list(ex.map(lambda p: Photo(p).iso or 0, files))
    low = [p for p, i in zip(files, isos) if i <= HIGH_ISO]
    high = [p for p, i in zip(files, isos) if i > HIGH_ISO]
    train_files, test_files = low[:: args.every], low[args.every // 2 :: args.every * 2]
    with ThreadPoolExecutor(6) as ex:
        train = [s for s in ex.map(load, train_files) if s]
        test = [s for s in ex.map(load, test_files) if s]
        high = [s for s in ex.map(load, high) if s]
    cameras = {s[2] for s in train}
    if len(cameras) != 1:
        sys.exit(f"Expected photos from one camera, found {sorted(map(str, cameras))}")
    camera = cameras.pop()
    print(f"{camera}: {len(train)} photos to fit on, {len(test)} to test, {len(high)} above ISO {HIGH_ISO}")

    table = fit_table(train)
    slope, iso0 = fit_shadow(high, table)
    profile = look.Profile(table, slope, iso0)

    print("colour difference to the camera's JPEG (CIELAB dE; colour-only in brackets):")
    neutral = look.identity_table()
    for label, samples in [("test photos", test), (f"photos above ISO {HIGH_ISO}", high)]:
        if not samples:
            continue
        before = colour_error(samples, neutral)
        after = colour_error(samples, table, profile.shadow_loss)
        print(f"  {label:24s} tone only {before[0]:.2f} ({before[1]:.2f})  ->  with profile {after[0]:.2f} ({after[1]:.2f})")
    print(f"high ISO: colour taken out of the darkest tones = {slope:.2f} x log2(ISO / {iso0:.0f})")
    look.PROFILES.mkdir(exist_ok=True)
    out = look.profile_path(camera)
    profile.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
