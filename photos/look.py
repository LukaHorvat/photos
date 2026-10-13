"""The camera's own look, as the default rendering.

  tone    Fitted per photo from the JPEG the camera embedded in the raw. The camera
          adapts its tone curve to each scene (Nikon's Active D-Lighting), and this
          way editing starts from what you saw while picking.
  colour  A profile per camera model (photos/profiles/<model>.npz), fitted once by
          scripts/calibrate_camera.py from raw + JPEG pairs: output colour by scene
          chromaticity and output brightness, including how bright colours fade to
          white and how much colour the camera takes out of dark tones at high ISO.

The develop shader (web/develop.frag) applies both as lookup textures. `model` below
is the same mapping in numpy, for fitting; the tests check the two agree.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
GRID = np.linspace(-14.0, 0.5, 117)  # tone curve samples: output luminance at these log2 scene luminances
PROFILES = Path(__file__).parent / "profiles"
TABLE_SIZE, TABLE_LEVELS = 33, 8
CONTRAST_MAX = 0.45  # contrast curve strength at +-100; above 0.5 it would stop being monotonic
WORK_SCALE = 4  # fitting a photo's tone happens at a quarter of its JPEG's size

# Without an embedded JPEG: one curve fitted to D3500 JPEGs, y = (1 + K) t^C / (t^C + K), t = G x.
FALLBACK_CURVE = (0.596, 1.650, 0.029)

_SRGB_TO_LINEAR = np.where(
    np.arange(256) / 255 <= 0.04045, np.arange(256) / 255 / 12.92, ((np.arange(256) / 255 + 0.055) / 1.055) ** 2.4
).astype(np.float32)


# --- The mapping -----------------------------------------------------------------------------


def tone_at(tone: np.ndarray, lum: np.ndarray) -> np.ndarray:
    return np.interp(np.log2(np.maximum(lum, 2.0**-16)), GRID, tone) * (lum > 0)


def tone_inverse(tone: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 ** np.interp(y, tone, GRID)


def fallback_tone() -> np.ndarray:
    g, c, k = FALLBACK_CURVE
    t = (2.0**GRID * g) ** c
    return np.minimum((1 + k) * t / (t + k), 1) + np.arange(GRID.size) * 1e-7


def contrast_curve(y: np.ndarray, contrast: float) -> np.ndarray:
    """An S-curve on roughly perceptual lightness (y^(1/2.4)); black and white stay put."""
    s = contrast / 100 * CONTRAST_MAX
    p = np.clip(y, 0, 1) ** (1 / 2.4)
    return np.clip(p + s * (p - 0.5) * 4 * p * (1 - p), 0, 1) ** 2.4


def chromaticity(rgb: np.ndarray) -> np.ndarray:
    """(r - g, b - g) of rgb / (r + g + b): in [-1, 1] for positive colours; neutral if there's no light."""
    total = rgb.sum(-1, keepdims=True)
    c = np.where(total > 1e-6, rgb / np.where(total > 1e-6, total, 1), 1 / 3)
    return np.clip(np.stack([c[..., 0] - c[..., 1], c[..., 2] - c[..., 1]], -1), -1, 1)


def identity_table(n: int = TABLE_SIZE, levels: int = TABLE_LEVELS) -> np.ndarray:
    """Output colour = input chromaticity, at luminance 1: a hue- and saturation-preserving tone curve."""
    u = np.linspace(-1, 1, n)
    cr, cb = np.meshgrid(u, u, indexing="ij")
    g = (1 - cr - cb) / 3
    rgb = np.stack([g + cr, g, g + cb], -1)
    rgb /= (rgb @ LUMA)[..., None]
    return np.repeat(rgb[:, :, None], levels, 2).astype(np.float32)


def lookup(table: np.ndarray, c: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Trilinear table lookup sampled like the shader's 3D texture: chromaticity grid nodes at the
    ends of [-1, 1], brightness levels centred in y^(1/2.4)."""
    n, levels = table.shape[0], table.shape[2]
    q = (c + 1) / 2 * (n - 1)
    i = np.clip(np.floor(q).astype(int), 0, n - 2)
    f = (q - i).astype(np.float32)
    e = np.clip(y, 0, 1) ** (1 / 2.4) * levels - 0.5
    j = np.clip(np.floor(e).astype(int), 0, levels - 2)
    fj = np.clip(e - j, 0, 1).astype(np.float32)
    out = np.zeros(y.shape + (3,), np.float32)
    for da in (0, 1):
        for db in (0, 1):
            for dj in (0, 1):
                w = (f[..., 0] if da else 1 - f[..., 0]) * (f[..., 1] if db else 1 - f[..., 1]) * (fj if dj else 1 - fj)
                out += w[..., None] * table[i[..., 0] + da, i[..., 1] + db, j + dj]
    return out


def lightness(y: np.ndarray) -> np.ndarray:
    return np.where(y > 216 / 24389, 116 * np.cbrt(np.maximum(y, 0)) - 16, 24389 / 27 * y)


def model(scene: np.ndarray, tone: np.ndarray, table: np.ndarray, shadow_loss: float = 0, contrast: float = 0) -> np.ndarray:
    """Scene-linear sRGB -> display-linear sRGB, as the shader's last steps do."""
    y = contrast_curve(tone_at(tone, scene @ LUMA), contrast)
    ratio = lookup(table, chromaticity(scene), y)
    keep = 1 - shadow_loss * np.exp(-lightness(y) / 15)
    return np.clip((1 + (ratio - 1) * keep[..., None]) * y[..., None], 0, 1)


# --- Camera profiles -----------------------------------------------------------------------------


@dataclass
class Profile:
    """A camera's colour rendering: table[chroma r-g, chroma b-g, brightness level] = output colour
    at luminance 1; at high ISO a share of colour, fading with lightness, is taken out of dark tones."""

    table: np.ndarray
    shadow_slope: float = 0.0  # share of colour lost in the darkest tones, per doubling of ISO above shadow_iso
    shadow_iso: float = 1e9

    def shadow_loss(self, iso: float | None) -> float:
        if not iso:
            return 0.0
        return float(np.clip(self.shadow_slope * np.log2(iso / self.shadow_iso), 0, 1))

    def save(self, path: Path):
        np.savez_compressed(path, table=self.table.astype(np.float32), shadow=np.array([self.shadow_slope, self.shadow_iso]))

    @classmethod
    def identity(cls) -> Profile:
        return cls(identity_table())


def profile_path(camera: str) -> Path:
    return PROFILES / (re.sub(r"[^A-Za-z0-9]+", "_", camera).strip("_") + ".npz")


@cache
def load_profile(camera: str | None) -> Profile:
    """The camera's profile, or a neutral one if it hasn't been calibrated."""
    if camera and profile_path(camera).exists():
        with np.load(profile_path(camera)) as f:
            slope, iso = f["shadow"]
            return Profile(f["table"], float(slope), float(iso))
    return Profile.identity()


# --- Fitting a photo's tone curve to its camera JPEG ----------------------------------------------


@dataclass
class Samples:
    """Scene vs camera colour (both linear) from one raw and its camera JPEG: averages of evenly
    coloured 16x16-pixel blocks, and single pixels near sensor clipping (P: brightest raw channel)."""

    S: np.ndarray
    C: np.ndarray
    hS: np.ndarray
    hC: np.ndarray
    hP: np.ndarray


def scene_rgb(rgb: np.ndarray, color) -> np.ndarray:
    """Camera RGB -> white-balanced scene-linear sRGB, as the shader does it (as shot)."""
    mul = color.as_shot.astype(np.float32)
    return np.minimum(rgb * mul, mul.min()) @ color.rgb_cam.T.astype(np.float32)


def samples(rgb: np.ndarray, color, jpeg: bytes, max_blocks: int = 20000, max_pixels: int = 30000, seed: int = 0) -> Samples | None:
    """rgb: camera RGB at any size (half from the preview's decode, full from an export's),
    oriented; jpeg: the camera's JPEG with its orientation tag. The work happens at a quarter
    of the JPEG's size, where a 4 x 4 block covers 16 x 16 full-size pixels."""
    h, w = rgb.shape[:2]
    im = Image.open(io.BytesIO(jpeg))
    full_w, full_h = im.size
    im.draft("RGB", (full_w // WORK_SCALE, full_h // WORK_SCALE))  # cheap: the decoder scales the DCT
    im = ImageOps.exif_transpose(im.convert("RGB"))
    if (im.size[0] < im.size[1]) != (full_w < full_h):  # rotated by its orientation tag
        full_w, full_h = full_h, full_w
    qw, qh = full_w // WORK_SCALE, full_h // WORK_SCALE
    if im.size != (qw, qh):
        im = im.resize((qw, qh), Image.Resampling.BOX)
    # The JPEG is a centred crop of the sensor, k rgb pixels per JPEG pixel: w - 2m = k jw, h - 2m = k jh.
    k = (w - h) / (full_w - full_h) if full_w != full_h else w / full_w
    fw, fh = k * full_w, k * full_h
    x0, y0 = (w - fw) / 2, (h - fh) / 2
    if x0 < -0.5 or y0 < -0.5:
        return None
    field = (x0, y0, x0 + fw, y0 + fh)
    raw = np.stack(
        [np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c])).resize((qw, qh), Image.Resampling.BOX, box=field)) for c in range(3)],
        -1,
    )
    cam8 = np.asarray(im)
    cam = _SRGB_TO_LINEAR[cam8]
    scene = scene_rgb(raw, color)

    b = 16 // WORK_SCALE
    bh, bw = (qh // b) * b, (qw // b) * b

    def block_mean(a):  # per channel, with Pillow's box filter: much faster than numpy here
        size = (bw // b, bh // b)
        return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(a[:bh, :bw, c])).resize(size, Image.Resampling.BOX)) for c in range(3)], -1)

    S = block_mean(scene).reshape(-1, 3)
    C = block_mean(cam).reshape(-1, 3)
    e = cam8.astype(np.float32) / 255
    spread = np.sqrt(np.maximum(block_mean(e * e) - block_mean(e) ** 2, 0)).max(-1).reshape(-1)
    ok = (spread < 0.03) & (C.max(1) < 0.985) & (S @ LUMA > 2**-12) & (S.sum(1) > 0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(np.flatnonzero(ok))[:max_blocks]

    # Near-clipped pixels from smooth areas (so a pixel of misalignment doesn't matter)
    P = np.maximum(np.maximum(raw[..., 0], raw[..., 1]), raw[..., 2])
    g8 = e @ LUMA
    g = np.abs(np.diff(g8, axis=0))[:, :-1] + np.abs(np.diff(g8, axis=1))[:-1]
    smooth = np.zeros(P.shape, bool)
    smooth[1:-1, 1:-1] = (g[:-1, :-1] < 0.03) & (g[1:, 1:] < 0.03)
    hidx = rng.permutation(np.flatnonzero((P > 0.7) & smooth))[:max_pixels]
    flat = lambda a: a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1)
    return Samples(S[idx], C[idx], flat(scene)[hidx], flat(cam)[hidx], flat(P)[hidx])


def fit_curve(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Monotone curve y = f(x) sampled at 2^GRID: median per half-stop, smoothed."""
    lx = np.log2(np.maximum(x, 2.0**-16))
    med = np.full(GRID.shape, np.nan)
    for i, g in enumerate(GRID):
        sel = (lx >= g - 0.25) & (lx < g + 0.25)
        if sel.sum() > 15:
            med[i] = np.median(y[sel])
    ok = ~np.isnan(med)
    if ok.sum() < 3:
        return None
    curve = np.interp(GRID, GRID[ok], med[ok])
    lo = GRID < GRID[ok][0]
    curve[lo] = med[ok][0] * 2.0 ** (2.0 * (GRID[lo] - GRID[ok][0]))  # fade to black below the data
    hi = GRID > GRID[ok][-1]
    curve[hi] = np.minimum(1, med[ok][-1] + (GRID[hi] - GRID[ok][-1]) * 0.3)
    curve = np.maximum.accumulate(curve)
    curve = np.convolve(np.pad(curve, 2, mode="edge"), np.ones(5) / 5, mode="valid")
    return curve + np.arange(curve.size) * 1e-7  # strictly increasing, so it can be inverted


def fit_tone(s: Samples) -> np.ndarray | None:
    """The photo's tone curve. Blocks set it up to the brightest evenly coloured areas; the
    near-clipped pixels set the top, where the camera reaches white before the sensor clips."""
    tone = fit_curve(s.S @ LUMA, s.C @ LUMA)
    if tone is None or len(s.hS) < 500:
        return tone
    x, y = s.hS @ LUMA, s.hC @ LUMA
    edge = np.log2(np.percentile(x[x > 0], 5))
    keep = np.repeat(GRID[GRID < edge], 40)  # enough samples per bin for fit_curve to keep it there
    refit = fit_curve(np.r_[2.0**keep, x], np.r_[np.interp(keep, GRID, tone), y])
    return tone if refit is None else refit


@dataclass
class Look:
    tone: np.ndarray
    table: np.ndarray
    shadow_loss: float
    anchor: float  # log2 scene luminance the tone curve turns into mid-grey: compression pivots here

    @classmethod
    def for_photo(cls, rgb: np.ndarray, color, jpeg: bytes | None, camera: str | None, iso: float | None) -> Look:
        """rgb: the photo's camera RGB, at any size."""
        profile = load_profile(camera)
        tone = None
        if jpeg:
            s = samples(rgb, color, jpeg)
            tone = fit_tone(s) if s is not None else None
        if tone is None:
            tone = fallback_tone()
        anchor = float(np.log2(tone_inverse(tone, 0.18)))
        return cls(tone.astype(np.float32), profile.table, profile.shadow_loss(iso), anchor)
