"""Noise reduction for exports, modelled on the camera's JPEGs (as look.py is for colour).

1. Raw: non-local means on the four Bayer planes jointly (R, G1, G2, B as one 4-channel
   image, so every channel is averaged with the same weights and colours stay put), after a
   generalized Anscombe transform with this photo's measured noise (variance = A * signal
   + B): the noise is ~1 everywhere, so one strength fits every brightness and ISO. On the
   GPU; the result goes back into LibRaw's buffer and gets demosaiced as usual.
2. After demosaicing: colour noise smoothed by a guided filter that follows brightness
   edges (gpu.py), at half resolution like a JPEG's colour.
3. Part of the removed luminance noise is given back, separately per scale, so the result
   has the camera's fine grain instead of looking plastic (in the develop shader).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .look import LUMA

AUTO_ISO = 1000  # automatic noise reduction from this ISO; below it the camera's JPEGs barely reduce noise either
STRENGTH = 0.68  # NLM filtering parameter h, in units of the (stabilised) noise
SEARCH = 15  # NLM search window, raw-plane pixels
PATCH = 5  # NLM patch size
GRAIN = (0.55, 0.4, 0.95)  # removed noise given back at GRAIN_ISO, fine, small, medium scale: the camera's texture
GRAIN_ISO = 5000
GRAIN_PER_STOP = 0.33  # the camera keeps more texture at lower ISOs: this much more grain per stop below GRAIN_ISO
CHROMA_RADIUS = 3  # guided filter radius for colour, half-resolution pixels
CHROMA_EPS = 0.004**2
PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))  # (row, column) of each Bayer plane within a 2x2 tile
CEILING = 1.5  # a photo measuring noisier than this times the camera's typical noise at its ISO,
CEILING_LEVEL = 0.18  # at this level (of the raw range: mid-grey), is denoised as typical instead


@dataclass
class Noise:
    """What the develop step needs to finish noise reduction after demosaicing."""

    grain: np.ndarray  # half-size scene luminance removed by NLM, oriented like the image
    as_shot: np.ndarray  # white balance and colour matrix the colour noise is judged in
    rgb_cam: np.ndarray
    weights: tuple  # how much of the grain is given back per scale (grain_weights)


def grain_weights(iso: int | None) -> tuple:
    m = 1 + GRAIN_PER_STOP * np.log2(GRAIN_ISO / iso) if iso else 1
    return tuple(float(np.clip(m, 0.5, 3)) * w for w in GRAIN)


def noise_samples(mosaic: np.ndarray, black: float, pattern: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Noise variance by brightness, measured in the photo: (signal levels, variances), raw
    units above black. From the two green photosites of every other 2x2 tile (same light,
    independent noise; every other is as accurate as all of them, 4x faster). Fine texture
    everywhere, like foliage, reads as noise too: see CEILING."""
    h, w = mosaic.shape[0] // 2 * 2, mosaic.shape[1] // 2 * 2
    greens = [p for p in PHASES if pattern[p] in (1, 3)]
    g1, g2 = (mosaic[dy:h:4, dx:w:4].astype(np.float32) - black for dy, dx in greens)
    m, d = (g1 + g2) / 2, g1 - g2
    xs, vs = [], []
    for lo, hi in itertools.pairwise(np.quantile(m, np.linspace(0.01, 0.995, 25))):
        dd = d[(m >= lo) & (m < hi)]
        if dd.size >= 100:
            xs.append((lo + hi) / 2)
            vs.append((np.median(np.abs(dd - np.median(dd))) / 0.6745) ** 2 / 2)
    return np.array(xs), np.array(vs)


def noise_model(mosaic: np.ndarray, black: float, pattern: np.ndarray) -> tuple[float, float]:
    """(A, B): variance = A * signal + B, raw units above black, measured in this photo."""
    A, B = np.polyfit(*noise_samples(mosaic, black, pattern), 1)
    return max(float(A), 1e-3), max(float(B), 1e-3)


def trusted(measured: tuple[float, float], typical: tuple[float, float], level: float) -> tuple[float, float]:
    """The measured noise (A, B), unless at `level` it's more than CEILING times the camera's
    typical noise: then that's texture, and the typical noise is used."""
    variance = lambda model: model[0] * level + model[1]
    return typical if variance(measured) > CEILING * variance(typical) else measured


def active(setting: float | None, iso: int | None) -> bool:
    """Whether a photo gets noise reduction: its setting (None: automatic, 0: off, 100: on)."""
    if setting is None:
        return bool(iso) and iso >= AUTO_ISO
    return setting > 0


def supported(raw) -> bool:
    return raw.raw_pattern is not None and np.asarray(raw.raw_pattern).shape == (2, 2)


def raw_nlm(raw, black: float, scale: float, color, typical: tuple[float, float] | None = None) -> tuple[np.ndarray, int]:
    """Denoise LibRaw's raw buffer in place (on the GPU).

    Returns the luminance removed, the grain's source (one value per 2x2 tile, scene units,
    as-shot white balance, not yet oriented), and the gain the buffer now has: denoised
    values are smooth to a fraction of a raw unit, and rounding them back to whole units
    leaves flat patches, different per colour, that show as colour blotches once deep
    shadows are lifted. So they're written as many times larger as 16 bits allow (16x for
    12-bit raws); black and white levels scale with them.

    The noise is measured in the photo itself. But a photo full of fine texture reads far
    noisier than it is (DSC_9331, ISO 200: 10x) and would be smeared, so if the camera's
    typical noise at this ISO is known (`typical`, look.Profile.noise_at) and the photo
    measures more than CEILING times that, the typical noise is used instead.

    scale: raw units -> our normalised camera RGB (1 / (saturation - black))."""
    from . import gpu

    args = nlm_args(raw, black, scale, color, typical)
    denoised, removed = gpu.raw_nlm(raw.raw_image_visible, *args)
    raw.raw_image_visible[...] = denoised
    return removed, args[-1]


def nlm_args(raw, black: float, scale: float, color, typical: tuple[float, float] | None = None) -> tuple:
    """raw_nlm()'s arguments for gpu.raw_nlm() after the mosaic (the gain last)."""
    mosaic = raw.raw_image_visible
    pattern = np.asarray(raw.raw_pattern)
    colours = [{0: 0, 1: 1, 2: 2, 3: 1}[int(pattern[p])] for p in PHASES]  # LibRaw calls the second green 3
    lum = (color.rgb_cam.T @ LUMA) * color.as_shot * scale  # luminance of each camera channel
    weights = [lum[c] / colours.count(c) for c in colours]  # the two greens are averaged
    gain = max(1, 65535 // max(int(raw.white_level), int(mosaic.max())))
    model = noise_model(mosaic, black, pattern)
    if typical:
        model = trusted(model, typical, CEILING_LEVEL * (raw.white_level - black))
    return (black, *model, weights, STRENGTH, SEARCH, PATCH, gain)


def orient(img: np.ndarray, flip: int) -> np.ndarray:
    """Turn an image as stored on the sensor like LibRaw turns the developed one."""
    return {0: img, 3: img[::-1, ::-1], 5: np.rot90(img, 1), 6: np.rot90(img, -1)}.get(flip, img)
