"""Noise reduction for exports, modelled on the camera's JPEGs (as look.py is for colour).

1. Raw: non-local means on the four Bayer planes jointly (R, G1, G2, B as one 4-channel
   image, so every channel is averaged with the same weights and colours stay put), after a
   generalized Anscombe transform with this photo's measured noise (variance = A * signal
   + B): the noise is ~1 everywhere, so one strength fits every brightness and ISO. The
   weights come from there, the raw values are averaged (unbiased, see gpu.py). On the GPU;
   the result goes back into LibRaw's buffer and gets demosaiced as usual.
2. After demosaicing: colour noise smoothed by a guided filter that follows brightness
   edges (gpu.py), at half resolution like a JPEG's colour.
3. Part of the removed luminance noise is given back, separately per scale, so the result
   has the camera's fine grain instead of looking plastic (in the develop shader).

Luminance and colour noise reduction have strengths of their own (strength()), as in
Lightroom. Weaker than standard, the standard result is mixed with the photo demosaiced
without noise reduction, its luminance and its colour separately: a weaker filter, or noise
left in the mosaic, comes out of AHD with dark worms at high ISO, and giving more grain back
can't bring back colour noise.

Instead of steps 1 and the demosaicing, a photo can have RawNIND do both (neural.py); steps 2
and 3 and the mixing stay.
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
GRAIN_FLOOR = 0.65  # and above it less, down to this share of GRAIN, reached at ISO ~10000 (the camera's texture at 11400-25600)
CHROMA_RADIUS = 3  # guided filter radius for colour, half-resolution pixels; larger bleeds skin colour into irises
CHROMA_EPS = 0.004**2
PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))  # (row, column) of each Bayer plane within a 2x2 tile
CEILING = 1.5  # a photo measuring noisier than this times the camera's typical noise at its ISO,
CEILING_LEVEL = 0.18  # at this level (of the raw range: mid-grey), is denoised as typical instead


@dataclass(frozen=True)
class Reduction:
    """How a photo's noise is reduced: luminance and colour strengths (strength()), and whether
    by RawNIND (neural.py) in place of non-local means and AHD."""

    luminance: float
    colour: float
    neural: bool = False

    def __bool__(self) -> bool:  # any noise reduction at all
        return bool(self.luminance or self.colour)


@dataclass
class Noise:
    """What the develop step needs to finish noise reduction after demosaicing."""

    grain: np.ndarray  # half-size scene luminance removed by NLM, oriented like the image
    as_shot: np.ndarray  # white balance and colour matrix the colour noise is judged in
    rgb_cam: np.ndarray
    weights: tuple  # how much of the grain is given back per scale (grain_weights())
    chroma_eps: float = CHROMA_EPS  # colour smoothing (chroma_eps())
    amount: tuple = (1.0, 1.0)  # luminance, colour: how much is noise reduced, the rest the photo without (gpu.Decoded.original)

    @classmethod
    def at(cls, grain: np.ndarray, color, iso: int | None, reduction: Reduction) -> Noise:
        """For a Reduction; color: develop.Color."""
        lum, col = reduction.luminance, reduction.colour
        return cls(grain, color.as_shot, color.rgb_cam, grain_weights(iso, lum), chroma_eps(col), (amount(lum), amount(col)))


def grain_weights(iso: int | None, s: float = 1.0) -> tuple:
    """How much of the removed noise is given back, per scale: at strength `s` 1 (strength())
    the camera's texture for this ISO; above 1 less, none at 2. Below 1, amount() of that:
    the rest of the photo is the one without noise reduction."""
    m = 1 + GRAIN_PER_STOP * np.log2(GRAIN_ISO / iso) if iso else 1
    share = amount(s) if s <= 1 else max(2 - s, 0.0)
    return tuple(float(np.clip(m, GRAIN_FLOOR, 3)) * w * share for w in GRAIN)


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


def strength(setting: float | None, iso: int | None) -> float:
    """How strong a photo's luminance or colour noise reduction is, 1 standard, 0 none: its
    setting in percent, or automatically, standard from AUTO_ISO up."""
    if setting is None:
        return 1.0 if iso and iso >= AUTO_ISO else 0.0
    return min(max(setting / 100, 0.0), 2.0)


def chroma_eps(s: float) -> float:
    """The colour guided filter's eps at strength `s`: stronger above 1."""
    return CHROMA_EPS * max(1.0, s) ** 2


def amount(s: float) -> float:
    """How much of a photo is the standard result at strength `s`, the rest the photo without
    noise reduction. Below 1 the noise left falls as (1 - s)^2: each step looks about as big
    (noise halved looks far from half gone; mixed linearly, little changed below ~60%)."""
    return 1 - (1 - min(s, 1.0)) ** 2


def mixed(reduction: Reduction) -> bool:
    """Whether a Reduction mixes in the photo without noise reduction."""
    return min(amount(reduction.luminance), amount(reduction.colour)) < 1


def supported(raw) -> bool:
    return raw.raw_pattern is not None and np.asarray(raw.raw_pattern).shape == (2, 2)


def raw_nlm(raw, black: float, scale: float, color, typical: tuple[float, float] | None = None, s: float = 1.0) -> tuple[np.ndarray, int]:
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

    Strength `s` (strength(), of luminance noise reduction): above 1, stronger averaging.
    Below 1 it's the same, and the result gets mixed with the photo without (amount()).

    scale: raw units -> our normalised camera RGB (1 / (saturation - black))."""
    from . import gpu

    args = nlm_args(raw, black, scale, color, typical, s)
    denoised, removed = gpu.raw_nlm(raw.raw_image_visible, *args)
    raw.raw_image_visible[...] = denoised
    return removed, args[-1]


def nlm_args(raw, black: float, scale: float, color, typical: tuple[float, float] | None = None, s: float = 1.0) -> tuple:
    """raw_nlm()'s arguments for gpu.raw_nlm() after the mosaic (the gain last)."""
    mosaic = raw.raw_image_visible
    pattern = np.asarray(raw.raw_pattern)
    colours = _colours(pattern)
    weights = luminance_weights(pattern, color, scale)
    gain = max(1, 65535 // max(int(raw.white_level), int(mosaic.max())))
    model = noise_model(mosaic, black, pattern)
    if typical:
        model = trusted(model, typical, CEILING_LEVEL * (raw.white_level - black))
    greens = tuple(k for k, c in enumerate(colours) if c == 1)
    return (black, *model, weights, STRENGTH * max(1.0, s), SEARCH, PATCH, greens, gain)


def _colours(pattern) -> list[int]:
    """The colour (0 R, 1 G, 2 B) of each photosite of a 2x2 tile, in PHASES order."""
    return [{0: 0, 1: 1, 2: 2, 3: 1}[int(pattern[p])] for p in PHASES]  # LibRaw calls the second green 3


def luminance_weights(pattern, color, scale: float) -> list[float]:
    """Per photosite of a 2x2 tile (PHASES order), its share of the tile's scene luminance (as-shot
    white balance); scale: raw units -> our normalised camera RGB."""
    colours = _colours(pattern)
    lum = (color.rgb_cam.T @ LUMA) * color.as_shot * scale  # luminance of each camera channel
    return [float(lum[c] / colours.count(c)) for c in colours]  # the two greens are averaged


def removed(mosaic: np.ndarray, rgb: np.ndarray, pattern, weights) -> np.ndarray:
    """What demosaicing and denoising to camera RGB `rgb` (H x W x 3) took out of `mosaic` (H x W,
    the same units), per 2x2 tile: sum(weights * removed) over its photosites (half size, float32),
    the grain's source like raw_nlm()'s."""
    out = np.zeros((mosaic.shape[0] // 2, mosaic.shape[1] // 2), np.float32)
    for (dy, dx), c, w in zip(PHASES, _colours(pattern), weights):
        site = np.s_[dy : 2 * out.shape[0] : 2, dx : 2 * out.shape[1] : 2]
        out += np.float32(w) * (mosaic[site] - rgb[site + (c,)])
    return out


def orient(img: np.ndarray, flip: int) -> np.ndarray:
    """Turn an image as stored on the sensor like LibRaw turns the developed one."""
    return {0: img, 3: img[::-1, ::-1], 5: np.rot90(img, 1), 6: np.rot90(img, -1)}.get(flip, img)
