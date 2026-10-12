"""Raw development: the processing behind the editor and the exporter.

The editor previews with a WebGL shader (web/develop.js) that mirrors `render`
below step for step, so sliders respond instantly; exports run the same maths
here at full resolution. Keep the two in sync.

The pipeline, in order:
  1. white balance: colour temperature (Kelvin) -> camera RGB multipliers
  2. camera RGB -> linear sRGB, with the Adobe matrix LibRaw has for the camera
  3. dynamic range compression: the contrast of a smooth, edge-aware "base" layer
     of log luminance is reduced around mid-grey; local detail is kept as it is
  4. exposure (EV), after compression
  5. saturation
  6. base tone curve, fitted to the camera's own JPEGs, then sRGB encoding
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import rawpy
from PIL import Image

DEFAULTS = {"compress": 0, "exposure": 0.0, "temp": None, "saturation": 0}
LIMITS = {"compress": (0, 100), "exposure": (-4.0, 4.0), "temp": (2000, 15000), "saturation": (-100, 100)}
TEMP_STEP = 50  # Kelvin; temperatures are kept on this grid, so the editor can be sent all of them

# Base tone curve y = (1 + K) t^C / (t^C + K), t = GAIN * x, clipped to 1. Fitted by
# histogram matching against D3500 JPEGs (scripts/calibrate_curve.py).
CURVE_GAIN = 0.596
CURVE_C = 1.650
CURVE_K = 0.029

# Scene-linear value the curve turns into display mid-grey (0.18); compression pivots here.
ANCHOR = float(np.log2((0.18 * CURVE_K / (1 + CURVE_K - 0.18)) ** (1 / CURVE_C) / CURVE_GAIN))
MAX_COMPRESSION = 0.75  # at 100, the base layer keeps a quarter of its contrast
LOG_FLOOR = 2.0**-12  # ~ the sensor's noise floor; darker than this is not lifted further

# Edge-aware smoothing (guided filter) of log luminance for the base layer, run at low
# resolution; its per-pixel linear coefficients are upsampled and applied at full size.
BASE_LONG_SIDE = 512
BASE_RADIUS = 20  # pixels at BASE_LONG_SIDE
BASE_EPS = 0.3  # log2 units squared; edges stronger than ~0.5 stop are kept

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
SRGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]]
)
BLACK_PAD = 128  # raw units kept below the black level, so shadow noise isn't clipped at zero
RAW_LINEAR = dict(
    use_camera_wb=False,
    use_auto_wb=False,
    user_wb=[1.0, 1.0, 1.0, 1.0],
    output_color=rawpy.ColorSpace.raw,
    gamma=(1, 1),
    no_auto_bright=True,
    output_bps=16,
    adjust_maximum_thr=0,
)


# --- Colour temperature --------------------------------------------------------------


def planck_xy(temp: float) -> tuple[float, float]:
    """CIE xy of a black body at `temp` Kelvin (Kim et al. 2002 cubic fit, 1667-25000 K)."""
    t = min(max(temp, 1667.0), 25000.0)
    if t <= 4000:
        x = -0.2661239e9 / t**3 - 0.2343589e6 / t**2 + 0.8776956e3 / t + 0.179910
    else:
        x = -3.0258469e9 / t**3 + 2.1070379e6 / t**2 + 0.2226347e3 / t + 0.240390
    if t <= 2222:
        y = -1.1063814 * x**3 - 1.34811020 * x**2 + 2.18555832 * x - 0.20219683
    elif t <= 4000:
        y = -0.9549476 * x**3 - 1.37418593 * x**2 + 2.09137015 * x - 0.16748867
    else:
        y = 3.0817580 * x**3 - 5.87338670 * x**2 + 3.75112997 * x - 0.37001483
    return x, y


def temp_multipliers(cam_xyz: np.ndarray, temp: float) -> np.ndarray:
    """Camera RGB multipliers (green = 1) that make a `temp` Kelvin light source neutral."""
    x, y = planck_xy(temp)
    cam = cam_xyz @ np.array([x / y, 1.0, (1 - x - y) / y])
    mul = 1.0 / cam
    return mul / mul[1]


@dataclass
class Color:
    """Per-photo colour data: the camera matrix and the as-shot white balance."""

    cam_xyz: np.ndarray  # XYZ -> camera RGB
    rgb_cam: np.ndarray  # white-balanced camera RGB -> linear sRGB
    as_shot: np.ndarray  # camera's multipliers, green = 1
    tint: np.ndarray  # as_shot / temp_multipliers(as_shot_temp): the off-locus (green/magenta) part
    as_shot_temp: float

    @classmethod
    def from_raw(cls, raw) -> Color:
        cam_xyz = np.array(raw.rgb_xyz_matrix[:3], np.float64)
        cam_rgb = cam_xyz @ SRGB_TO_XYZ
        cam_rgb /= cam_rgb.sum(axis=1, keepdims=True)  # so white maps to white
        wb = np.array(raw.camera_whitebalance[:3], np.float64)
        if not wb.all():
            wb = 1.0 / cam_rgb.sum(axis=1)
        as_shot = wb / wb[1]
        # Temperature moves red against blue; find the one matching the camera's red/blue
        # ratio (bisection in mired; the ratio falls as mired rises) and keep the rest as tint.
        target = np.log(as_shot[0] / as_shot[2])
        lo, hi = 1e6 / 25000, 1e6 / 1667
        for _ in range(50):
            mid = (lo + hi) / 2
            m = temp_multipliers(cam_xyz, 1e6 / mid)
            if np.log(m[0] / m[2]) > target:  # too warm a light: go to more mired
                lo = mid
            else:
                hi = mid
        temp = 1e6 / ((lo + hi) / 2)
        tint = as_shot / temp_multipliers(cam_xyz, temp)
        return cls(cam_xyz, np.linalg.inv(cam_rgb), as_shot, tint / tint[1], temp)

    def multipliers(self, temp: float | None) -> np.ndarray:
        if temp is None:
            return self.as_shot
        return temp_multipliers(self.cam_xyz, temp) * self.tint

    def header(self) -> dict:
        lo, hi = LIMITS["temp"]
        return {
            "rgbCam": self.rgb_cam.tolist(),
            "asShot": self.as_shot.tolist(),
            "asShotTemp": round(self.as_shot_temp),
            # The multipliers for every temperature the editor can set, so it needn't compute them.
            "wb": {t: self.multipliers(t).tolist() for t in range(lo, hi + 1, TEMP_STEP)},
        }


# --- Pipeline ---------------------------------------------------------------------------


def clean(settings: dict | None) -> dict:
    """Settings with every field present, of the right type and within range."""
    out = dict(DEFAULTS)
    for key, value in (settings or {}).items():
        if key not in DEFAULTS or (value is None and key == "temp"):
            continue
        lo, hi = LIMITS[key]
        value = min(max(float(value), lo), hi)
        if key == "exposure":
            out[key] = round(value, 2)
        elif key == "temp":
            out[key] = int(round(value / TEMP_STEP) * TEMP_STEP)
        else:
            out[key] = int(round(value))
    return out


def to_srgb(rgb: np.ndarray, color: Color, coeffs: np.ndarray, settings: dict) -> np.ndarray:
    """camera RGB (HxWx3 float32, linear, saturation = 1) -> sRGB uint8. Modifies rgb."""
    s = clean(settings)
    mul = color.multipliers(s["temp"]).astype(np.float32)
    rgb *= mul
    # A channel at the sensor's ceiling, scaled by white balance, would turn highlights
    # pink; clip every channel where the least-scaled one tops out, so they stay white.
    # Values below zero (shadow noise) are kept until the tone curve: clipping them
    # earlier would lift the average of each channel, and with red and blue amplified
    # by white balance, turn lifted shadows purple.
    np.minimum(rgb, mul.min(), out=rgb)
    rgb = rgb @ color.rgb_cam.T.astype(np.float32)

    strength = s["compress"] / 100 * MAX_COMPRESSION
    if strength:
        lum = np.log2(np.maximum(rgb @ LUMA, LOG_FLOOR))
        a, b = upsample(coeffs, *rgb.shape[:2])
        base = a * lum + b
        rgb *= np.exp2(strength * (ANCHOR - base))[..., None]

    rgb *= np.float32(2.0 ** s["exposure"])

    if s["saturation"]:
        lum = (rgb @ LUMA)[..., None]
        rgb -= lum
        rgb *= np.float32(1 + s["saturation"] / 100)
        rgb += lum

    rgb = tone_curve(rgb)
    return np.round(oetf(rgb) * 255).astype(np.uint8)


def tone_curve(x: np.ndarray) -> np.ndarray:
    t = (np.maximum(x, 0) * CURVE_GAIN) ** CURVE_C
    return np.minimum((1 + CURVE_K) * t / (t + CURVE_K), 1.0)


def oetf(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def base_coefficients(rgb_cam: np.ndarray, color: Color) -> np.ndarray:
    """Guided-filter coefficients (h x w x 2: a, b) for the base layer, at low resolution.

    base = a * log2(luminance) + b, with a and b upsampled bilinearly: smooth over
    regions, but following strong edges, so compression doesn't halo around them.
    Computed once per photo from the as-shot rendering.
    """
    h, w = rgb_cam.shape[:2]
    scale = BASE_LONG_SIDE / max(h, w)
    lh, lw = max(1, round(h * scale)), max(1, round(w * scale))
    lin = np.minimum(rgb_cam * color.as_shot.astype(np.float32), color.as_shot.min()) @ color.rgb_cam.T.astype(np.float32)
    lum = np.log2(np.maximum(lin @ LUMA, LOG_FLOOR)).astype(np.float32)
    guide = np.asarray(Image.fromarray(lum).resize((lw, lh), Image.Resampling.BOX))
    r, eps = BASE_RADIUS, BASE_EPS
    mean = _box(guide, r)
    var = _box(guide * guide, r) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    return np.stack([_box(a, r), _box(b, r)], axis=-1).astype(np.float32)


def _box(x: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, shrinking the window at the borders."""

    def along(a, axis):
        n = a.shape[axis]
        c = np.cumsum(a, axis=axis, dtype=np.float64)
        c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
        hi = np.minimum(np.arange(n) + r + 1, n)
        lo = np.maximum(np.arange(n) - r, 0)
        shape = [1, 1]
        shape[axis] = n
        return (np.take(c, hi, axis=axis) - np.take(c, lo, axis=axis)) / (hi - lo).reshape(shape)

    return along(along(x, 0), 1).astype(np.float32)


def upsample(coeffs: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear upsampling sampled at pixel centres, like a GL texture with LINEAR filtering."""

    def axis(n_out, n_in):
        pos = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        i0 = np.floor(pos).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (pos - i0).astype(np.float32)

    h, w = coeffs.shape[:2]
    x0, x1, wx = axis(width, w)
    rows = coeffs[:, x0] * (1 - wx)[None, :, None] + coeffs[:, x1] * wx[None, :, None]
    y0, y1, wy = axis(height, h)
    out = rows[y0] * (1 - wy)[:, None, None] + rows[y1] * wy[:, None, None]
    return out[..., 0], out[..., 1]


# --- Entry points -------------------------------------------------------------------------


def decode(data: bytes, half_size: bool) -> tuple[np.ndarray, Color]:
    """Linear camera RGB (HxWx3 float32, black 0, saturation 1), oriented, and its colour data.

    Noise below the black level is kept (as negative values): LibRaw is given a black
    level BLACK_PAD lower than the real one, and the difference is subtracted here.
    """
    with rawpy.imread(io.BytesIO(data)) as raw:
        color = Color.from_raw(raw)
        black = min(raw.black_level_per_channel)
        # Lossy NEFs saturate a few codes under the nominal white level (see raw.Photo.clipping_mask).
        top = int(raw.raw_image_visible.max())
        saturation = top if top >= 0.98 * raw.white_level else raw.white_level
        pad = min(black, BLACK_PAD)
        rgb = raw.postprocess(half_size=half_size, user_black=black - pad, user_sat=saturation, **RAW_LINEAR)
    # LibRaw outputs (raw - user_black) * 65535 / user_sat.
    rgb = rgb.astype(np.float32)
    rgb *= np.float32(saturation / 65535 / (saturation - black))
    rgb -= np.float32(pad / (saturation - black))
    return rgb, color


def preview(data: bytes, long_side: int) -> bytes:
    """Everything the browser needs to develop this photo itself, as one binary blob:

    uint32 header length, JSON header, padding to 2 bytes, then float16 camera RGB
    (height x width x 3) and float16 base coefficients (ch x cw x 2).
    """
    import json

    rgb, color = decode(data, half_size=True)
    h, w = rgb.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale < 1:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        rgb = np.stack(
            [np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c])).resize(size, Image.Resampling.BOX)) for c in range(3)], -1
        )
        h, w = rgb.shape[:2]
    coeffs = base_coefficients(rgb, color)
    header = {
        "width": w,
        "height": h,
        "coeffWidth": coeffs.shape[1],
        "coeffHeight": coeffs.shape[0],
        **color.header(),
    }
    head = json.dumps(header).encode()
    head += b" " * (len(head) % 2)
    return b"".join(
        [
            len(head).to_bytes(4, "little"),
            head,
            rgb.astype(np.float16).tobytes(),
            coeffs.astype(np.float16).tobytes(),
        ]
    )


def develop(data: bytes, settings: dict, half_size: bool = False) -> np.ndarray:
    """Full pipeline on a whole raw file -> sRGB uint8 (HxWx3)."""
    rgb, color = decode(data, half_size)
    coeffs = base_coefficients(rgb, color)
    return to_srgb(rgb, color, coeffs, settings)
