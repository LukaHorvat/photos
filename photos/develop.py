"""Raw development: the inputs of the develop shader, and running it for exports.

The pipeline itself is web/develop.frag. It runs in the browser for the editor's
preview, so sliders respond instantly, and on the server (gpu.py) for exports, so an
export is what the editor showed. This module decodes raws into linear camera RGB and
works out everything else the shader needs. LibRaw unpacks the raw data; at full size the
GPU demosaics it (gpu.demosaic, LibRaw's own AHD to the bit) and keeps it for the shader.

The pipeline, in order:
  1. white balance: colour temperature (Kelvin) -> camera RGB multipliers
  2. camera RGB -> linear sRGB, with the Adobe matrix LibRaw has for the camera
  3. dynamic range compression: the contrast of a smooth, edge-aware "base" layer
     of log luminance is reduced around mid-grey; local detail is kept as it is
  4. exposure (EV), after compression
  5. saturation
  6. the camera's look (look.py): this photo's tone curve, fitted to its camera JPEG,
     then contrast, then the camera's colour profile; then sRGB encoding

Exports also get noise reduction (denoise.py): on the raw data before demosaicing, then
colour noise in gpu.py before the shader, which adds back some grain at step 2.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import rawpy
from PIL import Image

from .look import GRID, LUMA, Look, jpeg_target, load_profile, samples_at

# temp None: as shot. tint 0: the camera's. noise None: automatic (denoise.active), else 0 off, 100 on.
DEFAULTS = {"compress": 0, "exposure": 0.0, "temp": None, "tint": 0, "saturation": 0, "contrast": 0, "noise": None}
LIMITS = {
    "compress": (0, 100),
    "exposure": (-4.0, 4.0),
    "temp": (2000, 15000),
    "tint": (-100, 100),
    "saturation": (-100, 100),
    "contrast": (-100, 100),
    "noise": (0, 100),
}
TEMP_STEP = 50  # Kelvin; temperatures are kept on this grid, so the editor can be sent all of them
TINT_STOPS = 1.0  # tint +100 / -100: red and blue this many stops up / down against green (magenta / green)

MAX_COMPRESSION = 0.75  # at 100, the base layer keeps a quarter of its contrast
LOG_FLOOR = 2.0**-12  # ~ the sensor's noise floor; darker than this is not lifted further

# Edge-aware smoothing (guided filter) of log luminance for the base layer, run at low
# resolution; its per-pixel linear coefficients are upsampled and applied at full size.
BASE_LONG_SIDE = 512
BASE_RADIUS = 20  # pixels at BASE_LONG_SIDE
BASE_EPS = 0.3  # log2 units squared; edges stronger than ~0.5 stop are kept

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
FULL_TILE = 512  # the editor's 100% view loads full-size data in tiles this big


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
    off_locus: np.ndarray  # as_shot / temp_multipliers(as_shot_temp): the camera's own green/magenta
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
        off_locus = as_shot / temp_multipliers(cam_xyz, temp)
        return cls(cam_xyz, np.linalg.inv(cam_rgb), as_shot, off_locus / off_locus[1], temp)

    def multipliers(self, temp: float | None, tint: float = 0) -> np.ndarray:
        """White balance: a temperature (None: as shot; the camera's own tint kept), and tint on
        top of it, which moves red and blue against green without changing brightness."""
        base = self.as_shot if temp is None else temp_multipliers(self.cam_xyz, temp) * self.off_locus
        f = 2.0 ** (tint / 100 * TINT_STOPS)
        w = self.tint_weights
        return base * np.array([f, 1.0, f]) / (w[0] * f + w[1] + w[2] * f)

    @property
    def tint_weights(self) -> np.ndarray:
        """Luminance of each white-balanced camera channel (they sum to 1)."""
        return self.rgb_cam.T @ LUMA

    def header(self) -> dict:
        lo, hi = LIMITS["temp"]
        return {
            "asShot": self.as_shot.tolist(),
            "asShotTemp": round(self.as_shot_temp),
            # The multipliers for every temperature the editor can set, so it needn't compute them;
            # tint on top as in multipliers(), and the white balance pickers' maths (editor.js).
            "wb": {t: self.multipliers(t).tolist() for t in range(lo, hi + 1, TEMP_STEP)},
            "tintStops": TINT_STOPS,
            "tintWeights": self.tint_weights.tolist(),
            "camRgb": np.linalg.inv(self.rgb_cam).tolist(),  # linear sRGB -> white-balanced camera RGB, rows
        }


# --- Pipeline ---------------------------------------------------------------------------


def clean(settings: dict | None) -> dict:
    """Settings with every field present, of the right type and within range."""
    out = dict(DEFAULTS)
    for key, value in (settings or {}).items():
        if key not in DEFAULTS or (value is None and DEFAULTS[key] is None):
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


def base_uniforms(color: Color, look: Look) -> dict:
    """The shader inputs that are fixed for a photo (see develop.frag)."""
    return {
        "rgbCam": color.rgb_cam.T.flatten().tolist(),  # GL matrices are column-major
        "maxCompression": MAX_COMPRESSION,
        "anchor": look.anchor,
        "logFloor": LOG_FLOOR,
        "toneRange": [float(GRID[0]), float(GRID[-1])],
        "toneSize": float(GRID.size),
        "shadowLoss": look.shadow_loss,
    }


def uniforms(color: Color, look: Look, settings: dict) -> dict:
    """All shader inputs for these settings. web/develop.js sets the same ones."""
    s = clean(settings)
    return {
        **base_uniforms(color, look),
        "mul": color.multipliers(s["temp"], s["tint"]).tolist(),
        "compress": s["compress"],
        "exposure": s["exposure"],
        "saturation": s["saturation"],
        "contrast": s["contrast"],
    }


def look_textures(look: Look) -> tuple[np.ndarray, np.ndarray]:
    """The look's lookup textures: tone curve (n,), colour table as depth x height x width x 3
    (level, b - g, r - g), the order a GL 3D texture is uploaded in."""
    return look.tone.astype(np.float32), np.ascontiguousarray(look.table.transpose(2, 1, 0, 3), np.float32)


def base_coefficients(rgb_cam: np.ndarray, color: Color) -> np.ndarray:
    """Guided-filter coefficients (h x w x 2: a, b) for the base layer, at low resolution.

    base = a * log2(luminance) + b, with a and b upsampled bilinearly: smooth over
    regions, but following strong edges, so compression doesn't halo around them.
    Computed once per photo from the as-shot rendering.
    """
    h, w = rgb_cam.shape[:2]
    # Linear luminance at the as-shot white balance (one weighted sum per pixel), averaged
    # down to the base layer's size, then logged. Averaging linear light gives the same
    # result whatever size it starts from, so the editor's preview and an export agree.
    mul, weights = base_luminance(color)
    lum = np.minimum(rgb_cam * mul, mul.min()) @ weights
    return coefficients(np.asarray(Image.fromarray(lum).resize(base_size(w, h), Image.Resampling.BOX)))


def base_luminance(color: Color) -> tuple[np.ndarray, np.ndarray]:
    """The base layer's luminance: min(rgb * mul, min(mul)) @ weights (float32)."""
    return color.as_shot.astype(np.float32), (color.rgb_cam.T @ LUMA).astype(np.float32)


def base_size(w: int, h: int) -> tuple[int, int]:
    """(width, height) the base layer is worked out at."""
    scale = BASE_LONG_SIDE / max(h, w)
    return max(1, round(w * scale)), max(1, round(h * scale))


def coefficients(small: np.ndarray) -> np.ndarray:
    """base_coefficients() from the luminance averaged down to base_size()."""
    guide = np.log2(np.maximum(small, LOG_FLOOR)).astype(np.float32)
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


# --- Entry points -------------------------------------------------------------------------


def decode(data: bytes, half_size: bool) -> tuple[np.ndarray, Color]:
    """Linear camera RGB (HxWx3 float32, black 0, saturation 1), oriented, and its colour data.

    Noise below the black level is kept (as negative values): LibRaw is given a black
    level BLACK_PAD lower than the real one, and the difference is subtracted here.
    """
    rgb, color, _, _ = _decode(data, half_size, denoise=False)
    return rgb, color


def decode_full(data: bytes, photo, denoise: bool):
    """decode() at full size; with `denoise`, noise reduction on the raw data before
    demosaicing (denoise.py). Also returns what the develop step needs to finish noise
    reduction (or None)."""
    typical = load_profile(photo.camera).noise_at(photo.iso) if denoise else None
    rgb, color, noise, _ = _decode(data, False, denoise, photo.iso, typical)
    return rgb, color, noise


def _decode(data: bytes, half_size: bool, denoise: bool, iso: int | None = None, typical_noise=None):
    """decode(), plus the noise reduction's leftovers and the full-size (width, height)."""
    from . import denoise as dn

    removed = None
    with rawpy.imread(io.BytesIO(data)) as raw:
        sizes = raw.sizes
        full_size = (sizes.height, sizes.width) if sizes.flip in (5, 6) else (sizes.width, sizes.height)
        color = Color.from_raw(raw)
        black = min(raw.black_level_per_channel)
        # Lossy NEFs saturate a few codes under the nominal white level (see raw.Photo.clipping_mask).
        top = int(raw.raw_image_visible.max())
        saturation = top if top >= 0.98 * raw.white_level else raw.white_level
        gain = 1  # raw values are this many times larger after noise reduction
        if denoise and dn.supported(raw):
            removed, gain = dn.raw_nlm(raw, black, 1 / (saturation - black), color, typical_noise)
            flip = raw.sizes.flip
        pad = min(black, BLACK_PAD)
        rgb = raw.postprocess(half_size=half_size, user_black=gain * (black - pad), user_sat=gain * saturation, **RAW_LINEAR)
    # LibRaw outputs (raw - user_black) * 65535 / user_sat.
    rgb = rgb.astype(np.float32)
    rgb *= np.float32(saturation / 65535 / (saturation - black))
    rgb -= np.float32(pad / (saturation - black))
    noise = None
    if removed is not None:
        noise = dn.Noise(dn.orient(removed, flip), color.as_shot, color.rgb_cam, dn.grain_weights(iso))
    return rgb, color, noise, full_size


def _on_gpu(photo, data: bytes, denoise: bool, base: bool):
    """decode_full() demosaiced on the GPU (gpu.demosaic: the same result, to the bit, without
    LibRaw's slow demosaicing and the copies around it), left there as the develop shader's
    input; the photo's look fitted (if it isn't yet) and with `base`, the base layer worked
    out, from what the GPU averages down. Returns (gpu.Decoded, color, noise, look, base
    coefficients or None), or None for a raw LibRaw has to demosaic (not a plain Bayer mosaic)."""
    from . import denoise as dn
    from . import gpu

    with rawpy.imread(io.BytesIO(data)) as raw:
        sizes = raw.sizes
        black_levels = set(raw.black_level_per_channel)
        if not dn.supported(raw) or raw.num_colors != 3 or len(black_levels) != 1 or sizes.pixel_aspect != 1:
            return None
        color = Color.from_raw(raw)
        black = black_levels.pop()
        top = int(raw.raw_image_visible.max())
        saturation = top if top >= 0.98 * raw.white_level else raw.white_level  # as _decode
        nlm, gain = None, 1
        if denoise:
            typical = load_profile(photo.camera).noise_at(photo.iso)
            nlm = dn.nlm_args(raw, black, 1 / (saturation - black), color, typical)
            gain = nlm[-1]
        mosaic, pattern, cam_xyz = raw.raw_image_visible.copy(), np.asarray(raw.raw_pattern), raw.rgb_xyz_matrix
    flip = sizes.flip
    size = (sizes.height, sizes.width) if flip & 4 else (sizes.width, sizes.height)
    pad = min(black, BLACK_PAD)
    to_cam = (saturation / 65535 / (saturation - black), pad / (saturation - black))  # as _decode
    look = _looks.get((photo.path, denoise))
    jpeg = photo.embedded_jpeg(data) if look is None else None
    target = jpeg_target(size, jpeg) if jpeg else None
    decoded = gpu.demosaic(
        mosaic, pattern, gain * (black - pad), gain * saturation, flip, cam_xyz, to_cam, nlm=nlm,
        look=None if target is None else (target.size, target.box), base=(*base_luminance(color), base_size(*size)) if base else None,
    )
    try:
        if look is None:
            samples = None if target is None else samples_at(decoded.look, target, color)
            look = _looks[photo.path, denoise] = Look.from_samples(samples, photo.camera, photo.iso)
        noise = None
        if nlm is not None:
            noise = dn.Noise(dn.orient(decoded.removed, flip), color.as_shot, color.rgb_cam, dn.grain_weights(photo.iso))
        return decoded, color, noise, look, None if decoded.base is None else coefficients(decoded.base)
    except BaseException:
        decoded.release()
        raise


# A photo's look is fitted from the full-size data an export develops (with noise reduction or
# without), so an export comes out the same whether the editor ran first or not. The quick
# preview fits a provisional one from its half-size data if that isn't there yet; the final
# view replaces it.
_looks: dict = {}  # (path, noise reduction on) -> Look
_quick_looks: dict = {}  # path -> Look


def look_for(photo, data: bytes, rgb: np.ndarray, color: Color, denoise: bool) -> Look:
    """The look, fitted from full-size `rgb` (with noise reduction if `denoise`) if it isn't yet."""
    look = _looks.get((photo.path, denoise))
    if look is None:
        look = _looks[photo.path, denoise] = Look.for_photo(rgb, color, photo.embedded_jpeg(data), photo.camera, photo.iso)
    return look


def quick_look(photo, data: bytes, rgb: np.ndarray, color: Color) -> Look:
    """For the quick preview: the photo's look if it's been fitted, else one from `rgb` (half size)."""
    look = _looks.get((photo.path, False)) or _looks.get((photo.path, True)) or _quick_looks.get(photo.path)
    if look is None:
        look = _quick_looks[photo.path] = Look.for_photo(rgb, color, photo.embedded_jpeg(data), photo.camera, photo.iso)
    return look


def preview(photo, data: bytes, long_side: int) -> bytes:
    """Everything the browser needs to develop this photo itself, as one binary blob:

    uint32 header length, JSON header, padding to 2 bytes, then float16 camera RGB
    (height x width x 3), base coefficients (ch x cw x 2), tone curve, colour table, and
    with header["grain"] the grain to give back (height x width).

    This is the quick one, from a half-size decode without noise reduction (~0.5 s);
    full_preview() is what an export will be, shrunk.
    """
    rgb, color, _, full_size = _decode(data, half_size=True, denoise=False)
    with ThreadPoolExecutor(1) as ex:  # the look fit and the shrink don't need each other, and mostly let go of the GIL
        small = ex.submit(_shrink, rgb, long_side)
        look = quick_look(photo, data, rgb, color)
        return _preview_blob(small.result(), None, color, look, full_size)


def full_preview(full: "Full", long_side: int) -> bytes:
    """preview() made from the export's own input (noise reduction included, if it has any)."""
    if long_side not in full.previews:
        from concurrent.futures import ThreadPoolExecutor

        h, w = full.cam.shape[:2]
        planes = [full.cam[..., c] for c in range(3)] + ([] if full.grain is None else [full.grain])
        with ThreadPoolExecutor(len(planes)) as ex:  # PIL and numpy let go of the GIL: ~4x faster
            small = list(ex.map(lambda plane: _shrink(plane, long_side), planes))
        grain = small[3] if len(small) > 3 else None
        full.previews[long_side] = _preview_blob(np.stack(small[:3], -1), grain, full.color, full.look, (w, h))
    return full.previews[long_side]


def _shrunk_size(w: int, h: int, long_side: int) -> tuple[int, int]:
    """(width, height) of w x h shrunk to fit long_side (never enlarged)."""
    scale = min(1.0, long_side / max(h, w))
    return (w, h) if scale == 1 else (max(1, round(w * scale)), max(1, round(h * scale)))


def _shrink(img: np.ndarray, long_side: int) -> np.ndarray:
    """Averaged down (in linear light) to fit long_side, float32."""
    h, w = img.shape[:2]
    size = _shrunk_size(w, h, long_side)
    if size == (w, h):
        return img.astype(np.float32)
    shrink = lambda c: np.asarray(Image.fromarray(np.ascontiguousarray(c, np.float32)).resize(size, Image.Resampling.BOX))
    if img.ndim == 2:
        return shrink(img)
    with ThreadPoolExecutor(img.shape[2]) as ex:  # a channel per thread: PIL and numpy let go of the GIL
        return np.stack(list(ex.map(lambda c: shrink(img[..., c]), range(img.shape[2]))), -1)


def _preview_blob(rgb, grain, color: Color, look: Look, full_size) -> bytes:
    import json

    h, w = rgb.shape[:2]
    coeffs = base_coefficients(rgb, color)
    tone, table = look_textures(look)
    header = {
        "width": w,
        "height": h,
        "coeffWidth": coeffs.shape[1],
        "coeffHeight": coeffs.shape[0],
        "toneSize": tone.size,
        "tableSize": [table.shape[2], table.shape[1], table.shape[0]],  # width, height, depth
        "grain": grain is not None,
        "uniforms": base_uniforms(color, look),
        "fullWidth": full_size[0],  # for the 100% view (Full)
        "fullHeight": full_size[1],
        "fullTile": FULL_TILE,
        **color.header(),
    }
    head = json.dumps(header).encode()
    head += b" " * (len(head) % 2)
    arrays = (rgb, coeffs, tone, table) + (() if grain is None else (grain,))
    return b"".join([len(head).to_bytes(4, "little"), head] + [a.astype(np.float16).tobytes() for a in arrays])


@dataclass
class Full:
    """What an export feeds the develop shader, at full size, for the editor: its 100% view
    (in tiles) and its final whole-photo view (full_preview). The browser runs the shader on
    it, so both show what the export will."""

    cam: np.ndarray  # camera RGB (after noise reduction, if any), H x W x 3 float16 (may be a view)
    grain: np.ndarray | None  # grain given back after noise reduction, H x W float16
    color: Color
    look: Look
    previews: dict = field(default_factory=dict)  # long side -> full_preview()

    def tile(self, i: int, j: int) -> bytes:
        """Tile (column i, row j) of FULL_TILE squares, smaller at the right and bottom edges:
        its camera RGB, then its grain."""
        rows, cols = slice(j * FULL_TILE, (j + 1) * FULL_TILE), slice(i * FULL_TILE, (i + 1) * FULL_TILE)
        cam = self.cam[rows, cols]
        grain = np.zeros(cam.shape[:2], np.float16) if self.grain is None else self.grain[rows, cols]
        return cam.tobytes() + grain.tobytes()


def full(photo, data: bytes, denoise: bool, previews=()) -> Full:
    """The export's input for the editor, with its full_preview() at each long side in
    `previews` made already (averaged down on the GPU, ~0.3 s quicker than from the Full)."""
    from . import gpu

    on_gpu = _on_gpu(photo, data, denoise, base=False)
    if on_gpu is None:
        rgb, color, noise = decode_full(data, photo, denoise)
        look = look_for(photo, data, rgb, color, denoise)
        if noise is None:
            return Full(rgb.astype(np.float16), None, color, look)
        decoded = gpu.upload(rgb)
    else:
        decoded, color, noise, look, _ = on_gpu
    w, h = decoded.width, decoded.height
    sizes = {s: _shrunk_size(w, h, s) for s in previews if _shrunk_size(w, h, s) != (w, h)}
    try:
        cam, grain, small = gpu.read_decoded(decoded, noise, set(sizes.values()))
    finally:
        decoded.release()
    out = Full(cam[..., :3], grain, color, look)
    for long_side, size in sizes.items():
        out.previews[long_side] = _preview_blob(*small[size], color, look, (w, h))
    return out


def develop(photo, data: bytes, settings: dict) -> np.ndarray:
    """A whole raw file, developed at full size -> sRGB uint8 (HxWx3)."""
    from . import denoise, gpu

    s = clean(settings)
    compress = s["compress"] > 0  # else the shader doesn't read the base layer
    nr = denoise.active(s["noise"], photo.iso)
    on_gpu = _on_gpu(photo, data, nr, base=compress)
    if on_gpu is None:
        rgb, color, noise = decode_full(data, photo, nr)
        look = look_for(photo, data, rgb, color, nr)
        coeffs = base_coefficients(rgb, color) if compress else None
        decoded = gpu.upload(rgb)
    else:
        decoded, color, noise, look, coeffs = on_gpu
    try:
        coeffs = np.zeros((1, 1, 2), np.float32) if coeffs is None else coeffs
        return gpu.render_decoded(decoded, coeffs, uniforms(color, look, s), *look_textures(look), noise=noise)
    finally:
        decoded.release()
