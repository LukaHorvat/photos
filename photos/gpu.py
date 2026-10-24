"""GPU work on the server: demosaicing, the develop shader for exports, and noise reduction.

The develop shader (web/develop.frag) also renders the editor's preview in the browser,
so an export looks like what you saw. Noise reduction (see denoise.py) only runs here:
non-local means on the raw planes, and after demosaicing a guided filter for colour
noise and the grain given back. Demosaicing is LibRaw's AHD, ported to give the same
result to the bit, ~10x faster than LibRaw's own; the camera RGB stays on the GPU for
the develop shader, and only what the CPU needs comes back. Everything runs on the GPU
when Mesa can reach one (on WSL, through D3D12) and on Mesa's software renderer
otherwise. A GL context belongs to one thread at a time, so all GL work happens on a
single thread of its own.
"""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

WEB = Path(__file__).parent / "web"
POOL_IDLE = 300  # seconds without GPU work, after which the texture pool (~1 GB) is freed; remaking it costs ~0.5 s
GREENS_FLAT = (0.5, 1.5)  # raw_out: brightness steps (noise sd) up to which a tile's two greens are evened out, fully to partly
_thread = ThreadPoolExecutor(1, thread_name_prefix="gpu")
_gl: dict = {}
# Free textures by (size, components, dtype), for reuse: new GPU memory is slow (on WSL
# ~2 ms/MB once the driver has given freed memory back, and an export uses ~1.5 GB).
_pool: dict = {}
_jobs = itertools.count(1)
_latest = 0


@dataclass
class Decoded:
    """Camera RGB on the GPU, as the develop shader's input (an RGBA16F texture, oriented,
    black 0, sensor saturation 1), until release(): from demosaic(), or upload(). With what
    the CPU needs of it, averaged down and read back by demosaic()."""

    width: int
    height: int
    cam: object  # the texture: for the GL thread only
    look: np.ndarray | None = None  # demosaic(look=...)
    base: np.ndarray | None = None  # demosaic(base=...)
    removed: np.ndarray | None = None  # demosaic(nlm=...): what raw_nlm() returns as removed
    original: object = None  # demosaic(original=True), upload(original=...): camera RGB without noise reduction

    def release(self):
        textures = [t for t in (self.cam, self.original) if t is not None]
        if textures:
            _run(_release, *textures)
        self.cam = self.original = None


def demosaic(mosaic, pattern, black: int, saturation: int, flip: int, cam_xyz, to_cam, nlm=None, look=None, base=None, original=False) -> Decoded:
    """What LibRaw's postprocess() makes of a Bayer mosaic with develop.RAW_LINEAR, then as
    camera RGB on the GPU: value * to_cam[0] - to_cam[1], in float32 like develop._decode.

    mosaic: uint16, the sensor's visible area; pattern: its 2x2 colours as LibRaw numbers
    them (3: the second green); black, saturation: LibRaw's user_black and user_sat (black
    the same for every colour); flip: LibRaw's sizes.flip; cam_xyz: rawpy's rgb_xyz_matrix,
    which AHD's CIELab needs. LibRaw's AHD is ported exactly: the result is the same to the bit.

    nlm: raw_nlm()'s arguments after the mosaic, to denoise it first (Decoded.removed); black
        and saturation then scale with its gain.
    original: with nlm, also demosaic the mosaic as it is (Decoded.original), for a weaker
        noise reduction to mix in (denoise.Noise.amount).
    look: ((width, height), box): camera RGB averaged down over box (x0, y0, x1, y1) as PIL's
        BOX resize does it, read back as Decoded.look (float32, height x width x 3).
    base: (mul, weights, (width, height)): luminance min(rgb * mul, min(mul)) @ weights averaged
        down the same way, as Decoded.base (float32, height x width)."""
    mosaic = np.ascontiguousarray(mosaic, np.uint16)
    xyz = _libraw_xyz_cam(tuple(np.asarray(cam_xyz, np.float32)[:3].ravel().tolist()))
    to_cam = tuple(float(np.float32(v)) for v in to_cam)
    if base is not None:
        mul, weights, size = base
        base = (tuple(np.float32(mul).tolist()), tuple(np.float32(weights).tolist()), size)
    return _run(_demosaic, mosaic, np.asarray(pattern), int(black), int(saturation), int(flip), xyz, to_cam, nlm, look, base, original)


def upload(rgb, original=None) -> Decoded:
    """Camera RGB (HxWx3 float32) as a Decoded, for what LibRaw demosaics; `original` the same
    without noise reduction, if it is to be mixed in (Decoded.original)."""
    h, w = rgb.shape[:2]
    return Decoded(w, h, _run(_upload, _rgba16(rgb)), original=None if original is None else _run(_upload, _rgba16(original)))


def render(rgb, coeffs, uniforms, tone, table, noise=None) -> np.ndarray:
    """camera RGB (HxWx3 float32) + base coefficients (hxwx2) + the look's tone curve (n) and
    colour table (depth x height x width x 3, see develop.look_textures) -> sRGB uint8 HxWx3.
    With `noise` (denoise.Noise), colour noise is smoothed and grain given back first."""
    decoded = upload(rgb)
    try:
        return render_decoded(decoded, coeffs, uniforms, tone, table, noise)
    finally:
        decoded.release()


def render_decoded(decoded: Decoded, coeffs, uniforms, tone, table, noise=None) -> np.ndarray:
    """render() of camera RGB already on the GPU.

    Conversions happen here, on the caller's thread, so the GL thread only does GL work
    (it is shared by all exports)."""
    grain = None if noise is None else _f32(noise.grain)
    out = _run(_render, decoded.cam, _f32(coeffs), uniforms, _f32(tone), _f32(table), noise, grain, decoded.original)
    # GL reads the bottom row first. Drawing upside down instead would need a vertex shader
    # of its own, and change a few pixels by a bit (sampling positions computed differently).
    return np.ascontiguousarray(out[::-1])


def read_decoded(decoded: Decoded, noise=None, shrink=()) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """What render_decoded() feeds the develop shader, read back: camera RGB (HxWx4 float16),
    with `noise` its colour noise smoothed (and Decoded.original mixed in), and the grain to
    give back (HxW float16) or None.
    The editor's 100% view runs the shader on these in the browser, so it shows the export.

    And for each (width, height) in `shrink`, both averaged down to it as PIL's BOX resize does
    it (develop._shrink): {size: (camera RGB, grain or None)}, float32."""
    h, w = decoded.height, decoded.width
    grain = None if noise is None else _f32(noise.grain)
    cam, grain, small = _run(_read_cam, decoded.cam, noise, grain, tuple(shrink), decoded.original)
    return np.frombuffer(cam, np.float16).reshape(h, w, 4), None if grain is None else np.frombuffer(grain, np.float16).reshape(h, w), small


def _rgba16(rgb) -> np.ndarray:
    cam = np.empty(rgb.shape[:2] + (4,), np.float16)  # RGBA16F: the fastest upload, as in the browser
    cam[..., :3] = rgb
    cam[..., 3] = 0
    return cam


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32)


def raw_nlm(mosaic: np.ndarray, black: float, A: float, B: float, weights, h: float, search: int, patch: int,
            greens: tuple[int, int] | None = None, gain: int = 1):
    """Non-local means on a Bayer mosaic (uint16, 2x2 pattern): its four planes as one
    4-channel image, so every colour gets the same weights, after a generalized Anscombe
    transform for noise variance A * signal + B (black subtracted), which makes the noise ~1.

    Returns the denoised mosaic, times `gain` (uint16: what denoising adds below one raw unit
    survives), and per 2x2 tile, sum(weights * removed) over the four photosites (float32,
    half size). `greens`: where a tile's two green photosites are (denoise.PHASES order), to
    even them out where the photo is flat (see raw_out)."""
    out, removed = _run(_raw_nlm, np.ascontiguousarray(mosaic, np.uint16), black, A, B, weights, h, search, patch, greens, gain)
    hh, ww = mosaic.shape[0] // 2, mosaic.shape[1] // 2
    return np.frombuffer(out, np.uint16).reshape(mosaic.shape), np.frombuffer(removed, np.float32).reshape(hh, ww)


# --- LibRaw's AHD: its constants, as LibRaw computes them -----------------------------------

_XYZ_RGB = ((0.4124564, 0.3575761, 0.1804375), (0.2126729, 0.7151522, 0.0721750), (0.0193339, 0.1191920, 0.9503041))
_D65 = tuple(float(np.float32(v)) for v in (0.95047, 1.0, 1.08883))


@cache
def _libraw_xyz_cam(cam_xyz: tuple) -> np.ndarray:
    """AHD's camera RGB -> XYZ (3x3 float32), from rawpy's rgb_xyz_matrix (3x3, flattened):
    LibRaw's cam_xyz_coeff (rgb_cam, in doubles) then cielab()'s set-up (in floats), every
    step in LibRaw's order, so that its CIELab and the homogeneity decisions come out the same."""
    cam_xyz = np.array(cam_xyz, np.float64).reshape(3, 3).tolist()
    cam_rgb = [[_sum_seq(cam_xyz[i][k] * _XYZ_RGB[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    for row in cam_rgb:
        num = _sum_seq(row)
        row[:] = [v / num for v in row]
    # LibRaw's pseudoinverse(cam_rgb)
    work = [[1.0 if j == i + 3 else 0.0 for j in range(6)] for i in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                work[i][j] += cam_rgb[k][i] * cam_rgb[k][j]
    for i in range(3):
        num = work[i][i]
        work[i] = [v / num for v in work[i]] if abs(num) > float(np.float32(1e-5)) else work[i]
        for k in range(3):
            if k != i:
                num = work[k][i]
                work[k] = [a - b * num for a, b in zip(work[k], work[i])]
    inverse = [[_sum_seq(work[j][k + 3] * cam_rgb[i][k] for k in range(3)) for j in range(3)] for i in range(3)]
    rgb_cam = np.float32([[inverse[j][i] for j in range(3)] for i in range(3)])
    xyz_cam = np.zeros((3, 3), np.float32)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                xyz_cam[i, j] = np.float32(xyz_cam[i, j] + np.float32(_XYZ_RGB[i][k] * float(rgb_cam[k, j]) / _D65[i]))
    return xyz_cam


def _sum_seq(values) -> float:
    """A sum in the given order (like a C loop), unlike Python's sum() since 3.12 (Neumaier)."""
    total = 0.0
    for v in values:
        total += v
    return total


@cache
def _cbrt_table() -> np.ndarray:
    """AHD's cube root table (LibRaw's cielab()), 65536 floats as a 256 x 256 texture."""
    r = np.arange(0x10000, dtype=np.float32) / np.float32(65535)
    cbrt = np.where(r > np.float32(0.008856), np.power(r, np.float32(1) / np.float32(3)), np.float32(7.787) * r + np.float32(16) / np.float32(116))
    return cbrt.astype(np.float32).reshape(256, 256)


@cache
def _box_bounds(n: int, lo: float, hi: float, size: int) -> np.ndarray:
    """What PIL's BOX resize averages along an axis, from n pixels over [lo, hi) down to size:
    for each output pixel, its input pixels [first, end) (int32, size x 2). Pillow takes the
    box as C floats, and weighs an input pixel 1 if its centre falls within the output
    pixel's support, 0 otherwise (Resample.c: precompute_coeffs, box_filter)."""
    lo32, hi32 = np.float32(lo), np.float32(hi)
    scale = float(hi32 - lo32) / size
    filterscale = max(scale, 1.0)
    support, ss = 0.5 * filterscale, 1.0 / filterscale
    out = np.empty((size, 2), np.int32)
    for i in range(size):
        center = float(lo32) + (i + 0.5) * scale
        xs = [x for x in range(max(int(center - support + 0.5), 0), min(int(center + support + 0.5), n)) if -0.5 < (x - center + 0.5) * ss <= 0.5]
        out[i] = xs[0], xs[-1] + 1
    return out


def info() -> dict:
    """The GL renderer in use (creates the context if needed)."""
    return _thread.submit(lambda: dict(_setup()["ctx"].info)).result()


def _run(fn, *args):
    """fn(*args) on the GL thread. The texture pool is freed once the GPU has been idle for POOL_IDLE."""
    global _latest
    out = _thread.submit(fn, *args).result()
    _latest = job = next(_jobs)
    timer = threading.Timer(POOL_IDLE, _thread.submit, (_trim, job))
    timer.daemon = True
    timer.start()
    return out


def _trim(job: int):
    if job == _latest:
        for tex in itertools.chain.from_iterable(_pool.values()):
            _delete(tex)
        _pool.clear()


# --- Shaders for the passes that only run here -----------------------------------------------

PASS_VERTEX = """#version 330
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }"""
PASS_HEADER = """#version 330
// Weights at offset i of box filters of radius 1, 2 and 4, each applied twice
#define TENT(i) (max(vec3(3.0, 5.0, 9.0) - abs(float(i)), 0.0) / vec3(9.0, 25.0, 81.0))
"""

# The variance-stabilising transform: noise of variance A x + B (raw units above black) becomes
# ~1 everywhere. c = 3/8 A^2 + B.
VST = """
    uniform float A, c;
    vec4 forward(vec4 x) { return 2.0 / A * sqrt(max(A * x + c, 0.0)); }
    vec4 inverse(vec4 z) { return ((z * A / 2.0) * (z * A / 2.0) - c) / A; }
"""
# The raw mosaic, a 2x2 tile at a time (photosites in denoise.PHASES order), and the NLM's
# result for a tile (raw units above black).
RAW = VST + """
    uniform usampler2D raw; uniform float black;
    vec4 tile(ivec2 t) {
      ivec2 q = 2 * t;
      return vec4(texelFetch(raw, q, 0).r, texelFetch(raw, q + ivec2(1, 0), 0).r,
                  texelFetch(raw, q + ivec2(0, 1), 0).r, texelFetch(raw, q + ivec2(1, 1), 0).r) - black;
    }
    uniform sampler2D sum; uniform sampler2D weight;
    vec4 denoised(ivec2 t) { return texelFetch(sum, t, 0) / texelFetch(weight, t, 0).r; }
"""

SHADERS = {
    "raw_planes": RAW + """
        out vec4 z;
        void main() { z = forward(tile(ivec2(gl_FragCoord.xy))); }""",
    # The denoised mosaic: the NLM result (sum / weight). Each plane is averaged on its own,
    # so the two greens of a tile keep different leftover noise, varying smoothly; AHD turns
    # that into a maze of short lines. Where the photo is flat (between the next tiles,
    # brightness changes by less than GREENS_FLAT noise sd), the two greens get their mean;
    # at edges they differ for real (a pixel apart) and stay.
    "raw_out": RAW + """
        uniform float gain;
        uniform ivec2 greens;  // the tile's two green photosites, or -1: leave them
        uniform vec2 flatness;  // GREENS_FLAT
        out uint v;
        float lum(ivec2 t) { return dot(denoised(clamp(t, ivec2(0), textureSize(sum, 0) - 1)), vec4(1.0)); }
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy), t = p / 2;
          float x = float(texelFetch(raw, p, 0).r);  // odd size: no whole tile here
          if (all(lessThan(t, textureSize(sum, 0)))) {
            vec4 d = denoised(t);
            if (greens.x >= 0) {
              float l = lum(t - ivec2(1, 0)), r = lum(t + ivec2(1, 0)), u = lum(t - ivec2(0, 1)), b = lum(t + ivec2(0, 1));
              float sd = sqrt(2.0 * (A * max((l + r + u + b) / 4.0, 0.0) + 4.0 * (c - 0.375 * A * A)));  // of a difference of two tiles' sums
              float even = (1.0 - smoothstep(flatness.x, flatness.y, max(abs(r - l), abs(b - u)) / sd)) * (d[greens.x] - d[greens.y]) / 2.0;
              d[greens.x] -= even;
              d[greens.y] += even;
            }
            x = d[(p.y & 1) * 2 + (p.x & 1)] + black;
          }
          v = uint(clamp(round(x * gain), 0.0, 65535.0));
        }""",
    "raw_removed": RAW + """
        uniform vec4 weights;
        out float v;
        void main() {
          ivec2 t = ivec2(gl_FragCoord.xy);
          v = dot(weights, tile(t) - denoised(t));
        }""",
    # Non-local means, four search offsets per pass. Each offset o is done once for both
    # pairs it stands for, (p, p + o) and (p - o, p): the patch distance is the same. The
    # weights come from patch distances in the stabilised domain, where noise is even, but
    # what's averaged is the values back in raw units: an average taken in the stabilised
    # domain reads low once back (the square root is concave), by about A / 4; at ISO 25600
    # ~9 raw units, half the blue of a black dress.
    "nlm_init": VST + """
        uniform sampler2D img;
        layout(location = 0) out vec4 sum; layout(location = 1) out float weight;
        void main() { sum = inverse(texelFetch(img, ivec2(gl_FragCoord.xy), 0)); weight = 1.0; }""",
    # Squared differences to the four offsets, summed along the patch's rows (its columns
    # are summed by a box pass).
    "nlm_diff": """
        uniform sampler2D img; uniform ivec2 o0, o1, o2, o3; uniform int r;
        out vec4 d;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy), last = textureSize(img, 0) - 1;
          d = vec4(0.0);
          for (int i = -r; i <= r; i++) {
            ivec2 q = clamp(p + ivec2(i, 0), ivec2(0), last);
            vec4 a = texelFetch(img, q, 0);
            vec4 d0 = a - texelFetch(img, clamp(q + o0, ivec2(0), last), 0);
            vec4 d1 = a - texelFetch(img, clamp(q + o1, ivec2(0), last), 0);
            vec4 d2 = a - texelFetch(img, clamp(q + o2, ivec2(0), last), 0);
            vec4 d3 = a - texelFetch(img, clamp(q + o3, ivec2(0), last), 0);
            d += vec4(dot(d0, d0), dot(d1, d1), dot(d2, d2), dot(d3, d3));
          }
        }""",
    "nlm_acc": VST + """
        uniform sampler2D dist; uniform sampler2D img; uniform ivec2 o0, o1, o2, o3; uniform float inv_n, inv_h2;
        layout(location = 0) out vec4 sum; layout(location = 1) out float weight;
        float wt(float s) { return exp(-max(s * inv_n - 2.0, 0.0) * inv_h2); }
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy), last = textureSize(img, 0) - 1;
          ivec2 o[4] = ivec2[4](o0, o1, o2, o3);
          vec4 here = texelFetch(dist, p, 0);
          sum = vec4(0.0); weight = 0.0;
          for (int k = 0; k < 4; k++) {
            if (o[k] == ivec2(0)) continue;  // padding
            float wf = wt(here[k]);
            ivec2 q = p - o[k];
            bool inside = q.x >= 0 && q.y >= 0 && q.x <= last.x && q.y <= last.y;
            float wb = inside ? wt(texelFetch(dist, q, 0)[k]) : 0.0;
            sum += wf * inverse(texelFetch(img, clamp(p + o[k], ivec2(0), last), 0)) + wb * inverse(texelFetch(img, clamp(q, ivec2(0), last), 0));
            weight += wf + wb;
          }
        }""",
    # Box filter along one axis: sum (for NLM's patch distances) or mean.
    "box": """
        uniform sampler2D src; uniform int r; uniform ivec2 dir; uniform float scale;
        out vec4 s;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy), last = textureSize(src, 0) - 1;
          s = vec4(0.0);
          for (int i = -r; i <= r; i++) s += texelFetch(src, clamp(p + i * dir, ivec2(0), last), 0);
          s *= scale;
        }""",
    # Colour noise: a guided filter on chroma at half size, guided by luminance, in a
    # square-root domain where noise is about even across brightness.
    "chroma_prep": """
        uniform sampler2D cam; uniform vec3 mul; uniform float clipLevel; uniform mat3 rgbCam;
        layout(location = 0) out vec4 t0; layout(location = 1) out vec4 t1;
        void main() {
          ivec2 q = ivec2(gl_FragCoord.xy), last = textureSize(cam, 0) - 1;
          vec3 s = vec3(0.0);
          for (int i = 0; i < 4; i++) {
            vec3 c = texelFetch(cam, min(2 * q + ivec2(i & 1, i >> 1), last), 0).rgb;
            vec3 scene = rgbCam * min(c * mul, vec3(clipLevel));
            s += sign(scene) * sqrt(abs(scene));
          }
          s *= 0.25;
          float I = dot(s, vec3(0.25, 0.5, 0.25));
          vec2 ch = vec2(s.r - s.g, s.b - s.g);
          t0 = vec4(I, ch, I * I);
          t1 = vec4(I * ch, 0.0, 0.0);
        }""",
    "chroma_coef": """
        uniform sampler2D m0; uniform sampler2D m1; uniform float eps;
        out vec4 coef;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy);
          vec4 a = texelFetch(m0, p, 0); vec2 b = texelFetch(m1, p, 0).rg;
          vec2 k = (b - a.r * a.gb) / (a.a - a.r * a.r + eps);
          coef = vec4(k.x, a.g - k.x * a.r, k.y, a.b - k.y * a.r);
        }""",
    "chroma_apply": """
        uniform sampler2D t0in; uniform sampler2D mc;
        layout(location = 0) out vec4 t0; layout(location = 1) out vec4 t1;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy);
          float I = texelFetch(t0in, p, 0).r; vec4 c = texelFetch(mc, p, 0);
          vec2 ch = vec2(c.x * I + c.y, c.z * I + c.w);
          t0 = vec4(I, ch, I * I);
          t1 = vec4(I * ch, 0.0, 0.0);
        }""",
    # Full size again: the pixel's own luminance, the smoothed chroma (bilinear), back to camera
    # RGB. Mixed with the photo without noise reduction, for a weaker one.
    "chroma_final": """
        uniform sampler2D cam; uniform sampler2D chroma; uniform vec3 mul; uniform float clipLevel; uniform mat3 rgbCam;
        uniform mat3 camRgb; uniform sampler2D original; uniform float amount;
        out vec4 result;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy);
          vec3 scene = rgbCam * min(texelFetch(cam, p, 0).rgb * mul, vec3(clipLevel));
          vec3 s = sign(scene) * sqrt(abs(scene));
          vec2 ch = texture(chroma, gl_FragCoord.xy / vec2(textureSize(cam, 0))).gb;
          float g = dot(s, vec3(0.25, 0.5, 0.25)) - 0.25 * (ch.x + ch.y);
          vec3 s2 = vec3(g + ch.x, g, g + ch.y);
          vec3 nr = (camRgb * (sign(s2) * s2 * s2)) / mul;
          result = vec4(amount < 1.0 ? mix(texelFetch(original, p, 0).rgb, nr, amount) : nr, 1.0);
        }""",
    # Grain: the removed luminance (half size) at full size, blurred at three scales (box
    # radius 1, 2 and 4 twice over, i.e. tent filters) and given back per scale. Rows, then columns.
    "grain_rows": """
        uniform sampler2D src; uniform vec2 size;
        out vec4 v;
        float x(vec2 p) { return texture(src, p / size).r; }
        void main() {
          vec2 p = gl_FragCoord.xy;
          vec3 b = vec3(0.0);
          for (int i = -8; i <= 8; i++)
            b += x(vec2(clamp(p.x + float(i), 0.5, size.x - 0.5), p.y)) * TENT(i);
          v = vec4(b, x(p));
        }""",
    "grain_columns": """
        uniform sampler2D rows; uniform vec3 w;
        out vec4 v;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy), last = textureSize(rows, 0) - 1;
          vec3 b = vec3(0.0);
          for (int i = -8; i <= 8; i++) b += texelFetch(rows, clamp(p + ivec2(0, i), ivec2(0), last), 0).rgb * TENT(i);
          float x0 = texelFetch(rows, p, 0).a;
          v = vec4(w.x * (x0 - b.x) + w.y * (b.x - b.y) + w.z * (b.y - b.z));
        }""",
}

# LibRaw's AHD demosaicing (src/demosaic/ahd_demosaic.cpp), pass by pass; LibRaw works in tiles,
# but every step only reads the mosaic's own values, so whole-image passes give the same.
# Integer maths as LibRaw's (GLSL's >> on ints is arithmetic, as GCC's); float maths rounded
# where LibRaw's is, see ROUND.
AHD = """
    uniform usampler2D mosaic;   // the scaled mosaic (ahd_scale), as on the sensor
    uniform ivec4 pat;           // its colours (0 R, 1 G, 2 B) at (row & 1) * 2 + (col & 1)
    int FC(ivec2 p) { return pat[(p.y & 1) * 2 + (p.x & 1)]; }
    int M(ivec2 p) { return int(texelFetch(mosaic, p, 0).r); }
"""
AHD_RGB = AHD + """
    uniform usampler2D green;    // green interpolated horizontally (r) and vertically (g)
    int G(ivec2 p, int d) { uvec2 g = texelFetch(green, p, 0).rg; return int(d == 0 ? g.x : g.y); }
    // ahd_interpolate_r_and_b_in_rgb: the pixel's colour with green interpolated along d
    ivec3 rgbAt(ivec2 p, int d) {
      int f = FC(p), x = M(p);
      ivec3 rgb;
      if (f == 1) {
        int c = FC(p + ivec2(0, 1)), t1 = 2 - c;
        int vh = x + ((M(p - ivec2(1, 0)) + M(p + ivec2(1, 0)) - G(p - ivec2(1, 0), d) - G(p + ivec2(1, 0), d)) >> 1);
        int vv = x + ((M(p - ivec2(0, 1)) + M(p + ivec2(0, 1)) - G(p - ivec2(0, 1), d) - G(p + ivec2(0, 1), d)) >> 1);
        rgb[t1] = clamp(vh, 0, 65535);
        rgb[c] = clamp(vv, 0, 65535);
        rgb[1] = x;
      } else {
        int c = 2 - f, g = G(p, d);
        int s = M(p + ivec2(-1, -1)) + M(p + ivec2(1, -1)) + M(p + ivec2(-1, 1)) + M(p + ivec2(1, 1))
              - G(p + ivec2(-1, -1), d) - G(p + ivec2(1, -1), d) - G(p + ivec2(-1, 1), d) - G(p + ivec2(1, 1), d) + 1;
        rgb[c] = clamp(g + (s >> 2), 0, 65535);
        rgb[f] = x;
        rgb[1] = g;
      }
      return rgb;
    }
"""
# LibRaw rounds every float product and sum on its own (x86-64: no fused multiply-add); the GPU's
# compiler fuses a multiply and an add into one rounding, and `precise` doesn't stop it through
# Mesa's D3D12 driver. A bit-level identity it can't see through (zero is a uniform, left at 0)
# makes it round there.
ROUND = """
    uniform uint zero;
    float r(float x) { return uintBitsToFloat(floatBitsToUint(x) ^ zero); }
    vec3 r(vec3 x) { return uintBitsToFloat(floatBitsToUint(x) ^ zero); }
"""
# LibRaw's output as develop._decode turns it into camera RGB, in float32 as numpy does it: a
# multiply, then a subtraction. toHalf: rounded to the nearest half float as numpy rounds, for
# an RGBA16F texture (D3D12 truncates float32 to float16): to its step at that exponent, ties to even.
CAMERA = ROUND + """
    uniform vec2 toCam;
    vec3 camera(ivec3 v) { return r(r(vec3(v) * toCam.x) - toCam.y); }
    vec3 toHalf(vec3 x) {
      ivec3 e = ivec3((floatBitsToUint(x) >> 23) & 255u) - 127;
      vec3 step = exp2(vec3(max(e, -14) - 10));
      return roundEven(x / step) * step;
    }
"""
TO_CAM = CAMERA + """
    uniform usampler2D rgb;  // LibRaw's output (ahd_combine), oriented
    vec3 camAt(ivec2 p) { return camera(ivec3(texelFetch(rgb, p, 0).rgb)); }
"""
# PIL's BOX resize, which averages each output pixel's rows of input pixels, then the rows (in
# doubles; here floats): what an output pixel averages, from _box_bounds() along x and y.
BOX = """
    uniform isampler2D boundsX; uniform isampler2D boundsY;
    ivec2 boxX() { return texelFetch(boundsX, ivec2(int(gl_FragCoord.x), 0), 0).xy; }
    ivec2 boxY() { return texelFetch(boundsY, ivec2(int(gl_FragCoord.y), 0), 0).xy; }
"""

SHADERS.update({
    # LibRaw's black subtraction (raw2image_ex) and scaling to 16 bits (scale_colors)
    "ahd_scale": """
        uniform usampler2D raw; uniform int black; uniform float scale;
        out uint v;
        void main() {
          int x = int(texelFetch(raw, ivec2(gl_FragCoord.xy), 0).r);
          v = uint(clamp(int(float(x > black ? x - black : 0) * scale), 0, 65535));
        }""",
    # ahd_interpolate_green_h_and_v (green photosites keep their own)
    "ahd_green": AHD + """
        out uvec2 g;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy);
          int x = M(p);
          if (FC(p) == 1) { g = uvec2(x); return; }
          int l = M(p - ivec2(1, 0)), r = M(p + ivec2(1, 0)), u = M(p - ivec2(0, 1)), d = M(p + ivec2(0, 1));
          int vh = ((l + x + r) * 2 - M(p - ivec2(2, 0)) - M(p + ivec2(2, 0))) >> 2;
          int vv = ((u + x + d) * 2 - M(p - ivec2(0, 2)) - M(p + ivec2(0, 2))) >> 2;
          g = uvec2(clamp(vh, min(l, r), max(l, r)), clamp(vv, min(u, d), max(u, d)));
        }""",
    # ..._and_convert_to_cielab, both directions
    "ahd_lab": AHD_RGB + ROUND + """
        uniform vec3 xc0, xc1, xc2;   // columns of LibRaw's xyz_cam
        uniform sampler2D cbrtTable;  // _cbrt_table()
        float cbrtAt(float x) {
          int i = clamp(int(x), 0, 65535);
          return texelFetch(cbrtTable, ivec2(i & 255, i >> 8), 0).r;
        }
        ivec4 cielab(ivec3 rgb) {
          vec3 xyz = vec3(0.5);
          xyz = r(xyz + r(xc0 * float(rgb.r)));
          xyz = r(xyz + r(xc1 * float(rgb.g)));
          xyz = r(xyz + r(xc2 * float(rgb.b)));
          float X = cbrtAt(xyz.x), Y = cbrtAt(xyz.y), Z = cbrtAt(xyz.z);
          return ivec4(int(r(r(116.0 * Y) - 16.0) * 64.0), int(32000.0 * r(X - Y)), int(12800.0 * r(Y - Z)), 0);
        }
        layout(location = 0) out ivec4 labH; layout(location = 1) out ivec4 labV;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy);
          labH = cielab(rgbAt(p, 0));
          labV = cielab(rgbAt(p, 1));
        }""",
    # ahd_interpolate_build_homogeneity_map
    "ahd_homo": """
        uniform isampler2D labH; uniform isampler2D labV;
        out uvec2 homo;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy);
          ivec2 dir[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
          ivec3 ch = texelFetch(labH, p, 0).xyz, cv = texelFetch(labV, p, 0).xyz;
          uint ldh[4], ldv[4], abh[4], abv[4];
          for (int i = 0; i < 4; i++) {
            ivec3 nh = texelFetch(labH, p + dir[i], 0).xyz, nv = texelFetch(labV, p + dir[i], 0).xyz;
            ivec2 dh = ch.yz - nh.yz, dv = cv.yz - nv.yz;
            ldh[i] = uint(abs(ch.x - nh.x));
            ldv[i] = uint(abs(cv.x - nv.x));
            abh[i] = uint(dh.x * dh.x + dh.y * dh.y);
            abv[i] = uint(dv.x * dv.x + dv.y * dv.y);
          }
          uint leps = min(max(ldh[0], ldh[1]), max(ldv[2], ldv[3]));
          uint abeps = min(max(abh[0], abh[1]), max(abv[2], abv[3]));
          uint hh = 0u, hv = 0u;
          for (int i = 0; i < 4; i++) {
            hh += (ldh[i] <= leps && abh[i] <= abeps) ? 1u : 0u;
            hv += (ldv[i] <= leps && abv[i] <= abeps) ? 1u : 0u;
          }
          homo = uvec2(hh, hv);
        }""",
    # ahd_interpolate_combine_homogeneous_pixels, and border_interpolate(5) for the 5-pixel
    # border, at the output's size: LibRaw's flip_index turns it.
    "ahd_combine": AHD_RGB + CAMERA + """
        uniform usampler2D homo; uniform int flip;
        layout(location = 0) out uvec4 rgbOut;  // LibRaw's output
        layout(location = 1) out vec4 cam;      // camera RGB, as develop._decode makes it of that
        void main() {
          ivec2 size = textureSize(mosaic, 0), o = ivec2(gl_FragCoord.xy);
          int row = o.y, col = o.x;
          if ((flip & 4) != 0) { int t = row; row = col; col = t; }
          if ((flip & 2) != 0) row = size.y - 1 - row;
          if ((flip & 1) != 0) col = size.x - 1 - col;
          ivec2 p = ivec2(col, row);
          ivec3 rgb;
          if (all(greaterThanEqual(p, ivec2(5))) && all(lessThan(p, size - 5))) {
            int hh = 0, hv = 0;
            for (int j = -1; j <= 1; j++)
              for (int i = -1; i <= 1; i++) {
                uvec2 h = texelFetch(homo, p + ivec2(i, j), 0).rg;
                hh += int(h.x);
                hv += int(h.y);
              }
            ivec3 a = rgbAt(p, 0), b = rgbAt(p, 1);
            rgb = hh != hv ? (hv > hh ? b : a) : (a + b) >> 1;
          } else {
            ivec3 sum = ivec3(0), n = ivec3(0);
            for (int j = -1; j <= 1; j++)
              for (int i = -1; i <= 1; i++) {
                ivec2 q = p + ivec2(i, j);
                if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size))) continue;
                sum[FC(q)] += M(q);
                n[FC(q)] += 1;
              }
            int f = FC(p);
            for (int c = 0; c < 3; c++) rgb[c] = c == f ? M(p) : int(uint(sum[c]) / uint(max(n[c], 1)));
          }
          rgbOut = uvec4(uvec3(rgb), 0u);
          cam = vec4(toHalf(camera(rgb)), 0.0);
        }""",
    # Camera RGB averaged down (box_cam) for the look, luminance (box_luminance) for the base
    # layer, a float texture (box_texture) for the editor's preview.
    "box_cam": TO_CAM + BOX + """
        out vec4 v;
        void main() {
          ivec2 bx = boxX(), by = boxY();
          vec3 s = vec3(0.0);
          for (int y = by.x; y < by.y; y++) {
            vec3 row = vec3(0.0);
            for (int x = bx.x; x < bx.y; x++) row += camAt(ivec2(x, y));
            s += row / float(bx.y - bx.x);
          }
          v = vec4(s / float(by.y - by.x), 0.0);
        }""",
    "box_luminance": TO_CAM + BOX + """
        uniform vec3 mul; uniform vec3 weights;
        out float v;
        void main() {
          ivec2 bx = boxX(), by = boxY();
          float clipLevel = min(mul.r, min(mul.g, mul.b)), s = 0.0;
          for (int y = by.x; y < by.y; y++) {
            float row = 0.0;
            for (int x = bx.x; x < bx.y; x++) row += dot(min(camAt(ivec2(x, y)) * mul, vec3(clipLevel)), weights);
            s += row / float(bx.y - bx.x);
          }
          v = s / float(by.y - by.x);
        }""",
    "box_texture": BOX + """
        uniform sampler2D src;
        out vec4 v;
        void main() {
          ivec2 bx = boxX(), by = boxY();
          vec4 s = vec4(0.0);
          for (int y = by.x; y < by.y; y++) {
            vec4 row = vec4(0.0);
            for (int x = bx.x; x < bx.y; x++) row += texelFetch(src, ivec2(x, y), 0);
            s += row / float(bx.y - bx.x);
          }
          v = s / float(by.y - by.x);
        }""",
})


def _setup() -> dict:
    if not _gl:
        import moderngl

        try:
            ctx = moderngl.create_standalone_context(backend="egl")
        except Exception as e:
            raise RuntimeError(f"Exports need OpenGL through EGL (Mesa's libegl1): {e}") from e
        prog = ctx.program(
            vertex_shader=(WEB / "develop.vert").read_text(),
            fragment_shader=(WEB / "develop.frag").read_text(),
        )
        quad = ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], "f4").tobytes())
        _gl.update(ctx=ctx, prog=prog, quad=quad, vao=ctx.vertex_array(prog, [(quad, "2f", "pos")]), passes={})
        _gl["c"] = _gl_functions()
        _gl["finish"] = ctx.finish if _gl["c"] is None else _gl["c"].glFinish
    return _gl


def _gl_functions():
    """GL through ctypes, which lets go of the GIL while a call runs, for the calls that take
    long: waiting for the GPU, and copying images to and from it (~100 ms for a full-size one).
    moderngl holds the GIL meanwhile, which stalls every other thread (the other exports'
    Python code, the editor's quick preview)."""
    import ctypes

    for lib in ("libGL.so.1", "libOpenGL.so.0"):
        try:
            gl = ctypes.CDLL(lib)
            gl.glFinish, gl.glPixelStorei, gl.glReadPixels, gl.glGetTexImage, gl.glTexSubImage2D, gl.glTexImage2D  # all there?
            gl.glGenTextures, gl.glDeleteTextures, gl.glBindTexture, gl.glTexParameteri
            return gl
        except (OSError, AttributeError):
            continue
    return None


# GL's pixel formats (normalized or float, integer) by components, and types by moderngl dtype
_FORMATS = {1: (0x1903, 0x8D94), 2: (0x8227, 0x8228), 3: (0x1907, 0x8D98), 4: (0x1908, 0x8D99)}
_TYPES = {"f1": 0x1401, "f2": 0x140B, "f4": 0x1406, "u1": 0x1401, "u2": 0x1403, "u4": 0x1405, "i1": 0x1400, "i2": 0x1402, "i4": 0x1404}
GL_TEXTURE_2D, GL_PACK_ALIGNMENT, GL_UNPACK_ALIGNMENT = 0x0DE1, 0x0D05, 0x0CF5


def _read(obj, components: int | None = None, dtype: str = "f1") -> np.ndarray:
    """A texture's or a framebuffer's (first attachment's) pixels, as bytes (uint8), once the
    GPU is done; framebuffers in `components` and `dtype`."""
    import ctypes

    import moderngl

    gl = _setup()
    gl["finish"]()
    texture = isinstance(obj, moderngl.Texture)
    if texture:
        components, dtype = obj.components, obj.dtype
    if gl["c"] is None:
        return np.frombuffer(obj.read() if texture else obj.read(components=components, dtype=dtype, alignment=1), np.uint8)
    w, h = obj.size
    out = np.empty(h * w * components * int(dtype[1]), np.uint8)
    fmt, typ, ptr = _FORMATS[components][dtype[0] in "ui"], _TYPES[dtype], ctypes.c_void_p(out.ctypes.data)
    gl["c"].glPixelStorei(GL_PACK_ALIGNMENT, 1)
    if texture:
        obj.use(0)
        gl["c"].glGetTexImage(GL_TEXTURE_2D, 0, fmt, typ, ptr)
    else:
        obj.use()
        gl["c"].glReadPixels(0, 0, w, h, fmt, typ, ptr)
    return out


# GL's internal formats by moderngl dtype, for 1 to 4 components
_INTERNAL = {
    "f1": (0x8229, 0x822B, 0x8051, 0x8058),
    "f2": (0x822D, 0x822F, 0x881B, 0x881A),
    "f4": (0x822E, 0x8230, 0x8815, 0x8814),
    "u1": (0x8232, 0x8238, 0x8D7D, 0x8D7C),
    "u2": (0x8234, 0x823A, 0x8D77, 0x8D76),
    "u4": (0x8236, 0x823C, 0x8D71, 0x8D70),
    "i1": (0x8231, 0x8237, 0x8D8F, 0x8D8E),
    "i2": (0x8233, 0x8239, 0x8D89, 0x8D88),
    "i4": (0x8235, 0x823B, 0x8D83, 0x8D82),
}


def _new_texture(size, components: int, dtype: str):
    """A new texture, made through ctypes (see _gl_functions): the driver can take ~0.2 s to
    find the memory for a full-size one, with moderngl holding the GIL. moderngl doesn't
    delete textures it didn't make: _delete() does."""
    import ctypes

    gl = _setup()
    if gl["c"] is None:
        return gl["ctx"].texture(size, components, dtype=dtype)
    c, name = gl["c"], ctypes.c_uint(0)
    c.glGenTextures(1, ctypes.byref(name))
    c.glBindTexture(GL_TEXTURE_2D, name.value)
    fmt = _FORMATS[components][dtype[0] in "ui"]
    c.glTexImage2D(GL_TEXTURE_2D, 0, _INTERNAL[dtype][components - 1], int(size[0]), int(size[1]), 0, fmt, _TYPES[dtype], None)
    c.glTexParameteri(GL_TEXTURE_2D, 0x813D, 0)  # GL_TEXTURE_MAX_LEVEL: no mipmaps
    return gl["ctx"].external_texture(name.value, size, components, 0, dtype)


def _delete(tex):
    import ctypes

    c = _setup()["c"]
    if c is not None:
        c.glDeleteTextures(1, ctypes.byref(ctypes.c_uint(tex.glo)))
    tex.release()


def _write(tex, data):
    """Upload data (its dtype the texture's) into a whole texture (see _gl_functions)."""
    import ctypes

    c = _setup()["c"]
    data = np.ascontiguousarray(data)
    if c is None:
        tex.write(data)
        return
    (w, h), fmt = tex.size, _FORMATS[tex.components][tex.dtype[0] in "ui"]
    tex.use(0)
    c.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    c.glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, fmt, _TYPES[tex.dtype], ctypes.c_void_p(data.ctypes.data))


def _pass(name: str, target, textures: dict, **uniforms):
    """Draw one full-target pass of SHADERS[name] into framebuffer `target`."""
    import moderngl

    gl = _setup()
    if name not in gl["passes"]:
        prog = gl["ctx"].program(vertex_shader=PASS_VERTEX, fragment_shader=PASS_HEADER + SHADERS[name])
        gl["passes"][name] = (prog, gl["ctx"].vertex_array(prog, [(gl["quad"], "2f", "pos")]))
    prog, vao = gl["passes"][name]
    for unit, (uname, tex) in enumerate(textures.items()):
        tex.use(unit)
        prog[uname].value = unit
    for uname, value in uniforms.items():
        prog[uname].value = tuple(np.asarray(value).ravel().tolist()) if np.ndim(value) else value
    target.use()
    vao.render(moderngl.TRIANGLE_STRIP)


def _texture(size, components, data=None, dtype="f4", linear=False):
    """A texture from the pool (or a new one), with data written to it if given; hand it back
    with _release."""
    import moderngl

    free = _pool.get((tuple(size), components, dtype))
    tex = free.pop() if free else _new_texture(tuple(size), components, dtype)
    if data is not None:
        _write(tex, data)
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR) if linear else (moderngl.NEAREST, moderngl.NEAREST)
    tex.repeat_x = tex.repeat_y = False
    return tex


def _release(*objs):
    """Textures back to the pool, anything else (framebuffers) freed."""
    import moderngl

    for obj in objs:
        if isinstance(obj, moderngl.Texture):
            _pool.setdefault((tuple(obj.size), obj.components, obj.dtype), []).append(obj)
        else:
            obj.release()


def _box(src, tmp, out, r: int):
    """2D box mean of src into out (via tmp), separable."""
    ctx = _setup()["ctx"]
    for a, b, d in ((src, tmp, (1, 0)), (tmp, out, (0, 1))):
        fb = ctx.framebuffer([b])
        _pass("box", fb, {"src": a}, r=r, dir=d, scale=1 / (2 * r + 1))
        fb.release()


def _raw_nlm(mosaic, *args):
    size = (mosaic.shape[1], mosaic.shape[0])
    raw = _texture(size, 1, mosaic, dtype="u2")
    out, removed = _nlm(raw, size, *args)
    try:
        return _read(out), _read(removed)
    finally:
        _release(raw, out, removed)


def _nlm(raw, size, black, A, B, weights, h, search, patch, greens, gain):
    """raw_nlm() of the mosaic in texture `raw`: the denoised mosaic (R16UI) and what was
    removed (R32F, half size), as textures to hand back with _release."""
    import moderngl

    ctx = _setup()["ctx"]
    half = (size[0] // 2, size[1] // 2)
    z, acc = _texture(half, 4), _texture(half, 4)
    d, dist = (_texture(half, 4, dtype="f2") for _ in range(2))  # half floats: 25% faster, differences < 0.01 noise sd
    wgt, out, removed = _texture(half, 1), _texture(size, 1, dtype="u2"), _texture(half, 1)
    fbs = {k: ctx.framebuffer(t) for k, t in [("z", [z]), ("acc", [acc, wgt]), ("d", [d]), ("dist", [dist]), ("out", [out]), ("removed", [removed])]}
    vst = dict(A=A, c=3 / 8 * A * A + B)
    try:
        _pass("raw_planes", fbs["z"], {"raw": raw}, black=float(black), **vst)
        _pass("nlm_init", fbs["acc"], {"img": z}, **vst)
        r, R = patch // 2, search // 2
        offsets = [(dx, dy) for dy in range(R + 1) for dx in range(-R, R + 1) if dy > 0 or dx > 0]
        offsets += [(0, 0)] * (-len(offsets) % 4)
        inv_n = 1 / ((2 * r + 1) ** 2 * 4)
        for g in range(0, len(offsets), 4):
            o = {f"o{k}": offsets[g + k] for k in range(4)}
            ctx.disable(moderngl.BLEND)
            _pass("nlm_diff", fbs["d"], {"img": z}, r=r, **o)
            _pass("box", fbs["dist"], {"src": d}, r=r, dir=(0, 1), scale=1.0)
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.ONE, moderngl.ONE
            _pass("nlm_acc", fbs["acc"], {"dist": dist, "img": z}, inv_n=inv_n, inv_h2=1 / (h * h), **vst, **o)
        ctx.disable(moderngl.BLEND)
        _pass("raw_out", fbs["out"], {"raw": raw, "sum": acc, "weight": wgt}, gain=float(gain), black=float(black),
              greens=greens or (-1, -1), flatness=GREENS_FLAT, **vst)
        _pass("raw_removed", fbs["removed"], {"raw": raw, "sum": acc, "weight": wgt}, weights=tuple(weights), black=float(black))
        return out, removed
    except BaseException:
        _release(out, removed)
        raise
    finally:
        ctx.disable(moderngl.BLEND)
        _release(z, acc, d, dist, wgt, *fbs.values())


def _check_size(w: int, h: int):
    limit = _setup()["ctx"].info["GL_MAX_TEXTURE_SIZE"]
    if max(w, h) > limit:
        raise ValueError(f"{w}x{h} is larger than this GPU's textures ({limit})")


def _upload(cam) -> object:
    h, w = cam.shape[:2]
    _check_size(w, h)
    return _texture((w, h), 4, cam, dtype="f2")


def _demosaic(mosaic, pattern, black, saturation, flip, xyz, to_cam, nlm, look, base, original) -> Decoded:
    h, w = mosaic.shape
    size, out = (w, h), ((h, w) if flip & 4 else (w, h))
    _check_size(w, h)
    pat = tuple(1 if c == 3 else int(c) for c in pattern.ravel())  # the second green is green
    raw = _texture(size, 1, mosaic, dtype="u2")
    owned, orig, cam = [raw], None, None
    try:
        if original:
            rgb, orig = _ahd(raw, pat, black, saturation, flip, xyz, to_cam, out)
            _release(rgb)  # its textures are reused below
        removed, gain = None, 1
        if nlm is not None:
            raw, removed_tex = _nlm(raw, size, *nlm)
            owned += [raw, removed_tex]
            removed = np.frombuffer(_read(removed_tex), np.float32).reshape(h // 2, w // 2)
            gain = nlm[-1]
        rgb, cam = _ahd(raw, pat, gain * black, gain * saturation, flip, xyz, to_cam, out)
        owned.append(rgb)
        decoded = Decoded(out[0], out[1], cam, removed=removed, original=orig)
        if look is not None:
            look_size, box = look
            decoded.look = _resize("box_cam", {"rgb": rgb}, out, look_size, box, 3, toCam=to_cam)
        if base is not None:
            mul, weights, base_size = base
            decoded.base = _resize("box_luminance", {"rgb": rgb}, out, base_size, (0, 0, *out), 1, toCam=to_cam, mul=mul, weights=weights)[..., 0]
        return decoded
    except BaseException:
        _release(*[t for t in (orig, cam) if t is not None])
        raise
    finally:
        _release(*owned)


def _ahd(raw, pat, black, saturation, flip, xyz, to_cam, out) -> tuple:
    """LibRaw's AHD of the mosaic texture `raw`: its RGB (16-bit, oriented to size `out`, as
    LibRaw outputs it) and that as camera RGB (RGBA16F), to hand back with _release."""
    ctx = _setup()["ctx"]
    size = raw.size
    temps = []

    def tex(*args, **kwargs):
        temps.append(_texture(*args, **kwargs))
        return temps[-1]

    def fb(*texs):
        temps.append(ctx.framebuffer(list(texs)))
        return temps[-1]

    rgb = cam = None
    try:
        scaled, green, homo = tex(size, 1, dtype="u2"), tex(size, 2, dtype="u2"), tex(size, 2, dtype="u1")
        lab_h, lab_v = tex(size, 4, dtype="i2"), tex(size, 4, dtype="i2")
        cbrt = tex((256, 256), 1, _cbrt_table())
        _pass("ahd_scale", fb(scaled), {"raw": raw}, black=black, scale=float(np.float32(65535) / np.float32(saturation)))
        _pass("ahd_green", fb(green), {"mosaic": scaled}, pat=pat)
        lab = {"mosaic": scaled, "green": green, "cbrtTable": cbrt}
        _pass("ahd_lab", fb(lab_h, lab_v), lab, pat=pat, xc0=xyz[:, 0], xc1=xyz[:, 1], xc2=xyz[:, 2])
        _pass("ahd_homo", fb(homo), {"labH": lab_h, "labV": lab_v})
        rgb, cam = _texture(out, 4, dtype="u2"), _texture(out, 4, dtype="f2")
        _pass("ahd_combine", fb(rgb, cam), {"mosaic": scaled, "green": green, "homo": homo}, pat=pat, flip=flip, toCam=to_cam)
        return rgb, cam
    except BaseException:
        _release(*[t for t in (rgb, cam) if t is not None])
        raise
    finally:
        _release(*temps)


def _resize(shader: str, source: dict, in_size, size, box, components: int, **uniforms) -> np.ndarray:
    """PIL's BOX resize of what `shader` (box_cam, box_luminance, box_texture) makes of its
    `source` texture (in_size), over `box`, down to `size`: float32, height x width x components."""
    ctx = _setup()["ctx"]
    (w, h), (ow, oh), (x0, y0, x1, y1) = in_size, size, box
    bx, by = _texture((ow, 1), 2, _box_bounds(w, x0, x1, ow), dtype="i4"), _texture((oh, 1), 2, _box_bounds(h, y0, y1, oh), dtype="i4")
    small = _texture((ow, oh), 4 if components > 1 else 1)
    fb = ctx.framebuffer([small])
    try:
        _pass(shader, fb, {**source, "boundsX": bx, "boundsY": by}, **uniforms)
        return _read(fb, components=components, dtype="f4").view(np.float32).reshape(oh, ow, components)
    finally:
        _release(bx, by, small, fb)


def _chroma_nr(cam, size, mul, rgb_cam, eps, original=None, amount=1.0):
    """Camera RGB with colour noise smoothed (RGBA16F texture); eps: denoise.chroma_eps(). Mixed
    with `original` (camera RGB without noise reduction), `amount` of it smoothed."""
    import moderngl

    from . import denoise

    ctx = _setup()["ctx"]
    w, h = size
    hs = ((w + 1) // 2, (h + 1) // 2)
    t0, t1, tmp, m0, m1, coef, mc = (_texture(hs, 4) for _ in range(7))
    common = dict(mul=mul, clipLevel=float(min(mul)), rgbCam=rgb_cam.T)
    fbs = []

    def fb(*texs):
        f = ctx.framebuffer(list(texs))
        fbs.append(f)
        return f

    try:
        _pass("chroma_prep", fb(t0, t1), {"cam": cam}, **common)
        for _ in range(2):
            _box(t0, tmp, m0, denoise.CHROMA_RADIUS)
            _box(t1, tmp, m1, denoise.CHROMA_RADIUS)
            _pass("chroma_coef", fb(coef), {"m0": m0, "m1": m1}, eps=eps)
            _box(coef, tmp, mc, denoise.CHROMA_RADIUS)
            _pass("chroma_apply", fb(t0, t1), {"t0in": t0, "mc": mc})
        t0.filter = (moderngl.LINEAR, moderngl.LINEAR)
        out = _texture(size, 4, dtype="f2")
        original, amount = (cam, 1.0) if original is None else (original, amount)
        _pass("chroma_final", fb(out), {"cam": cam, "chroma": t0, "original": original}, amount=amount, camRgb=np.linalg.inv(rgb_cam).T, **common)
        return out
    finally:
        _release(t0, t1, tmp, m0, m1, coef, mc, *fbs)


def _grain(removed, weights, size):
    """Full-size grain to add (R16F texture): the removed luminance, partly given back per scale."""
    ctx = _setup()["ctx"]
    half = _texture((removed.shape[1], removed.shape[0]), 1, removed, linear=True)
    rows, grain = _texture(size, 4, dtype="f2"), _texture(size, 1, dtype="f2")
    fbs = [ctx.framebuffer([t]) for t in (rows, grain)]
    try:
        _pass("grain_rows", fbs[0], {"src": half}, size=size)
        _pass("grain_columns", fbs[1], {"rows": rows}, w=weights)
        return grain
    finally:
        _release(half, rows, *fbs)


def _with_noise(cam, noise, removed, size, original=None) -> tuple:
    """The develop shader's camera RGB and grain textures, from the camera RGB texture (and the
    one without noise reduction, to mix in: Decoded.original)."""
    smoothed = _chroma_nr(cam, size, noise.as_shot.astype(np.float32), noise.rgb_cam, noise.chroma_eps, original, noise.amount)
    return smoothed, _grain(removed, noise.weights, size)


def _read_cam(cam, noise, removed, shrink, original) -> tuple:
    grain, owned = None, []
    if noise is not None:
        cam, grain = owned = _with_noise(cam, noise, removed, cam.size, original)
    try:
        small = {}
        for size in shrink:
            box = (0, 0, *cam.size)
            small[size] = _resize("box_texture", {"src": cam}, cam.size, size, box, 3), None if grain is None else _resize("box_texture", {"src": grain}, cam.size, size, box, 1)[..., 0]
        return _read(cam), None if grain is None else _read(grain), small
    finally:
        _release(*owned)


def _render(cam, coeffs, uniforms, tone, table, noise, removed, original) -> np.ndarray:
    """sRGB, HxWx3 uint8, bottom row first, from the camera RGB texture `cam`."""
    import moderngl

    gl = _setup()
    ctx, prog = gl["ctx"], gl["prog"]
    w, h = cam.size
    owned = []
    if noise is not None:
        cam, grain = _with_noise(cam, noise, removed, (w, h), original)
        owned.append(cam)
    else:
        grain = _texture((1, 1), 1, np.zeros((1, 1), np.float32))
    owned.append(grain)
    co = _texture((coeffs.shape[1], coeffs.shape[0]), 2, coeffs, linear=True)
    tn = _texture((tone.size, 1), 1, tone, linear=True)
    tb = ctx.texture3d((table.shape[2], table.shape[1], table.shape[0]), 3, table, dtype="f4")
    tb.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tb.repeat_x = tb.repeat_y = tb.repeat_z = False
    target = _texture((w, h), 4, dtype="f1")
    fbo = ctx.framebuffer([target])
    owned += [co, tn, tb, target, fbo]
    try:
        for unit, (tex, name) in enumerate([(cam, "cam"), (co, "coeffs"), (tn, "tone"), (tb, "colour"), (grain, "grain")]):
            tex.use(unit)
            prog[name].value = unit
        for name, value in uniforms.items():
            if prog.get(name, None) is not None:
                prog[name].value = tuple(value) if isinstance(value, (list, tuple, np.ndarray)) else value
        fbo.use()
        gl["vao"].render(moderngl.TRIANGLE_STRIP)
        return _read(fbo, components=3).reshape(h, w, 3)
    finally:
        _release(*owned)
