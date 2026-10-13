"""GPU work on the server: the develop shader for exports, and noise reduction.

The develop shader (web/develop.frag) also renders the editor's preview in the browser,
so an export looks like what you saw. Noise reduction (see denoise.py) only runs here:
non-local means on the raw planes, and after demosaicing a guided filter for colour
noise and the grain given back. Everything runs on the GPU when Mesa can reach one (on
WSL, through D3D12) and on Mesa's software renderer otherwise. A GL context belongs to
one thread at a time, so all GL work happens on a single thread of its own.
"""

from __future__ import annotations

import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WEB = Path(__file__).parent / "web"
POOL_IDLE = 30  # seconds without GPU work, after which the texture pool is freed
_thread = ThreadPoolExecutor(1, thread_name_prefix="gpu")
_gl: dict = {}
# Free textures by (size, components, dtype), for reuse: new GPU memory is slow (on WSL
# ~2 ms/MB once the driver has given freed memory back, and an export uses ~1.5 GB).
_pool: dict = {}
_jobs = itertools.count(1)
_latest = 0


def render(rgb, coeffs, uniforms, tone, table, noise=None) -> np.ndarray:
    """camera RGB (HxWx3 float32) + base coefficients (hxwx2) + the look's tone curve (n) and
    colour table (depth x height x width x 3, see develop.look_textures) -> sRGB uint8 HxWx3.
    With `noise` (denoise.Noise), colour noise is smoothed and grain given back first.

    Conversions happen here, on the caller's thread, so the GL thread only does GL work
    (it is shared by all exports)."""
    grain = None if noise is None else _f32(noise.grain)
    out = _run(_render, _rgba16(rgb), _f32(coeffs), uniforms, _f32(tone), _f32(table), noise, grain)
    return np.ascontiguousarray(out[::-1, :, :3])  # GL reads the bottom row first


def denoised(rgb, noise) -> tuple[np.ndarray, np.ndarray]:
    """What render() feeds the develop shader when given `noise`, read back: camera RGB with
    colour noise smoothed (HxWx4 float16) and the grain to give back (HxW float16). The
    editor's 100% view runs the shader on these in the browser, so it shows the export."""
    h, w = rgb.shape[:2]
    cam, grain = _run(_denoised, _rgba16(rgb), noise, _f32(noise.grain))
    return np.frombuffer(cam, np.float16).reshape(h, w, 4), np.frombuffer(grain, np.float16).reshape(h, w)


def _rgba16(rgb) -> np.ndarray:
    cam = np.empty(rgb.shape[:2] + (4,), np.float16)  # RGBA16F: the fastest upload, as in the browser
    cam[..., :3] = rgb
    cam[..., 3] = 0
    return cam


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, np.float32)


def raw_nlm(mosaic: np.ndarray, black: float, A: float, B: float, weights, h: float, search: int, patch: int, gain: int = 1):
    """Non-local means on a Bayer mosaic (uint16, 2x2 pattern): its four planes as one
    4-channel image, so every colour gets the same weights, after a generalized Anscombe
    transform for noise variance A * signal + B (black subtracted), which makes the noise ~1.

    Returns the denoised mosaic, times `gain` (uint16: what denoising adds below one raw unit
    survives), and per 2x2 tile, sum(weights * removed) over the four photosites (float32,
    half size)."""
    out, removed = _run(_raw_nlm, np.ascontiguousarray(mosaic, np.uint16), black, A, B, weights, h, search, patch, gain)
    hh, ww = mosaic.shape[0] // 2, mosaic.shape[1] // 2
    return np.frombuffer(out, np.uint16).reshape(mosaic.shape), np.frombuffer(removed, np.float32).reshape(hh, ww)


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
            tex.release()
        _pool.clear()


# --- Shaders for the passes that only run here -----------------------------------------------

PASS_VERTEX = """#version 330
in vec2 pos;
void main() { gl_Position = vec4(pos, 0.0, 1.0); }"""
PASS_HEADER = """#version 330
// Weights at offset i of box filters of radius 1, 2 and 4, each applied twice
#define TENT(i) (max(vec3(3.0, 5.0, 9.0) - abs(float(i)), 0.0) / vec3(9.0, 25.0, 81.0))
"""

# The raw mosaic, a 2x2 tile at a time (photosites in denoise.PHASES order), and the
# variance-stabilising transform.
RAW = """
    uniform usampler2D raw; uniform float black, A, c;
    vec4 tile(ivec2 t) {
      ivec2 q = 2 * t;
      return vec4(texelFetch(raw, q, 0).r, texelFetch(raw, q + ivec2(1, 0), 0).r,
                  texelFetch(raw, q + ivec2(0, 1), 0).r, texelFetch(raw, q + ivec2(1, 1), 0).r) - black;
    }
    vec4 forward(vec4 x) { return 2.0 / A * sqrt(max(A * x + c, 0.0)); }
    vec4 inverse(vec4 z) { return ((z * A / 2.0) * (z * A / 2.0) - c) / A; }
"""

SHADERS = {
    "raw_planes": RAW + """
        out vec4 z;
        void main() { z = forward(tile(ivec2(gl_FragCoord.xy))); }""",
    # The denoised mosaic: the NLM result (sum / weight) back in raw units.
    "raw_out": RAW + """
        uniform sampler2D sum; uniform sampler2D weight; uniform float gain;
        out uint v;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy), t = p / 2;
          float x = float(texelFetch(raw, p, 0).r);  // odd size: no whole tile here
          if (all(lessThan(t, textureSize(sum, 0))))
            x = inverse(texelFetch(sum, t, 0) / texelFetch(weight, t, 0).r)[(p.y & 1) * 2 + (p.x & 1)] + black;
          v = uint(clamp(round(x * gain), 0.0, 65535.0));
        }""",
    "raw_removed": RAW + """
        uniform sampler2D sum; uniform sampler2D weight; uniform vec4 weights;
        out float v;
        void main() {
          ivec2 t = ivec2(gl_FragCoord.xy);
          v = dot(weights, tile(t) - inverse(texelFetch(sum, t, 0) / texelFetch(weight, t, 0).r));
        }""",
    # Non-local means, four search offsets per pass. Each offset o is done once for both
    # pairs it stands for, (p, p + o) and (p - o, p): the patch distance is the same.
    "nlm_init": """
        uniform sampler2D img;
        layout(location = 0) out vec4 sum; layout(location = 1) out float weight;
        void main() { sum = texelFetch(img, ivec2(gl_FragCoord.xy), 0); weight = 1.0; }""",
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
    "nlm_acc": """
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
            sum += wf * texelFetch(img, clamp(p + o[k], ivec2(0), last), 0) + wb * texelFetch(img, clamp(q, ivec2(0), last), 0);
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
    # Full size again: the pixel's own luminance, the smoothed chroma (bilinear), back to camera RGB.
    "chroma_final": """
        uniform sampler2D cam; uniform sampler2D chroma; uniform vec3 mul; uniform float clipLevel; uniform mat3 rgbCam;
        uniform mat3 camRgb;
        out vec4 result;
        void main() {
          ivec2 p = ivec2(gl_FragCoord.xy);
          vec3 scene = rgbCam * min(texelFetch(cam, p, 0).rgb * mul, vec3(clipLevel));
          vec3 s = sign(scene) * sqrt(abs(scene));
          vec2 ch = texture(chroma, gl_FragCoord.xy / vec2(textureSize(cam, 0))).gb;
          float g = dot(s, vec3(0.25, 0.5, 0.25)) - 0.25 * (ch.x + ch.y);
          vec3 s2 = vec3(g + ch.x, g, g + ch.y);
          result = vec4((camRgb * (sign(s2) * s2 * s2)) / mul, 1.0);
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
        _gl["finish"] = _gl_finish() or ctx.finish
    return _gl


def _gl_finish():
    """glFinish through ctypes, which lets go of the GIL while it waits. moderngl holds the GIL
    while it blocks, so waiting for the GPU in a read stalls every other thread (the other
    exports' Python code)."""
    import ctypes

    for lib in ("libGL.so.1", "libOpenGL.so.0"):
        try:
            return ctypes.CDLL(lib).glFinish
        except (OSError, AttributeError):
            continue
    return None


def _read(obj, **kwargs) -> bytes:
    """Read a texture or framebuffer once the GPU is done (see _gl_finish)."""
    _setup()["finish"]()
    return obj.read(**kwargs)


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
    tex = free.pop() if free else _setup()["ctx"].texture(size, components, dtype=dtype)
    if data is not None:
        tex.write(data)
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


def _raw_nlm(mosaic, black, A, B, weights, h, search, patch, gain):
    import moderngl

    ctx = _setup()["ctx"]
    size, half = (mosaic.shape[1], mosaic.shape[0]), (mosaic.shape[1] // 2, mosaic.shape[0] // 2)
    raw, z, acc = _texture(size, 1, mosaic, dtype="u2"), _texture(half, 4), _texture(half, 4)
    d, dist = (_texture(half, 4, dtype="f2") for _ in range(2))  # half floats: 25% faster, differences < 0.01 noise sd
    wgt, out, removed = _texture(half, 1), _texture(size, 1, dtype="u2"), _texture(half, 1)
    fbs = {k: ctx.framebuffer(t) for k, t in [("z", [z]), ("acc", [acc, wgt]), ("d", [d]), ("dist", [dist]), ("out", [out]), ("removed", [removed])]}
    common = dict(black=float(black), A=A, c=3 / 8 * A * A + B)
    try:
        _pass("raw_planes", fbs["z"], {"raw": raw}, **common)
        _pass("nlm_init", fbs["acc"], {"img": z})
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
            _pass("nlm_acc", fbs["acc"], {"dist": dist, "img": z}, inv_n=inv_n, inv_h2=1 / (h * h), **o)
        ctx.disable(moderngl.BLEND)
        _pass("raw_out", fbs["out"], {"raw": raw, "sum": acc, "weight": wgt}, gain=float(gain), **common)
        _pass("raw_removed", fbs["removed"], {"raw": raw, "sum": acc, "weight": wgt}, weights=tuple(weights), **common)
        return _read(out), _read(removed)
    finally:
        ctx.disable(moderngl.BLEND)
        _release(raw, z, acc, d, dist, wgt, out, removed, *fbs.values())


def _chroma_nr(cam, size, mul, rgb_cam):
    """Camera RGB with colour noise smoothed (RGBA16F texture)."""
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
            _pass("chroma_coef", fb(coef), {"m0": m0, "m1": m1}, eps=denoise.CHROMA_EPS)
            _box(coef, tmp, mc, denoise.CHROMA_RADIUS)
            _pass("chroma_apply", fb(t0, t1), {"t0in": t0, "mc": mc})
        t0.filter = (moderngl.LINEAR, moderngl.LINEAR)
        out = _texture(size, 4, dtype="f2")
        _pass("chroma_final", fb(out), {"cam": cam, "chroma": t0}, camRgb=np.linalg.inv(rgb_cam).T, **common)
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


def _with_noise(cam, noise, removed, size) -> tuple:
    """The develop shader's camera RGB and grain textures, from the camera RGB texture."""
    smoothed = _chroma_nr(cam, size, noise.as_shot.astype(np.float32), noise.rgb_cam)
    return smoothed, _grain(removed, noise.weights, size)


def _denoised(cam, noise, removed) -> tuple[bytes, bytes]:
    h, w = cam.shape[:2]
    src = _texture((w, h), 4, cam, dtype="f2")
    smoothed, grain = _with_noise(src, noise, removed, (w, h))
    try:
        return _read(smoothed), _read(grain)
    finally:
        _release(src, smoothed, grain)


def _render(cam, coeffs, uniforms, tone, table, noise, removed) -> np.ndarray:
    """RGBA8, bottom row first."""
    import moderngl

    gl = _setup()
    ctx, prog = gl["ctx"], gl["prog"]
    h, w = cam.shape[:2]
    if max(h, w) > ctx.info["GL_MAX_TEXTURE_SIZE"]:
        raise ValueError(f"{w}x{h} is larger than this GPU's textures ({ctx.info['GL_MAX_TEXTURE_SIZE']})")
    owned = []
    cam = _texture((w, h), 4, cam, dtype="f2")
    owned.append(cam)
    if noise is not None:
        cam, grain = _with_noise(cam, noise, removed, (w, h))
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
        return np.frombuffer(_read(fbo, components=4), np.uint8).reshape(h, w, 4)
    finally:
        _release(*owned)
