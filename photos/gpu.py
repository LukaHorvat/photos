"""Runs the editor's develop shader (web/develop.frag) on the server, for exports.

The same shader renders the editor's preview in the browser, so an export looks like
what you saw. It runs on the GPU when Mesa can reach one (on WSL, through D3D12) and
on Mesa's software renderer otherwise. A GL context belongs to one thread at a time,
so all GL work happens on a single thread of its own.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

WEB = Path(__file__).parent / "web"
_thread = ThreadPoolExecutor(1, thread_name_prefix="gpu")
_gl: dict = {}


def render(rgb: np.ndarray, coeffs: np.ndarray, uniforms: dict, tone: np.ndarray, table: np.ndarray) -> np.ndarray:
    """camera RGB (HxWx3 float32) + base coefficients (hxwx2) + the look's tone curve (n) and
    colour table (depth x height x width x 3, see develop.look_textures) -> sRGB uint8 HxWx3."""
    return _thread.submit(_render, rgb, coeffs, uniforms, tone, table).result()


def info() -> dict:
    """The GL renderer in use (creates the context if needed)."""
    return _thread.submit(lambda: dict(_setup()["ctx"].info)).result()


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
        _gl.update(ctx=ctx, prog=prog, vao=ctx.vertex_array(prog, [(quad, "2f", "pos")]))
    return _gl


def _render(rgb: np.ndarray, coeffs: np.ndarray, uniforms: dict, tone: np.ndarray, table: np.ndarray) -> np.ndarray:
    import moderngl

    gl = _setup()
    ctx, prog = gl["ctx"], gl["prog"]
    h, w = rgb.shape[:2]
    if max(h, w) > ctx.info["GL_MAX_TEXTURE_SIZE"]:
        raise ValueError(f"{w}x{h} is larger than this GPU's textures ({ctx.info['GL_MAX_TEXTURE_SIZE']})")
    cam = ctx.texture((w, h), 3, np.ascontiguousarray(rgb, np.float32), dtype="f4")
    cam.filter = (moderngl.NEAREST, moderngl.NEAREST)  # output pixels sit exactly on its texels
    co = ctx.texture((coeffs.shape[1], coeffs.shape[0]), 2, np.ascontiguousarray(coeffs, np.float32), dtype="f4")
    co.filter = (moderngl.LINEAR, moderngl.LINEAR)
    co.repeat_x = co.repeat_y = False
    tn = ctx.texture((tone.size, 1), 1, np.ascontiguousarray(tone, np.float32), dtype="f4")
    tn.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tn.repeat_x = tn.repeat_y = False
    tb = ctx.texture3d((table.shape[2], table.shape[1], table.shape[0]), 3, np.ascontiguousarray(table, np.float32), dtype="f4")
    tb.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tb.repeat_x = tb.repeat_y = tb.repeat_z = False
    fbo = ctx.simple_framebuffer((w, h), components=4)
    try:
        for unit, (tex, name) in enumerate([(cam, "cam"), (co, "coeffs"), (tn, "tone"), (tb, "colour")]):
            tex.use(unit)
            prog[name].value = unit
        for name, value in uniforms.items():
            prog[name].value = tuple(value) if isinstance(value, (list, tuple, np.ndarray)) else value
        fbo.use()
        gl["vao"].render(moderngl.TRIANGLE_STRIP)
        out = np.frombuffer(fbo.read(components=3, alignment=1), np.uint8).reshape(h, w, 3)
        return np.ascontiguousarray(out[::-1])  # GL reads the bottom row first
    finally:
        for obj in (cam, co, tn, tb, fbo):
            obj.release()
