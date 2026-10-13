import re
from pathlib import Path

import numpy as np
import pytest

from photos import develop, look

FRAG = (Path(develop.__file__).parent / "web" / "develop.frag").read_text()
# Camera RGB that already is linear sRGB, white balanced: the pipeline's maths shows through.
PLAIN = develop.Color(np.eye(3), np.eye(3), np.ones(3), np.ones(3), 6500.0)
NEUTRAL = look.Look(look.fallback_tone().astype(np.float32), look.identity_table(), 0.0, 0.0)
NEUTRAL.anchor = float(np.log2(look.tone_inverse(NEUTRAL.tone, 0.18)))
CAMERA = look.load_profile("NIKON D3500")


@pytest.fixture(scope="module")
def gpu():
    from photos import gpu

    try:
        gpu.info()
    except Exception as e:
        pytest.skip(f"no OpenGL: {e}")
    return gpu


def render(gpu, rgb, lk=NEUTRAL, **settings):
    rgb = np.asarray(rgb, np.float32)
    coeffs = develop.base_coefficients(rgb, PLAIN)
    return gpu.render(rgb, coeffs, develop.uniforms(PLAIN, lk, settings), *develop.look_textures(lk)).astype(int)


def test_every_shader_input_is_supplied():
    declared = set(re.findall(r"uniform (?:float|vec2|vec3|mat3) (\w+);", FRAG))
    assert declared == set(develop.uniforms(PLAIN, NEUTRAL, {}))


def test_rows_come_out_in_order(gpu):
    img = np.zeros((40, 60, 3), np.float32)
    img[:20] = 0.5  # bright top half
    out = render(gpu, img)
    assert out.shape == (40, 60, 3)
    assert out[:20].min() > 100 and out[20:].max() == 0


def test_scene_mid_grey_becomes_display_mid_grey(gpu):
    out = render(gpu, np.full((8, 8, 3), 2**NEUTRAL.anchor))
    assert np.all(np.abs(out - 118) <= 1)  # sRGB-encoded 0.18


def test_sliders_move_the_right_way(gpu):
    img = np.full((64, 64, 3), 2**NEUTRAL.anchor, np.float32)
    img[:, :32] /= 16  # a deep shadow next to mid-grey
    base = render(gpu, img)
    assert render(gpu, img, exposure=1).mean() > base.mean()
    assert render(gpu, img, compress=100)[:, :16].mean() > base[:, :16].mean()  # shadows lifted
    assert render(gpu, img, contrast=100)[:, :16].mean() < base[:, :16].mean()  # shadows deepened
    grey = render(gpu, np.full((8, 8, 3), 0.05), saturation=100)
    assert np.ptp(grey) <= 1  # saturation leaves greys grey


def test_shader_matches_the_look_model(gpu):
    """The shader's last steps agree with look.model, which the calibration fits with."""
    rng = np.random.default_rng(1)
    scene = np.exp2(rng.uniform(-9, 0, (48, 64, 1))) * rng.uniform(0.15, 1.0, (48, 64, 3))  # below sensor clipping
    tone = np.clip(1.1 * (2.0**look.GRID) ** 0.5 - 0.02, 0, 1) + np.arange(look.GRID.size) * 1e-7
    lk = look.Look(tone.astype(np.float32), CAMERA.table, 0.6, float(np.log2(look.tone_inverse(tone, 0.18))))
    for contrast in (0, 60):
        out = render(gpu, scene, lk, contrast=contrast)
        expected = look.model(scene.astype(np.float32), lk.tone, lk.table, lk.shadow_loss, contrast)
        expected = np.round(np.where(expected <= 0.0031308, 12.92 * expected, 1.055 * expected ** (1 / 2.4) - 0.055) * 255)
        assert np.abs(out - expected).max() <= 2
