import re
from pathlib import Path

import numpy as np
import pytest

from photos import develop

FRAG = (Path(develop.__file__).parent / "web" / "develop.frag").read_text()
# Camera RGB that already is linear sRGB, white balanced: the pipeline's maths shows through.
PLAIN = develop.Color(np.eye(3), np.eye(3), np.ones(3), np.ones(3), 6500.0)


@pytest.fixture(scope="module")
def gpu():
    from photos import gpu

    try:
        gpu.info()
    except Exception as e:
        pytest.skip(f"no OpenGL: {e}")
    return gpu


def render(gpu, rgb, **settings):
    rgb = np.asarray(rgb, np.float32)
    return gpu.render(rgb, develop.base_coefficients(rgb, PLAIN), develop.uniforms(PLAIN, settings)).astype(int)


def test_every_shader_input_is_supplied():
    declared = set(re.findall(r"uniform (?:float|vec3|mat3) (\w+);", FRAG))
    assert declared == set(develop.uniforms(PLAIN, {}))


def test_rows_come_out_in_order(gpu):
    img = np.zeros((40, 60, 3), np.float32)
    img[:20] = 0.5  # bright top half
    out = render(gpu, img)
    assert out.shape == (40, 60, 3)
    assert out[:20].min() > 100 and out[20:].max() == 0


def test_scene_mid_grey_becomes_display_mid_grey(gpu):
    out = render(gpu, np.full((8, 8, 3), 2**develop.ANCHOR))
    assert np.all(np.abs(out - 118) <= 1)  # sRGB-encoded 0.18


def test_sliders_move_the_right_way(gpu):
    img = np.full((64, 64, 3), 2**develop.ANCHOR, np.float32)
    img[:, :32] /= 16  # a deep shadow next to mid-grey
    base = render(gpu, img)
    assert render(gpu, img, exposure=1).mean() > base.mean()
    assert render(gpu, img, compress=100)[:, :16].mean() > base[:, :16].mean()  # shadows lifted
    grey = render(gpu, np.full((8, 8, 3), 0.05), saturation=100)
    assert np.ptp(grey) <= 1  # saturation leaves greys grey
