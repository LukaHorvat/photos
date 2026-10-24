import numpy as np
import pytest

from photos import denoise, develop, neural

GAIN = 1e6  # the model's output gain is arbitrary


def nearest(tile):
    """A stand-in for the model: RGGB planes (4 x n x n) -> camera RGB (3 x 2n x 2n), each 2x2
    tile's colours repeated, at an arbitrary gain."""
    r, g1, g2, b = tile
    up = lambda p: np.repeat(np.repeat(p, 2, 0), 2, 1)
    return GAIN * np.stack([up(r), up((g1 + g2) / 2), up(b)])


def test_tiles_blend_into_one_photo():
    rng = np.random.default_rng(0)
    mosaic = rng.uniform(-0.01, 1, (1100, 1500)).astype(np.float32)  # 2 x 2 tiles, with mirrored edges
    rgb = neural.denoise(mosaic, np.array([[0, 1], [3, 2]]), run=nearest)
    r, g1, g2, b = (mosaic[i::2, j::2] for i, j in denoise.PHASES)
    up = lambda p: np.repeat(np.repeat(p, 2, 0), 2, 1)
    assert rgb.shape == (1100, 1500, 3) and rgb.dtype == np.float32
    assert np.allclose(rgb, np.stack([up(r), up((g1 + g2) / 2), up(b)], -1), atol=1e-6)  # no seams, gain undone


@pytest.mark.parametrize("pattern", [[[1, 2], [0, 3]], [[1, 0], [2, 3]], [[2, 1], [3, 0]]])  # GBRG, GRBG, BGGR
def test_red_is_moved_to_the_top_left(pattern):
    colour = np.float32([0.2, 0.5, 0.8])
    pattern = np.array(pattern)
    mosaic = np.tile(colour[np.where(pattern == 3, 1, pattern)], (551, 751))[:1101, :1501]  # odd sizes too
    rgb = neural.denoise(mosaic, pattern, run=nearest)
    assert rgb.shape == (1101, 1501, 3)
    assert np.allclose(rgb, colour, atol=1e-6)


def test_removed_is_what_each_photosite_lost():
    rng = np.random.default_rng(1)
    pattern = np.array([[0, 1], [3, 2]])
    rgb = rng.uniform(0, 1, (6, 8, 3)).astype(np.float32)
    mosaic = np.choose(np.tile(np.where(pattern == 3, 1, pattern), (3, 4)), np.moveaxis(rgb, -1, 0))
    weights = [0.1, 0.2, 0.3, 0.4]
    assert np.allclose(denoise.removed(mosaic, rgb, pattern, weights), 0)
    assert np.allclose(denoise.removed(mosaic + 0.5, rgb, pattern, weights), 0.5 * sum(weights))


def test_neural_caps_the_strengths_and_needs_cuda(monkeypatch):
    s = develop.clean({"noise": 150, "chroma": None, "neural": 1})
    assert s["neural"] is True and s["noise"] == 100 and s["chroma"] is None
    assert develop.clean({"noise": 150})["noise"] == 150 and develop.clean({})["neural"] is False
    monkeypatch.setattr(neural, "available", lambda: True)
    assert develop.reduction(s, 25600) == denoise.Reduction(1, 1, True)
    monkeypatch.setattr(neural, "available", lambda: False)  # non-local means instead
    assert develop.reduction(s, 25600) == denoise.Reduction(1, 1, False)
    assert not develop.reduction(develop.clean({"neural": True}), 200)  # Auto is off at low ISO, either way


@pytest.mark.skipif(not neural.available() or not (neural.CACHE / "rawnind-bayer-1.0.onnx").exists(), reason="needs CUDA and the model downloaded")
def test_rawnind_denoises_and_keeps_the_level():
    rng = np.random.default_rng(2)
    level = np.float32([0.10, 0.20, 0.20, 0.06])  # R G G B
    clean = np.tile(level.reshape(2, 2), (512, 512))
    noisy = (clean + rng.normal(0, 1, clean.shape) * np.sqrt(0.002 * clean + 1e-4)).astype(np.float32)
    rgb = neural.denoise(noisy, np.array([[0, 1], [3, 2]]))
    inner = rgb[100:-100, 100:-100]
    assert np.allclose(inner.mean((0, 1)), [0.10, 0.20, 0.06], rtol=0.02)
    assert inner[..., 1].std() < 0.2 * noisy[1::2, 0::2].std()
