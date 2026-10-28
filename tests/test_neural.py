import numpy as np
import pytest
from PIL import Image

from photos import denoise, develop, neural

GAIN = 1e6  # the model's output gain is arbitrary


@pytest.fixture(scope="module")
def gpu():
    from photos import gpu

    try:
        gpu.info()
    except Exception as e:
        pytest.skip(f"no OpenGL: {e}")
    return gpu


def nearest(tile):
    """A stand-in for the model: RGGB planes (4 x n x n) -> camera RGB (3 x 2n x 2n), each 2x2
    tile's colours repeated, at an arbitrary gain."""
    r, g1, g2, b = tile
    up = lambda p: np.repeat(np.repeat(p, 2, 0), 2, 1)
    return GAIN * np.stack([up(r), up((g1 + g2) / 2), up(b)])


def blended(mosaic, pattern, black=0.0, white=1.0, run=nearest) -> np.ndarray:
    """neural.denoise()'s tiles blended on the CPU, its gain applied: camera RGB (H x W x 3)."""
    h, w = mosaic.shape
    acc = np.zeros((h, w, 3))

    def blend(y, x, out, wy, wx):
        rows, cols = slice(max(-y, 0), min(h - y, out.shape[1])), slice(max(-x, 0), min(w - x, out.shape[2]))
        acc[y + rows.start : y + rows.stop, x + cols.start : x + cols.stop] += np.moveaxis(out[:, rows, cols], 0, -1) * wy[rows, None, None] * wx[cols, None]

    return acc * neural.denoise(mosaic, pattern, black, white, blend, run=run)


@pytest.fixture
def small_tiles(monkeypatch):
    monkeypatch.setattr(neural, "MAX_TILE", 256)  # many tiles, and not square, in small test photos


def test_tiling_covers_with_few_tiles():
    for n in (100, 1023, 1024, 1025, 2008, 3008, 5000):
        size, step, count = neural._tiling(n)
        assert size <= neural.MAX_TILE and size % 16 == 0 and size == step + 2 * neural.OVERLAP
        assert count * step >= n and (count - 1) * (neural.MAX_TILE - 2 * neural.OVERLAP) < n
    assert neural._tiling(2008) == neural._tiling(3008)[:2] + (2,) == (1072, 1008, 2)  # a 24 Mpx photo: 2 x 3 tiles


def test_tiles_blend_into_one_photo(small_tiles):
    rng = np.random.default_rng(0)
    mosaic = rng.uniform(-0.01, 1, (1100, 1500)).astype(np.float32)  # 3 x 4 tiles, with mirrored edges
    rgb = blended(mosaic, np.array([[0, 1], [3, 2]]))
    r, g1, g2, b = (mosaic[i::2, j::2] for i, j in denoise.PHASES)
    up = lambda p: np.repeat(np.repeat(p, 2, 0), 2, 1)
    assert rgb.shape == (1100, 1500, 3)
    assert np.allclose(rgb, np.stack([up(r), up((g1 + g2) / 2), up(b)], -1), atol=1e-6)  # no seams, gain undone


@pytest.mark.parametrize("pattern", [[[1, 2], [0, 3]], [[1, 0], [2, 3]], [[2, 1], [3, 0]]])  # GBRG, GRBG, BGGR
def test_red_is_moved_to_the_top_left(pattern, small_tiles):
    colour = np.float32([0.2, 0.5, 0.8])
    pattern = np.array(pattern)
    mosaic = np.tile(colour[np.where(pattern == 3, 1, pattern)], (551, 751))[:1101, :1501]  # odd sizes too
    rgb = blended(mosaic, pattern)
    assert rgb.shape == (1101, 1501, 3)
    assert np.allclose(rgb, colour, atol=1e-6)


@pytest.mark.parametrize("flip", [0, 3, 5, 6])
def test_gpu_blends_as_the_cpu_does(gpu, flip, small_tiles):
    """gpu.neural(): the tiles blended, scaled, turned and rounded as camera RGB; what that took
    out of each photosite; the look and base averaged down from it like PIL's BOX resize."""
    rng = np.random.default_rng(3)
    pattern, black, white = np.array([[1, 2], [0, 3]]), 100, 3900
    mosaic = rng.integers(0, 4000, (1101, 1501)).astype(np.uint16)  # odd sizes, 3 x 4 tiles
    calls = iter(range(100))
    run = lambda tile: nearest(tile) * (1 + 0.1 * next(calls))  # each tile different, where they overlap too
    weights = [0.1, 0.2, 0.3, 0.4]
    box, mul, lum_weights = (2.5, 1.0, 57.25, 55.0), np.float32([2.0, 1.0, 1.5]), np.float32([0.3, 0.6, 0.1])
    d = gpu.neural(mosaic, pattern, black, white, flip, lambda blend: neural.denoise(mosaic, pattern, black, white, blend, run=run),
                   weights, look=((17, 11), box), base=(mul, lum_weights, (23, 15)))
    try:
        cam = gpu.read_decoded(d)[0][..., :3].astype(np.float32)
    finally:
        d.release()
    calls = iter(range(100))
    rgb = blended(mosaic, pattern, black, white, run).astype(np.float32)
    assert np.allclose(cam, denoise.orient(rgb, flip), rtol=1e-3, atol=1e-7)  # float16, within its step
    x = (mosaic.astype(np.float32) - black) / np.float32(white - black)
    removed = sum(w * (x[dy : 1100 : 2, dx : 1500 : 2] - rgb[dy : 1100 : 2, dx : 1500 : 2, c])
                  for (dy, dx), c, w in zip(denoise.PHASES, [1, 2, 0, 1], weights))
    assert d.removed.shape == (550, 750) and np.allclose(d.removed, removed, atol=1e-5)
    pil = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(cam[..., c])).resize((17, 11), Image.Resampling.BOX, box=box)) for c in range(3)], -1)
    assert np.allclose(d.look, pil, rtol=1e-5, atol=1e-7)
    lum = np.minimum(cam * mul, mul.min()) @ lum_weights
    assert np.allclose(d.base, np.asarray(Image.fromarray(lum).resize((23, 15), Image.Resampling.BOX)), rtol=1e-5, atol=1e-7)


def test_neural_caps_the_strengths_and_needs_cuda(monkeypatch):
    s = develop.clean({"noise": 150, "chroma": None, "neural": 1})
    assert s["neural"] is True and s["noise"] == 100 and s["chroma"] is None
    assert develop.clean({"noise": 150, "neural": False})["noise"] == 150 and develop.clean({})["neural"] is True
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
    rgb = blended(noisy, np.array([[0, 1], [3, 2]]), run=None)
    inner = rgb[100:-100, 100:-100]
    assert np.allclose(inner.mean((0, 1)), [0.10, 0.20, 0.06], rtol=0.02)
    assert inner[..., 1].std() < 0.2 * noisy[1::2, 0::2].std()


@pytest.mark.skipif(not neural.available() or not (neural.CACHE / "rawnind-bayer-1.0.onnx").exists(), reason="needs CUDA and the model downloaded")
def test_float16_model_is_darktables():
    """_half(): the same result as darktable's model (float32), but for its scale, to float16's precision."""
    import onnxruntime as ort

    ort.preload_dlls()
    original = ort.InferenceSession(str(neural._original()), providers=["CUDAExecutionProvider"])
    rng = np.random.default_rng(3)
    tile = (np.kron(rng.uniform(0, 0.5, (4, 32, 32)), np.ones((1, 16, 16))) + rng.normal(0, 0.02, (4, 512, 512))).astype(np.float32)
    want = original.run(None, {"input": tile[None]})[0][0]
    got = neural._runner()(tile) * 2.0 ** neural.HALVINGS["output_module.0"]
    assert np.abs(got - want).mean() < 1e-3 * np.abs(want).mean() and np.abs(got - want).max() < 2e-2 * np.abs(want).mean()
    assert neural._runner()(tile[:, :336, :208]).shape == (3, 672, 416)  # any size divisible by 16
