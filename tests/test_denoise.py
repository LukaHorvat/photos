import numpy as np
import pytest

from photos import denoise, develop, look

PATTERN = np.array([[0, 1], [3, 2]])  # R G / G B, as LibRaw numbers them
BLACK = 600


@pytest.fixture(scope="module")
def gpu():
    from photos import gpu

    try:
        gpu.info()
    except Exception as e:
        pytest.skip(f"no OpenGL: {e}")
    return gpu


def noisy_mosaic(shape, A, B, seed=0):
    """A smooth scene, the same light on each 2x2 tile, with noise of variance A * signal + B
    (raw units, above black)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]] // 2 * 2
    signal = 200 + 3000 * (x / shape[1]) + 800 * np.sin(y / 20.0) ** 2
    noisy = signal + rng.normal(0, 1, shape) * np.sqrt(A * signal + B)
    return np.clip(np.round(noisy + BLACK), 0, 65535).astype(np.uint16), signal


def reference_nlm(z, h, search, patch):
    """Textbook non-local means (correct away from the borders, where np.roll doesn't wrap)."""
    R, r = search // 2, patch // 2
    num, den = np.zeros_like(z), np.zeros(z.shape[:2] + (1,))
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            other = np.roll(z, (-dy, -dx), (0, 1))  # other[p] = z[p + o]
            sq = ((z - other) ** 2).sum(-1)
            dist = sum(np.roll(sq, (-iy, -ix), (0, 1)) for iy in range(-r, r + 1) for ix in range(-r, r + 1))
            w = np.exp(-np.maximum(dist / ((2 * r + 1) ** 2 * z.shape[-1]) - 2, 0) / h**2)[..., None]
            num += w * other
            den += w
    return num / den


def test_noise_model_recovers_the_noise():
    A, B = 2.0, 60.0
    mosaic, _ = noisy_mosaic((800, 1200), A, B)
    a, b = denoise.noise_model(mosaic, BLACK, PATTERN)
    for signal in (300, 1000, 3000):  # what the stabilising transform needs: the variance at each level
        assert a * signal + b == pytest.approx(A * signal + B, rel=0.08)  # noise sd within 4%


def test_raw_nlm_is_non_local_means_in_the_stabilised_domain(gpu):
    A, B, h, search, patch = 2.0, 60.0, 0.7, 5, 3
    mosaic, signal = noisy_mosaic((64, 96), A, B)
    weights = [0.1, 0.2, 0.3, 0.4]
    out, removed = gpu.raw_nlm(mosaic, BLACK, A, B, weights, h, search, patch)
    assert out.shape == mosaic.shape and out.dtype == np.uint16 and removed.shape == (32, 48)

    c = 3 / 8 * A * A + B
    planes = np.stack([mosaic[dy::2, dx::2].astype(np.float64) - BLACK for dy, dx in denoise.PHASES], -1)
    z = 2 / A * np.sqrt(A * planes + c)
    expected = ((reference_nlm(z, h, search, patch) * A / 2) ** 2 - c) / A
    got = np.stack([out[dy::2, dx::2].astype(np.float64) - BLACK for dy, dx in denoise.PHASES], -1)
    inner = np.s_[search // 2 + patch // 2 : -(search // 2 + patch // 2)]
    assert np.abs(got[inner, inner] - expected[inner, inner]).max() <= 1.5  # rounding to raw units, half-float distances
    truth = np.stack([signal[dy::2, dx::2] for dy, dx in denoise.PHASES], -1)
    assert (got - truth)[inner, inner].std() < 0.7 * (planes - truth)[inner, inner].std()  # it did denoise
    # What was removed, weighted per photosite (from the unrounded result)
    assert np.abs(removed - (planes - got) @ weights).max() <= 0.5 * sum(weights) + 1e-3
    # With a gain, the result keeps fractions of a raw unit
    gain = 65535 // int(mosaic.max())
    out_fine, _ = gpu.raw_nlm(mosaic, BLACK, A, B, weights, h, search, patch, gain=gain)
    fine = np.stack([out_fine[dy::2, dx::2].astype(np.float64) / gain - BLACK for dy, dx in denoise.PHASES], -1)
    assert np.abs(fine - got).max() <= 0.5 + 0.5 / gain
    assert np.abs(fine[inner, inner] - expected[inner, inner]).max() < np.abs(got[inner, inner] - expected[inner, inner]).max()


def test_grain_follows_iso():
    assert denoise.grain_weights(denoise.GRAIN_ISO) == pytest.approx(denoise.GRAIN)
    low, high = denoise.grain_weights(100), denoise.grain_weights(6400)
    assert all(lo > hi for lo, hi in zip(low, high))  # the camera keeps more texture at low ISO
    assert denoise.grain_weights(None) == pytest.approx(denoise.GRAIN)


def test_render_with_noise_reduction(gpu):
    plain = develop.Color(np.eye(3), np.eye(3), np.ones(3), np.ones(3), 6500.0)
    lk = look.Look(look.fallback_tone().astype(np.float32), look.identity_table(), 0.0, -2.5)
    rng = np.random.default_rng(2)
    rgb = (0.2 + 0.02 * rng.normal(size=(64, 80, 3))).astype(np.float32)  # grey with colour noise
    args = (rgb, develop.base_coefficients(rgb, plain), develop.uniforms(plain, lk, {}), *develop.look_textures(lk))
    grain = rng.normal(0, 0.01, (32, 40)).astype(np.float32)
    noise = denoise.Noise(grain, np.ones(3), np.eye(3), (0.0, 0.0, 0.0))
    before, after = gpu.render(*args).astype(int), gpu.render(*args, noise=noise).astype(int)
    chroma = lambda img: (img - img.mean(-1, keepdims=True)).std()
    assert chroma(after) < chroma(before) / 2  # colour noise smoothed
    assert np.abs(after.mean((0, 1)) - before.mean((0, 1))).max() < 1  # colour kept
    with_grain = denoise.Noise(grain, np.ones(3), np.eye(3), (1.0, 1.0, 1.0))
    first, again = gpu.render(*args, noise=with_grain), gpu.render(*args, noise=with_grain)
    assert np.array_equal(first, again)  # pooled textures are reused cleanly
    assert np.abs(first.astype(int) - after).mean() > 1  # grain given back
