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


def reference_nlm(z, h, search, patch, values=None):
    """Textbook non-local means (correct away from the borders, where np.roll doesn't wrap): the
    weights from z, averaging `values` (z itself if not given)."""
    values = z if values is None else values
    R, r = search // 2, patch // 2
    num, den = np.zeros_like(z), np.zeros(z.shape[:2] + (1,))
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            other = np.roll(z, (-dy, -dx), (0, 1))  # other[p] = z[p + o]
            sq = ((z - other) ** 2).sum(-1)
            dist = sum(np.roll(sq, (-iy, -ix), (0, 1)) for iy in range(-r, r + 1) for ix in range(-r, r + 1))
            w = np.exp(-np.maximum(dist / ((2 * r + 1) ** 2 * z.shape[-1]) - 2, 0) / h**2)[..., None]
            num += w * np.roll(values, (-dy, -dx), (0, 1))
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
    expected = reference_nlm(z, h, search, patch, values=((z * A / 2) ** 2 - c) / A)  # raw values averaged: see gpu.py
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


def test_raw_nlm_keeps_the_level_in_the_dark(gpu):
    # ISO 25600's noise: the plain inverse transform left these 9 raw units darker
    A, B, gain = 36.0, 4700.0, 16
    rng = np.random.default_rng(3)
    for level in (0, 20, 50, 400):
        mosaic = np.round(BLACK + level + rng.normal(0, np.sqrt(A * level + B), (192, 192))).astype(np.uint16)
        out, _ = gpu.raw_nlm(mosaic, BLACK, A, B, [0.25] * 4, denoise.STRENGTH, denoise.SEARCH, denoise.PATCH, gain=gain)
        assert out.astype(float).mean() / gain - BLACK == pytest.approx(level, abs=1.5)


def test_raw_nlm_evens_out_the_greens_where_flat(gpu):
    A, B, gain = 2.0, 60.0, 16
    signal = np.full((128, 192), 500.0)
    signal[:, 97:] += 2000  # a step between the two columns of tile column 48
    rng = np.random.default_rng(4)
    mosaic = np.round(BLACK + signal + rng.normal(0, 1, signal.shape) * np.sqrt(A * signal + B)).astype(np.uint16)
    args = (BLACK, A, B, [0.25] * 4, denoise.STRENGTH, denoise.SEARCH, denoise.PATCH)
    evened, _ = gpu.raw_nlm(mosaic, *args, greens=(1, 2), gain=gain)  # R G / G B
    plain, _ = gpu.raw_nlm(mosaic, *args, gain=gain)
    diff = lambda out: (out[0::2, 1::2].astype(float) - out[1::2, 0::2]) / gain  # G1 - G2, per tile
    flat = np.s_[4:-4, 4:40]
    assert diff(plain)[flat].std() > 1 and np.abs(diff(evened)[flat]).max() <= 1 / gain
    assert diff(evened)[4:-4, 48].mean() == pytest.approx(2000, abs=50)  # the step: G1 is right of it, G2 left
    assert np.array_equal(diff(evened)[:, 48], diff(plain)[:, 48])


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


def test_texture_doesnt_pass_for_noise():
    typical, level = (0.5, 0.2), 700
    assert denoise.trusted((0.6, 10.0), typical, level) == (0.6, 10.0)  # a bit noisier than typical: as measured
    assert denoise.trusted((4.5, 500.0), typical, level) == typical  # 10x: foliage, not noise (DSC_9331)


def test_automatic_noise_reduction_follows_iso():
    assert denoise.strength(None, 200) == 0 and denoise.strength(None, denoise.AUTO_ISO) == denoise.strength(None, 6400) == 1
    assert denoise.strength(None, None) == 0  # no ISO to judge by
    assert denoise.strength(100, 200) == 1 and denoise.strength(0, 6400) == 0  # the photo's setting wins


def test_noise_setting_is_automatic_unless_set():
    assert develop.clean({})["noise"] is None and develop.clean({"noise": None})["noise"] is None
    assert develop.clean({"noise": 100})["noise"] == 100 and develop.clean({"noise": 250, "neural": False})["noise"] == 200
    assert develop.reduction(develop.clean({"noise": 0, "chroma": None, "neural": False}), 25600) == denoise.Reduction(0, 1)  # luminance off, colour auto
    assert develop.clean({"noise": 60})["chroma"] == 60  # saved when one setting did both
    assert develop.clean({"noise": 60, "chroma": None})["chroma"] is None


def test_strength_and_colour_smoothing():
    assert denoise.strength(60, 25600) == 0.6 and denoise.strength(150, 200) == 1.5 and denoise.strength(0, 6400) == 0
    assert denoise.chroma_eps(0.5) == denoise.chroma_eps(1) == denoise.CHROMA_EPS and denoise.chroma_eps(2) == 4 * denoise.CHROMA_EPS


def test_strength_sets_how_much_grain_is_given_back():
    for iso in (1000, 6400, 25600):
        standard = denoise.grain_weights(iso)
        assert denoise.grain_weights(iso, 1) == standard
        assert denoise.grain_weights(iso, 0.5) == pytest.approx(tuple(0.75 * w for w in standard))  # a quarter is without
        assert denoise.grain_weights(iso, 1.5) == pytest.approx(tuple(w / 2 for w in standard))  # stronger: less
        assert denoise.grain_weights(iso, 0) == denoise.grain_weights(iso, 2) == (0, 0, 0)
    assert denoise.amount(0) == 0 and denoise.amount(0.5) == 0.75 and denoise.amount(1) == denoise.amount(1.5) == 1


def test_luminance_and_colour_strengths_act_apart():
    color = develop.Color(np.eye(3), np.eye(3), np.ones(3), np.ones(3), 6500.0)
    noise = denoise.Noise.at(None, color, 25600, denoise.Reduction(0.5, 1.5))
    assert noise.weights == denoise.grain_weights(25600, 0.5) and noise.chroma_eps == denoise.chroma_eps(1.5)
    assert noise.amount == (0.75, 1) and denoise.mixed(denoise.Reduction(0.5, 1.5)) and not denoise.mixed(denoise.Reduction(1, 2))


def test_weaker_noise_reduction_mixes_in_the_photo_without(gpu):
    A, B, gain, saturation = 2.0, 60.0, 16, 4095
    mosaic, _ = noisy_mosaic((48, 64), A, B)
    xyz = np.array([[0.8821, -0.2938, -0.0785], [-0.4178, 1.2142, 0.2287], [-0.0824, 0.1651, 0.6860]], np.float32)  # the D3500's
    to_cam = (saturation / 65535 / (saturation - BLACK), 22 / (saturation - BLACK))
    nlm = (BLACK, A, B, [0.25] * 4, denoise.STRENGTH, denoise.SEARCH, denoise.PATCH, (1, 2), gain)
    plain = gpu.demosaic(mosaic, PATTERN, BLACK - 22, saturation, 0, xyz, to_cam)
    d = gpu.demosaic(mosaic, PATTERN, BLACK - 22, saturation, 0, xyz, to_cam, nlm=nlm, original=True)
    try:
        demosaiced, _, _ = gpu.read_decoded(plain)
        noise = lambda amount: denoise.Noise(d.removed, np.ones(3), np.eye(3), (0.0, 0.0, 0.0), amount=amount)
        amounts = [(0, 0), (0.25, 0.25), (1, 1), (1, 0), (0, 1)]
        cam = {a: gpu.read_decoded(d, noise(a))[0][..., :3].astype(np.float32) for a in amounts}
    finally:
        plain.release()
        d.release()
    without, reduced = cam[0, 0], cam[1, 1]
    assert np.array_equal(without, demosaiced[..., :3])  # the original is demosaiced as it is, to the bit
    assert np.abs(reduced - without).std() > 0.002  # noise reduced
    assert np.abs(cam[0.25, 0.25] - (0.75 * without + 0.25 * reduced)).max() < 1e-3
    # Luminance and colour apart (camera RGB is scene RGB here: grey is (1, 1, 1)), to float16's
    # rounding (~0.001 at 1, on both sides)
    y = lambda img: img @ look.LUMA
    colour = lambda img: img - y(img)[..., None]
    assert np.abs(y(cam[1, 0]) - y(reduced)).max() < 2e-3 and np.abs(colour(cam[1, 0]) - colour(without)).max() < 2e-3
    assert np.abs(y(cam[0, 1]) - y(without)).max() < 2e-3 and np.abs(colour(cam[0, 1]) - colour(reduced)).max() < 2e-3


def test_camera_noise_profile_round_trip(tmp_path):
    profile = look.Profile(look.identity_table(), noise=(0.06, 0.0013, 8e-6))
    profile.save(tmp_path / "cam.npz")
    with np.load(tmp_path / "cam.npz") as f:
        assert tuple(f["noise"]) == profile.noise
    A, B = profile.noise_at(400)
    assert A == pytest.approx(0.06 + 0.0013 * 400) and B == pytest.approx(8e-6 * 400**2)
    assert look.Profile(look.identity_table()).noise_at(400) is None
