import numpy as np

from photos import develop


def test_full_tiles_cover_the_photo():
    h, w, T = 1100, 1300, develop.FULL_TILE
    rng = np.random.default_rng(0)
    cam, grain = rng.random((h, w, 3)).astype(np.float16), rng.random((h, w)).astype(np.float16)
    full = develop.Full(cam, grain, None, None)
    rebuilt_cam, rebuilt_grain = np.zeros_like(cam), np.zeros_like(grain)
    for j in range(-(-h // T)):
        for i in range(-(-w // T)):
            th, tw = min(T, h - j * T), min(T, w - i * T)  # smaller at the right and bottom edges
            data = np.frombuffer(full.tile(i, j), np.float16)
            assert data.size == th * tw * 4
            rebuilt_cam[j * T : j * T + th, i * T : i * T + tw] = data[: th * tw * 3].reshape(th, tw, 3)
            rebuilt_grain[j * T : j * T + th, i * T : i * T + tw] = data[th * tw * 3 :].reshape(th, tw)
    assert np.array_equal(rebuilt_cam, cam) and np.array_equal(rebuilt_grain, grain)


def test_tint_moves_red_and_blue_against_green_and_keeps_brightness():
    rng = np.random.default_rng(1)
    cam_rgb = np.eye(3) + rng.uniform(-0.15, 0.15, (3, 3))
    cam_rgb /= cam_rgb.sum(axis=1, keepdims=True)  # white stays white, as in Color.from_raw
    color = develop.Color(cam_rgb @ np.linalg.inv(develop.SRGB_TO_XYZ), np.linalg.inv(cam_rgb), np.array([2.0, 1.0, 1.5]), np.array([1.02, 1.0, 0.99]), 5200.0)
    for temp in (None, 3500):
        base = color.multipliers(temp)
        assert np.allclose(base, color.as_shot if temp is None else develop.temp_multipliers(color.cam_xyz, temp) * color.off_locus)
        for tint in (-100, -30, 45, 100):
            mul = color.multipliers(temp, tint)
            f = 2 ** (tint / 100 * develop.TINT_STOPS)
            assert np.allclose(mul[[0, 2]] / mul[1], base[[0, 2]] / base[1] * f)  # magenta: red and blue up
            neutral = color.rgb_cam @ (mul / base)  # what a grey under that light becomes
            assert np.isclose(neutral @ develop.LUMA, 1)  # as bright as without tint
