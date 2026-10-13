import numpy as np

from photos import develop


def test_full_tiles_cover_the_photo():
    h, w, T = 1100, 1300, develop.FULL_TILE
    rng = np.random.default_rng(0)
    cam, grain = rng.random((h, w, 3)).astype(np.float16), rng.random((h, w)).astype(np.float16)
    full = develop.Full(cam, grain)
    rebuilt_cam, rebuilt_grain = np.zeros_like(cam), np.zeros_like(grain)
    for j in range(-(-h // T)):
        for i in range(-(-w // T)):
            th, tw = min(T, h - j * T), min(T, w - i * T)  # smaller at the right and bottom edges
            data = np.frombuffer(full.tile(i, j), np.float16)
            assert data.size == th * tw * 4
            rebuilt_cam[j * T : j * T + th, i * T : i * T + tw] = data[: th * tw * 3].reshape(th, tw, 3)
            rebuilt_grain[j * T : j * T + th, i * T : i * T + tw] = data[th * tw * 3 :].reshape(th, tw)
    assert np.array_equal(rebuilt_cam, cam) and np.array_equal(rebuilt_grain, grain)
