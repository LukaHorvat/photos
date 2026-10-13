import numpy as np

from photos import look


def test_fit_tone_recovers_the_cameras_curve():
    rng = np.random.default_rng(0)
    scene = np.exp2(rng.uniform(-10, 0, (20000, 1))) * rng.uniform(0.3, 1.0, (20000, 3))
    curve = lambda x: np.clip(1.15 * x**0.45 - 0.03, 0, 1)
    camera = curve(scene @ look.LUMA)[:, None] * np.ones(3)
    s = look.Samples(scene, camera * 0.999, np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
    tone = look.fit_tone(s)
    x = 2.0 ** np.linspace(-8, -1, 30)
    assert np.abs(look.tone_at(tone, x) - curve(x)).max() < 0.02


def test_identity_table_keeps_colours():
    scene = np.array([[0.2, 0.1, 0.05], [0.05, 0.1, 0.2], [0.1, 0.1, 0.1]], np.float32)
    tone = np.minimum(2.0**look.GRID, 1) + np.arange(look.GRID.size) * 1e-7  # straight line
    out = look.model(scene, tone, look.identity_table())
    np.testing.assert_allclose(out / out.sum(1, keepdims=True), scene / scene.sum(1, keepdims=True), atol=0.01)


def test_contrast_keeps_black_white_and_order():
    y = np.linspace(0, 1, 101)
    for c in (-100, -40, 40, 100):
        out = look.contrast_curve(y, c)
        assert out[0] == 0 and abs(out[-1] - 1) < 1e-9
        assert np.all(np.diff(out) >= 0)
    assert look.contrast_curve(np.array([0.05]), 100)[0] < 0.05 < look.contrast_curve(np.array([0.05]), -100)[0]


def test_shadow_loss_rises_with_iso():
    p = look.Profile(look.identity_table(), shadow_slope=0.4, shadow_iso=1345)
    assert p.shadow_loss(400) == 0 and 0 < p.shadow_loss(3200) < p.shadow_loss(6400) <= 1
