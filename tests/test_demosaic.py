"""The GPU's demosaicing against LibRaw's AHD, written out in numpy below (this reference gives
LibRaw's postprocess() output to the bit on D3500 raws, all orientations)."""

import numpy as np
import pytest
from PIL import Image

from photos import denoise

D3500 = np.array([[0.8821, -0.2938, -0.0785], [-0.4178, 1.2142, 0.2287], [-0.0824, 0.1651, 0.6860]], np.float32)  # rgb_xyz_matrix
PATTERN = np.array([[0, 1], [3, 2]])  # R G / G B, as LibRaw numbers them


@pytest.fixture(scope="module")
def gpu():
    from photos import gpu

    try:
        gpu.info()
    except Exception as e:
        pytest.skip(f"no OpenGL: {e}")
    return gpu


def libraw_ahd(mosaic, black, saturation, pattern, xyz_cam, cbrt):
    """LibRaw: black subtracted and scaled to 16 bits, AHD inside a 5-pixel border, the border
    averaged (border_interpolate). Not turned (flip 0)."""
    H, W = mosaic.shape
    scale = np.float32(65535) / np.float32(saturation)
    M = np.clip((np.maximum(mosaic.astype(np.int64) - black, 0).astype(np.float32) * scale).astype(np.int64), 0, 65535)
    pat = np.where(pattern == 3, 1, pattern)
    yy, xx = np.mgrid[:H, :W]
    FC = pat[yy & 1, xx & 1]
    at = lambda a, dy, dx: np.roll(a, (-dy, -dx), (0, 1))  # a[y + dy, x + dx] (wraps: only used inside)
    l, r, u, d = at(M, 0, -1), at(M, 0, 1), at(M, -1, 0), at(M, 1, 0)
    gh = np.where(FC == 1, M, np.clip(((l + M + r) * 2 - at(M, 0, -2) - at(M, 0, 2)) >> 2, np.minimum(l, r), np.maximum(l, r)))
    gv = np.where(FC == 1, M, np.clip(((u + M + d) * 2 - at(M, -2, 0) - at(M, 2, 0)) >> 2, np.minimum(u, d), np.maximum(u, d)))
    below = pat[(yy + 1) & 1, xx & 1]  # a green photosite's vertical neighbours' colour
    rgbs, labs = [], []
    for G in (gh, gv):
        across = np.clip(M + ((at(M, 0, -1) + at(M, 0, 1) - at(G, 0, -1) - at(G, 0, 1)) >> 1), 0, 65535)
        along = np.clip(M + ((at(M, -1, 0) + at(M, 1, 0) - at(G, -1, 0) - at(G, 1, 0)) >> 1), 0, 65535)
        diag = sum(at(M, dy, dx) - at(G, dy, dx) for dy in (-1, 1) for dx in (-1, 1))
        opposite = np.clip(G + ((diag + 1) >> 2), 0, 65535)
        rgb = np.stack([np.where(FC == 1, np.where(below == c, along, np.where(2 - below == c, across, M)), np.where(FC == c, M, G if c == 1 else opposite)) for c in range(3)], -1)
        f = rgb.astype(np.float32)
        xyz = np.full(rgb.shape, np.float32(0.5), np.float32)
        for c in range(3):
            xyz = xyz + xyz_cam[:, c] * f[..., c : c + 1]
        X, Y, Z = (cbrt.ravel()[np.clip(xyz[..., i].astype(np.int64), 0, 65535)] for i in range(3))
        lab = np.stack([(np.float32(116) * Y - np.float32(16)) * np.float32(64), np.float32(32000) * (X - Y), np.float32(12800) * (Y - Z)], -1)
        rgbs.append(rgb)
        labs.append(lab.astype(np.int64))
    near = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    ld = [[np.abs(lab[..., 0] - at(lab[..., 0], *o)) for o in near] for lab in labs]
    ab = [[((lab[..., 1:] - at(lab, *o)[..., 1:]) ** 2).sum(-1) for o in near] for lab in labs]
    leps = np.minimum(np.maximum(ld[0][0], ld[0][1]), np.maximum(ld[1][2], ld[1][3]))
    abeps = np.minimum(np.maximum(ab[0][0], ab[0][1]), np.maximum(ab[1][2], ab[1][3]))
    homo = [sum((ld[k][i] <= leps) & (ab[k][i] <= abeps) for i in range(4)).astype(np.int64) for k in range(2)]
    hm = [sum(at(h, dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) for h in homo]
    out = np.where((hm[0] != hm[1])[..., None], np.where((hm[1] > hm[0])[..., None], rgbs[1], rgbs[0]), (rgbs[0] + rgbs[1]) >> 1)
    for y, x in zip(*np.nonzero((yy < 5) | (yy >= H - 5) | (xx < 5) | (xx >= W - 5))):
        sums, n = np.zeros(3, np.int64), np.zeros(3, np.int64)
        for q in range(max(y - 1, 0), min(y + 2, H)):
            for p in range(max(x - 1, 0), min(x + 2, W)):
                sums[FC[q, p]] += M[q, p]
                n[FC[q, p]] += 1
        out[y, x] = [M[y, x] if c == FC[y, x] else sums[c] // n[c] for c in range(3)]
    return out


def test_demosaic_is_librarys_ahd(gpu):
    rng = np.random.default_rng(3)
    H, W, black, saturation = 48, 64, 150, 4095
    y, x = np.mgrid[:H, :W]
    scene = 600 + 2500 * (np.sin(x / 3.0) * np.cos(y / 5.0) > 0) + 400 * np.sin((x + y) / 7.0)  # edges both ways
    mosaic = np.clip(scene + rng.normal(0, 40, (H, W)) + black, 0, 4095).astype(np.uint16)
    expected = libraw_ahd(mosaic, black - 22, saturation, PATTERN, gpu._libraw_xyz_cam(tuple(D3500.ravel().tolist())), gpu._cbrt_table())
    to_cam = (saturation / 65535 / (saturation - black), 22 / (saturation - black))
    for flip in (0, 3, 5, 6):
        d = gpu.demosaic(mosaic, PATTERN, black - 22, saturation, flip, D3500, to_cam)
        try:
            cam, _, _ = gpu.read_decoded(d)
        finally:
            d.release()
        rgb = expected.astype(np.float32)
        rgb *= np.float32(to_cam[0])
        rgb -= np.float32(to_cam[1])
        oriented = denoise.orient(rgb, flip)
        assert cam.shape[:2] == oriented.shape[:2] == (d.height, d.width)
        assert np.array_equal(cam[..., :3], oriented.astype(np.float16))  # to the bit, rounded as numpy rounds


def test_demosaic_averages_down_like_pil(gpu):
    """What develop averages down on the CPU for the look and the base layer, with PIL's BOX
    resize, from camera RGB in float32 (with uneven steps and a box with fractions)."""
    rng = np.random.default_rng(4)
    mosaic = rng.integers(200, 4000, (60, 90)).astype(np.uint16)
    to_cam, box = (1 / 3945, 0.01), (2.5, 1.0, 57.25, 55.0)
    mul, weights = np.float32([2.0, 1.0, 1.5]), np.float32([0.3, 0.6, 0.1])
    for flip in (0, 6):
        d = gpu.demosaic(mosaic, PATTERN, 100, 4095, flip, D3500, to_cam, look=((17, 11), box), base=(mul, weights, (23, 15)))
        d.release()
        rgb = libraw_ahd(mosaic, 100, 4095, PATTERN, gpu._libraw_xyz_cam(tuple(D3500.ravel().tolist())), gpu._cbrt_table()).astype(np.float32)
        rgb *= np.float32(to_cam[0])
        rgb -= np.float32(to_cam[1])
        rgb = np.ascontiguousarray(denoise.orient(rgb, flip))
        pil = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c])).resize((17, 11), Image.Resampling.BOX, box=box)) for c in range(3)], -1)
        assert np.allclose(d.look, pil, rtol=1e-5, atol=0)  # PIL adds up in doubles, the GPU in floats
        lum = np.minimum(rgb * mul, mul.min()) @ weights
        assert np.allclose(d.base, np.asarray(Image.fromarray(lum).resize((23, 15), Image.Resampling.BOX)), rtol=1e-5, atol=0)


def test_shrinking_on_the_gpu_is_develops_shrink(gpu):
    from photos import develop

    rng = np.random.default_rng(5)
    rgb = rng.uniform(-0.01, 1.0, (70, 103, 3)).astype(np.float16).astype(np.float32)
    d = gpu.upload(rgb)
    try:
        _, _, small = gpu.read_decoded(d, shrink=[(40, 27)])
    finally:
        d.release()
    assert develop._shrunk_size(103, 70, 40) == (40, 27)
    assert np.allclose(small[(40, 27)][0], develop._shrink(rgb, 40), rtol=1e-5, atol=1e-7)
