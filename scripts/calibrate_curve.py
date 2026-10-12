"""Fit develop.py's base tone curve to the camera's own JPEGs.

For a sample of raws, compare luminance quantiles of the linear, as-shot white-balanced
render with those of the embedded JPEG (histogram matching, so no pixel alignment is
needed), and least-squares fit the curve's gain, contrast and shoulder.

    uv run python scripts/calibrate_curve.py /mnt/d/Radno
"""

import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from photos import develop
from photos.raw import Photo

Q = np.linspace(0.03, 0.97, 48)


def oetf(x):
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0.0031308), 1 / 2.4) - 0.055)


def samples(path: Path):
    p = Photo(path)
    rgb, color = develop.decode(path.read_bytes(), half_size=True)
    lin = np.minimum(rgb * color.as_shot.astype(np.float32), color.as_shot.min()) @ color.rgb_cam.T.astype(np.float32)
    x = np.quantile(np.maximum(lin @ develop.LUMA, 0)[::4, ::4], Q)
    jpg = np.asarray(Image.open(io.BytesIO(b"".join(p.jpeg(1000)))).convert("RGB"), np.float32) / 255
    jl = np.where(jpg <= 0.04045, jpg / 12.92, ((jpg + 0.055) / 1.055) ** 2.4) @ develop.LUMA
    y = np.quantile(jl, Q)
    return x, y


def main(folder: str):
    files = sorted(Path(folder).glob("*.NEF"))[::13]
    with ThreadPoolExecutor(8) as ex:
        pairs = list(ex.map(samples, files))
    x = np.concatenate([p[0] for p in pairs])
    y = oetf(np.concatenate([p[1] for p in pairs]))
    keep = (y > 0.02) & (y < 0.98)  # clipped tails carry no shape information
    x, y = x[keep], y[keep]

    def err(g, c, k):
        # The tone curve of web/develop.frag.
        t = (x * g) ** c
        return np.mean((oetf(np.minimum((1 + k) * t / (t + k), 1)) - y) ** 2)

    best = min(
        (err(g, c, k), g, c, k)
        for g in np.geomspace(0.5, 16, 60)
        for c in np.linspace(0.6, 2.2, 33)
        for k in np.geomspace(0.02, 4, 30)
    )
    print(f"{len(files)} photos, {len(x)} quantile pairs")
    print(f"CURVE_GAIN = {best[1]:.3f}\nCURVE_C = {best[2]:.3f}\nCURVE_K = {best[3]:.3f}\nrms error {np.sqrt(best[0]) * 255:.1f} / 255")


if __name__ == "__main__":
    main(sys.argv[1])
