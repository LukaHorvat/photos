"""Fit a camera's sensor noise to its raws and add it to photos/profiles/<model>.npz.

    uv run python scripts/calibrate_noise.py /mnt/d/Radno [more folders...]

Noise reduction measures how noisy each photo is (denoise.noise_model: variance = A * signal
+ B, raw units), but in a photo full of fine texture, like foliage, the texture reads as noise.
Noise belongs to the sensor and the ISO, not the scene, so this fits what's typical for the
camera over many photos, A = a0 + a1 ISO (from the quieter photos at each ISO: texture only
ever adds) and B = b2 ISO^2 (from the darkest measurements of high-ISO photos, where read noise shows).
A photo that measures more than denoise.CEILING times that is denoised as typical instead.
"""

import argparse
import io
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import rawpy  # noqa: E402

from photos import denoise, look  # noqa: E402
from photos.raw import RAW_EXTS, Photo, read_file  # noqa: E402

QUANTILE = 20  # percent: the quieter photos at an ISO
HIGH_ISO = 1000  # read noise (B) is only measurable above this


def measure(path: Path):
    photo = Photo(path)
    if not photo.iso:
        return None
    with rawpy.imread(io.BytesIO(read_file(path))) as raw:
        if not denoise.supported(raw):
            return None
        xs, vs = denoise.noise_samples(raw.raw_image_visible, min(raw.black_level_per_channel), np.asarray(raw.raw_pattern))
    A, _ = np.polyfit(xs, vs, 1)
    return photo.camera, photo.iso, A, (xs[0], vs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--every", type=int, default=2, help="use every n-th photo")
    args = ap.parse_args()

    files = sorted(p for f in args.folders for p in f.iterdir() if p.suffix.lower() in RAW_EXTS)[:: args.every]
    with ThreadPoolExecutor(6) as ex:
        rows = [r for r in ex.map(measure, files) if r]
    cameras = {r[0] for r in rows}
    if len(cameras) != 1:
        sys.exit(f"Expected photos from one camera, found {sorted(map(str, cameras))}")
    camera = cameras.pop()

    # A: the quieter photos of each ISO band (half stops), then a line through them.
    isos = np.array([r[1] for r in rows], float)
    slopes = np.array([r[2] for r in rows])
    bands = np.round(np.log2(isos) * 2) / 2
    points = [(np.median(isos[bands == b]), np.percentile(slopes[bands == b], QUANTILE), (bands == b).sum()) for b in np.unique(bands)]
    x, y, n = (np.array(v) for v in zip(*points))
    a1, a0 = np.polyfit(x, y, 1, w=np.sqrt(n))
    if a0 < 0:
        a0, a1 = 0.0, float(np.sum(np.sqrt(n) * x * y) / np.sum(np.sqrt(n) * x * x))

    # B: what's left of the darkest measurement after the A part, above HIGH_ISO.
    reads = [(v - (a0 + a1 * iso) * x) / iso**2 for _, iso, _, (x, v) in rows if iso >= HIGH_ISO]
    b2 = max(float(np.median(reads)), 0.0) if reads else 0.0

    print(f"{camera}: {len(rows)} photos, ISO {isos.min():.0f}-{isos.max():.0f}")
    print(f"  variance = ({a0:.4f} + {a1 * 100:.4f} ISO/100) signal + {b2 * 1e4:.4f} (ISO/100)^2")
    for iso, quiet, count in points:
        print(f"  ISO ~{iso:5.0f} ({count:3d} photos): A of the quieter ones {quiet:6.3f}, fitted {a0 + a1 * iso:6.3f}")
    out = look.profile_path(camera)
    profile = replace(look.load_profile(camera), noise=(float(a0), float(a1), b2))
    look.PROFILES.mkdir(exist_ok=True)
    profile.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
