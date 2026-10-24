"""RawNIND (Brummer & De Vleeschouwer, "Learned joint denoising and demosaicing of raw images",
2025): a neural network that demosaics and denoises a Bayer raw in one go. A photo can use it
("neural") in place of non-local means and AHD (denoise.py, gpu.py): its edges come out clean
where AHD on noisy data draws mazes and combs, and its colour without blotches. Tried against
eight other open models on DSC_0285 (ISO 25600), it came out best.

darktable ships it ("neural restore"); this runs the very same ONNX file with ONNX Runtime on an
NVIDIA GPU (CUDA). It is downloaded on first use and checked against darktable's published hash;
the model is GPL-3.0, so it lives in the user's cache rather than in this repo.

It takes no strength: the noise reduction strengths (denoise.Reduction) only mix in the photo
without it, and set how much grain is given back, as for non-local means up to 100%.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import numpy as np

URL = "https://github.com/darktable-org/darktable-ai/releases/download/release-5.6.0/rawdenoise-nind.dtmodel"
SHA256 = "d71b5f1e727c85a359e6f74dca9e2016c9d8fc3e2f7ac3e9b347d80ceca969af"  # darktable-ai's versions.json
MEMBER = "rawdenoise-nind/model_bayer.onnx"
CACHE = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "photos"
TILE = 512  # the model's input: 4 x 512 x 512, the RGGB planes (half size) -> 3 x 1024 x 1024 camera RGB
OVERLAP = 32  # half-size pixels each tile shares with its neighbours on each side, as darktable's
_lock = threading.Lock()  # one photo at a time: more would only share the GPU, in more memory


@cache
def available() -> bool:
    """Whether RawNIND can run here: ONNX Runtime with CUDA (the model comes on first use)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def denoise(mosaic: np.ndarray, pattern: np.ndarray, run=None) -> np.ndarray:
    """Camera RGB (H x W x 3 float32, not white balanced; a view) from a Bayer mosaic (H x W float32,
    black 0, white 1, noise below black kept), fed to the model as darktable does: its RGGB
    planes in tiles, mirrored at the photo's edges, overlapping and blended with linear ramps.
    The model's output has an arbitrary gain, so each channel is scaled to the mosaic's mean of
    that colour. pattern: LibRaw's 2x2 colours (0 R, 1 G, 2 B, 3 second G); run: one tile ->
    its output, for tests (default: the model)."""
    with _lock:
        return _denoise(mosaic, pattern, run or _runner())


def _runner():
    session = _session()
    name = session.get_inputs()[0].name
    return lambda tile: session.run(None, {name: tile[None]})[0][0]


def _denoise(mosaic, pattern, run):
    # The model wants red at the top left: mirroring a row or column in front moves it there.
    (dy,), (dx,) = np.nonzero(np.asarray(pattern) == 0)
    if np.asarray(pattern)[1 - dy, 1 - dx] != 2:
        raise ValueError(f"not a Bayer pattern: {pattern.tolist()}")
    h, w = mosaic.shape
    step = TILE - 2 * OVERLAP
    ny, nx = -(-((h + dy + 1) // 2) // step), -(-((w + dx + 1) // 2) // step)  # tiles to cover the half-size planes
    top, left = 2 * OVERLAP + dy, 2 * OVERLAP + dx  # sensor pixels in front of the photo
    bottom, right = 2 * ((ny - 1) * step + TILE) - top - h, 2 * ((nx - 1) * step + TILE) - left - w
    padded = np.pad(mosaic.astype(np.float32, copy=False), ((top, bottom), (left, right)), mode="reflect")
    planes = np.stack([padded[0::2, 0::2], padded[0::2, 1::2], padded[1::2, 0::2], padded[1::2, 1::2]])
    # Each tile's weights: a linear cross-fade over the 128 sensor px two tiles share, divided by
    # what all the tiles' add up to (1, but at the photo's edges), along y and x
    ramp = np.minimum(np.arange(2 * TILE) + 0.5, 2 * TILE - np.arange(2 * TILE) - 0.5) / (4 * OVERLAP)
    ramp = np.minimum(ramp, 1).astype(np.float32)
    tiles = [(ty * step, tx * step) for ty in range(ny) for tx in range(nx)]
    totals = [np.zeros(n, np.float32) for n in padded.shape]
    for t, total in zip(zip(*tiles), totals):
        for start in set(t):
            total[2 * start : 2 * (start + TILE)] += ramp
    acc = np.zeros(padded.shape + (3,), np.float32)
    # The model runs on a thread of its own (ONNX Runtime lets go of the GIL), the next tile
    # while this one is blended in
    with ThreadPoolExecutor(1) as pool:
        outs = pool.map(lambda t: run(np.ascontiguousarray(planes[:, t[0] : t[0] + TILE, t[1] : t[1] + TILE])), tiles)
        for (y, x), out in zip(tiles, outs):
            area = np.s_[2 * y : 2 * (y + TILE), 2 * x : 2 * (x + TILE)]
            weight = (ramp / totals[0][area[0]])[:, None, None] * (ramp / totals[1][area[1]])[None, :, None]
            acc[area] += out.transpose(1, 2, 0) * weight
    rgb = acc[top : top + h, left : left + w]
    # Each channel to the mosaic's mean of its colour
    colours = np.where(np.asarray(pattern) == 3, 1, pattern)
    want = [np.mean([mosaic[i::2, j::2].mean(dtype=np.float64) for i in range(2) for j in range(2) if colours[i, j] == c]) for c in range(3)]
    rgb *= np.float32(want / rgb.mean((0, 1), dtype=np.float64))
    return rgb


@cache
def _session():
    import onnxruntime as ort

    ort.preload_dlls()  # CUDA and cuDNN, from their pip packages
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(_model()), options, providers=["CUDAExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":  # ONNX Runtime falls back to the CPU: far too slow
        raise RuntimeError("RawNIND needs CUDA, which ONNX Runtime couldn't use")
    return session


def _model() -> Path:
    """The ONNX file, downloaded (and its archive checked) unless it's cached."""
    path = CACHE / "rawnind-bayer-1.0.onnx"
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as r:
            archive = r.read()
        if hashlib.sha256(archive).hexdigest() != SHA256:
            raise RuntimeError(f"{URL} isn't the file expected (its hash differs)")
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            model = z.read(MEMBER)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(model)
        os.replace(tmp, path)
    return path
