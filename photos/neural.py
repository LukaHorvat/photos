"""RawNIND (Brummer & De Vleeschouwer, "Learned joint denoising and demosaicing of raw images",
2025): a neural network that demosaics and denoises a Bayer raw in one go. A photo can use it
("neural") in place of non-local means and AHD (denoise.py, gpu.py): its edges come out clean
where AHD on noisy data draws mazes and combs, and its colour without blotches. Tried against
eight other open models on DSC_0285 (ISO 25600), it came out best.

darktable ships it ("neural restore"); this runs the same ONNX file with ONNX Runtime on an
NVIDIA GPU (CUDA), converted to float16 (_half()): twice as fast, the result the same to ~0.1%.
It is downloaded on first use and checked against darktable's published hash; the model is
GPL-3.0, so it lives in the user's cache rather than in this repo.

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
# The model takes the RGGB planes (half size, any size divisible by 16) and gives camera RGB at
# full size. darktable runs it on tiles of 512; larger ones waste less on overlaps, and the GPU
# has the memory: a 24 Mpx photo takes 2 x 3 tiles of 1072.
MAX_TILE = 1088
OVERLAP = 32  # half-size pixels each tile shares with its neighbours on each side, as darktable's
# Each convolution's output divided by 2 ** this, and what follows it accordingly (_half()), for
# float16, which holds up to 65504: the largest activations, on test tiles (DSC_0285 at ISO 25600,
# the brightest parts of ISO 200 photos, white, checkerboards, noise), reach 1.4e6 at the output;
# scaled, they stay under 4096. The model's output gain is arbitrary anyway (denoise()).
HALVINGS = {"convs4.0": 1, "convs4.2": 2, "bottom.0": 2, "bottom.2": 3, "tconvs1.0": 3, "tconvs1.2": 3, "tconvs2.0": 2, "tconvs2.2": 3,
            "up3": 1, "tconvs3.0": 2, "tconvs3.2": 4, "up4": 4, "tconvs4.0": 6, "tconvs4.2": 9, "output_module.0": 9}
_lock = threading.Lock()  # one photo at a time: more would only share the GPU, in more memory


@cache
def available() -> bool:
    """Whether RawNIND can run here: ONNX Runtime with CUDA, and onnx to convert the model (which
    comes on first use)."""
    try:
        import onnx  # noqa: F401
        import onnxruntime as ort
    except ImportError:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def denoise(mosaic: np.ndarray, pattern: np.ndarray, black: float, white: float, blend, run=None) -> np.ndarray:
    """RawNIND over a Bayer mosaic (H x W raw values), as darktable runs it: its RGGB planes,
    black 0 and white 1 (noise below black kept), in tiles mirrored at the photo's edges,
    overlapping and blended with linear ramps. The blending is the caller's (gpu.neural() does
    it on the GPU): blend(y, x, out, wy, wx) for each tile, whose output `out` (camera RGB, not
    white balanced, 3 x 2h x 2w float32 for a tile of h x w) belongs at row y, column x of the photo
    (negative: partly in front of it), times wy along its rows and wx along its columns (they
    add up to 1 at every pixel, over the tiles).

    Returns what the blended sum is to be multiplied by (3 float32): the model's output has an
    arbitrary gain, so each channel is scaled to the mosaic's mean of that colour.
    pattern: LibRaw's 2x2 colours (0 R, 1 G, 2 B, 3 second G); run: one tile -> its output,
    for tests (default: the model)."""
    with _lock:
        return _denoise(mosaic, pattern, black, white, blend, run or _runner())


def _runner():
    session = _session()
    name = session.get_inputs()[0].name
    return lambda tile: session.run(None, {name: tile[None]})[0][0]


def _denoise(mosaic, pattern, black, white, blend, run):
    # The model wants red at the top left: mirroring a row or column in front moves it there.
    (dy,), (dx,) = np.nonzero(np.asarray(pattern) == 0)
    if np.asarray(pattern)[1 - dy, 1 - dx] != 2:
        raise ValueError(f"not a Bayer pattern: {pattern.tolist()}")
    h, w = mosaic.shape
    (th, sy, ny), (tw, sx, nx) = _tiling((h + dy + 1) // 2), _tiling((w + dx + 1) // 2)
    top, left = 2 * OVERLAP + dy, 2 * OVERLAP + dx  # sensor pixels in front of the photo
    bottom, right = 2 * ((ny - 1) * sy + th) - top - h, 2 * ((nx - 1) * sx + tw) - left - w
    padded = np.pad(mosaic, ((top, bottom), (left, right)), mode="reflect")
    # Each tile's weights: a linear cross-fade over the 128 sensor px two tiles share, divided by
    # what all the tiles' add up to (1, but at the photo's edges), along y and x
    ramps = [np.minimum(np.minimum(np.arange(2 * t) + 0.5, 2 * t - np.arange(2 * t) - 0.5) / (4 * OVERLAP), 1).astype(np.float32) for t in (th, tw)]
    starts = [range(0, 2 * ny * sy, 2 * sy), range(0, 2 * nx * sx, 2 * sx)]  # of the tiles in `padded`, along y and x
    totals = [np.zeros(n, np.float32) for n in padded.shape]
    for total, ramp, axis in zip(totals, ramps, starts):
        for start in axis:
            total[start : start + len(ramp)] += ramp
    tiles = [(y, x) for y in starts[0] for x in starts[1]]

    def planes(t):  # a tile's input: the RGGB planes, black 0, white 1
        area = padded[t[0] : t[0] + 2 * th, t[1] : t[1] + 2 * tw]
        tile = np.stack([area[0::2, 0::2], area[0::2, 1::2], area[1::2, 0::2], area[1::2, 1::2]]).astype(np.float32)
        return (tile - np.float32(black)) / np.float32(white - black)

    sums = np.zeros(3)  # of the blended output over the photo
    # The model runs on a thread of its own (ONNX Runtime lets go of the GIL), the next tile
    # while this one is blended in
    with ThreadPoolExecutor(1) as pool:
        for (y, x), out in zip(tiles, pool.map(lambda t: run(planes(t)), tiles)):
            wy, wx = ramps[0] / totals[0][y : y + 2 * th], ramps[1] / totals[1][x : x + 2 * tw]
            y, x = y - top, x - left
            blend(y, x, out, wy, wx)
            rows, cols = slice(max(-y, 0), min(h - y, 2 * th)), slice(max(-x, 0), min(w - x, 2 * tw))  # within the photo
            sums += (out[:, rows, cols] @ wx[cols]) @ wy[rows]
    # Each channel to the mosaic's mean of its colour
    colours = np.where(np.asarray(pattern) == 3, 1, pattern)
    want = [np.mean([mosaic[i::2, j::2].mean(dtype=np.float64) for i in range(2) for j in range(2) if colours[i, j] == c]) for c in range(3)]
    return np.float32((np.float64(want) - black) / (white - black) / (sums / (h * w)))


def _tiling(n: int) -> tuple[int, int, int]:
    """Tiles to cover n half-size pixels along one side, with OVERLAP more on each side: their
    size (at most MAX_TILE, divisible by 16), step and number, as few as can."""
    count = -(-n // (MAX_TILE - 2 * OVERLAP))
    step = 16 * -(-n // (16 * count))
    return step + 2 * OVERLAP, step, count


@cache
def _session():
    import onnxruntime as ort

    ort.preload_dlls()  # CUDA and cuDNN, from their pip packages
    options = ort.SessionOptions()
    options.log_severity_level = 3
    providers = [("CUDAExecutionProvider", {"prefer_nhwc": "1"})]  # cuDNN's float16 convolutions are faster on channels last
    session = ort.InferenceSession(str(_model()), options, providers=providers)
    if session.get_providers()[0] != "CUDAExecutionProvider":  # ONNX Runtime falls back to the CPU: far too slow
        raise RuntimeError("RawNIND needs CUDA, which ONNX Runtime couldn't use")
    return session


def _model() -> Path:
    """The model to run (_half()), converted unless it's cached."""
    path = CACHE / "rawnind-bayer-1.0-fp16.onnx"
    if not path.exists():
        _save(path, _half(_original()))
    return path


def _original() -> Path:
    """darktable's ONNX file, downloaded (and its archive checked) unless it's cached."""
    path = CACHE / "rawnind-bayer-1.0.onnx"
    if not path.exists():
        with urllib.request.urlopen(URL, timeout=60) as r:
            archive = r.read()
        if hashlib.sha256(archive).hexdigest() != SHA256:
            raise RuntimeError(f"{URL} isn't the file expected (its hash differs)")
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            _save(path, z.read(MEMBER))
    return path


def _save(path: Path, data: bytes):
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _half(original: Path) -> bytes:
    """The model in float16, for any tile size, float32 in and out: its activations would
    overflow float16, so each convolution's output is divided by 2 ** HALVINGS[its name]. That
    changes nothing else: the convolutions are linear, and LeakyReLU and max pooling commute with
    a positive scale, so the next convolutions' weights take the scale out again (per input
    channel: a concatenation joins differently scaled ones), and their biases put it in."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(original))
    graph = model.graph
    params = {p.name: numpy_helper.to_array(p).astype(np.float64) for p in graph.initializer}
    scales = {"input": np.ones(4)}  # of each tensor against the original's, per channel
    for node in graph.node:
        if node.op_type in ("Conv", "ConvTranspose"):
            weight, bias = node.input[1:3]
            scale = 2.0 ** -HALVINGS.get(weight.rsplit(".", 1)[0], 0)
            ratio = scale / scales[node.input[0]]  # per input channel: weights out x in x ... (transposed: in x out x ...)
            params[weight] *= ratio[:, None, None] if node.op_type == "Conv" else ratio[:, None, None, None]
            params[bias] *= scale
            scales[node.output[0]] = np.full(len(params[bias]), scale)
        elif node.op_type == "Concat":
            scales[node.output[0]] = np.concatenate([scales[i] for i in node.input])
        elif node.op_type in ("LeakyRelu", "MaxPool"):
            scales[node.output[0]] = scales[node.input[0]]
        elif node.op_type != "DepthToSpace" or node.output[0] != "output":
            raise ValueError(f"RawNIND: {node.op_type} unexpected")
    for p in graph.initializer:
        p.CopyFrom(numpy_helper.from_array(params[p.name].astype(np.float16), p.name))
    for value, sizes in ((graph.input[0], ("h", "w")), (graph.output[0], ("2h", "2w"))):
        for dim, size in zip(value.type.tensor_type.shape.dim[2:], sizes):
            dim.dim_param = size
    del graph.value_info[:]  # shapes for 512 x 512
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == "input":
                node.input[i] = "input16"
        for i, name in enumerate(node.output):
            if name == "output":
                node.output[i] = "output16"
    graph.node.insert(0, helper.make_node("Cast", ["input"], ["input16"], to=TensorProto.FLOAT16))
    graph.node.append(helper.make_node("Cast", ["output16"], ["output"], to=TensorProto.FLOAT))
    return model.SerializeToString()
