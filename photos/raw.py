"""Fast access to the camera-rendered JPEGs embedded in raw files.

NEF (and CR2, ARW, DNG, PEF, ...) are TIFF containers. Next to the sensor data
the camera stores one or more ready-made JPEGs - on a Nikon D3500 a full-size
6000x4000 one and a 1620x1080 one, plus a tiny RGB thumbnail. Reading just
those bytes is ~100x faster than decoding the raw, and over a slow mount (WSL
reading /mnt/d) it means a few hundred KB per photo instead of the whole file.

Raws without a usable embedded JPEG fall back to a half-size rawpy render.
"""

from __future__ import annotations

import io
import struct
import threading
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

RAW_EXTS = {".nef", ".nrw", ".cr2", ".arw", ".dng", ".pef", ".orf", ".rw2", ".raf", ".srw"}
JPEG_EXTS = {".jpg", ".jpeg"}
IMAGE_EXTS = RAW_EXTS | JPEG_EXTS

THUMB_SIZE = 320

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
_TYPE_FMT = {1: "B", 3: "H", 4: "I", 6: "b", 8: "h", 9: "i", 13: "I"}

# PIL transposes that undo an EXIF orientation value.
_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


@dataclass
class Embedded:
    offset: int
    length: int
    width: int
    height: int


@dataclass
class RawInfo:
    jpegs: list[Embedded] = field(default_factory=list)  # smallest first
    orientation: int = 1
    rgb_thumb: tuple[int, int, int] | None = None  # (offset, width, height) of 8-bit RGB strip
    exif: dict = field(default_factory=dict)  # display summary
    tags: dict = field(default_factory=dict)  # {"ifd0": {tag: value}, "exif": {tag: value}}, for exports


# EXIF copied into exported JPEGs.
IFD0_TAGS = (0x010F, 0x0110, 0x0132, 0x013B, 0x8298)  # make, model, date, artist, copyright
EXIF_TAGS = (
    0x829A, 0x829D, 0x8822, 0x8827, 0x9003, 0x9004, 0x9010, 0x9011, 0x9204,
    0x9207, 0x9209, 0x920A, 0xA402, 0xA403, 0xA405, 0xA432, 0xA433, 0xA434,
)  # exposure, aperture, program, ISO, dates, offsets, bias, metering, flash, focal, modes, lens


class _Tiff:
    """Random access into a TIFF file, with the head buffered since IFDs usually live there."""

    HEAD = 256 * 1024

    def __init__(self, f):
        self.f = f
        self.head = f.read(self.HEAD)
        if self.head[:2] == b"II":
            self.e = "<"
        elif self.head[:2] == b"MM":
            self.e = ">"
        else:
            raise ValueError("not a TIFF file")

    def read(self, off: int, n: int) -> bytes:
        if off + n <= len(self.head):
            return self.head[off : off + n]
        self.f.seek(off)
        return self.f.read(n)

    def first_ifd(self) -> int:
        return struct.unpack(self.e + "I", self.read(4, 4))[0]

    def ifd(self, off: int) -> tuple[dict, int]:
        """Returns ({tag: values}, next_ifd_offset). Values are tuples; strings are str."""
        (n,) = struct.unpack(self.e + "H", self.read(off, 2))
        raw = self.read(off + 2, n * 12 + 4)
        tags = {}
        for i in range(n):
            tag, typ, cnt, val = struct.unpack_from(self.e + "HHI4s", raw, i * 12)
            size = _TYPE_SIZES.get(typ, 1) * cnt
            if size > 4:
                if size > 64 * 1024:  # maker notes and other blobs we don't need
                    continue
                val = self.read(struct.unpack(self.e + "I", val)[0], size)
            tags[tag] = self._decode(typ, cnt, val)
        (nxt,) = struct.unpack_from(self.e + "I", raw, n * 12)
        return tags, nxt

    def _decode(self, typ, cnt, val):
        if typ == 2:
            return val[:cnt].split(b"\0", 1)[0].decode("latin-1").strip()
        if typ in (5, 10):
            fmt = "I" if typ == 5 else "i"
            nums = struct.unpack(f"{self.e}{2 * cnt}{fmt}", val[: 8 * cnt])
            return tuple(a / b if b else 0.0 for a, b in zip(nums[::2], nums[1::2]))
        if typ in _TYPE_FMT:
            return struct.unpack(f"{self.e}{cnt}{_TYPE_FMT[typ]}", val[: _TYPE_SIZES[typ] * cnt])
        return val[:cnt]


def _jpeg_size(head: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's SOF marker, or None if it isn't a baseline/progressive JPEG.

    Lossless JPEG (SOF3, used for DNG raw data) is rejected - browsers can't show it.
    """
    if head[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(head):
        if head[i] != 0xFF:
            i += 1
            continue
        m = head[i + 1]
        if m in (0xC0, 0xC1, 0xC2):
            h, w = struct.unpack(">HH", head[i + 5 : i + 9])
            return w, h
        if 0xC3 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            return None
        if m == 0xFF or m == 0xD8 or 0xD0 <= m <= 0xD7 or m == 0x01:
            i += 1 if m == 0xFF else 2
            continue
        (seg,) = struct.unpack(">H", head[i + 2 : i + 4])
        i += 2 + seg
    return None


def inspect(path: Path) -> RawInfo:
    info = RawInfo()
    with open(path, "rb") as f:
        try:
            t = _Tiff(f)
        except ValueError:
            return info
        ifd0, nxt = t.ifd(t.first_ifd())
        info.orientation = ifd0.get(0x0112, (1,))[0]

        ifds = [ifd0]
        for off in ifd0.get(0x014A, ()):  # SubIFDs
            ifds.append(t.ifd(off)[0])
        while nxt and len(ifds) < 16:  # IFD1, IFD2, ... (CR2 keeps previews here)
            d, nxt = t.ifd(nxt)
            ifds.append(d)

        seen = set()
        for d in ifds:
            if 0x0201 in d and 0x0202 in d:
                off, length = d[0x0201][0], d[0x0202][0]
            elif d.get(0x0103, (0,))[0] in (6, 7) and len(d.get(0x0111, ())) == 1:
                off, length = d[0x0111][0], d[0x0117][0]
            else:
                continue
            if off in seen or length < 1024:
                continue
            seen.add(off)
            size = _jpeg_size(t.read(off, min(length, 64 * 1024)))
            if size:
                info.jpegs.append(Embedded(off, length, *size))
        info.jpegs.sort(key=lambda j: j.width * j.height)

        if (
            ifd0.get(0x0103, (0,))[0] == 1
            and ifd0.get(0x0106, (0,))[0] == 2
            and ifd0.get(0x0115, (0,))[0] == 3
            and ifd0.get(0x0102, (0,))[0] == 8
            and len(ifd0.get(0x0111, ())) == 1
        ):
            info.rgb_thumb = (ifd0[0x0111][0], ifd0[0x0100][0], ifd0[0x0101][0])

        if 0x8769 in ifd0:
            exif = t.ifd(ifd0[0x8769][0])[0]
            info.exif = _exif_summary(ifd0, exif)
            single = lambda v: v[0] if isinstance(v, tuple) and len(v) == 1 else v
            info.tags = {
                "ifd0": {k: single(ifd0[k]) for k in IFD0_TAGS if k in ifd0},
                "exif": {k: single(exif[k]) for k in EXIF_TAGS if k in exif},
            }
    return info


def _exif_summary(ifd0: dict, exif: dict) -> dict:
    out = {}
    if 0x0110 in ifd0:
        out["camera"] = ifd0[0x0110]
    if 0xA434 in exif:
        out["lens"] = exif[0xA434]
    if 0x9003 in exif:
        out["taken"] = exif[0x9003]
    if 0x829A in exif:
        s = exif[0x829A][0]
        out["shutter"] = f"1/{round(1 / s)}" if 0 < s < 0.5 else f"{s:g}s"
    if 0x829D in exif:
        out["aperture"] = f"f/{round(exif[0x829D][0], 1):g}"
    if 0x8827 in exif:
        out["iso"] = exif[0x8827][0]
    if 0x9204 in exif and exif[0x9204][0]:
        out["ev"] = f"{exif[0x9204][0]:+.1f} EV"
    if 0x920A in exif:
        out["focal"] = f"{exif[0x920A][0]:g}mm"
    return out


def orientation_app1(orientation: int) -> bytes:
    """A minimal EXIF APP1 segment carrying only an Orientation tag.

    Embedded JPEGs don't carry the camera's orientation (it's in the raw's IFD0),
    so we splice this in after SOI and let the browser rotate the image itself.
    """
    tiff = b"MM\x00\x2a\x00\x00\x00\x08" + struct.pack(">HHHIHHI", 1, 0x0112, 3, 1, orientation, 0, 0)
    return b"\xff\xe1" + struct.pack(">H", 2 + 6 + len(tiff)) + b"Exif\x00\x00" + tiff


def _mask_png(im: Image.Image) -> bytes:
    """A 0/1 'L' image as a 1-bit PNG: 1 is magenta, 0 transparent. Usually a few KB."""
    im.putpalette([0, 0, 0, 255, 0, 255])
    buf = io.BytesIO()
    im.save(buf, "PNG", transparency=0, bits=1, compress_level=1)
    return buf.getvalue()


class Photo:
    """One image file. Parsing is lazy and cached; methods are safe to call from many threads."""

    def __init__(self, path: Path):
        self.path = path
        self.is_raw = path.suffix.lower() in RAW_EXTS
        self._info: RawInfo | None = None
        self._thumb: bytes | None = None
        self._rendered: bytes | None = None
        self._clip: bytes | None = None
        self._clip_sized: dict[int, bytes] = {}
        self._lock = threading.RLock()

    @property
    def info(self) -> RawInfo:
        if self._info is None:
            try:
                self._info = inspect(self.path) if self.is_raw else RawInfo()
            except (OSError, ValueError, struct.error, KeyError, IndexError):
                self._info = RawInfo()
        return self._info

    def jpeg(self, min_side: int = 0) -> list[bytes]:
        """JPEG bytes (as chunks to write in order) for display.

        Picks the smallest embedded JPEG whose long side is at least ~min_side;
        min_side=0 means the largest available.
        """
        if not self.is_raw:
            return [self.path.read_bytes()]
        info = self.info
        if not info.jpegs:
            return [self._render()]
        pick = info.jpegs[-1]
        if min_side:
            for j in info.jpegs:
                if max(j.width, j.height) >= min_side * 0.9:
                    pick = j
                    break
        with open(self.path, "rb") as f:
            f.seek(pick.offset)
            data = f.read(pick.length)
        if info.orientation in (0, 1):
            return [data]
        return [data[:2], orientation_app1(info.orientation), data[2:]]

    def clipping(self, long_side: int = 0) -> bytes:
        """The clipping overlay (see clipping_mask), max-pooled down to about long_side.

        Pooling keeps a pixel flagged if anything under it was clipped, so a small
        blown highlight stays visible at fit-to-screen size instead of being averaged away.
        """
        full = self.clipping_mask()
        if not long_side:
            return full
        if long_side not in self._clip_sized:
            im = Image.open(io.BytesIO(full))
            scale = long_side / max(im.size)
            if scale >= 1:
                return full
            size = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
            small = im.convert("L").resize(size, Image.Resampling.BOX).point(lambda v: 1 if v else 0)
            self._clip_sized[long_side] = _mask_png(small)
        return self._clip_sized[long_side]

    def clipping_mask(self) -> bytes:
        """PNG overlay of where the *raw* data is clipped, one pixel per 2x2 Bayer block.

        Magenta where any photosite in the block is saturated, or all four are at or below
        the black level; transparent elsewhere. Cropped and rotated to line up with the
        embedded JPEG. Slow (~0.7 s): the whole raw has to be read and decompressed.
        """
        if self._clip is None:
            if not self.is_raw:
                raise ValueError("not a raw file")
            import numpy as np
            import rawpy

            with rawpy.imread(io.BytesIO(self.path.read_bytes())) as raw:
                img = raw.raw_image_visible
                img = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]
                # Lossy-compressed NEFs saturate a few codes below LibRaw's nominal white
                # level (4087 vs 4095 on a D3500), so data topping out just under it has hit
                # the ceiling. Data well below it hasn't clipped at all.
                top = int(img.max())
                saturated = top if top >= 0.98 * raw.white_level else raw.white_level
                hot = img >= saturated
                black = raw.black_level_per_channel
                if len(set(black)) == 1:
                    cold = img <= black[0]
                else:
                    colors = raw.raw_colors_visible[: img.shape[0], : img.shape[1]]
                    cold = img <= np.asarray(black, img.dtype)[colors]
            mask = (hot[0::2, 0::2] | hot[0::2, 1::2] | hot[1::2, 0::2] | hot[1::2, 1::2]) | (
                cold[0::2, 0::2] & cold[0::2, 1::2] & cold[1::2, 0::2] & cold[1::2, 1::2]
            )
            info = self.info
            if info.jpegs:
                # The embedded JPEG is a centred crop of the sensor (6000x4000 of 6016x4016).
                j = info.jpegs[-1]
                th, tw = j.height // 2, j.width // 2
                mh, mw = mask.shape
                if th <= mh and tw <= mw:
                    y, x = (mh - th) // 2, (mw - tw) // 2
                    mask = mask[y : y + th, x : x + tw]
            im = Image.fromarray(mask.view(np.uint8))
            if info.orientation in _TRANSPOSE:
                im = im.transpose(_TRANSPOSE[info.orientation])
            self._clip = _mask_png(im)
        return self._clip

    def size(self) -> tuple[int, int] | None:
        """Displayed (width, height) of the largest image, after orientation."""
        info = self.info
        if not info.jpegs:
            return None
        j = info.jpegs[-1]
        return (j.height, j.width) if info.orientation in (5, 6, 7, 8) else (j.width, j.height)

    def meta(self) -> dict:
        if not self.is_raw:
            return self._jpeg_meta()
        m = dict(self.info.exif)
        if size := self.size():
            m["width"], m["height"] = size
        return m

    def thumbnail(self) -> bytes:
        if self._thumb is None:
            with self._lock:
                if self._thumb is None:
                    self._thumb = self._make_thumbnail()
        return self._thumb

    def _make_thumbnail(self) -> bytes:
        info = self.info
        if self.is_raw and info.rgb_thumb and info.jpegs:
            # Tiny but nearly free: ~60 KB read. Nikon letterboxes it to 4:3, so crop the bars.
            off, w, h = info.rgb_thumb
            with open(self.path, "rb") as f:
                f.seek(off)
                im = Image.frombytes("RGB", (w, h), f.read(w * h * 3))
            big = info.jpegs[-1]
            ch = round(w * big.height / big.width)
            if ch < h:
                top = (h - ch) // 2
                im = im.crop((0, top, w, top + ch))
        else:
            if self.is_raw:
                src = b"".join(self.jpeg(THUMB_SIZE)) if info.jpegs else self._render()
            else:
                src = self.path.read_bytes()
            im = Image.open(io.BytesIO(src))
            im.draft("RGB", (THUMB_SIZE, THUMB_SIZE))
            orientation = im.getexif().get(0x0112, 1)
            im = im.convert("RGB")
            im.thumbnail((THUMB_SIZE, THUMB_SIZE))
            if not self.is_raw and orientation in _TRANSPOSE:
                im = im.transpose(_TRANSPOSE[orientation])
        if self.is_raw and info.orientation in _TRANSPOSE and info.jpegs:
            im = im.transpose(_TRANSPOSE[info.orientation])
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=85)
        return buf.getvalue()

    def _render(self) -> bytes:
        """Fallback for raws without an embedded JPEG: half-size demosaic (slow, ~1s)."""
        if self._rendered is None:
            with self._lock:
                if self._rendered is None:
                    import rawpy

                    with rawpy.imread(str(self.path)) as raw:
                        rgb = raw.postprocess(half_size=True, use_camera_wb=True, output_bps=8)
                    buf = io.BytesIO()
                    Image.fromarray(rgb).save(buf, "JPEG", quality=90)
                    self._rendered = buf.getvalue()
        return self._rendered

    def _jpeg_meta(self) -> dict:
        try:
            with Image.open(self.path) as im:
                exif = im.getexif()
                w, h = im.size
                if exif.get(0x0112, 1) in (5, 6, 7, 8):
                    w, h = h, w
                sub = exif.get_ifd(0x8769)
        except OSError:
            return {}
        m = {"width": w, "height": h}
        if 0x0110 in exif:
            m["camera"] = str(exif[0x0110]).strip()
        if 0x829A in sub:
            s = float(sub[0x829A])
            m["shutter"] = f"1/{round(1 / s)}" if 0 < s < 0.5 else f"{s:g}s"
        if 0x829D in sub:
            m["aperture"] = f"f/{round(float(sub[0x829D]), 1):g}"
        if 0x8827 in sub:
            m["iso"] = sub[0x8827]
        if 0x920A in sub:
            m["focal"] = f"{float(sub[0x920A]):g}mm"
        if 0x9003 in sub:
            m["taken"] = sub[0x9003]
        return m
