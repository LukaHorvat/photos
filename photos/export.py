"""Export: develop photos at full resolution to JPEG, keeping the camera's EXIF."""

from __future__ import annotations

import shutil
from functools import cache
from pathlib import Path

from PIL import Image, ImageCms

from . import develop
from .raw import Photo

QUALITY = 92


@cache
def _srgb_profile() -> bytes:
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def export_photo(photo: Photo, settings: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    if not photo.is_raw:  # a JPEG has nothing to develop
        out = out_dir / photo.path.name
        shutil.copyfile(photo.path, out)
        return out
    im = Image.fromarray(develop.develop(photo.path.read_bytes(), settings))
    exif = Image.Exif()
    tags = photo.info.tags
    for tag, value in tags.get("ifd0", {}).items():
        exif[tag] = value
    exif[0x0112] = 1  # rotated already
    exif[0x0131] = "photos"
    sub = exif.get_ifd(0x8769)
    for tag, value in tags.get("exif", {}).items():
        sub[tag] = value
    out = out_dir / f"{photo.path.stem}.jpg"
    tmp = out.with_suffix(".jpg.tmp")
    im.save(tmp, "JPEG", quality=QUALITY, subsampling=0, exif=exif, icc_profile=_srgb_profile())
    tmp.replace(out)
    return out
