"""Which settings each photo gets: series detection and edit propagation.

Photos are grouped into series: bursts of shots with no pause longer than
SERIES_GAP between them, in shooting order and over *all* photos, kept or not, so
picking or un-picking a photo never changes the groups. An edit is stored on the
photo where it starts:

    edits = {"DSC_0012.NEF": {"compress": 40, ...}}

A photo gets the settings of the nearest edit at or before it in its group, or the
defaults. So editing the first photo of a group changes the whole group, and
editing a later one splits the group just before it: earlier photos keep their
settings, it and the following ones take the new ones. Both come down to
`edits[name] = settings`. An edit that changes nothing compared to what the photo
would inherit is dropped, so moving a slider back merges a split again.

web/editor.js has the same `resolve` for display. Keep them in sync.
"""

from __future__ import annotations

from datetime import datetime

from .develop import DEFAULTS, clean

SERIES_GAP = 10  # seconds; a longer pause between two shots starts a new series


def taken(meta: dict) -> datetime | None:
    try:
        return datetime.strptime(meta["taken"], "%Y:%m:%d %H:%M:%S")
    except (KeyError, TypeError, ValueError):
        return None


def group_ids(metas: list[dict]) -> list[int]:
    """Series number for each photo, in shooting order. A photo without a capture time is a series of its own."""
    ids, prev = [], None
    for meta in metas:
        t = taken(meta)
        same = t is not None and prev is not None and abs((t - prev).total_seconds()) <= SERIES_GAP
        ids.append(ids[-1] + (not same) if ids else 0)
        prev = t
    return ids


def resolve(names: list[str], groups: list[int], edits: dict) -> list[dict]:
    """The settings of every photo."""
    out, current = [], DEFAULTS
    for i, name in enumerate(names):
        if i == 0 or groups[i] != groups[i - 1]:
            current = DEFAULTS
        current = edits.get(name, current)
        out.append(current)
    return out


def normalize(names: list[str], groups: list[int], edits: dict) -> dict:
    """Drop edits that are no-ops: the same as what that photo would inherit anyway."""
    out, current = {}, DEFAULTS
    for i, name in enumerate(names):
        if i == 0 or groups[i] != groups[i - 1]:
            current = DEFAULTS
        if name in edits:
            settings = clean(edits[name])
            if settings != current:
                out[name] = current = settings
    return out


def apply(names: list[str], groups: list[int], edits: dict, name: str, settings: dict) -> dict:
    """Set `name`'s settings: for it and the following photos of its group (until the next split)."""
    return normalize(names, groups, {**edits, name: clean(settings)})
