from photos.develop import DEFAULTS
from photos.edits import apply, group_ids, resolve

# Ten photos: a series of six (0-5), a single photo (6), a series of three (7-9).
NAMES = [f"P{i}" for i in range(10)]
SECONDS = [0, 2, 4, 10, 18, 28, 60, 100, 101, 111]
META = [{"taken": f"2026:06:30 12:{s // 60:02d}:{s % 60:02d}"} for s in SECONDS]
GROUPS = group_ids(META)


def settings(edits):
    return [s["compress"] for s in resolve(NAMES, GROUPS, edits)]


def edit(edits, i, **change):
    current = resolve(NAMES, GROUPS, edits)[i]
    return apply(NAMES, GROUPS, edits, NAMES[i], {**current, **change})


def test_a_pause_of_more_than_ten_seconds_starts_a_series():
    assert GROUPS == [0, 0, 0, 0, 0, 0, 1, 2, 2, 2]  # gaps of exactly 10 s stay together


def test_photos_without_capture_time_stand_alone():
    t = {"taken": "2026:06:30 12:00:00"}
    assert group_ids([t, {}, t, {"taken": "garbage"}, t]) == [0, 1, 2, 3, 4]


def test_editing_first_photo_changes_whole_group_only():
    e = edit({}, 0, compress=40)
    assert settings(e) == [40] * 6 + [0] * 4


def test_editing_later_photo_splits_group_before_it():
    e = edit({}, 0, compress=40)
    e = edit(e, 3, exposure=1.0)
    s = resolve(NAMES, GROUPS, e)
    assert [x["compress"] for x in s[:6]] == [40] * 6  # the change carries the group's other settings along
    assert [x["exposure"] for x in s[:6]] == [0, 0, 0, 1, 1, 1]
    assert set(e) == {"P0", "P3"}


def test_editing_split_start_changes_only_its_part():
    e = edit(edit({}, 0, compress=40), 3, compress=80)
    e = edit(e, 3, compress=70)
    assert settings(e)[:6] == [40, 40, 40, 70, 70, 70]
    e = edit(e, 1, compress=10)  # splits the first part again, stops at the existing split
    assert settings(e)[:6] == [40, 10, 10, 70, 70, 70]


def test_reverting_a_split_merges_it_back():
    e = edit(edit({}, 0, compress=40), 3, compress=80)
    e = edit(e, 3, compress=40)
    assert e == {"P0": {**DEFAULTS, "compress": 40}}
    e = edit(e, 0, compress=0)
    assert e == {}


def test_changes_never_cross_into_the_next_group():
    e = edit({}, 4, compress=50)
    assert settings(e) == [0, 0, 0, 0, 50, 50, 0, 0, 0, 0]


def test_unpicked_photos_are_grouped_like_picked_ones():
    # Only P2 and P4 picked; editing P2 (first *picked*, not first in group) splits before P2.
    # Picking P0 later shows it with the group's untouched settings; picking P3 shows it edited.
    e = edit({}, 2, compress=30)
    assert settings(e)[:6] == [0, 0, 30, 30, 30, 30]
