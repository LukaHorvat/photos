from PIL import Image

from photos import develop, edits, server


def shoot(folder, name, taken):
    exif = Image.Exif()
    exif.get_ifd(0x8769)[0x9003] = taken  # DateTimeOriginal
    Image.new("RGB", (8, 8)).save(folder / name, exif=exif)


def test_deleting_rejected_photos(tmp_path):
    for k, t in enumerate(["12:00:00", "12:00:05", "12:00:09", "12:30:00"], 1):
        shoot(tmp_path, f"p{k}.jpg", f"2026:05:01 {t}")  # p1-p3: a series
    for extra in ("p2.xmp", "p2.jpg.arp", "p3.xmp", "p20.xmp", "notes.txt"):
        (tmp_path / extra).write_text("x")
    lib = server.Library(tmp_path)
    for name, mark in [("p1.jpg", "keep"), ("p2.jpg", "reject"), ("p3.jpg", "keep")]:
        lib.set_mark(name, mark)
    lib.set_edit("p2.jpg", {"compress": 40})  # starts on the rejected photo, reaches p3
    assert lib.settings()["p3.jpg"]["compress"] == 40

    info = lib.rejected()
    assert info["photos"] == 1 and info["files"] == 3 and not info["recycle"]  # p2.jpg, p2.xmp, p2.jpg.arp
    result = lib.delete_rejected()

    assert result == {"deleted": 1, "errors": []}
    assert sorted(p.name for p in tmp_path.iterdir()) == [".photos.json", "notes.txt", "p1.jpg", "p20.xmp", "p3.jpg", "p3.xmp", "p4.jpg"]
    assert lib.names == ["p1.jpg", "p3.jpg", "p4.jpg"] and "p2.jpg" not in lib.state["marks"]
    settings = lib.settings()
    assert settings["p1.jpg"] == develop.DEFAULTS and settings["p3.jpg"]["compress"] == 40  # kept what it had
    assert server.Library(tmp_path).settings() == settings  # and saved so
    assert edits.normalize(lib.names, lib.groups(), lib.state["edits"]) == lib.state["edits"]


def in_use(times):
    """trash.remove where Windows refuses a file that's open, the first `times` attempts."""
    calls = []

    def remove(paths, done=lambda path, error: None):
        calls.append(list(paths))
        errors = {}
        for path in paths:
            if len(calls) <= times:
                errors[path] = "The process cannot access the file because it is being used by another process."
            else:
                path.unlink()
            done(path, errors.get(path))
        return errors

    return remove, calls


def test_a_photo_in_use_is_tried_again_and_else_stays_rejected(tmp_path, monkeypatch):
    for k, t in enumerate(["12:00:00", "12:00:05", "12:00:07"], 1):
        shoot(tmp_path, f"p{k}.jpg", f"2026:05:01 {t}")
    monkeypatch.setattr(server.time, "sleep", lambda s: None)
    lib = server.Library(tmp_path)
    lib.set_mark("p2.jpg", "reject")

    remove, calls = in_use(times=1)  # open for a moment: the second try works
    monkeypatch.setattr(server.trash, "remove", remove)
    steps = []
    assert lib.delete_rejected(lambda done, total: steps.append((done, total))) == {"deleted": 1, "errors": []}
    assert len(calls) == 2 and steps == [(1, 1)] and lib.names == ["p1.jpg", "p3.jpg"]

    lib.set_mark("p3.jpg", "reject")
    remove, calls = in_use(times=2)  # still open: it stays, marked rejected, and says why
    monkeypatch.setattr(server.trash, "remove", remove)
    result = lib.delete_rejected()
    assert result["deleted"] == 0 and result["errors"] == ["p3.jpg: The process cannot access the file because it is being used by another process."]
    assert lib.names == ["p1.jpg", "p3.jpg"] and lib.state["marks"]["p3.jpg"] == "reject"
