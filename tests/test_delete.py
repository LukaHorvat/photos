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
