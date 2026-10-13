# photos

Personal raw-photo workflow tool: pick, develop, export.

```sh
uv run photos cull /mnt/d/Radno     # opens the picker
uv run photos edit /mnt/d/Radno     # opens the editor (same app, E switches between them)
uv run photos export /mnt/d/Radno   # develops every kept photo to /mnt/d/Radno/export
```

The UI runs in the Windows browser; press `?` in it for keys.

## Cull

Shows the JPEG the camera embedded in each raw file, so browsing never decodes
raw data, and prefetches/decodes the next few photos so browsing is instant.
`←`/`→` browse, `↑`/`↓` keep/reject (auto-advancing to the next photo). `J` paints
magenta wherever the raw data itself is clipped: a saturated sensor channel, or
all four photosites of a Bayer block at the black level.

## Edit

Develops the kept photos: dynamic range compression, exposure (applied after
compression), colour temperature, contrast and saturation. One shader
(`web/develop.frag`) does it, in the browser for the preview, so sliders respond
instantly, and on the server's GPU for exports, so an export is what the editor
showed.

Unedited, a photo looks like the camera's own JPEG (`look.py`): its tone curve is
fitted to the JPEG embedded in that raw (the camera adapts it per scene), and
colour comes from a profile per camera model in `photos/profiles/`, fitted from
raw + JPEG pairs:

```sh
uv run python scripts/calibrate_camera.py /mnt/d/Radno
```

Exports also get noise reduction like the camera's (`denoise.py`): non-local
means on the raw data before demosaicing, using the noise measured in each photo
itself; colour noise smoothed along brightness edges; then part of the removed
grain given back, more at low ISO, as the camera keeps it. It runs on the GPU,
about 0.3 s per photo in a batch.

The editor shows what the export will be, noise reduction included. The server
prepares the export's noise-reduced input at full size (~3 s per photo, for the
photo on screen and then the next one, so at editing pace it's ready when you get
there); until then a quick preview without noise reduction stands in ("Reducing
noise…"). The browser develops a shrunk copy for the whole-photo view, and `Z` (or
a click) shows 100% at the cursor from tiles of the full size, so sliders stay
instant either way.

Photos are grouped into series: a pause of more than 10 seconds between two
consecutive shots starts a new one (`edits.SERIES_GAP`). A change applies to the
photo and the following ones of its series; changing a photo that isn't the first
of its series splits the series just before it. Series are worked out over all
photos, picked or not, so picking a photo later puts it in its series with that
series' settings.

## State

Everything is saved as you go to `<folder>/.photos.json`:

```json
{
  "marks": {"DSC_9282.NEF": "keep", "DSC_9283.NEF": "reject"},
  "edits": {"DSC_9290.NEF": {"compress": 40, "exposure": 0.3, "temp": null, "saturation": 10}},
  "version": 1
}
```

An edit sits on the photo where it starts to apply (`temp: null` means as shot).

## Tests

```sh
uv run pytest
```
