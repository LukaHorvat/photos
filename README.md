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
compression), colour temperature and saturation. The preview is rendered by a
WebGL shader in the browser (`web/develop.js`) from linear raw data the server
prepares once per photo, so sliders respond instantly; exports run the same
pipeline in numpy (`develop.py`) at full resolution. The default rendering's
tone curve is fitted to the camera's own JPEGs (`scripts/calibrate_curve.py`).

Photos are grouped into series: runs of consecutive shots with identical shutter
speed, aperture, ISO, exposure compensation and focal length (`edits.series_key`).
A change applies to the photo and the following ones of its series; changing a
photo that isn't the first of its series splits the series just before it. Series
are worked out over all photos, picked or not, so picking a photo later puts it in
its series with that series' settings.

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
