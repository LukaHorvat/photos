# photos

Personal raw-photo workflow tool: pick, develop, export.

```sh
uv run photos cull <folder>     # opens the picker
uv run photos edit <folder>     # opens the editor (same app, E switches between them)
uv run photos export <folder>   # develops every kept photo to <folder>/export
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
compression), white balance (temperature, and tint: green to magenta), contrast and
saturation. Two pickers set white balance from a click: `W` on something that should
be grey or white, `Shift W` on skin, which sets only the tint: skin's hue is about
the same on everyone, its warmth and saturation aren't. One shader
(`web/develop.frag`) does it, in the browser for the preview, so sliders respond
instantly, and on the server's GPU for exports, so an export is what the editor
showed. At full size the raw is demosaiced on the GPU too: LibRaw's own AHD, ported to
give the same result to the bit, ~10x faster than LibRaw runs it. A batch export takes
~0.5 s per photo.

Unedited, a photo looks like the camera's own JPEG (`look.py`): its tone curve is
fitted to the JPEG embedded in that raw (the camera adapts it per scene; beyond the
tones the photo has, it follows the camera's usual curve), and
colour comes from a profile per camera model in `photos/profiles/`, fitted from
raw + JPEG pairs:

```sh
uv run python scripts/calibrate_camera.py <folder>
```

Noise reduction works like the camera's (`denoise.py`): non-local means on the
raw data before demosaicing; colour noise smoothed along brightness edges; then
part of the removed grain given back, more at lower ISO, as the camera keeps it.
It runs on the GPU, about 0.3 s per photo in a batch. Like the camera's, it's on
from ISO 1000 (Auto); the editor's two sliders, as in Lightroom, override that per photo
(and, like the other sliders, for the rest of its series): Luminance for the grain,
Colour for colour speckles. 100% is the camera's; less mixes in that part of the photo
without noise reduction (the noise left falls as (1 - s)², which looks even across the
slider); more smooths further (luminance at 200%: no grain at all); 0 turns it off.
Luminance 0 with colour 100% keeps all the grain but no colour noise.
They apply when you let go of the slider, as they need the server to redo the photo.

**Neural (RawNIND)**, a toggle under the sliders, has a neural network do the
demosaicing and denoising instead (`neural.py`): RawNIND (Brummer & De Vleeschouwer,
2025), the model darktable 5.6 ships as "neural restore". At high ISO its edges and
colour come out cleaner: AHD draws mazes and combs from noisy data. The colour filter, the
grain and the sliders still apply, up to 100% (it has no strength of its own; less mixes in
the photo without, as above). It takes about as long as non-local means (~1.5 s for a 24 Mpx
export, on battery) and needs an NVIDIA GPU with CUDA, through ONNX Runtime: the `neural`
dependency group (~2.5 GB, installed by default; `uv sync --no-group neural` leaves it out).
The model (GPL-3.0) is downloaded from darktable-ai's GitHub release on first use, checked
against its published SHA-256, converted to float16 (twice as fast; the result differs by
~0.1%) and kept in `~/.cache/photos`.

It needs to know how noisy the photo is. That's measured in the photo itself, but
a photo full of fine texture (foliage) would read as very noisy and be smeared, so
the profile also holds the camera's typical noise per ISO, and a photo measuring
more than 1.5x that is denoised as typical:

```sh
uv run python scripts/calibrate_noise.py <folder>
```

The editor shows what the export will be. The server prepares the export's own
input at full size (1-3 s per photo, for the photo on screen and then the next
one, so at editing pace it's ready when you get there); until then a quick preview
without noise reduction stands in ("Reducing noise…"). The browser develops a
shrunk copy for the whole-photo view, and `Z` (or a click) shows 100% at the cursor
from tiles of the full size, so sliders stay instant either way.

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
  "edits": {"DSC_9290.NEF": {"compress": 40, "exposure": 0.3, "temp": null, "saturation": 10, "contrast": 0, "noise": null, "chroma": null, "neural": false}},
  "version": 1
}
```

An edit sits on the photo where it starts to apply (`temp: null` means as shot;
`noise` and `chroma`, luminance and colour noise reduction: `null` automatic, else the
strength in percent, 0 off, 100 the camera's; an edit saved before `chroma` existed
uses its `noise` for both; `neural`: by RawNIND).

## Tests

```sh
uv run pytest
```
