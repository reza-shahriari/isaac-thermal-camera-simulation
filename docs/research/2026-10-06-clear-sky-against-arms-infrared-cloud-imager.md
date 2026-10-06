# The clear LWIR sky against a calibrated sky camera — first report, 2026-10-06

**irsim's clear night sky is compared, elevation by elevation, with ARM's Infrared Cloud Imager,
an upward-looking LWIR camera whose pixels are calibrated in W/(m²·sr). On a cold, dry night the
model is the measurement to within 7 %. On a humid summer night it is 24–28 % too bright at every
elevation, and the instrument's own clear-sky model is within 3 %.**

This is roadmap step `XD.6`, part 1. The decisions are recorded in
[ADR 0183](../decisions/0183-the-clear-lwir-sky-is-held-to-arms-infrared-cloud-imager.md). The
humid misfit is roadmap step `AT.37`. The cloudy images in the same files are `XD.6` part 2, and
they have not been compared yet.

---

## 1. The short version

1. **This is the first radiometric check of the sky against reality.** Until now, the layered clear
   sky ([ADR 0071](../decisions/0071-layered-slant-path-atmosphere.md)) was calibrated on one
   anchor: a dry Tucson sky at −40 °C, 15° elevation. Every later check compared it with other
   models. The public IR imagery used elsewhere in the project is 8-bit and post-AGC, so no radiance
   survives in it ([ADR 0068](../decisions/0068-evaluation-data-and-what-it-may-claim.md)).
2. **The dry sky is right.** 11 Dec 2023, 2.5 °C, 0.86 cm of precipitable water: model/measured
   **1.00–1.07** at every elevation. That is closer than the instrument's own model (0.89–0.93).
3. **The humid sky is too bright.** 6 Aug 2023, 27.0 °C, 3.96 cm: **1.24–1.28** at every
   elevation. The misfit grows with humidity: May (1.41 cm) sits between the two, at 1.03–1.18.
4. **The cause is the humid water-vapour absorption, and partly the vertical profile.** It is not
   the camera band. The fix is `AT.37`. It needs each night's measured dew point before anything
   is refitted.

---

## 2. The instrument and the data

ARM's Infrared Cloud Imager (ICI) was deployed by NWB Sensors at the Southern Great Plains site
from May to December 2023 ([doi:10.5439/3001561](https://doi.org/10.5439/3001561); ARM data, no
use constraints, cite the DOI).

- **Sensor.** An upward-looking 7.3–14 µm uncooled microbolometer imaging the full sky.
- **What each image file holds.** Calibrated radiance per pixel, a cloud mask, and the
  instrument's own modelled clear sky. A separate file maps every pixel to its azimuth and
  elevation.
- **Calibration.** The deployment's intercomparison report holds it to ARM's AERI spectrometer to
  0.17 W/(m²·sr).

The owner fetched **92 images over five nights**. The raw files are 2.3 MB each and are not in
git. `scripts/ici_extract_profiles.py` reduces every image to a few numbers:

- the median radiance of the clear pixels within ±1.5° of eight elevations (10° to 88°);
- the instrument's model at the same elevations;
- the image's cloud fraction.

The result is one 15 kB file: [`data/validation/ici_sgp2023_clear_sky.csv`](../../data/validation/ici_sgp2023_clear_sky.csv).
An image counts as clear when at most 0.5 % of its sky is masked as cloud. 30 images qualify, on
three nights.

**The surface weather came off a video.** The companion meteorology datastreams arrived empty, or
for other dates. The instrument's processing prints the surface air temperature and GNSS
precipitable water vapour (PWV) in the title of every frame of the deployment time-lapse.
`scripts/ici_met_from_video.py` reads those titles with a glyph-template reader, trained on seven
frames transcribed by eye. It takes the frame nearest each image, never more than a minute away.
These are the same inputs the instrument's own clear-sky model ran on. The values change smoothly
from minute to minute, which a misread digit would not; a test checks this.

## 3. How the model is driven

`irsim.validation.ici.model_clear_sky` runs the layered atmosphere (`us_standard_clear` preset)
with one constant weather sample per image:

- **Air temperature** is the measured surface temperature.
- **Humidity** is set so that the measured column is held. The model's water falls off as
  `w0 · e^(−h/H_w)`, whose column is `w0 · H_w`. So `w0 = PWV / H_w`, with the preset's
  `H_w` = 2 km. If that would need more than 100 % surface humidity, it is clamped and flagged.
  None of the three clear nights needed that: 39 %, 77 % and 75 %.
- **Band.** The ICI's response shape is not published; its own report names this as the main
  limitation of its calibration. The comparison uses the project's ESTIMATED VOx microbolometer
  curve as a stand-in.

## 4. Results

Model/measured at each elevation, re-run on 2026-10-06 with `scripts/validate_sky_ici.py`.

| Night | Surface T | PWV | Images | 10° | 20° | 30° | 45° | 60° | 88° | ICI's own model |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 Dec | 2.5 °C | 0.86 cm | 5 | 1.059 | 1.043 | 1.025 | 1.024 | 1.020 | 1.004 | 0.89–0.93 |
| 21 May | 20.8 °C | 1.41 cm | 10 | 1.026 | 1.087 | 1.136 | 1.171 | 1.170 | 1.180 | 0.97–1.07 |
| 6 Aug | 27.0 °C | 3.96 cm | 15 | 1.277 | 1.245 | 1.239 | 1.250 | 1.271 | 1.270 | 1.01–1.03 |

The measured zenith radiance on those nights was 7.1, 10.6 and 21.6 W/(m²·sr).

The plot of all three nights (measured points, irsim and the instrument's own model against
elevation) is written by the script to `outputs/validation/ici/ici_clear_sky.png`. It is not in
git; the site gallery shows it as the `sky-ici` entry.

## 5. What the misfit is, and what it is not

**It is not the band.** A different response shape scales every sky roughly alike. It cannot make
a misfit that grows from 0 % to 25 % with humidity. Swapping the VOx curve for 7.5–13.5 µm and
8–14 µm top-hats moves the December zenith by under 7 %; a test checks this.

**It is mostly how the humid column absorbs.** The session that ran the first comparison split the
band by spectral class, and those diagnostics are recorded in ADR 0183. They were not re-run for
this report.

- **Water lines** (7.8–8.3 and 12.5–13.2 µm): from December to August, their emissivity rises
  from 0.41 to 0.97.
- **Window class:** from 0.047 to 0.31.

`AT.27`'s squared self-continuum term
([ADR 0160](../decisions/0160-lwir-humidity-carries-a-self-continuum-square.md)) is the first
suspect. With the dry anchor held, dropping that term brings August to 1.06–1.10 and leaves
December alone.

**Part of it is the vertical profile.** May's zenith stays at about 1.2 under every continuum
coefficient tried. The model spreads the measured PWV over a fixed 2 km water scale height and a
constant surface sample, so an evening inversion and a deeper real column are both missing.
Fitting the continuum alone would bake that profile error into a spectroscopic constant. ADR 0160
already warned against exactly that.

## 6. What is next

- **`AT.37` — fix the humid sky.** First pin each night's column with its measured surface dew
  point (ARM `sgpmetE13.b1` for the five dates), so the column depth is known independently of the
  PWV. Then refit the humid terms with the dry anchor held. This moves every LWIR golden.
  `tests/unit/test_ici_clear_sky.py` already holds the target, as a strict expected failure: the
  August night within 10 % at every elevation, with December still within 7 %.
- **`XD.6` part 2 — the cloudy images.** The same 92 files hold 62 cloudy images. They can be held
  against two things: the cloud-clutter model
  ([ADR 0070](../decisions/0070-cloud-clutter-model.md)), and the radiance of cloud bases at the
  temperature the deck assigns them.
- **More nights.** The deployment ran for nine months. Only three of the five nights were clear, one
  in each humidity regime. A longer, clear-sky-filtered sample would put a spread on each
  number.

## 7. Reproduce

```bash
python scripts/validate_sky_ici.py                 # JSON + plot → outputs/validation/ici/
python -m pytest tests/unit/test_ici_clear_sky.py  # 4 pass, 1 strict xfail (AT.37)
```

Rebuilding the CSV from the raw files needs the ARM request unpacked locally, plus `h5py`, OpenCV
and `ffmpeg`. Both scripts' docstrings give the commands.
