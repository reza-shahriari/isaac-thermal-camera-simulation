# ADR 0149 — AGC is a set of vendor-neutral families, and a band's display follows its detector

**Status:** Accepted
**Date:** 2026-09-26

## Context

ADR 0147 approximated one camera's default AGC, the FLIR Boson's. That raised two questions that
matter more than which Boson setting is right:

1. **Do other cameras do something fundamentally different?** Reading more than one vendor's
   documentation, they don't. The uncooled LWIR cores all build on one 14-bit → 8-bit histogram
   and differ in which controls they expose and what they call them:
   - FLIR Boson [R51]: plateau value, Linear Percent, tail rejection, max gain, damping, ACE,
     Information-Based Equalization, DDE.
   - FLIR Lepton (its Software IDD): HEQ with a **high clip limit** (the plateau), a **low clip
     limit** (a constant added to every occupied bin), a Linear Percent defined differently from
     the Boson's, damping and an ROI.
   - Xenics and Sensors Unlimited SWIR cores: auto-gain and histogram equalisation with a
     "maximal allowed stretching" (a max gain) and an equalisation strength.
   - Most of the rest (InfiRay, HIKMICRO, Teledyne DALSA) publish nothing, and the literature
     describes CLAHE-style local equalisation for them.

   The algorithm *families* are shared. The parameters and some proprietary weighting are not.
2. **Is AGC the same across bands?** Not quite, and the difference sits in front of the AGC:
   - **Emissive bands (LWIR, MWIR)** see a scene whose radiance changes by a factor of a few
     between night and noon, so exposure is fixed (an MWIR camera picks an integration time) and
     all the display work is in the AGC.
   - **Reflective bands (NIR, SWIR)** see sunlight, which changes by five to six decades between
     noon and starlight. Their cameras are exposure-driven like a visible camera: auto-exposure
     sets the integration time (and switches gain modes), then a comparatively light AGC runs.
   - A silicon NIR sensor's display is the visible-camera chain: black level, linear stretch,
     display gamma. It is not a bolometer's equaliser.

   Every sensor config in this repository, including `example_nir_si_1280.yaml`, carried the
   Boson's `isp` block copied verbatim: plateau 0.012, DDE, gamma 1.

## Decision

- **The `isp.agc` modes are families, not cameras.** `linear`, `plateau_equalization`,
  `plateau_local` (tiled, CLAHE construction), `information_based` (detail-weighted) and `none`.
  A specific camera is a **parameter set** in its sensor YAML.
- **The controls shared across vendors are config fields on every equalising mode:**
  - `linear_percent` (ADR 0147).
  - `clip_limit_low`: the Lepton low clip, a fraction of N added to every occupied bin.
  - `max_gain`: a cap in display codes per DN. When it binds, the occupied range is centred on
    mid-grey.

  Each is zero by default, and zero is the old operator bit for bit. At its default a field is
  also left out of both the config hash and the ISP hash.
- **The Lepton family's own Linear Percent is not added as a separate field.** It adds a fraction
  of N to every non-zero bin, which is `clip_limit_low` by another name. One control, one name.
- **The NIR config gets a visible-camera display:** `agc: linear`, `gamma: 2.2`, no DDE. SWIR keeps
  histogram equalisation, because its vendors document it on-board.
- **Auto-exposure for photon detectors is a separate step (`SC.27`).** It changes the detector,
  not the display. Until it exists, a reflective-band scene is valid only near the exposure its
  config was written for.
- **The device (Warp) AGC refuses any of these controls** rather than silently dropping them. It
  had been ignoring `linear_percent` since ADR 0147, and that gap is now closed with an error.

## Consequences

- The user selects an AGC with `isp.agc` in the sensor YAML, or with `--agc` in
  `scripts/redisplay_planes.py`. `--agc all` renders every family on the same frames.
- Measured, and held by `tests/unit/test_agc_vendor_controls.py`:
  - **Clear sky with a 0.6 % target:** the low clip at 10⁻³ takes the target from ≤ 3 to ≥ 20
    codes, with its parts in order.
  - **Bland 33 DN sky:** `max_gain = 1.25` cuts the noise from all 256 codes to ≤ 43, centred
    on mid-grey.
  - **Cluttered sky:** the low clip barely helps, because every bin is dense.
- **The NIR golden moved on its display only.** `nir_frame_display8` changed by up to 71 codes;
  `nir_frame_dn16` and `nir_frame_radiance` are bit-identical.
- **Error not bounded:** no camera's actual parameter values are fitted. The families are right,
  but the numbers in each YAML stay ESTIMATED until a Tier 4 comparison against public footage.

## Revisit when

- A public clip from a named camera is fitted (the `XD` lane). Its fitted parameters become that
  camera's preset.
- `SC.27` lands auto-exposure. The NIR and SWIR display parameters then need re-checking at the
  exposures it chooses.
