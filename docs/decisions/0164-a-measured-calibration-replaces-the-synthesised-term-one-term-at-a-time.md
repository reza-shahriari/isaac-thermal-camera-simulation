# 0164 — A measured calibration replaces the synthesised term, one term at a time

Date: 2026-09-29

**Status:** Accepted (2026-09-29, SC.33).

## Context

Every camera in the catalogue is described by design: a focal length and a pitch make the
pinhole, `optics.distortion` names a lens model whose coefficients nobody measured, the
fixed-pattern noise is drawn from `noise.ratios_3d` with the sensor seed, the bad pixels come
from a Neyman–Scott cluster process, and the noise magnitude is anchored on the datasheet's
NETD. That is the right description of a camera nobody has held. It is the wrong description of
the unit on the bench, whose OpenCV calibration, gain and offset maps, bad-pixel map, NETD and
SITF are ordinary outputs of the tests every real camera goes through -- and which were
un-representable here: `DistortionSpec` was schema only, and nothing applied it (roadmap SC.33).

Two things had to be decided: where a measured quantity goes when the model already has a
synthesised one, and how the engine-free path, which has no camera prim to distort its image,
shows a measured lens.

## Options

1. **A separate "calibrated camera" schema**, loaded by a different code path. Rejected: two
   schemas drift apart, and every consumer of a sensor config would have to know both.
2. **Overrides on the existing fields** -- write the measured NETD into `noise.netd_mk_at_300k`,
   the measured lens into `optics.distortion`. Rejected: a file would then no longer say what
   was designed and what was measured, and a gain map has no existing field to overwrite.
3. **An optional `calibration:` block whose every member is optional, each replacing exactly one
   synthesised term when present** (chosen). `calibration.geometric` is OpenCV's pinhole and
   Brown-Conrady lens in pixels -- pasted from `cv2.calibrateCamera`, not converted -- and
   replaces the designed pinhole and `optics.distortion` wherever a projection is made
   (`Intrinsics.from_sensor`, `SensorSpec.effective_distortion`): the USD camera and its lens
   schema, the ROS `CameraInfo`, the point targets, the rotor discs, the CPU builder.
   `calibration.radiometric` names `.npy` sidecars and two figures: `gain_map` and `offset_map`
   become the fixed pattern the frames carry (the drawn V/H/VH pattern is zeroed, the drift of
   ADR 0058 breathes from them); `bad_pixel_map` replaces the drawn cluster map, every flagged
   pixel on the factory map and classed DEAD, the one class a map that only says "bad" supports;
   `netd_k` is the anchor of ADR 0025 whatever the FPA's kind (a measurement on the unit beats a
   datasheet electron budget); `sitf_dn_per_k` is the NUC residual's millikelvin-to-DN
   conversion of ADR 0056. Absent, each term is synthesised as before; absent altogether, the
   file is the camera it always was and hashes the same (the SC.32 rule: no bump).

For the engine-free path: the Isaac camera distorts every AOV at render time through the lens
schema on the prim (ADR 0015), so the frame and its visible companion arrive distorted together.
The CPU builder is that path's engine, so it is the builder that distorts -- every plane at once,
resampled from the ideal pinhole grid onto the distorted one with the same intrinsics and the
same model (`irsim.optics.calibration`): floats bilinearly, ids and masks nearest, the encoded
temperature re-derived, point targets moved to where the lens puts their ray. `run_frame` is not
told about lenses at all, which is what keeps the two paths agreeing about where a ray lands.

## Consequences

- A user pastes their camera's calibration into its YAML and the simulator renders that unit:
  the checkerboard test straightens under `cv2.undistort` to 0.1 px, a loaded gain map
  reproduces its own flat field to 0.01 DN, a loaded bad-pixel map is the map the chain replaces.
- The maps' content hashes enter the config hash, so a frame records which calibration made it
  and a different map is a different camera; a file with no block, or an empty one, hashes as
  before, and every golden recorded before SC.33 stays valid.
- ESTIMATED where it matters: the maps are the residual non-uniformity *after* the unit's own
  correction, applied as a constant; a camera whose maps change with FPA temperature needs the
  gain/offset(T_FPA) polynomials of ADR 0053, which this does not replace.
- Not done here: the truth planes of the CPU builder's `resolved` boxes stay on the ideal grid;
  the Isaac-side round trip (render through the prim's schema, undistort with the same numbers)
  is `IG.19`'s render to make.
