# ADR 0130 — A cumulus has a size and a shape, and both bands march it

**Status:** Accepted. The deck's depth map becomes a **height ramp clipped by an inversion**
rather than a soft threshold; the field is **band-limited** at the smallest real cumulus; the deck
**tiles** instead of ending at a footprint; and the visible dome **marches** the deck exactly as
the infrared band does instead of sampling its column depth at the base.
**Date:** 2026-09-23
Roadmap: AT.15 (§5.3 a)

## Context

ADR 0127 gave the cloud a top and the frame gained 19.7 K of structure it did not have. The owner,
on the clip it produced: *"the rgb is much better now / the IR is awful now xD / seems to be some
random points those are not in the rgb at all."*

The two halves of that are one defect seen from two sides, and the visible frame is what proved
it: the RGB showed a handful of soft wisps over a mostly clear sky, and the LWIR frame of the
**same directions at the same instant** was covered in bright vertical smears. A rendered pair
cannot disagree like that about where a cloud is — ADR 0076 exists to forbid exactly this — so at
least one of the two was not reading the deck the renderer was showing.

Measured on the frame geometry of `phantom3_outbound_pointwise` (20° boresight, 31° × 25° field,
so 7.4°–32.4° of elevation), against the deck ADR 0127 built:

| | dome (RGB) | marched (LWIR) |
|---|---|---|
| cloud in frame | 26 % of pixels | 57 % of pixels |
| what it reads | column depth **at the base crossing** | optical depth **along the ray** |

Three separate causes, each visible once the depth map itself was plotted rather than inferred:

1. **The depth map was a mesa field.** `deck_field` mapped the synthesised field through a
   smoothstep of width 0.45 σ, which is a *membership* function — right for AT.11's plane-parallel
   sheet, where the number means "how much of this column is cloud". Read as a **top height** it
   says every covered column is the deck's full 1.2 km, so the deck was flat-topped blocks with
   vertical walls. Straight up that is invisible: a vertical ray through a mesa and through a
   tower of the same column depth read the same number, which is why the AT.12 tests passed. At
   20° it is the whole picture — an oblique ray runs *along* a 1.2 km wall for kilometres. Marched
   optical depth over the frame averaged **7.07** and reached **27.8**, which is opaque
   everywhere, which is the flat white AT.11 was written to remove.
2. **The field had no smallest cloud.** A `1/f^β` synthesis is scale-free, so a 25 m grid carries
   25 m clouds. A cumulus field does not: its size distribution has a mode near half a kilometre
   and falls off sharply below, because a parcel that small entrains dry air faster than it can
   condense. Sub-cloud structure is not a cosmetic error *here* precisely because the two bands
   read the deck differently — a 25 m speck stays a speck on the dome and smears over the
   kilometres the ray spends in the deck. That is, literally, "some random points those are not in
   the rgb at all".
3. **The deck had an edge.** The footprint ran to `base / tan(10°)`, and below that a ray left it
   and read clear sky — a hard, straight, cold band across the bottom of every oblique frame,
   in exactly the direction where a real cumulus field puts its densest wall.

## Decision

**Depth is a ramp clipped by an inversion, not a switch.**
`depth = smoothstep(min(1, (field − threshold) / depth_scale))`, with `depth_scale` = 1.5 σ. A
cumulus grows as far above its base as the thermal that made it can carry it, thermal strength in
a field of them is distributed about a mean, and the capping inversion clips the strongest — which
is why a fair-weather cumulus field has a common top with rounded shoulders below it. The
smoothstep rounds both ends, so there is no cusp where a tower meets the inversion and no crease
where it meets clear air. At the presets' 0.45 coverage the mean cloudy column comes out at half
the deck and 12 % of them reach the cap: a 600 m mean depth under a 1.2 km lid.

A side effect worth stating: the authored coverage is now **exact**. The ramp is zero at the
threshold, so the covered set is the set above the quantile; the old smoothstep spread cloud half
its softness past it.

**The field is band-limited at `DEFAULT_SMALLEST_CLOUD_M` = 400 m**, as a Gaussian roll-off at
400/4 m applied in the Fourier domain — where the field was synthesised, and which keeps it
exactly periodic. The blurred field is renormalised to unit variance, because both the threshold
and the ramp are expressed in σ and a blur removes variance.

**The deck tiles.** The depth map is an inverse FFT and therefore exactly periodic, so wrapping
the lookup is seamless to the bit and there is no edge to reach. `min_elevation_deg` now sizes the
*tile* — 12 km, against the 8 km the shallowest ray of a 25° frame reaches, so a frame sees at
most one repeat.

**The dome marches.** `visible_sky.environment_map` calls `CloudDeck.march` on each texel and
takes `α = 1 − exp(−τ)`, Beer's law on the authored (visible) optical depth, where the infrared
band applies `CLOUD_OD_RATIO` to the same number first. One geometry, one integration, two bands.
It is baked once per render, so the cost is a one-off where a frame is marched three hundred
times.

**The march is sized by path length, not by cells.** It is a midpoint rule over an integrand with
a jump in it — the cloud's own boundary — so it converges at first order and the error is set by
how precisely a step lands on that boundary, which is a distance in metres. Measured against a
1536-step reference over a rendered frame: 36 m steps hold the band emissivity to 0.02 at the 99th
percentile, 72 m to 0.044, 100 m to 0.061. `MARCH_STEP_M` = 36 m.

## Consequences

Measured on the same frame geometry, after:

* **Largest 0.25 K bin: 6.4 % of the frame.** It was 53.9 % before ADR 0126, 39.6 % after 0127.
* **In-cloud spread 28.8 K** (p1–p99), against the 19.7 K of ADR 0127 and the 1.25 K of the sheet.
* **Emission level 15 m to 1065 m** above the base (p1–p99).
* **The two bands draw cloud in the same pixels, by construction** — the dome test asserts the
  dome's drawn set equals the set where the marched optical depth is non-zero, and separately
  asserts that the old sampled quantity is a *different* cloud, so a dome that agreed with it
  would be the bug.
* **The camera marches at half the native pitch**, not a quarter. The cloud edge now carries 40 K
  across a pixel, so the interpolation error at stride 4 was 1.39 K at p99 and 4.6 K at worst;
  at stride 2 it is 0.46 K and 2.2 K. The box filter over the supersamples barely helps, because
  bilinear error over a 4 × 4 block is correlated and does not average away. Sizing the march by
  path length made this affordable: a 20° ray now takes 98 steps where it took 256.

### What this does not fix

* **A cloud is still a vertical extrusion.** `density_at` is a function of the column depth and
  the height within it, so a cloud is a *dome standing on the base plane* — widest at the bottom,
  tapering upward. A real cumulus bulges above its base. A single compact column, marched, comes
  out as an elliptical skirt with a tower over it, which is recognisably a cloud and is not a
  cauliflower. Genuine 3-D shape needs the volume in the renderer: **AT.13**, then **AT.14**.
* **The visible cloud has no internal shading.** `_cloud_base` is one Lambertian radiance for
  every cloudy texel, so now that α saturates over the body of a cloud the body is flat. Edges
  still gradate. A two-stream reflectance in τ would be one line and is deliberately *not* taken
  here: what a ground observer sees at 20° is mostly sunlit cloud *sides*, and the geometry that
  decides how much of a ray's path was lit is the same geometry AT.14 is about. Recorded rather
  than guessed.
* **Every base is at one altitude.** Real cumulus bases vary by ±50–150 m across a field, and a
  perfectly flat base is a degenerate geometry for a near-horizontal ray, which skims it for
  kilometres. This is part of why the frame's lower third is a continuous wall.
* **No advection, no cloud shadow on the target, no cloud occluding the aircraft in LWIR.**
  Unchanged from ADR 0127.

## Alternatives considered

**Leave the infrared band marching and the dome sampling.** Rejected: it is the disagreement the
owner reported, and ADR 0076 is explicit that the two bands may not place a cloud differently.

**Make the infrared band sample the way the dome did**, so the two agree cheaply. Rejected: the
sampled quantity is a *vertical* thickness read for an *oblique* look. It is the wrong one, and
adopting it would have thrown away the 19.7 K of structure AT.12 was built for.

**Keep the mesa field and tame the smear by truncating the march.** Rejected: there is no physical
length at which to truncate a slant path, so the parameter would have been fitted to the picture.

**Apply the band limit by synthesising the field on a coarser grid.** Equivalent in effect, but it
would decouple the depth map's grid from the voxel grid the visible band wants, and `density_at`
is deliberately the one definition both bands read.
