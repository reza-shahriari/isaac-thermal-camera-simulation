# ADR 0135 — Steam is droplets, and Mie is computed rather than tabulated

**Status:** Accepted
**Date:** 2026-09-23

## Context

`PH.9` asks for steam and droplet plumes: "the slab of `PH.4` with a droplet extinction from
`cloud.py`'s Mie tables scaled by liquid water content, emitting at the droplet temperature."

Two of that row's premises did not survive contact with the repository.

**There are no Mie tables.** `irsim.atmosphere.cloud` carries a single measured band ratio,
`CLOUD_OD_RATIO`, and its own comment says in as many words that "deriving the ratio per band from
droplet Mie theory is not in scope; measuring it is." Nothing in `src/` or `data/` computes or
stores a Mie efficiency. The row was sized **S** on the assumption that the hard part already
existed.

**The verification criterion is under-specified.** "A pure-gas H₂O slab at 373 K has τ_LWIR ≥ 0.9
(NIRATAM)" names no path length, and the path is what decides it. Measured against `PH.5`'s own
tables at a full atmosphere of H₂O — saturated *pure* steam, the most opaque vapour there can be —
τ_LWIR is 0.879 over half a metre, 0.844 over one metre and 0.767 over five. The criterion as
written is true only inside about 40 cm.

What the repository does have is `data/nk/water.csv`: Segelstein's complex refractive index of
liquid water over 0.65–15.6 µm, which is the *input* Mie theory needs and which covers both
infrared bands with room to spare.

## Options considered

**Use the geometric limit `Q_ext = 2`.** Free, and wrong in the band that matters. `Q_ext` reaches
2 only for size parameters well above ten; a 2 µm droplet is `x = 3.1` against MWIR and `x = 1.2`
against LWIR, and the measured efficiencies there are 1.64 and 0.40. Assuming 2 would answer
`PH.9`'s band-comparison question with the assumption, and would overstate LWIR extinction fivefold
for fine droplets.

**Scale `cloud.py`'s single band ratio.** It is a measured visible-to-infrared absorption ratio for
a cloud, not an extinction efficiency, and it carries no droplet size — the one parameter that
decides the answer. It also cannot produce a MWIR-versus-LWIR difference, which is what was asked.

**Compute Mie from the refractive index the repository already carries. (chosen)**

## Decision

`irsim.atmosphere.mie` implements Bohren & Huffman's `BHMIE` (1983, §4.8) in NumPy — no SciPy, as
`irsim.optics.mtf` does for `J₁` — with the **downward** recursion for the logarithmic derivative
`Dₙ(mx)`. The direction is not a detail: upward is unstable once `k` is appreciable, and for water
in LWIR it is (`k = 0.05` at 10 µm, `0.20` at 12 µm), so the obvious implementation passes a
transparent test case and fails the case this module exists for.

`irsim.atmosphere.droplets` band-averages `Q_ext` over the camera's own response weighted by the
Planck emission of the droplets — the same weighting `soot_band_kappa_per_m` uses — and converts to
an extinction coefficient through the standard cloud-optics identity

    β_ext = 3 · LWC · Q_ext / (4 ρ_water r)

which follows from `β = N π r² Q_ext` and `LWC = N (4/3) π r³ ρ_water` by eliminating `N`.

`GasSlab` gains `lwc_kg_m3` and `droplet_radius_um`. The droplets are taken to be at the slab's own
temperature, which is right for condensing steam — droplet and vapour are in equilibrium at the
saturation temperature — and is the assumption to revisit for a spray injected into hot gas.

**The oracle is the exact Rayleigh limit, not a published table value.** `Q_sca = (8/3)x⁴|K|²` and
`Q_abs = 4x·Im K` with `K = (m²−1)/(m²+2)` are closed form as `x → 0`, and they pin the `a₁`/`b₁`
coefficients and the `2/x²` normalisation — between them, everything the series is built from.
Matched to 2e-4 relative including an absorbing case. A half-remembered book value was deliberately
not used as the check.

## Consequences

**The criterion is restated as the comparison it was standing in for.** Not "τ_LWIR ≥ 0.9 for pure
gas", which depends on an unnamed path, but: at a realistic plume's water loading the condensed
phase dominates the vapour several times over. Measured at 373 K over one metre in LWIR — pure
saturated vapour leaves an optical depth of **0.169**, and 5 g/m³ of 5 µm droplets leaves
**1.02**, six times as much. The two are equal at **0.83 g/m³**, which is the number a scene author
needs and which the row did not have.

**`PH.9`'s band criterion is true, and only for small droplets.** At 1 g/m³, MWIR extinction
exceeds LWIR by **4.1× at 2 µm** and **2.5× at 5 µm** — but by 10 µm the ratio is 0.92 and by 20 µm
it is 0.99, because both bands are then past `x = 6` and `Q_ext` has settled near 2 in each. Fresh
condensate is in the first regime, so the criterion is the right one to state; a reader who took it
as "MWIR is always more opaque" would size a coarse plume's contrast wrongly, so the convergence is
asserted in its own test rather than left to be discovered.

**Extinction goes as 1/r at fixed water content.** The same kilogram of water is *more* opaque
spread over smaller droplets, so a plume does not clear as it condenses further. This inverts the
intuition that bigger droplets mean a thicker plume and is tested directly.

**A droplet-only slab is allowed below the 300 K gas floor,** down to 273.15 K. That floor exists
because `PH.5`'s absorption tables start at 300 K, and a slab carrying only condensed water needs
no such table — fog at 285 K was being refused for a reason that did not apply to it. The new floor
is freezing, below which the scatterers are ice: a different refractive index and not a sphere, so
Mie on liquid water stops applying.

**Cost.** One Mie series per quadrature node per call, cached only by the `lru_cache` on the
refractive-index table. A band average is a few hundred series of a few dozen terms — microseconds,
and it is called once per slab rather than per pixel. If a scene ever varies droplet radius per
pixel, this becomes a table lookup and the table is generated from this code.
