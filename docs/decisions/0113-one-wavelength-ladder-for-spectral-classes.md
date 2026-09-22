# ADR 0113 — One wavelength ladder for the atmosphere's spectral classes

**Status:** Accepted. Pays the debt [ADR 0092](0092-band-scalability-guard-and-the-atmosphere-carve-out.md) recorded as the
one carve-out that was neither physics nor a schema constraint; extends ADR 0071's model without
changing its calibration.
**Date:** 2026-09-22
Roadmap: AT.10 (§7.1, §7.4, §12.1)

## Context

[ADR 0071](0071-layered-slant-path-atmosphere.md) splits each band into **spectral classes** — a
window, water-vapour lines, a band-edge region, an opaque CO₂ core — and gives each its own
extinction, so the band transmittance is a sum of exponentials rather than one grey γ. The classes
lived in `BAND_CLASSES`, a dict keyed by **camera band name** with five hand-written tables.

That is spectroscopy filed under camera names, and it had the three faults filing always has.

**Two tables disagreed about the same air.** NIR resolved the 0.94 µm water band as its own class
at ×10; SWIR's window ran 0.80–1.10 µm at ×0.5 and treated the same band as clear. A scene rendered
in both bands described two different atmospheres, a factor of twenty apart, over one sky. `AT.4`
measured it and left it; this ADR is the fix.

**Two stretches of spectrum belonged to no class at all**, 1.80–2.00 and 6.00–7.00 µm, which are
the 1.9 and 6.3 µm water bands. `class_weights` raises on weight outside every class, so a future
camera reaching either would have failed at load — and the only reason nobody hit it is that the
four shipped responses stop exactly short of both.

**A fifth band meant a sixth table in `src/`**, which is the thing CLAUDE.md's "bands are data, not
code" exists to prevent. `ADR 0092` recorded this as the last of six band-aware carve-outs and the
only one that was not a fact about the world or a field name already written into YAML.

## Decision

One `ATMOSPHERE_LADDER`: a **wavelength-ordered, gap-free** sequence of rungs from 0.35 to 14.5 µm,
each a contiguous interval with one extinction behaviour. A band's classes are **derived** by
`classes_for(band, response)`, which takes every rung the band's span overlaps, whole, and merges
rungs of the same name into one class with several intervals — the shape `SpectralClass` already
had. The ladder's ordering and gap-freeness is checked at import, because a gap is a band that
raises at the far end of a render.

**A band's span is its nominal range *and* its response**, not either alone. The nominal range
alone was `AT.3`'s defect: the NIR classes stopped at 1.05 µm while `nir_si.csv` reaches 1.10,
carrying 0.188 % of the band outside every class. The response alone would let a camera with a
narrow filter shrink the model of the air it looks through.

**Overlap is half-open** (`lo < span_hi and hi > span_lo`), so a band ending exactly on a rung
boundary does not acquire the rung beyond it — LWIR's response stops at 7.00 µm and does not pick
up the 6.3 µm water band, which carries none of its light. The class set's own **top** edge is
closed instead, or a response ending exactly there (a nominal top-hat does, at full height) would
fall outside every class.

**A band reaching past the ladder is refused**, with the reason, rather than given the nearest
class. The numbers outside are spectroscopy nobody has written down here, and a nearest-class
fallback would look exactly like a model.

## Consequences

**The calibrated bands are untouched.** LWIR and NIR come out **bit-identical** — the ladder
reproduces their hand-written tables exactly, which is the property that let this ship at all,
since ADR 0071's window multipliers are fitted to R13's Tucson sky and any drift there is a change
to a measured anchor. MWIR is identical to **one ulp**: its classes now come back in wavelength
order rather than the order someone typed them, so the sum runs in a different order. `visible`
moves by 1.5 × 10⁻¹⁴ relative, which is the anchor solver's own tolerance: it has one non-opaque
class, so its multiplier is entirely absorbed by the 200 m anchor and any positive value gives the
same air.

**SWIR changes, on purpose, and this is the point.** 0.90–0.98 µm moves from `window` (×0.5) to
`h2o_0p94` (×10): **16.9 %** of the Planck-weighted InGaAs band leaves the window, and the band
loses **6.5 %** of its transmittance at 5 km (0.4148 → 0.3877). The **200 m anchor is exact**,
because the anchor solve re-fits the free classes to the grey preset at exactly that distance —
which is why no test caught this, and why the error grows with range: it is invisible where the
model is pinned and largest where it is extrapolated.

**Nothing in the golden store moves.** All eight golden arrays come from one Boson LWIR config
(`GT.2` is the row that fixes that), and LWIR is bit-identical, so the promised golden update
turned out to be no update at all. That is luck rather than design, and it is an argument for
`GT.2` rather than against this change.

**The class *set* now depends on the response**, where it used to depend only on the band. A
`LayeredAtmosphere` built without responses gets a narrower set derived from the nominal range, and
the weights vector is a different length than the same band's with a real detector. That is
consistent — the no-response path already uses a nominal top-hat for the weights, and already warns
loudly (`AT.2`) — but it means a caller comparing the two must compare transmittance, not weight
vectors. The measured MWIR `h2o_wing` figure moves with it: the nominal top-hat over 3.0–5.0 µm
does not reach the 5.0–6.0 µm half of that class at all, so the nominal weight reads 0.0192 where
the hand-written table handed it 0.0238 regardless of whether the camera could see there. Against
the real response's 0.1303 that is a **6.8×** gap rather than 5.5×.

**The carve-out is down from seven offences to two.** What is left in
`atmosphere/layered.py` is `gamma_aerosol_visible` and `aerosol_ratio_to_visible`, which is the
same photopic definition `atmosphere/extinction.py` carries: meteorological optical range *is*
defined against the eye, and the ratio has to name what it is taken against. A test now asserts
that **no carve-out enumerates the band registry**, which is the property that made `BAND_CLASSES`
a second registry whatever it was called.

**The multipliers are still ESTIMATED.** This ADR moves where the spectroscopy lives and makes it
consistent; it does not measure it. The window values remain ADR 0071's R13 calibration and
everything else remains an estimate, including the two new rungs — the 1.9 and 6.3 µm bands are
modelled opaque on the same footing as the 1.4 and 2.7 µm bands, which is right in kind and
unmeasured in degree.

## Revisit when

A band needs spectroscopy outside 0.35–14.5 µm — VLWIR, or the sub-millimetre — at which point the
ladder is extended rather than a table added, and the extension is the place to ask whether the
multipliers deserve a line-by-line source; or `PH.5`'s HITEMP tables arrive, which would give the
hot-gas path real absorption coefficients and make the estimated multipliers here look like what
they are.
