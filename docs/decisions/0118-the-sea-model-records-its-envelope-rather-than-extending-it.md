# ADR 0118 — The sea model records its validity envelope rather than extending it

**Status:** Accepted. Qualifies [ADR 0078](0078-maritime-phase-and-analytic-sea.md) and
[ADR 0079](0079-sea-water-optical-constants-and-slope-model.md): they chose the facet-integrated
Fresnel sea and recorded what it approximates, without saying over what range of angles anyone has
checked it against a sea.
**Date:** 2026-09-22
Roadmap: SE.1 (§5.3)

## Context

`irsim.atmosphere.sea.SeaModel` computes a band-effective emissivity for any view direction, from
nadir to grazing, by integrating the Fresnel emissivity of water over a Cox-Munk slope
distribution. That is a Masuda-class model, and it answers everywhere.

Published in-situ angular radiometry does not. A CE 312 four-channel radiometer over
buoy-instrumented sea, in the 8-14 µm interval, validates the model only to **about 50° from
nadir**; beyond that, multiple reflections between facets of the roughened surface produce
discrepancies and the Wu-Smith model is required for 8-13 µm. Emissivity falls roughly 2-3 % by a
55° view angle. (`docs/research/2026-09-15-field-survey.md`, medium confidence.)

The gap between those two sentences is not academic for this project. A camera on a shore or a
mast — the maritime deployment the lane is being built for — sees the sea at grazing incidence
across essentially its whole frame. Measured on real Boson geometry:

| frame | view zenith | beyond 50° |
|---|---|---|
| shore camera, 20 m, boresight −1° | 76.6–89.9° | **1.0000** |
| shore camera, 20 m, boresight −20° | 57.6–82.6° | **1.0000** |
| mast camera, 100 m, boresight −5° | 72.7–89.8° | **1.0000** |
| UAV, 2 km, boresight −60° | 17.7–44.5° | 0.0000 |
| UAV, 2 km, boresight −90° | 0.0–19.3° | 0.0000 |

At a 20 m eye height the 50° limit is crossed at **31 m of slant range**. Everything past a boat
length is extrapolation.

The obvious reading of "the sea model is validated to 50°" is that this is a caveat to note. It is
not: for the deployment in question it is the entire operating band, and the model was shipped
without anything recording that.

## Decision

**Record the envelope; do not extend it, and do not gate on it.**

A new module `irsim.atmosphere.sea_envelope` carries the limit, its source, and the geometry that
converts a camera depression into the angle the limit is expressed in. `SeaModel` gains
`view_zenith_rad`, `beyond_envelope` and `envelope_report`; `MaritimeScene` carries an
`EnvelopeReport` in its metadata and recomputes it on demand; the maritime Tier 3 report prints
the frame fraction beyond.

Three sub-decisions, each of which could have gone the other way:

**1. The envelope is indexed by view zenith at the surface, not by camera depression.** These
differ by the horizon dip, and `sin θ = (1 + h/R) cos δ` is used rather than `90° − δ` so that
θ is *exactly* 90° at the geometric horizon for any camera height. The flat-earth form is short by
0.14° at 20 m and by 3.2° at 10 km, i.e. it errs precisely in the quantity the envelope is about.

This axis is pinned by its own test because the wrong axis does not fail loudly — it fails
**backwards**. Reading the same 50° against depression reports the shore frame as 0.000 beyond
(fully validated, when none of it is) and the oblique airborne frame as 0.867 beyond (when all of
it is inside). A wrong-axis implementation would certify the one deployment that is entirely
extrapolated and warn about the one that is not.

**2. The flag is a record, not a gate.** `beyond_envelope` answers a question; nothing refuses to
render. Refusing would disable the maritime lane outright, since it has almost no in-envelope
pixels, and would trade a known and measured extrapolation for no capability at all. The model
keeps answering past 50° and the answer is now labelled.

**3. Wu-Smith is not implemented.** The survey gives one sentence about it — that it is required
for 8-13 µm past 50° — and no coefficients, no functional form, and no error bound. Implementing
a named model from a one-line claim would produce something that looks like an improvement and
whose agreement with reality is *less* knowable than the current model's, because the current
model's disagreement is at least attributable to a stated missing mechanism. Closing this properly
needs the Wu & Smith paper and a digitised comparison; that is future work, not this step.

## Consequences

**A finding the roadmap row did not anticipate: the envelope has two axes, not one.** The
published 2-3 % drop at 55° is a measurement *of a sea*, so it carries that sea's roughness. Our
facet-integrated drop at 55° is:

| wind | 0 | 2 | 5 | 10 | 15 m/s |
|---|---|---|---|---|---|
| drop at 55° | 2.07 % | 2.29 % | 2.67 % | 3.43 % | 4.33 % |

It reproduces the published range for winds up to **7.3 m/s** and exceeds 3 % above that — at an
angle the published range is supposed to cover. So a rough-sea frame at 45° is outside the
published figure even though it is inside the published angle. `VALIDATED_WIND_M_S` carries this,
and it is the only thing in the module that is measured here rather than cited.

**This is the first external check on the angular emissivity.** The 2.07 % calm-sea drop landing
inside the published 2-3 % is the only comparison in the repo between this chain's angular
behaviour and a measurement of the sea. Everything else about the sea model is self-consistency.

**The isothermal identity is explicitly demoted for this purpose.** It is the load-bearing test in
`test_sea_surface.py` because it holds for *any* emissivity — which is exactly why it is blind to
the question SE.1 asks. `test_the_isothermal_identity_cannot_see_the_envelope` demonstrates this
rather than asserting it: replacing ε(θ) with a flat 0.5, wrong by up to 0.49, leaves the identity
satisfied to the same 1 mK. Anyone tempted to treat the identity as evidence that the sea is
right at grazing should read that test first.

**What is now known and still unfixed.** The maritime lane's Tier 3 findings — the 2.9 K
near-horizon inversion, the cold trough a few degrees down, the 57 % overcast collapse — are all
measured in a band with no published backing. They are not thereby wrong; they are unverified,
and the report now says so on every run instead of leaving a reader to infer it. `SE.3`
(ECOSTRESS SST) is the nearest thing to an external check the lane has planned, and it constrains
the *temperature*, not the angular emissivity.

**Not addressed here.** Wave shadowing and inter-facet reflection remain unbounded (ADR 0078
already records this, and it is the same physics the 50° limit is about). The slope distribution
stays isotropic. The envelope is recorded for LWIR only, because the published figure is an
8-14 µm one and the MWIR sea has not been looked for.
