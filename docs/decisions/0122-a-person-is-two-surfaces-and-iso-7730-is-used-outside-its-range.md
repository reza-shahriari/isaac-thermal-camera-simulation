# ADR 0122 — A person is two surfaces, and ISO 7730 is used outside its range

**Status:** Accepted. Adds the first surface in this project whose temperature is **authored from a
set point rather than solved**, alongside one that is solved by a standard's own equation rather
than by [ADR 0036](0036-rk2-for-the-surface-balance.md)'s balance.
**Date:** 2026-09-22
Roadmap: PH.12 (§6.1, §16.2)

## Context

A person is the most common target an infrared camera is pointed at, and a single temperature per
person gets the most important thing about them wrong. Measured at 0 °C in still air with 1 clo:
skin **34.07 °C**, clothing **13.96 °C** — a **20.1 K step across one body**, four hundred times a
50 mK NETD. That is why a face and hands are the brightest things in a winter street scene while
the coat between them is nearly background.

Indoors at 22 °C in 0.5 clo the same person's step is **4.9 K**. How much of a person stands out is
a property of the weather, not of the person, and a one-temperature human renders both identically.

## Decision

**Two models, because they are two problems.** Skin is thermoregulated and is *authored* from
`t_sk = 35.7 − 0.028 (M − W)` °C. That is the physics, not a shortcut: a body holds its skin near
this value across a wide range of weather by varying blood flow and sweating, so a passive energy
balance on skin would answer a question the body does not ask. (It also falls as work rises —
34.07 °C seated, 32.44 °C walking — because more of the heat leaves by sweat than by the surface.
A model that raised skin temperature with activity would look sensible and be wrong.)

Clothing is passive and *is* solved, from ISO 7730's own implicit balance rather than from
`irsim.thermal.balance`. The standard's coefficients — 3.96e-8 for the radiative term, the
1.290/0.645 area factors, the 2.38/12.1 convection pair — were fitted together as a set.
Re-deriving the equation from this project's balance would produce something close and comparable
to nothing published, and the point of modelling a person is that published numbers exist for them.

**Solved by bisection, not by the textbook fixed point.** ISO 7730 is usually written as an
iteration. An undamped one **diverges for a coat in wind**: at 1 m/s the quartic and the `h_c`
term together push the map's slope past −1, and damping by a half only postpones it — that failure
was hit during implementation, at ordinary wind speeds, not at an extreme. The residual is
strictly decreasing in `t_cl`, so a bracketed root cannot fail.

## Consequences

### PH.12's third criterion does not hold at the condition it names, and should not

The row expects 1 clo at 0 °C to land the clothing 10–15 °C below skin. ISO 7730's equation gives
**20.1 K** there. The band the row names is what the *same equation* produces at **10–15 °C of
air** — which is exactly ISO 7730 Annex A's validity floor:

| air | step at 1 clo |
|---|---|
| 0 °C | **20.1 K** |
| 5 °C | 17.1 K |
| 10 °C | **14.2 K** ← in band |
| 15 °C | **11.2 K** ← in band |
| 20 °C | 8.2 K |

So the criterion was written for an in-range condition and then quoted at an out-of-range one. The
test records the measured 20.1 K and asserts *why* the band belongs to 10–15 °C, rather than
loosening the band until 20.1 fits — which would have hidden a correct implementation behind a
tolerance and invited someone later to "fix" it.

**This module uses an indoor comfort standard outdoors, deliberately.** Every winter figure here
is past Annex A's 10–30 °C floor. Nothing else published gives a clothing surface temperature, so
it is the right extrapolation; but the numbers below 10 °C are the equation's, not the standard's,
and both the module and the tests say so.

### Wind does not always cool the coat

PH.12's second criterion is "more wind lowers t_cl", and that is the special case. Wind couples
the surface to the **air**, so it moves it toward air temperature from whichever side it is on. At
−10 °C air under a −40 °C sky the coat is above air at 1 and 2 clo and wind cools it; by **3 clo**
the radiative loss has dragged it *below* air (−10.31 °C) and the same wind pushes it **up**.

Only a regime sweep finds this. A monotone-cooling test passes at every insulation a person
actually wears and hides the sign change beyond it — and the first version of the test asserted
the one-way version and failed, which is how it was found.

### The fourth criterion is not met, and the gap is declared

PH.12 asks for agreement with `pythermalcomfort`'s two-node model to 0.5 K. The package is not
installed and is not a dependency. Rather than drop the criterion:

* a `comfort` optional extra declares it in `pyproject.toml`, following the `validation` and `ml`
  precedent, so the check is installable with `pip install -e '.[comfort]'`;
* the comparison test uses `importorskip` and **skips loudly** with that instruction;
* a second test asserts the extra exists *and* that nothing under `src/irsim` imports it, so the
  dev-only-oracle rule is enforced rather than remembered.

A skipped test that nobody can see is worse than a missing one, so the gap is recorded in three
places rather than left to a green run to imply.

### Not modelled

No sweating, shivering, skin blood flow, radiant asymmetry or posture — those belong to a two-node
comfort model, and this is the surface a camera sees at a steady state, which is a smaller
question. The effective radiation area factor of 0.72 is folded into the standard's 3.96e-8 and
cannot be varied for a seated versus standing person. And, as with `PH.7`'s fire, `PH.10`'s snow
and `PH.11`'s leaf, no scene declares a human: binding two patches to one prim is scene authoring
on machinery that already exists, and it is not this step.
