# ADR 0115 — Fire: authored by what it emits, and the air above it is not the weather's

**Status:** Accepted. Adds the two terms a fire puts into §6.1's balance; the flame's *image* stays
[ADR 0098](0098-participating-media-and-the-phenomena-tier.md)'s slab, placed by
[ADR 0114](0114-the-exhaust-plume-as-a-chord-per-pixel.md)'s injector.
**Date:** 2026-09-22
Roadmap: PH.7 (§6.1, §6.6)

## Context

`PH.4`–`PH.6` built what a camera sees of hot gas. A fire also *heats things*, and the two routes
it takes to a surface — radiation from the flame, convection to the plume above it — are not in
the model at all. §6.6 scripts vehicle heat sources as load-driven schedules; neither applies.

Both have a plausible wrong answer that would produce numbers nobody could challenge from a
single frame, which is what makes them worth an ADR rather than a docstring.

## Options considered

**A hot rectangle at a flame temperature, radiating `ε σ T⁴`.** What
[ADR 0088](0088-spatial-heat-sources-and-sky-occlusion.md) already does for an engine bay, and it
would need no new code at all. Rejected: a flame's temperature and emissivity are **not separately
knowable**. 1200 K at ε = 1 is 118 kW/m², and so is 1500 K at ε = 0.41. Authoring a temperature
therefore amounts to authoring an emissivity nobody measured, and the pair is under-determined in
exactly the direction a scene author would guess wrong — upward, because a flame looks hot.

**A view-factor model with the flame as a cylinder.** More faithful geometry than a rectangle, and
the configuration factors are published. Deferred rather than rejected: ADR 0088's parallel-plane
closed form is already in the repository, already clamped by
[ADR 0090](0090-clamped-view-factors-for-nested-radiators.md), and a rectangle standing beside a
wall is the geometry most fire-protection separation calculations use. A cylinder changes `F`, not
the term `F` multiplies.

**Ambient air above the fire.** The default if nothing is done, and wrong by hundreds of kelvin:
a facet over a fire convects to the plume. Rejected on sight; recorded because it is what every
scene did until now.

## Decision

**Radiation: a flame carries a surface emissive power, not a temperature.**
`RadiantRectangle.sep_w_m2` is the authored product, and `emitted_flux_w_m2` answers to one of the
two — a rectangle that carries a SEP **refuses** a temperature rather than preferring one, because
the two numbers disagree by design and silently picking would make which is used an implementation
detail. Values are the literature's ranges and are marked ESTIMATED to them: **135 kW/m²** for the
luminous unobscured flame (Mudan, Considine: 100–170) and **40 kW/m²** for the smoke-obscured
upper region (30–50), which is most of a large pool fire's height and the reason a bigger fire is
not proportionally worse to stand beside.

**The absorbed flux uses two different coefficients, deliberately.**

    q_int = F · (α · SEP − ε · L_occluded)

`α` is the surface's absorptivity **for the flame's spectrum**, which peaks between 1 and 5 µm;
`ε` is its own long-wave emissivity, and only the sky the flame blocks is weighted by it. They
coincide only for a grey surface, and a painted one is not. Using one number for both is the easy
mistake and it errs in the direction that matters: it under-predicts what the fire delivers. The
occlusion term defaults to nothing, which is right for the common geometry — a flame stands
*beside* a wall, not over it — and is available for when it is not.

**Convection: Heskestad's centreline, held at the tip below the flame height.**

    ΔT₀ = 9.1 (T_∞ / (g c_p² ρ_∞²))^(1/3) · Q_c^(2/3) · (z − z₀)^(−5/3)

with `Q_c` in kW, `z₀ = 0.083 Q^(2/5) − 1.02 D` and the mean flame height
`L = 0.235 Q^(2/5) − 1.02 D`. The correlation describes the buoyant plume *above* the flame; inside
the continuous flame there is no `z^(−5/3)` decay to describe, so below `L` the value is held at
the tip's. That is not smoothing. It makes the profile continuous at the tip by construction, and
the tip value then falls out **independent of Q and D** — the `Q_c^(2/3)` and `(L − z₀)^(−5/3)`
dependences cancel exactly — at **452.7 K** above ambient, which is Heskestad's own statement of
what the mean flame height means. Measured across a 0.3 m gas ring at 50 kW and a 6 m pool at
40 MW: the same number to a tenth of a kelvin. It is the cheapest transcription check the module
has and it is in the suite.

`Q_c` is authored in **kilowatts** and a value above 100 MW is refused as watts — no pool fire
reaches there, and a 1 MW room fire mistakenly given in watts arrives as 700 000, producing a
plume a hundred times too strong that nothing downstream would question.

**Off the axis, one estimated number.** The excess falls as a Gaussian of half-width
`0.13 (z − z₀)`, never narrower than the pool that feeds it. The velocity profile's spread is
about 0.12 and the temperature profile is some 10 % wider (Heskestad, SFPE Handbook). This is the
only fitted quantity in the module; it sets how fast the plume's heat falls away from the axis and
never the centreline value the tests pin.

**The flame a camera sees is solved from the flame a wall feels.** `flame_plume` builds ADR 0098's
soot slab on a cone standing on the pool, and its **mixing length is solved, not authored**, so the
slab reaches exactly the centreline excess Heskestad gives at the tip. One model, two consumers,
and a scene cannot set them apart. Soot only: no gas table is consulted, because a luminous flame's
own emission *is* its soot.

## Consequences

**Easy now:** a wall beside a fire, a ceiling above one, and a flame in frame, from one `PoolFire`.
`PH.8` takes the frame this produces into the ISP's gain state and AGC.

**Measured, and worth carrying forward:** soot is grey-ish and gas is not, as a ratio rather than a
claim. Between 3–5 µm and 7.5–13.5 µm the soot band means differ by **2.47**, which is just the
ratio of the bands' emission-weighted `⟨1/λ⟩`; hot CO₂ over the same pair differs by **5.49**,
because its absorption is in lines and the lines are in one band. So a camera that cannot see an
exhaust plume can still see a fire — and that is why `PH.7` needs no gas table where `PH.6` could
not exist without one.

**Error introduced, bounded where it can be:** a rectangle rather than a cylinder for `F`; one
Gaussian width, ESTIMATED; the SEP ranges are ranges, so a scene that cares should author its own;
and Heskestad's correlation is for a free axisymmetric plume, so a fire against a wall or under a
ceiling entrains differently than this says.

**Deliberately not modelled:** wind tilt and a plume that is not vertical; flame pulsation;
radiation from the smoke layer as something distinct from the flame; and any feedback from the
heated surface back to the fire. Each changes the geometry rather than these two terms. A fire is
also not yet authorable in a scene file — `PH.7`'s acceptance is at cell level and that is where it
is met; a `fire:` block is a schema step of its own.

## Revisit when

A validation needs the flux on a surface close to a large fire, where the rectangle's `F` and the
free-plume entrainment both stop describing the geometry; or a scene needs several fires, or one
in wind, at which point the plume is no longer axisymmetric and the centreline is no longer where
the correlation puts it.
