# ADR 0121 — A leaf is solved, not stepped, and has its own boundary layer

**Status:** Accepted. Uses the latent term [ADR 0101](0101-the-latent-term-and-the-film.md) added and
the stability guard [ADR 0094](0094-imex-for-conduction.md) introduced for `TC.1`; adds the first
surface in this project whose boundary layer is *not* the bulk aerodynamic one.
**Date:** 2026-09-22
Roadmap: PH.11 (§6.1)

## Context

A leaf is the opposite of every other surface in the library. Its areal heat capacity is
**630 J m⁻² K⁻¹** against 63 000 for a snowpack and 180 000 for asphalt, so it has no thermal
memory; and it controls its own evaporation through a stomatal resistance that is a property of
the plant, not of the weather.

That is worth modelling because a well-watered canopy reads **below** air temperature on a hot dry
afternoon and a stressed one reads well above it, under identical weather. Every crop water stress
index ever flown rests on that difference.

## Decision

Three decisions, and the second is the one that nearly went wrong.

### 1. Solve the root, do not step it

`leaf_steady_state_k` bisects `net_flux(T) = 0` rather than integrating. Measured: with the leaf's
own boundary layer at 2 m/s, `h = 34.8 W m⁻² K⁻¹`, the time constant is **15.4 s** and the
midpoint rule is stable only to **30.8 s**. A scene ticking at 60 s would oscillate and diverge.
A snowpack's limit under the same forcing is 3089 s and asphalt's 8825 s, so the leaf is two
orders of magnitude more restrictive than anything else here.

A surface with a 15 s memory has nothing to remember across a minute-long tick, so its steady
state *is* its answer — the same argument `TC.1` made for a conduction ladder whose fast mode sits
below the tick. Bisection rather than Newton because the balance is monotone decreasing in T
(emission, convection and evaporation all rise with it), so it cannot fail, and the derivative
Newton would need is exactly what a future latent term would make ugly.

### 2. A leaf's boundary layer is not the project's bulk one

This is the part that produced the wrong physics before it was caught. `bulk_conductance_kg_m2_s`
describes a metres-deep surface layer over soil or sea and gives an aerodynamic resistance of
**435 s/m** at 2 m/s. A 5 cm leaf in the same wind has **33 s/m** — thirteen times smaller,
because the air only has to cross five centimetres.

Built on the bulk value, a well-watered leaf could not transpire fast enough to go below air at
all: it read **+6.4 K** at a 3.4 kPa deficit instead of −1.9 K. That is the wrong *sign* of the
only effect PH.11 exists to produce, and it would have passed anything except a test that asked
for the sign. `leaf_boundary_conductance_mol_m2_s` implements Campbell & Norman's
`g_Ha = 1.4 · 0.135 √(u/d)` instead, and `leaf_forcing` derives both `h` and `g_e` from that one
conductance so the sensible and latent paths cannot describe different turbulence.

### 3. The oracle is written in different units on purpose

`leaf_temperature_cn` implements Campbell & Norman's closed form (2nd ed., §14.5) in **molar**
units — conductances in mol m⁻² s⁻¹, vapour as a mole fraction, γ = c_p/λ — while this project's
balance is SI mass-based with `E = ρ_a Δq/(r_a + r_s)`. Neither is derived from the other, so
agreement is evidence rather than algebra. The bridge is the two molar masses and nothing else.

The molar constants are **derived from this project's own SI constants rather than pasted from the
book**, which keeps the oracle tied to the balance it checks and costs a visible gap:
`c_p,molar = 29.105` against their 29.3 (0.67 %, this project carries dry-air c_p = 1005 where
they use a moist-air value) and `γ = 6.594e-4` against their 6.66e-4 (0.99 %). The test asserts
the derived values and names the published ones, because a tolerance loosened until 29.105 counted
as 29.3 would hide the only thing there worth knowing.

## Consequences

**The four acceptance criteria, measured** at 30 °C, 2 m/s, 800 W/m²:

| check | result |
|---|---|
| closing the stomata → dry reference | +7.48 K, exactly, and monotone in r_s |
| well-watered leaf at 3.18 kPa deficit | **−1.93 K** (stressed: **+7.00 K**) |
| Idso slope | **−1.84 °C/kPa**, inside the published [−3.8, −1.1] |
| Campbell & Norman vs this balance | **0.10 K** near air, **0.37 K** at 4.6 K departure |

**The fourth criterion holds to 0.1 K only near air temperature, and that is stated rather than
quoted selectively.** Two causes, separated by measurement. The ~0.1 K floor is the unit bridge:
this project's specific humidity uses the exact `q = ε e/(p − (1−ε)e)` while the molar side's
`Δq = (M_w/M_a) Δw` is the `q ≈ ε e/p` form, worth 1.6 % of the deficit at 30 °C. The growth with
departure is Campbell & Norman's own double linearisation — of the saturation curve *and* of σT⁴,
both about air temperature — which is what a closed form buys its closed form with.

Where both vanish (no net isothermal radiation, saturated air) the two agree to **4e-10 K**, the
bisection tolerance. That degenerate case is the test that says the formulations are the same
equation; the finite ones say how far their approximations drift.

**The Idso band is an acceptance, not a fit.** `IDSO_SLOPE_C_PER_KPA` records the published
non-water-stressed baseline and the model is asked whether it lands inside. Nothing in the module
was tuned to it — the slope falls out of the stomatal resistance, the boundary layer and the
saturation curve, and a model tuned to the band would have proved nothing.

**Not done, and each would change a number above.** Stomatal resistance is a constant per regime,
not a function of light, humidity or leaf water potential, so a modelled canopy cannot close its
stomata as the day dries — `WELL_WATERED_R_S_S_M` and `STRESSED_R_S_S_M` are ESTIMATED brackets,
not a species. There is no canopy: one leaf, one temperature, no sunlit/shaded separation and no
radiation trapping between layers, which is what a real thermal image of a canopy is mostly made
of. `LEAF_WIDTH_M` is a single ESTIMATED 5 cm. And no material carries a stomatal resistance and
no scene declares vegetation, so like `PH.7`'s fire and `PH.10`'s snow this ships as physics with
its tests and no scene-schema surface.
