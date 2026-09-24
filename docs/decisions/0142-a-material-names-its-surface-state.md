# 0142 — A material file names its surface state, and the field has no default

Date: 2026-09-24
**Status:** Accepted
Roadmap: AT.17 (§4.5)

## Context

§4.5 is unambiguous: emissivity is a property of the *surface*, not of the substance under it. One
measurement campaign on aluminium window profiles reports **0.834–0.856** normal total emissivity
for anodised exterior surfaces and **0.055–0.82** across untreated all-aluminium cavities in the
same frames [R39] — one metal, one paper, a fifteen-fold spread. Polished aluminium is 0.04 at
8 µm; rough wrought iron is 0.94 against 0.28 polished [R47]. Oxidation dominates and roughness
comes second.

The library did not carry that. `configs/materials/bare_aluminium.yaml` opened with the words
*"Bare / polished aluminium"* and held ε_LWIR = 0.09, which is an **oxidised** surface: the two
states are 0.04 and 0.09, a factor of 2.25 in the single term that decides whether a surface
reports its own temperature or the sky's. Nothing could see the contradiction, because neither
number had a state written beside it. §16.2's own note says as much — *"these are surface states,
unwritten"* — and no code enforced it.

## Decision

**`surface_treatment` is a required field on every material, with no default.** Material schema
goes to v2.

The vocabulary is §4.5's own list — `polished`, `machined`, `oxidised`, `anodised`, `painted`,
`sandblasted`, `weathered` — plus two:

* **`as_manufactured`** for substances that leave the works in the state they keep: a moulded
  plastic, a woven fabric, a cast tyre, float glass.
* **`natural`** for substances nobody treated at all: skin, snow, soil, water, a leaf.

Both are still *states* and still have to be chosen. The alternative — making the field optional,
or allowing a free string — puts a blank on exactly the materials whose state is least obvious,
which is the failure this ADR exists to end.

**`bare_aluminium` keeps its name.** It is referenced by scene configs, asset maps, the material
probe stage and some thirty tests, so renaming it is a change to *those* rather than to this
library, and a rename is not what makes the file correct. What makes it correct is that it now
declares `oxidised` and says so in its description. Two siblings hold the other two states:
`aluminium_polished` (0.04) and `aluminium_anodised` (0.845, the midpoint of [R39]'s range).

## Consequences

**The library demonstrates the rule instead of violating it.** The three aluminium files differ
only in surface state: ρ, c_p and k are identical, because those belong to the metal, and ε spans
0.805, because that belongs to the surface. A file copied from one to another without changing its
optics fails a test rather than becoming a plausible new material.

**One real physical distinction becomes expressible.** `aluminium_anodised` *falls* toward grazing
where the two bare metals rise, because its emitting surface is micrometres of oxide — a rough
dielectric — over the same metal, so it takes Level B where they take Level A (Fresnel). A library
keyed on substance cannot hold that; one keyed on surface state can, and
`tests/unit/test_directional_dispatch.py` asserts the whole committed set either way.

**A schema bump costs the 21 committed files one line each**, and any material authored outside
this repository fails to load until it names a state. That is the intended cost: a v1 file is one
whose surface is unstated, and loading it silently would be the same defect under a version number.

**What this does not fix.** §4.5 (b) is about anodic *film thickness* — ε ≈ 0.15 at 1 µm of film,
0.45 at 2 µm, 0.91 above about 15 µm [R40] — and that stays unobtainable, because no mesh carries
a film thickness and `thermal.thickness_m` is §6.1's heat-capacity parameter that nothing optical
may read. The state is authored; the film is never solved. Nor does naming a state make a value a
measurement: every §16.2 row remains literature or ESTIMATED, now with its surface written down.

## Revisit when

A second substance in this library needs more than one state — painted steel against rusted steel
is the obvious next pair — or when a spectral library covering all four bands [R41][R42] replaces
the §16.2 rows, at which point the state becomes the key the spectra are looked up by rather than
an annotation beside them.
