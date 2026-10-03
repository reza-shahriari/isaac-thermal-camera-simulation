# ADR 0041 — Provenance and licences of checked-in optical data

**Status:** Accepted; amended by [ADR 0179](0179-ecostress-curves-with-jpls-credit-line-as-a-last-resort.md) (ECOSTRESS, with its credit line, where no CC0 or public-domain source exists)
**Date:** 2026-09-15

## Context

This repository checks in numbers that came from somewhere else: complex refractive indices
(`data/nk/`), spectral emissivity curves (`data/spectra/materials/`), detector response curves
(`data/spectra/responses/`). Every radiometric result the simulator produces is a function of them,
so a number whose origin is unknown makes every result downstream unfalsifiable — not wrong,
*uncheckable*, which is worse, because there is no experiment that can find the error.

Three separate risks travel together here and are easy to conflate:

1. **Attribution.** A reader who wants to check a value needs to know which published dataset it is,
   and to be able to go and read it.
2. **Licensing.** Data that may not be redistributed must not be committed, however convenient.
   Several widely-used optical-constant compilations (Palik's handbook is the obvious one) are
   copyrighted books, not public datasets.
3. **Substitution.** The number in the file may not be a measurement of the substance named on it.
   This project has three kinds already: *measured* (Segelstein water), *modelled* (the Drude
   aluminium of M7.5), and *proxy* — a measurement of a genuinely different material standing in for
   one whose constants are not freely available. A proxy that is not labelled reads exactly like a
   measurement.

M7.5 forced the issue on all three at once by needing glass and paint, neither of which has freely
available optical constants for the actual substance across 0.75–13.5 µm.

## Options considered

1. **A licence/provenance manifest beside the data.** One index file listing each dataset, its
   source and its licence. Rejected: it is a second thing to update, so it drifts, and the loader
   cannot enforce it per file without reading a file the data file does not mention.
2. **Provenance in the data file, checked as lint.** A CI rule that every `data/**.csv` has a
   `# source:` line. Better, but lint is advisory — it runs on the repository, not on the file a
   user points `$IRSIM_DATA_DIR` at, which is precisely where an unsourced file arrives from.
3. **Provenance in the data file, enforced by the loader.** A table without a `# source:` header
   fails to load. Chosen.
4. **Fetch at run time from the upstream database instead of committing.** Rejected: `make ci` has
   no network, and a radiometric result that depends on a live download is not reproducible. The
   fetch is a *script*, run deliberately, and its output is committed.

## Decision

**Provenance is data, not metadata, and the loader enforces it.** `load_nk_table` refuses a table
with no `# source:` block (`irsim/materials/nk.py`). The block is free prose, carried on the loaded
`NKTable`, so it reaches anyone holding the table rather than only someone reading the file.

**Only redistributable data is committed.** In practice that means public-domain or CC0 sources.
The two M7.5 tables come from the RefractiveIndex.INFO database, which is CC0 1.0; Segelstein's
water table is a public thesis dataset hosted by the Oregon Medical Laser Center. Palik and other
copyrighted compilations are *not* checked in, even where they are the better data — where one is
the right source, the alternative is a modelled table with the discrepancy recorded, as aluminium
does.

**Every file declares which of three kinds it is, in the source block itself:**

| kind | meaning | in the tree |
|---|---|---|
| measured | a published measurement of this substance | `water.csv` (Segelstein 1981) |
| MODELLED | a closed-form model, parameters stated | `aluminium.csv` (Drude, M7.5) |
| PROXY | a measurement of a *different* substance | `glass.csv` (fused silica for soda-lime), `paint_proxy.csv` (PMMA for clearcoat) |

The kind goes **inside** the `# source:` block, not in a separate comment. The loader ends the
provenance it carries at the first blank comment line, so a note after one is invisible to every
consumer — which is how the first draft of `glass.csv` shipped its PROXY label where nothing could
see it. `test_a_proxy_table_declares_that_it_is_a_proxy` now asserts the label survives the load.

**Regeneration is a script, and the script is the provenance.** `scripts/fetch_nk_tables.py`
(measured data, from URLs) and `scripts/generate_nk_tables.py` (modelled data, from parameters)
write the header they document themselves, including the retrieval date and the URL. Neither runs
in a test.

## Consequences

**What this makes easy.** Any value can be traced to a source without leaving the file it lives in.
A proxy cannot quietly become a measurement in a reader's mind. Adding a dataset is a scripted,
repeatable act rather than a paste.

**What this makes hard, deliberately.** Data cannot be dropped in. A new table needs a source, a
licence check and a kind, and the loader will not take it otherwise.

**The error this introduces.** Proxy data carries a real, unbounded-in-general error in *absolute*
terms: fused silica is not soda-lime, PMMA is not paint. The project contains that error by a
convention established in M7.5 — the table supplies the angular **shape** and the material's
authored per-band value supplies the **magnitude** (`directional_emissivity`). So the residual
error is in how emissivity *falls with angle*, which is set by the interface and the band's n and
k, and not in the emissivity itself. That is the bounded part; it is not zero, and for glass in the
transparent bands there is a second approximation on top of it (below).

**A known limit, stated rather than buried.** Level A computes ε = 1 − R, which for a
semi-transparent material is absorptance *plus* transmittance, not emissivity. For glass this is
exact in LWIR, where τ = 0, and approximate in NIR and SWIR, where τ is 0.77 and 0.70: it neglects
the lengthening of the internal path toward grazing, which raises true absorptance relative to
1 − R. The magnitude is unaffected (it comes from the authored value); what is approximate is the
shape, in the two bands where ε is smallest. Improving it needs the §4.4 slab form, which is a
different model, not a better table.

## Revisit when

- A required dataset exists only under a licence that forbids redistribution — then the choice is a
  modelled table plus the recorded discrepancy, or a documented fetch the user runs themselves. Do
  not commit it.
- A measured soda-lime glass or a measured pigmented-coating dataset covering 0.75–13.5 µm becomes
  freely available: swap the file, keep everything else. The loader and every consumer are unchanged
  by the swap, which is the point of the header being the only place the source is named.
- Tier 4 shows a per-class angular bias that tracks the proxy materials specifically — that would
  mean the shape-from-proxy assumption, not just the magnitude, needs the real substance.
