# The material library

Every surface irsim renders is made of a library material. A material says how much the surface
**emits** in each camera band (near infrared, short-wave, mid-wave and long-wave), how much
**sunlight** it absorbs, and how it stores and conducts **heat**. Those numbers decide what the
camera sees: a surface that emits well shows its own temperature, and one that hardly emits
shows a reflection of the sky. This page says where every number came from and how you can
check it yourself.

The library has 49 materials. 26 of them were measured and imported by a script; the rest are
the project's first, literature-based values, and say so in their own files. The files are in
[`configs/materials/`](../configs/materials/), one per material.

## How to read a material

- **Emissivity (ε)**, between 0 and 1, per band. 1 is a perfect emitter (a black body); a
  polished metal is below 0.05. What is not emitted is reflected, or, for glass, transmitted:
  the three always add up to exactly 1 (Kirchhoff's law), and the library refuses a file that
  breaks this.
- **Surface state.** Emissivity belongs to the *surface*, not the substance. Aluminium measures
  0.04 polished and 0.85 anodised, so every material names its state: polished, oxidised,
  painted, weathered, as manufactured or natural.
- **Sunlight absorbed** (the solar absorptivity) sets how much a surface warms in the sun. A white
  plastic absorbs about a quarter of it; tar absorbs almost all of it.
- **Thermal body**: density, specific heat, conductivity and a typical thickness, from standard
  engineering handbooks.

## Where the numbers come from

### Polished metals, from their measured optical constants

Seven metals: copper, steel, chrome plating, titanium, gold, silver and magnesium. Physicists
measure a metal's *optical constants* (its complex refractive index) in the laboratory, and for a
clean, polished metal the emissivity follows from them exactly (the Fresnel equations). We take
the published measurements from the RefractiveIndex.INFO database, which is in the public domain,
and compute each metal's emissivity from 0.3 to 16 µm. The same data gives how emissivity changes
with the viewing angle.

These are the **polished** states: the lowest emissivity each metal can have. A real part in
service is rougher and a little oxidised, and emits more. That is a different material.

### Surfaces measured in two parts: USGS and a thermal-infrared library

Nineteen surfaces: roads, roofs, bricks, concrete, wood, paint, leaves, grass, sand, desert
playa, sea water, snow and two plastics. No single free source measures a real brick or road
across all four bands, so each of these combines two measurements:

- **The U.S. Geological Survey Spectral Library, version 7** measured how much light hundreds of
  real samples reflect, from 0.35 to 2.5 µm: weathered bricks, an old road, roof tar, pine
  beams, leaves picked in the field. It is one of the most used reference libraries in remote
  sensing, it is published as an official USGS data release with a peer-reviewed description,
  and it is in the public domain. From it we take the **near-infrared and short-wave** values and
  the **sunlight absorbed**. The curves themselves are committed in
  [`data/spectra/materials/usgs/`](../data/spectra/materials/usgs/), each with the USGS record
  number it came from.
- **The MODIS UCSB Emissivity Library** measured emissivity directly from 3.3 to 14.5 µm, with an
  integrating sphere, for building materials, soils, vegetation, water and snow. It was built to
  support NASA's MODIS land-surface-temperature product. From it we take the **mid-wave and
  long-wave** values. It states no licence, so we cite and extract only the band values; its
  curves are not copied into this repository.
- **For plastics** (white PVC and black PET), the long-wave values are computed from the
  polymer's own measured optical constants, the same way as the metals.

Each material's file names both samples and explains the pairing. A pairing is a judgement: it
matches the same *kind* of surface (a clay brick with a clay brick), not the same physical
sample.

### The rest: literature starting values

The remaining 23 materials (car paints, rubber, human skin, cotton, glass, carbon fibre and
others) carry the literature values the project started with, in
[physics model §16.2](physics-model.md). Where a value was estimated rather than looked up, the
file marks it `ESTIMATED`.

## How you can be sure the values are right

**1. Every number can be traced.** Each file names its source down to the sample: the USGS record
number, the UCSB file name, the paper and its DOI for the optical constants, the handbook table
for the thermal body. You can open the source and compare.

**2. The numbers are recomputed from the data at every change.** The project's test suite
re-derives each imported value from the committed data: the near-infrared and short-wave values
and the sunlight absorbed from the USGS curves, and the metals' and plastics' emissivity from the
optical constants. If a number in a file stopped matching its data, the tests would fail and the
change could not be committed.

**3. The values must obey physics the data cannot fake.** The tests also check properties that
any correct measurement of these surfaces has, independent of the data:

- **Metals.** A metal's long-wave emissivity is tied to its electrical resistance (the
  Hagen–Rubens relation). Each metal's value lands within a factor of two of that prediction;
  silver, copper and gold (the best conductors) are the best mirrors; each metal gets brighter
  toward grazing angles, as metals do.
- **Leaves** show the "red edge": they reflect strongly in the near infrared and are almost black
  in the thermal bands.
- **Snow** is white in the near infrared and black in the short-wave, where ice absorbs.
- **Quartz sand** is the least emissive natural surface in the long wave, from quartz's
  well-known reflection band at 8–9.5 µm.
- **Water** is dark in every band.
- **Kirchhoff's law** holds for every material in every band, to one part in a million.

**4. The values agree with independent literature.** The table below sets measured materials
beside the literature values the project started from, for the same kind of surface. In the long
wave, the band most thermal cameras use, every pair agrees within 0.04. The larger differences
have physical reasons, and they are why measured values replace the generic ones: melting snow
absorbs about twice the sunlight of the fresh snow the literature row describes; a bright desert
playa reflects more in the mid-wave than generic soil, because of its minerals; and the two
measured leaves absorb more sunlight than the generic leaf row assumed.

<!-- comparison table: written by scripts/material_catalogue.py -->

| measured material | literature row (§16.2) | LWIR measured / literature | MWIR measured / literature | sunlight absorbed measured / literature |
|---|---|---|---|---|
| `asphalt_road_aged` | Asphalt (dry) | 0.97 / 0.94 | 0.97 / 0.92 | 0.89 / 0.90 |
| `concrete_pavement` | Concrete | 0.96 / 0.92 | 0.93 / 0.90 | 0.71 / 0.65 |
| `cinder_block` | Concrete | 0.96 / 0.92 | 0.95 / 0.90 | 0.65 / 0.65 |
| `leaf_maple` | Vegetation (leaf) | 0.96 / 0.97 | 0.96 / 0.96 | 0.69 / 0.50 |
| `conifer_needles` | Vegetation (leaf) | 0.98 / 0.97 | 0.98 / 0.96 | 0.63 / 0.50 |
| `playa_dry_mud` | Soil (dry) | 0.95 / 0.92 | 0.81 / 0.90 | 0.55 / 0.75 |
| `sand_beach` | Soil (dry) | 0.90 / 0.92 | 0.87 / 0.90 | 0.72 / 0.75 |
| `seawater` | Water | 0.98 / 0.96 | 0.98 / 0.98 | 0.97 / 0.93 |
| `snow_melting` | Snow | 0.99 / 0.99 | 0.99 / 0.98 | 0.32 / 0.15 |

<!-- end of comparison table -->

**5. The limits are stated, not hidden.** The UCSB measurements start at 3.34 µm, so the mid-wave
value covers 83 % of that band. The sunlight absorbed covers the 97 % of the sun's energy that USGS
measured. Paired samples are the same kind of surface, not the same sample. Paper, fabrics,
polyethylene, ABS and rubber do not have measured thermal-infrared values yet.

## Every material

The values the simulator uses, per band (a Planck-weighted average over the band at 300 K):

<!-- catalogue table: written by scripts/material_catalogue.py -->

| material | surface | NIR | SWIR | MWIR | LWIR | sunlight absorbed | where the optics come from |
|---|---|---|---|---|---|---|---|
| `abs_plastic_white` | as manufactured | 0.12 | 0.22 | 0.90 | 0.95 | 0.25 | literature, partly estimated |
| `aircraft_aluminium_painted` | painted | 0.25 | 0.35 | 0.88 | 0.90 | 0.30 | literature, partly estimated |
| `aluminium_anodised` | anodised | 0.15 | 0.25 | 0.82 | 0.84 | 0.35 | literature, partly estimated |
| `aluminium_polished` | polished | 0.04 | 0.03 | 0.03 | 0.04 | 0.08 | literature, partly estimated |
| `asphalt_dry` | weathered | 0.92 | 0.90 | 0.92 | 0.94 | 0.90 | literature, partly estimated |
| `asphalt_road_aged` | weathered | 0.86 | 0.80 | 0.97 | 0.97 | 0.89 | measured: USGS + UCSB |
| `bare_aluminium` | oxidised | 0.08 | 0.05 | 0.06 | 0.09 | 0.15 | literature, partly estimated |
| `brick_red` | as manufactured | 0.77 | 0.75 | 0.67 | 0.95 | 0.83 | measured: USGS + UCSB |
| `brick_tan` | as manufactured | 0.83 | 0.82 | 0.92 | 0.96 | 0.86 | measured: USGS + UCSB |
| `car_paint_black` | painted | 0.94 | 0.94 | 0.88 | 0.90 | 0.94 | literature, partly estimated |
| `car_paint_white` | painted | 0.30 | 0.40 | 0.88 | 0.90 | 0.28 | literature, partly estimated |
| `carbon_fibre` | as manufactured | 0.90 | 0.88 | 0.88 | 0.90 | 0.90 | literature, partly estimated |
| `chrome_plated` | polished | 0.37 | 0.32 | 0.07 | 0.03 | 0.35 | measured optical constants |
| `cinder_block` | weathered | 0.65 | 0.65 | 0.95 | 0.96 | 0.65 | measured: USGS + UCSB |
| `concrete` | weathered | 0.65 | 0.72 | 0.90 | 0.92 | 0.65 | literature, partly estimated |
| `concrete_pavement` | weathered | 0.69 | 0.66 | 0.93 | 0.96 | 0.71 | measured: USGS + UCSB |
| `conifer_needles` | natural | 0.35 | 0.60 | 0.98 | 0.98 | 0.63 | measured: USGS + UCSB |
| `copper_polished` | polished | 0.05 | 0.03 | 0.01 | 0.01 | 0.19 | measured optical constants |
| `cotton_clothing` | as manufactured | 0.45 | 0.55 | 0.93 | 0.95 | 0.70 | literature, partly estimated |
| `etics_render` | weathered | 0.88 | 0.88 | 0.90 | 0.91 | 0.30 | literature, partly estimated |
| `glass_windshield` | as manufactured | 0.15 | 0.22 | 0.85 | 0.88 | 0.10 | literature, partly estimated |
| `gold_polished` | polished | 0.01 | 0.01 | 0.01 | 0.01 | 0.15 | measured optical constants |
| `grass_dry` | natural | 0.67 | 0.67 | 0.91 | 0.96 | 0.76 | measured: USGS + UCSB |
| `human_skin` | natural | 0.50 | 0.70 | 0.97 | 0.98 | 0.65 | literature, partly estimated |
| `leaf_maple` | natural | 0.38 | 0.64 | 0.96 | 0.96 | 0.69 | measured: USGS + UCSB |
| `magnesium_polished` | polished | 0.06 | 0.13 | 0.05 | 0.04 | 0.06 | measured optical constants |
| `painted_aluminium_green` | painted | 0.58 | 0.70 | 0.65 | 0.97 | 0.56 | measured: USGS + UCSB |
| `painted_composite` | painted | 0.30 | 0.40 | 0.90 | 0.92 | 0.35 | literature, partly estimated |
| `pet_black` | as manufactured | 0.94 | 0.95 | 0.96 | 0.94 | 0.94 | measured: USGS + polymer n, k |
| `playa_dry_mud` | natural | 0.46 | 0.44 | 0.81 | 0.95 | 0.55 | measured: USGS + UCSB |
| `plywood` | as manufactured | 0.19 | 0.40 | 0.89 | 0.95 | 0.44 | measured: USGS + UCSB |
| `propeller_rubber` | as manufactured | 0.94 | 0.92 | 0.94 | 0.95 | 0.94 | literature, partly estimated |
| `pvc_white` | as manufactured | 0.31 | 0.55 | 0.96 | 0.96 | 0.38 | measured: USGS + polymer n, k |
| `roof_shingle_dark` | weathered | 0.90 | 0.92 | 0.97 | 0.97 | 0.90 | measured: USGS + UCSB |
| `roof_tar_black` | weathered | 0.97 | 0.97 | 0.97 | 0.97 | 0.97 | measured: USGS + UCSB |
| `rubber_tyre` | as manufactured | 0.95 | 0.94 | 0.94 | 0.95 | 0.94 | literature, partly estimated |
| `rusted_steel` | oxidised | 0.62 | 0.68 | 0.82 | 0.85 | 0.80 | literature, partly estimated |
| `sand_beach` | natural | 0.68 | 0.57 | 0.87 | 0.90 | 0.72 | measured: USGS + UCSB |
| `seawater` | natural | 0.98 | 0.98 | 0.98 | 0.98 | 0.97 | measured: USGS + UCSB |
| `silver_polished` | polished | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | measured optical constants |
| `snow` | natural | 0.15 | 0.90 | 0.98 | 0.99 | 0.15 | literature, partly estimated |
| `snow_melting` | natural | 0.41 | 0.98 | 0.99 | 0.99 | 0.32 | measured: USGS + UCSB |
| `soil_dry` | natural | 0.70 | 0.74 | 0.90 | 0.92 | 0.75 | literature, partly estimated |
| `soil_wet` | natural | 0.85 | 0.90 | 0.95 | 0.96 | 0.85 | literature, partly estimated |
| `steel_polished` | polished | 0.36 | 0.25 | 0.08 | 0.02 | 0.39 | measured optical constants |
| `titanium_polished` | polished | 0.39 | 0.35 | 0.19 | 0.07 | 0.40 | measured optical constants |
| `vegetation_leaf` | natural | 0.10 | 0.25 | 0.96 | 0.97 | 0.50 | literature, partly estimated |
| `water` | natural | 0.95 | 0.98 | 0.98 | 0.96 | 0.93 | literature, partly estimated |
| `wood_pine` | as manufactured | 0.34 | 0.51 | 0.89 | 0.95 | 0.53 | measured: USGS + UCSB |

<!-- end of catalogue table -->

## Adding or updating materials

The importers are scripts, so the library can be rebuilt from its sources:

- `scripts/import_material_spectra.py metals` fetches the metals' optical constants and writes
  their curves and files.
- `scripts/import_paired_spectra.py --usgs <folder>` reads the unzipped USGS library and the
  UCSB spectra and writes the paired materials.
- `scripts/material_catalogue.py` rewrites the tables on this page.

To add one material by hand, the [Blender add-on](blender-addon.md) has a form that checks it with
the library's rules before writing it ([tutorial step 5](tutorials/blender-addon/05-new-material.md)).

## Sources

- R. F. Kokaly, R. N. Clark, G. A. Swayze et al. (2017). *USGS Spectral Library Version 7*. U.S.
  Geological Survey Data Series 1035. [doi:10.3133/ds1035](https://doi.org/10.3133/ds1035);
  data: [doi:10.5066/F7RR1WDJ](https://doi.org/10.5066/F7RR1WDJ). Public domain.
- Z. Wan et al., *MODIS UCSB Emissivity Library*, Institute for Computational Earth System
  Science, University of California, Santa Barbara.
  [icess.eri.ucsb.edu/modis/EMIS](https://icess.eri.ucsb.edu/modis/EMIS/html/em.html).
- M. N. Polyanskiy (2024). *Refractiveindex.info database of optical constants*. Scientific Data
  11, 94. [doi:10.1038/s41597-023-02898-2](https://doi.org/10.1038/s41597-023-02898-2). Public
  domain (CC0). The measurements it compiles, one per material:
  - copper and iron: M. R. Querry (1985), *Optical constants*, Contractor Report CRDC-CR-85034;
  - chromium and titanium: A. D. Rakić, A. B. Djurišić, J. M. Elazar, M. L. Majewski (1998),
    *Appl. Opt.* 37, 5271. [doi:10.1364/AO.37.005271](https://doi.org/10.1364/AO.37.005271);
  - gold: R. L. Olmon et al. (2012), *Phys. Rev. B* 86, 235147.
    [doi:10.1103/PhysRevB.86.235147](https://doi.org/10.1103/PhysRevB.86.235147);
  - silver: H. U. Yang et al. (2015), *Phys. Rev. B* 91, 235137.
    [doi:10.1103/PhysRevB.91.235137](https://doi.org/10.1103/PhysRevB.91.235137);
  - magnesium: H.-J. Hagemann, W. Gudat, C. Kunz (1975), *J. Opt. Soc. Am.* 65, 742.
    [doi:10.1364/JOSA.65.000742](https://doi.org/10.1364/JOSA.65.000742);
  - PVC and PET: X. Zhang, J. Qiu, X. Li, J. Zhao, L. Liu (2020), *Appl. Opt.* 59, 2337.
    [doi:10.1364/AO.383831](https://doi.org/10.1364/AO.383831).
- F. P. Incropera, D. P. DeWitt, T. L. Bergman, A. S. Lavine, *Fundamentals of Heat and Mass
  Transfer*, Tables A.1 (metals) and A.3 (building and natural materials).
- E. Hagen, H. Rubens (1903). *Annalen der Physik* 316, 873: the link between a metal's
  electrical resistance and its infrared emissivity used in the checks above.
