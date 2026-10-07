# ADR 0202 — The cloudscape tiles its weather map with hashed offsets and a histogram-preserving blend

**Status:** Accepted (WX.22 part 1)
**Date:** 2026-10-07

## Context

The per-pixel cloudscape both bands march (`WX.23`, `WX.26`) takes "which lattice cells hold a
cloud, and how large" from one weather map. That map is one 32 km tile, sampled with wrap. A
camera sees 80 km (`max_range_m`), so a low frame held the same cover pattern two or three times.

The clouds themselves did not repeat. Each 3.2 km lattice cell's cloud has its position, size,
rotation and patch hashed from the cell index, so the cloudy-column mask correlated only 0.14 with
itself one tile away. What repeated was the grouping: the clusters and the clear lanes between
them. As a block average at 3 km, the mask correlated 0.24 one tile away.

The voxel `CloudField` behind the dome and the path tracer's volumes is a separate source. It
tiles every 15.36 km, and its sun light maps and volume tiles assume that period.

## Options considered

1. **A larger map.** Moving the period past 80 km needs 2.5× the texels per side and a cover
   solve over 6× the area. It does not remove the period; it hides it.
2. **Value noise per lattice cell, no texture.** This is aperiodic, but it throws away the
   weather map's measured structure, the WX.6 recipe whose sizes and outlines meet §7.5.
3. **Hashed tile offsets with a cross-fade.** Each 32 km tile of the plane reads the same map at
   its own hashed offset. A plain cross-fade of uniform values shrinks their variance in the blend
   band, which moves the cover. Blending on the coverage's normal score with variance-preserving
   weights (Heitz & Neyret 2018) keeps a uniform coverage uniform: `Φ(Σ w g_k / √Σ w²)`.

## Decision

Option 3, in weather-fx (`core.cloudscape.sample_aperiodic`), baked:

- Tile `(i, j)` reads the map at `(u, v)` plus `(hash(i, j), hash(i, j) + 1)`. The origin's tile
  keeps no offset.
- Neighbouring tiles cross-fade over a quarter tile at each border (`APERIODIC_BORDER`).
- The coverage is blended on its normal score; the type channel is blended about its mean with
  the same weights.
- The blend is evaluated once, at every texel centre of a field `APERIODIC_TILES` = 5 tiles a
  side, at the map's own 512 texels a tile, with the tiles' indices taken modulo 5. The result,
  `Cloudscape.weather_field`, wraps without a seam. It repeats every 160 km, twice the camera's
  80 km reach.
- Both sides read that field with one bilinear lookup. NumPy uses `sample_wrapped`; the kernel
  uses its hardware-filtered texture, with `weather_tile_m` set to the field's period. The kernel
  source is the one from before this change (weather-fx `bd77d3d`), apart from the array it
  uploads.

### Why baked, not evaluated in the march

The first version (weather-fx `58df21d`) evaluated the blend inside the march: four filtered
reads and an `erf`/`erfinv` pair per density sample, with manual float32 filtering. Outside Isaac
Sim it ran: a 1280×720 still took 75 ms on the A6000, and the GPU matched NumPy. Inside Isaac Sim's
CUDA context it failed in two ways, with the same PTX:

- With the blend's constants in the kernel's `Layer` struct, the launch faulted with CUDA 700.
  compute-sanitizer reported an invalid `__local__` read. Growing the struct by one float was
  enough; raising the context's stack limit did not help.
- With the constants moved to `wp.constant`s, the kernel never returned from synchronize. A
  traceback dump showed it there after 4 minutes.

This was not root-caused. The bake removes the question: the kernel does what it did before, and
the cost moves to the cloudscape's build.

## Consequences

Measured on the plan view (`scripts/cloud_field_repeat.py`: cumulus, 35 %, 96 km square, 384
columns per side):

| | Before (wrapped) | After |
|---|---|---|
| Cover pattern (3 km blocks), correlation one tile away | 0.21 | **0.00** |
| Column-by-column mask, correlation one tile away | 0.15 | 0.11 |
| Cloudy share | 15.8 % | 15.7 % |

weather-fx's own tests check four more things:

- the coverage quantiles through the borders are within 0.03 of the plain map's;
- nothing jumps across a tile border, or across the bake's own wrap;
- the interior of the origin's tile is the old lookup to 1e-6;
- the bake agrees with the per-point blend: at 512 texels a tile, 0.002 in coverage on average
  and 0.011 at the 99th percentile. They differ only between texel centres, where the blend bends.

The origin sits on the corner where four tiles meet, so the clouds within 8 km of it do change.

- The bake costs about 3 s and 52 MB (2560² × 2 float32) per cloudscape, on top of a 10 s build.
  The GPU holds the same 52 MB as its weather texture. The march itself costs what it did.
- The field repeats every 160 km. A camera at the field's centre could see the same point only
  at opposite horizons.
- The voxel `CloudField` (dome, path-traced tier) still tiles every 15.36 km. Its light maps
  and tiled volumes would need the same treatment; that is `WX.22` part 2.
- The plan view shows a more visible regularity than the tile: straight clear lanes along every
  3.2 km lattice line, both ways. Each patch is faded to zero before its cell's border
  (`_lattice_density`'s border smoothstep) and cannot cross it. That is `WX.29`.

## Revisit when

- The Isaac-only kernel failure is understood. Then an in-march blend would drop the bake's
  memory and its 160 km period.
- `WX.29` lets a patch straddle its cell's border (a 2 × 2 cell neighbourhood).
- `WX.22` part 2 moves the voxel field the same way.
