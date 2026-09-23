# ADR 0129 — Defocus is Hopkins, and defocus is achromatic

**Status:** Accepted
**Date:** 2026-09-23

## Context

`docs/physics-model.md` §8.3 names `MTF_defocus` in the cascade and then declines to define it —
"in practice fit a Gaussian … from a measured slant-edge, rather than deriving it". ADR 0059 took
that advice and folded aberration and defocus into one Gaussian; `SC.4`/ADR 0117 then solved that
Gaussian's σ from FLIR's published on-axis MTF at Nyquist, which is an **in-focus** figure. The net
effect was that the one number in the repository that could have carried defocus was authored to
mean its opposite, and no camera had a focus distance at all (`OC.1`).

`OC.2` has to choose the model that turns W020 into an OTF. The choice is not obvious, because the
usual rule of thumb — "use the geometric disk once the blur is bigger than the Airy disc" — is a
visible-wavelength habit, and λ is twenty times larger here.

## Options considered

1. **Geometric disk (pillbox), OTF = 2J₁(πcξ)/(πcξ).** One closed form, no quadrature. Valid only
   for W020 > 2λ, i.e. a blur circle of 16Fλ. For a Boson at F/1.0 in LWIR that is **168 µm —
   fourteen pixels** — reached only inside 1.2 m. It is invalid across the entire range this
   project renders.
2. **A fatter Gaussian.** Free, and it is what §8.3 suggests. It is strictly positive and monotone,
   so it cannot express contrast reversal at all, and it is the model the repository already has.
3. **Hopkins' defocus OTF**, the quadrature `4/(πa)∫₀^√(1−s²) sin[a(√(1−y²)−s)] dy`,
   `a = 8π W020 s/λ`. One term covering diffraction *and* defocus.
4. **Full wave-optics pupil**, `P(ρ) = A(ρ)exp(i2πW020ρ²/λ)`, PSF = |ℱ{P}|². The general form, and
   the one that extends to Zernike aberrations and a real cold stop.

## Decision

**Hopkins (3), with (1) and (2) kept as selectable models for ablation.** Option 4 is deferred until
something needs an aberration that is not a fitted Gaussian.

Hopkins reduces analytically — and numerically, to the 7.7e-9 Simpson floor — to `mtf_diffraction`
at W020 = 0. It therefore **replaces** the cascade's diffraction factor rather than multiplying onto
it, and ADR 0117's in-focus aberration Gaussian still multiplies on top, unchanged.

Measured at Nyquist for a Boson 640 focused at infinity looking at 10 m (c = 19.6 µm, W020 = 0.23λ),
all three as the bare lens OTF with no aberration term: in focus **0.461**, Hopkins **0.356**,
Gaussian **0.203**, geometric disk **0.173**. The cheap models lose about half the contrast that is
actually there.

## Consequences

**The geometric rule of thumb is looser than it is usually quoted.** The gap between Hopkins and the
geometric disk peaks at 0.22 in the transition band, is still **0.07** at the W020 = 2λ threshold
`geometric_regime_blur_um` encodes, and only falls under 0.02 by sixteen waves. The threshold is an
asymptote marker, not a switchover point, and `test_defocus_otf.py` asserts the whole series so that
nobody tightens it on the strength of the name.

**Defocus is achromatic, which contradicts how `OC` was planned.** Substituting `s = ξλF` into
`a = 8π W020 s/λ` cancels the wavelength: `a = 8π W020 ξ F`. The blur circle `c = 8 F W020` has no λ
in it either. So the defocus zeros sit at fixed spatial frequencies and band averaging cannot smear
them — the single-λ ringing the lane was planned to avoid does not exist. What *is* chromatic is
diffraction, whose cut-off runs 133 down to 74 cyc/mm across 7.5–13.5 µm. Measured at W020 = 2λ, the
band-averaged OTF differs from the 10.5 µm one by under 0.3 % wherever the OTF still carries
contrast, and reshapes only the far tail. `band_average_otf` is therefore kept, and kept honest:
worth having for the cut-off, not the rescue it was billed as.

**The band weight is R(λ), not R(λ)·L(λ,T).** The physically exact broadband OTF weights each
wavelength by the spectral radiance actually reaching the detector, which depends on the scene and
would force the kernel to be rebuilt per frame. R(λ) alone is the convention every published MTF
bench uses. Given the paragraph above — averaging moves the contrast-carrying part of the OTF by
under 0.3 % — the error this approximation carries is smaller than that, so it is not worth a
per-frame rebuild. Revisit if a scene ever puts a 1200 K flame and a 250 K sky in one frame and the
tail structure turns out to matter.

**SciPy is still not a dependency.** NumPy has no Bessel functions, so `bessel_j1` is the integral
representation `(1/π)∫₀^π cos(θ − x sinθ)dθ` evaluated by the same Simpson grid as Hopkins, exact to
1e-9 against published values, rather than a rational approximation with a validity band to get
wrong.

**Cost.** Both quadratures are 2049-node Simpson over a vectorised grid. They are evaluated when a
kernel is built, not per pixel, so the cost lands in `OC.3`'s kernel bank and not in the frame loop.
