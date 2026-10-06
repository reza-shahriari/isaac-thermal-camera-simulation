"""Do the two bands agree about where the cloud is? (WX.26; docs/physics-model.md §7.5)

Criterion 3 of §7.5 says the visible and the infrared cover the same pixels with cloud, because
both sample one field from one origin. That was first *asserted* by construction and judged by
eye on a rendered pair (ADR 0190). This module measures it: the visible companion's own march
leaves a transmittance per pixel (weather-fx's ``CloudLayerEffect.transmittance``), the infrared
march leaves its own (the ``cloud_transmittance`` truth plane), and the two are compared as the
same physical quantity -- the band's emissivity -- through the one number that links them,
``τ_B = r_B τ_vis`` (§7.5, ADR 0126, 0162)::

    ε_B(predicted from the visible) = 1 − T_vis ** r_B
    ε_B(marched)                    = 1 − T_B

The score is three things: the intersection over union of the two cloud masks at the same
emissivity threshold, the 95th-percentile and mean absolute emissivity error, and the Pearson
correlation of the two emissivities over the pixels either band put cloud on. The visible march
stops a ray at a transmittance floor (0.004 in weather-fx), so below it the visible says only
"opaque"; an infrared emissivity at or above what the floor implies counts as agreement there.

Pure NumPy, so the comparison runs on saved planes with no engine;
``scripts/cloud_band_agreement.py`` does that on a rendered directory and ``render_phantom4.py``
does it on every frame it renders.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from irsim.atmosphere.cloud import CLOUD_OD_RATIO

__all__ = [
    "IOU_BAR",
    "MIN_CLOUD_FRACTION_FOR_IOU",
    "P95_BAR",
    "VISIBLE_TRANSMITTANCE_FLOOR",
    "unreached_ray_mask",
    "CloudBandAgreement",
    "band_emissivity_from_visible",
    "cloud_band_agreement",
]

#: weather-fx's per-pixel march ends a ray once its transmittance falls under this (its
#: ``cloud_march`` kernel: ``transmittance > 0.004``), so a saved visible transmittance at or
#: below it means "opaque", not a value.
VISIBLE_TRANSMITTANCE_FLOOR = 0.004
#: The bar (ADR 0191). The IoU is the roadmap step's own. The emissivity tolerance is the
#: engine-free test's: once both marches read one shell round the planet to one range, weather-fx's
#: layer at its capture settings and the infrared deck agree to 0.005-0.03 at the 95th percentile
#: on cumulus and cirrus at 10 and 30 degrees, so 0.05 holds between the two integrators as it
#: does between the deck and the function's own dense integral.
IOU_BAR = 0.9
P95_BAR = 0.05
#: Below this cloud fraction in both bands the IoU is not judged: a wisp of a few hundred pixels
#: at a frame's edge makes the ratio swing with a one-pixel shift, and says nothing about the
#: cloud. The emissivity error is judged on every frame.
MIN_CLOUD_FRACTION_FOR_IOU = 0.01
#: The cloud mask's threshold on the **band** emissivity, the same one the ``cloud_id`` truth
#: plane uses on the band transmittance (``irsim.io.truth.CLOUD_THRESHOLD``).
MASK_EMISSIVITY = 0.5


@dataclass(frozen=True)
class CloudBandAgreement:
    """One frame's agreement between the visible and the infrared cloud."""

    #: Intersection over union of the two cloud masks (band emissivity >= 0.5 in each).
    iou: float
    #: 95th percentile and mean of |ε_B(marched) − ε_B(predicted from the visible)| over the
    #: pixels either band put cloud on (ε > 0.01 in either); 0 when neither did anywhere.
    p95_emissivity_error: float
    mean_emissivity_error: float
    #: Pearson correlation of the two emissivities over the same pixels; 1 when they are one
    #: constant (an all-opaque frame), NaN when there were none.
    correlation: float
    #: Fraction of the compared pixels each band calls cloud.
    cloud_fraction_ir: float
    cloud_fraction_vis: float
    #: Pixels compared, after the mask.
    pixels: int

    def passes(self, *, iou_min: float = IOU_BAR, p95_max: float = P95_BAR) -> bool:
        """WX.26's bar: masks at IoU >= 0.9 (the roadmap's own number) where either band has at
        least :data:`MIN_CLOUD_FRACTION_FOR_IOU` of cloud, and the band emissivity within
        :data:`P95_BAR` at the 95th percentile on every frame (ADR 0191)."""
        if self.pixels == 0:
            return False
        judged = max(self.cloud_fraction_ir, self.cloud_fraction_vis) >= MIN_CLOUD_FRACTION_FOR_IOU
        return (self.iou >= iou_min or not judged) and self.p95_emissivity_error <= p95_max

    def as_dict(self) -> dict[str, Any]:
        return {
            k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in asdict(self).items()
        }


def band_emissivity_from_visible(
    transmittance_vis: Any, od_ratio: float = CLOUD_OD_RATIO
) -> NDArray[np.float64]:
    """The band emissivity a visible transmittance implies: ``1 − T_vis ** r_B``.

    ``τ_B = r_B τ_vis`` (§7.5) and ``T = e^{−τ}``, so ``T_B = T_vis ** r_B``. Exactly what the
    infrared march computes sample by sample from the same density (``absorbed = 1 − e^{−r dτ}``),
    so the only differences left between this and the marched emissivity are the two
    integrators' own errors -- which is what the comparison is for.
    """
    t = np.clip(np.asarray(transmittance_vis, dtype=np.float64), 0.0, 1.0)
    if not 0.0 < float(od_ratio) <= 1.0:
        raise ValueError(f"the band's optical-depth ratio must lie in (0, 1], got {od_ratio}")
    return np.asarray(1.0 - np.power(t, float(od_ratio)), dtype=np.float64)


def unreached_ray_mask(
    elevation_rad: Any, *, thickness_m: float, max_path_m: float
) -> NDArray[np.bool_]:
    """Rays the infrared march crosses the whole layer on: elevation above
    ``asin(thickness / max_path)``.

    The deck ends a ray after ``max_path_m`` inside the layer; a ray shallower than this crosses
    more layer than that and the infrared stops short of what the visible march (80 km of range)
    went on to see -- a difference of path caps, not of cloud. ``thickness_m`` <= 0 or an
    infinite cap keeps every ray.
    """
    el = np.asarray(elevation_rad, dtype=np.float64)
    if not np.isfinite(max_path_m) or thickness_m <= 0.0 or max_path_m <= 0.0:
        return np.ones(el.shape, dtype=bool)
    return np.asarray(el > math.asin(min(thickness_m / max_path_m, 1.0)), dtype=bool)


def cloud_band_agreement(
    transmittance_ir: Any,
    transmittance_vis: Any,
    *,
    od_ratio: float = CLOUD_OD_RATIO,
    mask: Any = None,
    visible_floor: float = VISIBLE_TRANSMITTANCE_FLOOR,
    touched: float = 0.01,
) -> CloudBandAgreement:
    """Score one frame: the infrared band's marched transmittance against the visible march's.

    ``transmittance_ir`` is the **band** transmittance (the ``cloud_transmittance`` plane) and
    ``transmittance_vis`` the visible one, on one pixel grid; ``mask`` keeps a subset of pixels
    (a sky mask, a region). Both arrays may hold NaN, which drops the pixel.
    """
    t_ir = np.asarray(transmittance_ir, dtype=np.float64)
    t_vis = np.asarray(transmittance_vis, dtype=np.float64)
    if t_ir.shape != t_vis.shape:
        raise ValueError(f"the two planes differ in shape: {t_ir.shape} vs {t_vis.shape}")
    keep = np.isfinite(t_ir) & np.isfinite(t_vis)
    if mask is not None:
        keep &= np.asarray(mask, dtype=bool)
    e_ir = 1.0 - np.clip(t_ir[keep], 0.0, 1.0)
    e_vis = band_emissivity_from_visible(t_vis[keep], od_ratio)
    opaque_vis = t_vis[keep] <= float(visible_floor)
    # Below its floor the visible march reports only "opaque": an infrared emissivity at or
    # above what the floor implies is agreement, not an error.
    error = np.abs(e_ir - e_vis)
    error = np.where(opaque_vis & (e_ir >= e_vis), 0.0, error)
    cloud_ir = e_ir >= MASK_EMISSIVITY
    cloud_vis = e_vis >= MASK_EMISSIVITY
    union = cloud_ir | cloud_vis
    n = int(keep.sum())
    iou = float((cloud_ir & cloud_vis).sum() / union.sum()) if union.any() else 1.0
    either = (e_ir > touched) | (e_vis > touched)
    if either.any():
        p95 = float(np.percentile(error[either], 95.0))
        mean = float(error[either].mean())
        a, b = e_ir[either], e_vis[either]
        if a.std() < 1e-9 and b.std() < 1e-9:
            corr = 1.0
        elif a.std() < 1e-9 or b.std() < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(a, b)[0, 1])
    else:
        p95, mean, corr = 0.0, 0.0, float("nan")
    return CloudBandAgreement(
        iou=iou,
        p95_emissivity_error=p95,
        mean_emissivity_error=mean,
        correlation=corr,
        cloud_fraction_ir=float(cloud_ir.mean()) if n else 0.0,
        cloud_fraction_vis=float(cloud_vis.mean()) if n else 0.0,
        pixels=n,
    )
