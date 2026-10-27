"""Elasto-petrophysical logs of the pseudo-wells (rock physics of the manuscript, Table 2).

The facies of each sample (Markov-chain facies of a pseudo-well, or the interpreted facies of a
well) sets its relations, in two-way time t (s):
  shale            Vp = a2 t + b2 + eta_vp,  Vs = 0.50 Vp - 56.207,  rho = 0.34 Vp^0.251,
                   porosity 0.01 + eta_phi
  sand (silt)      porosity = 2 f (a1 t + b1) + eta_phi   (silt: 2 (a4 t + b4) + eta_phi),
                   Kdry = -35.5 phi + 28,  Gdry = 0.8 Kdry - 2.5,  Gassmann (brine),
                   mixed linearly with shale according to Vshale
  calcareous sand  Vp = a3 t + b3 + eta_vp,  Vs = 0.52 Vp - 56.207,  rho = 0.32 Vp^0.251,
                   porosity 0.01 + eta_phi
  Vshale           mean of the facies at Well 2X + AR(4) fluctuation, clipped to [0, 1]
eta_vp and eta_phi are AR(4) processes fitted to the residuals log - relations at Well 2X; they
act on the quantities that drive the relations, so that Vs and rho follow through the relations
and Gassmann. All fluctuations are drawn independently for every pseudo-well. f scales the sand
porosity trend to a maximum of 0.18 at Well 2X. No smoothing. Calibration (f, Vshale statistics,
AR models): calibrate_well2x.py -> Data/rock_physics_2x.json.

Facies codes: 0 shale, 1 silt, 2 sand, 3 calcareous sand.
"""
import json
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

PHI_SCALE = 2.0          # porosity = PHI_SCALE (a t + b) in sand and silt
KDRY_SLOPE = -35.5       # Kdry = KDRY_SLOPE phi + 28 (GPa)
BURN = 100               # samples discarded at the start of every AR series
STATS_FILE = Path(__file__).resolve().parents[2] / "Data" / "rock_physics_2x.json"
_STATS = None


def stats_2x(path=None):
    """Calibration of Well 2X (calibrate_well2x.py)."""
    global _STATS
    if _STATS is None or path is not None:
        _STATS = json.loads(Path(path or STATS_FILE).read_text())
    return _STATS


def ar_filter(c, e):
    """AR process with the coefficients and innovation std of c, driven by the white noise e
    (ns + BURN, nsim); the first BURN samples are discarded."""
    return lfilter([1.0], np.r_[1.0, -np.asarray(c["ar"])], e * c["sigma"], axis=0)[BURN:]


def ar_series(c, ns, nsim, rng):
    """(ns, nsim) independent stationary AR series of mean c["mean"]."""
    return c["mean"] + ar_filter(c, rng.standard_normal((ns + BURN, nsim)))


def facies_vshale(f, st, rng):
    """Vshale: mean of the facies at Well 2X + AR fluctuation, clipped to [0, 1]."""
    ns, nsim = f.shape
    vsh = np.ones((ns, nsim))
    for code, key in ((0, "shale_vsh"), (1, "silt_vsh"), (2, "sand_vsh"), (3, "calc_vsh")):
        vsh = np.where(f == code, np.clip(ar_series(st[key], ns, nsim, rng), 0.0, 1.0), vsh)
    return vsh


def table_relations(t, fac, vsh, rp, sand_factor, d_vp=0.0, d_phi=0.0):
    """Relations of Table 2; t (ns,), fac and vsh (ns, nsim). d_vp and d_phi are the fluctuations
    of shale / calcareous-sand Vp and of porosity (0 = trend values). Returns vp, vs, rho, por."""
    t = np.asarray(t, float)[:, None]
    f = np.asarray(fac)
    vp_sh = rp["a2"] * t + rp["b2"] + d_vp
    vs_sh = 0.50 * vp_sh - 56.207
    rho_sh = 0.34 * vp_sh ** 0.251
    phi = np.where(f == 2, PHI_SCALE * sand_factor * (rp["a1"] * t + rp["b1"]),
                   PHI_SCALE * (rp["a4"] * t + rp["b4"])) + d_phi
    phi = np.clip(phi, 0.005, 0.35)
    kdry = np.maximum(KDRY_SLOPE * phi + 28.0, 0.5)
    gdry = np.maximum(0.8 * kdry - 2.5, 0.3)
    km, kfl, rfl = rp["Kmat"], rp["Kw"], rp["rhow"]
    ksat = kdry + (1 - kdry / km) ** 2 / (phi / kfl + (1 - phi) / km - kdry / km ** 2)
    rho_sd = rfl * phi + (1 - phi) * rp["rhomat"]
    vp_sd = 1000 * np.sqrt((ksat + 4 / 3 * gdry) / rho_sd)
    vs_sd = 1000 * np.sqrt(gdry / rho_sd)
    vp_c = rp["a3"] * t + rp["b3"] + d_vp
    sand = (f == 1) | (f == 2)
    vp = np.where(f == 0, vp_sh, np.where(f == 3, vp_c, vp_sd * (1 - vsh) + vp_sh * vsh))
    vs = np.where(f == 0, vs_sh, np.where(f == 3, 0.52 * vp_c - 56.207, vs_sd * (1 - vsh) + vs_sh * vsh))
    rho = np.where(f == 0, rho_sh, np.where(f == 3, 0.32 * vp_c ** 0.251,
                                            rho_sd * (1 - vsh) + 0.34 * vp_sh ** 0.25 * vsh))
    por = np.where(sand, phi, np.clip(0.01 + d_phi, 0.0, 0.4))
    return vp, vs, rho, por


def pseudo_well_properties(time, facies, rp, rng):
    """(vp, vs, rho, por, vsh), each (ns, nsim), for the facies columns `facies` (ns, nsim)."""
    st = stats_2x(rp.get("stats_2x"))
    f = np.asarray(facies)
    ns, nsim = f.shape
    vsh = facies_vshale(f, st, rng)
    res = st["residual_ar"]
    d_vp = ar_filter(res["vp"], rng.standard_normal((ns + BURN, nsim)))
    d_phi = ar_filter(res["por"], rng.standard_normal((ns + BURN, nsim)))
    vp, vs, rho, por = table_relations(time, f, vsh, rp, st["sand_factor"], d_vp, d_phi)
    return vp, vs, rho, por, vsh
