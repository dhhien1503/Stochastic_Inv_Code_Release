"""Calibration of the pseudo-well rock physics (pseudowell/rock_physics.py, Table 2 of the
manuscript) on Well 2X, over ML1-ML3 in two-way time, on the 1 ms grid of the pseudo-wells:

  sand_factor     scale of the sand porosity trend, so that its maximum over the sand of Well 2X
                  is --sand-phi-max (0.18)
  residual_ar     AR(4) models (Yule-Walker) of the residuals log - relations of Vp and porosity
                  (measured Vshale in the mixing; mean residual of each facies removed): the
                  fluctuations eta_vp and eta_phi of the pseudo-wells
  shale_vsh       Vshale of shale: mean and AR(4) fit of the 1 ms log (74 samples)
  shale_vp        AR(4) fit of the shale Vp log, whose correlation in time is used for the
                  Vshale fluctuations of the other facies
  <facies>_vsh    Vshale of silt, sand and calcareous sand: mean and standard deviation of the
                  log (1 ms log for sand; full-resolution log for silt and calcareous sand, which
                  have 5 and 1 samples at 1 ms), AR(4) shape of shale_vp scaled to that deviation

Usage:  python calibrate_well2x.py            # -> Data/rock_physics_2x.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import lfilter

from paths import DATA_DIR
from odisi_inversion import CFG
from pseudowell.rock_physics import PHI_SCALE, table_relations

ML_DEPTH = (3500.0, 3550.0, 3631.0, 3672.0)      # macrolayer tops at Well 2X (m)
SHALE, SILT, SAND, CALC = 0, 1, 2, 3


def yule_walker(x, p):
    x = x - x.mean()
    n = len(x)
    acf = np.array([np.dot(x[:n - k], x[k:]) / n for k in range(p + 1)])
    R = np.array([[acf[abs(i - j)] for j in range(p)] for i in range(p)])
    phi = np.linalg.solve(R, acf[1:])
    return phi, float(np.sqrt(max(acf[0] - phi @ acf[1:], 0.0)))


def ar_gain(phi, n=2000):
    """Standard deviation of the AR process for unit innovations."""
    h = lfilter([1.0], np.r_[1.0, -np.asarray(phi)], np.r_[1.0, np.zeros(n - 1)])
    return float(np.sqrt((h ** 2).sum()))


def bin_1ms(t_log, x, t):
    i = np.round((t_log - t[0]) / 0.001).astype(int)
    ok = (i >= 0) & (i < len(t)) & np.isfinite(x)
    s = np.bincount(i[ok], x[ok], len(t))
    n = np.bincount(i[ok], minlength=len(t))
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def bin_mode(t_log, f, t):
    i = np.round((t_log - t[0]) / 0.001).astype(int)
    out = np.zeros(len(t), int)
    for k in range(len(t)):
        v = f[i == k].astype(int)
        out[k] = np.bincount(v, minlength=4).argmax() if len(v) else out[k - 1]
    return out


def well_2x(well_mat):
    """Logs of Well 2X over ML1-ML3: full resolution (dict), macrolayer times, 1 ms grid t,
    facies and logs on that grid."""
    m = loadmat(well_mat, squeeze_me=True)
    w = {k: np.asarray(m[k], float) for k in ("times1", "depth1", "vp1", "vs1", "rho1", "por1",
                                              "vsh1", "sw1", "litho1")}
    ml_t = np.interp(ML_DEPTH, w["depth1"], w["times1"])
    sel = (w["times1"] >= ml_t[0]) & (w["times1"] <= ml_t[-1])
    w = {k: v[sel] for k, v in w.items()}
    t = np.arange(np.ceil(ml_t[0] * 1000), np.floor(ml_t[-1] * 1000) + 1) / 1000
    logs = {k: bin_1ms(w["times1"], w[k + "1"], t) for k in ("vp", "vs", "rho", "por", "vsh")}
    return w, ml_t, t, bin_mode(w["times1"], w["litho1"], t), logs


def calibrate(well_mat, order=4, sand_phi_max=0.18):
    w, ml_t, t, fac, logs = well_2x(well_mat)

    def fitted(x):
        v = x[(fac == SHALE) & np.isfinite(x)]
        phi, sig = yule_walker(v, order)
        return dict(mean=float(v.mean()), std=float(v.std()), ar=phi.tolist(), sigma=sig,
                    n_samples=int(len(v)), source="1 ms log, AR fit")

    shale_vp = fitted(logs["vp"])

    def vsh_stats(v, source):
        v = v[np.isfinite(v)]
        return dict(mean=float(v.mean()), std=float(v.std()), ar=shale_vp["ar"],
                    sigma=float(v.std()) / ar_gain(shale_vp["ar"]), n_samples=int(len(v)),
                    source=source + ", AR shape of shale Vp")

    rp = CFG["rock"]
    trend = PHI_SCALE * (rp["a1"] * t[fac == SAND] + rp["b1"])
    sand_factor = float(sand_phi_max / trend.max())
    vsh_m = np.where(np.isfinite(logs["vsh"]), logs["vsh"], 1.0)[:, None]
    est = dict(zip(("vp", "vs", "rho", "por"),
                   table_relations(t, fac[:, None], vsh_m, rp, sand_factor)))
    residual_ar = {}
    for name in ("vp", "por"):
        r = logs[name] - est[name][:, 0]
        ok = np.isfinite(r)
        for f in np.unique(fac):
            sel = ok & (fac == f)
            r[sel] -= r[sel].mean()
        phi, sig = yule_walker(r[ok], order)
        residual_ar[name] = dict(ar=phi.tolist(), sigma=sig, std=float(r[ok].std()))
    return dict(
        well="2X", window_twt=[float(ml_t[0]), float(ml_t[-1])], ml_depth=list(ML_DEPTH),
        ar_order=order, sand_phi_max=sand_phi_max, sand_factor=sand_factor,
        residual_ar=residual_ar, shale_vp=shale_vp,
        shale_vsh=fitted(logs["vsh"]),
        sand_vsh=vsh_stats(logs["vsh"][fac == SAND], "1 ms log"),
        silt_vsh=vsh_stats(w["vsh1"][w["litho1"] == SILT], "full-resolution log"),
        calc_vsh=vsh_stats(w["vsh1"][w["litho1"] == CALC], "full-resolution log"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--well-mat", default=str(DATA_DIR / "well2x_data.mat"))
    ap.add_argument("--out", default=str(DATA_DIR / "rock_physics_2x.json"))
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--sand-phi-max", type=float, default=0.18)
    args = ap.parse_args()
    cal = calibrate(args.well_mat, args.order, args.sand_phi_max)
    Path(args.out).write_text(json.dumps(cal, indent=1))
    print(f"sand porosity trend x {cal['sand_factor']:.4f} (maximum {cal['sand_phi_max']} in the 2X sand)")
    for k, c in cal["residual_ar"].items():
        print(f"residual {k:4s} AR " + ", ".join(f"{v:.3f}" for v in c["ar"]) +
              f"  innovation {c['sigma']:.4g}  std {c['std']:.4g}")
    for k in ("shale_vp", "shale_vsh", "silt_vsh", "sand_vsh", "calc_vsh"):
        c = cal[k]
        print(f"{k:9s} mean {c['mean']:9.4g}  std {c['std']:8.4g}  AR " +
              ", ".join(f"{v:.3f}" for v in c["ar"]) + f"  innovation {c['sigma']:.4g}  "
              f"({c['n_samples']} samples, {c['source']})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
