"""CNN training set: pseudo-wells generated with the prior of ODiSI (same facies model and rock
physics, Table 2 of the manuscript).

Steps:
  1. macro layers ML1-ML3 of every trace from live_window.sgy (first+5, first+17, last-19,
     last-5); the pseudo-wells are simulated once, on the window of the thickest trace
     (184 samples at 1 ms, TWT 2.828-3.011 s)
  2. facies: Markov chain per macro layer (ML1 and ML3: T2/lambda2, ML2: T3/lambda3, Table 1),
     with carry-over of the initial-facies probabilities (pseudowell/markov.py)
  3. rock physics: pseudowell/rock_physics.pseudo_well_properties (relations of Table 2 in
     two-way time, AR(4) fluctuations and facies Vshale calibrated on Well 2X)
  4. EEI(chi) = eeimp2(vp, vs, rho, chi) per pseudo-well, Butterworth 3-60 Hz (order 4,
     filtfilt), chi = -60, -7.5, 25, 45 deg

Outputs in --outdir:
  data.mat               seis1..4, vp1, vs1, rho1, por1, vsh1 (184 x N), litho (184 x N, uint8,
                         0 shale 1 silt 2 sand 3 calc. sand), timev (1 x 184)  (--format mat or both)
  eei.npy, props.npy, facies.npy, meta.npz, config.json
                         training_data format for train_cnn.py / classify_facies.py
                                                              (--format npy or both)

Usage:
  python make_datamat.py --outdir training_data --n 100000
  python make_datamat.py --n 2000 --outdir /tmp/test --format npy     # quick test
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

from cnn_common import FACIES, PROPS, WIN_LEN, geometry
from paths import DATA_DIR
from pseudowell.markov import markov_chain_simulation_lambda
from pseudowell.rock_physics import pseudo_well_properties
from pseudowell.seismic import butter_bandpass, eeimp2, filtfilt_odd
from odisi_inversion import CFG

CHIS = (-60.0, -7.5, 25.0, 45.0)
BAND = (3.0, 60.0)


def bed_stats(f):
    """Mean thickness (samples) of the beds that start and end inside the segment."""
    out = {}
    for j in range(f.shape[1]):
        col = f[:, j]
        ch = np.flatnonzero(np.diff(col)) + 1
        for s, e in zip(np.r_[0, ch], np.r_[ch, len(col)]):
            if s > 0 and e < len(col):
                out.setdefault(int(col[s]), []).append(e - s)
    return {FACIES[k]: round(float(np.mean(v)), 2) for k, v in sorted(out.items())}


def summary(d, seg):
    lit = d["litho"]
    s = dict(layers={}, props={})
    for name, (a, b) in seg.items():
        m = lit[a:b]
        s["layers"][name] = dict(props=(np.bincount(m.ravel(), minlength=4) / m.size).round(4).tolist(),
                                 beds_ms=bed_stats(m[:, :5000]))
    for f in range(4):
        m = lit == f
        s["props"][FACIES[f]] = {k: [round(float(d[k][m].mean()), 4), round(float(d[k][m].std()), 4)]
                                 for k in ("vp1", "vs1", "rho1", "por1", "vsh1")}
    s["eei_rms"] = [round(float(np.sqrt((d[f"seis{c + 1}"] ** 2).mean())), 1) for c in range(4)]
    return s


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="folder with live_window.sgy, avo_intercept_A.sgy, avo_gradient_B.sgy")
    ap.add_argument("--outdir", default="training_data_python")
    ap.add_argument("--n", type=int, default=100000, help="number of pseudo-wells")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--format", choices=("mat", "npy", "both"), default="npy")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rock = CFG["rock"]
    t0 = time.time()

    # ---- 1. geometry of the thickest trace -------------------------------------------------
    tops, bases, _, _ = geometry(args.data_dir)
    k = int(np.argmax(bases[-1] - tops[0]))            # first maximum
    q = int(round(CFG["dt"] / CFG["ds"]))
    ns_layer = [q * int(b[k] - t[k]) for t, b in zip(tops, bases)]
    ns = sum(ns_layer)
    # times of the 1 ms samples of the window
    timev = (q * (int(tops[0][k]) + 1) + np.arange(ns)) * CFG["ds"]
    print(f"thickest trace {k}: layers {ns_layer} ms, TWT {timev[0]:.3f}-{timev[-1]:.3f} s")

    # ---- 2. facies ---------------------------------------------------------------------------
    # ML1 and ML3: T2/lambda2, ML2: T3/lambda3 (CFG["layers"])
    litho = np.vstack([markov_chain_simulation_lambda(L["T"], n_l, args.n, L["lam"], CFG["ds"],
                                                      seed=int(rng.integers(2 ** 31 - 1)), carry=True)
                       for L, n_l in zip(CFG["layers"], ns_layer)])
    print(f"facies {time.time() - t0:.0f} s")

    # ---- 3. rock physics (as ODiSI) ----------------------------------------------------------
    vp, vs, rho, por, vsh = pseudo_well_properties(timev, litho, rock, rng)
    print(f"rock physics {time.time() - t0:.0f} s")

    # ---- 4. EEI ------------------------------------------------------------------------------
    b_f, a_f = butter_bandpass(BAND[0], BAND[1], 1.0 / CFG["ds"], 4)
    seis = [filtfilt_odd(b_f, a_f, eeimp2(vp, vs, rho, chi), axis=0) for chi in CHIS]
    print(f"EEI {time.time() - t0:.0f} s")

    d = dict(seis1=seis[0], seis2=seis[1], seis3=seis[2], seis4=seis[3], vp1=vp, vs1=vs, rho1=rho,
             por1=por, vsh1=vsh, litho=litho.astype(np.uint8), timev=timev[None])
    if args.format in ("mat", "both"):
        sio.savemat(out / "data.mat", d, do_compression=False)
    if args.format in ("npy", "both"):
        X = np.zeros((args.n, len(CHIS), WIN_LEN), np.float32)
        Y = np.zeros((args.n, len(PROPS), WIN_LEN), np.float32)
        F = np.full((args.n, WIN_LEN), -1, np.int8)
        for c in range(len(CHIS)):
            X[:, c, :ns] = seis[c].T
        for i, a in enumerate((vp, vs, rho, por, vsh)):
            Y[:, i, :ns] = a.T
        F[:, :ns] = litho.T
        np.save(out / "eei.npy", X)
        np.save(out / "props.npy", Y)
        np.save(out / "facies.npy", F)
        np.savez(out / "meta.npz", trace=np.full(args.n, k), ns=np.full(args.n, ns, np.int32),
                 twt0=np.full(args.n, timev[0]), chis=np.array(CHIS), band=np.array(BAND),
                 props=np.array(PROPS), facies=np.array(FACIES))
    seg, a = {}, 0
    for L, n_l in zip(CFG["layers"], ns_layer):
        seg[L["name"]] = (a, a + n_l)
        a += n_l
    stats = summary(d, seg)
    (out / "config.json").write_text(json.dumps(dict(
        n=args.n, seed=args.seed, trace=k, ns_layer=ns_layer,
        twt=[timev[0], timev[-1]], chis=CHIS, band=BAND, layers=CFG["layers"],
        rock=rock, fixed_window=True,
        stats=stats), indent=1))
    print(f"wrote {out}/ ({args.format}) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
