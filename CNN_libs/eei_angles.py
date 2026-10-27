"""Training set with the EEI of the same pseudo-wells at other angles (EEI-angle ablation,
notebooks 03-07).

Reads a training set written by make_datamat.py (props.npy: Vp, Vs, rho, porosity, Vshale of
every pseudo-well) and recomputes the EEI at the angles --chis exactly as make_datamat.py
(eeimp2 + Butterworth 3-60 Hz, filtfilt). props.npy and facies.npy are the same as in the input
set and are hard-linked (copied if links are not possible).

Usage:
  python eei_angles.py --train-dir ../Notebooks/outputs/CNN/training_data --chis -7.5 \\
      --outdir ../Notebooks/outputs/CNN/training_data_chi_m7.5
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from cnn_common import WIN_LEN
from odisi_inversion import CFG
from pseudowell.seismic import butter_bandpass, eeimp2, filtfilt_odd

BAND = (3.0, 60.0)


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-dir", required=True, help="training set of make_datamat.py")
    ap.add_argument("--chis", type=float, nargs="+", required=True, help="EEI angles (deg)")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    src, out = Path(args.train_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    Y = np.load(src / "props.npy", mmap_mode="r")
    meta = dict(np.load(src / "meta.npz"))
    n, ns = len(Y), int(meta["ns"][0])
    assert np.ptp(meta["ns"]) == 0, "one fixed window expected (make_datamat.py)"
    vp, vs, rho = (np.asarray(Y[:, i, :ns], float).T for i in range(3))     # (ns, N)
    b_f, a_f = butter_bandpass(BAND[0], BAND[1], 1.0 / CFG["ds"], 4)
    X = np.zeros((n, len(args.chis), WIN_LEN), np.float32)
    for c, chi in enumerate(args.chis):
        X[:, c, :ns] = filtfilt_odd(b_f, a_f, eeimp2(vp, vs, rho, chi), axis=0).T
    np.save(out / "eei.npy", X)
    for name in ("props.npy", "facies.npy"):
        link_or_copy(src / name, out / name)
    meta["chis"] = np.array(args.chis)
    np.savez(out / "meta.npz", **meta)
    cfg = json.loads((src / "config.json").read_text())
    cfg.update(chis=list(args.chis), source=str(src.resolve()), eei_recomputed=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"{n} pseudo-wells x {len(args.chis)} angles x {ns} ms -> {out}/")


if __name__ == "__main__":
    main()
