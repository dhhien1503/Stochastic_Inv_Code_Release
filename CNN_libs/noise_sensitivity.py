"""Sensitivity of regression CNNs trained on different EEI angles to noise in the AVO gradient.

The field EEI is a combination of the AVO intercept A and gradient B, EEI(chi) ~ A cos(chi) +
B sin(chi), and B is the noisier of the two. Here the synthetic EEI of the test pseudo-wells is
perturbed as EEI(chi) + s * rms(b) * w * sin(chi), where b is the gradient term of the
least-squares fit a cos(chi) + b sin(chi) of the 4-angle EEI (per sample), w a band-limited
(3-60 Hz) random trace of unit rms and s the noise level. The same w is used for every network,
so networks with one, four or 25 angles see the same perturbation of the same pseudo-wells.

Usage:
  python noise_sensitivity.py --runs ../Notebooks/outputs/CNN/regression_s0 \
      ../Notebooks/outputs/CNN/ablation_* --ref-train-dir ../Notebooks/outputs/CNN/training_data \
      --out ../Notebooks/outputs/CNN/noise_sensitivity.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cnn_common import PROPS, aux_channels, band_response
from cnn_model import build_model


def two_term_b(X, chis):
    """Gradient term b of the per-sample fit a cos(chi) + b sin(chi); X (n, nchi, L)."""
    c = np.deg2rad(chis)
    G = np.stack([np.cos(c), np.sin(c)], 1)
    return np.einsum("j,njl->nl", np.linalg.pinv(G)[1], X)


def masked_median_corr(a, b, m):
    out = []
    for k in range(a.shape[1]):
        r = []
        for j in range(len(a)):
            x, y = a[j, k, m[j]], b[j, k, m[j]]
            r.append(np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else np.nan)
        out.append(float(np.nanmedian(r)))
    return out


@torch.no_grad()
def predict(ck, model, X, meta, idx, device, bs=4096):
    L = ck["win_len"]
    model.eval()
    out = []
    for s in range(0, len(idx), bs):
        i = idx[s:s + bs]
        x = np.concatenate([X[s:s + bs] / ck["s_chi"][None, :, None],
                            aux_channels(meta["ns"][i], meta["twt0"][i], L)], 1)
        out.append(model(torch.as_tensor(x, dtype=torch.float32, device=device))[0].float().cpu().numpy())
    return np.concatenate(out) * ck["y_std"][None, :, None] + ck["y_mean"][None, :, None]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ref-train-dir", required=True, help="4-angle training set (defines b)")
    ap.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ref = Path(args.ref_train_dir)
    meta = np.load(ref / "meta.npz")
    Fac = np.load(ref / "facies.npy")
    Y = np.load(ref / "props.npy")
    N, L = Fac.shape
    n_te = int(0.1 * N)
    idx = np.random.default_rng(0).permutation(N)[:n_te]          # test pseudo-wells (split seed 0)
    m = Fac[idx] >= 0
    b = two_term_b(np.load(ref / "eei.npy", mmap_mode="r")[idx], meta["chis"]) * m
    b_rms = np.sqrt((b ** 2).sum() / m.sum())
    rng = np.random.default_rng(args.seed)
    w = np.fft.irfft(np.fft.rfft(rng.standard_normal((n_te, L))) * band_response(L, (3.0, 60.0)), n=L)
    w = w / np.sqrt((w ** 2).mean(1, keepdims=True))

    res = {}
    for run in args.runs:
        run = Path(run)
        ck = torch.load(run / "model.pt", map_location=device, weights_only=False)
        model = build_model(ck["model_args"]).to(device)
        model.load_state_dict(ck["state_dict"])
        td = Path(json.loads((run / "metrics.json").read_text())["args"]["train_dir"])
        if not td.is_absolute():
            td = (run / ".." / ".." / td).resolve() if not td.exists() else td
        X0 = np.load(td / "eei.npy", mmap_mode="r")[idx].astype(np.float32)
        sinc = np.sin(np.deg2rad(ck["chis"]))[None, :, None]
        res[run.name] = {"chis": [float(c) for c in ck["chis"]]}
        for s in args.levels:
            X = X0 + (s * b_rms * w[:, None, :] * sinc * m[:, None, :]).astype(np.float32)
            mu = predict(ck, model, X, meta, idx, device)
            res[run.name][f"{s:g}"] = dict(zip(PROPS, masked_median_corr(Y[idx], mu, m)))
            print(f"{run.name:22s} noise {s:4g}: " +
                  " ".join(f"{p} {v:.3f}" for p, v in res[run.name][f'{s:g}'].items()), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
