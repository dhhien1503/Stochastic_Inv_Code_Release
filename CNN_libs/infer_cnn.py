"""Apply the trained EEI -> elasto-petrophysics CNN to the field line.

Builds the field EEI at the training angles from the AVO intercept/gradient
volumes (cnn_common.field_components), applies the intercept/gradient gains
fitted at training time, and runs the network with Monte Carlo dropout.

Uncertainty: each pass t gives a mean mu_t and an aleatoric variance s_t^2 per
sample. The predictive mean is mean_t(mu_t), the predictive variance
mean_t(s_t^2) + var_t(mu_t) (aleatoric + epistemic), and the quantiles come
from samples of the Gaussian mixture over the passes. Facies probabilities
are averaged over the passes.

Outputs in --outdir (SEG-Y on the same 1 ms grid and alignment as
odisi_inversion.py, so make_paper_figures.py reads them directly):
  <prop>_<stat>_<tag>.segy   prop: vp vs rho poro vshale; stat: mean std q05 q25 q75 q95
  facies_<tag>.segy          most probable facies (0 shale, 1 silt, 2 sand, 3 calc. sand)
  prob_<facies>_<tag>.segy   facies probabilities
  cnn_<tag>.npz              everything above as arrays, plus the input EEI
  sections_<tag>.png         mean sections

Usage:
  python infer_cnn.py --run results/regression_s0 --mc 100 --tag cnn_s0
  python make_paper_figures.py --input-dir results/regression_s0/field --tag cnn_s0 \
      --run-name CNN_s0 --outdir results/regression_s0/figures
"""
import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn_common import (aux_channels, field_components, field_eei, field_eei_4ms, geometry,
                        stretch_field, unstretch)
from paths import DATA_DIR
from cnn_model import build_model, enable_keras_training_mode, enable_mc_dropout
from pseudowell.plotting import NEW_CMAP
from pseudowell.segy_io import resolve_path, write_segy
from odisi_inversion import CFG

QUANTILES = dict(q05=0.05, q25=0.25, q75=0.75, q95=0.95)
BOUNDED = ("poro", "vshale")          # clipped to [0, 1]


@torch.no_grad()
def run_model(model, x, mc, samples_per_pass, seed, bn_batch_stats=False):
    """Returns mean, std, quantiles (standardised units) and facies probabilities."""
    g = torch.Generator(device=x.device).manual_seed(seed)
    if bn_batch_stats:
        enable_keras_training_mode(model)
    elif mc > 1:
        enable_mc_dropout(model)
    else:
        model.eval()
    mus, vars_, probs, draws = [], [], [], []
    for _ in range(max(mc, 1)):
        mu, lv, lg = model(x)
        var = lv.exp()
        mus.append(mu)
        vars_.append(var)
        probs.append(lg.softmax(1))
        e = torch.randn((samples_per_pass,) + mu.shape, device=x.device, generator=g)
        draws.append(mu + var.sqrt() * e)
    mu_t = torch.stack(mus)
    mean = mu_t.mean(0)
    std = (torch.stack(vars_).mean(0) + mu_t.var(0, unbiased=False)).sqrt()
    # quantiles by sorting (torch.quantile is limited to 2^24 elements)
    draws = torch.cat(draws).sort(0).values
    n = draws.shape[0]
    q = []
    for p in QUANTILES.values():
        i = p * (n - 1)
        lo, w = int(np.floor(i)), i - np.floor(i)
        q.append(draws[lo] * (1 - w) + draws[min(lo + 1, n - 1)] * w)
    return mean, std, torch.stack(q), torch.stack(probs).mean(0)


def stretch_input(args, ck, fixed, L):
    """Field EEI for a network trained on one fixed window (make_datamat.py): every field
    window is resampled onto the training window and standardised per angle with its own
    mean and std over the line."""
    from pseudowell.segy_io import read_segy
    from odisi_inversion import layer_bounds
    trc, _ = read_segy(resolve_path(CFG["horizon_file"], args.data_dir))
    nt = trc.shape[0]
    live = trc != 0
    i_first = live.argmax(axis=0)
    i_last = nt - 1 - live[::-1].argmax(axis=0)
    b = layer_bounds(CFG["layers"], i_first, i_last)
    i0, i1 = b[0][0], b[-1][1]                     # top of macro layer 2 .. base of macro layer 4
    n_in = fixed["ns"] if fixed else L
    gains = getattr(args, "stretch_gains_ab", (1.0, 1.0))
    eei4 = field_eei_4ms(args.data_dir, ck["chis"], tuple(ck["band"]), gains=gains)
    S = stretch_field(eei4, i0, i1, n_in)
    S = (S - S.mean((0, 2), keepdims=True)) / S.std((0, 2), keepdims=True)
    X = np.zeros((S.shape[0], S.shape[1], L), np.float32)
    X[..., :n_in] = S
    nx = S.shape[0]
    twt0 = fixed["twt0"] if fixed else 0.0
    aux = aux_channels(np.full(nx, n_in), np.full(nx, twt0), L)
    print(f"stretch: field windows of {4 * (i1 - i0).min()}-{4 * (i1 - i0).max()} ms "
          f"resampled onto {n_in} samples"
          + (f", intercept/gradient gains {gains[0]:.3f}/{gains[1]:.3f}" if gains != (1.0, 1.0) else ""))
    return X, dict(i0=np.asarray(i0), i1=np.asarray(i1), ns=None,
                   twt0=(np.asarray(i0) + 1) * CFG["dt"], top=None), aux


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="results/regression_s0", help="folder with model.pt")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="field SEG-Y folder")
    ap.add_argument("--outdir", default=None, help="default: <run>/field")
    ap.add_argument("--tag", default="CNN_EEI25")
    ap.add_argument("--mc", type=int, default=50, help="MC-dropout passes (1 = deterministic)")
    ap.add_argument("--samples-per-pass", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grid-shift", action="store_true",
                    help="as in odisi_inversion.py: no 3-sample shift of the 1 ms grid")
    ap.add_argument("--calibration", choices=("train", "angle"), default="train",
                    help="train: intercept/gradient gains fitted at training time (default); "
                         "angle: scale the field EEI to unit rms at every angle")
    ap.add_argument("--gains", type=float, nargs=2, metavar=("GA", "GB"),
                    help="override the intercept and gradient gains")
    ap.add_argument("--field-mode", choices=("auto", "window", "stretch"), default="auto",
                    help="window: each trace's ODiSI window on the 1 ms grid, zero padded; "
                         "stretch: each field window (top of macro layer 2 .. base of macro layer 4) "
                         "resampled onto the single training window of make_datamat.py; "
                         "auto: stretch when the training set has one fixed window")
    ap.add_argument("--bn-batch-stats", action="store_true",
                    help="Keras model(X, training=True): BatchNorm uses the field batch statistics")
    ap.add_argument("--stretch-gains", choices=("none", "fit"), default="none",
                    help="stretch mode: fit = scale the intercept and gradient by the gains fitted at "
                         "training time (field EEI power vs pseudo-well EEI power over all angles, no "
                         "well data; --gains overrides); the per-angle standardisation removes the "
                         "overall scale but not the intercept/gradient ratio, which sets the angle at "
                         "which the EEI changes polarity")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    out = Path(args.outdir or run / "field")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run / "model.pt", map_location=device, weights_only=False)
    props, facies = list(ck["props"]), list(ck["facies"])
    cal = ck["calibration"]
    L = ck["win_len"]

    # ---- field input -----------------------------------------------------------------
    t0 = time.time()
    fixed = ck.get("fixed_window")
    stretch = args.field_mode == "stretch" or (args.field_mode == "auto" and fixed is not None)
    if stretch:
        if args.gains:
            args.stretch_gains_ab = tuple(args.gains)
        elif args.stretch_gains == "fit":
            args.stretch_gains_ab = (cal["gain_a_nonoise"], cal["gain_b_nonoise"])
        else:
            args.stretch_gains_ab = (1.0, 1.0)
        X, comp, aux = stretch_input(args, ck, fixed, L)
        cal = dict(cal, mode="stretch (ODiSI window), per-angle standardisation",
                   gain_a=args.stretch_gains_ab[0], gain_b=args.stretch_gains_ab[1])
    else:
        comp = field_components(args.data_dir, ck["band"])
        aux = aux_channels(comp["ns"], comp["twt0"], L)
    if stretch:
        pass                      # already standardised per angle in stretch_input
    elif args.calibration == "angle":
        # one amplitude scale per angle (as the single-angle ODiSI scale), which
        # keeps the field waveform at each angle; ignores the fitted gains
        X = field_eei(comp, ck["chis"])
        m = aux[:, :1] > 0
        rms = np.sqrt((X ** 2 * m).sum((0, 2)) / m.sum())
        X = (X / rms[None, :, None]).astype(np.float32)
        cal = dict(cal, mode="angle (inference)")
    else:
        ga, gb = args.gains if args.gains else (cal["gain_a"], cal["gain_b"])
        X = field_eei(comp, ck["chis"], ga, gb)
        X = X / ck["s_chi"][None, :, None].astype(np.float32)
        cal = dict(cal, gain_a=ga, gain_b=gb)
    X = X * aux[:, :1]
    x = torch.as_tensor(np.concatenate([X, aux], 1), device=device)
    print(f"field EEI: {X.shape[0]} traces x {X.shape[1]} angles "
          f"({ck['chis'][0]:g}..{ck['chis'][-1]:g} deg); calibration {cal['mode']}: "
          f"intercept gain {cal['gain_a']:.3f}, gradient gain {cal['gain_b']:.3f}")

    model = build_model(ck["model_args"]).to(device)
    model.load_state_dict(ck["state_dict"])
    t1 = time.time()
    mean, std, q, prob = run_model(model, x, args.mc, args.samples_per_pass, args.seed,
                                   args.bn_batch_stats)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time() - t1
    print(f"inference: {args.mc} MC-dropout passes in {t_inf:.2f} s "
          f"(input preparation {t1 - t0:.1f} s)")

    ym = ck["y_mean"][None, :, None]
    ys = ck["y_std"][None, :, None]
    res = dict(mean=mean.cpu().numpy() * ys + ym, std=std.cpu().numpy() * ys)
    for i, k in enumerate(QUANTILES):
        res[k] = q[i].cpu().numpy() * ys + ym
    for i, p in enumerate(props):
        if p in BOUNDED:
            for k in ("mean", *QUANTILES):
                res[k][:, i] = res[k][:, i].clip(0, 1)
    prob = prob.cpu().numpy()                          # (nx, n_facies, L)
    if stretch:                                        # back onto each trace's 1 ms window
        n_in = fixed["ns"] if fixed else L
        for k in list(res):
            res[k], _ = unstretch(res[k][..., :n_in], comp["i0"], comp["i1"], n_in)
        prob, comp["ns"] = unstretch(prob[..., :n_in], comp["i0"], comp["i1"], n_in)
        comp["top"] = comp["i0"]
    fac = prob.argmax(1)

    # ---- back onto the 1 ms grid of the line ----------------------------------------------
    _, _, nt, nx = geometry(args.data_dir)
    q4 = int(round(CFG["dt"] / CFG["ds"]))
    shift = 0 if args.no_grid_shift else q4 - 1
    tmpl = resolve_path(CFG["horizon_file"], args.data_dir)

    def to_grid(a):                                    # a: (nx, L) -> (q4*nt, nx)
        v = np.zeros((q4 * nt, nx), np.float32)
        for k in range(nx):
            s = q4 * comp["top"][k]
            v[s:s + comp["ns"][k], k] = a[k, :comp["ns"][k]]
        if shift:
            v[shift:] = v[:-shift].copy()
            v[:shift] = 0
        return v

    vols = {}
    for st in ("mean", "std", *QUANTILES):
        for i, p in enumerate(props):
            vols[f"{p}_{st}"] = to_grid(res[st][:, i])
            write_segy(out / f"{p}_{st}_{args.tag}.segy", vols[f"{p}_{st}"], CFG["ds"], tmpl)
    vols["facies"] = to_grid(fac.astype(np.float32))
    write_segy(out / f"facies_{args.tag}.segy", vols["facies"], CFG["ds"], tmpl)
    for j, f in enumerate(facies):
        key = "prob_" + f.replace(" ", "_")
        vols[key] = to_grid(prob[:, j])
        write_segy(out / f"{key}_{args.tag}.segy", vols[key], CFG["ds"], tmpl)
    np.savez_compressed(out / f"cnn_{args.tag}.npz", eei_input=X, chis=ck["chis"],
                        ns=comp["ns"], twt0=comp["twt0"], top=comp["top"],
                        inference_time_s=t_inf, mc=args.mc, **vols)
    print(f"wrote SEG-Y and cnn_{args.tag}.npz to {out}/")
    if args.no_plots:
        return

    # ---- quick-look sections -------------------------------------------------------------
    t1ms = (np.arange(q4 * nt) + 1) * CFG["ds"]
    x_m = (np.arange(nx) + 1) * CFG["dx"]
    live = vols["vp_mean"] != 0
    rows = np.where(live.any(1))[0]
    ylim = (t1ms[rows[-1]], t1ms[rows[0]])
    panels = [("vp_mean", "Mean Vp (m/s)", (2800, 4300)), ("vs_mean", "Mean Vs (m/s)", (1500, 2300)),
              ("rho_mean", r"Mean $\rho$ (g/cc)", (2.45, 2.7)),
              ("poro_mean", "Mean porosity", (0, 0.18)), ("vshale_mean", "Mean Vshale", (0, 1)),
              ("facies", "Most probable facies", (0, 3)),
              ("vp_std", "Std Vp (m/s)", None), ("poro_std", "Std porosity", None),
              ("prob_sand", "P(sand)", (0, 1))]
    fig, axs = plt.subplots(3, 3, figsize=(16, 11))
    for ax, (k, title, clim) in zip(axs.ravel(), panels):
        a = np.where(live, vols[k], np.nan)
        im = ax.imshow(a, aspect="auto", cmap=NEW_CMAP, interpolation="nearest",
                       extent=[x_m[0], x_m[-1], t1ms[-1], t1ms[0]],
                       vmin=None if clim is None else clim[0], vmax=None if clim is None else clim[1])
        fig.colorbar(im, ax=ax)
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("TWT (s)")
    fig.tight_layout()
    fig.savefig(out / f"sections_{args.tag}.png", dpi=150)
    plt.close(fig)
    print(f"wrote sections_{args.tag}.png")


if __name__ == "__main__":
    main()
