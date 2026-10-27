"""Manuscript figures of the two CNNs (regression and facies classification), on the pseudo-wells.

Regression CNN (train_cnn.py: Vp, Vs, rho, porosity, Vshale):
  training_curve                       MSE, training and validation
  training_result_elasto_petrophysics  one training pseudo-well: logs and CNN prediction
  testing_result_elasto_petrophysics   one test pseudo-well (not seen in training)
  correlation_elasto_petrophysics      correlation coefficient per pseudo-well, training vs test
Facies classification CNN (classify_facies.py: shale, silt, sand, calcareous sand):
  facies_confusion                     confusion matrices, (a) training and (b) test pseudo-wells,
                                       tuned decision weights, synthetic EEI without augmentation

Both networks use the same split of the 100,000 pseudo-wells (numpy default_rng(seed).permutation:
10 % test, 10 % validation, 80 % training), which is rebuilt here. Predictions are deterministic
(dropout off), as in the metrics of train_cnn.py and classify_facies.py.

Usage (after notebook 02, which also calls this script):
  python make_training_figures.py --reg-run ../Notebooks/outputs/CNN/regression_s0 \
      --facies-run ../Notebooks/outputs/CNN/facies_s0 --train-dir ../Notebooks/outputs/CNN/training_data \
      --outdir ../Notebooks/outputs/CNN/regression_s0/figures
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cnn_common import PROPS, aux_channels
from cnn_model import build_model
from classify_facies import K, confusion, make_model, scores

TRACKS = (("vp", "Vp (m/s)"), ("vs", "Vs (m/s)"), ("rho", r"$\rho$ (g/cc)"),
          ("poro", r"$\phi$ (v/v)"), ("vshale", "Vshale (v/v)"))
BOX_ORDER = (("vp", "Vp"), ("vs", "Vs"), ("rho", r"$\rho$"), ("poro", r"$\phi$"), ("vshale", "Vshale"))
FACIES_LABELS = ("Shale", "Silt", "Sand", "Cal. sand")


def masked_corr(a, b, m):
    """Correlation coefficient of a and b along the last axis, over the samples where m."""
    n = m.sum(-1)
    am = np.where(m, a, 0).sum(-1) / n
    bm = np.where(m, b, 0).sum(-1) / n
    da, db = np.where(m, a - am[:, None], 0), np.where(m, b - bm[:, None], 0)
    sab, saa, sbb = (da * db).sum(-1), (da * da).sum(-1), (db * db).sum(-1)
    ok = (saa > 1e-12 * n) & (sbb > 1e-12 * n)
    return np.where(ok, sab / np.sqrt(np.where(ok, saa * sbb, 1)), np.nan)


def save(fig, path, formats, dpi):
    for fmt in formats:
        fig.savefig(path.with_suffix("." + fmt), dpi=dpi)
    plt.close(fig)


@torch.no_grad()
def predict_reg(ck, model, X, meta, idx, device, bs=4096):
    L = ck["win_len"]
    out = []
    model.eval()
    for s in range(0, len(idx), bs):
        i = idx[s:s + bs]
        x = np.concatenate([X[i] / ck["s_chi"][None, :, None],
                            aux_channels(meta["ns"][i], meta["twt0"][i], L)], 1)
        out.append(model(torch.as_tensor(x, dtype=torch.float32, device=device))[0].float().cpu().numpy())
    mu = np.concatenate(out)
    return mu * ck["y_std"][None, :, None] + ck["y_mean"][None, :, None]


@torch.no_grad()
def predict_facies(fk, model, X, idx, x_rms, device, bs=4096):
    """Facies of the classification CNN: argmax of the weighted probabilities (as classify_facies.py)."""
    n = fk["length"]
    out = []
    model.eval()
    for s in range(0, len(idx), bs):
        x = torch.as_tensor(X[idx[s:s + bs], :, :n] / x_rms[None, :, None], dtype=torch.float32,
                            device=device)
        lg = model(x)[0] if fk["arch"] == "keras" else model(x)[2]
        p = lg.float().softmax(1).cpu().numpy()
        out.append((p * fk["decision_weights"][None, :, None]).argmax(1))
    return np.concatenate(out)


def well_plot(path, t, true, pred, formats, dpi, actual_label="Actual"):
    fig, axs = plt.subplots(1, 5, figsize=(13, 7.2), sharey=True)
    for ax, (i, (p, lbl)) in zip(axs, enumerate(TRACKS)):
        ax.plot(pred[i], t, "k", lw=1.6, label="Predicted")
        ax.plot(true[i], t, "r--", lw=1.4, label=actual_label)
        ax.set_xlabel(lbl, fontsize=15)
        ax.tick_params(labelsize=11)
    axs[0].set_ylim(t[-1], t[0])
    axs[0].set_ylabel("Time (sec)", fontsize=15)
    axs[2].legend(loc="lower left", fontsize=12)
    fig.tight_layout()
    save(fig, path, formats, dpi)


def confusion_plot(path, panels, formats, dpi):
    """Row-normalised confusion matrices (recall), with the sample counts."""
    fig, axs = plt.subplots(1, len(panels), figsize=(6.8 * len(panels), 5.8))
    for ax, (title, s) in zip(np.atleast_1d(axs), panels):
        cm = np.array(s["confusion"], float)
        rows = cm.sum(1, keepdims=True)
        cmn = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(cm)):
            for j in range(len(cm)):
                txt = f"{cmn[i, j]:.2f}\n({int(cm[i, j]):,})" if rows[i, 0] > 0 else "-"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color="w" if cmn[i, j] > 0.55 else "k")
        k = np.arange(len(cm))
        ax.set_xticks(k)
        ax.set_xticklabels(FACIES_LABELS, fontsize=12)
        ax.set_yticks(k)
        ax.set_yticklabels([f"{l}\n(n={int(r):,})" for l, r in zip(FACIES_LABELS, rows[:, 0])],
                           fontsize=11)
        ax.set_xlabel("Predicted facies", fontsize=14)
        ax.set_ylabel("Actual facies", fontsize=14)
        ax.set_title(f"{title}\naccuracy {s['accuracy']:.2f}, balanced accuracy "
                     f"{s['balanced_accuracy']:.2f}, macro-F1 {s['macro_f1']:.2f}", fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fraction of actual facies")
    fig.tight_layout()
    save(fig, path, formats, dpi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reg-run", required=True)
    ap.add_argument("--facies-run", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=None, help="default: seed in metrics.json")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--actual-label", default="Actual",
                    help='legend of the pseudo-well logs, e.g. "Actual, high-cut 60 Hz"')
    args = ap.parse_args()
    reg, fac, td, out = Path(args.reg_run), Path(args.facies_run), Path(args.train_dir), Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rm = json.loads((reg / "metrics.json").read_text())["args"]
    fm = json.loads((fac / "metrics.json").read_text())
    seed = rm["seed"] if args.seed is None else args.seed
    assert fm["args"]["seed"] == seed and fm["args"]["test_frac"] == rm["test_frac"] \
        and fm["args"]["val_frac"] == rm["val_frac"], "the two networks must share the split"
    X = np.load(td / "eei.npy")
    Y = np.load(td / "props.npy")
    Fac = np.load(td / "facies.npy")
    meta = np.load(td / "meta.npz")
    N = len(X)
    perm = np.random.default_rng(seed).permutation(N)
    n_te, n_va = int(rm["test_frac"] * N), int(rm["val_frac"] * N)
    sets = dict(train=perm[n_te + n_va:], test=perm[:n_te])
    print(f"pseudo-wells: training {len(sets['train']):,}, validation {n_va:,}, test {n_te:,}")

    ck = torch.load(reg / "model.pt", map_location=device, weights_only=False)
    rnet = build_model(ck["model_args"]).to(device)
    rnet.load_state_dict(ck["state_dict"])
    fk = torch.load(fac / "model.pt", map_location=device, weights_only=False)
    fnet = make_model(fk["arch"], fk["n_chi"], fk["length"], fk["p_drop"], fk["width"]).to(device)
    fnet.load_state_dict(fk["state_dict"])
    n = fk["length"]
    x_rms = np.sqrt((X[sets["train"], :, :n] ** 2).mean((0, 2)))      # as classify_facies.py

    # ---- training curve of the regression CNN -------------------------------------------
    h = np.loadtxt(reg / "history.csv", delimiter=",", skiprows=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(h[:, 0], h[:, 2], lw=2, label="train loss")
    ax.plot(h[:, 0], h[:, 5], lw=2, label="validation loss")
    ax.set_xlabel("Epochs", fontsize=15)
    ax.set_ylabel("Mean square error", fontsize=15)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=13)
    fig.tight_layout()
    save(fig, out / "training_curve", args.formats, args.dpi)

    # ---- both networks on the training and test pseudo-wells -----------------------------
    res = {}
    for name, idx in sets.items():
        mu = predict_reg(ck, rnet, X, meta, idx, device)
        m = Fac[idx] >= 0
        r = {p: masked_corr(Y[idx, i], mu[:, i], m) for i, p in enumerate(PROPS)}
        fp = predict_facies(fk, fnet, X, idx, x_rms, device)
        cls = scores(confusion(Fac[idx, :n].ravel(), fp.ravel()))
        res[name] = dict(idx=idx, mu=mu, r=r, facies=cls)

    fig, ax = plt.subplots(figsize=(7, 5.2))
    pos = np.arange(len(BOX_ORDER))
    for off, c, name, lbl in ((-0.18, "b", "train", "Training"), (0.18, "r", "test", "Test")):
        d = [res[name]["r"][p][np.isfinite(res[name]["r"][p])] for p, _ in BOX_ORDER]
        bp = ax.boxplot(d, positions=pos + off, widths=0.3, showfliers=False)
        for key in bp:
            plt.setp(bp[key], color=c, lw=1.3)
        ax.plot([], [], c, lw=2, label=lbl)
    ax.set_xticks(pos)
    ax.set_xticklabels([l for _, l in BOX_ORDER], fontsize=15)
    ax.set_ylabel("Correlation coefficient", fontsize=15)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(loc="lower left", fontsize=13)
    fig.tight_layout()
    save(fig, out / "correlation_elasto_petrophysics", args.formats, args.dpi)

    # one training and one test pseudo-well: with sand, and whose mean correlation is the
    # median of such pseudo-wells (a typical, not a best-case, example)
    summary = {}
    for name, fname in (("train", "training_result_elasto_petrophysics"),
                        ("test", "testing_result_elasto_petrophysics")):
        R = res[name]
        sand = (Fac[R["idx"]] == 2).mean(1)
        rmean = np.nanmean(np.stack([R["r"][p] for p in PROPS]), 0)
        cand = np.where(sand > 0.1)[0]
        j = cand[np.argsort(rmean[cand])[len(cand) // 2]]
        k = R["idx"][j]
        m = Fac[k] >= 0
        t = meta["twt0"][k] + np.arange(Y.shape[2]) * 0.001
        well_plot(out / fname, t[m], Y[k][:, m], R["mu"][j][:, m], args.formats, args.dpi,
                  args.actual_label)
        summary[name] = dict(example_pseudowell=int(k), n_pseudowells=len(R["idx"]),
                             median_r={p: float(np.nanmedian(R["r"][p])) for p in PROPS},
                             facies=R["facies"])

    # ---- facies classification CNN: confusion matrices -------------------------------------
    confusion_plot(out / "facies_confusion",
                   [(f"(a) Training pseudo-wells ({len(sets['train']):,})", res["train"]["facies"]),
                    (f"(b) Test pseudo-wells ({n_te:,})", res["test"]["facies"])],
                   args.formats, args.dpi)
    ref = fm["test"]["clean"]["tuned"]["confusion"]
    print("test confusion identical to classify_facies.py metrics.json:",
          res["test"]["facies"]["confusion"] == ref)

    (out / "training_figures_summary.json").write_text(json.dumps(summary, indent=1))
    for name, s in summary.items():
        f = s["facies"]
        print(f"{name:5s} (n={s['n_pseudowells']:,}, example #{s['example_pseudowell']}): median r " +
              " ".join(f"{p} {v:.3f}" for p, v in s["median_r"].items()) +
              f"; facies accuracy {f['accuracy']:.3f}, balanced {f['balanced_accuracy']:.3f}, "
              f"macro-F1 {f['macro_f1']:.3f}, recall " +
              " ".join(f"{v:.2f}" for v in f["recall"].values()))
    print(f"wrote figures to {out}/")


if __name__ == "__main__":
    main()
