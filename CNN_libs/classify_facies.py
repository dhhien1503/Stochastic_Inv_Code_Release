"""Facies classification from multi-angle EEI, trained on the pseudo-wells of make_datamat.py.

The manuscript derives facies from the inverted Vshale with cut-offs (0.5 / 0.72), so it
has three classes, cannot produce calcareous sand and has no facies probabilities. This
script trains a dedicated classifier on the same 100,000 pseudo-wells
(training_data/, from make_datamat.py): EEI at -60, -7.5, 25, 45 deg -> one of
shale / silt / sand / calcareous sand for every 1 ms sample, with probabilities.

What it does differently from the regression networks:
  * plain cross-entropy, so the softmax outputs are calibrated probabilities under the
    prior; class imbalance (calcareous sand is 2.5 % of the samples) is handled in the
    decision rule, not in the loss: the class weights of argmax(w * p) are tuned on the
    validation pseudo-wells for macro-F1 (--decision macro_f1; --decision map = plain argmax)
  * augmentation for the field: random time stretch of the window (the field windows,
    152-184 ms, are stretched onto the 184-sample training window, up to 1.21x), random
    gain per angle and band-limited (3-60 Hz) noise
  * MC dropout at inference: facies probabilities averaged over the passes

The field EEI is prepared as in infer_cnn.py --field-mode stretch (A/B, 3-60 Hz, ODiSI window,
1 ms alignment).

Evaluation:
  * held-out pseudo-wells: accuracy, balanced accuracy, macro-F1, per-class F1, confusion
  * well 2X (trace --trace-2x, times1 of well2x_data.mat): a training well of the prior,
    used to choose settings
  * well 9P (blind; well path and window of make_paper_figures.py): reported, not tuned on
  (the well logs are confidential and not distributed: without the .mat files the wells
  are skipped)
  * baseline in the same metrics: Vshale cut-offs of a regression run (--baseline-run)

Outputs in --outdir: model.pt, metrics.json, summary.md, facies_<tag>.segy,
prob_<facies>_<tag>.segy, facies_sections.png, wells.png, confusion.png, training_curve.png

Usage:
  python classify_facies.py --train-dir training_data --arch keras --outdir results/facies_s0
  python classify_facies.py --train-dir training_data --arch keras --epochs 2 --max-train 5000 \
      --outdir results/facies_test                           # quick test
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from cnn_common import FACIES, band_response, unstretch
from paths import DATA_DIR
from cnn_model import EEINet, KerasNet, enable_mc_dropout
from infer_cnn import stretch_input
from pseudowell.segy_io import read_segy, resolve_path, write_segy
from odisi_inversion import CFG

FACIES_CMAP = ListedColormap([[0.5, 0.5, 0], [0.5, 0.5, 0.5], [1, 1, 0], [1, 0, 0]])
K = len(FACIES)
DX = 12.5
# well 9P path (distance and TWT of its two ends) and scored window, on the two-way time of the line
WELL9P_X, WELL9P_T, WIN9P = np.array([234, 255]) * DX, (2.8, 3.08), (2.886, 3.016)


# ---- metrics ------------------------------------------------------------------------------
def confusion(t, p):
    cm = np.zeros((K, K), int)
    np.add.at(cm, (np.asarray(t, int), np.asarray(p, int)), 1)
    return cm


def scores(cm):
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    present = cm.sum(1) > 0
    return dict(accuracy=float(np.trace(cm) / max(cm.sum(), 1)),
                balanced_accuracy=float(rec[present].mean()),
                macro_f1=float(f1[present].mean()),
                f1=dict(zip(FACIES, f1.round(4).tolist())),
                recall=dict(zip(FACIES, rec.round(4).tolist())),
                precision=dict(zip(FACIES, prec.round(4).tolist())),
                confusion=cm.tolist())


def tune_weights(prob, true, iters=3):
    """Class weights w maximising macro-F1 of argmax(w * p) (coordinate search, log grid)."""
    w = np.ones(K)
    grid = np.exp(np.linspace(np.log(0.2), np.log(20), 41))
    best = scores(confusion(true, (prob * w[:, None]).argmax(0)))["macro_f1"]
    for _ in range(iters):
        for k in range(1, K):                         # shale fixed at 1
            for g in grid:
                w2 = w.copy()
                w2[k] = g
                s = scores(confusion(true, (prob * w2[:, None]).argmax(0)))["macro_f1"]
                if s > best:
                    best, w = s, w2
    return w, best


# ---- training data and augmentation ---------------------------------------------------------
class Data:
    def __init__(self, X, Fac, device, band, n):
        self.X = torch.as_tensor(X[:, :, :n], device=device)
        self.F = torch.as_tensor(Fac[:, :n].astype(np.int64), device=device)
        self.H2 = torch.as_tensor(band_response(n, band), dtype=torch.float32, device=device)
        self.n, self.device = n, device

    def batch(self, idx, aug=None, g=None):
        x, f = self.X[idx], self.F[idx]
        if not aug:
            return x, f
        B, C, L = x.shape
        r = lambda *s: torch.rand(*s, device=self.device, generator=g)
        if aug["stretch"] > 1:
            # a sub-window of length L/s, resampled onto L samples (field windows shorter
            # than the training window are stretched the same way)
            s = 1 + (aug["stretch"] - 1) * r(B)
            w = (L - 1) / s
            off = r(B) * (L - 1 - w)
            pos = off[:, None] + torch.arange(L, device=self.device)[None] * (w / (L - 1))[:, None]
            i0 = pos.floor().long().clamp(0, L - 2)
            fr = (pos - i0)[:, None]
            gi = lambda a, i: a.gather(2, i[:, None].expand(-1, C, -1))
            x = gi(x, i0) * (1 - fr) + gi(x, i0 + 1) * fr
            f = f.gather(1, pos.round().long().clamp(0, L - 1))
        if aug["gain"] > 0:
            x = x * (1 + aug["gain"] * (2 * r(B, C, 1) - 1))
        if aug["noise"] > 0:
            nz = torch.randn(B, C, L, device=self.device, generator=g)
            nz = torch.fft.irfft(torch.fft.rfft(nz) * self.H2, n=L)
            nz = nz / nz.pow(2).mean(-1, keepdim=True).sqrt()
            x = x + aug["noise"] * r(B, 1, 1) * nz
        x = x / x.pow(2).mean((0, 2), keepdim=True).sqrt()     # unit rms per angle, as the field
        return x, f


def make_model(arch, n_chi, n, p_drop, width):
    if arch == "keras":
        return KerasNet(n_chi, K, K, n, p_drop)                # its "mu" output = K logits
    return EEINet(n_chi, 1, K, width, p_drop=p_drop)


def logits_of(model, arch, x):
    out = model(x)
    return out[0] if arch == "keras" else out[2]


@torch.no_grad()
def predict(model, arch, x, mc=0, bs=4096):
    if mc > 1:
        enable_mc_dropout(model)
    else:
        model.eval()
    out = []
    for s in range(0, len(x), bs):
        xb = x[s:s + bs]
        out.append(torch.stack([logits_of(model, arch, xb).float().softmax(1)
                                for _ in range(max(mc, 1))]).mean(0))
    model.train()
    return torch.cat(out).cpu().numpy()                        # (n, K, L)


# ---- field ----------------------------------------------------------------------------------
def to_grid(a, i0, ns, nt, shift):
    """(nx, L) on each trace's 1 ms window -> (4 nt, nx) on the line grid (infer_cnn.py)."""
    q = int(round(CFG["dt"] / CFG["ds"]))
    v = np.zeros((q * nt, a.shape[0]), np.float32)
    for k in range(a.shape[0]):
        v[q * i0[k]:q * i0[k] + ns[k], k] = a[k, :ns[k]]
    if shift:
        v[shift:] = v[:-shift].copy()
        v[:shift] = 0
    return v


def block_log(t_log, lab_log, t, half=0.0005):
    """Most frequent log facies within +-half of every time t (logs are ~0.05 ms)."""
    out = np.full(len(t), -1)
    for i, ti in enumerate(t):
        m = np.abs(t_log - ti) <= half
        if m.any():
            out[i] = np.bincount(lab_log[m].astype(int), minlength=K).argmax()
    return out


def along_path(vol_prob, live, x, t1, xs, ts):
    """Probabilities (K, n) and live flag along a path, linear in x and t."""
    pts = np.column_stack([ts, xs])
    P = np.stack([RegularGridInterpolator((t1, x), v, bounds_error=False, fill_value=np.nan)(pts)
                  for v in vol_prob])
    L = RegularGridInterpolator((t1, x), live.astype(float), method="nearest",
                                bounds_error=False, fill_value=0)(pts) > 0
    return P, L


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-dir", default="training_data")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--outdir", default="results/facies_s0")
    ap.add_argument("--tag", default=None, help="default: name of --outdir")
    ap.add_argument("--arch", choices=("unet", "keras"), default="unet")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--aug-stretch", type=float, default=1.25, help="max time-stretch factor (1 = off)")
    ap.add_argument("--aug-gain", type=float, default=0.2, help="max relative gain per angle")
    ap.add_argument("--aug-noise", type=float, default=0.2, help="max noise rms / signal rms")
    ap.add_argument("--decision", choices=("macro_f1", "map"), default="macro_f1")
    ap.add_argument("--mc", type=int, default=50, help="MC-dropout passes on the field")
    ap.add_argument("--trace-2x", type=int, default=0)
    ap.add_argument("--well2x-mat", default=str(DATA_DIR / "well2x_data.mat"))
    ap.add_argument("--well9p-mat", default=str(DATA_DIR / "well9p_data.mat"))
    ap.add_argument("--baseline-run", default=str(Path("results") / "regression_s0" / "field"),
                    help="folder with vshale_mean_<tag>.segy of a regression run (Vshale cut-off baseline)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or out.name
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data -------------------------------------------------------------------------------
    td = Path(args.train_dir)
    meta = np.load(td / "meta.npz")
    n = int(meta["ns"][0])
    assert np.ptp(meta["ns"]) == 0, "classify_facies.py expects one fixed window (data.mat)"
    X = np.load(td / "eei.npy")
    Fac = np.load(td / "facies.npy")
    chis, band = meta["chis"], tuple(meta["band"])
    N = len(X)
    perm = rng.permutation(N)
    n_te, n_va = int(args.test_frac * N), int(args.val_frac * N)
    idx_te, idx_va, idx_tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    if args.max_train:
        idx_tr = idx_tr[:args.max_train]
    X = X / np.sqrt((X[idx_tr, :, :n] ** 2).mean((0, 2)))[None, :, None]
    data = Data(X.astype(np.float32), Fac, device, band, n)
    del X
    prior = np.bincount(Fac[idx_tr, :n].ravel(), minlength=K) / (len(idx_tr) * n)
    print(f"{N} pseudo-wells x {len(chis)} angles x {n} ms; facies prior " +
          " ".join(f"{f} {p:.3f}" for f, p in zip(FACIES, prior)))

    # ---- training --------------------------------------------------------------------------
    model = make_model(args.arch, len(chis), n, args.dropout, args.width).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{args.arch}: {n_par / 1e6:.2f} M parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * int(np.ceil(len(idx_tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    aug = dict(stretch=args.aug_stretch, gain=args.aug_gain, noise=args.aug_noise)
    tr_t, va_t = torch.as_tensor(idx_tr, device=device), torch.as_tensor(idx_va, device=device)
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    hist, best = [], np.inf
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tot, nb = 0.0, 0
        for b in torch.randperm(len(tr_t), device=device).split(args.batch):
            x, f = data.batch(tr_t[b], aug)
            with torch.autocast(**amp):
                lg = logits_of(model, args.arch, x)
            loss = F.cross_entropy(lg.float(), f, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        model.eval()
        g = torch.Generator(device=device).manual_seed(123)
        vl, nv = 0.0, 0
        with torch.no_grad():
            for b in va_t.split(4096):
                x, f = data.batch(b, aug, g)                  # validation with the same augmentation
                with torch.autocast(**amp):
                    lg = logits_of(model, args.arch, x)
                vl += F.cross_entropy(lg.float(), f).item()
                nv += 1
        model.train()
        hist.append([ep, tot / nb, vl / nv])
        flag = ""
        if vl / nv < best:
            best, flag = vl / nv, " *"
            torch.save(model.state_dict(), out / "best_weights.pt")
        print(f"epoch {ep:3d}  train CE {tot / nb:.4f}  val CE {vl / nv:.4f}  "
              f"{time.time() - t0:.0f} s{flag}", flush=True)
    train_time = time.time() - t0
    model.load_state_dict(torch.load(out / "best_weights.pt", map_location=device))

    # ---- decision rule and held-out pseudo-wells ----------------------------------------------
    def pseudo(idx, augment):
        g = torch.Generator(device=device).manual_seed(7)
        xs, fs = [], []
        for b in torch.as_tensor(idx, device=device).split(4096):
            x, f = data.batch(b, aug if augment else None, g)
            xs.append(x)
            fs.append(f)
        x, f = torch.cat(xs), torch.cat(fs).cpu().numpy()
        p = predict(model, args.arch, x)
        return p.transpose(1, 0, 2).reshape(K, -1), f.ravel()

    p_va, f_va = pseudo(idx_va, True)
    if args.decision == "macro_f1":
        w, _ = tune_weights(p_va[:, ::7], f_va[::7])          # subsample for speed
    else:
        w = np.ones(K)
    print("decision weights " + " ".join(f"{f} {v:.2f}" for f, v in zip(FACIES, w)))
    metrics = dict(args=vars(args), train_time_s=train_time, n_parameters=n_par,
                   prior=prior.tolist(), decision_weights=w.tolist(), test={})
    for name, augment in (("clean", False), ("augmented", True)):
        p, f = pseudo(idx_te, augment)
        metrics["test"][name] = dict(map=scores(confusion(f, p.argmax(0))),
                                     tuned=scores(confusion(f, (p * w[:, None]).argmax(0))))

    # ---- field ------------------------------------------------------------------------------
    ns_ = SimpleNamespace(data_dir=args.data_dir)
    ck = dict(chis=chis, band=band)
    Xf, comp, _ = stretch_input(ns_, ck, dict(ns=n, twt0=float(meta["twt0"][0])), n)
    xf = torch.as_tensor(Xf[:, :, :n], device=device)
    pf = predict(model, args.arch, xf, mc=args.mc)              # (nx, K, n)
    pf, ns_win = unstretch(pf, comp["i0"], comp["i1"], n)
    trc, _ = read_segy(resolve_path(CFG["horizon_file"], args.data_dir))
    nt, nx = trc.shape
    shift = int(round(CFG["dt"] / CFG["ds"])) - 1
    prob_vol = np.stack([to_grid(pf[:, k], comp["i0"], ns_win, nt, shift) for k in range(K)])
    live = prob_vol.sum(0) > 0
    fac_vol = np.where(live, (prob_vol * w[:, None, None]).argmax(0), 0).astype(np.float32)
    tmpl = resolve_path(CFG["horizon_file"], args.data_dir)
    write_segy(out / f"facies_{tag}.segy", fac_vol, CFG["ds"], tmpl)
    for k, fname in enumerate(FACIES):
        write_segy(out / f"prob_{fname.replace(' ', '_')}_{tag}.segy", prob_vol[k], CFG["ds"], tmpl)
    t1 = (np.arange(prob_vol.shape[1]) + 1) * CFG["ds"]
    x = (np.arange(nx) + 1) * DX

    # baseline: Vshale cut-offs of a regression run (manuscript facies, 3 classes)
    base = None
    vf = sorted(Path(args.baseline_run).glob("vshale_mean_*.segy"))
    if vf:
        vsh = read_segy(vf[0])[0]
        vlive = read_segy(Path(str(vf[0]).replace("vshale_mean", "vp_mean")))[0] != 0
        lab = np.zeros_like(vsh)
        lab[(vsh < 0.72) & (vsh >= 0.5) & vlive] = 1
        lab[(vsh < 0.5) & vlive] = 2
        base = np.stack([(lab == k).astype(float) for k in range(K)])     # one-hot "probabilities"
        print(f"baseline: Vshale cut-offs of {vf[0].name}")

    # wells: 9P along its path (blind), 2X vertical at trace_2x (prior well)
    wells = {}
    if Path(args.well9p_mat).exists():
        m9 = loadmat(args.well9p_mat, squeeze_me=True)
        xs = np.linspace(*WELL9P_X, 800)
        ts = np.linspace(*WELL9P_T, 800)
        sel = (ts >= WIN9P[0]) & (ts <= WIN9P[1])
        wells["9P (blind)"] = dict(xs=xs[sel], ts=ts[sel], t_log=m9["times2"], lab_log=m9["litho2"])
    if Path(args.well2x_mat).exists():
        m2 = loadmat(args.well2x_mat, squeeze_me=True)
        k2 = args.trace_2x
        t2 = t1[live[:, k2]]
        t2 = t2[(t2 >= np.nanmin(m2["times1"])) & (t2 <= np.nanmax(m2["times1"]))]
        wells["2X (prior well)"] = dict(xs=np.full(len(t2), x[k2]), ts=t2, t_log=m2["times1"],
                                        lab_log=m2["litho1"])
    if not wells:
        print("well logs not found: facies scores at the wells skipped")
    metrics["wells"] = {}
    for wn, wd in wells.items():
        ok_log = np.isfinite(wd["t_log"])
        wd["true"] = block_log(wd["t_log"][ok_log], wd["lab_log"][ok_log], wd["ts"])
        P, L = along_path(prob_vol, live, x, t1, wd["xs"], wd["ts"])
        good = L & (wd["true"] >= 0) & np.isfinite(P).all(0)
        wd["prob"], wd["pred"] = P, (P * w[:, None]).argmax(0)
        res = dict(cnn_map=scores(confusion(wd["true"][good], P[:, good].argmax(0))),
                   cnn=scores(confusion(wd["true"][good], wd["pred"][good])))
        if base is not None:
            Bp, _ = along_path(base, live, x, t1, wd["xs"], wd["ts"])
            wd["base"] = Bp.argmax(0)
            res["vshale_cutoff"] = scores(confusion(wd["true"][good], wd["base"][good]))
        res["n_samples"] = int(good.sum())
        res["true_fractions"] = (np.bincount(wd["true"][good], minlength=K) / good.sum()).round(3).tolist()
        metrics["wells"][wn] = res
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    torch.save(dict(state_dict=model.state_dict(), arch=args.arch, n_chi=len(chis), length=n,
                    width=args.width, p_drop=args.dropout, chis=chis, band=band,
                    decision_weights=w, facies=FACIES), out / "model.pt")

    # ---- summary ----------------------------------------------------------------------------
    def row(name, s):
        return (f"| {name} | {s['accuracy']:.3f} | {s['balanced_accuracy']:.3f} | {s['macro_f1']:.3f} | "
                + " | ".join(f"{s['f1'][f]:.2f}" for f in FACIES) + " |")
    lines = [f"# Facies classification: {tag}", "",
             f"arch {args.arch}, {n_par / 1e6:.2f} M parameters, {args.epochs} epochs, "
             f"{train_time:.0f} s; decision {args.decision} "
             f"(weights {', '.join(f'{v:.2f}' for v in w)})", "",
             "| | accuracy | balanced acc. | macro-F1 | F1 " + " | F1 ".join(FACIES) + " |",
             "|---" * (4 + K) + "|"]
    for nm in ("clean", "augmented"):
        lines.append(row(f"test pseudo-wells, {nm}, argmax", metrics["test"][nm]["map"]))
        lines.append(row(f"test pseudo-wells, {nm}, tuned", metrics["test"][nm]["tuned"]))
    for wn, res in metrics["wells"].items():
        lines.append(row(f"{wn}, CNN argmax", res["cnn_map"]))
        lines.append(row(f"{wn}, CNN tuned", res["cnn"]))
        if "vshale_cutoff" in res:
            lines.append(row(f"{wn}, Vshale cut-offs", res["vshale_cutoff"]))
    if metrics["wells"]:
        lines += ["", "Well facies fractions in the scored window (shale/silt/sand/calc.): " +
                  "; ".join(f"{wn} {res['true_fractions']}" for wn, res in metrics["wells"].items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figures ----------------------------------------------------------------------------
    h = np.array(hist)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(h[:, 0], h[:, 1], label="train")
    ax.plot(h[:, 0], h[:, 2], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_curve.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 1 + len(metrics["wells"]), figsize=(5 * (1 + len(metrics["wells"])), 4.5),
                            squeeze=False)
    axs = axs[0]
    items = [("test pseudo-wells", metrics["test"]["augmented"]["tuned"])] + \
            [(wn, r["cnn"]) for wn, r in metrics["wells"].items()]
    for ax, (nm, s) in zip(axs, items):
        cm = np.array(s["confusion"], float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        color="w" if cmn[i, j] > 0.5 else "k", fontsize=9)
        ax.set_xticks(range(K))
        ax.set_xticklabels(FACIES, rotation=30, fontsize=8)
        ax.set_yticks(range(K))
        ax.set_yticklabels(FACIES, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{nm}\nmacro-F1 {s['macro_f1']:.2f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=150)
    plt.close(fig)

    rows = np.where(live.any(1))[0]
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    panels = [(fac_vol, "Most probable facies (tuned)", FACIES_CMAP, (0, 3))] + \
             [(prob_vol[k], f"P({f})", "viridis", (0, 1)) for k, f in enumerate(FACIES)]
    if base is not None:
        panels.append((base.argmax(0).astype(float), "Vshale cut-offs (manuscript)", FACIES_CMAP, (0, 3)))
    for ax, (v, title, cmap, clim) in zip(axs.ravel(), panels):
        im = ax.imshow(np.where(live, v, np.nan), aspect="auto", cmap=cmap, vmin=clim[0], vmax=clim[1],
                       extent=[x[0], x[-1], t1[-1], t1[0]], interpolation="nearest")
        ax.set_ylim(t1[rows[-1]], t1[rows[0]])
        ax.plot(WELL9P_X, WELL9P_T, "w--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("TWT (s)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "facies_sections.png", dpi=150)
    plt.close(fig)

    if not wells:
        print(f"wrote {out}/")
        return
    fig, axs = plt.subplots(1, 2 * len(wells), figsize=(4.2 * len(wells) * 2, 7))
    for j, (wn, wd) in enumerate(wells.items()):
        t = wd["ts"]
        cols = [("well", wd["true"]), ("CNN", wd["pred"])] + \
               ([("Vsh cut-off", wd["base"])] if "base" in wd else [])
        a = axs[2 * j]
        a.imshow(np.stack([c[1] for c in cols], 1).astype(float), aspect="auto", cmap=FACIES_CMAP,
                 vmin=0, vmax=3, extent=[0, len(cols), t[-1], t[0]], interpolation="nearest")
        a.set_xticks(np.arange(len(cols)) + 0.5)
        a.set_xticklabels([c[0] for c in cols])
        a.set_title(wn)
        a.set_ylabel("TWT (s)")
        a = axs[2 * j + 1]
        cum = np.cumsum(np.nan_to_num(wd["prob"]), 0)
        for k in range(K):
            lo = cum[k - 1] if k else np.zeros_like(cum[0])
            a.fill_betweenx(t, lo, cum[k], color=FACIES_CMAP.colors[k], label=FACIES[k])
        a.set_ylim(t[-1], t[0])
        a.set_xlim(0, 1)
        a.set_xlabel("probability (stacked)")
        a.set_title(f"{wn}: CNN probabilities")
        if j == 0:
            a.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "wells.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
