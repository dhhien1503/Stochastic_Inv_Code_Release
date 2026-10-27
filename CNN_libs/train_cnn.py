"""Train the EEI -> elasto-petrophysics CNN on the pseudo-well training set.

Regression of vp, vs, rho, porosity and vshale (Gaussian negative log-likelihood,
so the network also predicts its aleatoric uncertainty; --reg-loss mse gives a
plain least-squares fit), and classification of the four facies (weighted
cross-entropy), on every 1 ms sample of the window.

Field amplitudes: by default (--calibration none) the field EEI is used at its
raw amplitude (0.65 * convz(A cos + B sin, wavelet), as in ODiSI), which ties
the well 9P synthetic EEI to within ~10-20% at chi <= 0. --calibration noise
instead fits separate intercept/gradient gains and an additive noise level so
that the field EEI power matches the prior pseudo-wells at all 25 angles
(cnn_common.calibrate), and adds that noise (nA cos(chi) + nB sin(chi), band-
limited) to the training EEI. With the current prior this is harmful: the prior
under-predicts EEI at chi > 10 deg (too little density/Vs contrast in sands),
so the fit inflates the field gains and treats real signal as noise. A random
+-10% gain is always applied to the training EEI.

Outputs in --outdir: model.pt (weights + every normalisation and calibration
constant needed by infer_cnn.py), history.csv, metrics.json and QC figures.

Usage:
  python train_cnn.py                                   # training_data/ -> results/eei25
  python train_cnn.py --calibration noise --outdir results/eei25_noise
  python train_cnn.py --epochs 5 --max-train 5000 --outdir results/test
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from cnn_common import (FACIES, PROPS, aux_channels, band_response, calibrate,
                        field_components)
from paths import DATA_DIR
from cnn_model import build_model, enable_mc_dropout

UNITS = dict(vp="m/s", vs="m/s", rho="g/cc", poro="v/v", vshale="v/v")


class Data:
    """Whole training set on the GPU, with on-the-fly augmentation."""

    def __init__(self, X, Y, Fac, aux, s_chi, y_mean, y_std, chis, cal, H2, device):
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=device)
        self.X, self.aux = t(X), t(aux)
        self.Y = (t(Y) - t(y_mean)[None, :, None]) / t(y_std)[None, :, None]
        self.F = t(Fac, torch.int64)
        self.mask = self.F >= 0
        self.Y = self.Y * self.mask[:, None]
        self.s_chi = t(s_chi)[None, :, None]
        c = np.deg2rad(chis)
        # noise of the intercept and gradient traces, projected on each angle,
        # in units of the normalised input
        self.na = t(cal["noise_a"] * np.cos(c))[None, :, None] / self.s_chi
        self.nb = t(cal["noise_b"] * np.sin(c))[None, :, None] / self.s_chi
        self.H2 = t(H2)
        self.device = device

    def batch(self, idx, noise=(0.0, 0.0), gain_jitter=0.0, generator=None):
        x = self.X[idx] / self.s_chi
        m = self.mask[idx]
        B, _, L = x.shape
        if noise[1] > 0:
            w = torch.randn(B, 2, L, device=self.device, generator=generator)
            w = torch.fft.irfft(torch.fft.rfft(w) * self.H2, n=L)
            w = w / w.pow(2).mean(-1, keepdim=True).sqrt()
            lvl = noise[0] + (noise[1] - noise[0]) * torch.rand(
                B, 1, 1, device=self.device, generator=generator)
            x = x + lvl * (self.na * w[:, :1] + self.nb * w[:, 1:])
        if gain_jitter > 0:
            x = x * (1 + gain_jitter * (2 * torch.rand(B, 1, 1, device=self.device,
                                                       generator=generator) - 1))
        x = x * m[:, None]
        return torch.cat([x, self.aux[idx]], 1), self.Y[idx], self.F[idx], m


def loss_fn(mu, logvar, logits, y, f, m, args, cw):
    mf = m[:, None].float()
    if args.reg_loss == "nll":
        var = logvar.exp()
        reg = 0.5 * (logvar + (y - mu) ** 2 / var)
        if args.beta > 0:                     # beta-NLL (Seitzer et al. 2022)
            reg = reg * var.detach() ** args.beta
    else:
        reg = (y - mu) ** 2
    reg = (reg * mf).sum() / (mf.sum() * mu.shape[1])
    cls = F.cross_entropy(logits, f, weight=cw, ignore_index=-1)
    return reg + args.cls_weight * cls, reg, cls


@torch.no_grad()
def predict(model, data, idx, noise, mc=0, bs=2048, seed=0):
    """Mean, total std (standardised units) and facies probabilities."""
    g = torch.Generator(device=data.device).manual_seed(seed)
    if mc:
        enable_mc_dropout(model)
    else:
        model.eval()
    mus, vars_, probs = [], [], []
    for s in range(0, len(idx), bs):
        x, *_ = data.batch(idx[s:s + bs], noise=noise, generator=g)
        acc = []
        for _ in range(max(mc, 1)):
            mu, lv, lg = model(x)
            acc.append((mu.float(), lv.float().exp(), lg.float().softmax(1)))
        mu_t = torch.stack([a[0] for a in acc])
        mus.append(mu_t.mean(0))
        vars_.append(torch.stack([a[1] for a in acc]).mean(0) + mu_t.var(0, unbiased=False))
        probs.append(torch.stack([a[2] for a in acc]).mean(0))
    model.train()
    return torch.cat(mus), torch.cat(vars_).sqrt(), torch.cat(probs)


def evaluate(model, data, idx, y_mean, y_std, noise, mc=0):
    mu, sd, pr = predict(model, data, idx, noise, mc)
    m = data.mask[idx].cpu().numpy()
    y = data.Y[idx].cpu().numpy()
    mu, sd, pr = mu.cpu().numpy(), sd.cpu().numpy(), pr.cpu().numpy()
    out = {}
    corr = {}
    for i, p in enumerate(PROPS):
        r = []
        for j in range(len(idx)):
            a, b = y[j, i, m[j]], mu[j, i, m[j]]
            r.append(np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else np.nan)
        corr[p] = np.array(r)
        err = (mu[:, i] - y[:, i])[m]
        z = np.abs(err) / sd[:, i][m]
        out[p] = dict(corr_median=float(np.nanmedian(r)), corr_mean=float(np.nanmean(r)),
                      rmse=float(np.sqrt((err ** 2).mean()) * y_std[i]),
                      coverage50=float((z <= 0.674).mean()), coverage90=float((z <= 1.645).mean()))
    f_true = data.F[idx].cpu().numpy()[m]
    f_pred = pr.argmax(1)[m]
    K = len(FACIES)
    cm = np.zeros((K, K), int)
    np.add.at(cm, (f_true, f_pred), 1)
    prec = np.diag(cm) / np.maximum(cm.sum(0), 1)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    out["facies"] = dict(accuracy=float(np.trace(cm) / cm.sum()), f1=dict(zip(FACIES, f1.tolist())),
                         macro_f1=float(f1.mean()), confusion=cm.tolist())
    return out, corr, (mu, sd, pr)


def plots(out_dir, hist, corr_tr, corr_te, metrics, data, idx_ex, pred_ex, y_mean, y_std, ts):
    h = np.array(hist)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for j, (lbl, a, b) in enumerate((("total", 1, 4), ("regression", 2, 5), ("facies CE", 3, 6))):
        ax[j].plot(h[:, 0], h[:, a], label="train")
        ax[j].plot(h[:, 0], h[:, b], label="validation")
        ax[j].set_title(lbl)
        ax[j].set_xlabel("epoch")
        ax[j].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    pos = np.arange(len(PROPS))
    for off, c, cd, lbl in ((-0.18, "b", corr_tr, "Training"), (0.18, "r", corr_te, "Test")):
        bp = ax.boxplot([cd[p][np.isfinite(cd[p])] for p in PROPS], positions=pos + off,
                        widths=0.3, showfliers=False)
        for k in bp:
            plt.setp(bp[k], color=c)
        ax.plot([], [], c, label=lbl)
    ax.set_xticks(pos)
    ax.set_xticklabels(["Vp", "Vs", r"$\rho$", r"$\phi$", "Vshale"])
    ax.set_ylabel("Correlation coefficient per pseudo-well")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_elasto_petrophysics.png", dpi=150)
    plt.close(fig)

    cm = np.array(metrics["test"]["facies"]["confusion"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(FACIES)):
        for j in range(len(FACIES)):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="w" if cmn[i, j] > 0.5 else "k")
    ax.set_xticks(range(len(FACIES)))
    ax.set_xticklabels(FACIES, rotation=30)
    ax.set_yticks(range(len(FACIES)))
    ax.set_yticklabels(FACIES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "facies_confusion.png", dpi=150)
    plt.close(fig)

    mu, sd, pr = pred_ex
    fig, axs = plt.subplots(len(idx_ex), 6, figsize=(15, 4.2 * len(idx_ex)), sharey="row")
    for r, j in enumerate(idx_ex):
        m = data.mask[j].cpu().numpy()
        t = ts[j][m]
        y = data.Y[j].cpu().numpy()
        for i, p in enumerate(PROPS):
            ax = axs[r, i]
            tr = y[i, m] * y_std[i] + y_mean[i]
            pm = mu[r, i, m] * y_std[i] + y_mean[i]
            ps = sd[r, i, m] * y_std[i]
            ax.plot(tr, t, "k", lw=1, label="pseudo-well")
            ax.plot(pm, t, "b", lw=1.2, label="CNN mean")
            ax.fill_betweenx(t, pm - 1.645 * ps, pm + 1.645 * ps, color="b", alpha=0.2,
                             label="90% interval")
            ax.set_xlabel(f"{p} ({UNITS[p]})")
        f_true = data.F[j].cpu().numpy()[m]
        f_pred = pr[r].argmax(0)[m]
        axs[r, 5].imshow(np.stack([f_true, f_pred], 1), aspect="auto", cmap="tab10",
                         vmin=0, vmax=9, extent=[0, 2, t[-1], t[0]], interpolation="nearest")
        axs[r, 5].set_xticks([0.5, 1.5])
        axs[r, 5].set_xticklabels(["true", "CNN"])
        axs[r, 0].set_ylabel("TWT (s)")
        axs[r, 0].set_ylim(t[-1], t[0])
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "test_examples.png", dpi=120)
    plt.close(fig)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train-dir", default="training_data")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="field SEG-Y folder")
    ap.add_argument("--outdir", default="results/regression_s0")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--arch", choices=("unet", "keras"), default="unet",
                    help="unet: EEINet (default); keras: port of the manuscript's Keras "
                         "PetroNet (regression only, use with --reg-loss mse --cls-weight 0)")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--reg-loss", choices=("nll", "mse"), default="nll")
    ap.add_argument("--beta", type=float, default=0.5, help="beta-NLL exponent (0 = plain NLL)")
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--calibration", choices=("noise", "gains", "none"), default="none",
                    help="none: raw field amplitudes, no added noise (default); "
                         "noise: fitted intercept/gradient gains + fitted noise; "
                         "gains: fitted gains only")
    ap.add_argument("--noise-range", type=float, nargs=2, default=(0.25, 1.5),
                    help="training noise level, as a fraction of the fitted noise")
    ap.add_argument("--gain-jitter", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--max-train", type=int, default=None, help="debug: cap the training set")
    ap.add_argument("--mc-test", type=int, default=20, help="MC-dropout passes for test coverage")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    td = Path(args.train_dir)

    # ---- data ---------------------------------------------------------------------
    t0 = time.time()
    X = np.load(td / "eei.npy")
    Y = np.load(td / "props.npy")
    Fac = np.load(td / "facies.npy")
    meta = np.load(td / "meta.npz")
    chis, band = meta["chis"], tuple(meta["band"])
    N, n_chi, L = X.shape
    perm = rng.permutation(N)
    n_te, n_va = int(args.test_frac * N), int(args.val_frac * N)
    idx_te, idx_va, idx_tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    if args.max_train:
        idx_tr = idx_tr[:args.max_train]
    mask = Fac >= 0
    mtr = mask[idx_tr]
    syn_power = (X[idx_tr] ** 2).sum((0, 2)) / mtr.sum()
    s_chi = np.sqrt(syn_power)
    y_mean = np.array([Y[idx_tr, i][mtr].mean() for i in range(len(PROPS))])
    y_std = np.array([Y[idx_tr, i][mtr].std() for i in range(len(PROPS))])
    freq = np.bincount(Fac[idx_tr][mtr], minlength=len(FACIES)) / mtr.sum()
    cw = 1 / np.sqrt(freq)
    cw = cw / (cw * freq).sum()
    print(f"loaded {N} pseudo-wells x {n_chi} angles x {L} ms in {time.time() - t0:.0f} s; "
          f"train/val/test {len(idx_tr)}/{len(idx_va)}/{len(idx_te)}")

    # ---- field calibration ----------------------------------------------------------------
    comp = field_components(args.data_dir, band)
    cal = calibrate(comp, syn_power, chis)
    if args.calibration == "gains":
        cal.update(gain_a=cal["gain_a_nonoise"], gain_b=cal["gain_b_nonoise"], noise_a=0.0, noise_b=0.0)
    elif args.calibration == "none":
        cal.update(gain_a=1.0, gain_b=1.0, noise_a=0.0, noise_b=0.0)
    cal["mode"] = args.calibration
    print(f"field calibration ({args.calibration}): intercept gain {cal['gain_a']:.3f}, gradient "
          f"gain {cal['gain_b']:.3f}, noise (intercept, gradient) = {cal['noise_a']:.1f}, "
          f"{cal['noise_b']:.1f}")
    print("  fitted/field EEI rms ratio per angle: " +
          " ".join(f"{r:.2f}" for r in cal["fit_ratio"]))
    noise = tuple(args.noise_range) if cal["noise_a"] + cal["noise_b"] > 0 else (0.0, 0.0)

    aux = aux_channels(meta["ns"], meta["twt0"], L)
    # one window for every pseudo-well (make_datamat.py): the field must be stretched onto it
    fixed_window = (dict(ns=int(meta["ns"][0]), twt0=float(meta["twt0"][0]))
                    if np.ptp(meta["ns"]) == 0 and np.ptp(meta["twt0"]) == 0 else None)
    data = Data(X, Y, Fac, aux, s_chi, y_mean, y_std, chis, cal, band_response(L, band), device)
    del X, Y
    cw_t = torch.as_tensor(cw, dtype=torch.float32, device=device)

    # ---- model ------------------------------------------------------------------------
    model_args = dict(arch=args.arch, n_in=n_chi + 2, n_reg=len(PROPS), n_cls=len(FACIES),
                      width=args.width, p_drop=args.dropout, length=L)
    model = build_model(model_args).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"{type(model).__name__}: {n_par / 1e6:.2f} M parameters, device {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * int(np.ceil(len(idx_tr) / args.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    idx_tr_t = torch.as_tensor(idx_tr, device=device)
    idx_va_t = torch.as_tensor(idx_va, device=device)
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    hist, best = [], np.inf
    t_train = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = np.zeros(3)
        nb = 0
        for b in torch.randperm(len(idx_tr_t), device=device).split(args.batch):
            x, y, f, m = data.batch(idx_tr_t[b], noise, args.gain_jitter)
            with torch.autocast(**amp):
                mu, lv, lg = model(x)
            loss, reg, cls = loss_fn(mu.float(), lv.float(), lg.float(), y, f, m, args, cw_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += [loss.item(), reg.item(), cls.item()]
            nb += 1
        # validation at the nominal (fitted) noise level, fixed noise draw
        model.eval()
        g = torch.Generator(device=device).manual_seed(123)
        vt = np.zeros(3)
        nv = 0
        with torch.no_grad():
            for b in idx_va_t.split(2048):
                lvl = (1.0, 1.0) if noise[1] > 0 else (0.0, 0.0)
                x, y, f, m = data.batch(b, lvl, generator=g)
                with torch.autocast(**amp):
                    mu, lv, lg = model(x)
                vt += [v.item() for v in loss_fn(mu.float(), lv.float(), lg.float(), y, f, m,
                                                 args, cw_t)]
                nv += 1
        tr, va = tot / nb, vt / nv
        hist.append([ep, *tr, *va])
        flag = ""
        if va[0] < best:
            best = va[0]
            flag = " *"
            torch.save(model.state_dict(), out_dir / "best_weights.pt")
        print(f"epoch {ep:3d}  train {tr[0]:.4f} (reg {tr[1]:.4f}, CE {tr[2]:.4f})  "
              f"val {va[0]:.4f} (reg {va[1]:.4f}, CE {va[2]:.4f})  "
              f"{time.time() - t_train:.0f} s{flag}", flush=True)
    train_time = time.time() - t_train
    np.savetxt(out_dir / "history.csv", np.array(hist), delimiter=",", fmt="%.6f",
               header="epoch,train_loss,train_reg,train_ce,val_loss,val_reg,val_ce", comments="")

    # ---- evaluation on the held-out pseudo-wells ------------------------------------------
    model.load_state_dict(torch.load(out_dir / "best_weights.pt", map_location=device))
    nominal = (1.0, 1.0) if noise[1] > 0 else (0.0, 0.0)
    metrics = dict(train_time_s=train_time, n_parameters=n_par, calibration=cal,
                   args=vars(args))
    sub_tr = idx_tr[:len(idx_te)]
    metrics["train"], corr_tr, _ = evaluate(model, data, sub_tr, y_mean, y_std, nominal)
    metrics["test"], corr_te, _ = evaluate(model, data, idx_te, y_mean, y_std, nominal)
    metrics["test_noise_free"], _, _ = evaluate(model, data, idx_te, y_mean, y_std, (0.0, 0.0))
    if args.mc_test:
        sub = idx_te[:2000]
        metrics[f"test_mc{args.mc_test}"], _, _ = evaluate(model, data, sub, y_mean, y_std,
                                                           nominal, mc=args.mc_test)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))

    def row(name, d):
        return (f"{name:18s}" + "".join(f"{d[p]['corr_median']:8.3f}" for p in PROPS) +
                f"   facies acc {d['facies']['accuracy']:.3f}, macro-F1 {d['facies']['macro_f1']:.3f}")
    print("\nmedian correlation per pseudo-well   " + "".join(f"{p:>8s}" for p in PROPS))
    for k in [k for k in metrics if k.startswith(("train", "test")) and isinstance(metrics[k], dict)
              and "facies" in metrics[k]]:
        print(row(k, metrics[k]))
    for k in [k for k in metrics if k.startswith("test")]:
        print(f"coverage of 50%/90% intervals, {k}: " + ", ".join(
            f"{p} {metrics[k][p]['coverage50']:.2f}/{metrics[k][p]['coverage90']:.2f}" for p in PROPS))

    # ---- checkpoint for inference ---------------------------------------------------------
    torch.save(dict(state_dict=model.state_dict(),
                    model_args=model_args, fixed_window=fixed_window,
                    chis=chis, band=band, win_len=L, s_chi=s_chi, y_mean=y_mean, y_std=y_std,
                    props=PROPS, facies=FACIES, calibration=cal, reg_loss=args.reg_loss),
               out_dir / "model.pt")

    ex = idx_te[:3]
    ts = meta["twt0"][:, None] + np.arange(L)[None] * 0.001
    mu, sd, pr = predict(model, data, ex, nominal)
    plots(out_dir, hist, corr_tr, corr_te, metrics, data, ex,
          (mu.cpu().numpy(), sd.cpu().numpy(), pr.cpu().numpy()), y_mean, y_std, ts)
    print(f"\nsaved model and QC figures to {out_dir}/ (training {train_time:.0f} s)")


if __name__ == "__main__":
    main()
