"""Well 2X in two-way time: manuscript Figures 2 and 3.

Figure 2  Well_2x_Macro_layer_time         Vshale, porosity, macrolayers ML1-ML3 and facies
Figure 3  Well_2x_elasto_petrophysics_time  logs at full resolution (grey) and averaged on the 1 ms
          grid of the pseudo-wells (black), and one realisation (blue) of the pseudo-well rock physics
          (pseudowell/rock_physics.py, Table 2) computed with the facies interpreted at the well, on
          the same 1 ms grid; Sw (full resolution) after Vshale
Figure 4  Well_2x_EEI_Rsquare               correlation coefficient between the normalised EEI of the
          logs (full resolution, ML1-ML3) and Vp, Vs, rho, Vshale and porosity, chi from -90 to 90 deg
well2x_fit.txt: correlation and rms difference between the logs (1 ms averages) and the realisation.

Usage:
  python make_well2x_figures.py --outdir ../Notebooks/outputs/Well2X --formats png pdf
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

from calibrate_well2x import well_2x
from paths import DATA_DIR
from odisi_inversion import CFG
from pseudowell.rock_physics import pseudo_well_properties
from pseudowell.seismic import eeimp2

FACIES_CMAP = ListedColormap([[0.5, 0.5, 0], [0.5, 0.5, 0.5], [1, 1, 0], [1, 0, 0]])
NSIM = 1000          # realisations drawn; Figure 3 shows the first one


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--well-mat", default=str(DATA_DIR / "well2x_data.mat"))
    ap.add_argument("--outdir", default="figures_well2x")
    ap.add_argument("--formats", nargs="+", default=["png"])
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    w, ml_t, t, fac, logs = well_2x(args.well_mat)
    sims = pseudo_well_properties(t, np.repeat(fac[:, None], NSIM, 1), CFG["rock"],
                                  np.random.default_rng(args.seed))
    real = {k: v[:, 0] for k, v in zip(("vp", "vs", "rho", "por", "vsh"), sims)}

    lines = [f"Well 2X, {t[0]:.3f}-{t[-1]:.3f} s ({len(t)} samples of 1 ms), macrolayer boundaries "
             f"at {', '.join(f'{x:.4f}' for x in ml_t)} s",
             "one pseudo-well realisation on the facies of the well: r, rms difference"]
    for k in ("por", "vp", "vs", "rho", "vsh"):
        x = logs[k]
        ok = np.isfinite(x)
        lines.append(f"{k:8s}  {np.corrcoef(x[ok], real[k][ok])[0, 1]:6.3f}  "
                     f"{np.sqrt(np.mean((x[ok] - real[k][ok]) ** 2)):10.4g}")
    (out / "well2x_fit.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    g_fmt = FuncFormatter(lambda v, _: f"{v:g}")
    rc = {"font.family": "sans-serif", "xtick.direction": "in", "ytick.direction": "in",
          "xtick.top": True, "ytick.right": True, "axes.linewidth": 0.8, "grid.color": "0.85",
          "grid.linewidth": 0.6}
    ylim = (ml_t[-1], ml_t[0])
    yt = np.arange(np.ceil(ml_t[0] * 100) / 100, ml_t[-1] + 1e-9, 0.01)

    def facies_track(ax):
        ax.imshow(w["litho1"][:, None], aspect="auto", cmap=FACIES_CMAP, vmin=-0.5, vmax=3.5,
                  extent=[0, 1, w["times1"][-1], w["times1"][0]], interpolation="nearest")
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_xlabel("Facies", fontsize=16, fontweight="bold")
        cax = ax.inset_axes([1.06, 0.03, 0.12, 0.3])     # just right of the facies column
        cb = plt.colorbar(plt.cm.ScalarMappable(cmap=FACIES_CMAP, norm=plt.Normalize(-0.5, 3.5)),
                          cax=cax, ticks=[0, 1, 2, 3])
        cb.set_ticklabels(["Shale", "Silt", "Sand", "Cal. Sand"])
        cb.ax.tick_params(labelsize=11, length=0)

    def save(fig, name):
        fig.tight_layout()
        for fmt in args.formats:
            fig.savefig(out / f"{name}.{fmt}", dpi=args.dpi)
        plt.close(fig)

    # ---- Figure 2: macrolayers in time --------------------------------------------------------
    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, 4, figsize=(8, 6), sharey=True)
        for ax, k, lab, xl in ((axs[0], "vsh1", "Vshale (v/v)", (0, 1)),
                               (axs[1], "por1", r"$\phi$ (v/v)", (0, 0.25))):
            ax.plot(w[k], w["times1"], "k", lw=1.2)
            ax.set_xlim(xl)
            ax.set_xlabel(lab, fontsize=16, fontweight="bold")
            ax.grid(True)
            ax.xaxis.set_major_formatter(g_fmt)
        a = axs[2]
        for i, name in enumerate(("ML1", "ML2", "ML3")):
            a.add_patch(plt.Rectangle((0, ml_t[i]), 1, ml_t[i + 1] - ml_t[i], fill=False, lw=1.5))
            a.text(0.5, 0.5 * (ml_t[i] + ml_t[i + 1]), name, ha="center", va="center", fontsize=16,
                   fontweight="bold")
        a.set_xlim(0, 1)
        a.set_xticks([])
        a.set_xlabel("ML", fontsize=16, fontweight="bold")
        facies_track(axs[3])
        axs[0].set_ylim(*ylim)
        axs[0].set_yticks(yt)
        axs[0].set_ylabel("Time (sec)", fontsize=16, fontweight="bold")
        for ax in axs:
            ax.tick_params(labelsize=12)
        save(fig, "Well_2x_Macro_layer_time")

    # ---- Figure 3: logs and one pseudo-well realisation on the facies of the well -------------
    tracks = [("por", "por1", r"$\phi$ (v/v)", (0, 0.3), 1),
              ("vp", "vp1", "Vp (km/s)", (2.5, 5), 1e-3),
              ("vs", "vs1", "Vs (km/s)", (1, 3), 1e-3),
              ("rho", "rho1", r"$\rho$ (g/cm$^3$)", (2.2, 2.8), 1),
              ("vsh", "vsh1", "Vshale (v/v)", (0, 1), 1),
              ("sw", "sw1", "Sw", (0.5, 1.02), 1)]
    with plt.rc_context(rc):
        fig, axs = plt.subplots(1, 7, figsize=(14, 7.4), sharey=True)
        for ax, (k, wk, lab, xl, s) in zip(axs, tracks):
            if k in real:
                ax.plot(w[wk] * s, w["times1"], color="0.65", lw=0.8, label="Log, full resolution")
                ax.plot(logs[k] * s, t, "k", lw=1.2, label="Log, 1 ms average")
                ax.plot(real[k] * s, t, "b", lw=1.4, label="Pseudo-well, 1 ms")
            else:
                ax.plot(w[wk] * s, w["times1"], "k", lw=1.0)
            ax.set_xlim(xl)
            ax.set_xlabel(lab, fontsize=15, fontweight="bold")
            ax.grid(True)
            ax.xaxis.set_major_formatter(g_fmt)
            ax.tick_params(labelsize=12)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=3, fontsize=13, frameon=False)
        facies_track(axs[6])
        axs[0].set_ylim(*ylim)
        axs[0].set_yticks(yt)
        axs[0].set_ylabel("Time (sec)", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        for fmt in args.formats:
            fig.savefig(out / f"Well_2x_elasto_petrophysics_time.{fmt}", dpi=args.dpi)
        plt.close(fig)
    # ---- Figure 4: correlation between EEI and the logs as a function of chi ----------------
    chis = np.arange(-90, 90.01, 0.5)
    ok = np.all([np.isfinite(w[k]) for k in ("vp1", "vs1", "rho1", "vsh1", "por1")], axis=0)
    eei = [eeimp2(w["vp1"][ok], w["vs1"][ok], w["rho1"][ok], c) for c in chis]
    curves = (("vp1", "Vp", "k"), ("vs1", "Vs", "r"), ("rho1", r"$\rho$", (0, 0.8, 0)),
              ("vsh1", "Vshale", "b"), ("por1", r"$\phi$", (0.9, 0.7, 0)))
    lines = ["chi of the largest |r| and r at the angles of the inversion (-60, -7.5, 25, 45 deg)"]
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(11, 6))
        for k, lab, c in curves:
            r = np.array([np.corrcoef(e, w[k][ok])[0, 1] for e in eei])
            ax.plot(chis, r, color=c, lw=2.2, label=lab)
            i = np.abs(r).argmax()
            lines.append(f"{k[:-1]:4s} chi {chis[i]:6.1f}  r {r[i]:6.3f}   " + "  ".join(
                f"{a:g}: {r[np.argmin(np.abs(chis - a))]:6.3f}" for a in (-60, -7.5, 25, 45)))
        for a in (-60, -7.5, 25, 45):
            ax.axvline(a, color="0.5", ls=":", lw=1.2)
        ax.set_xlim(-90, 90)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-90, 91, 30))
        ax.set_yticks(np.arange(-1, 1.01, 0.5))
        ax.yaxis.set_major_formatter(g_fmt)
        ax.xaxis.set_major_formatter(g_fmt)
        ax.tick_params(axis="x", pad=8)
        ax.set_xlabel(r"$\chi$ (degree)", fontsize=20)
        ax.set_ylabel("Correlation coefficient", fontsize=20)
        ax.tick_params(labelsize=17)
        ax.grid(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=17, framealpha=1,
                  edgecolor="k")
        save(fig, "Well_2x_EEI_Rsquare")
    (out / "well2x_eei_sensitivity.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"figures in {out}/")


if __name__ == "__main__":
    main()
