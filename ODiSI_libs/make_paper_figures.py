"""Publication figures of the manuscript.

Figure 1  Stochastic_inversion_<run>.png      mean Vp, Vs, rho, porosity, Vshale + facies
                                              (discrete facies colours, as in the well panels)
Figure 2  Cal_seismic_data_<run>.png          EEI(-60, -7.5, 25, 45) computed from the mean
                                              Vp/Vs/rho, band-passed 3-50 Hz
Figure 3  Well_9p_elasto_petrophysics_<run>.png  well 9P logs vs inversion (mean, q25, q75)
                                              along the well path + correlation coefficients

Inputs are the inversion SEG-Ys (<prop>_<stat>_<tag>.segy, on the 1 ms grid) of
odisi_inversion.py or infer_cnn.py, plus well9p_data.mat (times2, vp2, vs2, rho2, por2,
vsh2, litho2) for the logs and correlations of Figure 3. The well logs are confidential and
not distributed: without them Figure 3 shows the inversion along the well path only.

Usage:
  python make_paper_figures.py --input-dir ../Notebooks/outputs/ODiSI/chi_45 --formats png pdf
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import RegularGridInterpolator

from paths import DATA_DIR
from pseudowell.plotting import NEW_CMAP
from pseudowell.segy_io import read_segy, resolve_path
from pseudowell.seismic import butter_bandpass, eeimp2, filtfilt_odd
from odisi_inversion import CFG, layer_bounds

DT, DT1, DX = 0.004, 0.001, 12.5
WELL9P_X = np.array([234, 255]) * DX        # well 9P path: distance and TWT of its two ends
WELL9P_T = np.array([2.8, 3.08])
# well 9P logs (times2) are on the two-way time of the seismic line; no time shift is applied
YLIM_WELL = (2.886, 3.016)                  # Well 9P window: plotted and scored (s)
FACIES_CMAP = ListedColormap([[0.5, 0.5, 0], [0.5, 0.5, 0.5], [1, 1, 0], [1, 0, 0]])
Q75_COLOR = np.array([255, 165, 0]) / 255

plt.rcParams.update({"font.size": 14, "axes.titlesize": 16, "savefig.facecolor": "white"})


def load_volumes(input_dir, tag):
    vol = {}
    for p in ("vp", "vs", "rho", "poro", "vshale"):
        for st in ("mean", "q25", "q75"):
            f = Path(input_dir) / f"{p}_{st}_{tag}.segy"
            if f.exists():
                vol[f"{p}_{st}"] = read_segy(f)[0]
            else:
                print(f"  missing {f.name} — panels/curves using it are left out")
    return vol


def facies_from_vshale(vsh, live=None):
    """Facies from Vshale: 1 where 0.5 <= Vsh < 0.72, 2 where Vsh <= 0.5, else 0,
    inside the inversion window.

    A test ``Vsh > 0`` would skip the zero padding outside the window, but it also
    turns clean sand with Vsh = 0 into shale (the CNN clips
    Vshale at 0). The window is therefore taken from ``live`` (e.g. Vp != 0)
    when given."""
    if live is None:
        live = vsh > 0
    litho = np.zeros_like(vsh)
    litho[(vsh < 0.72) & (vsh >= 0.5) & live] = 1
    litho[(vsh < 0.5) & live] = 2
    return litho


def image(ax, data, x, t, cmap, clim=None):
    """Image with pixel centres at x, t."""
    hx, ht = (x[1] - x[0]) / 2, (t[1] - t[0]) / 2
    return ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest",
                     vmin=None if clim is None else clim[0],
                     vmax=None if clim is None else clim[1],
                     extent=[x[0] - hx, x[-1] + hx, t[-1] + ht, t[0] - ht])


def section_panel(fig, ax, data, x, t1, title, clim, horizons, ylim):
    im = image(ax, data, x, t1, NEW_CMAP, clim)
    cb = fig.colorbar(im, ax=ax)
    for h in horizons:
        ax.plot(x, h, "--k", lw=1)
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("TWT (sec)")
    ax.set_title(title, fontsize=16)
    return cb


def save(fig, out_dir, name, formats, dpi=500):
    for fmt in formats:
        fig.savefig(out_dir / f"{name}.{fmt}", dpi=dpi)
    plt.close(fig)
    print(f"  wrote {name}.{'/'.join(formats)}")


def ssim_global(x, y):
    """Structural similarity of two curves over the whole interval (one window): luminance
    (means), contrast (standard deviations) and structure (covariance) terms, with the
    constants C1 = (0.01 L)^2, C2 = (0.03 L)^2 of Wang et al. (2004), L = range of the log x."""
    L = x.max() - x.min()
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mx, my = x.mean(), y.mean()
    sxy = ((x - mx) * (y - my)).mean()
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (x.var() + y.var() + c2))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-dir", default="results")
    ap.add_argument("--tag", default="2x_9P_line_EEI_run_10_Nov_25")
    ap.add_argument("--run-name", default="10_Nov", help="suffix of the figure names")
    ap.add_argument("--data-dir", default=str(DATA_DIR), help="folder with live_window.sgy, avo_intercept_A.sgy, avo_gradient_B.sgy")
    ap.add_argument("--well-mat", default=str(DATA_DIR / "well9p_data.mat"))
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--formats", nargs="+", default=["png"], help="e.g. png pdf svg")
    ap.add_argument("--dpi", type=int, default=500)
    ap.add_argument("--facies-classified", "--facies-from-cnn", dest="facies_classified",
                    action="store_true",
                    help="plot facies_<tag>.segy of --input-dir (facies classification, 4 classes: "
                         "ODiSI posterior or CNN classifier) instead of the Vshale cut-offs")
    ap.add_argument("--facies-segy", default=None,
                    help="facies volume to plot instead of the Vshale cut-offs (e.g. the "
                         "facies_<tag>.segy of classify_facies.py); implies --facies-classified")
    args = ap.parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- geometry: live window and macro-layer horizons -------------------------
    trcdat, _ = read_segy(resolve_path(CFG["horizon_file"], args.data_dir))
    nt, nx = trcdat.shape
    live = trcdat != 0
    i_first = live.argmax(axis=0)
    i_last = nt - 1 - live[::-1].argmax(axis=0)
    t = (np.arange(nt) + 1) * DT                    # sample i at (i+1)*dt
    x = (np.arange(nx) + 1) * DX
    ylim = (t[i_first.min()], t[i_last.max()])
    bounds = layer_bounds(CFG["layers"], i_first, i_last)
    top, base = bounds[0][0], bounds[-1][1]
    horizons = [t[b[0]] for b in bounds] + [t[base]]

    print(f"reading {args.input_dir}/*_{args.tag}.segy")
    vol = load_volumes(args.input_dir, args.tag)
    nt1 = vol["vshale_mean"].shape[0]
    t1 = (np.arange(nt1) + 1) * DT1
    live = vol["vp_mean"] != 0 if "vp_mean" in vol else None
    litho = facies_from_vshale(vol["vshale_mean"], live)
    classified = False
    fac_file = Path(args.facies_segy or Path(args.input_dir) / f"facies_{args.tag}.segy")
    if args.facies_classified or args.facies_segy:
        if fac_file.exists():
            litho = read_segy(fac_file)[0]
            classified = True
            print(f"  facies from the classification ({fac_file})")
        else:
            print(f"  {fac_file.name} not found — facies from Vshale cut-offs")

    # ---- Figure 1: inversion results ------------------------------------------------
    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    panels = [("vp_mean", "Mean Vp", (2800, 4300)), ("vs_mean", "Mean Vs", (1500, 2300)),
              ("rho_mean", "Mean Rho", (2.45, 2.7)), ("poro_mean", "Mean Porosity", (0, 0.18)),
              ("vshale_mean", "Mean Vshale", (0, 1.0))]
    for ax, (key, title, clim) in zip(axs.ravel(), panels):
        if key in vol:
            section_panel(fig, ax, vol[key], x, t1, title, clim, horizons, ylim)
        else:
            ax.set_axis_off()
    # facies: discrete colours (those of the well panels), one per class, outside the
    # inversion window the background colour of the property sections
    n_fac = 4 if (litho.max() > 2 or classified) else 3
    ax = axs[2, 1]
    fac_cmap = ListedColormap(FACIES_CMAP.colors[:n_fac])
    fac_cmap.set_bad(NEW_CMAP(0))
    in_window = live if live is not None else np.ones(litho.shape, bool)
    im = image(ax, np.ma.masked_where(~in_window, litho), x, t1, fac_cmap, (-0.5, n_fac - 0.5))
    cb = fig.colorbar(im, ax=ax, ticks=np.arange(n_fac))
    cb.set_ticklabels(["Shale", "Silt", "Sand", "Cal. Sand"][:n_fac])
    cb.ax.tick_params(labelsize=13, length=0)
    for h in horizons:
        ax.plot(x, h, "--k", lw=1)
    ax.set_ylim(ylim[1], ylim[0])
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("TWT (sec)")
    ax.set_title("Lithological facies", fontsize=16)
    fig.tight_layout()
    save(fig, out_dir, f"Stochastic_inversion_run_{args.run_name}_25", args.formats, args.dpi)

    # ---- Figure 2: EEI computed from the mean elastic volumes -------------------------
    b, a = butter_bandpass(3, 50, 1000, 4)
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.6))
    for ax, chi in zip(axs.ravel(), (-60, -7.5, 25, 45)):
        eei_cal = np.zeros_like(vol["vp_mean"])
        for k in range(nx):
            # window of the trace on the 1 ms grid
            s = slice(4 * top[k] + 3, 4 * base[k] + 3)
            e = eeimp2(vol["vp_mean"][s, k], vol["vs_mean"][s, k], vol["rho_mean"][s, k], chi)
            eei_cal[s, k] = filtfilt_odd(b, a, e)
        section_panel(fig, ax, eei_cal, x, t1, f"Cal. seismic EEI({chi:g})", None,
                      horizons, ylim)
    fig.tight_layout()
    save(fig, out_dir, f"Cal_seismic_data_{args.run_name}", args.formats, args.dpi)

    # ---- Figure 3: well 9P vs inversion along the well path -----------------------------
    npoints = 800
    x_line = np.linspace(*WELL9P_X, npoints)
    t_line = np.linspace(*WELL9P_T, npoints)
    pts = np.column_stack([t_line, x_line])
    along = {kk: RegularGridInterpolator((t1, x), v, bounds_error=False)(pts)
             for kk, v in vol.items()}
    # facies are classes: nearest neighbour, not linear interpolation of the codes
    litho_seis = RegularGridInterpolator((t1, x), litho, method="nearest",
                                         bounds_error=False)(pts)

    well = None
    if Path(args.well_mat).exists():
        from scipy.io import loadmat
        m = loadmat(args.well_mat, squeeze_me=True)
        well = {kk: np.asarray(m[kk], float) for kk in
                ("times2", "vp2", "vs2", "rho2", "por2", "vsh2", "litho2")}
        for kk in ("vp2", "vs2", "rho2", "por2", "vsh2"):
            well[kk][well[kk] <= -999] = np.nan            # LAS null value -999.25
        well["t"] = well["times2"]
    else:
        print(f"  {args.well_mat} not found — Figure 3 shows the inversion only, "
              "no well logs or correlations")

    # inversion curves and facies along the well path, on the same two-way time as the logs
    t_plot = t_line

    # (inversion key, well variable, x label, x limits, x ticks)
    curves = [("poro", "por2", r"$\phi$ (v/v)", (0, 0.25), [0, 0.1, 0.2]),
              ("vp", "vp2", "Vp (m/s)", (2800, 4300), [3000, 4000]),
              ("vs", "vs2", "Vs (m/s)", (1500, 2500), [1600, 2400]),
              ("rho", "rho2", r"$\rho$ (g/cm$^3$)", (2.40, 2.70), [2.4, 2.6]),
              ("vshale", "vsh2", "Vshale(v/v)", (0, 1.0), [0, 0.5, 1])]
    g_fmt = FuncFormatter(lambda v, _: f"{v:g}")             # tick labels: 2.9, 3, 0.5
    # style: Helvetica-like font, inward ticks on all sides, faint grid, pure-green q25
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica",
                                             "Nimbus Sans", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans",
                         "mathtext.it": "Liberation Sans:italic",
                         "mathtext.bf": "Liberation Sans:bold",
                         "mathtext.sf": "Liberation Sans", "mathtext.cal": "Liberation Sans",
                         "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True,
                         "axes.linewidth": 0.8, "grid.color": "0.85", "grid.linewidth": 0.6}):
        fig, axs = plt.subplots(1, 7, figsize=(12, 7), sharey=True)
        # 7 equal panels from x=0.13 to 0.905
        fig.subplots_adjust(left=0.13, right=0.905, bottom=0.11, top=0.925, wspace=0.58)
        for i, (ax, (p, wk, label, xlim, ticks)) in enumerate(zip(axs, curves)):
            if well is not None:
                ax.plot(well[wk], well["t"], "-k", lw=1.5, label="Actual")
            for st, color, name in (("mean", "b", "Inverted"),
                                    ("q25", (0, 1, 0), "Quantile 25"),
                                    ("q75", Q75_COLOR, "Quantile 75")):
                if f"{p}_{st}" in along:
                    ax.plot(along[f"{p}_{st}"], t_plot, color=color, lw=1.5, label=name)
            ax.set_xlim(xlim)
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(g_fmt)
            ax.tick_params(labelsize=14, length=4)
            ax.tick_params(axis="x", pad=10)
            ax.set_xlabel(label, fontsize=16, fontweight="bold")
            ax.grid(True)
            if i > 0:
                ax.tick_params(labelleft=False)
        axs[0].set_ylabel("Time (sec)", fontsize=16, fontweight="bold")
        axs[0].set_ylim(YLIM_WELL[1], YLIM_WELL[0])
        axs[0].set_yticks(np.arange(2.90, 3.0161, 0.02))
        axs[0].yaxis.set_major_formatter(g_fmt)
        # legend of the rho panel, placed across the Vs/rho gap
        axs[3].legend(loc="upper left", bbox_to_anchor=(0.39, 0.882),
                      bbox_transform=fig.transFigure, fontsize=13, fancybox=False,
                      framealpha=1, edgecolor="k", borderaxespad=0, handlelength=2.0,
                      handletextpad=0.3, labelspacing=0.25, borderpad=0.3)

        image(axs[5], litho_seis[:, None], np.array([1.0, 2.0]), t_plot, FACIES_CMAP, (0, 3))
        axs[5].set_xlabel("Est. Facies", fontsize=16, fontweight="bold")
        if well is not None:
            im = image(axs[6], well["litho2"][:, None], np.array([1.0, 2.0]), well["t"],
                         FACIES_CMAP, (0, 3))
        else:
            im = plt.cm.ScalarMappable(cmap=FACIES_CMAP, norm=plt.Normalize(0, 3))
        axs[6].set_xlabel("Facies", fontsize=16, fontweight="bold")
        for ax in axs[5:]:
            ax.grid(False)
            ax.tick_params(left=False, right=False, top=False, bottom=False,
                           labelleft=False, labelbottom=False)
            ax.set_ylim(YLIM_WELL[1], YLIM_WELL[0])
        cax = fig.add_axes([0.90, 0.125, 0.02, 0.2])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0.4, 1.1, 1.8, 2.6])
        cb.set_ticklabels(["Shale", "Silt", "Sand", "Cal. Sand"])
        cb.ax.tick_params(labelsize=11, length=0, pad=6)
        cb.outline.set_linewidth(0.8)
        save(fig, out_dir, f"Well_9p_elasto_petrophysics_{args.run_name}", args.formats, args.dpi)

    # ---- correlation between well logs and inversion ----------------------------------
    if well is not None:
        idx = (t_plot >= YLIM_WELL[0]) & (t_plot <= YLIM_WELL[1])
        lines = []
        err = ["# property  rmse  bias (inversion - log)  std of the log  rmse/std  ssim   "
               f"(Well 9P, {YLIM_WELL[0]:.3f}-{YLIM_WELL[1]:.3f} s, same samples as the correlation)"]
        for p, wk, *_ in [curves[1], curves[2], curves[3], curves[4], curves[0]]:  # order of the output
            good = np.isfinite(well[wk])
            w = np.interp(t_plot, well["t"][good], well[wk][good], left=np.nan, right=np.nan)
            ok = idx & np.isfinite(w) & np.isfinite(along.get(f"{p}_mean", np.nan * w))
            r = np.corrcoef(w[ok], along[f"{p}_mean"][ok])[0, 1] if ok.sum() > 2 else np.nan
            lines.append(f"{p:7s} r = {r:.3f}")
            d = along[f"{p}_mean"][ok] - w[ok]
            rmse, sd = np.sqrt(np.mean(d ** 2)), w[ok].std()
            err.append(f"{p:7s} {rmse:.5g} {d.mean():.5g} {sd:.5g} {rmse / sd:.3f} "
                       f"{ssim_global(w[ok], along[f'{p}_mean'][ok]):.3f}")
        print(f"correlation well vs inversion ({YLIM_WELL[0]:.3f}-{YLIM_WELL[1]:.3f} s):\n  " + "\n  ".join(lines))
        (out_dir / f"well_correlation_{args.run_name}.txt").write_text("\n".join(lines) + "\n")
        (out_dir / f"well_errors_{args.run_name}.txt").write_text("\n".join(err) + "\n")


if __name__ == "__main__":
    main()
