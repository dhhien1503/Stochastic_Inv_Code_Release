"""ODiSI: per-trace stochastic inversion of EEI with pseudo-wells (manuscript configuration).

Workflow per trace:
  1. observed EEI(chi) = 0.65 * convz(A cos(chi) + B sin(chi), wavelet.txt), muted outside the
     live window
  2. Markov-chain facies pseudo-wells for the 3 macro layers ML1-ML3 (pseudowell/markov.py)
  3. rock physics of Table 2 of the manuscript (pseudowell/rock_physics.py)
     -> Vp, Vs, rho, porosity, Vshale
  4. synthetic = global amplitude scale * filtfilt(Butterworth 3-60 Hz, EEI) at 1 ms, resampled
     to 4 ms; the scale is calibrated on the line (Connolly & Hughes 2016)
  5. misfit = sum of squared differences per macro layer, each pseudo-well with its best static
     shift within +-4 ms; the best 50 pseudo-wells of each layer are joined into composites
  6. mean / std / quartiles and facies probabilities of the 50 composites

Usage:
  python odisi_inversion.py --chi 45 --outdir results/chi_45      # full line
  python odisi_inversion.py --nsim 500 --traces 0:60 --no-plots    # quick test
"""
import argparse
import time as _time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from paths import DATA_DIR
from pseudowell.markov import markov_chain_simulation_lambda
from pseudowell.plotting import NEW_CMAP, litho_qc, pseudowell_panel, section, wiggle_qc
from pseudowell.rock_physics import pseudo_well_properties
from pseudowell.segy_io import read_segy, resolve_path, write_segy
from pseudowell.seismic import butter_bandpass, convz, eeimp2, filtfilt_odd, r_square, resamp

T2 = [[0, 0.6667, 0.3333], [1, 0, 0], [0.6667, 0.3333, 0]]
LAMBDA2 = [63.9, 558.6, 1123.6]
T3 = [[0, 0.5, 0.5, 0], [1, 0, 0, 0], [0.5, 0, 0, 0.5], [0, 0, 1.0, 0]]
LAMBDA3 = [198.8423, 300.5908, 312.7351, 845.823]

CFG = dict(
    horizon_file="live_window.sgy",    # 1 inside the live window of the seismic line, 0 outside
    avo_a_file="avo_intercept_A.sgy",
    avo_b_file="avo_gradient_B.sgy",
    wavelet_file="wavelet.txt",
    seis_scale=0.65,
    chi=25.0,
    dt=0.004,
    ds=0.001,
    dx=12.5,
    nsim=15000,
    num_samp=50,
    # macro layers ML1-ML3, (anchor, offset): anchor = first/last live sample of the trace;
    # ML1 and ML3 share the transition matrix and bed thicknesses (Table 1 of the manuscript)
    layers=[
        dict(name="layer2", top=("first", 5), base=("first", 17), T=T2, lam=LAMBDA2),
        dict(name="layer3", top=("first", 17), base=("last", -19), T=T3, lam=LAMBDA3),
        dict(name="layer4", top=("last", -19), base=("last", -5), T=T2, lam=LAMBDA2),
    ],
    # trends of Table 2 (t in s): sand porosity a1 t + b1, shale Vp a2 t + b2, calcareous-sand
    # Vp a3 t + b3, silt porosity a4 t + b4; brine and mineral moduli (GPa) and densities (g/cm3)
    rock=dict(rhow=1.05, Kw=2.438, Kmat=36.5, rhomat=2.7,
              a1=0.0278, b1=0.0272, a2=2.6823e3, b2=-4.4394e3, a3=9.2290e3, b3=-2.2821e4,
              a4=-0.1085, b4=0.3575),
    butter=dict(f_low=3, f_high=60, fs=1000, order=4),
    match=dict(scale="auto", shifts=(-4, -2, 0, 2, 4), per_layer=True),
    qc_trace=48,                       # trace of the QC figures (0-based)
    carry=True,                        # carry-over of the initial-facies probabilities
)
SEGY_TAG = "2x_9P_line_EEI_run_10_Nov_25"
PNG_TAG = "2x_9p_log_lambda_dummy_eei58_run_10_Nov_25"
RUN_TAG = "run_10_Nov_25"

PROPS = ("vp", "vs", "rho", "poro", "vshale")
FACIES = ("shale", "silt", "sand", "calcareous_sand")   # facies codes 0..3


# ----------------------------------------------------------------------------
def layer_bounds(layers, i_first, i_last):
    def idx(spec):
        anchor, off = spec
        return (i_first if anchor == "first" else i_last) + off
    bounds = [(idx(L["top"]), idx(L["base"])) for L in layers]
    for (_, b), (t, _) in zip(bounds, bounds[1:]):
        if not np.all(b == t):
            raise ValueError("macro layers must be contiguous")
    return bounds


def load_observed(cfg, data_dir):
    """seis = 0.65 * convz(A cos(chi) + B sin(chi), wavelet) .* mask"""
    trcdat, _ = read_segy(resolve_path(cfg["horizon_file"], data_dir))
    avo_a, _ = read_segy(resolve_path(cfg["avo_a_file"], data_dir))
    avo_b, _ = read_segy(resolve_path(cfg["avo_b_file"], data_dir))
    wavelet = np.loadtxt(resolve_path(cfg["wavelet_file"], data_dir))
    chi = np.deg2rad(cfg["chi"])
    ref = avo_a * np.cos(chi) + avo_b * np.sin(chi)
    nt, nx = trcdat.shape
    live = trcdat != 0
    i_first = live.argmax(axis=0)
    i_last = nt - 1 - live[::-1].argmax(axis=0)
    rows = np.arange(nt)[:, None]
    mask = (rows >= i_first) & (rows <= i_last)
    seis = cfg["seis_scale"] * convz(ref, wavelet) * mask
    return seis, i_first, i_last


def simulate(facies, twt, rng, cfg, b_f, a_f):
    """Rock physics + band-limited EEI synthetic for each facies column."""
    vp, vs, rho, por, vsh = pseudo_well_properties(twt, facies, cfg["rock"], rng)
    props = dict(vp=vp, vs=vs, rho=rho, poro=por, vshale=vsh, litho=facies)
    return props, forward(props, cfg, b_f, a_f)


def forward(props, cfg, b_f, a_f):
    """EEI -> Butterworth band-pass (filtfilt) -> global amplitude scale."""
    eei = eeimp2(props["vp"], props["vs"], props["rho"], cfg["chi"])
    return cfg["match"]["scale"] * filtfilt_odd(b_f, a_f, eei, axis=0)


def shift_rows(a, sh):
    """Static shift by sh samples along axis 0 (positive = later), edge-padded."""
    if sh == 0:
        return a
    out = np.empty_like(a)
    if sh > 0:
        out[sh:], out[:sh] = a[:-sh], a[:1]
    else:
        out[:sh], out[sh:] = a[-sh:], a[-1:]
    return out


def rank_matches(syn, obs, cfg, edges):
    """Rank pseudo-wells against the observed trace by the sum of squared differences.

    Returns, per macro layer, the realisation order and the static shift (1 ms samples) chosen
    for each realisation. Without per-layer matching (amplitude calibration) every layer gets the
    same whole-window order.
    """
    m = cfg["match"]
    shifts = list(m["shifts"])
    ds, dt = cfg["ds"], cfg["dt"]
    nl = len(edges) - 1
    sse = np.empty((nl, len(shifts), syn.shape[1]))
    for i, sh in enumerate(shifts):
        res2 = (resamp(shift_rows(syn, sh), ds, dt) - obs[:, None]) ** 2
        for L in range(nl):
            sse[L, i] = res2[edges[L]:edges[L + 1]].sum(axis=0)
    sse = np.nan_to_num(sse, nan=np.inf)
    if not m["per_layer"]:
        tot = sse.sum(axis=0)                              # (nshift, nsim)
        best_sh = tot.argmin(axis=0)
        order = np.argsort(tot[best_sh, np.arange(tot.shape[1])], kind="stable")
        return [(order, np.take(shifts, best_sh)[order])] * nl
    out = []
    for L in range(nl):
        best_sh = sse[L].argmin(axis=0)
        order = np.argsort(sse[L][best_sh, np.arange(sse.shape[2])], kind="stable")
        out.append((order, np.take(shifts, best_sh)[order]))
    return out


def composite(props, ranking, edges_fine, n):
    """Build the n best pseudo-wells: layer L of composite i is the i-th best
    (shifted) realisation for that layer."""
    out = {}
    for name, a in props.items():
        c = np.empty((a.shape[0], n), dtype=a.dtype)
        for L, (order, shifts) in enumerate(ranking):
            fs = slice(edges_fine[L], edges_fine[L + 1])
            for i in range(n):
                c[fs, i] = shift_rows(a[:, order[i]:order[i] + 1], int(shifts[i]))[fs, 0]
        out[name] = c
    return out


def invert_trace(k, tops, bases, seis_col, cfg, seed, qc_mode, return_syn=0):
    rng = np.random.default_rng([seed, k])
    ds, dt = cfg["ds"], cfg["dt"]
    q = int(round(dt / ds))
    b_f, a_f = butter_bandpass(**cfg["butter"])

    facies = np.vstack([
        markov_chain_simulation_lambda(L["T"], q * (b - t), cfg["nsim"], L["lam"], ds,
                                       seed=int(rng.integers(2**31 - 1)), carry=cfg["carry"])
        for L, t, b in zip(cfg["layers"], tops, bases)])
    ns = facies.shape[0]
    top, base = tops[0], bases[-1]
    twt = (top + 1) * dt + np.arange(ns) * ds          # times of the window samples
    tnew = (top + 1) * dt + (np.arange(ns) + 1) * ds    # times one sample later (QC figures)
    edges = [t - top for t in tops] + [base - top]      # layer edges, coarse samples
    edges_fine = [q * e for e in edges]

    props, syn = simulate(facies, twt, rng, cfg, b_f, a_f)
    obs = seis_col[top:base]
    if return_syn:                     # used to calibrate the amplitude scale
        r = np.nan_to_num(r_square(resamp(syn, ds, dt), obs), nan=-np.inf)
        best = np.argsort(-r)[:return_syn]
        return dict(k=k, syn=resamp(syn[:, best], ds, dt), obs=obs)
    ranking = rank_matches(syn, obs, cfg, edges)

    best = composite(props, ranking, edges_fine, cfg["num_samp"])
    syn_b = forward(best, cfg, b_f, a_f)
    out = dict(k=k)
    for name in PROPS:
        a = best[name]
        out[name + "_mean"] = a.mean(axis=1)
        out[name + "_std"] = a.std(axis=1, ddof=1)                         # sample std
        out[name + "_q25"] = np.quantile(a, 0.25, axis=1, method="hazen")  # piecewise-linear quantile
        out[name + "_q75"] = np.quantile(a, 0.75, axis=1, method="hazen")
    for j, f in enumerate(FACIES):        # facies posterior: fraction of the best pseudo-wells
        out[f"prob_{f}"] = (best["litho"] == j).mean(axis=1)
    out["ntg_mean"] = 1.0 - out["vshale_mean"]
    out["syn_mean"] = syn_b.mean(axis=1)
    # match quality of the delivered models: best pseudo-well and mean synthetic
    out["r_best"] = r_square(resamp(syn_b[:, :1], ds, dt), obs)[0]
    out["r_mean"] = r_square(resamp(out["syn_mean"][:, None], ds, dt), obs)[0]

    if qc_mode:
        # observed trace interpolated onto the 1 ms pseudo-well times
        xp = (np.arange(top, base + 1) + 1) * dt
        qc = dict(tnew=tnew, obs=np.interp(tnew, xp, seis_col[top:base + 1]),
                  litho100=composite(dict(litho=facies), ranking, edges_fine, 100)["litho"])
        if qc_mode == "best100":
            for name in ("best", "first"):
                p = ({kk: v[:, :100] for kk, v in props.items()} if name == "first"
                     else composite(props, ranking, edges_fine, 100))
                qc[name] = dict(p, syn=forward(p, cfg, b_f, a_f))
        out["qc"] = qc
    return out


def calibrate_scale(cfg, bounds, seis, traces, seed, n_jobs, nbest=30, n_iter=4):
    """Global synthetic-to-seismic amplitude scale (Connolly & Hughes 2016):
    rank, least-squares scale of the best matches over a set of traces, repeat."""
    sub = list(traces)[::10]
    c = dict(cfg, nsim=min(cfg["nsim"], 3000),
             match=dict(cfg["match"], scale=1.0, shifts=(0,), per_layer=False))
    scale = 1.0
    for _ in range(n_iter):
        cc = dict(c, match=dict(c["match"], scale=scale))
        res = Parallel(n_jobs=n_jobs)(delayed(invert_trace)(
            k, [b[0][k] for b in bounds], [b[1][k] for b in bounds], seis[:, k], cc,
            seed + 7, None, return_syn=nbest) for k in sub)
        num = sum((r["syn"] * r["obs"][:, None]).sum() for r in res)
        den = sum((r["syn"] ** 2).sum() for r in res)
        scale *= num / den
    return scale


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--nsim", type=int, default=CFG["nsim"])
    ap.add_argument("--num-samp", type=int, default=CFG["num_samp"])
    ap.add_argument("--chi", type=float, default=CFG["chi"])
    ap.add_argument("--traces", default=None, help="python slice, e.g. 0:60")
    ap.add_argument("--n-jobs", type=int, default=-1, help="default: all cores")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-grid-shift", action="store_true",
                    help="write SEG-Y aligned to the input time axis instead of "
                         "the default 1 ms grid, which is 3 samples late")
    args = ap.parse_args()

    cfg = dict(CFG, chi=args.chi, num_samp=args.num_samp, nsim=args.nsim, match=dict(CFG["match"]))
    n_jobs = args.n_jobs
    out_dir = Path(args.outdir)
    dt, ds = cfg["dt"], cfg["ds"]
    q = int(round(dt / ds))

    seis, i_first, i_last = load_observed(cfg, args.data_dir)
    nt, nx = seis.shape
    t = (np.arange(nt) + 1) * dt
    x = (np.arange(nx) + 1) * cfg["dx"]
    ylim = (t[i_first.min()], t[i_last.max()])
    bounds = layer_bounds(cfg["layers"], i_first, i_last)
    top, base = bounds[0][0], bounds[-1][1]
    horizons = [t[b[0]] for b in bounds] + [t[base]]
    hz = dict(horizons=horizons)
    chi_lbl = r"($\chi$=%g)" % cfg["chi"]

    traces = range(nx) if args.traces is None else range(nx)[slice(*[
        int(s) if s else None for s in args.traces.split(":")])]
    print(f"{len(traces)} traces x {cfg['nsim']} pseudo-wells, chi={cfg['chi']}, n_jobs={n_jobs}")

    cfg["match"]["scale"] = calibrate_scale(cfg, bounds, seis, traces, args.seed, n_jobs)
    m = cfg["match"]
    print(f"band {cfg['butter']['f_low']:g}-{cfg['butter']['f_high']:g} Hz, "
          f"scale {m['scale']:.3f}, shifts {list(m['shifts'])} ms, per-layer {m['per_layer']}")

    last = traces[-1]
    qc_of = lambda k: "best100" if k == cfg["qc_trace"] else "litho" if k == last else None
    jobs = (delayed(invert_trace)(k, [b[0][k] for b in bounds], [b[1][k] for b in bounds],
                                  seis[:, k], cfg, args.seed, qc_of(k))
            for k in traces)
    t0 = _time.time()
    results = list(tqdm(Parallel(n_jobs=n_jobs, return_as="generator")(jobs), total=len(traces)))
    print(f"total {_time.time() - t0:.1f} s")

    # ---- assemble on the 1 ms grid --------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [kk for kk in results[0] if kk not in ("k", "r_best", "r_mean", "qc")]
    vol = {kk: np.zeros((q * nt, nx)) for kk in keys}
    r_best = np.full(nx, np.nan)
    r_mean = np.full(nx, np.nan)
    qc = {}
    for res in results:
        k = res["k"]
        s = slice(q * top[k], q * base[k])
        for kk in keys:
            vol[kk][s, k] = res[kk]
        r_best[k] = res["r_best"]
        r_mean[k] = res["r_mean"]
        if "qc" in res:
            qc[k] = res["qc"]
    np.savez_compressed(out_dir / f"inversion_{PNG_TAG}.npz", r_best=r_best, r_mean=r_mean,
                        t_coarse=t, x=x, ds=ds, seis=seis, **vol)
    print(f"R^2 vs observed EEI: best pseudo-well {np.nanmean(r_best):.3f}, "
          f"mean of best {cfg['num_samp']} {np.nanmean(r_mean):.3f}")

    # ---- SEG-Y outputs --------------------------------------------------------
    tmpl = resolve_path(cfg["horizon_file"], args.data_dir)
    shift = 0 if args.no_grid_shift else q - 1
    for st in ("mean", "std", "q25", "q75"):
        for name in ("vshale", "poro", "vp", "vs", "rho"):
            a = np.zeros_like(vol[f"{name}_{st}"])
            a[shift:] = vol[f"{name}_{st}"][:a.shape[0] - shift]
            write_segy(out_dir / f"{name}_{st}_{SEGY_TAG}.segy", a, ds, tmpl)
    # facies classification: posterior probabilities and the most probable facies
    probs = np.stack([vol[f"prob_{f}"] for f in FACIES])
    live = vol["vp_mean"] != 0
    vol["facies"] = np.where(live, probs.argmax(0), 0).astype(float)
    for name in [f"prob_{f}" for f in FACIES] + ["facies"]:
        a = np.zeros_like(vol[name])
        a[shift:] = vol[name][:a.shape[0] - shift]
        write_segy(out_dir / f"{name}_{SEGY_TAG}.segy", a, ds, tmpl)
    print(f"outputs in {out_dir}/")
    if args.no_plots:
        return

    # ---- QC figures -------------------------------------------------------------
    P = lambda name: out_dir / f"{name}_{PNG_TAG}.png"
    sec = lambda name, a, title, **kw: section(P(name), a, x, t, title, ylim=ylim, **kw)
    sec("seismic_input", seis, f"Seismic EEI {chi_lbl}", cmap=NEW_CMAP, **hz)
    sec("P_mean_invert", vol["vp_mean"], f"Mean P velocity {chi_lbl}", cmap=NEW_CMAP,
        clim=(3000, 4500), **hz)
    sec("S_mean_invert", vol["vs_mean"], f"Mean S velocity {chi_lbl}", cmap=NEW_CMAP,
        clim=(1500, 2500), **hz)
    sec("Rho_mean_invert", vol["rho_mean"], f"Mean Density {chi_lbl}", cmap=NEW_CMAP,
        clim=(2.44, 2.80), **hz)
    sec("poro_mean_invert", vol["poro_mean"], f"Mean Porosity {chi_lbl}", clim=(0, 0.15))
    sec("vshale_mean_invert", vol["vshale_mean"], f"Mean Vshale {chi_lbl}", cmap=NEW_CMAP,
        clim=(0, 1), **hz)
    sec("Litho_mean_invert", vol["facies"], f"Most probable facies {chi_lbl}", cmap=NEW_CMAP,
        clim=(0, 3), **hz)
    rest = PNG_TAG.split("_", 2)[2]       # tag without the leading '2x_9p_'
    if last in qc:
        litho_qc(out_dir / f"Best_50_litho_well2x_{rest}.png", qc[last]["tnew"],
                 qc[last]["litho100"])
    d = qc.get(cfg["qc_trace"])
    if d is not None:
        for which, label in (("best", "Best"), ("first", "First")):
            b = d[which]
            wiggle_qc(out_dir / f"{label}_100_psedowells_seismic_at_well_9p_{rest}.png",
                      d["tnew"], b["syn"], d["obs"], size_cm=(20, 10), dpi=600,
                      xlabel="Pseudo Well #")
            pseudowell_panel(out_dir / f"{label}_100_wells_{RUN_TAG}.png", d["tnew"],
                             b["vp"], b["vs"], b["rho"], b["poro"], b["vshale"], b["litho"])


if __name__ == "__main__":
    main()
