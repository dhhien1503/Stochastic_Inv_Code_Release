"""Small helpers shared by the notebooks: paths, running the library scripts, showing figures.

The notebooks call the scripts of ODiSI_libs/ and CNN_libs/ exactly as they are, so that the
results are the same as from the command line. Every step writes its full log next to its
outputs; the notebook prints only the last lines.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ODISI_LIBS = ROOT / "ODiSI_libs"
CNN_LIBS = ROOT / "CNN_libs"
DATA_DIR = ROOT / "Data"
OUT = ROOT / "Notebooks" / "outputs"          # everything the notebooks compute goes here
ODISI_OUT = OUT / "ODiSI"
CNN_OUT = OUT / "CNN"
TRAIN_DIR = CNN_OUT / "training_data"         # 100,000 pseudo-wells, EEI at the four angles

for p in (str(ODISI_LIBS), str(CNN_LIBS)):
    if p not in sys.path:
        sys.path.insert(0, p)

CHIS_4 = (-60.0, -7.5, 25.0, 45.0)             # EEI angles of the manuscript
PROPS = ("vp", "vs", "rho", "poro", "vshale")
PROP_LABELS = dict(vp="Vp (m/s)", vs="Vs (m/s)", rho="Density (g/cc)", poro="Porosity (v/v)",
                   vshale="Vshale (v/v)")
FACIES = ("shale", "silt", "sand", "calcareous sand")
FACIES_COLORS = ([0.5, 0.5, 0], [0.5, 0.5, 0.5], [1, 1, 0], [1, 0, 0])
WELL9P = DATA_DIR / "well9p_data.mat"          # proprietary, not distributed

# settings of the regression network of the manuscript (also used for the angle tests)
REG_ARGS = ["--arch", "keras", "--epochs", "150", "--batch", "32", "--lr", "1e-3",
            "--weight-decay", "0", "--dropout", "0.3", "--reg-loss", "mse", "--cls-weight", "0",
            "--calibration", "none", "--gain-jitter", "0", "--mc-test", "0"]


def _clean(text):
    """Keep the final state of progress-bar lines (carriage returns), drop the others."""
    lines = [line.split("\r")[-1] for line in text.splitlines()]
    return "\n".join(l for l in lines if "it/s]" not in l or l.lstrip().startswith("100%"))


def run(script, *args, log, tail=12):
    """Run ODiSI_libs/<script> or CNN_libs/<script> with the kernel's Python.

    The full output is written to `log`; the last `tail` lines are printed.
    Returns the elapsed time in seconds.
    """
    path = ODISI_LIBS / script if (ODISI_LIBS / script).exists() else CNN_LIBS / script
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(path), *map(str, args)]
    print("$ python", path.relative_to(ROOT), *[str(a).replace(str(ROOT) + "/", "") for a in args])
    t0 = time.time()
    r = subprocess.run(cmd, cwd=path.parent, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONUNBUFFERED="1"))
    dt = time.time() - t0
    out = _clean(r.stdout + ("\n" + r.stderr if r.stderr.strip() else ""))
    out = out.replace(str(ROOT) + "/", "")          # paths relative to the package root
    log.write_text(out + "\n")
    lines = [l for l in out.splitlines() if l.strip()]
    print("  ...\n" if len(lines) > tail else "", end="")
    print("\n".join("  " + l for l in lines[-tail:]))
    print(f"[{dt:.0f} s, full log: {log.relative_to(ROOT)}]")
    if r.returncode:
        raise RuntimeError(f"{script} failed, see {log}")
    return dt


def show(*paths, width=950):
    """Display PNG figures in the notebook."""
    from IPython.display import Image, display
    for p in paths:
        p = Path(p)
        if p.exists():
            display(Image(filename=str(p), width=width))
        else:
            print(f"(figure not found: {p.relative_to(ROOT)})")


def first(folder, pattern):
    """First file in `folder` matching `pattern` (figure names carry run tags)."""
    hits = sorted(Path(folder).glob(pattern))
    return hits[0] if hits else Path(folder) / pattern


def well_logs_available():
    return WELL9P.exists()


def read_well_correlation(txt):
    """Correlation coefficients at Well 9P written by make_paper_figures.py ({} if absent)."""
    txt = Path(txt)
    if not txt.exists():
        return {}
    r = {}
    for line in txt.read_text().splitlines():
        if "r =" in line:
            p, v = line.split("r =")
            r[p.strip()] = float(v)
    return r


def test_correlations(run_dir):
    """Median correlation per property on the test pseudo-wells (metrics.json of train_cnn.py)."""
    m = json.loads((Path(run_dir) / "metrics.json").read_text())
    return {p: m["test"][p]["corr_median"] for p in PROPS}


def environment():
    import platform
    print(f"Python {platform.python_version()}, numpy {np.__version__}")
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
        print(f"PyTorch {torch.__version__}, GPU: {gpu}")
    except ImportError:
        print("PyTorch not installed (needed for the CNN notebooks)")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Package root: {ROOT.name}/")
    print(f"Well 9P logs: {'found' if well_logs_available() else 'not distributed (blind-well comparison skipped)'}")


# ---------------------------------------------------------------------------------------------
# plots made directly in the notebooks

def plot_field_eei(chis=CHIS_4, band=(3.0, 60.0)):
    """AVO intercept, gradient and field EEI at the given angles, with the inversion window."""
    import matplotlib.pyplot as plt
    from cnn_common import field_eei_4ms, geometry
    from pseudowell.segy_io import read_segy

    a, dt = read_segy(DATA_DIR / "avo_intercept_A.sgy")
    b, _ = read_segy(DATA_DIR / "avo_gradient_B.sgy")
    eei = field_eei_4ms(DATA_DIR, chis, band=band)
    tops, bases, nt, nx = geometry(DATA_DIR)
    bounds = [np.asarray(tp, int) for tp in tops] + [np.asarray(bases[-1], int)]
    i0, i1 = max(bounds[0].min() - 25, 0), min(bounds[-1].max() + 25, nt)
    t = (np.arange(nt) + 1) * dt
    x = np.arange(1, nx + 1)
    panels = [("AVO intercept A", a), ("AVO gradient B", b)] + \
             [(f"EEI({c:g}°)", e) for c, e in zip(chis, eei)]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(15, 3.6 * nrow), sharey=True, squeeze=False)
    for ax, (title, d) in zip(axs.flat, panels):
        v = np.percentile(np.abs(d[i0:i1]), 98)
        ax.imshow(d[i0:i1], aspect="auto", cmap="seismic", vmin=-v, vmax=v,
                  extent=(x[0], x[-1], t[i1 - 1], t[i0]))
        for bd in bounds:
            ax.plot(x, t[bd], "k--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Trace")
    for ax in axs[:, 0]:
        ax.set_ylabel("TWT (s)")
    for ax in list(axs.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Field data of the line (dashed: macrolayer boundaries)", y=1.0)
    fig.tight_layout()
    plt.show()


def plot_pseudowells(train_dir=TRAIN_DIR, idx=(0, 1, 2), chis=None):
    """A few training pseudo-wells: facies, the five properties and their synthetic EEI."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    train_dir = Path(train_dir)
    cfg = json.loads((train_dir / "config.json").read_text())
    chis = chis or cfg["chis"]
    X = np.load(train_dir / "eei.npy", mmap_mode="r")
    Y = np.load(train_dir / "props.npy", mmap_mode="r")
    F = np.load(train_dir / "facies.npy", mmap_mode="r")
    n = int(sum(cfg["ns_layer"]))
    t = cfg["twt"][0] + np.arange(n) * 0.001
    cmap = ListedColormap(FACIES_COLORS)
    for i in idx:
        fig, axs = plt.subplots(1, 7, figsize=(15, 4.5), sharey=True,
                                gridspec_kw=dict(width_ratios=[0.4, 1, 1, 1, 1, 1, 1.4]))
        axs[0].imshow(F[i, :n, None], aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
                      extent=(0, 1, t[-1], t[0]), interpolation="nearest")
        axs[0].set_xticks([])
        axs[0].set_title("Facies")
        axs[0].set_ylabel("TWT (s)")
        for k, p in enumerate(PROPS):
            axs[k + 1].plot(Y[i, k, :n], t, "b", lw=1)
            axs[k + 1].set_title(PROP_LABELS[p], fontsize=10)
        gap = 2.2 if len(chis) <= 6 else 1.0
        for j, c in enumerate(chis):
            e = X[i, j, :n]
            axs[6].plot(e / (np.abs(e).max() + 1e-12) + gap * j, t, lw=1)
        step = 1 if len(chis) <= 6 else 6
        axs[6].set_xticks([gap * j for j in range(0, len(chis), step)])
        axs[6].set_xticklabels([f"{c:g}°" for c in chis[::step]], fontsize=9)
        axs[6].set_title("Synthetic EEI", fontsize=10)
        axs[0].set_ylim(t[-1], t[0])
        fig.suptitle(f"Pseudo-well #{i}", y=1.0)
        fig.tight_layout()
        plt.show()
    frac = np.bincount(np.asarray(F[:, :n]).ravel(), minlength=4) / F[:, :n].size
    print("Facies fractions over the training set: " +
          ", ".join(f"{f} {100 * v:.1f}%" for f, v in zip(FACIES, frac)))


def plot_history(run_dir, title="Regression CNN"):
    """Training and validation loss from history.csv of train_cnn.py."""
    import matplotlib.pyplot as plt
    h = np.genfromtxt(Path(run_dir) / "history.csv", delimiter=",", names=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(h["epoch"], h["train_reg"], label="training")
    ax.plot(h["epoch"], h["val_reg"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (standardised properties)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()


def table(rows, cols, fmt="{:.3f}", index_name=""):
    """Show a small table (dict of dicts) as Markdown in the notebook."""
    from IPython.display import Markdown, display
    head = f"| {index_name} | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    body = []
    for name, r in rows.items():
        cells = [fmt.format(r[c]) if isinstance(r.get(c), (int, float)) else str(r.get(c, "–"))
                 for c in cols]
        body.append(f"| {name} | " + " | ".join(cells) + " |")
    display(Markdown("\n".join([head, sep, *body])))
