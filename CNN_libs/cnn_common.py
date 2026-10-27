"""Shared settings and helpers for the EEI -> elasto-petrophysics CNN.

The training pseudo-wells, the synthetic EEI and the field EEI must be built in
exactly the same way; everything that both make_datamat.py and
infer_cnn.py depend on lives here.

Grid: every sample is a window from the top of macro layer 2 to the base of
macro layer 4 of one trace (the ODiSI window, CFG["layers"]), on the 1 ms
pseudo-well grid, zero-padded to WIN_LEN samples. Sample j of trace k is at
TWT (top_k + 1) * dt + j * ds, the same time axis as odisi_inversion.py,
so the CNN volumes can be written to SEG-Y on the same grid as the ODiSI ones.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ODiSI_libs"))

import numpy as np
from scipy.signal import resample_poly

from pseudowell.segy_io import read_segy, resolve_path
from pseudowell.seismic import butter_bandpass, convz, filtfilt_odd
from odisi_inversion import CFG, layer_bounds

CHIS = np.arange(-60.0, 60.1, 5.0)                 # 25 EEI angles (degrees)
PROPS = ("vp", "vs", "rho", "poro", "vshale")      # regression targets
FACIES = ("shale", "silt", "sand", "calcareous sand")   # classification target
WIN_LEN = 192        # 1 ms samples; the line's windows are 152-184 ms long
BAND = (6.0, 30.0)   # Hz; band of the observed EEI (see README, --improved)
TWT_REF, TWT_SCALE = 2.9, 0.1                       # normalisation of the time channel


def geometry(data_dir):
    """Macro-layer bounds of every trace, in 4 ms samples.

    Returns tops (list of arrays, one per layer), bases (same), nt, nx.
    """
    trcdat, _ = read_segy(resolve_path(CFG["horizon_file"], data_dir))
    nt, nx = trcdat.shape
    live = trcdat != 0
    i_first = live.argmax(axis=0)
    i_last = nt - 1 - live[::-1].argmax(axis=0)
    bounds = layer_bounds(CFG["layers"], i_first, i_last)
    tops = [b[0] for b in bounds]
    bases = [b[1] for b in bounds]
    return tops, bases, nt, nx


def window(tops, bases, k):
    """First 4 ms sample, number of 1 ms samples and start time of trace k's window."""
    q = int(round(CFG["dt"] / CFG["ds"]))
    top, base = int(tops[0][k]), int(bases[-1][k])
    return top, q * (base - top), (top + 1) * CFG["dt"]


def field_components(data_dir, band=BAND):
    """Intercept and gradient parts of the field EEI on the 1 ms window grid.

    The ODiSI scripts build EEI(chi) = 0.65 * convz(A cos(chi) + B sin(chi),
    wavelet.txt). That is linear in A and B, so it is kept here as two traces,
    EA = 0.65 * convz(A, wavelet) and EB = 0.65 * convz(B, wavelet), each with
    the same Butterworth band-pass as the synthetic EEI of the pseudo-wells and
    up-sampled from 4 ms to 1 ms (the signal is band-limited far below the 4 ms
    Nyquist, so this loses nothing). EEI(chi) = cos(chi) EA + sin(chi) EB.

    Returns dict(EA, EB (nx, WIN_LEN) float64, ns, twt0, top (nx,)).
    """
    avo_a, _ = read_segy(resolve_path(CFG["avo_a_file"], data_dir))
    avo_b, _ = read_segy(resolve_path(CFG["avo_b_file"], data_dir))
    wavelet = np.loadtxt(resolve_path(CFG["wavelet_file"], data_dir))
    tops, bases, nt, nx = geometry(data_dir)
    q = int(round(CFG["dt"] / CFG["ds"]))
    b_f, a_f = butter_bandpass(band[0], band[1], 1.0 / CFG["dt"], CFG["butter"]["order"])
    win = [window(tops, bases, k) for k in range(nx)]
    out = {}
    for name, ref in (("EA", avo_a), ("EB", avo_b)):
        seis = CFG["seis_scale"] * convz(ref, wavelet)
        seis = filtfilt_odd(b_f, a_f, seis, axis=0)
        # 4 ms sample i is at (i+1)*dt, so 1 ms sample q*i is at the same time
        up = resample_poly(seis, q, 1, axis=0)
        E = np.zeros((nx, WIN_LEN))
        for k, (top, ns, _) in enumerate(win):
            E[k, :ns] = up[q * top:q * top + ns, k]
        out[name] = E
    out["ns"] = np.array([w[1] for w in win])
    out["twt0"] = np.array([w[2] for w in win])
    out["top"] = np.array([w[0] for w in win])
    return out


def field_eei_4ms(data_dir, chis, band=None, gains=(1.0, 1.0)):
    """Field EEI on the full 4 ms grid of the line, (n_chi, nt, nx) float64:
    0.65 * convz(A cos(chi) + B sin(chi), wavelet.txt), band-passed when `band` is given;
    gains = (gA, gB) scale the intercept and the gradient.
    """
    avo_a, _ = read_segy(resolve_path(CFG["avo_a_file"], data_dir))
    avo_b, _ = read_segy(resolve_path(CFG["avo_b_file"], data_dir))
    wavelet = np.loadtxt(resolve_path(CFG["wavelet_file"], data_dir))
    out = []
    for chi in np.deg2rad(np.asarray(chis, float)):
        seis = CFG["seis_scale"] * convz(gains[0] * avo_a * np.cos(chi) + gains[1] * avo_b * np.sin(chi),
                                         wavelet)
        if band is not None:
            b_f, a_f = butter_bandpass(band[0], band[1], 1.0 / CFG["dt"], CFG["butter"]["order"])
            seis = filtfilt_odd(b_f, a_f, seis, axis=0)
        out.append(seis)
    return np.stack(out)


def stretch_field(eei4, i0, i1, n_out, upsample=True):
    """Resample the window [i0, i1) (4 ms samples) of every trace onto n_out samples,
    to fit the fixed 184-sample training window.

    upsample=False: linear interpolation of the 4 ms samples i0 .. i1-1;
    upsample=True (default) first up-samples to 1 ms (band-limited) and
    interpolates the 1 ms samples 4 i0 .. 4 i1 - 1.
    Returns (nx, n_chi, n_out) float32.
    """
    n_chi, nt, nx = eei4.shape
    q = int(round(CFG["dt"] / CFG["ds"]))
    src = resample_poly(eei4, q, 1, axis=1) if upsample else eei4
    f = q if upsample else 1
    X = np.zeros((nx, n_chi, n_out), np.float32)
    for k in range(nx):
        a, b = f * int(i0[k]), f * int(i1[k])
        grid = np.arange(a, b)
        tq = np.linspace(a, b - 1, n_out)
        for c in range(n_chi):
            X[k, c] = np.interp(tq, grid, src[c, a:b, k])
    return X


def unstretch(Y, i0, i1, n_in):
    """Inverse of stretch_field for the outputs: (nx, ..., n_in) on the stretched axis ->
    (nx, ..., 4*(i1-i0)) on the 1 ms grid of each window (zero padded to the longest)."""
    q = int(round(CFG["dt"] / CFG["ds"]))
    ns = q * (np.asarray(i1) - np.asarray(i0))
    out = np.zeros(Y.shape[:-1] + (int(ns.max()),), Y.dtype)
    src = np.linspace(0, 1, n_in)
    for k in range(Y.shape[0]):
        dst = np.linspace(0, 1, ns[k])
        flat = Y[k].reshape(-1, n_in)
        out[k].reshape(-1, out.shape[-1])[:, :ns[k]] = np.stack([np.interp(dst, src, r) for r in flat])
    return out, ns


def field_eei(comp, chis=CHIS, gain_a=1.0, gain_b=1.0):
    """EEI(chi) = gain_a cos(chi) EA + gain_b sin(chi) EB, (nx, nchi, WIN_LEN) float32."""
    c = np.deg2rad(np.asarray(chis))[None, :, None]
    return (gain_a * np.cos(c) * comp["EA"][:, None] +
            gain_b * np.sin(c) * comp["EB"][:, None]).astype(np.float32)


def calibrate(comp, syn_power, chis=CHIS):
    """Fit the field EEI amplitudes to the synthetic ones, angle by angle.

    Model: the power of the calibrated field EEI at angle chi,
        gA^2 <EA^2> cos^2 + gB^2 <EB^2> sin^2 + 2 gA gB <EA EB> sin cos,
    equals the power of the synthetic EEI plus that of additive noise
        P_syn(chi) + sA^2 cos^2 + sB^2 sin^2,
    where the noise is that of the intercept (sA) and gradient (sB) traces.
    gA and gB are separate intercept and gradient gains (the single global
    scale of ODiSI is the special case gA = gB); sA and sB are used to add
    realistic noise to the training EEI. Fitted in log power over all angles.

    syn_power: mean square of the synthetic EEI per angle (signal only), in the
    same units as the training EEI.
    """
    from scipy.optimize import least_squares
    m = np.arange(WIN_LEN)[None, :] < comp["ns"][:, None]
    n = m.sum()
    a = (comp["EA"] ** 2).sum() / n
    b = (comp["EB"] ** 2).sum() / n
    ab = (comp["EA"] * comp["EB"]).sum() / n
    c = np.deg2rad(np.asarray(chis))
    C2, S2, SC = np.cos(c) ** 2, np.sin(c) ** 2, np.sin(c) * np.cos(c)
    p_syn = np.asarray(syn_power, float)
    p_field = lambda gA, gB: gA ** 2 * a * C2 + gB ** 2 * b * S2 + 2 * gA * gB * ab * SC

    # gains without noise first, as the starting point
    r0 = least_squares(lambda g: np.log(p_field(*g)) - np.log(p_syn), [1.0, 1.0])
    s0 = np.log(0.1 * np.sqrt(p_syn.mean()))
    res = lambda p: (np.log(p_syn + np.exp(2 * p[2]) * C2 + np.exp(2 * p[3]) * S2)
                     - np.log(p_field(p[0], p[1])))
    r = least_squares(res, [*r0.x, s0, s0])
    gA, gB, lsA, lsB = r.x
    sA, sB = np.exp(lsA), np.exp(lsB)
    ratio = np.sqrt((p_syn + sA ** 2 * C2 + sB ** 2 * S2) / p_field(gA, gB))
    return dict(gain_a=float(gA), gain_b=float(gB), noise_a=float(sA), noise_b=float(sB),
                gain_a_nonoise=float(r0.x[0]), gain_b_nonoise=float(r0.x[1]),
                fit_ratio=ratio.tolist(),
                snr=np.sqrt(p_syn / (sA ** 2 * C2 + sB ** 2 * S2 + 1e-30)).tolist(),
                corr_ab_field=float(ab / np.sqrt(a * b)))


def band_response(length=WIN_LEN, band=BAND, ds=CFG["ds"]):
    """|H(f)|^2 of the zero-phase (filtfilt) Butterworth band-pass on the
    rfft frequencies of a `length`-sample trace; used to colour the noise."""
    from scipy.signal import freqz
    b_f, a_f = butter_bandpass(band[0], band[1], 1.0 / ds, CFG["butter"]["order"])
    f = np.fft.rfftfreq(length, ds)
    _, h = freqz(b_f, a_f, worN=f, fs=1.0 / ds)
    return np.abs(h) ** 2


def aux_channels(ns, twt0, length=WIN_LEN, ds=CFG["ds"]):
    """Mask and normalised-TWT channels, (n, 2, length) float32.

    The rock-physics trends depend on absolute time, so the network is given
    the TWT of every sample; the mask tells it where the window ends.
    """
    j = np.arange(length)
    mask = (j[None, :] < np.asarray(ns)[:, None]).astype(np.float32)
    twt = (np.asarray(twt0)[:, None] + j[None, :] * ds - TWT_REF) / TWT_SCALE
    return np.stack([mask, (twt * mask).astype(np.float32)], axis=1)
