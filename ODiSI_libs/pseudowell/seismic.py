"""EEI, wavelet, convolution, resampling and match metric.

Extended elastic impedance, Ormsby wavelet, convolution and resampling, plus
r_square (implemented as the coefficient of determination
against the observed trace, so ranking by it equals ranking by rms error,
the metric used by Connolly & Hughes 2016).
"""
import numpy as np
from scipy.signal import butter, filtfilt, oaconvolve
from scipy.signal.windows import tukey


def eeimp2(vp, vs, rho, chi_deg):
    """Normalised extended elastic impedance, column-wise."""
    chi = np.deg2rad(chi_deg)
    K = np.mean((vs / vp) ** 2, axis=0)
    mvp, mvs, mrho = vp.mean(axis=0), vs.mean(axis=0), rho.mean(axis=0)
    x1 = np.cos(chi) + np.sin(chi)
    x2 = np.cos(chi) - 4.0 * K * np.sin(chi)
    x3 = -8.0 * K * np.sin(chi)
    return mvp * mrho * (vp / mvp) ** x1 * (rho / mrho) ** x2 * (vs / mvs) ** x3


def ormsby(f1, f2, f3, f4, tlen, dt, taper_pct=20):
    """Zero-phase Ormsby wavelet (CREWES ormsby): raised-cosine end tapers and
    normalised so a sine at the dominant frequency passes with unit gain."""
    n = int(round(tlen / dt)) + 1
    nzero = int(np.ceil((n + 1) / 2))
    nr = n - nzero
    nl = n - nr - 1
    tw = dt * np.arange(-nl, nr + 1)
    s = lambda f: np.pi * f ** 2 * np.sinc(f * tw) ** 2
    w = (s(f4) - s(f3)) / (f4 - f3) - (s(f2) - s(f1)) / (f2 - f1)
    win = tukey(n, 2 * taper_pct / 100)
    w *= win
    # Truncation + taper leave ~5% gain at 0 Hz. Because EEI has a mean of
    # ~1e4, that DC leak would add a large offset to every synthetic and
    # dominate the R^2 ranking, so restore the zero DC gain an Ormsby
    # band-pass is designed to have.
    w -= win * w.sum() / win.sum()
    w /= np.abs(np.fft.rfft(w, 8192)).max()
    return w, tw


def convolve_padded(trc, w, pad=100):
    """convz with the driver's edge-value padding (the 'zero tap' block)."""
    padded = np.pad(trc, ((pad, pad), (0, 0)), mode="edge")
    return oaconvolve(padded, w[:, None], mode="same", axes=0)[pad:-pad]


def resamp(trc, dt_in, dt_out, pass_frac=0.6, pad=64):
    """Anti-aliased down-sampling along axis 0 (CREWES resamp equivalent).

    Zero-phase low-pass: flat to pass_frac*Nyquist_out, cosine roll-off to
    Nyquist_out; output samples at t0, t0+dt_out, ...
    """
    q = int(round(dt_out / dt_in))
    padded = np.pad(trc, ((pad, pad), (0, 0)), mode="edge")
    n = padded.shape[0]
    f = np.fft.rfftfreq(n, dt_in)
    fn = 0.5 / dt_out
    f0 = pass_frac * fn
    H = np.where(f <= f0, 1.0,
                 np.where(f >= fn, 0.0, 0.5 * (1 + np.cos(np.pi * (f - f0) / (fn - f0)))))
    out = np.fft.irfft(np.fft.rfft(padded, axis=0) * H[:, None], n, axis=0)
    return out[pad:-pad][::q]


def r_square(syn, obs):
    """Coefficient of determination of each synthetic column vs observed."""
    obs = obs[:, None]
    sse = np.sum((obs - syn) ** 2, axis=0)
    sst = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - sse / sst


def convz(r, w, nzero=None):
    """CREWES convz: convolve each column with w, output the same length as r,
    with sample nzero (1-based, default round((len(w)+1)/2)) of w as time zero.
    (The optional 5% end taper only touches the first/last ~60 samples of the
    1251-sample traces, outside the live window, so it is not applied.)"""
    w = np.asarray(w, dtype=float).ravel()
    if nzero is None:
        nzero = int(np.floor((len(w) + 1) / 2 + 0.5))     # round half up
    r2 = r if r.ndim == 2 else r[:, None]
    full = oaconvolve(r2, w[:, None], mode="full", axes=0)
    out = full[nzero - 1:nzero - 1 + r2.shape[0]]
    return out if r.ndim == 2 else out[:, 0]


def butter_bandpass(f_low, f_high, fs, order=4):
    """[b, a] = butter(order, [f_low f_high]/(fs/2), 'bandpass')."""
    return butter(order, [f_low / (fs / 2), f_high / (fs / 2)], btype="bandpass")


def filtfilt_odd(b, a, x, axis=0):
    """Zero-phase forward-backward filter along ``axis``: odd reflection of
    3*(nfilt-1) samples at both ends and steady-state initial conditions."""
    nfilt = max(len(a), len(b))
    return filtfilt(b, a, x, axis=axis, padtype="odd", padlen=3 * (nfilt - 1))
