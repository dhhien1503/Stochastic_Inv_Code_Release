"""Markov chain lithofacies simulation, following SeReMpy (Grana et al.).

Reference: SeReMpy/Geostats.py ``MarkovChainSimulation`` and ``RandDisc``
(https://github.com/seismicreservoirmodeling/SeReMpy, MIT licence, Dario
Grana 2020; Grana, Mukerji & Doyen 2021, Seismic Reservoir Modeling, ch. 4.4).

* ``rand_disc`` / ``markov_chain_simulation`` reproduce SeReMpy: the initial
  facies is drawn from the first row of T^100 and every sample is drawn from
  the T row of the previous sample (discrete-time chain, 0-based codes).
* ``markov_chain_simulation_lambda`` is the SeReMpy chain made continuous-time,
  as in Connolly & Hughes 2016:
  each facies is held for an exponential thickness -log(u)/lambda[facies]
  (lambda in 1/s), rounded to whole samples, before the next RandDisc draw
  from T. Facies codes are 0..n-1, and the first facies of every T is shale.
"""
import numpy as np
from numba import njit
from numpy.linalg import matrix_power


# --------------------------------------------------------------------------
# SeReMpy reference implementation
# --------------------------------------------------------------------------
def rand_disc(p):
    """SeReMpy RandDisc: sample an index from probability mass function p."""
    p = np.asarray(p, dtype=float).ravel()
    u = np.random.rand()
    index = 0
    s = p[0]
    while u > s and index < p.size - 1:
        index += 1
        s += p[index]
    return index


def markov_chain_simulation(T, ns, nsim):
    """SeReMpy MarkovChainSimulation: (ns, nsim) realisations, codes 0..n-1."""
    T = np.asarray(T, dtype=float)
    fsim = np.zeros((ns, nsim), dtype=int)
    fprior = matrix_power(T, 100)[0, :]
    for j in range(nsim):
        fsim[0, j] = rand_disc(fprior)
        for i in range(1, ns):
            fsim[i, j] = rand_disc(T[fsim[i - 1, j], :])
    return fsim


# --------------------------------------------------------------------------
# Lambda (continuous-time) extension used by the inversion
# --------------------------------------------------------------------------
def facies_prior(T, lam=None, prior="seremp"):
    """Initial-facies probabilities.

    "seremp"     : first row of T^100, exactly as SeReMpy.
    "stationary" : stationary distribution of T weighted by mean bed thickness
                   1/lambda, i.e. the expected time fraction of each facies.
                   Use this when T is periodic (e.g. the layer-3 matrix, whose
                   facies split into {1,4} and {2,3}), where row 0 of T^100
                   never converges and only facies 1 or 4 can start a layer.
    """
    T = np.asarray(T, dtype=float)
    if prior == "seremp":
        p = matrix_power(T, 100)[0, :]
    elif prior == "stationary":
        w, v = np.linalg.eig(T.T)
        p = np.abs(np.real(v[:, np.argmin(np.abs(w - 1.0))]))
        p = p / np.asarray(lam, dtype=float)
    else:
        raise ValueError(prior)
    return p / p.sum()


@njit(cache=True)
def _rand_disc(p):
    # same search as SeReMpy RandDisc
    u = np.random.random()
    index = 0
    s = p[0]
    while u > s and index < p.size - 1:
        index += 1
        s += p[index]
    return index


@njit(cache=True)
def _simulate_lambda(T, fprior, lam, ns, nsim, ds, seed, carry=False):
    # Each bed:
    #   thickness = -log(u)/lambda, u uniform on (0, 1)
    #   n = thickness/samp rounded to the nearest integer
    #   samples i..i+n are set, then i advances by n, so the next bed
    #   overwrites the last one: each bed occupies n samples and a bed with
    #   n == 0 vanishes (the chain still transitions from it).
    #   The next facies is drawn from the row of T of the current facies.
    # carry=True: the initial-facies probabilities start as the first row of T^100
    #   and are then overwritten with the T row of the current facies, so
    #   realisation j+1 starts from the T row of the facies of the second-to-last
    #   bed of realisation j instead of from T^100.
    np.random.seed(seed)
    fsim = np.zeros((ns, nsim), dtype=np.int8)
    p = fprior.copy()
    for j in range(nsim):
        f = _rand_disc(p if carry else fprior)
        i = 0
        while i < ns:
            thick = -np.log(np.random.random()) / lam[f]
            n = int(np.floor(thick / ds + 0.5))
            stop = min(i + n, ns)
            fsim[i:stop, j] = f
            if stop == ns:     # no transition after the last bed
                break
            i = stop
            p = T[f].copy()
            f = _rand_disc(T[f])
    return fsim


def markov_chain_simulation_lambda(T, ns, nsim, lam, ds, seed=0, prior="seremp", carry=False):
    """CTMC facies realisations (ns, nsim), codes 0..n-1.

    T   : transition probability matrix (zero diagonal => adjacent beds differ)
    lam : exponential bed-thickness rate per facies, in 1/(units of ds)
    carry : the initial-facies probabilities of each realisation are
            carried over from the previous one (see _simulate_lambda)
    """
    T = np.atleast_2d(np.asarray(T, dtype=np.float64))
    lam = np.atleast_1d(np.asarray(lam, dtype=np.float64))
    if lam.size != T.shape[0]:
        raise ValueError(f"lambda has {lam.size} values but T is {T.shape}")
    fprior = facies_prior(T, lam, prior)
    return _simulate_lambda(T, fprior, lam, int(ns), int(nsim), float(ds), int(seed), bool(carry))
