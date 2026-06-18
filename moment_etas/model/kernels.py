"""ETAS triggering kernels: productivity and inverse-CDF samplers (spec §3.3-3.5, §6).

The branching simulator samples offspring delays and displacements directly
from the normalized Omori-Utsu and power-law spatial kernels; no densities or
envelopes are needed for simulation. All times in days, distances in km.

Magnitude-dependent laws are anchored at the reference magnitude m_ref (the
spec's Mc) — the scaling anchor, not the simulation cutoff m_min (see Params).
"""

import numpy as np

from .rupture import rupture_radius


def productivity(m, k, alpha, m_ref):
    """Expected direct offspring count ν(M) = k 10^(α (M − m_ref))."""
    return k * 10.0 ** (alpha * (np.asarray(m) - m_ref))


def sample_omori(rng: np.random.Generator, n: int, c: float, p: float) -> np.ndarray:
    """Delays τ from g(τ) = (p−1)/c (1 + τ/c)^(−p) via inverse CDF."""
    u = rng.random(n)
    return c * (u ** (-1.0 / (p - 1.0)) - 1.0)


def spatial_scale(m, c_d, a0, m_ref):
    """Triggering scale coupled to rupture geometry: d(M) = c_d · R(M), km.

    The aftershock zone tracks the rupture radius (spec §3.5), so the plateau of
    the spatial kernel shares the rupture-area magnitude scaling exactly; c_d is
    its size relative to the rupture (c_d = 1 → d(M) = R(M)).
    """
    return c_d * rupture_radius(m, a0, m_ref)


def sample_displacement(
    rng: np.random.Generator, n: int, d: float, q: float
) -> tuple[np.ndarray, np.ndarray]:
    """Offspring displacements (dx, dy) from the isotropic power-law kernel.

    Radial inverse CDF of h(r) (2πr-weighted): r = d sqrt(u^(−1/(q−1)) − 1).
    """
    u = rng.random(n)
    r = d * np.sqrt(u ** (-1.0 / (q - 1.0)) - 1.0)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    return r * np.cos(theta), r * np.sin(theta)
