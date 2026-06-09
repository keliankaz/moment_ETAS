"""Discrete truncated Gutenberg-Richter magnitude distribution (spec §2).

Magnitudes live on bins M_k = m_min + k*DM. The pmf is the discretized
exponential GR, truncated at the largest supportable bin k_max.
"""

import numpy as np

from ..params import DM


def gr_pmf(k_max: int, b: float) -> np.ndarray:
    """Pmf over bins k = 0..k_max: P(k) ∝ 10**(-b k DM)."""
    if k_max < 0:
        raise ValueError("k_max < 0: no supportable bin (locked location)")
    w = 10.0 ** (-b * DM * np.arange(k_max + 1))
    return w / w.sum()


def sample_magnitude(rng: np.random.Generator, m_min: float, k_max: int, b: float) -> float:
    """Draw one magnitude from the truncated discrete GR.

    Inverse-CDF sampling of the truncated geometric (one uniform draw, no
    allocation — this is the per-event hot path): with ρ = 10^(−b·DM),
    CDF(k) = (1 − ρ^(k+1)) / (1 − ρ^(k_max+1)).
    """
    if k_max < 0:
        raise ValueError("k_max < 0: no supportable bin (locked location)")
    rho = 10.0 ** (-b * DM)
    u = rng.random()
    k = int(np.log(1.0 - u * (1.0 - rho ** (k_max + 1))) / np.log(rho))
    return m_min + min(k, k_max) * DM
