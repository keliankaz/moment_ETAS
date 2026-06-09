"""Rupture geometry and moment conversion (spec §1.1, §1.3).

Hanks-Kanamori in SI units: M0 = 10**(1.5 M + 9.05) N·m.
Self-similar rupture-area scaling: A(M) = A0 * 10**(M - m_min), R = sqrt(A/pi).
"""

import numpy as np


def moment(m):
    """Seismic moment M0 (N·m) for magnitude(s) m."""
    return 10.0 ** (1.5 * np.asarray(m) + 9.05)


def rupture_area(m, a0, m_min):
    """Rupture area A(M) in km²."""
    return a0 * 10.0 ** (np.asarray(m) - m_min)


def rupture_radius(m, a0, m_min):
    """Equivalent circular rupture radius R(M) in km."""
    return np.sqrt(rupture_area(m, a0, m_min) / np.pi)


def stress_density(m, a0, m_min):
    """Depletion level M0(M)/A(M): moment density removed over the disk, N·m/km²."""
    return moment(m) / rupture_area(m, a0, m_min)
