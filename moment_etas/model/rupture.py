"""Rupture geometry and moment conversion (spec §1.1, §1.3).

Hanks-Kanamori in SI units: M0 = 10**(1.5 M + 9.05) N·m.
Self-similar rupture-area scaling anchored at the reference magnitude m_ref
(the spec's Mc): A(M) = a0 * 10**(M - m_ref), R = sqrt(A/pi). Note m_ref is
the scaling anchor, not the simulation cutoff m_min — see Params.
"""

import numpy as np


def moment(m):
    """Seismic moment M0 (N·m) for magnitude(s) m."""
    return 10.0 ** (1.5 * np.asarray(m) + 9.05)


def magnitude_from_moment(m0):
    """Inverse of moment(): the magnitude whose seismic moment is m0 (N·m)."""
    return (np.log10(m0) - 9.05) / 1.5


def rupture_area(m, a0, m_ref):
    """Rupture area A(M) in km²; a0 is the area at m == m_ref."""
    return a0 * 10.0 ** (np.asarray(m) - m_ref)


def rupture_radius(m, a0, m_ref):
    """Equivalent circular rupture radius R(M) in km."""
    return np.sqrt(rupture_area(m, a0, m_ref) / np.pi)
