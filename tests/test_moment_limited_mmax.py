"""Moment-limited Mmax: the ceiling is set by accumulated moment, not footprint.

    python tests/test_moment_limited_mmax.py
    pytest tests/test_moment_limited_mmax.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moment_etas import Params
from moment_etas.model.moment_field import GriddedField
from moment_etas.model.rupture import moment, rupture_radius
from moment_etas.params import DM


def _brute_kmax(fld, x, y, t, p, m_top=10.0):
    """Reference smallest-crossing scan via the exact enclosed_moment (clips to
    domain), independent of the field's internal _k_hard."""
    k, kk = -1, 0
    while p.m_min + kk * DM <= m_top:
        m = p.m_min + kk * DM
        if fld.enclosed_moment(x, y, rupture_radius(m, p.a0, p.m_ref), t) < moment(m):
            break
        k, kk = kk, kk + 1
    return k


def test_mmax_matches_brute_force_across_charge_levels():
    """local_kmax tracks the exact enclosed-moment scan from dormant to charged."""
    for f0 in (1e17, 1e18, 5e18):
        fld = GriddedField(Params(f0=f0))
        k = fld.local_kmax(50.0, 50.0, 0.0)
        assert abs(k - _brute_kmax(fld, 50.0, 50.0, 0.0, fld.p)) <= 1


def test_charged_domain_exceeds_footprint_cap():
    """A strongly charged domain supports events beyond the domain-footprint
    magnitude, via the moment ceiling (else-branch of the scan)."""
    p = Params(f0=5e18)
    fld = GriddedField(p)
    k = fld.local_kmax(50.0, 50.0, 0.0)
    assert k > fld._k_hard, "moment ceiling should exceed the geometric scan extent"
    assert p.m_min + k * DM > 8.5, "should exceed the old max(lx,ly) footprint cap"


if __name__ == "__main__":
    test_mmax_matches_brute_force_across_charge_levels()
    test_charged_domain_exceeds_footprint_cap()
    print("OK: Mmax is moment-limited and matches the exact enclosed-moment scan")
