"""Spatial F₀ / Ṁ_load: equivalence on uniform input + gradient direction.

    python tests/test_spatial_baseline.py     # standalone
    pytest tests/test_spatial_baseline.py     # if pytest installed
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moment_etas import Params, simulate_catalog
from moment_etas.model.moment_field import GriddedField
from moment_etas.params import DM


def test_uniform_callable_and_array_match_scalar():
    """A constant callable and constant array reproduce the scalar catalog."""
    nx, ny = 100, 100
    p_callable = Params(
        f0=lambda X, Y: np.full_like(X, 2.0e16),
        mdot=lambda X, Y: np.full_like(X, 5.0e11),
    )
    p_array = Params(f0=np.full((nx, ny), 2.0e16), mdot=np.full((nx, ny), 5.0e11))

    base = simulate_catalog(Params(), t_max=3650.0, seed=42)        # scalar
    via_callable = simulate_catalog(p_callable, t_max=3650.0, seed=42)
    via_array = simulate_catalog(p_array, t_max=3650.0, seed=42)

    # the spatial code path (non-uniform flag) must reproduce the scalar result
    assert not GriddedField(p_callable)._uniform
    for other in (via_callable, via_array):
        assert len(other) == len(base)
        assert np.array_equal(other.m, base.m)
        assert np.allclose(other.field.depletion, base.field.depletion, rtol=1e-12)


def test_higher_initial_field_supports_higher_mmax():
    """Instantaneous Mmax increases with the local baseline field (t=0, no depletion)."""
    lx = 100.0
    p = Params(f0=lambda X, Y: 2.0e16 * (0.4 + 1.2 * X / lx))
    fld = GriddedField(p)
    k_low = fld.local_kmax(10.0, 50.0, 0.0)    # low-F₀ side
    k_high = fld.local_kmax(90.0, 50.0, 0.0)   # high-F₀ side
    assert k_high > k_low, (
        f"Mmax should rise with F₀: low M{p.m_min + k_low*DM:.1f} "
        f"vs high M{p.m_min + k_high*DM:.1f}"
    )


if __name__ == "__main__":
    test_uniform_callable_and_array_match_scalar()
    test_higher_initial_field_supports_higher_mmax()
    print("OK: uniform callable/array match scalar; Mmax rises with the baseline field")
