"""Characterization test: pins current default behavior as a regression net.

Not a correctness proof — it locks the present output so behavior-preserving
refactors (and the spatial-loading / moment-limited-Mmax work) can be verified
not to change the default scalar-loading model.

Runnable two ways:
    python tests/test_characterization.py        # standalone, plain asserts
    pytest tests/test_characterization.py        # if pytest is installed
"""

import sys
from pathlib import Path

import numpy as np

# Allow running standalone (python tests/...) without the package installed;
# under the conda env the editable install makes this a no-op.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moment_etas import Params, simulate_catalog

# Golden values: Params() defaults, seed 42, 10-yr (3650-day) run.
GOLDEN_N_EVENTS = 51
GOLDEN_DEPLETION_SUM = 1.729205396375566e16  # N·m / km², summed over the grid
GOLDEN_M_MAX = 4.4


def _default_catalog():
    return simulate_catalog(Params(), t_max=3650.0, seed=42)


def test_default_run_is_unchanged():
    """Default seeded run reproduces the pinned catalog and field."""
    cat = _default_catalog()
    assert len(cat) == GOLDEN_N_EVENTS, f"event count drifted: {len(cat)}"
    assert np.isclose(
        cat.field.depletion.sum(), GOLDEN_DEPLETION_SUM, rtol=1e-12
    ), f"depletion sum drifted: {cat.field.depletion.sum()!r}"
    assert np.isclose(cat.m.max(), GOLDEN_M_MAX), f"max magnitude drifted: {cat.m.max()}"


if __name__ == "__main__":
    test_default_run_is_unchanged()
    print("OK: default run matches golden "
          f"(n={GOLDEN_N_EVENTS}, sum={GOLDEN_DEPLETION_SUM:.6e}, Mmax={GOLDEN_M_MAX})")
