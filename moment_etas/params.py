"""Model parameters (spec §5). All times in days, densities in N·m/km²."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

#: Magnitude bin width (spec §2)
DM = 0.1

#: A static baseline field (F₀ or Ṁ_load): a scalar, a callable f(X, Y)
#: evaluated on cell centers, or a 2D array matching the grid (spec §1.1-1.2).
BaselineSpec = float | Callable | np.ndarray


@dataclass
class Params:
    # Magnitude
    m_min: float = 3.0      # simulation cutoff: GR lower bound, bin origin
    m_ref: float = 3.0      # reference magnitude Mc anchoring the scaling laws:
                            # a0, k, d_km are the values AT m_ref. Decoupled from
                            # m_min so raising the catalog floor does not silently
                            # recalibrate rupture areas, productivity, or d(M).
    b: float = 1.0          # GR b-value

    # Domain
    lx: float = 100.0       # domain extent, km
    ly: float = 100.0
    cell: float = 1.0       # grid cell size Δ, km

    # Field — f0 and mdot are scalar | callable f(X,Y) | (nx,ny) array (spec §1.1-1.2)
    f0: BaselineSpec = 2.0e16   # initial moment density, N·m/km² (~ supports M6.5)
    mdot: BaselineSpec = 5.0e11  # loading rate, N·m/km²/day (~ recharge M6.5 in ~100 yr)
    a0: float = 0.1         # rupture area at m_ref, km² (A ≈ 10^(M−4) km²)

    # ETAS
    mu0: float = 1.0e-6     # background rate, events/day/km²
    k: float = 0.05         # productivity amplitude at m_ref
    alpha: float = 1.0      # productivity magnitude scaling
    c: float = 0.01         # Omori offset, days
    p: float = 1.2          # Omori exponent (> 1 strictly)
    tau_max: float = 36525.0  # aftershock delay cutoff, days (100 yr); None = untruncated
    d_km: float = 1.0       # triggering spatial scale at m_ref, km
    gamma: float = 0.5      # triggering spatial magnitude scaling
    q: float = 1.8          # spatial power-law exponent (> 1 strictly)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Check parameter constraints.

        Called at construction and again by simulate_catalog, since fields are
        commonly mutated after construction (which skips __post_init__).
        """
        if self.p <= 1:
            raise ValueError(f"Omori p must be > 1 strictly, got {self.p}")
        if self.q <= 1:
            raise ValueError(f"spatial q must be > 1 strictly, got {self.q}")
        if self.b <= 0:
            raise ValueError(f"GR b must be positive, got {self.b}")
        if self.tau_max is not None and self.tau_max <= 0:
            raise ValueError(f"tau_max must be positive or None, got {self.tau_max}")
        for extent, name in ((self.lx, "lx"), (self.ly, "ly")):
            n = extent / self.cell
            if abs(n - round(n)) > 1e-9:
                raise ValueError(
                    f"cell={self.cell} must divide {name}={extent} evenly "
                    f"(grid extent would mismatch the domain)"
                )
        grid_shape = (round(self.lx / self.cell), round(self.ly / self.cell))
        for spec, name in ((self.f0, "f0"), (self.mdot, "mdot")):
            if isinstance(spec, np.ndarray) and spec.shape != grid_shape:
                raise ValueError(
                    f"{name} array shape {spec.shape} != grid {grid_shape} "
                    f"(nx, ny from lx/ly/cell)"
                )

    @property
    def area(self) -> float:
        return self.lx * self.ly
