"""Model parameters (spec §5). All times in days, densities in N·m/km²."""

from dataclasses import dataclass

#: Magnitude bin width (spec §2)
DM = 0.1


@dataclass
class Params:
    # Magnitude
    m_min: float = 3.0      # completeness / minimum magnitude (= Mc)
    b: float = 1.0          # GR b-value

    # Domain
    lx: float = 100.0       # domain extent, km
    ly: float = 100.0
    cell: float = 1.0       # grid cell size Δ, km

    # Field
    f0: float = 2.0e16      # initial moment density, N·m/km² (~ supports M6.5)
    mdot: float = 5.0e11    # loading rate, N·m/km²/day (~ recharge M6.5 in ~100 yr)
    a0: float = 0.1         # rupture area at m_min, km² (A ≈ 10^(M−4) km²)

    # ETAS
    mu0: float = 1.0e-6     # background rate, events/day/km²
    k: float = 0.05         # productivity amplitude
    alpha: float = 1.0      # productivity magnitude scaling
    c: float = 0.01         # Omori offset, days
    p: float = 1.2          # Omori exponent (> 1 strictly)
    tau_max: float = 36525.0  # aftershock delay cutoff, days (100 yr); None = untruncated
    d_km: float = 1.0       # triggering spatial scale at m_min, km
    gamma: float = 0.5      # triggering spatial magnitude scaling
    q: float = 1.8          # spatial power-law exponent (> 1 strictly)

    def __post_init__(self):
        if self.p <= 1:
            raise ValueError(f"Omori p must be > 1 strictly, got {self.p}")
        if self.q <= 1:
            raise ValueError(f"spatial q must be > 1 strictly, got {self.q}")

    @property
    def area(self) -> float:
        return self.lx * self.ly
