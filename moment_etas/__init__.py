"""Moment-bounded spatio-temporal ETAS model.

Standard ETAS triggering plus a signed seismic-moment density field that bounds
the local maximum magnitude via rupture-disk supportability (see spec.md).

Time convention: all times in days. All field quantities are moment densities
in N·m / km².
"""

from .params import DM, Params
from .simulation.simulate import Catalog, simulate_catalog

__version__ = "0.1.0"

__all__ = ["DM", "Params", "Catalog", "simulate_catalog"]
