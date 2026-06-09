"""Spatiotemporal moment density field (spec §1).

Two equivalent backends behind one interface:

- ``GriddedField`` — density on a regular grid with a summed-area table for
  O(1) enclosed-moment queries (quadrature accuracy set by cell size).
- ``AnalyticField`` — mesh-free superposition over the event history using
  closed-form disk-overlap integrals (exact).

The field is signed: depletion applies no floor, and negative values are
strain deficits (spec §1.4). The supportability root-find for the local
maximum magnitude takes the *smallest* crossing (spec §1.5).
"""

from abc import ABC, abstractmethod


class MomentField(ABC):
    """Interface shared by the gridded and analytic field backends."""

    @abstractmethod
    def load(self, dt: float) -> None:
        """Advance tectonic loading by ``dt`` days (spec §1.2)."""

    @abstractmethod
    def deplete(self, x: float, y: float, magnitude: float) -> None:
        """Subtract M0(M)/A(M) uniformly over the rupture disk (spec §1.4)."""

    @abstractmethod
    def enclosed_moment(self, x: float, y: float, radius: float) -> float:
        """Signed moment integral over the disk of given radius (spec §1.5)."""

    @abstractmethod
    def local_mmax(self, x: float, y: float) -> float:
        """Smallest supportability crossing; -inf if locked (spec §1.5)."""


class GriddedField(MomentField):
    """Density grid + summed-area table backend."""


class AnalyticField(MomentField):
    """Mesh-free superposition backend (closed-form disk overlaps)."""
