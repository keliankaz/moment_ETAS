"""Spatiotemporal moment density field (spec §1).

The field is signed and linear in the event history:

    F(x, y, t) = F0 + mdot * t - D(x, y)

where D is the accumulated depletion density. Loading is uniform and
deterministic, so only D is stored ("lazy loading", spec §6 Notes).

``GriddedField`` evaluates disk integrals by direct masked sums over cells
(spec §8 build note: no summed-area table until profiling justifies one).
``AnalyticField`` (closed-form disk overlaps) is deferred; it will share this
interface and serve as the cross-validation backend.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..params import DM, Params
from .rupture import moment, rupture_radius


class MomentField(ABC):
    """Interface shared by the gridded and analytic field backends."""

    @abstractmethod
    def deplete(self, x: float, y: float, m: float) -> None:
        """Subtract M0(M)/A(M) uniformly over the in-domain rupture disk (spec §1.4)."""

    @abstractmethod
    def enclosed_moment(self, x: float, y: float, radius: float, t: float) -> float:
        """Signed moment integral of F over the disk (N·m), in-domain part only (spec §1.5)."""

    @abstractmethod
    def local_kmax(self, x: float, y: float, t: float) -> int:
        """Largest supportable magnitude bin index k_max; -1 if locked (spec §1.5, §2)."""


class GriddedField(MomentField):
    def __init__(self, params: Params):
        self.p = params
        self.nx = int(round(params.lx / params.cell))
        self.ny = int(round(params.ly / params.cell))
        self.cell_area = params.cell**2
        self.depletion = np.zeros((self.nx, self.ny))
        # cell-center coordinates
        self._xc = (np.arange(self.nx) + 0.5) * params.cell
        self._yc = (np.arange(self.ny) + 0.5) * params.cell
        # hard stop for the supportability scan: disk would swallow the domain
        r_domain = max(params.lx, params.ly)
        m_hard = params.m_min
        while rupture_radius(m_hard + DM, params.a0, params.m_min) < r_domain:
            m_hard += DM
        self._k_hard = int(round((m_hard - params.m_min) / DM))

    def field(self, t: float) -> np.ndarray:
        """Full field F(x, y, t) on the grid (N·m/km²)."""
        return self.p.f0 + self.p.mdot * t - self.depletion

    def cell_index(self, x: float, y: float) -> tuple[int, int]:
        """Grid indices of the cell containing (x, y), clamped to the domain."""
        i = min(max(int(x / self.p.cell), 0), self.nx - 1)
        j = min(max(int(y / self.p.cell), 0), self.ny - 1)
        return i, j

    def _disk_mask(self, x: float, y: float, radius: float):
        """In-domain disk cells plus the unclipped cell count.

        Returns (i0, i1, j0, j1, mask, n_unclipped) where the mask selects
        in-domain cells whose centers fall in the disk, and n_unclipped counts
        all such cell centers ignoring the domain boundary (for conservative
        normalization: the clipped share of an event's moment is lost, §7).
        """
        c = self.p.cell
        i0f = int(np.floor((x - radius) / c))
        i1f = int(np.floor((x + radius) / c)) + 1
        j0f = int(np.floor((y - radius) / c))
        j1f = int(np.floor((y + radius) / c)) + 1
        dxf = (np.arange(i0f, i1f) + 0.5) * c - x
        dyf = (np.arange(j0f, j1f) + 0.5) * c - y
        full = dxf[:, None] ** 2 + dyf[None, :] ** 2 <= radius**2
        n_unclipped = int(full.sum())

        i0, i1 = max(i0f, 0), min(i1f, self.nx)
        j0, j1 = max(j0f, 0), min(j1f, self.ny)
        mask = full[i0 - i0f : i1 - i0f, j0 - j0f : j1 - j0f]
        return i0, i1, j0, j1, mask, n_unclipped

    def deplete(self, x: float, y: float, m: float) -> None:
        """Remove the event's moment over its discretized disk (exactly conserving).

        The level is M0 / (n_unclipped * cell_area), not M0/A(M): normalizing by
        the discretized disk makes the removed moment exactly M0 times the
        in-domain cell fraction, eliminating discretization error in the moment
        budget. Sub-cell disks deposit their full moment into the host cell.
        """
        r = rupture_radius(m, self.p.a0, self.p.m_min)
        i0, i1, j0, j1, mask, n_unclipped = self._disk_mask(x, y, r)
        if n_unclipped == 0:
            i, j = self.cell_index(x, y)
            self.depletion[i, j] += moment(m) / self.cell_area
            return
        level = moment(m) / (n_unclipped * self.cell_area)
        self.depletion[i0:i1, j0:j1][mask] += level

    def drop_level_at(self, i: int, j: int, x: float, y: float, m: float) -> float:
        """Depletion level an event at (x, y) with magnitude m applies to cell (i, j).

        Mirrors `deplete` exactly (same mask and normalization rules) for
        per-cell reconstruction of the field history; returns 0.0 if the cell
        is not covered. Keep the two methods in sync.
        """
        r = rupture_radius(m, self.p.a0, self.p.m_min)
        i0, i1, j0, j1, mask, n_unclipped = self._disk_mask(x, y, r)
        if n_unclipped == 0:
            return moment(m) / self.cell_area if (i, j) == self.cell_index(x, y) else 0.0
        if i0 <= i < i1 and j0 <= j < j1 and mask[i - i0, j - j0]:
            return moment(m) / (n_unclipped * self.cell_area)
        return 0.0

    def enclosed_moment(self, x: float, y: float, radius: float, t: float) -> float:
        i0, i1, j0, j1, mask, _ = self._disk_mask(x, y, radius)
        if not mask.any():
            # disk smaller than a cell: use the host cell's density times disk area
            i, j = self.cell_index(x, y)
            f = self.p.f0 + self.p.mdot * t - self.depletion[i, j]
            return f * np.pi * radius**2
        dep = self.depletion[i0:i1, j0:j1][mask]
        f = self.p.f0 + self.p.mdot * t - dep
        return f.sum() * self.cell_area

    def local_kmax(self, x: float, y: float, t: float) -> int:
        """First-failure scan up the magnitude bins (smallest crossing, spec §1.5)."""
        p = self.p
        k = -1
        while k < self._k_hard:
            m = p.m_min + (k + 1) * DM
            r = rupture_radius(m, p.a0, p.m_min)
            if self.enclosed_moment(x, y, r, t) < moment(m):
                break
            k += 1
        return k


class AnalyticField(MomentField):
    """Mesh-free superposition backend (closed-form disk overlaps). Not yet implemented."""

    def __init__(self, params: Params):
        raise NotImplementedError("deferred: GriddedField first (spec §8 build order)")

    def deplete(self, x, y, m):  # pragma: no cover
        raise NotImplementedError

    def enclosed_moment(self, x, y, radius, t):  # pragma: no cover
        raise NotImplementedError

    def local_kmax(self, x, y, t):  # pragma: no cover
        raise NotImplementedError
