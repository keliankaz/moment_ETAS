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
from .rupture import magnitude_from_moment, moment, rupture_radius


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

        # Static baseline fields (spec §1.1-1.2): resolve scalar | callable |
        # array to a scalar (uniform) or an (nx, ny) grid. All reads go through
        # self.f0/self.mdot, never params.f0/params.mdot.
        self.f0, f0_uniform = self._resolve_baseline(params.f0)
        self.mdot, mdot_uniform = self._resolve_baseline(params.mdot)
        self._uniform = f0_uniform and mdot_uniform
        # domain cell-sums of the baselines (for the total-moment ceiling, §1.5):
        # total domain moment at t = (f0_domain + t*mdot_domain - depletion.sum()) * cell_area
        ncells = self.nx * self.ny
        self._f0_domain = float(self.f0.sum()) if not f0_uniform else self.f0 * ncells
        self._mdot_domain = float(self.mdot.sum()) if not mdot_uniform else self.mdot * ncells
        # Extent of the geometric scan: the domain diagonal, so a disk of this
        # radius covers the whole domain from ANY epicenter (even a corner).
        # Beyond it the enclosed moment is frozen at the domain total and Mmax
        # is moment-limited (handled analytically in local_kmax, spec §1.5).
        # _k_hard is a geometric/memory bound now, not a correctness cap.
        r_domain = float(np.hypot(params.lx, params.ly))
        m_hard = params.m_min
        while rupture_radius(m_hard + DM, params.a0, params.m_ref) < r_domain:
            m_hard += DM
        self._k_hard = int(round((m_hard - params.m_min) / DM))

        # precomputed tables for the supportability scan (one-time cost):
        # per-bin radius, required moment, and the annulus of integer cell
        # offsets each bin adds to the previous bin's disk (centered on the
        # epicenter's host cell — a sub-cell approximation within quadrature
        # accuracy; deplete() keeps exact epicenter geometry)
        ks = np.arange(self._k_hard + 1)
        mags = params.m_min + ks * DM
        self._scan_radius = rupture_radius(mags, params.a0, params.m_ref)
        self._scan_m0 = moment(mags)
        self._scan_annuli = []
        r_max = int(np.ceil(self._scan_radius[-1] / params.cell))
        oi = np.arange(-r_max, r_max + 1)
        dist2 = (oi[:, None] ** 2 + oi[None, :] ** 2) * params.cell**2
        prev = np.zeros_like(dist2, dtype=bool)
        for r in self._scan_radius:
            disk = dist2 <= r**2
            ann = disk & ~prev
            ii, jj = np.nonzero(ann)
            self._scan_annuli.append((oi[ii], oi[jj]))
            prev = disk

    def _resolve_baseline(self, spec):
        """Resolve a BaselineSpec to (value, is_uniform).

        Scalar → passthrough (uniform); callable → evaluated on the cell-center
        meshgrid; (nx, ny) array → used as-is.
        """
        if callable(spec):
            xx, yy = np.meshgrid(self._xc, self._yc, indexing="ij")
            return np.asarray(spec(xx, yy), dtype=float), False
        arr = np.asarray(spec, dtype=float)
        if arr.ndim == 0:
            return float(arr), True
        return arr, False

    def field(self, t: float) -> np.ndarray:
        """Full field F(x, y, t) on the grid (N·m/km²)."""
        return self.f0 + self.mdot * t - self.depletion

    def baseline_at(self, i: int, j: int, t):
        """Loading baseline F₀ + Ṁ_load·t at cell (i, j); t may be an array.

        Single accessor for the local baseline so scalar and (Stage 1) grid
        storage are handled in one place.
        """
        f0 = self.f0[i, j] if np.ndim(self.f0) else self.f0
        md = self.mdot[i, j] if np.ndim(self.mdot) else self.mdot
        return f0 + md * t

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
        r = rupture_radius(m, self.p.a0, self.p.m_ref)
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
        r = rupture_radius(m, self.p.a0, self.p.m_ref)
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
            f = self.baseline_at(i, j, t) - self.depletion[i, j]
            return f * np.pi * radius**2
        dep = self.depletion[i0:i1, j0:j1][mask]
        f0 = self.f0[i0:i1, j0:j1][mask] if np.ndim(self.f0) else self.f0
        md = self.mdot[i0:i1, j0:j1][mask] if np.ndim(self.mdot) else self.mdot
        f = f0 + md * t - dep
        return f.sum() * self.cell_area

    def local_kmax(self, x: float, y: float, t: float) -> int:
        """First-failure scan up the magnitude bins (smallest crossing, spec §1.5).

        Incremental over precomputed per-bin annuli: each bin adds only its new
        cells to running (in-domain cell count, depletion) sums, so the whole
        scan costs one pass over the largest disk instead of one full disk sum
        per bin.
        """
        ci, cj = self.cell_index(x, y)
        uniform = self._uniform
        if uniform:
            f_load = self.f0 + self.mdot * t   # scalar; fast path
        else:
            f0_sum = 0.0
            mdot_sum = 0.0
        n_in = 0
        dep_sum = 0.0
        k = -1
        for kk in range(self._k_hard + 1):
            di, dj = self._scan_annuli[kk]
            if len(di):
                ii = ci + di
                jj = cj + dj
                valid = (ii >= 0) & (ii < self.nx) & (jj >= 0) & (jj < self.ny)
                vi, vj = ii[valid], jj[valid]
                n_in += vi.size
                dep_sum += float(self.depletion[vi, vj].sum())
                if not uniform:
                    f0_sum += float(self.f0[vi, vj].sum()) if np.ndim(self.f0) \
                        else self.f0 * vi.size
                    mdot_sum += float(self.mdot[vi, vj].sum()) if np.ndim(self.mdot) \
                        else self.mdot * vi.size
            if n_in == 0:
                # disk smaller than a cell: host-cell density times disk area
                f = self.baseline_at(ci, cj, t) - self.depletion[ci, cj]
                enclosed = f * np.pi * self._scan_radius[kk] ** 2
            else:
                base_sum = f_load * n_in if uniform else f0_sum + t * mdot_sum
                enclosed = (base_sum - dep_sum) * self.cell_area
            if enclosed < self._scan_m0[kk]:
                break
            k = kk
        else:
            # Still supportable at full-domain coverage: the enclosed moment is
            # frozen at the domain total, so Mmax is moment-limited, not
            # footprint-limited (spec §1.5). Take the largest bin the total
            # accumulated moment can release.
            m_total = (self._f0_domain + t * self._mdot_domain
                       - float(self.depletion.sum())) * self.cell_area
            if m_total > 0.0:
                m_ceiling = magnitude_from_moment(m_total)
                k = max(k, int(np.floor((m_ceiling - self.p.m_min) / DM + 1e-9)))
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
