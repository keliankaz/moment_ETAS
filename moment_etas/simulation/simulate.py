"""Chronological branching (cluster) simulation of the moment-bounded ETAS model (spec §6).

Background events seed a time-ordered priority queue; each popped event is
assigned a magnitude from the then-current field (or discarded if the location
is locked), depletes the field over its rupture disk, and pushes its offspring
sampled directly from the Omori and spatial kernels. No thinning envelope.
"""

import heapq
from dataclasses import dataclass

import numpy as np

from ..params import DM, Params
from ..model.kernels import productivity, sample_displacement, sample_omori, spatial_scale
from ..model.magnitude import sample_magnitude
from ..model.moment_field import GriddedField, MomentField


@dataclass
class Catalog:
    """Simulation result: event arrays plus bookkeeping."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    m: np.ndarray
    parent: np.ndarray          # index of triggering event, -1 for background
    n_locked: int               # queued events discarded at locked locations
    field: MomentField          # final field state
    params: Params

    def __len__(self) -> int:
        return len(self.t)


def simulate_catalog(
    params: Params,
    t_max: float,
    seed: int | None = None,
) -> Catalog:
    """Run the branching simulation for t_max days. Times in days, coords in km."""
    rng = np.random.default_rng(seed)
    fld = GriddedField(params)

    # heap entries: (time, sequence, x, y, parent_index)
    heap: list[tuple[float, int, float, float, int]] = []
    seq = 0

    n_bg = rng.poisson(params.mu0 * params.area * t_max)
    for t in rng.uniform(0.0, t_max, n_bg):
        x = rng.uniform(0.0, params.lx)
        y = rng.uniform(0.0, params.ly)
        heapq.heappush(heap, (t, seq, x, y, -1))
        seq += 1

    ts, xs, ys, ms, parents = [], [], [], [], []
    n_locked = 0

    while heap:
        t, _, x, y, parent = heapq.heappop(heap)

        k_max = fld.local_kmax(x, y, t)
        if k_max < 0:
            n_locked += 1
            continue

        m = sample_magnitude(rng, params.m_min, k_max, params.b)
        fld.deplete(x, y, m)

        idx = len(ts)
        ts.append(t)
        xs.append(x)
        ys.append(y)
        ms.append(m)
        parents.append(parent)

        n_off = rng.poisson(productivity(m, params.k, params.alpha, params.m_min))
        if n_off > 0:
            tau = sample_omori(rng, n_off, params.c, params.p)
            d = spatial_scale(m, params.d_km, params.gamma, params.m_min)
            dx, dy = sample_displacement(rng, n_off, d, params.q)
            for j in range(n_off):
                tc, xc, yc = t + tau[j], x + dx[j], y + dy[j]
                if tc <= t_max and 0.0 <= xc <= params.lx and 0.0 <= yc <= params.ly:
                    heapq.heappush(heap, (tc, seq, xc, yc, idx))
                    seq += 1

    return Catalog(
        t=np.array(ts),
        x=np.array(xs),
        y=np.array(ys),
        m=np.array(ms),
        parent=np.array(parents, dtype=int),
        n_locked=n_locked,
        field=fld,
        params=params,
    )
