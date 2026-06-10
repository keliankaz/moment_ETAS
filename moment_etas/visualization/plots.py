"""Plotting helpers (spec §8): field maps, Mmax maps, space-time plots, GR, sawtooth."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, LogNorm

from ..params import DM


def magnitude_time(cat, ax=None):
    """Magnitude vs time, background vs triggered distinguished."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    yrs = cat.t / 365.25
    bg = cat.parent == -1
    ax.vlines(yrs, cat.params.m_min, cat.m, lw=0.3, color="0.7", zorder=1)
    ax.scatter(yrs[~bg], cat.m[~bg], s=8, c="tab:orange", label="triggered", zorder=2)
    ax.scatter(yrs[bg], cat.m[bg], s=8, c="tab:blue", label="background", zorder=2)
    ax.set(xlabel="time (yr)", ylabel="magnitude", title="catalog")
    ax.legend(loc="upper left", fontsize=8)
    return ax


def space_time(cat, ax=None):
    """x-coordinate vs time, marker size by magnitude (fault-diagram style)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    s = 2.0 * 2.0 ** (cat.m - cat.params.m_min)
    sc = ax.scatter(cat.t / 365.25, cat.x, s=s, c=cat.m, cmap="viridis", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="magnitude")
    ax.set(xlabel="time (yr)", ylabel="x (km)", title="space-time")
    return ax


def epicenter_map(cat, ax=None):
    """Map view of epicenters, size by magnitude."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    s = 2.0 * 2.0 ** (cat.m - cat.params.m_min)
    sc = ax.scatter(cat.x, cat.y, s=s, c=cat.t / 365.25, cmap="plasma", alpha=0.7)
    plt.colorbar(sc, ax=ax, label="time (yr)")
    ax.set(xlabel="x (km)", ylabel="y (km)", title="epicenters",
           xlim=(0, cat.params.lx), ylim=(0, cat.params.ly), aspect="equal")
    return ax


def epicenter_density(cat, bin_km=2.0, m_min=None, ax=None):
    """Heat map of epicentral locations (counts per bin, log color scale).

    Useful when the catalog is too dense for a scatter. Optional m_min
    restricts to events at or above that magnitude.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    p = cat.params
    sel = slice(None) if m_min is None else cat.m >= m_min
    h, _, _ = np.histogram2d(
        cat.x[sel], cat.y[sel],
        bins=(np.arange(0, p.lx + bin_km, bin_km), np.arange(0, p.ly + bin_km, bin_km)),
    )
    im = ax.imshow(np.where(h > 0, h, np.nan).T, origin="lower",
                   extent=(0, p.lx, 0, p.ly), cmap="inferno",
                   norm=LogNorm(vmin=1, vmax=max(h.max(), 1)))
    plt.colorbar(im, ax=ax, label=f"events / {bin_km:g}×{bin_km:g} km²")
    title = "epicentral density" if m_min is None else f"epicentral density (M ≥ {m_min:g})"
    ax.set(xlabel="x (km)", ylabel="y (km)", title=title,
           xlim=(0, p.lx), ylim=(0, p.ly), aspect="equal")
    return ax


def magnitude_distribution(cat, ax=None):
    """Incremental and cumulative magnitude-frequency with the input GR slope."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    p = cat.params
    bins = np.arange(p.m_min, cat.m.max() + DM, DM)
    counts = np.array([(np.abs(cat.m - mb) < DM / 2).sum() for mb in bins])
    cum = np.array([(cat.m >= mb - DM / 2).sum() for mb in bins])
    ax.semilogy(bins, cum, "s-", ms=4, label="cumulative")
    ax.semilogy(bins[counts > 0], counts[counts > 0], "o", ms=4, label="incremental")
    ref = cum[0] * 10.0 ** (-p.b * (bins - p.m_min))
    ax.semilogy(bins, ref, "k--", lw=1, label=f"GR slope b={p.b}")
    ax.set(xlabel="magnitude", ylabel="count", title="magnitude-frequency")
    ax.legend(fontsize=8)
    return ax


def field_map(cat, t=None, ax=None):
    """Map of the moment density field F(x, y, t); defaults to the final state."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    p = cat.params
    if t is None:
        t = cat.t.max() if len(cat) else 0.0
    f = cat.field.field(t)
    im = ax.imshow(f.T, origin="lower", extent=(0, p.lx, 0, p.ly), cmap="RdBu",
                   norm=CenteredNorm(vcenter=0.0))
    plt.colorbar(im, ax=ax, label="F (N·m/km²)")
    ax.set(xlabel="x (km)", ylabel="y (km)", title=f"moment density, t = {t/365.25:.1f} yr")
    return ax


def mmax_map(cat, t=None, coarse=4, ax=None):
    """Map of local Mmax via the supportability bin scan on a coarsened grid."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    p = cat.params
    if t is None:
        t = cat.t.max() if len(cat) else 0.0
    xs = np.arange(coarse * p.cell / 2, p.lx, coarse * p.cell)
    ys = np.arange(coarse * p.cell / 2, p.ly, coarse * p.cell)
    mm = np.full((len(xs), len(ys)), np.nan)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            k = cat.field.local_kmax(x, y, t)
            if k >= 0:
                mm[i, j] = p.m_min + k * DM
    im = ax.imshow(mm.T, origin="lower", extent=(0, p.lx, 0, p.ly), cmap="magma")
    plt.colorbar(im, ax=ax, label="local Mmax")
    ax.set(xlabel="x (km)", ylabel="y (km)", title=f"supportable Mmax, t = {t/365.25:.1f} yr")
    return ax


def field_sawtooth(cat, x, y, n_t=2000, ax=None):
    """F at a fixed point vs time, reconstructed from the event history.

    Uses the same discretized-disk normalization as GriddedField.deplete, so
    the trace matches the simulation's field exactly at the cell containing
    (x, y).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    p = cat.params
    fld = cat.field
    i, j = fld.cell_index(x, y)

    drops = []  # (time, depletion level at this cell)
    for te, xe, ye, me in zip(cat.t, cat.x, cat.y, cat.m):
        lvl = fld.drop_level_at(i, j, xe, ye, me)
        if lvl > 0.0:
            drops.append((te, lvl))

    t_end = cat.t.max() if len(cat) else 1.0
    tt = np.linspace(0.0, t_end, n_t)
    f = p.f0 + p.mdot * tt
    for te, lvl in drops:
        f[tt >= te] -= lvl
    ax.plot(tt / 365.25, f, lw=1)
    ax.axhline(0.0, color="k", lw=0.5, ls=":")
    ax.set(xlabel="time (yr)", ylabel="F (N·m/km²)",
           title=f"field at ({x:.0f}, {y:.0f}) km — {len(drops)} covering events")
    return ax


def field_animation(cat, path="field_evolution.gif", t_start=None, t_end=None,
                    n_frames=80, fps=10, cmap="RdBu"):
    """Write a GIF of the moment field F(x, y, t) over [t_start, t_end].

    The field is reconstructed by replaying the catalog's depletions into a
    fresh grid (exact — same rules as the simulation), so no snapshots are
    needed and any time window works. Defaults to the last 20% of the run.
    Times in days; the duration covered and frame count set the GIF size, so
    keep n_frames modest for fine grids.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    from ..model.moment_field import GriddedField

    p = cat.params
    if t_end is None:
        t_end = float(cat.t.max()) if len(cat) else 0.0
    if t_start is None:
        t_start = 0.8 * t_end
    frame_times = np.linspace(t_start, t_end, n_frames)

    # replay depletions: pre-roll everything before t_start, then advance
    # the event pointer incrementally between frames
    fld = GriddedField(p)
    order = np.argsort(cat.t, kind="stable")
    te, xe, ye, me = cat.t[order], cat.x[order], cat.y[order], cat.m[order]
    ptr = 0
    frames = np.empty((n_frames, fld.nx, fld.ny), dtype=np.float32)
    for f, tf in enumerate(frame_times):
        while ptr < len(te) and te[ptr] <= tf:
            fld.deplete(xe[ptr], ye[ptr], me[ptr])
            ptr += 1
        frames[f] = fld.field(tf)

    vmax = max(abs(float(frames.min())), abs(float(frames.max())))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames[0].T, origin="lower", extent=(0, p.lx, 0, p.ly),
                   cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="F (N·m/km²)")
    ax.set(xlabel="x (km)", ylabel="y (km)")
    title = ax.set_title("")

    def draw(f):
        im.set_data(frames[f].T)
        title.set_text(f"moment density, t = {frame_times[f]/365.25:.1f} yr")
        return im, title

    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def overview(cat):
    """Six-panel summary figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.3))
    magnitude_time(cat, fig.add_subplot(gs[0, :]))
    space_time(cat, fig.add_subplot(gs[1, :]))
    magnitude_distribution(cat, fig.add_subplot(gs[2, 0]))
    if len(cat):
        i_big = int(np.argmax(cat.m))
        field_sawtooth(cat, cat.x[i_big], cat.y[i_big], ax=fig.add_subplot(gs[2, 1]))
    fig.tight_layout()
    return fig
