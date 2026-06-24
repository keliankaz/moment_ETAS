"""Visualize a single ETAS cluster as a branching process.

A cluster is an index array from ``Catalog.clusters()`` — a background root
plus its triggered descendants. Tree edges are ``(parent[i], i)`` for every
non-root member. All views take the catalog and one such member array.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .plots import marker_size

YEAR = 365.25


def _clean(ax):
    """Strip the axis apparatus (frame, ticks, tick labels, grid) for a clean
    network-diagram look (à la complexity-explorables). Title and colorbar,
    which carry meaning, are kept."""
    ax.axis("off")


def _tree(cat, members):
    """Return (root, children dict, generation dict) for the cluster."""
    mset = set(int(i) for i in members)
    children = {int(i): [] for i in members}
    root = int(members[0])
    for i in members:
        p = int(cat.parent[i])
        if p < 0:
            root = int(i)
        elif p in mset:
            children[p].append(int(i))
    gen = {root: 0}
    # members are chronological, so a parent's generation is set before its child
    for i in members:
        p = int(cat.parent[i])
        if p >= 0 and p in mset:
            gen[int(i)] = gen[p] + 1
    return root, children, gen


def _tidy_y(root, children):
    """Tidy-tree row coordinate per node: leaves get successive integers,
    each parent is centered on its children (iterative post-order)."""
    y, leaf = {}, [0]
    stack = [(root, False)]
    while stack:
        node, done = stack.pop()
        if done:
            kids = children[node]
            if kids:
                y[node] = float(np.mean([y[c] for c in kids]))
            else:
                y[node] = float(leaf[0])
                leaf[0] += 1
        else:
            stack.append((node, True))
            for c in reversed(children[node]):
                stack.append((c, False))
    return y


def cluster_tree(cat, members, ax=None):
    """Genealogy tree of the cluster (the family tree).

    x = generation depth (left→right cascade), y = tidy-tree layout. Nodes
    sized by magnitude and colored by time since the root — so structure (x +
    edges), size, and timing are all visible. (Time is *not* used as the x-axis
    because Omori bunches the immediate aftershocks at t≈0.)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    root, children, gen = _tree(cat, members)
    y = _tidy_y(root, children)
    mi = np.array(list(y))
    dt = (cat.t[mi] - cat.t[root]) / YEAR

    segs = [[(gen[int(cat.parent[i])], y[int(cat.parent[i])]), (gen[i], y[i])]
            for i in members if int(cat.parent[i]) in y]
    ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.6, zorder=1))

    sc = ax.scatter([gen[i] for i in mi], [y[i] for i in mi],
                    s=marker_size(cat.m[mi], cat.params.m_min),
                    c=dt, cmap="plasma", zorder=2)
    ax.scatter([gen[root]], [y[root]], s=marker_size(cat.m[root], cat.params.m_min),
               c="red", edgecolor="k", zorder=3, label=f"root M{cat.m[root]:.1f}")
    plt.colorbar(sc, ax=ax, label="time since root (yr)")
    ax.set_title(f"cluster genealogy — {len(members)} events, {max(gen.values())} generations")
    _clean(ax)
    return ax


def cluster_tree_radial(cat, members, ax=None):
    """Radial genealogy tree: angle = tidy-tree layout (spread over ±180°),
    radius = time since the root (years). Root at the center; nodes sized by
    magnitude, colored by generation.

    Complements ``cluster_tree`` (generation on a linear x-axis): here time is
    the radial axis, so the Omori cascade radiates outward while branches fan
    around the circle. Pass a polar ``ax`` (``subplot_kw={"projection": "polar"}``)
    or let one be created.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"projection": "polar"})
    root, children, gen = _tree(cat, members)
    y = _tidy_y(root, children)
    ys = np.array(list(y.values()))
    span = float(ys.max() - ys.min()) or 1.0
    # spread the layout across (just inside) ±180° so the ends don't wrap onto each other
    theta = {n: 0.98 * np.pi * (2 * (v - ys.min()) / span - 1) for n, v in y.items()}
    radius = {n: (cat.t[n] - cat.t[root]) / YEAR for n in y}

    segs = [[(theta[int(cat.parent[i])], radius[int(cat.parent[i])]), (theta[i], radius[i])]
            for i in members if int(cat.parent[i]) in y]
    ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.6, zorder=1))

    mi = np.array(list(y))
    sc = ax.scatter([theta[i] for i in mi], [radius[i] for i in mi],
                    s=marker_size(cat.m[mi], cat.params.m_min),
                    c=[gen[i] for i in mi], cmap="viridis", zorder=2)
    ax.scatter([theta[root]], [0.0], s=marker_size(cat.m[root], cat.params.m_min),
               c="red", edgecolor="k", zorder=3, label=f"root M{cat.m[root]:.1f}")
    plt.colorbar(sc, ax=ax, label="generation", shrink=0.7, pad=0.1)
    ax.set_title(f"radial genealogy — {len(members)} events\n(radius = time since root, yr)")
    _clean(ax)
    return ax


def cluster_map(cat, members, ax=None, color_by="generation"):
    """Map of the cluster with parent→child links; the branching in space.

    Nodes sized by magnitude; segments and nodes colored by generation (or
    time). Shows where offspring land relative to parents (the spatial kernel).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    root, _, gen = _tree(cat, members)
    val = (np.array([gen[int(i)] for i in members]) if color_by == "generation"
           else (cat.t[members] - cat.t[root]) / YEAR)
    vmin, vmax = float(val.min()), float(val.max())

    segs, segc = [], []
    for k, i in enumerate(members):
        p = int(cat.parent[i])
        if p >= 0:
            segs.append([(cat.x[p], cat.y[p]), (cat.x[i], cat.y[i])])
            segc.append(val[k])
    lc = LineCollection(segs, array=np.array(segc), cmap="viridis",
                        linewidths=0.6, alpha=0.6, zorder=1)
    lc.set_clim(vmin, vmax)
    ax.add_collection(lc)

    sc = ax.scatter(cat.x[members], cat.y[members],
                    s=marker_size(cat.m[members], cat.params.m_min),
                    c=val, cmap="viridis", vmin=vmin, vmax=vmax,
                    edgecolor="k", linewidth=0.2, zorder=2)
    ax.scatter([cat.x[root]], [cat.y[root]], s=marker_size(cat.m[root], cat.params.m_min),
               c="red", edgecolor="k", zorder=3, label=f"root M{cat.m[root]:.1f}")
    plt.colorbar(sc, ax=ax, label=color_by)
    ax.set_aspect("equal")
    ax.set_title(f"cluster in space — {len(members)} events")
    _clean(ax)
    return ax


def cluster_diagnostics(cat, members, axes=None):
    """Cumulative count since the root (Omori-cumulative) and events per generation."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4))
    root, _, gen = _tree(cat, members)

    dt = np.sort(cat.t[members] - cat.t[root]) / YEAR
    axes[0].step(dt, np.arange(1, len(dt) + 1), where="post")
    axes[0].set(xlabel="time since root (yr)", ylabel="cumulative events",
                title="cluster growth (Omori-cumulative)")

    g = np.array([gen[int(i)] for i in members])
    counts = np.bincount(g)
    axes[1].bar(np.arange(len(counts)), counts)
    axes[1].set(xlabel="generation", ylabel="events",
                title="events per generation")
    return axes
