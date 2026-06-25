"""Visualize a single ETAS cluster as a branching process.

A cluster is an index array from ``Catalog.clusters()`` — a background root
plus its triggered descendants. Tree edges are ``(parent[i], i)`` for every
non-root member. All views take the catalog and one such member array.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from .plots import marker_size

YEAR = 365.25
# Uniform node style (no color coding); the root gets a single red accent so it
# stays identifiable — it is not reliably the largest event in the cluster.
_marker_style = {"edgecolor": "k", "linewidth": 0.2, "color": "slategray"}
_root_style = {**_marker_style, "color": "red"}


def _clean(ax):
    """Strip the axis apparatus (frame, ticks, tick labels, grid) for a clean
    network-diagram look (à la complexity-explorables); the title is kept."""
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

    x = generation depth (left→right cascade), y = tidy-tree layout. Nodes are
    uniform (no color coding), sized by magnitude; the root is red. (Time is
    *not* the x-axis because Omori bunches the immediate aftershocks at t≈0.)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    root, children, gen = _tree(cat, members)
    y = _tidy_y(root, children)
    mi = np.array(list(y))

    segs = [
        [(gen[int(cat.parent[i])], y[int(cat.parent[i])]), (gen[i], y[i])]
        for i in members
        if int(cat.parent[i]) in y
    ]
    ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.6, zorder=1))

    ax.scatter(
        [gen[i] for i in mi],
        [y[i] for i in mi],
        s=marker_size(cat.m[mi], cat.params.m_min),
        **_marker_style,
        zorder=2,
    )
    ax.scatter(
        [gen[root]],
        [y[root]],
        s=marker_size(cat.m[root], cat.params.m_min),
        **_root_style,
        zorder=3,
    )
    ax.set_title(
        f"cluster genealogy — {len(members)} events, {max(gen.values())} generations"
    )
    _clean(ax)
    return ax


def cluster_tree_radial(cat, members, ax=None):
    """Radial genealogy tree: angle = tidy-tree layout (spread over ±180°),
    radius = time since the root (years). Root (red) at the center; nodes
    uniform (no color coding), sized by magnitude.

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

    segs = [
        [(theta[int(cat.parent[i])], radius[int(cat.parent[i])]), (theta[i], radius[i])]
        for i in members
        if int(cat.parent[i]) in y
    ]
    ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.6, zorder=1))

    mi = np.array(list(y))
    ax.scatter(
        [theta[i] for i in mi],
        [radius[i] for i in mi],
        s=marker_size(cat.m[mi], cat.params.m_min),
        **_marker_style,
        zorder=2,
    )
    ax.scatter(
        [theta[root]],
        [0.0],
        s=marker_size(cat.m[root], cat.params.m_min),
        **_root_style,
        zorder=3,
    )
    ax.set_title(
        f"radial genealogy — {len(members)} events\n(radius = time since root, yr)"
    )
    _clean(ax)
    return ax


def cluster_map(cat, members, ax=None):
    """Map of the cluster with parent→child links; the branching in space.

    Nodes sized by magnitude. Shows
    where offspring land relative to parents (the spatial kernel).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 6))
    root, _, _ = _tree(cat, members)

    segs = []

    for i in members:
        p = int(cat.parent[i])
        if p >= 0:
            segs.append([(cat.x[p], cat.y[p]), (cat.x[i], cat.y[i])])
    ax.add_collection(LineCollection(segs, colors="0.8", linewidths=0.3, zorder=1))

    ax.scatter(
        cat.x[members],
        cat.y[members],
        s=marker_size(cat.m[members], cat.params.m_min),
        **_marker_style,
        zorder=2,
    )
    ax.scatter(
        [cat.x[root]],
        [cat.y[root]],
        s=marker_size(cat.m[root], cat.params.m_min),
        **_root_style,
        zorder=3,
    )
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
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="time since root (yr)",
        ylabel="cumulative events",
        title="cluster growth (Omori-cumulative)",
    )

    g = np.array([gen[int(i)] for i in members])
    counts = np.bincount(g)
    axes[1].bar(np.arange(len(counts)), counts)
    axes[1].set(xlabel="generation", ylabel="events", title="events per generation")
    return axes


# --- git-graph view ---------------------------------------------------------

def _heavy_branches(cat, members):
    """Heavy-path decomposition of the cluster tree.

    Returns (root, branches, node_branch) where each branch is a list of node
    indices forming a heavy path (each node continues into its largest-subtree
    child); lighter children start new branches. node_branch maps node → branch
    index. This collapses linear runs into single lanes, the git way.
    """
    root, children, _ = _tree(cat, members)
    nodes = [int(i) for i in members]
    # subtree sizes (iterative post-order)
    size = {}
    stack = [(root, False)]
    while stack:
        n, done = stack.pop()
        if done:
            size[n] = 1 + sum(size[c] for c in children[n])
        else:
            stack.append((n, True))
            for c in children[n]:
                stack.append((c, False))
    heavy = {n: (max(children[n], key=lambda c: size[c]) if children[n] else None)
             for n in nodes}
    branches, node_branch = [], {}
    for n in nodes:
        par = int(cat.parent[n])
        if par < 0 or heavy.get(par) != n:          # a head: root or a light child
            path, cur = [], n
            while cur is not None:
                path.append(cur)
                node_branch[cur] = len(branches)
                cur = heavy[cur]
            branches.append(path)
    return root, branches, node_branch


def cluster_git(cat, members, ax=None, pad=0.12):
    """Git-graph view of the cluster.

    x = log10(time since root, days); each branch (heavy path) is a horizontal
    lane. Lanes are packed greedily (a freed lane is reused by a later,
    non-overlapping branch — height = max concurrent branches, not total), with
    the lane nearest the parent's preferred to keep edges short, and median
    epicentral distance from the root as the tiebreak. Children peel off their
    parent with a bezier curve. Nodes sized by magnitude; root in red.

    Trade-off (by design): lane reuse gives up strict planarity, so a few edges
    may cross — exactly the git aesthetic.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5))
    root, branches, node_branch = _heavy_branches(cat, members)

    # x = log10 days since root; root sits just left of the earliest event
    dt = (cat.t[members] - cat.t[root])
    pos = dt[dt > 0]
    x_root = np.log10(pos.min()) - 0.5 if len(pos) else 0.0
    x = {int(i): (np.log10(d) if d > 0 else x_root) for i, d in zip(members, dt)}
    dist = {int(i): float(np.hypot(cat.x[i] - cat.x[root], cat.y[i] - cat.y[root]))
            for i in members}

    nb = len(branches)
    x_head = np.array([x[b[0]] for b in branches])
    x_tail = np.array([x[b[-1]] for b in branches])
    med_dist = np.array([np.median([dist[n] for n in b]) for b in branches])
    parent_of_head = [int(cat.parent[b[0]]) for b in branches]   # -1 for root branch

    # greedy nearest-free-lane packing, processed by head time then distance
    order = sorted(range(nb), key=lambda b: (x_head[b], med_dist[b]))
    free_after = []                  # x beyond which each lane is free
    branch_lane = {}
    for b in order:
        pn = parent_of_head[b]
        target = 0 if pn < 0 else branch_lane[node_branch[pn]]
        cands = [ln for ln, fa in enumerate(free_after) if fa <= x_head[b] - pad]
        lane = (min(cands, key=lambda ln: (abs(ln - target), ln)) if cands
                else len(free_after))
        if lane == len(free_after):
            free_after.append(0.0)
        free_after[lane] = x_tail[b] + pad
        branch_lane[b] = lane
    node_lane = {n: branch_lane[bi] for n, bi in node_branch.items()}

    # within-branch lanes (horizontal) + bezier peel-offs from parent to head
    for b, path in enumerate(branches):
        ln = branch_lane[b]
        ax.plot([x[n] for n in path], [ln] * len(path), color="0.6", lw=0.9, zorder=1)
        pn = parent_of_head[b]
        if pn >= 0:
            x0, y0, x1, y1 = x[pn], node_lane[pn], x[path[0]], ln
            xm = 0.5 * (x0 + x1)
            ax.add_patch(PathPatch(
                Path([(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
                     [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                fill=False, edgecolor="0.6", lw=0.9, zorder=1))

    nodes = np.array([int(i) for i in members])
    s = np.clip(marker_size(cat.m[nodes], cat.params.m_min), 8, 250)
    ax.scatter([x[n] for n in nodes], [node_lane[n] for n in nodes], s=s,
               **_marker_style, zorder=2)
    ax.scatter([x[root]], [0], s=np.clip(marker_size(cat.m[root], cat.params.m_min), 8, 250),
               **_root_style, zorder=3)

    # readable time x-axis; lanes have no quantitative meaning so hide y
    tick_days = [1 / 24, 1, 30, YEAR, 10 * YEAR, 100 * YEAR]
    tick_lab = ["1 hr", "1 d", "1 mo", "1 yr", "10 yr", "100 yr"]
    xmin = x_root
    xmax = max(x_tail) if nb else 1.0
    keep = [(np.log10(d), lab) for d, lab in zip(tick_days, tick_lab)
            if xmin <= np.log10(d) <= xmax]
    if keep:
        ax.set_xticks([t for t, _ in keep])
        ax.set_xticklabels([lab for _, lab in keep])
    ax.set_yticks([])
    for sp in ("top", "left", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(-1, len(free_after))
    ax.set_xlabel("time since root")
    ax.set_title(f"cluster git-graph — {len(members)} events, {len(free_after)} lanes")
    return ax
