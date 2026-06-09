# Moment-Bounded Spatio-Temporal ETAS — Model Specification

## Overview

This model extends the standard **Epidemic Type Aftershock Sequence (ETAS)** model with a
**spatiotemporal moment field**: a gridded scalar field tracking the cumulative *available*
seismic moment at every location. The field's single role is to **bound the local maximum
magnitude**: the largest event achievable at a point is set by the moment available in its
rupture footprint. Earthquakes deplete the field near their epicenter; tectonic loading slowly
recharges it.

The rate of events is left as plain ETAS — the field does not modulate the intensity, except for
a hard zero where the budget cannot support even the smallest event. Depletion otherwise
influences seismicity *indirectly*: where the field is low, only small events are possible, small
events have low productivity, and so aftershock cascades self-limit in depleted ground.

The result is a coupled field/point-process system. ETAS supplies the triggering structure; the
moment field supplies a spatially resolved, history-dependent **energy budget** that caps event
size, location by location.

Physical analogy: each grid cell is a fault patch that charges with tectonic strain and
discharges seismic moment when it ruptures (or is ruptured by a nearby event's footprint).

---

## 1. Spatiotemporal Moment Field

### 1.1 The Field

$F(x, y, t)$ — cumulative available seismic moment **density** (N·m / km²) at location $(x,y)$
and time $t$.

The field is stored on a grid of cell size $\Delta \times \Delta$, but the grid is only a
quadrature mesh: every physical quantity is defined through *area integrals* of $F$, so the model
is resolution-independent in the limit $\Delta \to 0$ (see §11 for the original per-cell version
this replaced).

Magnitude ↔ moment conversion (Hanks-Kanamori):

$$
M_0(M) = 10^{1.5\,M + 9.05}\ \text{N}\cdot\text{m}
$$

(Hanks & Kanamori 1979 in SI units; an earlier draft used $C = 10^{-10.7}$, which is the
dyn·cm-convention constant misapplied to N·m.)

### 1.2 Tectonic Loading (Continuous)

The field recharges everywhere at a constant rate:

$$
\frac{\partial F}{\partial t} = \dot{M}_{\text{load}}(x, y)
\qquad [\,\text{N}\cdot\text{m} / \text{km}^2 / \text{day}\,]
$$

Spatially uniform ($\dot{M}_{\text{load}}(x,y) = \dot{M}_{\text{load}}$) for now; extensible to a
strain-rate map. Over an interval $\Delta t$ with no events:

$$
F(x, y, t + \Delta t) = F(x, y, t) + \dot{M}_{\text{load}} \, \Delta t
$$

### 1.3 Rupture Geometry

Each event of magnitude $M$ has a circular rupture footprint whose area follows self-similar
stress-drop scaling (one decade of area per magnitude unit, since $M_0 \propto 10^{1.5M}$ and
$M_0 \propto \Delta\sigma\, A^{3/2}$):

$$
A(M) = A_0 \, 10^{M - M_c} \quad [\text{km}^2], \qquad
R(M) = \sqrt{A(M)/\pi} \quad [\text{km}]
$$

The disk of radius $R(M)$ centered on the epicenter is the region the event both **draws moment
from** (§1.5) and **depletes** (§1.4).

### 1.4 Depletion at Each Event (Instantaneous)

When event $i$ occurs at $(x_i, y_i)$ with magnitude $M_i$, it subtracts its full seismic moment
**uniformly** over its rupture disk:

$$
F(x, y, t_i^+) = F(x, y, t_i^-) - \frac{M_0(M_i)}{A(M_i)},
\qquad \|(x,y) - (x_i,y_i)\| \le R(M_i)
$$

- **The field is signed: $F$ may go negative.** No floor is applied. A negative value is a
  **strain deficit** — an overdrawn budget that tectonic loading must repay before the location
  can host events again. The physical guarantee moves from the field to the *rate*: where the
  budget cannot support even $M_{\min}$, the effective event rate is clamped to zero (§1.5, §3).
- **Exact accounting**: every event removes exactly $M_0(M_i)$ (modulo disk area clipped at the
  domain edge). No solve is needed — depletion is linear in the event history.
- **Closed-form superposition**: because loading and depletion are both linear, the field is
  expressible ETAS-style as

$$
F(x,y,t) = F_0 + \dot{M}_{\text{load}}\, t
- \sum_{i:\, t_i < t} \frac{M_0(M_i)}{A(M_i)}\,
\mathbf{1}\!\big[\|(x,y) - (x_i,y_i)\| \le R(M_i)\big]
$$

  so a gridded field (with summed-area-table queries) and a mesh-free evaluation (disk–disk
  overlap areas) are exactly equivalent implementations; the grid is purely a performance choice.

### 1.5 Local Maximum Magnitude (Supportability)

Define the **enclosed available moment** within the rupture disk of a hypothetical magnitude-$M$
event nucleating at $(x,y)$:

$$
\mathcal{M}(x, y, t;\, M) = \int_{\|(x',y') - (x,y)\| \le R(M)} F(x', y', t)\; dA'
$$

The event is **supportable** iff the enclosed moment covers what it would release:

$$
\mathcal{M}(x, y, t;\, M) \;\ge\; M_0(M)
$$

In a fully charged field the left side grows with the disk area ($\sim 10^{M}$), the right with
released moment ($\sim 10^{1.5M}$), so the right eventually dominates. The local maximum magnitude
is defined by the **first failure of supportability** as $M$ increases from $M_{\min}$:

$$
M_{\max}(x, y, t) = \sup \left\{ M \ge M_{\min} \;:\;
\mathcal{M}(x, y, t;\, M') \ge M_0(M') \ \ \forall\, M' \in [M_{\min},\, M] \right\}
$$

Note the integrand $F$ is **signed** (§1.4): deficit pockets inside the disk subtract from the
enclosed moment, so $\mathcal{M}(M)$ need not be monotone in $M$ and the supportability condition
can hold on disconnected intervals. $M_{\max}$ is the **smallest crossing** — the end of the
*contiguous* supportable interval starting at $M_{\min}$. Physically: a rupture grows through
intermediate sizes, so it cannot jump over an unsupportable band; growth stops at the first
magnitude the regional budget cannot sustain. This also guarantees every magnitude in the
truncated GR support $[M_{\min}, M_{\max}]$ is itself supportable. Numerically: magnitudes live
on a discrete grid of width $\delta = 0.1$ (§2), so the scan is a finite walk up the bins
$M_k = M_{\min} + k\delta$ — $M_{\max}$ is the last bin before the first failure of
$\mathcal{M}(M_k) \ge M_0(M_k)$; no bisection or continuity care is needed. If even
$M = M_{\min}$ is not supportable ($\mathcal{M}(x,y,t;M_{\min}) < M_0(M_{\min})$), the location is
**locked**: the effective rate there is zero until loading repays the deficit.

Because $M_{\max}$ integrates moment over a region rather than a single cell, an isolated charged
cell no longer bottlenecks event size, and the largest possible earthquake is set by the regional
budget — not by grid resolution.

### 1.6 Field Trajectory

A point's density is a **stochastic sawtooth**: linear recharge from loading, punctuated by a
fixed drop of $M_0(M_i)/A(M_i)$ whenever it falls inside event $i$'s rupture disk. Large events
drop a broad neighborhood at once and can drive it well below zero (deep deficit); dense swarms
can collectively lock a region until loading repays the debt.

---

## 2. Magnitude Distribution

Magnitudes are **discrete**, on bins of width $\delta = 0.1$ (matching catalog reporting
resolution): $M_k = M_{\min} + k\delta$, $k = 0, 1, 2, \ldots$ When an event is generated at
location $(x, y)$ at time $t$, its magnitude is drawn from a **truncated geometric GR** over the
supportable bins:

$$
\Pr(M = M_k \mid x, y, t) = \frac{10^{-b\,k\delta}}{\sum_{j=0}^{k_{\max}} 10^{-b\,j\delta}},
\qquad k = 0, \ldots, k_{\max}(x,y,t)
$$

where $M_{k_{\max}} = M_{\max}(x,y,t)$ is the largest supportable bin (§1.5). This is the
discretization of the exponential GR density $f(M) \propto e^{-\beta M}$, $\beta = b\ln 10$;
sampling is a single finite categorical draw and the supportability scan is a finite walk over
the same bins.

- Well-charged locations ($M_{\max} \gg M_{\min}$) behave like standard GR.
- Near-depleted locations ($k_{\max} = 0$) produce only $M_{\min}$ events.
- Locked locations (no supportable bin) generate no events.

---

## 3. Conditional Intensity (ETAS)

The conditional intensity is the standard ETAS form, multiplied only by a **hard lock indicator**
— zero rate where the budget cannot support even $M_{\min}$, untouched everywhere else:

$$
\lambda(t, x, y \mid \mathcal{H}_t) = \Big[ \mu(x, y)
+ \sum_{i:\, t_i < t} \nu(M_i)\, g(t - t_i)\, h(x - x_i,\, y - y_i;\, M_i) \Big]
\cdot \mathbf{1}\big[ M_{\max}(x,y,t) \ge M_{\min} \big]
$$

The indicator is not an extra mechanism — it is the statement that the truncated GR of §2 has
empty support at locked locations, expressed at the rate level. In the branching simulation (§6)
it is implemented by **discarding** queued events that pop at locked locations; equivalently, in
a thinning construction it only ever lowers the rate below the ungated ETAS envelope, so either
algorithm remains valid.

Beyond the lock, the field's influence on seismicity is through the **magnitude truncation** of
§1.5–§2: a
depleted region can host only small events, and small events have low productivity $\nu(M)$, so
the rate feedback is *emergent* rather than imposed. Concretely:

- A partially depleted patch still hosts **many small** aftershocks at the full ETAS rate — only
  the sizes are capped. (Caveat: a mainshock consumes roughly its whole supportability margin, so
  its core can drop briefly below even the $M_{\min}$ threshold — *locked during the early Omori
  peak* — pushing the earliest aftershocks into the surrounding ring. In-core activity resumes on
  the recharge timescale of $M_{\min}$, which is short: time to re-support magnitude $M$ scales
  like $10^{0.5M}/\dot{M}_{\text{load}}$. This is an emergent, testable prediction of the model,
  at odds with the abundant immediate in-rupture-area aftershocks in real catalogs — watch for it
  in simulations.)
- But those small events seed few offspring (since $\nu(M) = K\,10^{\alpha(M-M_c)}$ falls steeply
  with magnitude), so the local branching ratio $n_{\text{local}} = \int \nu(M)\, f(M\mid\text{loc})\, dM$
  drops and the cascade dies out faster there.
- When a region depletes all the way to $M_{\max} < M_{\min}$ it is **locked** (§1.5): the
  indicator zeroes the rate there until loading repays the deficit.

(An earlier draft multiplied $\lambda$ by a moment-availability gate $a(F)$; it was dropped because
the truncation already supplies the rate feedback, the gate conflated the strain-budget timescale
with co-seismic stress triggering, and it would have wrongly suppressed small aftershocks in the
rupture zone.)

### 3.2 Background Rate

$$
\mu(x, y) = \mu_0 \qquad [\text{events / day / km}^2] \quad (\text{spatially uniform for now})
$$

### 3.3 Productivity

$$
\nu(M) = K \, 10^{\alpha(M - M_c)}
$$

### 3.4 Temporal Kernel (Omori-Utsu)

$$
g(\tau) = \frac{p - 1}{c} \left(1 + \frac{\tau}{c}\right)^{-p}, \qquad \tau > 0
$$

Normalized to integrate to 1 over $(0, \infty)$, which **requires $p > 1$ strictly** — at $p = 1$
the prefactor vanishes and the integral diverges (no proper density exists). Values just above 1
are legal but very long-tailed (the mean delay diverges for $p \le 2$).

### 3.5 Spatial Triggering Kernel (Isotropic Power-Law)

$$
h(r; M) = \frac{q - 1}{\pi\, d^2(M)} \left(1 + \frac{r^2}{d^2(M)}\right)^{-q},
\qquad
d(M) = D \, 10^{\gamma(M - M_c)/2} \quad [\text{km}]
$$

Note: the **triggering** kernel $h$ (where aftershocks are *placed*) and the **rupture disk**
$R(M)$ (where moment is *removed*, §1.3–1.4) are distinct and have different scales — $h$ governs
offspring locations, $R(M)$ governs moment consumption.

---

## 4. Model Coupling

```
ETAS  →  Field:   each event subtracts M₀(Mᵢ)/A(Mᵢ) from F over its rupture disk R(Mᵢ)
Field →  size:    supportability Mmax(x,y,t) bounds the magnitude drawn at (x,y)
Field →  rate:    hard lock only — λ = 0 where Mmax < Mmin (empty magnitude support)
size  →  rate:    smaller events ⇒ lower productivity ν(M) ⇒ fewer offspring (emergent)
Loading → Field:  continuous recharge Ṁ_load rebuilds the budget (and repays deficits)
```

The **marks** (magnitudes) are history-dependent through the shared field $F$ — the central
departure from standard ETAS, where marks are i.i.d. The rate is plain ETAS; depletion influences
it only *indirectly*, by forcing smaller (less productive) events in depleted regions.

---

## 5. Domain, Parameters & Diagnostics

### 5.1 Spatial Domain

A rectangle $[0, L_x] \times [0, L_y]$ (default $100 \times 100$ km), discretized at cell size
$\Delta$ (default 1 km). Loading and background rate are uniform over it. Boundary conventions:

- **Triggered offspring** whose sampled location falls outside the domain are discarded
  (standard ETAS boundary leakage; keep the domain comfortably larger than the analysis region).
- **Rupture disks** are clipped at the boundary: integration and depletion use only the
  in-domain portion (§7).

### 5.2 Parameters

**Time convention: all times are in days** — $t$, $\tau$, $c$, the simulation horizon
$T_{\max}$, and all rates ($\mu_0$, $\dot{M}_{\text{load}}$) are per day. Annual quantities from
the literature (e.g. tectonic loading per year) must be divided by 365.25 on input.

| Symbol | Description | Units | Typical range |
|--------|-------------|-------|---------------|
| $M_{\min}$ | Completeness / minimum magnitude | — | 2.0 – 4.0 |
| $b$ | GR b-value | — | 0.8 – 1.2 |
| **Domain** | | | |
| $L_x \times L_y$ | Domain extent | km | default 100 × 100 |
| $\Delta$ | Grid cell size (quadrature resolution) | km | 1 – 10 |
| **Field** | | | |
| $F_0$ | Initial moment density (sets initial $M_{\max}$) | N·m / km² | region-specific |
| $\dot{M}_{\text{load}}$ | Tectonic moment loading rate (density) | N·m / km² / day | region-specific |
| $A_0$ | Rupture area at $M_c$ (sets $R(M)$) | km² | ~ 1 – 100 |
| **ETAS** | | | |
| $\mu_0$ | Background rate | events/day/km² | region-specific |
| $K$ | Productivity amplitude | — | 0.01 – 0.1 |
| $\alpha$ | Productivity magnitude scaling | — | 0.5 – 2.0 |
| $c$ | Omori time offset | days | 0.001 – 1.0 |
| $p$ | Omori temporal decay exponent | — | 1.05 – 1.5 (strictly $p > 1$) |
| $D$ | Triggering spatial scale at $M_c$ | km | 1 – 20 |
| $\gamma$ | Triggering spatial magnitude scaling | — | 0.3 – 1.0 |
| $q$ | Triggering spatial power-law exponent | — | 1.5 – 2.5 |

### 5.3 Criticality Note

Because productivity rises with magnitude, the **local branching ratio depends on the local
magnitude gap** $\Delta_M = M_{\max}(x,y,t) - M_{\min}$ (taking $M_c = M_{\min}$):

$$
n(\Delta_M) = \int_{M_{\min}}^{M_{\max}} \nu(M)\, f(M)\, dM
= K\,\frac{b}{b-\alpha}\cdot\frac{1 - 10^{-(b-\alpha)\Delta_M}}{1 - 10^{-b\Delta_M}}
\qquad (\alpha \ne b),
$$

$$
n(\Delta_M) = K\, b \ln 10 \cdot \frac{\Delta_M}{1 - 10^{-b\Delta_M}} \qquad (\alpha = b).
$$

$n(\Delta_M)$ is increasing in the gap, so a well-charged region can be locally supercritical
($n > 1$). This is left **unguarded by design**: the supercritical contribution comes from the
largest events, which are also the events that deplete the field hardest, so the cascade and its
quench act on the same channel and are expected to keep each other in check. Whether that balance
holds is an empirical question for the simulations; revisit only if runaway cascades actually
appear.

---

## 6. Simulation Algorithm

**Chronological branching (cluster) simulation** with a priority queue. Offspring times and
displacements are sampled *directly* from the normalized kernels (no thinning envelope), but
events are processed strictly in time order so that magnitudes can be assigned from the
then-current field state.

```
Algorithm: simulate_catalog(T_max, domain, params)

Initialize:
  D      ← zero grid                     (accumulated depletion density)
  events ← []
  queue  ← empty min-heap keyed on time

Seed background ("immigrant") events:
  N_bg ~ Poisson(μ₀ · |domain| · T_max)
  for each: t ~ Uniform(0, T_max), (x,y) ~ Uniform(domain); push (t, x, y)

While queue not empty:
  pop earliest (t, x, y)

  1. Mmax(x,y,t) ← finite bin scan of supportability (§1.5), with
       F(x', y', t) = F₀ + Ṁ_load · t − D(x', y')          (loading is lazy/analytic)
  2. If no supportable bin: discard (location locked — this IS the lock indicator)
  3. Draw magnitude:  M ~ discrete truncated GR over bins [Mmin, Mmax]   (§2)
  4. Deplete:         D += M₀(M)/A(M) over the in-domain part of disk R(M)
  5. Record event (t, x, y, M)
  6. Spawn offspring:
       N_off ~ Poisson(ν(M))
       for each offspring:
         τ ~ Omori:    τ = c · (u^(−1/(p−1)) − 1),  u ~ U(0,1)
         r ~ spatial:  r = d(M) · sqrt(u^(−1/(q−1)) − 1),  u ~ U(0,1);  θ ~ U(0, 2π)
         child ← (t + τ,  x + r·cosθ,  y + r·sinθ)
         push child if  t + τ ≤ T_max  and child location ∈ domain   (else discard)

Return events and the final field state
```

### Notes
- **Equivalence to the intensity formulation (§3)**: the queue realizes the exact cluster
  representation of ETAS — each event independently spawns Poisson($\nu(M)$) offspring with
  i.i.d. kernel-distributed delays/displacements. Magnitude assignment at *pop time* is valid
  because pops occur in chronological order, so the field state seen by each event is exactly the
  history $\mathcal{H}_t$ it would see under the intensity formulation; discarding pops at locked
  sites implements the lock indicator.
- **No envelope, no rejection loop**: inverse-CDF sampling of $g$ and $h$ replaces thinning
  entirely; cost is $O(N \log N)$ in catalog size with no efficiency collapse as parents
  accumulate. (The locked-core pops are the only discarded work, and each costs one bin scan.)
- **Lazy loading**: loading is uniform and deterministic, so the field never needs a loading
  update — store only the depletion grid $D$ and evaluate
  $F = F_0 + \dot{M}_{\text{load}}\,t - D$ at query time.
- **Enclosed-moment queries**: direct masked disk sums over $D$'s cells (§8 build note); only one
  scan per popped event.
- **Field at intermediate times**: no snapshotting is done during simulation — the catalog fully
  determines the field at any $t$ (replay depletions for events with $t_i \le t$ into a fresh
  $D$, then $F = F_0 + \dot{M}_{\text{load}}\,t - D$). Reconstruct on demand for visualization.

---

## 7. Edge Cases & Constraints

| Situation | Handling |
|-----------|----------|
| Enclosed moment $< M_0(M_{\min})$ at a point | $M_{\max} < M_{\min}$ → location locked; rate clamped to zero until loading repays the deficit |
| Negative field density | Allowed by design: a strain deficit (§1.4); prevented from producing events by the lock, not by a floor |
| Non-monotone $\mathcal{M}(M)$ (deficit pockets in disk) | Supportability can hold on disconnected intervals; $M_{\max}$ is the end of the contiguous interval from $M_{\min}$ (smallest crossing, §1.5) |
| Rupture disk extends past domain edge | Integrate/deplete only over the in-domain portion; the outside share of removed moment is lost (or use reflecting edges — config flag) |
| Dense swarm | Cumulative depletion can lock a whole region → quiescence until recharge (emergent) |
| Offspring sampled outside domain or beyond $T_{\max}$ | Discarded at push time (standard branching-boundary leakage; keep domain larger than analysis region) |

---

## 8. Planned File Structure

```
.                                 ← repo root
├── spec.md                       ← this file
├── environment.yml               ← conda env (installs package editable)
├── pyproject.toml                ← package metadata
├── moment_etas/                  ← installable package
│   ├── model/
│   │   ├── moment_field.py       ← MomentField interface + GriddedField (summed-area table)
│   │   │                           and AnalyticField (closed-form disk overlaps) backends;
│   │   │                           load(), deplete(), enclosed_moment(), local_mmax()
│   │   ├── magnitude.py          ← truncated_gr_sample(), truncated_gr_pdf()
│   │   ├── rupture.py            ← rupture_area A(M), rupture_radius R(M)
│   │   ├── kernels.py            ← omori/spatial inverse-CDF samplers, productivity()
│   │   └── intensity.py          ← conditional_intensity() — not used by the simulator
│   │                               (branching needs no rate evaluation); kept for diagnostics
│   ├── simulation/
│   │   └── simulate.py           ← simulate_catalog() — chronological branching (§6)
│   └── visualization/
│       └── plots.py              ← field_animation(), mmax_map(), space_time_plot(), mag_dist()
└── notebooks/
    └── 01_exploration.ipynb      ← interactive sandbox
```

The two field backends implement one interface; their agreement on identical simulations is the
primary correctness check (§9, grid ↔ closed form). **Build order**: `GriddedField` first (with
direct masked disk sums — no summed-area table until profiling justifies one); `AnalyticField`
later as the cross-validation backend.

---

## 9. Verification Checks

| Test | Expected result |
|------|-----------------|
| **Moment accounting** | Exact: $\int_{\text{domain}} F_{\text{final}}\, dA = \int F_0\, dA + \dot{M}_{\text{load}}\, T \, \lvert\text{domain}\rvert - \sum_i M_0(M_i)$, up to disk area clipped at domain edges (trackable) |
| **Grid ↔ closed form** | Gridded $F$ matches the superposition formula of §1.4 at any $(x,y,t)$ to quadrature accuracy |
| **GR recovery** | With large $\dot{M}_{\text{load}}$ (near-static field), pooled magnitude histogram recovers input $b$ |
| **Field sawtooth** | Plot $F$ at a fixed point over time: linear ramp with fixed drops $M_0(M_i)/A(M_i)$ from covering events; may dip below zero after large events |
| **Size suppression** | In a depleted region the local $M_{\max}$ (and the largest observed magnitude) drops; the local branching ratio $n_{\text{local}}$ falls and cascades die out faster — while small events persist |
| **Large-event migration** | After a large event, the depleted core hosts only small aftershocks; larger aftershocks occur preferentially in the still-charged surrounding ring, infilling as the core recharges |
| **Quiescence/recharge cycles** | Heavily ruptured regions show gaps in seismicity whose length $\propto \dot{M}_{\text{load}}^{-1}$ |
| **Omori recovery** | Stack aftershock sequences in still-charged regions; recover $p$, $c$ |

---

## 10. Future Extensions

- Parameter estimation (EM / MLE) with the latent field treated as observed-from-catalog
- Spatially variable loading $\dot{M}_{\text{load}}(x,y)$ from geodetic strain-rate maps
- Anisotropic / directed rupture disks (propagation toward charged regions) and fault-aligned $h$
- Free rupture-area exponent $d_A$ (relax the self-similar $A \propto 10^{M}$ assumption)
- Tapered radial depletion profile inside $R(M)$ (vs. the flat uniform-subtraction default)
- Coulomb-style stress transfer (signed field changes: some neighbors loaded, others relaxed)
- Comparison against fixed-$M_{\max}$ and scalar-budget ETAS baselines

---

## 11. Notes on the Supportability Formulation

The rupture-disk supportability definition of $M_{\max}$ (§1.5) replaced an earlier per-cell cap,
in which the maximum magnitude at a point was set by a *single cell's* budget,
$M_{\max} = \tfrac{2}{3}\log_{10}(F_{\text{cell}}/C)$, with depletion spread over a separate
Gaussian footprint. That version was discarded for two reasons:

- **Incoherence**: the size cap (one cell) and the depletion footprint (many cells) disagreed
  about how large an event "is" in space.
- **Resolution dependence**: shrinking $\Delta$ shrank the largest possible earthquake.

The current formulation removes both problems by making one region — the rupture disk $R(M)$ —
serve as the area an event draws from, is bounded by, and depletes. Key consequences:

- **Self-consistent loop**: bigger $M$ ⇒ bigger disk ⇒ more enclosed moment required *and* more
  consumed.
- **Resolution-independent**: everything is an area integral of a density; the grid is only
  quadrature — and since depletion is linear (§1.4), the field even has a closed-form
  superposition equal to the gridded version exactly.
- **Signed field, clamped rate**: $F$ may go negative (strain deficit); the physical guarantee
  is enforced at the rate level via the lock indicator (§3), not by flooring the field. An
  earlier draft floored $F$ at zero with a conserving level-solve; it was dropped because the
  floor broke linearity (and with it the superposition form) while the lock already prevents
  events where the budget is overdrawn.
- **Emergent realism**: large events leave broad depleted patches → wider quiescence and outward
  aftershock migration.

### Open choices / refinements

- **Nucleation vs. centroid**: the disk is currently centered on the nucleation point and
  isotropic; real ruptures propagate asymmetrically toward charged ground (see §10).
- **Root-find cost**: $M_{\max}$ is a 1-D bisection per accepted candidate; a summed-area table of
  $F$ makes each enclosed-moment query $O(1)$ (§6 Notes).
- **Disk approximation**: a square (integral-image) approximation of the disk is fast; a true
  circular mask is more accurate at the cost of a few more lookups.
- **Edge bookkeeping**: disks clipped at the domain boundary lose their outside share of removed
  moment; track it if exact global accounting matters (§7).
