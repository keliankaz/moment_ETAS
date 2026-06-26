# Implementation Plan

## Unsorted ideas

### Testing
  Make sure all the samplers are working as expected 
  - Omori's law
  - Spatial 'displacements' (make sure that the cutoff matches the rupture radius)
  - Root finding for the supportability scan (plot both the support and the earthquake moment)

Suspicious behavior:
  - Spatial density of events has steps (from the incremental supportability scan?)

### Parameter space exploration

Create helper functions to determine dimensionless paramers/time scales:
  - Time scales:
    - aftershock time scale
    - recurrence of Mmax from loading
    - recurrence of Mmax from mu/b
    - recurrence of Mmin from mu
    - Explosion time scale (related to aftershock time scale?)
  - Structural parameters:
    - alpha vs beta
    - branching ratio
  - Spatial parameters:
    - Lx/ly ratio
    - L(M0max) / L_min (long and skinny ruptures)
    - heterogeneous loading
      - sigma over delta r
      - step over jump scale (L_min/Dx)

### Understanding emergent behavior:
  - Mmax (long term average and its variance)
  - Cyclicity
  - Spatial distribution of events on the margins
  - Emergent high b-value from low b-value simulations

### Reproducing/illustrating observed behavior:
  - Mmax
  - Cyclicity
  - Corona's
  - Mogi doughnuts
  - Stepovers

### Visualizations
- branching ratio over time
- phase space? (n vs mu/Mdot)
  - Characteristic sequences
  - Cyclicity


  
