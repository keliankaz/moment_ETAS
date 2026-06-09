# moment-ETAS

Experimental **moment-bounded spatio-temporal ETAS** model: standard ETAS triggering, plus a
signed seismic-moment density field that bounds the local maximum magnitude via rupture-disk
supportability. Tectonic loading recharges the field; events deplete it.

See [spec.md](spec.md) for the full model specification.

## Setup

```bash
conda env create -f environment.yml
conda activate moment-etas
```

This installs the `moment_etas` package in editable mode.

## Layout

- `moment_etas/model/` — field, magnitude distribution, rupture geometry, ETAS kernels
- `moment_etas/simulation/` — thinning-based catalog simulator
- `moment_etas/visualization/` — field maps, space-time plots, magnitude distributions
- `notebooks/` — exploration notebooks
