# Projection-Induced Discontinuities

**Quantifying topological aliasing in high-dimensional dynamical systems**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

When high-dimensional dynamical systems are observed through low-dimensional coordinates, continuous trajectories can appear discontinuous. We call this phenomenon **topological aliasing**: the "jumps" are projection artifacts, not features of the underlying dynamics.

This toolkit provides computational methods to:
- Quantify aliasing rates in projected data
- Detect false discontinuities ("teleportations") in shadows of chaotic attractors
- Measure neighborhood preservation in dimensionality reduction (t-SNE, UMAP)
- Map regime boundaries where binary hypothesis testing breaks down

## Key Results

Using the **Lorenz attractor** as a minimal model:
- **47%** of dynamical states are misclassified under 2D projection
- **199** false discontinuities appear in the shadow that don't exist in the 3D flow

Validated on **scRNA-seq data** (n = 90,300 cells across 4 datasets):
- **75.5%** of apparent neighbors in t-SNE projections were not neighbors in high-dimensional space

## Installation

```bash
pip install -e .
```

For scRNA-seq analysis:
```bash
pip install -e ".[scanpy]"
```

## Quick Start

### Lorenz Attractor Demo

```python
from projection_discontinuities import (
    generate_lorenz_trajectory,
    compute_shadow_aliasing,
    count_teleportations
)

# Generate Lorenz trajectory
xyz = generate_lorenz_trajectory(n_points=10000)

# Compute aliasing in (y,z) projection
aliasing = compute_shadow_aliasing(xyz)
print(f"Aliasing rate: {aliasing:.1%}")

# Count false discontinuities
teleports = count_teleportations(xyz)
print(f"False discontinuities: {teleports}")
```

### scRNA-seq Analysis

```python
from projection_discontinuities import (
    load_scanpy_dataset,
    analyze_dataset
)

# Load standard dataset
data, name, species = load_scanpy_dataset('pbmc3k')

# Compute aliasing metrics
results = analyze_dataset(data, name)
print(f"Effective dimension: {results['d_eff']:.1f}")
print(f"Aliasing rate: {results['aliasing']:.1%}")
print(f"Coverage: {results['coverage']:.4%}")
```

## Demos

All demonstrations are in `demos.py`:

```bash
python demos.py              # Run all demos
python demos.py lorenz       # Lorenz shadow box
python demos.py chaotic      # Lorenz, Rossler, Henon comparison
python demos.py timeseries   # Mackey-Glass time series
python demos.py scrna        # scRNA-seq aliasing
python demos.py regime       # Falsifiability regimes
python demos.py memory       # Non-ergodic memory
```

## Generating Paper Figures

```bash
python generate_figures.py
```

Outputs to `figures/`:
- `fig_shadow_box.pdf` - Lorenz projection with aliasing zones
- `fig_multi_dataset.pdf` - scRNA-seq aliasing across datasets
- `fig_scale_dependent.pdf` - Regime boundary diagram
- `fig_sub_landauer_sr.pdf` - Stochastic resonance scaling

## Core Functions

### Metrics
- `participation_ratio(X)` - Effective dimensionality via eigenvalue concentration
- `compute_aliasing(X, X_proj)` - Fraction of neighbors destroyed by projection
- `compute_coverage(X)` - State space coverage estimate

### Lorenz Analysis
- `generate_lorenz_trajectory()` - Integrate Lorenz system
- `compute_shadow_aliasing()` - Aliasing rate in 2D projection
- `count_teleportations()` - False discontinuities in shadow

### Visualization
- `plot_shadow_box()` - Lorenz attractor with projection overlay
- `plot_metrics_dashboard()` - Multi-panel aliasing summary
- `plot_regime_boundary()` - Binary test failure regions

## Paper

**Projection-Induced Discontinuities in Nonlinear Dynamical Systems: Quantifying Topological Aliasing in High-Dimensional Data**

Todd, I. (2025). *Chaos, Solitons & Fractals* (submitted).

## Citation

```bibtex
@article{todd2025projection,
  title={Projection-Induced Discontinuities in Nonlinear Dynamical Systems:
         Quantifying Topological Aliasing in High-Dimensional Data},
  author={Todd, Ian},
  journal={Chaos, Solitons \& Fractals},
  year={2025},
  note={Submitted}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
