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

**Chaotic systems** (Lorenz, Rössler, Hénon):
- **50-56%** aliasing rates across canonical attractors
- **199** false discontinuities in Lorenz shadow that don't exist in the 3D flow

**Time series** (Mackey-Glass delay equation):
- **50-61%** aliasing for low-chaos (τ=17), **76-91%** for high-chaos (τ=30)

**scRNA-seq benchmarks** (PBMC 3k, Paul15):
- **70.5%** aliasing (t-SNE), **79.3%** (UMAP), error bars <0.2% across 5 seeds

## Installation

```bash
pip install -e .
```

For scRNA-seq analysis:
```bash
pip install -e ".[scanpy]"
```

## Quick Start

The easiest way to run the analyses is via `demos.py`:

```bash
# Lorenz shadow box demonstration
python demos.py lorenz

# Compare Lorenz, Rössler, and Hénon attractors
python demos.py chaotic

# Mackey-Glass time series embedding
python demos.py timeseries

# scRNA-seq aliasing (requires scanpy)
python demos.py scrna
```

Each demo prints quantitative results and saves figures to `figures/`.

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

## Reproducing the Paper

```bash
python generate_figures.py
```

This regenerates all figures from the manuscript into `figures/`.

**Note:** scRNA-seq analyses require the optional `scanpy` dependency and download ~1GB of data.

Outputs:
- `fig_shadow_box.pdf` - Lorenz projection with aliasing zones
- `fig_multi_dataset.pdf` - scRNA-seq aliasing across datasets
- `fig_scale_dependent.pdf` - Regime boundary diagram
- `fig_sub_landauer_sr.pdf` - Stochastic resonance scaling

## Core Metrics

The toolkit computes three key metrics:

- **Participation ratio** ($D_{\text{sys}}$): Effective dimensionality via eigenvalue concentration
- **Topological aliasing rate**: Fraction of k-NN neighbors destroyed by projection
- **Coverage**: State space coverage estimate (for curse of dimensionality analysis)

See `projection_discontinuities.py` for the full implementation.

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
