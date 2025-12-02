# The Limits of Falsifiability

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Regime: Ensemble](https://img.shields.io/badge/Regime-Ensemble-red.svg)](#the-falsifiability-regimes)

**When biology's shadows lie: measuring how much dimensional projections destroy topology**

---

## Quick Start

### Installation

```bash
cd toolkit
pip install .
```

### Usage

```bash
cd toolkit
python demo_scrna.py           # Analyze synthetic data
python demo_scrna.py --dataset pbmc3k  # Real data (requires scanpy)
python demo_lorenz.py          # Shadow box demonstration
```

---

## What This Is

This repository contains:

| Component | Description |
|-----------|-------------|
| **Paper 1** | "The Limits of Falsifiability" — BioSystems 258, Oct 2025 (published) |
| **Paper 2** | "The Geometry of Biological Shadows" — computational companion (in prep) |
| **Toolkit** | Clean Python library for measuring topological aliasing |

**The core finding:** When you project scRNA-seq data (D_sys ≈ 10-40) into t-SNE/UMAP (D_obs = 2), approximately **75.5% of apparent neighbors are wrong**. The clusters you see are partially hallucinated. This was validated across 4 standard datasets (n = 90,300 cells total).

---

## Repository Structure

```
limits-of-falsifiability/
├── toolkit/                    # ← START HERE
│   ├── falsifiability.py       # Complete library (~600 lines)
│   ├── demo_lorenz.py          # Shadow box: topological aliasing
│   ├── demo_scrna.py           # scRNA-seq aliasing analysis
│   ├── demo_regime.py          # Popper vs Ensemble regimes
│   ├── demo_memory.py          # Non-ergodic memory
│   ├── setup.py                # pip install .
│   └── README.md               # API documentation
│
├── biosystems_2025_published.pdf   # Paper 1 (published)
├── paper2_shadow_geometry.pdf      # Paper 2 (in prep)
├── paper2_shadow_geometry.tex      # Paper 2 source
│
├── figures/                    # Generated figures
├── archive_sims/               # Original simulation scripts (historical)
└── archive/                    # Other archived materials
```

---

## The Argument

### The Problem

Computational biology routinely:
1. Projects 10,000+ dimensional gene expression into 2D
2. Draws cluster boundaries
3. Calls them "cell types"
4. Makes biological claims

### What We Show

When D_sys >> D_obs:
- **75.5% of neighbors are wrong** (topological aliasing)
- **~40% of cluster assignments are wrong** (cluster aliasing)
- **<0.001% of state space is sampled** (coverage collapse)

This isn't a bug in t-SNE/UMAP. It's geometry.

### The Solution

**Think like cosmologists**: Accept fundamental observational limits and build epistemology around them, rather than pretending the shadow is the territory.

---

## Key Metrics

| Metric | What It Measures | Typical Value |
|--------|------------------|---------------|
| **D_sys** | Intrinsic dimensionality (participation ratio) | 10-40 for scRNA-seq |
| **Aliasing** | Fraction of 2D neighbors that weren't high-D neighbors | 66-83% |
| **Coverage** | Fraction of high-D space sampled | <0.001% |

---

## The Falsifiability Regimes

Binary hypothesis testing (Popperian falsification) works in some regimes but fails in others:

| Regime | Dimensionality | Signal | Method |
|--------|----------------|--------|--------|
| **Popper** (green) | Low (n < 10) | High | Single binary tests work |
| **Ensemble** (red) | High (n > 10) | Low | Only multivariate methods work |

**Most of biology (scRNA-seq, neuroscience, ecology) operates in the Ensemble regime.** The red badge on this repo indicates that the methods described here are designed for high-dimensional biological systems where classical falsification is geometrically incoherent.

---

## Citation

```bibtex
@article{todd2025limits,
  title={The limits of falsifiability},
  author={Todd, Ian},
  journal={BioSystems},
  volume={258},
  pages={105608},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.biosystems.2025.105608}
}
```

---

## License

Code: MIT
Papers: CC-BY

---

## Author

Ian Todd, University of Sydney
