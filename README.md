# The Limits of Falsifiability

**When biology's shadows lie: measuring how much dimensional projections destroy topology**

---

## Quick Start

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

**The core finding:** When you project scRNA-seq data (D_sys ≈ 14) into t-SNE/UMAP (D_obs = 2), approximately **67% of apparent neighbors are wrong**. The clusters you see are partially hallucinated.

---

## Repository Structure

```
limits-of-falsifiability/
├── toolkit/                    # ← START HERE
│   ├── falsifiability.py       # Complete library (~600 lines)
│   ├── demo_lorenz.py          # Shadow box demonstration
│   ├── demo_scrna.py           # scRNA-seq analysis
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
- **67% of neighbors are wrong** (topological aliasing)
- **~40% of cluster assignments are wrong** (cluster aliasing)
- **0.0002% of state space is sampled** (coverage collapse)

This isn't a bug in t-SNE/UMAP. It's geometry.

### The Solution

**Think like cosmologists**: Accept fundamental observational limits and build epistemology around them, rather than pretending the shadow is the territory.

---

## Key Metrics

| Metric | What It Measures | Typical Value |
|--------|------------------|---------------|
| **D_sys** | Intrinsic dimensionality (participation ratio) | ~10-20 for scRNA-seq |
| **Aliasing** | Fraction of 2D neighbors that weren't high-D neighbors | ~60-70% |
| **Coverage** | Fraction of high-D space sampled | ~0.0002% |

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
