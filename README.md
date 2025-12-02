# The Limits of Falsifiability

**Demonstrating that standard bioinformatics workflows produce topologically invalid results**

---

## What This Repository Is

This repository contains two related papers and their supporting simulations:

| Paper | Status | What It Does |
|-------|--------|--------------|
| **Paper 1:** "The Limits of Falsifiability" | Published (BioSystems 258, Oct 2025) | Philosophical argument that Popperian falsification fails in high-D biological systems |
| **Paper 2:** "The Geometry of Biological Shadows" | In preparation | Computational proof with real data showing *how much* standard methods lie |

**Paper 1** says: "Binary hypothesis testing is mathematically incoherent when D_sys >> D_obs."

**Paper 2** says: "Here's exactly how bad it is: 67% of your t-SNE neighbors are wrong."

---

## The Uncomfortable Claim

Every computational biologist knows that t-SNE/UMAP "distorts distances." What they may not realize is that these methods **create hallucinated topology**:

```
Sade-Feldman Melanoma Dataset (16,291 cells):
  D_sys (intrinsic dimensionality): 13.6
  D_obs (UMAP/t-SNE):               2

  Topological Aliasing:             67%
  → 2/3 of apparent neighbors in the 2D plot
    were NOT neighbors in the original space

  Cluster Misassignment:            ~40%
  → Cells that look "cleanly separated" in 2D
    are actually topologically entangled
```

This isn't a limitation of one dataset or one method. It's a **geometric inevitability** when you project 14 dimensions into 2.

---

## Papers

| File | Description |
|------|-------------|
| `biosystems_2025_published.pdf` | Paper 1 as published in BioSystems 258 |
| `paper2_shadow_geometry.pdf` | Paper 2 (in preparation) |

**Author:** Ian Todd, University of Sydney
**Paper 1 DOI:** [10.1016/j.biosystems.2025.105608](https://doi.org/10.1016/j.biosystems.2025.105608)

---

## The Core Distinction: System vs Shadow

| Concept | Definition | Example |
|---------|------------|---------|
| **The System (D_sys)** | The high-dimensional manifold where biology actually happens | Gene expression space, neural state space |
| **The Shadow (D_obs)** | The low-dimensional projection we can visualize | t-SNE, UMAP, a binary classifier |

Standard practice assumes the shadow faithfully represents the system. These simulations prove it doesn't.

---

## Simulation Modules

### Theoretical Demonstrations (Paper 2)

| Script | What It Proves |
|--------|----------------|
| `01_binary_projection.py` | Information preserved by binary tests → 1/k^n (approaches zero) |
| `02_sub_landauer_sr.py` | Signals below Landauer limit require ensemble averaging (SNR ∝ √N) |
| `03_predictability_horizon.py` | Chaotic systems have hard prediction limits (T ∝ ln(1/Δx)) |
| `04_scale_dependent.py` | Maps the boundary between "Popper regime" and "Ensemble regime" |
| `05_shadow_box.py` | **FLAGSHIP:** The Lorenz attractor shadow box (47% aliasing, 199 "teleportations") |
| `06_nonergodic_memory.py` | Time averaging fails when systems have hidden memory |
| `07_sample_complexity.py` | Coverage collapses exponentially: n=15 → 0.01% of space sampled |

### Real-World Validation (NEW)

| Script | What It Proves |
|--------|----------------|
| `08_scrna_aliasing.py` | **THE HAIRBALL:** Applies aliasing metrics to real scRNA-seq (Sade-Feldman melanoma) |
| `09_multi_dataset_aliasing.py` | Tests aliasing across 4 datasets (human/mouse, blood/tumor/marrow) |

---

## The Inference Trilemma

Paper 2 establishes that **all three classical escape routes from measurement uncertainty are blocked**:

| Escape Route | Why It Fails | Simulation |
|---|---|---|
| **Time averaging** | Non-ergodicity: hidden memory makes time avg ≠ ensemble avg | `06_nonergodic_memory.py` |
| **Ensemble averaging** | Curse of dimensionality: N ~ k^n samples required | `07_sample_complexity.py` |
| **Direct measurement** | Perturbation: energy injection destroys the phenomenon | (physical principle) |

This is not a technological limitation. It is a structural feature of high-dimensional systems.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the flagship theoretical simulation
python sims/05_shadow_box.py

# Run the real-data validation (requires scanpy)
python sims/08_scrna_aliasing.py

# Run multi-dataset comparison
python sims/09_multi_dataset_aliasing.py
```

Figures are saved to `figures/`.

---

## Why This Matters

The entire apparatus of computational biology—differential expression, cluster annotation, trajectory inference—rests on the assumption that 2D projections preserve meaningful structure.

We show that when D_sys ≈ 14 and D_obs = 2:
- **67% of neighbors are wrong** (topological aliasing)
- **~40% of cluster assignments are wrong** (cluster aliasing)
- **0.0002% of state space is sampled** (coverage collapse)

This means:
1. "Cell types" defined by 2D cluster boundaries are partially hallucinated
2. Binary classifications (responder/non-responder) have a ~40-67% baseline error rate from geometry alone
3. Trajectory inference in 2D may be following paths that don't exist in the real space

**We're not saying the field is wrong. We're saying the field needs to think like cosmologists**: accept fundamental observational limits and build epistemology around them, rather than pretending the shadow is the territory.

---

## Key Equations

| Equation | Description |
|----------|-------------|
| Ω_preserved / Ω_total = 1/k^n | Information preserved under binary projection |
| E_Landauer = k_B T ln 2 | Landauer limit (~3×10⁻²¹ J at 310K) |
| T_pred ≲ (1/λ) ln(L/Δx) | Predictability horizon for chaotic systems |
| SNR ∝ √N | Stochastic resonance scaling |
| D_PR = (Σλ_i)² / Σ(λ_i²) | Participation ratio (operational D_sys) |

---

## Definitions

| Term | Definition |
|------|------------|
| **D_sys** | Intrinsic dimensionality of the biological system |
| **D_obs** | Dimensionality of the observation/projection |
| **Topological Aliasing** | When low-D neighbors were not high-D neighbors |
| **Participation Ratio** | Operational measure of D_sys from eigenvalue spectrum |
| **Sub-Landauer Pattern** | Structure with E < k_B T ln 2, requiring ensemble detection |

---

## Related Work

Part of a research program on dimensional constraints in biology:

- **Todd (2025a):** "The limits of falsifiability" — BioSystems 258
- **Todd (2025b):** "Timing inaccessibility and the projection bound" — BioSystems
- **Todd (2025c):** "The geometry of biological shadows" — in preparation
- **Todd (2025d):** "The physics of immune cooperation" — submitted

---

## License

Code: MIT License
Papers: © 2025 Ian Todd (Open Access CC-BY)
