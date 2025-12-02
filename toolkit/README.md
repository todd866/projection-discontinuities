# Falsifiability Toolkit

**Cosmological-style tools for asking "what can we know?" about high-dimensional biological systems.**

---

## The 30-Second Version

```python
from falsifiability import analyze_dataset, generate_synthetic

# Generate test data
data, labels = generate_synthetic(n_samples=3000, n_dims=50)

# Run analysis
results = analyze_dataset(data, "My Data")

print(f"D_sys: {results['d_sys']:.1f}")        # Intrinsic dimensionality
print(f"Aliasing: {results['aliasing']:.1%}")  # How much the shadow lies
```

Output:
```
D_sys: 8.3
Aliasing: 65.2%  ← 2/3 of your t-SNE neighbors are WRONG
```

---

## What This Toolkit Does

When you project high-dimensional data (like scRNA-seq) into 2D (like t-SNE/UMAP), the projection **lies about topology**:

- Points that appear as neighbors in 2D often weren't neighbors in the original space
- Clusters that look "cleanly separated" are actually entangled
- Boundaries you draw are wrong for ~60-70% of points near them

This toolkit measures exactly how much the projection lies.

---

## Files

| File | Purpose | Trilemma Leg |
|------|---------|--------------|
| `falsifiability.py` | The complete library (~600 lines) | Core metrics |
| `demo_lorenz.py` | Lorenz shadow box | Spatial (aliasing) |
| `demo_scrna.py` | scRNA-seq aliasing | Spatial (aliasing) |
| `demo_regime.py` | Falsifiability regime diagram | When Popper fails |
| `demo_memory.py` | Non-ergodic memory | Temporal (time avg fails) |
| `README.md` | This file | |
| `requirements.txt` | Dependencies | |

---

## The Inference Trilemma

The toolkit demonstrates that **all three classical escape routes from measurement uncertainty are blocked**:

| Escape Route | Why It Fails | Demo |
|--------------|--------------|------|
| **Ensemble averaging** | Curse of dimensionality: can't sample high-D space | `demo_scrna.py` |
| **Time averaging** | Non-ergodicity: hidden memory breaks ergodic assumption | `demo_memory.py` |
| **Direct measurement** | Perturbation: energy injection destroys the phenomenon | (physical principle) |

This is not a technological limitation. It is a structural feature of high-dimensional systems.

---

## Key Concepts

### D_sys (Intrinsic Dimensionality)

How many "effective dimensions" does your data occupy? Measured via participation ratio:

```
D_sys = (Σλ_i)² / Σ(λ_i²)
```

where λ_i are eigenvalues of the covariance matrix.

- Typical scRNA-seq: D_sys ≈ 10-20
- If D_sys = 14 and D_obs = 2, you're compressing 14D into 2D

### Aliasing

What fraction of 2D neighbors were NOT neighbors in the original space?

```
Aliasing = 1 - mean(Jaccard similarity of neighbor sets)
```

- Aliasing = 0%: Perfect projection
- Aliasing = 50%: Half your neighbors are hallucinated
- Aliasing = 67%: Two-thirds of what you see is a lie

### Coverage

What fraction of the high-D space have you actually sampled?

```
Coverage = (occupied cells) / (total possible cells)
```

With 10,000 cells in 20D space: Coverage ≈ 0.0002%

---

## Usage Examples

### Analyze synthetic data (no dependencies beyond sklearn)

```bash
python demo_scrna.py
```

### Analyze real scRNA-seq (requires scanpy)

```bash
pip install scanpy
python demo_scrna.py --dataset pbmc3k
```

### Run the Lorenz shadow box

```bash
python demo_lorenz.py
```

### Use in your own code

```python
from falsifiability import (
    participation_ratio,
    compute_aliasing,
    compute_coverage,
    analyze_dataset,
    plot_hairball
)

# Your data (n_cells x n_genes)
data = load_my_data()

# Quick metrics
d_sys, eigenvalues = participation_ratio(data)
print(f"Intrinsic dimensionality: {d_sys:.1f}")

# Full analysis
results = analyze_dataset(data, "My Dataset")

# The "hairball of truth" - visual proof of aliasing
plot_hairball(results['data_2d'], results['data_pca'])
```

---

## The Philosophical Point

Standard bioinformatics treats 2D plots as ground truth:
- Draw cluster boundaries
- Call them "cell types"
- Make biological claims

We show that ~60-70% of the neighborhood relationships in these plots are **wrong**.

This isn't a criticism of t-SNE or UMAP specifically. It's a geometric inevitability when D_sys >> D_obs.

**The solution isn't better algorithms. The solution is acknowledging the limits of what 2D projections can tell you.**

Think like a cosmologist:
- Cosmologists can't see past the cosmic microwave background
- They don't pretend they can
- They build theory around what IS accessible

Biology needs the same epistemological honesty about dimensional projections.

---

## Sharing with Other AI Assistants

This toolkit is designed to be shared via copy-paste. To get another AI up to speed:

1. Paste `falsifiability.py` (the complete library)
2. Paste this `README.md`
3. Optionally paste one demo script

The library has a "CONTEXT FOR AI ASSISTANTS" section at the top explaining the framework.

---

## Citation

If you use this toolkit, please cite:

```
Todd, I. (2025). The limits of falsifiability. BioSystems, 258, 105608.
https://doi.org/10.1016/j.biosystems.2025.105608
```

---

## License

MIT License
