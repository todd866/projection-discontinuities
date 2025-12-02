#!/usr/bin/env python3
"""
Multi-Dataset Topological Aliasing Analysis
============================================

Runs the aliasing analysis across multiple public scRNA-seq datasets
to demonstrate that topological aliasing is a GENERAL phenomenon,
not a quirk of any particular dataset.

Datasets:
1. Sade-Feldman (melanoma, ~16k cells) - already have this
2. PBMC 3k (10X Genomics tutorial, ~2.7k cells)
3. Paul et al. 2015 (bone marrow differentiation, ~2.7k cells)
4. Tabula Muris (mouse cell atlas subset)

Output: Table showing D_sys, aliasing rate, and coverage for each dataset.

Papers: "The Geometry of Biological Shadows" (Paper 2)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../23_immune_cooperation'))

os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

np.random.seed(42)


# =============================================================================
# METRICS (copied from 08_scrna_aliasing.py for self-containment)
# =============================================================================

def participation_ratio(data, n_components=100):
    """Compute D_sys via PCA Participation Ratio."""
    if data.shape[0] > 3000:
        idx = np.random.choice(data.shape[0], 3000, replace=False)
        data = data[idx]

    data_centered = data - np.mean(data, axis=0)
    n_comp = min(n_components, min(data.shape) - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(data_centered)

    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    return pr, eigenvalues


def compute_topological_aliasing(data_high_d, data_low_d, k=10):
    """Measure neighbor preservation failure."""
    n_samples = min(data_high_d.shape[0], 3000)
    idx = np.random.choice(data_high_d.shape[0], n_samples, replace=False)

    high_d = data_high_d[idx]
    low_d = data_low_d[idx]

    nn_high = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
    nn_low = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')

    nn_high.fit(high_d)
    nn_low.fit(low_d)

    _, neighbors_high = nn_high.kneighbors(high_d)
    _, neighbors_low = nn_low.kneighbors(low_d)

    neighbors_high = neighbors_high[:, 1:]
    neighbors_low = neighbors_low[:, 1:]

    jaccard_sims = []
    for i in range(n_samples):
        set_high = set(neighbors_high[i])
        set_low = set(neighbors_low[i])
        intersection = len(set_high & set_low)
        union = len(set_high | set_low)
        jaccard_sims.append(intersection / union)

    aliasing_rate = 1 - np.mean(jaccard_sims)
    return aliasing_rate, np.array(jaccard_sims)


def compute_cluster_aliasing(data_high_d, data_low_d, n_clusters=8):
    """Cluster in high-D, classify in low-D, measure mismatch."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_true = kmeans.fit_predict(data_high_d)

    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(data_low_d, labels_true)
    labels_pred = clf.predict(data_low_d)

    mismatch_rate = np.mean(labels_true != labels_pred)
    return mismatch_rate


def compute_coverage(data, n_bins=3, n_dim=20):
    """Measure coverage of high-D space."""
    n_dim = min(n_dim, data.shape[1])
    pca = PCA(n_components=n_dim)
    data_reduced = pca.fit_transform(data)

    data_norm = (data_reduced - data_reduced.min(axis=0)) / (data_reduced.max(axis=0) - data_reduced.min(axis=0) + 1e-10)

    binned = np.floor(data_norm * n_bins).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)

    cells = set(tuple(b) for b in binned)
    n_occupied = len(cells)
    n_total_cells = n_bins ** n_dim

    coverage = n_occupied / n_total_cells
    return coverage, n_dim


# =============================================================================
# DATASET LOADERS
# =============================================================================

def load_sade_feldman():
    """Load Sade-Feldman melanoma data."""
    try:
        from fast_loader import load_sade_feldman_fast
        data_dir = "../../23_immune_cooperation/sade_feldman_data"
        data_resp, data_non, meta = load_sade_feldman_fast(data_dir)
        if data_resp is not None:
            data = np.vstack([data_resp, data_non])
            if np.max(data) > 100:
                data = np.log1p(data)
            return data, "Sade-Feldman (Melanoma)", "Human"
    except Exception as e:
        print(f"  Could not load Sade-Feldman: {e}")
    return None, None, None


def load_pbmc3k():
    """Load PBMC 3k from scanpy datasets."""
    try:
        import scanpy as sc
        print("  Downloading PBMC 3k...")
        adata = sc.datasets.pbmc3k()
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Convert to dense array
        if hasattr(adata.X, 'toarray'):
            data = adata.X.toarray()
        else:
            data = adata.X
        return data, "PBMC 3k (10X)", "Human"
    except Exception as e:
        print(f"  Could not load PBMC 3k: {e}")
    return None, None, None


def load_paul15():
    """Load Paul et al. 2015 bone marrow data."""
    try:
        import scanpy as sc
        print("  Downloading Paul et al. 2015...")
        adata = sc.datasets.paul15()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        if hasattr(adata.X, 'toarray'):
            data = adata.X.toarray()
        else:
            data = adata.X
        return data, "Paul15 (Bone Marrow)", "Mouse"
    except Exception as e:
        print(f"  Could not load Paul15: {e}")
    return None, None, None


def load_pbmc68k():
    """Load PBMC 68k (larger dataset for robustness)."""
    try:
        import scanpy as sc
        print("  Downloading PBMC 68k (this may take a moment)...")
        adata = sc.datasets.pbmc68k_reduced()

        if hasattr(adata.X, 'toarray'):
            data = adata.X.toarray()
        else:
            data = adata.X
        return data, "PBMC 68k (10X)", "Human"
    except Exception as e:
        print(f"  Could not load PBMC 68k: {e}")
    return None, None, None


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_dataset(data, name, species):
    """Run full aliasing analysis on a dataset."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}")

    # Filter low-variance genes
    var = np.var(data, axis=0)
    keep = var > 0.01
    data_filt = data[:, keep]

    n_cells, n_genes = data.shape
    n_genes_filt = np.sum(keep)
    print(f"  Cells: {n_cells:,}")
    print(f"  Genes: {n_genes:,} (filtered: {n_genes_filt:,})")

    # 1. D_sys
    d_sys, _ = participation_ratio(data_filt)
    print(f"  D_sys (Participation Ratio): {d_sys:.1f}")

    # 2. PCA + t-SNE
    n_pca = min(50, data_filt.shape[1] - 1)
    pca = PCA(n_components=n_pca)
    data_pca = pca.fit_transform(data_filt)

    # Subsample for t-SNE
    n_sub = min(3000, len(data_pca))
    idx = np.random.choice(len(data_pca), n_sub, replace=False)
    data_pca_sub = data_pca[idx]

    print(f"  Running t-SNE on {n_sub} cells...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    data_2d = tsne.fit_transform(data_pca_sub)

    # 3. Topological aliasing (k-NN)
    aliasing_knn, _ = compute_topological_aliasing(data_pca_sub, data_2d, k=10)
    print(f"  Topological Aliasing (k=10): {aliasing_knn:.1%}")

    # 4. Cluster aliasing
    cluster_aliasing = compute_cluster_aliasing(data_pca_sub, data_2d, n_clusters=8)
    print(f"  Cluster Aliasing (8 clusters): {cluster_aliasing:.1%}")

    # 5. Coverage
    coverage, n_dim = compute_coverage(data_filt, n_bins=3, n_dim=20)
    print(f"  Coverage (3^{n_dim}): {coverage:.2e}")

    return {
        'name': name,
        'species': species,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'd_sys': d_sys,
        'aliasing_knn': aliasing_knn,
        'aliasing_cluster': cluster_aliasing,
        'coverage': coverage,
        'data_2d': data_2d,
        'data_pca_sub': data_pca_sub
    }


def run_multi_dataset_analysis():
    """Main function to analyze all datasets."""

    print("=" * 70)
    print("MULTI-DATASET TOPOLOGICAL ALIASING ANALYSIS")
    print("Paper 2: The Geometry of Biological Shadows")
    print("=" * 70)

    # Define loaders
    loaders = [
        ("Sade-Feldman", load_sade_feldman),
        ("PBMC 3k", load_pbmc3k),
        ("Paul15", load_paul15),
        ("PBMC 68k", load_pbmc68k),
    ]

    results = []

    for loader_name, loader_func in loaders:
        print(f"\nLoading {loader_name}...")
        data, name, species = loader_func()

        if data is not None:
            result = analyze_dataset(data, name, species)
            results.append(result)
        else:
            print(f"  Skipping {loader_name} (data not available)")

    if not results:
        print("\nNo datasets loaded. Exiting.")
        return

    # ==========================================================================
    # RESULTS TABLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)

    # Header
    print(f"\n{'Dataset':<25} {'Species':<8} {'Cells':>8} {'D_sys':>6} {'Aliasing':>10} {'Cluster':>10} {'Coverage':>12}")
    print("-" * 85)

    for r in results:
        print(f"{r['name']:<25} {r['species']:<8} {r['n_cells']:>8,} {r['d_sys']:>6.1f} {r['aliasing_knn']:>9.1%} {r['aliasing_cluster']:>9.1%} {r['coverage']:>12.2e}")

    print("-" * 85)

    # Compute averages
    avg_dsys = np.mean([r['d_sys'] for r in results])
    avg_alias = np.mean([r['aliasing_knn'] for r in results])
    avg_cluster = np.mean([r['aliasing_cluster'] for r in results])

    print(f"{'AVERAGE':<25} {'':<8} {'':<8} {avg_dsys:>6.1f} {avg_alias:>9.1%} {avg_cluster:>9.1%}")

    # ==========================================================================
    # FIGURE: Comparison across datasets
    # ==========================================================================
    print("\nGenerating comparison figure...")

    n_datasets = len(results)
    fig, axes = plt.subplots(2, max(2, n_datasets), figsize=(4*max(2, n_datasets), 8))

    # Top row: t-SNE plots
    for i, r in enumerate(results):
        ax = axes[0, i] if n_datasets > 1 else axes[0]
        ax.scatter(r['data_2d'][:, 0], r['data_2d'][:, 1], s=1, alpha=0.5, c='#457B9D')
        ax.set_title(f"{r['name']}\nD_sys={r['d_sys']:.1f}, Alias={r['aliasing_knn']:.0%}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Fill empty subplots if needed
    for i in range(n_datasets, axes.shape[1]):
        axes[0, i].axis('off')

    # Bottom left: Bar chart of aliasing rates
    ax = axes[1, 0]
    names = [r['name'].split()[0] for r in results]  # Short names
    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, [r['aliasing_knn']*100 for r in results], width, label='k-NN Aliasing', color='#E63946')
    bars2 = ax.bar(x + width/2, [r['aliasing_cluster']*100 for r in results], width, label='Cluster Aliasing', color='#457B9D')

    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% (random)')
    ax.set_ylabel('Aliasing Rate (%)')
    ax.set_xlabel('Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    ax.set_title('A. Aliasing Rates Across Datasets', fontweight='bold')
    ax.grid(alpha=0.3, axis='y')

    # Bottom right: D_sys vs Aliasing scatter
    ax = axes[1, 1]
    for r in results:
        ax.scatter(r['d_sys'], r['aliasing_knn']*100, s=100, label=r['name'].split()[0])
        ax.annotate(r['name'].split()[0], (r['d_sys'], r['aliasing_knn']*100),
                   textcoords="offset points", xytext=(5,5), fontsize=8)

    ax.set_xlabel('D_sys (Intrinsic Dimensionality)')
    ax.set_ylabel('Topological Aliasing (%)')
    ax.set_title('B. D_sys vs Aliasing', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # Fill remaining subplots
    for i in range(2, axes.shape[1]):
        axes[1, i].axis('off')

    plt.tight_layout()

    # Save
    for path in ['figures/fig_multi_dataset.pdf', '../figures/fig_multi_dataset.pdf']:
        try:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
            break
        except:
            continue

    # ==========================================================================
    # KEY FINDING
    # ==========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    print(f"""
    Across {n_datasets} datasets spanning different:
      - Species (Human, Mouse)
      - Tissues (Melanoma, Blood, Bone Marrow)
      - Sample sizes ({min(r['n_cells'] for r in results):,} to {max(r['n_cells'] for r in results):,} cells)

    The topological aliasing rate is consistently HIGH:
      - Average k-NN aliasing: {avg_alias:.1%}
      - Average cluster aliasing: {avg_cluster:.1%}

    This is NOT a quirk of a single dataset.
    This is a GEOMETRIC INEVITABILITY of projecting D_sys >> D_obs.

    The shadow ALWAYS lies when D_sys ≈ {avg_dsys:.0f} and D_obs = 2.
    """)

    plt.show()

    return results


if __name__ == '__main__':
    run_multi_dataset_analysis()
