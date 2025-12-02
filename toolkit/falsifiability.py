"""
FALSIFIABILITY TOOLKIT
======================
Cosmological-style tools for asking "what can we know?" about
high-dimensional biological systems.

CONTEXT FOR AI ASSISTANTS:
--------------------------
This toolkit implements the framework from "The Limits of Falsifiability"
(BioSystems 258, 2025) and its computational companion "The Geometry of
Biological Shadows".

THE CORE ARGUMENT:
- Biological systems operate in high-dimensional spaces (D_sys ~ 10-20)
- We observe them through low-dimensional projections (D_obs = 2 for UMAP/t-SNE)
- When D_sys >> D_obs, the projection creates "topological aliasing":
  points that appear as neighbors in the shadow were NOT neighbors in reality
- This isn't a bug in UMAP - it's a geometric inevitability

THE KEY METRICS:
- D_sys: Intrinsic dimensionality (participation ratio of eigenvalues)
- Aliasing: Fraction of low-D neighbors that weren't high-D neighbors
- Coverage: Fraction of high-D space actually sampled (approaches 0 fast)

THE PHILOSOPHICAL POINT:
Standard bioinformatics treats 2D plots as ground truth. We show they're
shadows that actively lie about topology. Biology needs to think like
cosmology: accept fundamental observational limits and build epistemology
around them.

SECTIONS:
1. METRICS - Core measurements (D_sys, aliasing, coverage)
2. LOADERS - Data loading (synthetic, scanpy datasets)
3. VISUALIZATION - Plotting (hairball, comparison figures)
4. ANALYSIS - High-level analysis pipelines
5. UTILITIES - Helper functions

USAGE:
    from falsifiability import analyze_dataset, load_scanpy_dataset

    data, name, species = load_scanpy_dataset('pbmc3k')
    results = analyze_dataset(data, name)
    print(f"Aliasing: {results['aliasing']:.1%}")

Author: Ian Todd, University of Sydney
License: MIT
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Optional imports - gracefully handle missing packages
try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import scanpy as sc
    HAS_SCANPY = True
except ImportError:
    HAS_SCANPY = False

__version__ = "1.0.0"


# =============================================================================
# SECTION 1: METRICS
# =============================================================================
# These are the core measurements that quantify "how much does the shadow lie?"

def participation_ratio(data, n_components=100):
    """
    Compute intrinsic dimensionality (D_sys) via Participation Ratio.

    THE CONCEPT:
    D_sys tells us how many "effective dimensions" the data occupies.
    A dataset spread evenly across 10 dimensions has PR ≈ 10.
    A dataset concentrated on 1 dimension has PR ≈ 1.

    THE MATH:
    PR = (Σλ_i)² / Σ(λ_i²)
    where λ_i are eigenvalues of the covariance matrix.

    WHY IT MATTERS:
    If D_sys = 14 and D_obs = 2, we're compressing 14D into 2D.
    Information MUST be lost. The question is: how much topology is destroyed?

    Parameters
    ----------
    data : array (n_samples, n_features)
    n_components : int, max PCs to compute

    Returns
    -------
    pr : float, participation ratio (effective dimensionality)
    eigenvalues : array, the eigenvalue spectrum
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for participation_ratio")

    # Subsample for speed if needed
    if data.shape[0] > 3000:
        idx = np.random.choice(data.shape[0], 3000, replace=False)
        data = data[idx]

    # Center and compute PCA
    data_centered = data - np.mean(data, axis=0)
    n_comp = min(n_components, min(data.shape) - 1)

    pca = PCA(n_components=n_comp)
    pca.fit(data_centered)

    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Remove numerical zeros

    pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    return pr, eigenvalues


def compute_aliasing(data_high_d, data_low_d, k=10):
    """
    Measure topological aliasing: how often does the shadow lie about neighbors?

    THE CONCEPT:
    For each point, find its k nearest neighbors in high-D and low-D.
    If the projection were faithful, these sets would be identical.
    Aliasing = 1 - (overlap between the sets)

    THE INTERPRETATION:
    - Aliasing = 0%: Perfect projection, all neighbors preserved
    - Aliasing = 50%: Half of apparent neighbors are "hallucinated"
    - Aliasing = 67%: Two-thirds of what you see is a lie

    WHY IT MATTERS:
    When you draw a cluster boundary in t-SNE, you're implicitly saying
    "these cells are similar." If aliasing is 67%, that statement is
    wrong for 2/3 of the cells.

    Parameters
    ----------
    data_high_d : array (n_samples, high_dims)
    data_low_d : array (n_samples, low_dims), typically 2
    k : int, number of neighbors to compare

    Returns
    -------
    aliasing_rate : float, fraction of neighbors that don't match
    jaccard_similarities : array, per-point Jaccard similarities
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for compute_aliasing")

    n_samples = min(data_high_d.shape[0], 3000)  # Subsample for speed
    idx = np.random.choice(data_high_d.shape[0], n_samples, replace=False)

    high_d = data_high_d[idx]
    low_d = data_low_d[idx]

    # Find k-NN in each space
    nn_high = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
    nn_low = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')

    nn_high.fit(high_d)
    nn_low.fit(low_d)

    _, neighbors_high = nn_high.kneighbors(high_d)
    _, neighbors_low = nn_low.kneighbors(low_d)

    # Exclude self (first neighbor)
    neighbors_high = neighbors_high[:, 1:]
    neighbors_low = neighbors_low[:, 1:]

    # Compute Jaccard similarity for each point
    jaccard_sims = []
    for i in range(n_samples):
        set_high = set(neighbors_high[i])
        set_low = set(neighbors_low[i])
        intersection = len(set_high & set_low)
        union = len(set_high | set_low)
        jaccard_sims.append(intersection / union if union > 0 else 0)

    aliasing_rate = 1 - np.mean(jaccard_sims)
    return aliasing_rate, np.array(jaccard_sims)


def compute_cluster_aliasing(data_high_d, data_low_d, n_clusters=8):
    """
    Measure cluster-based aliasing: how often does 2D classification fail?

    THE CONCEPT:
    1. Cluster in high-D (the "true" labels)
    2. Train a classifier using only 2D coordinates
    3. Measure mismatch = how often the shadow lies about cluster identity

    This parallels the Lorenz "shadow box" analysis where we ask:
    "If I draw a decision boundary in the shadow, how often is it wrong?"

    Parameters
    ----------
    data_high_d : array (n_samples, high_dims)
    data_low_d : array (n_samples, 2)
    n_clusters : int, number of clusters

    Returns
    -------
    mismatch_rate : float, fraction of misclassified points
    labels_true : array, high-D cluster labels
    labels_pred : array, 2D-predicted labels
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for compute_cluster_aliasing")

    # Cluster in high-D
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_true = kmeans.fit_predict(data_high_d)

    # Train classifier on 2D to predict high-D labels
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(data_low_d, labels_true)
    labels_pred = clf.predict(data_low_d)

    mismatch_rate = np.mean(labels_true != labels_pred)
    return mismatch_rate, labels_true, labels_pred


def compute_coverage(data, n_bins=3, n_dims=20):
    """
    Measure coverage: what fraction of high-D space is sampled?

    THE CONCEPT:
    Discretize the high-D space into a grid. Count how many grid cells
    contain at least one data point. Coverage = occupied / total.

    THE MATH:
    With n_bins per dimension and n_dims dimensions:
    Total cells = n_bins^n_dims

    For n_bins=3, n_dims=20: Total = 3^20 ≈ 3.5 billion cells
    Even with 10,000 samples, coverage ≈ 0.0003%

    WHY IT MATTERS:
    You can't characterize a space you've never visited. If coverage is
    0.0003%, there are vast regions of possible cellular states that
    you've never observed and cannot make claims about.

    Parameters
    ----------
    data : array (n_samples, n_features)
    n_bins : int, bins per dimension
    n_dims : int, dimensions to consider

    Returns
    -------
    coverage : float, fraction of space covered
    n_occupied : int, number of cells with data
    n_total : int, total possible cells
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for compute_coverage")

    n_dims = min(n_dims, data.shape[1])

    # PCA to reduce to n_dims
    pca = PCA(n_components=n_dims)
    data_reduced = pca.fit_transform(data)

    # Normalize to [0, 1]
    data_norm = (data_reduced - data_reduced.min(axis=0)) / \
                (data_reduced.max(axis=0) - data_reduced.min(axis=0) + 1e-10)

    # Discretize
    binned = np.floor(data_norm * n_bins).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)

    # Count unique cells
    cells = set(tuple(b) for b in binned)
    n_occupied = len(cells)
    n_total = n_bins ** n_dims

    coverage = n_occupied / n_total
    return coverage, n_occupied, n_total


def compute_aliasing_vs_dimension(data_pca, d_obs_values=[2, 3, 5, 10, 15, 20], k=10):
    """
    Sweep D_obs and measure aliasing at each level.

    THE CONCEPT:
    As D_obs approaches D_sys, aliasing should decrease.
    This shows that aliasing isn't a bug in the algorithm -
    it's a consequence of dimensional compression.

    Parameters
    ----------
    data_pca : array (n_samples, n_components), PCA-transformed data
    d_obs_values : list, dimensions to test
    k : int, neighbors for aliasing computation

    Returns
    -------
    results : list of (d_obs, aliasing_rate) tuples
    """
    results = []
    for d_obs in d_obs_values:
        if d_obs >= data_pca.shape[1]:
            continue
        data_low = data_pca[:, :d_obs]
        aliasing, _ = compute_aliasing(data_pca, data_low, k=k)
        results.append((d_obs, aliasing))
    return results


# =============================================================================
# SECTION 2: LOADERS
# =============================================================================
# Functions to load data for analysis

def generate_synthetic(n_samples=5000, n_dims=50, n_clusters=5):
    """
    Generate synthetic high-dimensional data with cluster structure.

    Useful for testing the toolkit without external data dependencies.

    Parameters
    ----------
    n_samples : int
    n_dims : int, number of dimensions
    n_clusters : int, number of clusters

    Returns
    -------
    data : array (n_samples, n_dims)
    labels : array (n_samples,), cluster assignments
    """
    data = []
    labels = []

    for i in range(n_samples):
        cluster = np.random.randint(n_clusters)
        # Each cluster has a different mean in a subset of dimensions
        center = np.zeros(n_dims)
        start = cluster * (n_dims // n_clusters)
        end = (cluster + 1) * (n_dims // n_clusters)
        center[start:end] = 3.0

        point = center + np.random.randn(n_dims) * 0.5
        data.append(point)
        labels.append(cluster)

    return np.array(data), np.array(labels)


def load_scanpy_dataset(name):
    """
    Load a standard scRNA-seq dataset via scanpy.

    Available datasets:
    - 'pbmc3k': 2,700 PBMCs from 10X Genomics
    - 'paul15': Mouse bone marrow (Paul et al. 2015)
    - 'pbmc68k': 68,000 PBMCs (reduced version)

    Parameters
    ----------
    name : str, dataset name

    Returns
    -------
    data : array (n_cells, n_genes), log-normalized expression
    dataset_name : str, full name for display
    species : str, 'Human' or 'Mouse'
    """
    if not HAS_SCANPY:
        raise ImportError("scanpy required for load_scanpy_dataset")

    if name == 'pbmc3k':
        print("  Loading PBMC 3k...")
        adata = sc.datasets.pbmc3k()
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        dataset_name = "PBMC 3k (10X)"
        species = "Human"

    elif name == 'paul15':
        print("  Loading Paul et al. 2015...")
        adata = sc.datasets.paul15()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        dataset_name = "Paul15 (Bone Marrow)"
        species = "Mouse"

    elif name == 'pbmc68k':
        print("  Loading PBMC 68k (reduced)...")
        adata = sc.datasets.pbmc68k_reduced()
        dataset_name = "PBMC 68k (10X)"
        species = "Human"

    else:
        raise ValueError(f"Unknown dataset: {name}. Options: pbmc3k, paul15, pbmc68k")

    # Convert to dense array
    if hasattr(adata.X, 'toarray'):
        data = adata.X.toarray()
    else:
        data = np.array(adata.X)

    return data, dataset_name, species


# =============================================================================
# SECTION 3: VISUALIZATION
# =============================================================================
# Plotting functions for results

def plot_hairball(data_2d, data_high_d, n_lines=300, k=5, save_path=None):
    """
    The 'Hairball of Truth': visualize high-D neighbors on the 2D plot.

    THE CONCEPT:
    Draw lines connecting points that are TRUE neighbors in high-D space.
    If the 2D projection were faithful, lines would be short and local.
    Long lines crossing "cluster boundaries" prove topological aliasing.

    Parameters
    ----------
    data_2d : array (n_samples, 2)
    data_high_d : array (n_samples, high_dims)
    n_lines : int, number of source points to draw from
    k : int, neighbors per source point
    save_path : str, optional path to save figure

    Returns
    -------
    fig : matplotlib Figure
    """
    if not HAS_MATPLOTLIB or not HAS_SKLEARN:
        raise ImportError("matplotlib and sklearn required for plot_hairball")

    n_total = data_2d.shape[0]
    idx_sources = np.random.choice(n_total, min(n_lines, n_total), replace=False)

    # Find true high-D neighbors
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(data_high_d)
    _, indices = nn.kneighbors(data_high_d[idx_sources])

    # Create line segments
    segments = []
    for i, source_idx in enumerate(idx_sources):
        source_pos = data_2d[source_idx]
        for neighbor_idx in indices[i, 1:]:  # Skip self
            target_pos = data_2d[neighbor_idx]
            segments.append([source_pos, target_pos])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: The Shadow (clean t-SNE)
    ax1 = axes[0]
    ax1.scatter(data_2d[:, 0], data_2d[:, 1], c='#457B9D', s=8, alpha=0.6)
    ax1.set_title("A. The Shadow (Standard 2D Projection)", fontsize=12, fontweight='bold')
    ax1.axis('off')

    # Right: The Truth (with hairball)
    ax2 = axes[1]
    ax2.scatter(data_2d[:, 0], data_2d[:, 1], c='lightgray', s=5, alpha=0.4)
    lc = LineCollection(segments, colors='#8B0000', linewidths=0.6, alpha=0.4)
    ax2.add_collection(lc)
    ax2.scatter(data_2d[idx_sources, 0], data_2d[idx_sources, 1], c='black', s=12, zorder=10)
    ax2.set_title("B. The Truth (High-D Neighbors Connected)", fontsize=12, fontweight='bold')
    ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def plot_metrics_dashboard(results, save_path=None):
    """
    Plot a summary dashboard of falsifiability metrics.

    Parameters
    ----------
    results : dict from analyze_dataset()
    save_path : str, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plot_metrics_dashboard")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A: Eigenvalue spectrum
    ax = axes[0, 0]
    eigenvalues = results['eigenvalues'][:50]
    ax.semilogy(eigenvalues, 'o-', color='#2A9D8F', markersize=4)
    d_sys = results['d_sys']
    if int(d_sys) < len(eigenvalues):
        ax.axhline(y=eigenvalues[int(d_sys)], color='red', linestyle='--',
                   label=f'D_sys = {d_sys:.0f}')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_title(f'A. Eigenvalue Spectrum (D_sys = {d_sys:.1f})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # B: 2D embedding
    ax = axes[0, 1]
    data_2d = results['data_2d']
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c='#457B9D', s=1, alpha=0.5)
    ax.set_title('B. 2D Projection (D_obs = 2)', fontweight='bold')
    ax.axis('off')

    # C: Aliasing distribution
    ax = axes[1, 0]
    jaccard = results['jaccard_similarities']
    ax.hist(jaccard, bins=30, color='#E63946', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(jaccard), color='black', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(jaccard):.2f}')
    ax.set_xlabel('Jaccard Similarity (High-D vs 2D Neighbors)')
    ax.set_ylabel('Count')
    ax.set_title(f'C. Neighbor Preservation (Aliasing = {results["aliasing"]:.1%})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # D: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    summary = f"""
    FALSIFIABILITY METRICS
    ━━━━━━━━━━━━━━━━━━━━━━
    Dataset: {results['name']}
    Cells: {results['n_cells']:,}

    D_sys (intrinsic):     {results['d_sys']:.1f}
    D_obs (projection):    2
    Compression ratio:     {results['d_sys']/2:.1f}x

    Topological Aliasing:  {results['aliasing']:.1%}
    Cluster Aliasing:      {results['cluster_aliasing']:.1%}
    Coverage:              {results['coverage']:.2e}

    ━━━━━━━━━━━━━━━━━━━━━━

    INTERPRETATION:
    {results['aliasing']:.0%} of apparent neighbors in
    the 2D plot were NOT neighbors in the
    original high-dimensional space.

    Any classification based on 2D cluster
    boundaries has a ~{results['aliasing']:.0%} baseline
    error rate from geometry alone.
    """
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax.set_title('D. Summary', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


# =============================================================================
# SECTION 4: ANALYSIS
# =============================================================================
# High-level analysis pipelines

def analyze_dataset(data, name="Dataset", verbose=True, embedding_fn=None, random_state=42):
    """
    Run full falsifiability analysis on a dataset.

    This is the main entry point. Given expression data, it computes
    all the key metrics and returns a results dictionary.

    Parameters
    ----------
    data : array (n_samples, n_features)
    name : str, dataset name for display
    verbose : bool, print progress
    embedding_fn : callable, optional
        Custom embedding function. Should take (n_samples, n_features) array
        and return (n_samples, 2) array. If None, uses t-SNE.
        Example: lambda X: umap.UMAP().fit_transform(X)
    random_state : int, optional
        Random seed for reproducibility. Default 42.

    Returns
    -------
    results : dict with keys:
        - name, n_cells, n_genes
        - d_sys, eigenvalues
        - aliasing, jaccard_similarities
        - cluster_aliasing
        - coverage
        - data_pca, data_2d
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for analyze_dataset")

    rng = np.random.default_rng(random_state)

    if verbose:
        print(f"\nAnalyzing: {name}")
        print("=" * 50)

    # Filter low-variance features
    var = np.var(data, axis=0)
    keep = var > 0.01
    data_filt = data[:, keep]

    n_cells, n_genes = data.shape
    n_genes_filt = np.sum(keep)

    if verbose:
        print(f"  Cells: {n_cells:,}")
        print(f"  Features: {n_genes:,} (filtered: {n_genes_filt:,})")

    # 1. D_sys
    if verbose:
        print("  Computing D_sys...")
    d_sys, eigenvalues = participation_ratio(data_filt)
    if verbose:
        print(f"    D_sys = {d_sys:.1f}")

    # 2. PCA + embedding
    if verbose:
        print("  Computing 2D embedding...")
    n_pca = min(50, data_filt.shape[1] - 1)
    pca = PCA(n_components=n_pca, random_state=random_state)
    data_pca = pca.fit_transform(data_filt)

    # Subsample for embedding
    n_sub = min(3000, len(data_pca))
    idx = rng.choice(len(data_pca), n_sub, replace=False)
    data_pca_sub = data_pca[idx]

    if embedding_fn is not None:
        data_2d = embedding_fn(data_pca_sub)
    else:
        tsne = TSNE(n_components=2, perplexity=30, random_state=random_state, max_iter=500)
        data_2d = tsne.fit_transform(data_pca_sub)

    # 3. Aliasing
    if verbose:
        print("  Computing aliasing...")
    aliasing, jaccard = compute_aliasing(data_pca_sub, data_2d, k=10)
    if verbose:
        print(f"    Aliasing = {aliasing:.1%}")

    # 4. Cluster aliasing
    if verbose:
        print("  Computing cluster aliasing...")
    cluster_aliasing, _, _ = compute_cluster_aliasing(data_pca_sub, data_2d, n_clusters=8)
    if verbose:
        print(f"    Cluster aliasing = {cluster_aliasing:.1%}")

    # 5. Coverage
    if verbose:
        print("  Computing coverage...")
    coverage, _, _ = compute_coverage(data_filt, n_bins=3, n_dims=20)
    if verbose:
        print(f"    Coverage = {coverage:.2e}")

    return {
        'name': name,
        'n_cells': n_cells,
        'n_genes': n_genes,
        'd_sys': d_sys,
        'eigenvalues': eigenvalues,
        'aliasing': aliasing,
        'jaccard_similarities': jaccard,
        'cluster_aliasing': cluster_aliasing,
        'coverage': coverage,
        'data_pca': data_pca_sub,
        'data_2d': data_2d,
    }


def analyze_with_embedding(data_high_d, data_2d, name="Dataset", verbose=True, k=10):
    """
    Analyze aliasing for a pre-computed embedding.

    Use this when you already have a 2D embedding (e.g., from scanpy's
    adata.obsm['X_umap']) and want to measure how much it lies.

    Parameters
    ----------
    data_high_d : array (n_samples, n_features)
        High-dimensional data (e.g., PCA coordinates, not raw counts)
    data_2d : array (n_samples, 2)
        Pre-computed 2D embedding
    name : str
        Dataset name for display
    verbose : bool
        Print progress
    k : int
        Number of neighbors for aliasing computation

    Returns
    -------
    results : dict with keys:
        - d_sys, aliasing, cluster_aliasing, coverage

    Example
    -------
    >>> import scanpy as sc
    >>> adata = sc.read_h5ad("my_data.h5ad")
    >>> results = analyze_with_embedding(
    ...     adata.obsm['X_pca'],
    ...     adata.obsm['X_umap'],
    ...     name="My dataset"
    ... )
    >>> print(f"Aliasing: {results['aliasing']:.1%}")
    """
    if verbose:
        print(f"\nAnalyzing: {name}")
        print("=" * 50)

    # D_sys
    d_sys, eigenvalues = participation_ratio(data_high_d)
    if verbose:
        print(f"  D_sys = {d_sys:.1f}")

    # Aliasing
    aliasing, jaccard = compute_aliasing(data_high_d, data_2d, k=k)
    if verbose:
        print(f"  Aliasing = {aliasing:.1%}")

    # Cluster aliasing
    cluster_aliasing, _, _ = compute_cluster_aliasing(data_high_d, data_2d)
    if verbose:
        print(f"  Cluster aliasing = {cluster_aliasing:.1%}")

    # Coverage
    coverage, _, _ = compute_coverage(data_high_d, n_bins=3, n_dims=20)
    if verbose:
        print(f"  Coverage = {coverage:.2e}")

    return {
        'name': name,
        'd_sys': d_sys,
        'eigenvalues': eigenvalues,
        'aliasing': aliasing,
        'jaccard_similarities': jaccard,
        'cluster_aliasing': cluster_aliasing,
        'coverage': coverage,
    }


def compare_datasets(datasets, verbose=True):
    """
    Compare falsifiability metrics across multiple datasets.

    Parameters
    ----------
    datasets : list of (data, name, species) tuples
    verbose : bool

    Returns
    -------
    results : list of result dicts
    summary : dict with averages
    """
    results = []

    for data, name, species in datasets:
        r = analyze_dataset(data, name, verbose=verbose)
        r['species'] = species
        results.append(r)

    # Compute averages
    summary = {
        'avg_d_sys': np.mean([r['d_sys'] for r in results]),
        'avg_aliasing': np.mean([r['aliasing'] for r in results]),
        'avg_cluster_aliasing': np.mean([r['cluster_aliasing'] for r in results]),
        'n_datasets': len(results),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Dataset':<25} {'D_sys':>8} {'Aliasing':>10} {'Cluster':>10}")
        print("-" * 55)
        for r in results:
            print(f"{r['name']:<25} {r['d_sys']:>8.1f} {r['aliasing']:>9.1%} {r['cluster_aliasing']:>9.1%}")
        print("-" * 55)
        print(f"{'AVERAGE':<25} {summary['avg_d_sys']:>8.1f} {summary['avg_aliasing']:>9.1%} {summary['avg_cluster_aliasing']:>9.1%}")

    return results, summary


# =============================================================================
# SECTION 5: UTILITIES
# =============================================================================

def preprocess_expression(data, log_normalize=True, min_variance=0.01):
    """
    Standard preprocessing for expression data.

    Parameters
    ----------
    data : array (n_cells, n_genes)
    log_normalize : bool, apply log1p if max > 100 (raw counts)
    min_variance : float, filter genes with var < this

    Returns
    -------
    data_processed : array
    """
    # Log normalize if raw counts
    if log_normalize and np.max(data) > 100:
        data = np.log1p(data)

    # Filter low-variance genes
    var = np.var(data, axis=0)
    keep = var > min_variance

    return data[:, keep]


def print_results_table(results_list):
    """Print a formatted results table."""
    print(f"\n{'Dataset':<25} {'Species':<8} {'Cells':>8} {'D_sys':>6} {'Aliasing':>10} {'Cluster':>10}")
    print("-" * 75)
    for r in results_list:
        species = r.get('species', 'N/A')
        print(f"{r['name']:<25} {species:<8} {r['n_cells']:>8,} {r['d_sys']:>6.1f} {r['aliasing']:>9.1%} {r['cluster_aliasing']:>9.1%}")


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == '__main__':
    print("FALSIFIABILITY TOOLKIT")
    print("=" * 50)
    print("Testing with synthetic data...")

    # Generate and analyze synthetic data
    data, labels = generate_synthetic(n_samples=2000, n_dims=30)
    results = analyze_dataset(data, "Synthetic Test")

    print(f"\nResults:")
    print(f"  D_sys: {results['d_sys']:.1f}")
    print(f"  Aliasing: {results['aliasing']:.1%}")
    print(f"  Cluster aliasing: {results['cluster_aliasing']:.1%}")
    print(f"  Coverage: {results['coverage']:.2e}")

    print("\nToolkit loaded successfully!")
