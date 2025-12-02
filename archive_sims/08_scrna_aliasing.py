#!/usr/bin/env python3
"""
Topological Aliasing in scRNA-seq: Real-World Application
==========================================================

Applies the Paper 2 falsifiability metrics to real single-cell data.

THE QUESTION:
When we project high-dimensional gene expression to UMAP (D_obs = 2),
how much topological aliasing do we introduce?

METRICS:
1. D_sys: Intrinsic dimensionality via participation ratio
2. D_obs: UMAP embedding dimension (2)
3. Aliasing rate: How often do UMAP neighbors differ from high-D neighbors?
4. Coverage: What fraction of high-D space is actually sampled?
5. **The Hairball:** Visualizing the high-D connections on the low-D plot.

Dataset: GSE120575 (Sade-Feldman melanoma, ~16k cells, ~55k genes)

Papers: "The Geometry of Biological Shadows" (Paper 2) & "The Limits of Falsifiability" (Paper 1)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../23_immune_cooperation'))

os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Set random seed
np.random.seed(42)


def participation_ratio(data, n_components=100):
    """
    Compute effective dimensionality using PCA Participation Ratio.
    PR = (sum(lambda_i))^2 / sum(lambda_i^2)
    This is D_sys - the intrinsic dimensionality of the data manifold.
    """
    # Subsample for speed if too large
    if data.shape[0] > 3000:
        idx = np.random.choice(data.shape[0], 3000, replace=False)
        data = data[idx]

    # Center the data
    data_centered = data - np.mean(data, axis=0)

    # PCA
    n_comp = min(n_components, min(data.shape) - 1)
    pca = PCA(n_components=n_comp)
    pca.fit(data_centered)

    eigenvalues = pca.explained_variance_
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    pr = np.sum(eigenvalues)**2 / np.sum(eigenvalues**2)
    return pr, eigenvalues


def compute_topological_aliasing(data_high_d, data_low_d, k=10):
    """
    Measure topological aliasing: how often do low-D neighbors
    differ from high-D neighbors?
    """
    n_samples = min(data_high_d.shape[0], 5000)  # Subsample for speed
    idx = np.random.choice(data_high_d.shape[0], n_samples, replace=False)

    high_d = data_high_d[idx]
    low_d = data_low_d[idx]

    # Find k-nearest neighbors in each space
    nn_high = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
    nn_low = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')

    nn_high.fit(high_d)
    nn_low.fit(low_d)

    # Get neighbor indices (excluding self)
    _, neighbors_high = nn_high.kneighbors(high_d)
    _, neighbors_low = nn_low.kneighbors(low_d)

    neighbors_high = neighbors_high[:, 1:]  # Exclude self
    neighbors_low = neighbors_low[:, 1:]

    # Compute Jaccard similarity for each point
    jaccard_sims = []
    for i in range(n_samples):
        set_high = set(neighbors_high[i])
        set_low = set(neighbors_low[i])
        intersection = len(set_high & set_low)
        union = len(set_high | set_low)
        jaccard_sims.append(intersection / union)

    aliasing_rate = 1 - np.mean(jaccard_sims)
    return aliasing_rate, np.array(jaccard_sims)


def compute_coverage(data, n_bins=3):
    """Measure coverage of high-dimensional space."""
    # Reduce to manageable dimensions for coverage computation
    n_dim = min(20, data.shape[1])
    pca = PCA(n_components=n_dim)
    data_reduced = pca.fit_transform(data)

    # Normalize to [0, 1] for binning
    data_norm = (data_reduced - data_reduced.min(axis=0)) / (data_reduced.max(axis=0) - data_reduced.min(axis=0) + 1e-10)

    # Discretize into bins
    binned = np.floor(data_norm * n_bins).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)

    # Count unique cells occupied
    cells = set(tuple(b) for b in binned)
    n_occupied = len(cells)

    # Total possible cells
    n_total_cells = n_bins ** n_dim

    coverage = n_occupied / n_total_cells
    return coverage, n_occupied, n_total_cells, n_dim


def compute_cluster_aliasing(data_high_d, data_low_d, n_clusters=8):
    """
    Mirror the Lorenz shadow box analysis:
    1. Cluster in high-D space (ground truth labels)
    2. Train classifier on low-D coordinates
    3. Measure mismatch = how often the shadow lies about cluster identity

    This directly parallels the Lorenz lobe misclassification analysis.
    """
    # Cluster in high-D
    kmeans_high = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_true = kmeans_high.fit_predict(data_high_d)

    # Train classifier on low-D to predict high-D labels
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(data_low_d, labels_true)
    labels_pred = clf.predict(data_low_d)

    # Misclassification rate = aliasing
    mismatch_rate = np.mean(labels_true != labels_pred)

    return mismatch_rate, labels_true, labels_pred


def compute_trustworthiness(data_high_d, data_low_d, k=10):
    """
    Standard trustworthiness metric (Venna & Kaski 2006).
    Measures how many low-D neighbors are "false" (not neighbors in high-D).

    T(k) = 1 - (2 / (Nk(2n-3k-1))) * sum of rank errors for false neighbors

    Returns value in [0, 1] where 1 = perfect preservation.
    """
    n = len(data_high_d)

    # Get k-NN in both spaces
    nn_high = NearestNeighbors(n_neighbors=n, algorithm='auto')
    nn_low = NearestNeighbors(n_neighbors=k+1, algorithm='auto')

    nn_high.fit(data_high_d)
    nn_low.fit(data_low_d)

    # Get all distances and indices for rank computation
    dist_high, _ = nn_high.kneighbors(data_high_d)
    _, neighbors_low = nn_low.kneighbors(data_low_d)
    neighbors_low = neighbors_low[:, 1:]  # Exclude self

    # Compute ranks in high-D space
    ranks_high = np.argsort(np.argsort(dist_high, axis=1), axis=1)

    # Compute trustworthiness
    penalty = 0
    for i in range(n):
        for j_idx in range(k):
            j = neighbors_low[i, j_idx]
            r_ij = ranks_high[i, j]
            if r_ij > k:  # j is a false neighbor
                penalty += (r_ij - k)

    # Normalize
    trust = 1 - (2 / (n * k * (2 * n - 3 * k - 1))) * penalty
    return max(0, min(1, trust))


def compute_aliasing_sweep(data_pca, d_obs_values=[2, 3, 5, 10, 15], k=10):
    """
    Sweep D_obs and measure aliasing at each level.
    Shows that aliasing decreases as D_obs approaches D_sys.
    """
    results = []

    for d_obs in d_obs_values:
        if d_obs >= data_pca.shape[1]:
            continue

        # Use PCA projection to D_obs (fast, deterministic)
        data_low = data_pca[:, :d_obs]

        # Compute aliasing
        aliasing, _ = compute_topological_aliasing(data_pca, data_low, k=k)
        results.append((d_obs, aliasing))

    return results


def compare_embedding_methods(data_pca, n_subsample=3000):
    """
    Compare aliasing across different embedding methods:
    - PCA to 2D
    - t-SNE to 2D
    """
    # Subsample for speed
    idx = np.random.choice(len(data_pca), min(n_subsample, len(data_pca)), replace=False)
    data_sub = data_pca[idx]

    results = {}

    # PCA 2D
    pca_2d = data_sub[:, :2]
    aliasing_pca, _ = compute_topological_aliasing(data_sub, pca_2d, k=10)
    results['PCA-2D'] = aliasing_pca

    # t-SNE 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=500)
    tsne_2d = tsne.fit_transform(data_sub)
    aliasing_tsne, _ = compute_topological_aliasing(data_sub, tsne_2d, k=10)
    results['t-SNE-2D'] = aliasing_tsne

    return results, idx


def plot_aliasing_network(data_2d, data_high_d, n_lines=300, k=5):
    """
    The 'Hairball of Truth': Visualize high-D neighbors on the low-D plot.

    If the projection were perfect, lines would be short and local.
    Long lines crossing the plot indicate topological aliasing.
    """
    print("\nGenerating 'Hairball of Truth' visualization...")

    # Subsample points to draw connections for
    n_total = data_2d.shape[0]
    idx_sources = np.random.choice(n_total, min(n_lines, n_total), replace=False)

    # Find true high-D neighbors for these source points
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(data_high_d)
    _, indices = nn.kneighbors(data_high_d[idx_sources])

    # Create line segments: from source to each of its k true neighbors
    segments = []
    for i, source_idx in enumerate(idx_sources):
        source_pos = data_2d[source_idx]
        # Skip the first neighbor (itself)
        neighbor_indices = indices[i, 1:]

        for neighbor_idx in neighbor_indices:
            target_pos = data_2d[neighbor_idx]
            segments.append([source_pos, target_pos])

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Panel A: The Illusion (Standard t-SNE)
    ax1 = axes[0]
    ax1.scatter(data_2d[:, 0], data_2d[:, 1], c='#457B9D', s=10, alpha=0.6, edgecolors='none')
    ax1.set_title("A. The Shadow (Standard t-SNE)", fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')

    # Panel B: The Truth (High-D Neighbors Overlay)
    ax2 = axes[1]
    # Draw faint points
    ax2.scatter(data_2d[:, 0], data_2d[:, 1], c='lightgray', s=5, alpha=0.5, edgecolors='none')

    # Draw edges
    lc = LineCollection(segments, colors='magenta', linewidths=0.5, alpha=0.3)
    ax2.add_collection(lc)

    # Draw source points on top
    ax2.scatter(data_2d[idx_sources, 0], data_2d[idx_sources, 1], c='black', s=15, zorder=10)

    ax2.set_title(f"B. The Truth (High-D Neighbors)\n(Magenta lines connect true neighbors)", fontsize=14, fontweight='bold')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')

    # Add text explaining the point
    explanation = """
    THE SHADOW LIE:
    The magenta lines connect cells that are neighbors
    in the real high-dimensional space.

    Notice how they stretch across the "gaps" between
    clusters. This proves that the clusters are
    topologically entangled in reality, even if they
    look separated in the shadow.
    """
    ax2.text(0.02, 0.02, explanation, transform=ax2.transAxes, fontsize=10,
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()
    return fig


def generate_synthetic_baseline(n_samples, n_genes, n_clusters=5):
    """Generate synthetic scRNA-seq-like data for comparison."""
    data = []
    for _ in range(n_samples):
        cluster = np.random.randint(n_clusters)
        center = np.zeros(n_genes)
        center[cluster * (n_genes // n_clusters):(cluster + 1) * (n_genes // n_clusters)] = 5
        noise = np.random.exponential(0.5, n_genes)
        data.append(center + noise)
    return np.array(data)


def run_simulation():
    """Main simulation with real or synthetic data."""

    print("=" * 60)
    print("TOPOLOGICAL ALIASING IN scRNA-seq DATA")
    print("Paper 2: The Geometry of Biological Shadows")
    print("=" * 60)

    # Try to load real data
    try:
        from fast_loader import load_sade_feldman_fast
        print("\nLoading Sade-Feldman melanoma data...")
        # Try multiple paths
        data_dir = "../../23_immune_cooperation/sade_feldman_data"
        data_resp, data_non, meta = load_sade_feldman_fast(data_dir)

        if data_resp is not None:
            # Combine for full analysis
            data = np.vstack([data_resp, data_non])
            data_source = "GSE120575 (Sade-Feldman)"
            print(f"Loaded {data.shape[0]} cells x {data.shape[1]} genes")

            # Log normalize if it looks like raw counts (max value > 100)
            if np.max(data) > 100:
                print("Log-normalizing data...")
                data = np.log1p(data)

        else:
            raise ValueError("Data loading returned None")

    except Exception as e:
        print(f"Could not load real data: {e}")
        print("Using synthetic data for demonstration...")
        data = generate_synthetic_baseline(5000, 1000)
        data_source = "Synthetic"

    # Filter zero-variance genes
    print("\nFiltering low-variance genes...")
    var = np.var(data, axis=0)
    keep = var > 0.01
    data_filt = data[:, keep]
    print(f"Kept {np.sum(keep)} / {len(keep)} genes")

    # =========================================================================
    # 1. COMPUTE D_sys (Intrinsic Dimensionality)
    # =========================================================================
    print("\n" + "-" * 40)
    print("[1] INTRINSIC DIMENSIONALITY (D_sys)")
    print("-" * 40)

    d_sys, eigenvalues = participation_ratio(data_filt)
    print(f"  Participation Ratio: D_sys = {d_sys:.1f}")

    # =========================================================================
    # 2. COMPUTE LOW-D EMBEDDING (D_obs = 2)
    # =========================================================================
    print("\n" + "-" * 40)
    print("[2] LOW-DIMENSIONAL EMBEDDING (D_obs = 2)")
    print("-" * 40)

    # First reduce with PCA for speed
    print("  PCA reduction to 50 components...")
    pca = PCA(n_components=min(50, data_filt.shape[1] - 1))
    data_pca = pca.fit_transform(data_filt)

    # Then t-SNE (faster than UMAP without extra dependency)
    print("  t-SNE embedding to 2D...")
    n_subsample = min(5000, len(data_pca))
    idx_sub = np.random.choice(len(data_pca), n_subsample, replace=False)

    data_pca_sub = data_pca[idx_sub]

    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    data_2d = tsne.fit_transform(data_pca_sub)

    print(f"  Embedded {n_subsample} cells into 2D")

    # =========================================================================
    # 3. COMPUTE TOPOLOGICAL ALIASING
    # =========================================================================
    print("\n" + "-" * 40)
    print("[3] TOPOLOGICAL ALIASING")
    print("-" * 40)

    aliasing_rate, jaccard_sims = compute_topological_aliasing(
        data_pca_sub, data_2d, k=10
    )

    print(f"  Aliasing rate (k=10 neighbors): {aliasing_rate:.1%}")
    print(f"  Mean Jaccard similarity: {np.mean(jaccard_sims):.2f}")

    # =========================================================================
    # 4. COMPUTE COVERAGE
    # =========================================================================
    print("\n" + "-" * 40)
    print("[4] COVERAGE OF HIGH-D SPACE")
    print("-" * 40)

    coverage, n_occ, n_total, n_dim = compute_coverage(data_filt, n_bins=3)

    print(f"  Using first {n_dim} PCs with 3 bins per dimension")
    print(f"  Total possible cells: 3^{n_dim} = {n_total:,.0f}")
    print(f"  Cells occupied: {n_occ:,}")
    print(f"  Coverage: {coverage:.2e} ({coverage*100:.4f}%)")

    # =========================================================================
    # 5. CLUSTER-BASED ALIASING (Mirror Lorenz shadow box)
    # =========================================================================
    print("\n" + "-" * 40)
    print("[5] CLUSTER-BASED ALIASING (Lorenz-style)")
    print("-" * 40)

    cluster_aliasing, labels_true, labels_pred = compute_cluster_aliasing(
        data_pca_sub, data_2d, n_clusters=8
    )
    print(f"  Clusters in high-D: 8 (KMeans)")
    print(f"  Classifier in 2D: Logistic Regression")
    print(f"  Cluster misassignment rate: {cluster_aliasing:.1%}")
    print(f"  → {cluster_aliasing*100:.0f}% of cells are assigned to the wrong")
    print(f"     cluster when we classify based on 2D coordinates alone")

    # =========================================================================
    # 6. D_obs SWEEP
    # =========================================================================
    print("\n" + "-" * 40)
    print("[6] D_obs SWEEP (Aliasing vs Embedding Dimension)")
    print("-" * 40)

    d_obs_values = [2, 3, 5, 10, 15, 20, 30]
    aliasing_sweep = compute_aliasing_sweep(data_pca_sub, d_obs_values, k=10)

    print("  D_obs → Aliasing Rate:")
    for d_obs, alias in aliasing_sweep:
        bar = "█" * int(alias * 30)
        print(f"    {d_obs:3d}D → {alias:.1%} {bar}")

    # =========================================================================
    # 7. EMBEDDING METHOD COMPARISON
    # =========================================================================
    print("\n" + "-" * 40)
    print("[7] EMBEDDING METHOD COMPARISON")
    print("-" * 40)

    print("  Comparing PCA-2D vs t-SNE-2D...")
    method_results, _ = compare_embedding_methods(data_pca_sub, n_subsample=2000)

    print("  Method → Aliasing Rate:")
    for method, alias in method_results.items():
        bar = "█" * int(alias * 30)
        print(f"    {method:10s} → {alias:.1%} {bar}")

    # =========================================================================
    # PLOTTING
    # =========================================================================
    print("\n" + "-" * 40)
    print("Generating figures...")
    print("-" * 40)

    # FIGURE 1: Metrics Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Eigenvalue spectrum
    ax = axes[0, 0]
    ax.semilogy(eigenvalues[:50], 'o-', color='#2A9D8F', markersize=4)
    if int(d_sys) < len(eigenvalues):
        ax.axhline(y=eigenvalues[int(d_sys)], color='red', linestyle='--',
                   label=f'D_sys = {d_sys:.0f}')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_title(f'A. Eigenvalue Spectrum\n(D_sys = {d_sys:.1f})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel B: t-SNE embedding
    ax = axes[0, 1]
    ax.scatter(data_2d[:, 0], data_2d[:, 1], c='#457B9D', s=1, alpha=0.5)
    ax.set_title(f'B. 2D Embedding (D_obs = 2)', fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel C: Jaccard similarity distribution
    ax = axes[1, 0]
    ax.hist(jaccard_sims, bins=30, color='#E63946', alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(jaccard_sims), color='black', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(jaccard_sims):.2f}')
    ax.set_xlabel('Jaccard Similarity (high-D vs 2D neighbors)')
    ax.set_ylabel('Count')
    ax.set_title(f'C. Neighbor Preservation\n(Aliasing = {aliasing_rate:.1%})', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel D: Summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    TOPOLOGICAL ALIASING IN scRNA-seq
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Dataset: {data_source}
    Cells: {data.shape[0]:,}
    Genes: {data.shape[1]:,} (filtered: {np.sum(keep):,})

    SYSTEM (D_sys):
      Participation Ratio = {d_sys:.1f}

    SHADOW (D_obs):
      t-SNE dimension = 2

    ALIASING:
      Rate = {aliasing_rate:.1%}
      → {aliasing_rate*100:.0f}% of high-D neighbors are
        "lost" in the 2D projection

    COVERAGE:
      {coverage:.2e} of 3^{n_dim} cells occupied
      → The data covers a vanishing fraction
        of the high-D space

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    IMPLICATION:
    Binary classifications based on 2D cluster
    assignments have a ~{aliasing_rate*100:.0f}% baseline error
    rate due to topological aliasing alone.
    """
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax.set_title('D. Summary', fontweight='bold')

    plt.tight_layout()

    # Save Figure 1
    for path in ['figures/fig_scrna_metrics.pdf', '../figures/fig_scrna_metrics.pdf']:
        try:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
            break
        except Exception as e:
            continue

    # FIGURE 2: THE HAIRBALL (Visual Proof)
    fig2 = plot_aliasing_network(data_2d, data_pca_sub, n_lines=400, k=5)

    # Save Figure 2
    for path in ['figures/fig_scrna_hairball.pdf', '../figures/fig_scrna_hairball.pdf']:
        try:
            fig2.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved: {path}")
            break
        except Exception as e:
            continue

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: FALSIFIABILITY METRICS ON REAL DATA")
    print("=" * 60)
    print(f"""
    D_sys (intrinsic dimensionality):  {d_sys:.1f}
    D_obs (observation dimensionality): 2
    Ratio D_sys / D_obs:               {d_sys/2:.1f}x

    Topological aliasing rate:         {aliasing_rate:.1%}
    Coverage of state space:           {coverage:.2e}

    KEY INSIGHT:

    The gene expression manifold has ~{d_sys:.0f} effective dimensions,
    but we routinely visualize it in 2D. This ~{d_sys/2:.0f}x compression
    introduces {aliasing_rate:.0%} aliasing: cells that are neighbors in
    the 2D plot often weren't neighbors in the original space.

    This is not a failure of t-SNE/UMAP—it's a geometric inevitability
    when D_sys >> D_obs. The "structure" we see in 2D plots is partially
    real and partially artifact.

    The falsifiability implication: binary classifications based on
    2D cluster assignments have a ~{aliasing_rate:.0%} baseline error rate
    due to topological aliasing alone.
    """)

    plt.show()


if __name__ == '__main__':
    run_simulation()
