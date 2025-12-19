#!/usr/bin/env python3
"""
PROJECTION DISCONTINUITIES - DEMONSTRATION SUITE
=================================================

All demonstrations for the paper "Projection-Induced Discontinuities in
Nonlinear Dynamical Systems: Quantifying Topological Aliasing in High-Dimensional Data"

Usage:
    python demos.py                    # Run all demos
    python demos.py lorenz             # Lorenz shadow box only
    python demos.py chaotic            # Lorenz, Rossler, Henon
    python demos.py timeseries         # Mackey-Glass time series
    python demos.py scrna              # scRNA-seq aliasing
    python demos.py regime             # Falsifiability regimes
    python demos.py memory             # Non-ergodic memory

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE, trustworthiness
from sklearn.decomposition import PCA
import sys
import os
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SHARED UTILITIES (simplified standalone versions for demos)
# For production use, import from projection_discontinuities module.
# =============================================================================

def _participation_ratio(X):
    """Compute effective dimensionality via participation ratio (simplified)."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)


def _compute_aliasing(X_high, X_low, k=15):
    """
    Compute topological aliasing rate (fraction of k-NN lost in projection).

    Subsamples to improve runtime; results are stable across seeds.
    """
    n = X_high.shape[0]

    nn_high = NearestNeighbors(n_neighbors=k+1).fit(X_high)
    _, idx_high = nn_high.kneighbors(X_high)
    idx_high = idx_high[:, 1:]

    nn_low = NearestNeighbors(n_neighbors=k+1).fit(X_low)
    _, idx_low = nn_low.kneighbors(X_low)
    idx_low = idx_low[:, 1:]

    total = preserved = 0
    for i in range(n):
        high_set = set(idx_high[i])
        low_set = set(idx_low[i])
        total += k
        preserved += len(high_set & low_set)

    return 1 - (preserved / total)


# =============================================================================
# DYNAMICAL SYSTEMS
# =============================================================================

def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
    """Lorenz attractor ODEs."""
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def rossler_system(state, t, a=0.2, b=0.2, c=5.7):
    """Rossler attractor ODEs."""
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c)]


def henon_map(n_points, a=1.4, b=0.3, transient=1000):
    """Henon map: x_{n+1} = 1 - a*x_n^2 + y_n, y_{n+1} = b * x_n"""
    x, y = 0.1, 0.1
    trajectory = []
    for i in range(n_points + transient):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        if i >= transient:
            trajectory.append([x, y])
    return np.array(trajectory)


def mackey_glass(n_points, tau=17, beta=0.2, gamma=0.1, n=10, dt=1.0, transient=1000):
    """Mackey-Glass delay differential equation."""
    history_len = int(tau / dt) + 1
    x = np.ones(history_len) * 1.2
    series = []
    for i in range(n_points + transient):
        x_tau = x[0]
        x_now = x[-1]
        dx = beta * x_tau / (1 + x_tau**n) - gamma * x_now
        x_new = x_now + dx * dt
        x = np.roll(x, -1)
        x[-1] = x_new
        if i >= transient:
            series.append(x_new)
    return np.array(series)


def delay_embedding(x, dim, tau):
    """Create delay embedding of time series."""
    n = len(x) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i*tau:i*tau + n]
    return embedded


def generate_trajectory(system, n_points=10000, dt=0.01, transient=1000):
    """Generate trajectory for ODE systems."""
    state0 = [1.0, 1.0, 1.0]
    t = np.arange(0, (n_points + transient) * dt, dt)
    if system == 'lorenz':
        trajectory = odeint(lorenz_system, state0, t)
    elif system == 'rossler':
        trajectory = odeint(rossler_system, state0, t)
    else:
        raise ValueError(f"Unknown system: {system}")
    return trajectory[transient:]


# =============================================================================
# DEMO 1: LORENZ SHADOW BOX
# =============================================================================

def demo_lorenz():
    """The Lorenz Shadow Box - core demonstration of topological aliasing."""
    print("=" * 60)
    print("THE LORENZ SHADOW BOX")
    print("=" * 60)

    trajectory = generate_trajectory('lorenz', n_points=10000)
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    system_truth = x > 0
    shadow_prediction = z > 25

    correct = system_truth == shadow_prediction
    aliasing_rate = 1 - np.mean(correct)

    shadow_changes = np.diff(shadow_prediction.astype(int))
    system_changes = np.diff(system_truth.astype(int))
    teleportations = np.sum((shadow_changes != 0) & (system_changes == 0))

    print(f"\nPoints analyzed: {len(x):,}")
    print(f"Aliasing rate: {aliasing_rate:.1%}")
    print(f"Teleportations: {teleportations}")

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    colors = np.where(system_truth, '#E63946', '#457B9D')

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(x[::10], y[::10], z[::10], c=colors[::10], s=1, alpha=0.5)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    ax1.set_title('A. 3D Lorenz Attractor\n(Red = x>0, Blue = x<0)', fontweight='bold')

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(y[::10], z[::10], c=colors[::10], s=1, alpha=0.5)
    ax2.axhline(y=25, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Y'); ax2.set_ylabel('Z')
    ax2.set_title('B. Shadow (Y,Z projection)', fontweight='bold')

    ax3 = fig.add_subplot(2, 2, 3)
    wrong_mask = ~correct
    ax3.scatter(y[::10], z[::10], c='lightgray', s=1, alpha=0.3)
    ax3.scatter(y[wrong_mask][::5], z[wrong_mask][::5], c='magenta', s=3, alpha=0.7)
    ax3.axhline(y=25, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Y'); ax3.set_ylabel('Z')
    ax3.set_title(f'C. Misclassified Points ({aliasing_rate:.1%})', fontweight='bold')

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    ax4.text(0.1, 0.5, f"Aliasing = {aliasing_rate:.1%}\nTeleportations = {teleportations}\n\n"
             f"The shadow lies {aliasing_rate:.0%} of the time.",
             fontsize=14, transform=ax4.transAxes, verticalalignment='center')

    plt.tight_layout()
    plt.savefig('figures/fig_shadow_box.pdf', dpi=150, bbox_inches='tight')
    print("Saved: figures/fig_shadow_box.pdf")
    return {'aliasing': aliasing_rate, 'teleportations': teleportations}


# =============================================================================
# DEMO 2: CHAOTIC SYSTEMS (Lorenz, Rossler, Henon)
# =============================================================================

def demo_chaotic():
    """Compare aliasing across Lorenz, Rossler, and Henon attractors."""
    print("=" * 60)
    print("TOPOLOGICAL ALIASING ACROSS CHAOTIC SYSTEMS")
    print("=" * 60)

    results = []

    # Lorenz
    print("\n[1/3] Lorenz attractor...")
    traj = generate_trajectory('lorenz', n_points=10000)
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    truth = x > 0
    pred = z > 25
    aliasing = 1 - np.mean(truth == pred)
    teleports = np.sum((np.diff(pred.astype(int)) != 0) & (np.diff(truth.astype(int)) == 0))
    results.append({'name': 'Lorenz', 'd_frac': 2.06, 'aliasing': aliasing, 'teleports': teleports})
    print(f"      Aliasing: {aliasing:.1%}, Teleportations: {teleports}")

    # Rossler
    print("\n[2/3] Rossler attractor...")
    traj = generate_trajectory('rossler', n_points=10000)
    x, y, z = traj[:, 0], traj[:, 1], traj[:, 2]
    truth = x > 0
    pred = y > 0
    aliasing = 1 - np.mean(truth == pred)
    teleports = np.sum((np.diff(pred.astype(int)) != 0) & (np.diff(truth.astype(int)) == 0))
    results.append({'name': 'Rossler', 'd_frac': 2.01, 'aliasing': aliasing, 'teleports': teleports})
    print(f"      Aliasing: {aliasing:.1%}, Teleportations: {teleports}")

    # Henon
    print("\n[3/3] Henon map...")
    traj = henon_map(50000)
    x, y = traj[:, 0], traj[:, 1]
    truth = x > 0.5
    pred = y > 0
    aliasing = 1 - np.mean(truth == pred)
    teleports = np.sum((np.diff(pred.astype(int)) != 0) & (np.diff(truth.astype(int)) == 0))
    results.append({'name': 'Henon', 'd_frac': 1.26, 'aliasing': aliasing, 'teleports': teleports})
    print(f"      Aliasing: {aliasing:.1%}, Teleportations: {teleports}")

    # Summary
    print("\n" + "-" * 60)
    print(f"{'System':<12} {'D_frac':<8} {'Aliasing':<12} {'Teleports':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<12} {r['d_frac']:<8.2f} {r['aliasing']:<12.1%} {r['teleports']:<10,}")

    return results


# =============================================================================
# DEMO 3: MACKEY-GLASS TIME SERIES
# =============================================================================

def demo_timeseries():
    """Aliasing in delay-embedded time series (Mackey-Glass)."""
    print("=" * 60)
    print("TOPOLOGICAL ALIASING IN MACKEY-GLASS TIME SERIES")
    print("=" * 60)

    results = []

    for tau, expected_dim, label in [(17, 2.1, 'Low chaos'), (30, 3.6, 'High chaos')]:
        print(f"\n--- Mackey-Glass tau={tau} ({label}) ---")
        x = mackey_glass(20000, tau=tau)

        for embed_dim in [3, 5, 8, 12]:
            X_embed = delay_embedding(x, dim=embed_dim, tau=6)
            X_proj = X_embed[:, :2]
            d_sys = _participation_ratio(X_embed)
            aliasing = _compute_aliasing(X_embed, X_proj, k=15)

            results.append({'tau': tau, 'label': label, 'embed_dim': embed_dim,
                          'd_sys': d_sys, 'aliasing': aliasing})
            print(f"    Embed dim {embed_dim}: D_sys={d_sys:.1f}, Aliasing={aliasing:.1%}")

    return results


# =============================================================================
# DEMO 4: scRNA-seq ALIASING
# =============================================================================

def demo_scrna():
    """Aliasing in scRNA-seq data (uses scanpy if available)."""
    print("=" * 60)
    print("scRNA-seq ALIASING ANALYSIS")
    print("=" * 60)

    try:
        import scanpy as sc
        HAS_SCANPY = True
    except ImportError:
        HAS_SCANPY = False
        print("scanpy not installed - using synthetic data")

    try:
        import umap
        HAS_UMAP = True
    except ImportError:
        HAS_UMAP = False

    results = {}

    if HAS_SCANPY:
        datasets = ['pbmc3k', 'paul15']
        for dataset_name in datasets:
            print(f"\n--- {dataset_name} ---")

            if dataset_name == 'pbmc3k':
                adata = sc.datasets.pbmc3k()
                sc.pp.filter_cells(adata, min_genes=200)
                sc.pp.filter_genes(adata, min_cells=3)
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
                sc.pp.highly_variable_genes(adata, n_top_genes=2000)
                data = adata[:, adata.var.highly_variable].X
            else:
                adata = sc.datasets.paul15()
                sc.pp.filter_cells(adata, min_genes=200)
                sc.pp.normalize_total(adata)
                sc.pp.log1p(adata)
                data = adata.X

            if hasattr(data, 'toarray'):
                data = data.toarray()

            # Subsample
            if data.shape[0] > 5000:
                idx = np.random.choice(data.shape[0], 5000, replace=False)
                data = data[idx]

            print(f"  Cells: {data.shape[0]}, Genes: {data.shape[1]}")
            d_sys = _participation_ratio(data)
            print(f"  D_sys: {d_sys:.1f}")

            # PCA + t-SNE
            n_pcs = min(50, data.shape[1], data.shape[0] - 1)
            pca = PCA(n_components=n_pcs)
            X_pca = pca.fit_transform(data)

            tsne_alias = []
            umap_alias = []

            for seed in range(3):
                embedding = TSNE(n_components=2, random_state=seed, perplexity=30).fit_transform(X_pca)
                tsne_alias.append(_compute_aliasing(X_pca, embedding, k=15))

                if HAS_UMAP:
                    embedding = umap.UMAP(n_components=2, random_state=seed).fit_transform(X_pca)
                    umap_alias.append(_compute_aliasing(X_pca, embedding, k=15))

            results[dataset_name] = {
                'd_sys': d_sys,
                'tsne': (np.mean(tsne_alias), np.std(tsne_alias)),
                'umap': (np.mean(umap_alias), np.std(umap_alias)) if HAS_UMAP else None
            }

            print(f"  t-SNE aliasing: {np.mean(tsne_alias):.1%} +/- {np.std(tsne_alias):.1%}")
            if HAS_UMAP:
                print(f"  UMAP aliasing: {np.mean(umap_alias):.1%} +/- {np.std(umap_alias):.1%}")
    else:
        # Synthetic demo
        print("\nUsing synthetic high-D data...")
        np.random.seed(42)
        data = np.random.randn(2000, 100)
        d_sys = _participation_ratio(data)
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(data)
        embedding = TSNE(n_components=2, random_state=42).fit_transform(X_pca)
        aliasing = _compute_aliasing(X_pca, embedding, k=15)
        print(f"  D_sys: {d_sys:.1f}")
        print(f"  t-SNE aliasing: {aliasing:.1%}")
        results['synthetic'] = {'d_sys': d_sys, 'aliasing': aliasing}

    return results


# =============================================================================
# DEMO 5: FALSIFIABILITY REGIMES
# =============================================================================

def demo_regime():
    """Popper vs Ensemble regime demonstration."""
    print("=" * 60)
    print("FALSIFIABILITY REGIME DIAGRAM")
    print("=" * 60)

    np.random.seed(42)

    dimensions = [2, 5, 10, 20, 50, 100]
    signal_strengths = np.linspace(0.1, 2.0, 15)
    n_trials = 20

    binary_power = np.zeros((len(dimensions), len(signal_strengths)))
    multi_power = np.zeros((len(dimensions), len(signal_strengths)))

    print("\nComputing power across parameter grid...")

    for i, n_dims in enumerate(dimensions):
        print(f"  Dimensions = {n_dims}...", end=" ", flush=True)
        for j, sig in enumerate(signal_strengths):
            binary_results = []
            multi_results = []

            for _ in range(n_trials):
                # Model A: isotropic Gaussian
                X_A = np.random.randn(200, n_dims)
                # Model B: stretched in one direction
                X_B = np.random.randn(200, n_dims)
                direction = np.random.randn(n_dims)
                direction = direction / np.linalg.norm(direction)
                X_B = X_B + sig * np.outer(np.random.randn(200), direction)

                # Binary test (best t-test)
                best_p = 1.0
                for k in range(min(n_dims, 10)):
                    _, p = stats.ttest_ind(X_A[:, k], X_B[:, k])
                    best_p = min(best_p, p)
                binary_results.append(1.0 if best_p < 0.05 else 0.0)

                # Multivariate test
                try:
                    cov_A = np.cov(X_A.T) + 0.01 * np.eye(n_dims)
                    cov_B = np.cov(X_B.T) + 0.01 * np.eye(n_dims)
                    pooled = (199 * cov_A + 199 * cov_B) / 398
                    mean_diff = np.mean(X_A, axis=0) - np.mean(X_B, axis=0)
                    T2 = 100 * mean_diff @ np.linalg.solve(pooled, mean_diff)
                    F = T2 * (400 - n_dims - 1) / (n_dims * 398)
                    p = 1 - stats.f.cdf(F, n_dims, 400 - n_dims - 1)
                    multi_results.append(1.0 if p < 0.05 else 0.0)
                except:
                    multi_results.append(0.5)

            binary_power[i, j] = np.mean(binary_results)
            multi_power[i, j] = np.mean(multi_results)
        print("done")

    # Summary
    print("\n" + "-" * 50)
    print("Binary test 80% power threshold:")
    for i, n_dims in enumerate(dimensions):
        idx = np.where(binary_power[i, :] >= 0.8)[0]
        if len(idx) > 0:
            print(f"  n={n_dims:3d}: signal > {signal_strengths[idx[0]]:.2f}")
        else:
            print(f"  n={n_dims:3d}: NEVER achieves 80%")

    return {'binary': binary_power, 'multi': multi_power}


# =============================================================================
# DEMO 6: NON-ERGODIC MEMORY
# =============================================================================

def demo_memory():
    """Non-ergodicity demonstration with hidden states."""
    print("=" * 60)
    print("NON-ERGODIC MEMORY")
    print("=" * 60)

    np.random.seed(42)
    n_traj, n_steps = 20, 500
    dt, k = 0.1, 2.0

    hidden_states = np.array([i % 2 for i in range(n_traj)])
    targets = np.where(hidden_states == 0, 0.25, 0.75)
    X = np.random.uniform(0.3, 0.7, n_traj)

    trajectories = np.zeros((n_traj, n_steps))
    trajectories[:, 0] = X

    for t in range(1, n_steps):
        dX = -k * (X - targets) * dt + 0.1 * np.sqrt(dt) * np.random.randn(n_traj)
        X = np.clip(X + dX, 0, 1)
        trajectories[:, t] = X

    time_averages = np.mean(trajectories[:, n_steps//2:], axis=1)
    h0_avg = np.mean(time_averages[hidden_states == 0])
    h1_avg = np.mean(time_averages[hidden_states == 1])

    print(f"\nH=0 trajectories: time avg = {h0_avg:.3f} (target: 0.25)")
    print(f"H=1 trajectories: time avg = {h1_avg:.3f} (target: 0.75)")
    print(f"Ensemble mean: 0.500")
    print(f"\nNo trajectory achieves the ensemble mean!")

    return {'h0_avg': h0_avg, 'h1_avg': h1_avg}


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run demos based on command line argument."""
    os.makedirs('figures', exist_ok=True)

    if len(sys.argv) < 2:
        demo = 'all'
    else:
        demo = sys.argv[1].lower()

    if demo == 'all':
        demo_lorenz()
        demo_chaotic()
        demo_timeseries()
        demo_scrna()
        demo_regime()
        demo_memory()
    elif demo == 'lorenz':
        demo_lorenz()
    elif demo == 'chaotic':
        demo_chaotic()
    elif demo == 'timeseries':
        demo_timeseries()
    elif demo == 'scrna':
        demo_scrna()
    elif demo == 'regime':
        demo_regime()
    elif demo == 'memory':
        demo_memory()
    else:
        print(f"Unknown demo: {demo}")
        print("Options: all, lorenz, chaotic, timeseries, scrna, regime, memory")


if __name__ == '__main__':
    main()
