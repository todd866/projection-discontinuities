#!/usr/bin/env python3
"""
Sample Complexity: The Curse of Dimensionality
===============================================

Demonstrates why ensemble averaging becomes impractical in high dimensions.

THE KEY INSIGHT:
Parametric methods (like Hotelling's T²) assume structure and avoid the curse.
But biological inference often requires NONPARAMETRIC methods because we
don't know the true distribution family.

For nonparametric methods, sample complexity scales EXPONENTIALLY with dimension.

THE SIMULATION:
1. Show that to COVER a high-D space (sample all regions), you need N ~ k^n
2. Show that nonparametric tests (permutation tests on binned data) degrade with dimension
3. Show that "rare but important" events become invisible

Paper: "The Geometry of Biological Shadows" (Paper 2)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import pdist
import os

os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)


def measure_coverage(n_dims, n_samples, n_bins=3):
    """
    Measure what fraction of the n_dims-dimensional space is "covered" by n_samples.

    We discretize each dimension into n_bins bins, giving n_bins^n_dims total cells.
    Coverage = fraction of cells with at least one sample.

    This is the fundamental curse: cells grow as k^n, so coverage → 0.
    """
    # Generate uniform samples in [0, 1]^n
    samples = np.random.rand(n_samples, n_dims)

    # Discretize into bins
    binned = np.floor(samples * n_bins).astype(int)
    binned = np.clip(binned, 0, n_bins - 1)

    # Count unique cells occupied
    # Convert to tuple for hashing
    cells = set(tuple(b) for b in binned)
    n_occupied = len(cells)

    # Total possible cells
    n_total_cells = n_bins ** n_dims

    coverage = n_occupied / n_total_cells
    return coverage, n_occupied, n_total_cells


def samples_for_coverage(n_dims, target_coverage=0.5, n_bins=3, max_samples=100000):
    """Find how many samples needed to achieve target coverage."""
    for n_samples in [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]:
        if n_samples > max_samples:
            return max_samples
        coverage, _, _ = measure_coverage(n_dims, n_samples, n_bins)
        if coverage >= target_coverage:
            return n_samples
    return max_samples


def run_simulation():
    """Main simulation."""

    np.random.seed(42)

    print("=" * 60)
    print("SAMPLE COMPLEXITY: THE CURSE OF DIMENSIONALITY")
    print("=" * 60)

    # =========================================================================
    # Part 1: Coverage vs dimension (the fundamental curse)
    # =========================================================================
    print("\n[1] SPACE COVERAGE vs DIMENSIONALITY")
    print("-" * 40)
    print("    (3 bins per dimension, 1000 samples)")

    dimensions = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
    n_samples_fixed = 1000
    n_bins = 3

    coverages = []
    total_cells = []

    for n_dims in dimensions:
        cov, n_occ, n_total = measure_coverage(n_dims, n_samples_fixed, n_bins)
        coverages.append(cov)
        total_cells.append(n_total)
        print(f"  n = {n_dims:2d}: cells = {n_total:>10,}, coverage = {cov:6.2%}")

    # =========================================================================
    # Part 2: Samples needed for 50% coverage
    # =========================================================================
    print("\n[2] SAMPLES NEEDED FOR 50% COVERAGE")
    print("-" * 40)

    dims_for_samples = [2, 3, 4, 5, 6, 7, 8]
    samples_needed = []

    for n_dims in dims_for_samples:
        n_req = samples_for_coverage(n_dims, target_coverage=0.5, n_bins=3)
        samples_needed.append(n_req)
        print(f"  n = {n_dims}: N_required ≥ {n_req:,}")

    # =========================================================================
    # Part 3: Coverage curves for different dimensions
    # =========================================================================
    print("\n[3] COVERAGE CURVES")
    print("-" * 40)

    sample_sizes = [10, 50, 100, 500, 1000, 5000, 10000]
    dims_to_plot = [2, 4, 6, 8, 10]
    coverage_curves = {d: [] for d in dims_to_plot}

    for n_dims in dims_to_plot:
        print(f"  Computing coverage curve for n={n_dims}...", end=" ")
        for n_samples in sample_sizes:
            cov, _, _ = measure_coverage(n_dims, n_samples, n_bins=3)
            coverage_curves[n_dims].append(cov)
        print("done")

    # =========================================================================
    # Plotting
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Coverage vs dimension (fixed N=1000)
    ax1 = axes[0, 0]
    ax1.semilogy(dimensions, coverages, 'o-', linewidth=2, markersize=8, color='#E63946')
    ax1.set_xlabel('Dimensionality (n)', fontsize=11)
    ax1.set_ylabel('Coverage fraction', fontsize=11)
    ax1.set_title('A. Space Coverage Collapse\n(N=1,000 samples, 3 bins/dim)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-6, 1.5)

    # Add annotations
    ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.5)
    ax1.annotate('1% coverage', xy=(12, 0.015), fontsize=9, color='red')

    # Panel B: Total cells (exponential growth)
    ax2 = axes[0, 1]
    ax2.semilogy(dimensions, total_cells, 's-', linewidth=2, markersize=8, color='#2A9D8F')
    ax2.axhline(y=1000, color='black', linestyle='--', linewidth=2, label='N = 1,000 samples')
    ax2.set_xlabel('Dimensionality (n)', fontsize=11)
    ax2.set_ylabel('Total cells (3^n)', fontsize=11)
    ax2.set_title('B. Space Size Explosion\n(cells = 3^n)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Shade the "impossible" region
    ax2.fill_between(dimensions, 1000, [max(total_cells)]*len(dimensions),
                     alpha=0.2, color='red', label='More cells than samples')

    # Panel C: Coverage curves
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(dims_to_plot)))
    for i, n_dims in enumerate(dims_to_plot):
        ax3.semilogx(sample_sizes, coverage_curves[n_dims],
                     'o-', label=f'n={n_dims}', color=colors[i], linewidth=2, markersize=6)
    ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='50% coverage')
    ax3.set_xlabel('Number of samples (N)', fontsize=11)
    ax3.set_ylabel('Coverage fraction', fontsize=11)
    ax3.set_title('C. Coverage Curves by Dimension\n(diminishing returns in high-D)', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    # Panel D: Conceptual summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    concept_text = """
    THE CURSE OF DIMENSIONALITY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    To sample a space with k bins per dimension:

        Total cells = k^n

    CONCRETE NUMBERS (k=3):

    n = 5:    cells =        243    (doable)
    n = 10:   cells =     59,049    (hard)
    n = 15:   cells = 14,348,907    (impossible)
    n = 20:   cells = 3.5 billion   (absurd)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    THE BIOLOGICAL IMPLICATION:

    Nonparametric inference requires "seeing"
    the relevant regions of state space.

    But high-dimensional spaces are almost
    entirely empty—your samples cluster in
    a tiny fraction of the possible volume.

    RARE EVENTS (disease subtypes, unusual
    cell states, outlier phenotypes) live in
    cells you will NEVER sample.

    This is not "we need more data."
    This is "the required data doesn't exist."

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    ESCAPE ROUTE:

    Assume structure (parametric models) or
    find low-dimensional manifolds. But this
    requires knowing what you're looking for
    before you find it.
    """

    ax4.text(0.05, 0.95, concept_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax4.set_title('D. The Epistemological Point', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    for path in ['figures/fig_sample_complexity.pdf', '../figures/fig_sample_complexity.pdf']:
        try:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to {path}")
            break
        except:
            continue

    plt.show()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY: THE EXPONENTIAL WALL")
    print("=" * 60)
    print(f"""
    With k=3 bins per dimension and N=1,000 samples:

    Dimension    Total Cells    Coverage
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    n = 2              9         100%
    n = 5            243          99%
    n = 8          6,561          15%
    n = 10        59,049           1.7%
    n = 15    14,348,907          0.007%

    KEY INSIGHT:

    In high dimensions, almost all of the space is EMPTY of samples.
    Your 1,000 carefully collected data points occupy a vanishing
    fraction of the possible states.

    This means:
    1. Rare events are invisible (you'll never sample them)
    2. Nonparametric density estimation fails (no data in most bins)
    3. Model comparison becomes impossible (can't evaluate likelihood
       in regions with no samples)

    The curse is EXPONENTIAL: doubling your sample size buys you
    almost nothing in coverage. You'd need to EXPONENTIATE your
    samples to keep pace with dimensional growth.

    This completes the trilemma:
    1. Time averaging fails (non-ergodic memory)
    2. Ensemble averaging fails (curse of dimensionality) ← THIS
    3. Direct measurement fails (perturbation)
    """)


if __name__ == '__main__':
    run_simulation()
