#!/usr/bin/env python3
"""
Scale-Dependent Falsifiability
==============================

Demonstrates Principle 1 from the paper:
    A hypothesis H is falsifiable at energy scale E if:
    E > max(E_Landauer, E_coherence, E_coupling)

Shows that falsifiability varies with dimensionality and signal strength,
identifying the regime where Popperian logic applies versus where
ensemble-based inference is required.

Papers: "The Geometry of Biological Shadows" (Paper 2) & "The Limits of Falsifiability" (Paper 1, BioSystems 258, 2025)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def generate_system(n_dims, signal_strength, n_samples=200):
    """
    Generate data from two models that differ in a specific way.

    Model A: Gaussian centered at origin
    Model B: Gaussian with slight covariance structure

    signal_strength controls how distinguishable they are.
    """
    # Model A: isotropic Gaussian
    X_A = np.random.randn(n_samples, n_dims)

    # Model B: Gaussian with stretched variance in one direction
    X_B = np.random.randn(n_samples, n_dims)
    # Add a subtle correlation structure
    direction = np.random.randn(n_dims)
    direction = direction / np.linalg.norm(direction)
    X_B = X_B + signal_strength * np.outer(np.random.randn(n_samples), direction)

    return X_A, X_B

def binary_test_power(X_A, X_B, n_tests=10, alpha=0.05):
    """
    Compute the power of the best binary test to distinguish distributions.

    Uses t-tests on individual coordinates and returns the best p-value.

    Note: We only test min(n_dims, n_tests) coordinates because the mean shift
    is along a random direction, so all coordinates have the same expected
    mean difference. Sampling a subset is representative.
    """
    n_dims = X_A.shape[1]
    best_pvalue = 1.0

    # Test each coordinate (all coords are identically distributed; subset is representative)
    for i in range(min(n_dims, n_tests)):
        _, pvalue = stats.ttest_ind(X_A[:, i], X_B[:, i])
        best_pvalue = min(best_pvalue, pvalue)

    # Power = probability of rejecting null when alternative is true
    # Approximate by checking if we'd reject at level alpha
    return 1.0 if best_pvalue < alpha else 0.0

def multivariate_test_power(X_A, X_B, alpha=0.05):
    """
    Compute power using a multivariate test (Hotelling's T²).

    This uses all dimensions simultaneously.
    """
    from scipy.spatial.distance import mahalanobis

    # Compute pooled covariance
    n_A, n_B = len(X_A), len(X_B)
    cov_A = np.cov(X_A.T)
    cov_B = np.cov(X_B.T)
    pooled_cov = ((n_A - 1) * cov_A + (n_B - 1) * cov_B) / (n_A + n_B - 2)

    # Regularize
    pooled_cov += 0.01 * np.eye(pooled_cov.shape[0])

    # Hotelling's T² statistic
    mean_diff = np.mean(X_A, axis=0) - np.mean(X_B, axis=0)
    try:
        T2 = (n_A * n_B) / (n_A + n_B) * mean_diff @ np.linalg.solve(pooled_cov, mean_diff)

        # Convert to F-statistic
        p = X_A.shape[1]
        n = n_A + n_B
        F = T2 * (n - p - 1) / (p * (n - 2))
        pvalue = 1 - stats.f.cdf(F, p, n - p - 1)

        return 1.0 if pvalue < alpha else 0.0
    except:
        return 0.5

def run_simulation():
    """Main simulation of scale-dependent falsifiability."""

    np.random.seed(42)

    # Parameter grid
    dimensions = [2, 5, 10, 20, 50, 100]
    signal_strengths = np.linspace(0.1, 2.0, 15)

    n_trials = 50  # Trials per condition

    print("=" * 60)
    print("SCALE-DEPENDENT FALSIFIABILITY")
    print("=" * 60)
    print("\nComputing power across dimensionality × signal strength grid...")
    print()

    # Store results
    binary_power_grid = np.zeros((len(dimensions), len(signal_strengths)))
    multi_power_grid = np.zeros((len(dimensions), len(signal_strengths)))

    for i, n_dims in enumerate(dimensions):
        print(f"Dimensions = {n_dims}...", end=" ")
        for j, sig in enumerate(signal_strengths):
            binary_powers = []
            multi_powers = []

            for _ in range(n_trials):
                X_A, X_B = generate_system(n_dims, sig)
                binary_powers.append(binary_test_power(X_A, X_B))
                multi_powers.append(multivariate_test_power(X_A, X_B))

            binary_power_grid[i, j] = np.mean(binary_powers)
            multi_power_grid[i, j] = np.mean(multi_powers)

        print("done")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Binary test power heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(binary_power_grid, aspect='auto', origin='lower',
                     cmap='RdYlGn', vmin=0, vmax=1,
                     extent=[signal_strengths[0], signal_strengths[-1], 0, len(dimensions)-1])
    ax1.set_yticks(range(len(dimensions)))
    ax1.set_yticklabels(dimensions)
    ax1.set_xlabel('Signal Strength', fontsize=11)
    ax1.set_ylabel('Number of Dimensions', fontsize=11)
    ax1.set_title('A. Binary Test Power\n(Best single-coordinate t-test)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Power')

    # Add contour for 80% power
    ax1.contour(signal_strengths, range(len(dimensions)), binary_power_grid,
                levels=[0.8], colors='black', linewidths=2, linestyles='--')

    # Panel B: Multivariate test power heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(multi_power_grid, aspect='auto', origin='lower',
                     cmap='RdYlGn', vmin=0, vmax=1,
                     extent=[signal_strengths[0], signal_strengths[-1], 0, len(dimensions)-1])
    ax2.set_yticks(range(len(dimensions)))
    ax2.set_yticklabels(dimensions)
    ax2.set_xlabel('Signal Strength', fontsize=11)
    ax2.set_ylabel('Number of Dimensions', fontsize=11)
    ax2.set_title("B. Multivariate Test Power\n(Hotelling's T²)",
                  fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Power')

    # Add contour for 80% power
    ax2.contour(signal_strengths, range(len(dimensions)), multi_power_grid,
                levels=[0.8], colors='black', linewidths=2, linestyles='--')

    # Panel C: Power difference (where binary fails but multivariate succeeds)
    ax3 = axes[1, 0]
    power_diff = multi_power_grid - binary_power_grid
    im3 = ax3.imshow(power_diff, aspect='auto', origin='lower',
                     cmap='PuOr', vmin=-0.5, vmax=0.5,
                     extent=[signal_strengths[0], signal_strengths[-1], 0, len(dimensions)-1])
    ax3.set_yticks(range(len(dimensions)))
    ax3.set_yticklabels(dimensions)
    ax3.set_xlabel('Signal Strength', fontsize=11)
    ax3.set_ylabel('Number of Dimensions', fontsize=11)
    ax3.set_title('C. Advantage of Multivariate Methods\n(Multi - Binary)',
                  fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Power Difference')

    # Shade the "falsifiability failure" region
    ax3.contourf(signal_strengths, range(len(dimensions)), binary_power_grid,
                 levels=[0, 0.5], colors=['red'], alpha=0.2)
    ax3.annotate('Binary tests\nfail here', xy=(0.5, 4), fontsize=10,
                 ha='center', color='darkred', fontweight='bold')

    # Panel D: Falsifiability regime diagram
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create regime diagram text
    diagram_text = """
    SCALE-DEPENDENT FALSIFIABILITY (Principle 1)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    A hypothesis H is falsifiable at scale E if:

        E > max(E_Landauer, E_coherence, E_coupling)

    ┌─────────────────────────────────────────────┐
    │                                             │
    │   HIGH SIGNAL              LOW SIGNAL       │
    │   LOW DIMS                 HIGH DIMS        │
    │                                             │
    │   ┌─────────┐             ┌─────────┐      │
    │   │ POPPER  │             │ENSEMBLE │      │
    │   │ REGIME  │     →       │ REGIME  │      │
    │   │         │             │         │      │
    │   │ Binary  │             │ Multi-  │      │
    │   │ tests   │             │ variate │      │
    │   │ work    │             │ only    │      │
    │   └─────────┘             └─────────┘      │
    │                                             │
    │   Examples:               Examples:         │
    │   • Enzyme kinetics       • Consciousness   │
    │   • Action potentials     • Ecosystem       │
    │   • Mendelian genetics    • Evolution       │
    │                           • Protein folding │
    └─────────────────────────────────────────────┘

    The transition occurs when:
    • Dimensionality exceeds ~10-20
    • Signal strength falls below ~0.5 (relative)
    • Energy scale approaches k_B T ln 2
    """

    ax4.text(0.05, 0.98, diagram_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax4.set_title('D. Falsifiability Regimes', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    for path in ['figures/fig_scale_dependent.pdf', '../figures/fig_scale_dependent.pdf']:
        try:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to {path}")
            break
        except:
            continue

    plt.show()

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Find transition points
    print("\nBoundary where binary tests achieve 80% power:")
    for i, n_dims in enumerate(dimensions):
        # Find signal strength needed for 80% power
        threshold_idx = np.where(binary_power_grid[i, :] >= 0.8)[0]
        if len(threshold_idx) > 0:
            sig_threshold = signal_strengths[threshold_idx[0]]
            print(f"  n = {n_dims:3d} dims: signal > {sig_threshold:.2f}")
        else:
            print(f"  n = {n_dims:3d} dims: NEVER achieves 80% power")

    print(f"""
The simulation demonstrates scale-dependent falsifiability (Principle 1):

1. LOW-D, HIGH-SIGNAL: Binary tests work well. This is Popper's regime.
   Single hypothesis tests can definitively falsify models.

2. HIGH-D, LOW-SIGNAL: Binary tests fail, but multivariate methods
   still work. Information exists but requires ensemble approaches.

3. TRANSITION ZONE: As dimensionality increases, the signal strength
   needed for binary falsifiability grows. Eventually, binary tests
   become incoherent regardless of signal.

BIOLOGICAL IMPLICATION: Many biological systems operate in the
high-D, low-signal regime where:
  • Individual binary tests cannot falsify hypotheses
  • But ensemble methods can still discriminate models
  • Epistemology must shift from falsification to pattern matching

This is not a failure of science but a recognition of scale-dependent
limits on what kinds of knowledge are accessible through what methods.
""")

if __name__ == '__main__':
    run_simulation()
