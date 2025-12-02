#!/usr/bin/env python3
"""
DEMO: THE FALSIFIABILITY REGIME DIAGRAM
========================================

THE QUESTION:
Under what conditions does Popperian falsification work?
When does it break down?

THE CONCEPT:
Falsifiability is SCALE-DEPENDENT. It works in some regimes and fails in others.

The two key parameters:
- DIMENSIONALITY: How many effective degrees of freedom?
- SIGNAL STRENGTH: How strong is the effect relative to noise?

THE REGIMES:
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
└─────────────────────────────────────────────┘

THE IMPLICATION:
Most of biology operates in the ENSEMBLE regime, where binary hypothesis
testing is mathematically incoherent. We need different epistemology.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def generate_distinguishable_data(n_dims, signal_strength, n_samples=200):
    """
    Generate data from two models that differ by a specified amount.

    Model A: Isotropic Gaussian at origin
    Model B: Gaussian with extra variance along a random direction

    signal_strength controls how distinguishable they are.
    """
    # Model A: isotropic Gaussian
    X_A = np.random.randn(n_samples, n_dims)

    # Model B: Gaussian with stretched variance in one direction
    X_B = np.random.randn(n_samples, n_dims)
    direction = np.random.randn(n_dims)
    direction = direction / np.linalg.norm(direction)
    X_B = X_B + signal_strength * np.outer(np.random.randn(n_samples), direction)

    return X_A, X_B


def binary_test_power(X_A, X_B, n_tests=10, alpha=0.05):
    """
    Compute power of the best binary test (t-test on individual coordinates).

    This is what Popperian falsification assumes you can do:
    find ONE coordinate that distinguishes the models.
    """
    n_dims = X_A.shape[1]
    best_pvalue = 1.0

    for i in range(min(n_dims, n_tests)):
        _, pvalue = stats.ttest_ind(X_A[:, i], X_B[:, i])
        best_pvalue = min(best_pvalue, pvalue)

    return 1.0 if best_pvalue < alpha else 0.0


def multivariate_test_power(X_A, X_B, alpha=0.05):
    """
    Compute power using Hotelling's T² (multivariate test).

    This uses ALL dimensions simultaneously - ensemble approach.
    """
    n_A, n_B = len(X_A), len(X_B)
    cov_A = np.cov(X_A.T)
    cov_B = np.cov(X_B.T)
    pooled_cov = ((n_A - 1) * cov_A + (n_B - 1) * cov_B) / (n_A + n_B - 2)

    # Regularize
    pooled_cov += 0.01 * np.eye(pooled_cov.shape[0])

    mean_diff = np.mean(X_A, axis=0) - np.mean(X_B, axis=0)

    try:
        T2 = (n_A * n_B) / (n_A + n_B) * mean_diff @ np.linalg.solve(pooled_cov, mean_diff)
        p = X_A.shape[1]
        n = n_A + n_B
        F = T2 * (n - p - 1) / (p * (n - 2))
        pvalue = 1 - stats.f.cdf(F, p, n - p - 1)
        return 1.0 if pvalue < alpha else 0.0
    except:
        return 0.5


def run_regime_analysis():
    """Generate the falsifiability regime diagram."""

    print("=" * 60)
    print("FALSIFIABILITY REGIME DIAGRAM")
    print("When does Popperian falsification work?")
    print("=" * 60)

    np.random.seed(42)

    # Parameter grid
    dimensions = [2, 5, 10, 20, 50, 100]
    signal_strengths = np.linspace(0.1, 2.0, 15)
    n_trials = 30

    print("\nComputing power across dimensionality × signal strength grid...")

    # Store results
    binary_power = np.zeros((len(dimensions), len(signal_strengths)))
    multi_power = np.zeros((len(dimensions), len(signal_strengths)))

    for i, n_dims in enumerate(dimensions):
        print(f"  Dimensions = {n_dims}...", end=" ", flush=True)
        for j, sig in enumerate(signal_strengths):
            binary_results = []
            multi_results = []

            for _ in range(n_trials):
                X_A, X_B = generate_distinguishable_data(n_dims, sig)
                binary_results.append(binary_test_power(X_A, X_B))
                multi_results.append(multivariate_test_power(X_A, X_B))

            binary_power[i, j] = np.mean(binary_results)
            multi_power[i, j] = np.mean(multi_results)

        print("done")

    # Find the transition boundary (80% power for binary tests)
    print("\n" + "-" * 40)
    print("TRANSITION BOUNDARY (Binary tests achieve 80% power):")
    print("-" * 40)

    for i, n_dims in enumerate(dimensions):
        threshold_idx = np.where(binary_power[i, :] >= 0.8)[0]
        if len(threshold_idx) > 0:
            sig_threshold = signal_strengths[threshold_idx[0]]
            print(f"  n = {n_dims:3d} dims: signal > {sig_threshold:.2f}")
        else:
            print(f"  n = {n_dims:3d} dims: NEVER achieves 80% power (Ensemble only)")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Binary test power
    ax = axes[0, 0]
    im = ax.imshow(binary_power, aspect='auto', origin='lower',
                   cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[signal_strengths[0], signal_strengths[-1], 0, len(dimensions)-1])
    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels(dimensions)
    ax.set_xlabel('Signal Strength')
    ax.set_ylabel('Number of Dimensions')
    ax.set_title('A. Binary Test Power\n(Popperian Falsification)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Power')
    ax.contour(signal_strengths, range(len(dimensions)), binary_power,
               levels=[0.8], colors='black', linewidths=2, linestyles='--')

    # Panel B: Multivariate test power
    ax = axes[0, 1]
    im = ax.imshow(multi_power, aspect='auto', origin='lower',
                   cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[signal_strengths[0], signal_strengths[-1], 0, len(dimensions)-1])
    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels(dimensions)
    ax.set_xlabel('Signal Strength')
    ax.set_ylabel('Number of Dimensions')
    ax.set_title("B. Multivariate Test Power\n(Ensemble Methods)", fontweight='bold')
    plt.colorbar(im, ax=ax, label='Power')
    ax.contour(signal_strengths, range(len(dimensions)), multi_power,
               levels=[0.8], colors='black', linewidths=2, linestyles='--')

    # Panel C: Advantage of multivariate
    ax = axes[1, 0]
    power_diff = multi_power - binary_power
    im = ax.imshow(power_diff, aspect='auto', origin='lower',
                   cmap='PuOr', vmin=-0.5, vmax=0.5,
                   extent=[signal_strengths[0], signal_strengths[-1], 0, len(dimensions)-1])
    ax.set_yticks(range(len(dimensions)))
    ax.set_yticklabels(dimensions)
    ax.set_xlabel('Signal Strength')
    ax.set_ylabel('Number of Dimensions')
    ax.set_title('C. Ensemble Advantage\n(Multivariate - Binary)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Power Difference')

    # Shade binary failure region
    ax.contourf(signal_strengths, range(len(dimensions)), binary_power,
                levels=[0, 0.5], colors=['red'], alpha=0.2)
    ax.text(0.6, 4.5, 'Binary tests\nFAIL here', fontsize=10,
            ha='center', color='darkred', fontweight='bold')

    # Panel D: Regime diagram
    ax = axes[1, 1]
    ax.axis('off')

    diagram_text = """
    THE FALSIFIABILITY REGIMES
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Hypothesis H is falsifiable at scale E if:
        E > max(E_Landauer, E_coherence, E_coupling)

    ┌────────────────────────────────────────┐
    │                                        │
    │   POPPER REGIME         ENSEMBLE REGIME│
    │   (Low-D, High-Signal)  (High-D, Low-S)│
    │                                        │
    │   Binary tests work     Binary tests   │
    │   Single hypothesis     fail           │
    │   can be falsified      Must use       │
    │                         multivariate   │
    │   Examples:             Examples:      │
    │   • Mendelian genes     • Cell types   │
    │   • Drug response       • Consciousness│
    │   • Enzyme kinetics     • Ecosystems   │
    │                                        │
    └────────────────────────────────────────┘

    IMPLICATION FOR BIOLOGY:

    Most biological systems (D_sys ~ 10-20) operate
    in the ENSEMBLE regime where:

    • Single binary tests cannot falsify hypotheses
    • But multivariate methods CAN discriminate
    • Epistemology must shift from falsification
      to pattern matching and model comparison
    """

    ax.text(0.02, 0.98, diagram_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax.set_title('D. Regime Summary', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig_regime_diagram.pdf', dpi=150, bbox_inches='tight')
    print("\nSaved: fig_regime_diagram.pdf")

    plt.show()

    return {
        'binary_power': binary_power,
        'multi_power': multi_power,
        'dimensions': dimensions,
        'signal_strengths': signal_strengths
    }


if __name__ == '__main__':
    results = run_regime_analysis()
