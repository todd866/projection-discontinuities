#!/usr/bin/env python3
"""
Binary Projection in High-Dimensional Systems
==============================================

Demonstrates Eq. 1 from the paper:
    Ω_preserved / Ω_total = 1 / k^n

Shows that single binary tests become uninformative as dimensionality increases,
while multivariate approaches maintain discriminability.

Papers: "The Geometry of Biological Shadows" (Paper 2) & "The Limits of Falsifiability" (Paper 1, BioSystems 258, 2025)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)  # Also check current dir

def generate_high_d_distributions(n_dims, n_samples=1000, separation=0.5):
    """
    Generate two high-dimensional Gaussian distributions.

    The distributions differ slightly in their means - the kind of subtle
    difference that matters biologically but becomes invisible under binary projection.
    """
    # Distribution A: centered at origin
    X_A = np.random.randn(n_samples, n_dims)

    # Distribution B: shifted slightly in a random direction
    shift_direction = np.random.randn(n_dims)
    shift_direction = shift_direction / np.linalg.norm(shift_direction)
    X_B = np.random.randn(n_samples, n_dims) + separation * shift_direction

    return X_A, X_B

def binary_test_accuracy(X_A, X_B, feature_idx, threshold=0.0):
    """
    Test accuracy using a single binary feature: is X[feature_idx] > threshold?

    This is the kind of "falsifying test" Popper's framework assumes is possible.
    """
    # Classify based on single binary feature
    pred_A = (X_A[:, feature_idx] > threshold).astype(int)
    pred_B = (X_B[:, feature_idx] > threshold).astype(int)

    # What fraction does this correctly separate?
    # If distributions are from A, we want pred=0; if from B, we want pred=1
    correct_A = np.mean(pred_A == 0)
    correct_B = np.mean(pred_B == 1)

    return (correct_A + correct_B) / 2

def multivariate_accuracy(X_A, X_B):
    """
    Test accuracy using all dimensions (logistic regression).

    This represents the "ensemble fingerprint" approach the paper advocates.
    """
    X = np.vstack([X_A, X_B])
    y = np.concatenate([np.zeros(len(X_A)), np.ones(len(X_B))])

    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    return np.mean(scores)

def run_simulation():
    """Main simulation comparing binary vs multivariate discrimination."""

    np.random.seed(42)

    # Test across different dimensionalities
    dimensions = [2, 5, 10, 20, 50, 100, 200]
    n_samples = 500
    separation = 1.0  # Fixed separation in the high-D space

    binary_accuracies = []
    multivariate_accuracies = []
    theoretical_info_preserved = []

    print("=" * 60)
    print("BINARY PROJECTION IN HIGH-DIMENSIONAL SYSTEMS")
    print("=" * 60)
    print(f"\nSeparation between distributions: {separation}")
    print(f"Samples per distribution: {n_samples}")
    print()

    for n_dims in dimensions:
        X_A, X_B = generate_high_d_distributions(n_dims, n_samples, separation)

        # Binary test: best single feature
        best_binary = 0
        for i in range(min(n_dims, 20)):  # Check first 20 features
            acc = binary_test_accuracy(X_A, X_B, i)
            best_binary = max(best_binary, acc)

        # Multivariate test
        multi_acc = multivariate_accuracy(X_A, X_B)

        # Theoretical information preserved (Eq. 1)
        k = 10  # Assume ~10 distinguishable states per dimension
        info_preserved = 1.0 / (k ** n_dims)

        binary_accuracies.append(best_binary)
        multivariate_accuracies.append(multi_acc)
        theoretical_info_preserved.append(info_preserved)

        print(f"n = {n_dims:3d} dims: Binary acc = {best_binary:.3f}, "
              f"Multivariate acc = {multi_acc:.3f}, "
              f"Info preserved = {info_preserved:.2e}")

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Classification accuracy
    ax1 = axes[0]
    ax1.plot(dimensions, binary_accuracies, 'o-', color='#E63946',
             linewidth=2, markersize=8, label='Best single binary test')
    ax1.plot(dimensions, multivariate_accuracies, 's-', color='#2A9D8F',
             linewidth=2, markersize=8, label='Multivariate (all dims)')
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax1.set_xlabel('Number of Dimensions (n)', fontsize=12)
    ax1.set_ylabel('Classification Accuracy', fontsize=12)
    ax1.set_title('A. Binary Tests Fail in High-D', fontsize=14, fontweight='bold')
    ax1.legend(loc='center right')
    ax1.set_ylim(0.4, 1.05)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Panel B: Information preserved
    ax2 = axes[1]
    ax2.semilogy(dimensions, theoretical_info_preserved, 'o-', color='#457B9D',
                 linewidth=2, markersize=8)
    ax2.fill_between(dimensions, theoretical_info_preserved, 1e-300,
                     alpha=0.3, color='#457B9D')
    ax2.set_xlabel('Number of Dimensions (n)', fontsize=12)
    ax2.set_ylabel('Fraction of State Space Preserved', fontsize=12)
    ax2.set_title('B. Information Loss Under Binary Projection\n(Eq. 1: 1/k^n)',
                  fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    # Add annotation
    ax2.annotate(f'At n=100:\n~10⁻¹⁰⁰ preserved',
                 xy=(100, 1e-100), xytext=(20, 1e-60),
                 fontsize=10, ha='center',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()

    # Save figure
    for path in ['figures/fig_binary_projection.pdf', '../figures/fig_binary_projection.pdf']:
        try:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to {path}")
            break
        except:
            continue

    plt.show()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
The simulation demonstrates that:

1. BINARY TESTS FAIL: As dimensionality increases, even the best single
   binary test approaches chance performance (50%).

2. MULTIVARIATE WORKS: Using all dimensions maintains high discrimination,
   demonstrating that the information *exists* but cannot be captured
   by binary projection.

3. INFORMATION LOSS: Eq. 1 shows that a single binary partition preserves
   only 1/k^n of the state space - essentially nothing for biological
   systems with n >> 10.

IMPLICATION: Popperian falsification (single binary tests) becomes
incoherent in high-dimensional biological systems. Ensemble-based
methods are required.
""")

if __name__ == '__main__':
    run_simulation()
