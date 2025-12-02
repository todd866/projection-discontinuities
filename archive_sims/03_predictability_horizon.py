#!/usr/bin/env python3
"""
Predictability Horizon in Chaotic Systems
==========================================

Demonstrates Eq. 3-4 from the paper:
    T_pred ≲ (1/λ) ln(L/Δx)

Shows that chaotic systems have a fundamental predictability horizon
that depends on the Lyapunov exponent and measurement precision.
At quantum-limited precision, this horizon is finite and cannot be
extended by any improvement in measurement technology.

Papers: "The Geometry of Biological Shadows" (Paper 2) & "The Limits of Falsifiability" (Paper 1, BioSystems 258, 2025)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def logistic_map(x, r=3.9):
    """
    The logistic map: x_{n+1} = r * x_n * (1 - x_n)

    For r = 3.9, this is in the chaotic regime with
    Lyapunov exponent λ ≈ 0.5 (per iteration).
    """
    return r * x * (1 - x)

def compute_lyapunov(r=3.9, n_iter=10000):
    """
    Numerically estimate the Lyapunov exponent of the logistic map.

    λ = lim_{n→∞} (1/n) Σ ln|f'(x_i)|

    For the logistic map: f'(x) = r(1 - 2x)
    """
    x = 0.5
    lyap_sum = 0

    # Transient
    for _ in range(1000):
        x = logistic_map(x, r)

    # Compute
    for _ in range(n_iter):
        lyap_sum += np.log(abs(r * (1 - 2 * x)))
        x = logistic_map(x, r)

    return lyap_sum / n_iter

def divergence_time(x0, delta0, epsilon, r=3.9, max_iter=1000):
    """
    Compute time for two trajectories to diverge beyond epsilon.

    Starting from x0 and x0 + delta0, iterate until |x1 - x2| > epsilon.
    Returns the number of iterations (discrete time).
    """
    x1 = x0
    x2 = x0 + delta0

    for t in range(max_iter):
        if abs(x1 - x2) > epsilon:
            return t
        x1 = logistic_map(x1, r)
        x2 = logistic_map(x2, r)

    return max_iter

def run_simulation():
    """Main simulation of predictability horizons."""

    np.random.seed(42)

    # Parameters
    r = 3.9  # Chaotic regime
    L = 1.0  # System size (logistic map is in [0,1])

    # Compute Lyapunov exponent
    lyapunov = compute_lyapunov(r)
    print("=" * 60)
    print("PREDICTABILITY HORIZON IN CHAOTIC SYSTEMS")
    print("=" * 60)
    print(f"\nLogistic map parameter r = {r}")
    print(f"Lyapunov exponent λ ≈ {lyapunov:.3f} per iteration")
    print()

    # Test different tolerances (epsilon)
    epsilons = np.logspace(-1, -12, 20)

    # Different initial separations (delta0)
    delta0_values = [1e-6, 1e-10, 1e-14]

    print("Testing divergence times...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Divergence time vs tolerance
    ax1 = axes[0]

    for delta0 in delta0_values:
        T_divs = []
        for eps in epsilons:
            # Average over multiple initial conditions
            T_list = []
            for _ in range(50):
                x0 = np.random.uniform(0.1, 0.9)
                T = divergence_time(x0, delta0, eps, r)
                T_list.append(T)
            T_divs.append(np.mean(T_list))

        ax1.semilogx(epsilons, T_divs, 'o-', markersize=4,
                     label=f'δ₀ = {delta0:.0e}')

    # Theoretical prediction: T = (1/λ) ln(ε/δ0)
    eps_theory = np.logspace(-1, -12, 100)
    for delta0 in delta0_values:
        T_theory = (1/lyapunov) * np.log(eps_theory / delta0)
        T_theory = np.maximum(T_theory, 0)
        ax1.semilogx(eps_theory, T_theory, '--', alpha=0.5, linewidth=1)

    ax1.set_xlabel('Tolerance ε', fontsize=11)
    ax1.set_ylabel('Divergence Time T', fontsize=11)
    ax1.set_title('A. Predictability Horizon vs Measurement Precision\n(Eq. 3-4)',
                  fontsize=12, fontweight='bold')
    ax1.legend(title='Initial separation')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # Annotate
    ax1.annotate('Coarse precision\n(always predictable)',
                 xy=(0.1, 5), fontsize=9, ha='center')
    ax1.annotate('Fine precision\n(limited horizon)',
                 xy=(1e-10, 30), fontsize=9, ha='center')

    # Panel B: Exponential divergence visualization
    ax2 = axes[1]

    # Show actual trajectory divergence
    x0 = 0.5
    delta0 = 1e-10

    x1_traj = [x0]
    x2_traj = [x0 + delta0]
    diff_traj = [delta0]

    for t in range(50):
        x1_traj.append(logistic_map(x1_traj[-1], r))
        x2_traj.append(logistic_map(x2_traj[-1], r))
        diff_traj.append(abs(x1_traj[-1] - x2_traj[-1]))

    t_axis = np.arange(len(diff_traj))

    ax2.semilogy(t_axis, diff_traj, 'b-', linewidth=2, label='Actual divergence')

    # Theoretical exponential growth
    theory = delta0 * np.exp(lyapunov * t_axis)
    ax2.semilogy(t_axis, theory, 'r--', linewidth=1.5,
                 label=f'Theory: δ₀ exp(λt), λ={lyapunov:.2f}')

    # Mark different tolerance levels
    tolerances_to_mark = [1e-6, 1e-3, 0.1]
    colors = ['green', 'orange', 'red']
    for eps, c in zip(tolerances_to_mark, colors):
        # Find crossing time
        crossing = np.where(np.array(diff_traj) > eps)[0]
        if len(crossing) > 0:
            T_cross = crossing[0]
            ax2.axhline(y=eps, color=c, linestyle=':', alpha=0.7)
            ax2.axvline(x=T_cross, color=c, linestyle=':', alpha=0.7)
            ax2.annotate(f'ε={eps:.0e}\nT={T_cross}',
                        xy=(T_cross, eps), xytext=(T_cross+3, eps*5),
                        fontsize=8, color=c)

    ax2.set_xlabel('Time (iterations)', fontsize=11)
    ax2.set_ylabel('Trajectory Separation |x₁ - x₂|', fontsize=11)
    ax2.set_title('B. Exponential Divergence of Nearby Trajectories',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(1e-12, 10)

    plt.tight_layout()

    # Save figure
    for path in ['figures/fig_predictability_horizon.pdf', '../figures/fig_predictability_horizon.pdf']:
        try:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to {path}")
            break
        except:
            continue

    plt.show()

    # Compute specific predictions
    print("\n" + "=" * 60)
    print("QUANTITATIVE PREDICTIONS")
    print("=" * 60)

    # For different measurement precisions
    precisions = {
        'Classical (mm scale)': 1e-3,
        'Optical (μm scale)': 1e-6,
        'Molecular (nm scale)': 1e-9,
        'Quantum limit (~ℏ/Δp)': 1e-15,
    }

    print(f"\nFor Lyapunov exponent λ = {lyapunov:.3f}:")
    print(f"System size L = {L}")
    print()

    for name, dx in precisions.items():
        T_pred = (1/lyapunov) * np.log(L / dx)
        print(f"{name:25s}: Δx = {dx:.0e}, T_pred ≈ {T_pred:.1f} iterations")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
The simulation demonstrates the predictability horizon (Eq. 3-4):

1. EXPONENTIAL DIVERGENCE: Nearby trajectories separate as
   |δ(t)| ~ δ₀ exp(λt), where λ = {lyapunov:.3f} is the Lyapunov exponent.

2. FINITE HORIZON: For any tolerance ε, the time until trajectories
   diverge beyond ε is bounded by:
   T_pred ≲ (1/λ) ln(L/Δx) ≈ (1/{lyapunov:.2f}) ln(1/Δx)

3. QUANTUM LIMIT: Even at quantum-limited precision (Δx ~ 10⁻¹⁵),
   the horizon is only ~{(1/lyapunov) * np.log(1/1e-15):.0f} iterations.

IMPLICATION: In chaotic biological systems, complete specification
is impossible in principle beyond T_pred. This is not a technological
limitation but a fundamental constraint from the conjunction of
chaos and quantum mechanics.

Binary falsification requires deterministic prediction of outcomes.
When T_pred is shorter than the relevant biological timescale,
falsification becomes incoherent.
""")

if __name__ == '__main__':
    run_simulation()
