#!/usr/bin/env python3
"""
DEMO: THE LORENZ SHADOW BOX
===========================

THE QUESTION:
If a deterministic system (Lorenz attractor) is observed through a
2D projection, how often does a "clean" binary classification in the
shadow give the WRONG answer about the true system state?

THE SETUP:
- System: Lorenz attractor (3D chaotic flow, D_sys ≈ 2.06)
- Shadow: Projection to (y, z) plane (D_obs = 2)
- Binary test: "Is z > 25?" (looks like a clean horizontal cut)
- Truth: "Is x > 0?" (which lobe is the system actually in?)

THE RESULT:
~47% of points classified by the shadow test are WRONG about the true lobe.
The shadow lies almost half the time.

THE IMPLICATION:
If this happens with a simple 3D→2D projection of a well-understood
mathematical system, imagine what happens when we project 14D gene
expression into a 2D t-SNE plot.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
    """Lorenz attractor ODEs."""
    x, y, z = state
    return [
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ]


def generate_lorenz_trajectory(n_points=10000, dt=0.01, transient=1000):
    """Generate points on the Lorenz attractor."""
    # Initial condition
    state0 = [1.0, 1.0, 1.0]

    # Integrate
    t = np.arange(0, (n_points + transient) * dt, dt)
    trajectory = odeint(lorenz_system, state0, t)

    # Discard transient
    return trajectory[transient:]


def run_shadow_box_analysis():
    """Main analysis demonstrating the shadow box concept."""

    print("=" * 60)
    print("THE LORENZ SHADOW BOX")
    print("Plato's Cave for Falsifiability")
    print("=" * 60)

    # Generate trajectory
    print("\nGenerating Lorenz attractor trajectory...")
    trajectory = generate_lorenz_trajectory(n_points=10000)
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # THE SYSTEM TRUTH: Which lobe are we in?
    # x > 0 = right lobe, x < 0 = left lobe
    system_truth = x > 0  # True = right lobe

    # THE SHADOW OBSERVATION: Project to (y, z) plane
    # We can't see x directly - only y and z
    shadow_y = y
    shadow_z = z

    # THE SHADOW TEST: A "clean" binary cut at z = 25
    # This looks reasonable in the shadow - separates two clusters
    shadow_prediction = shadow_z > 25

    # THE CONFRONTATION: How often does the shadow lie?
    correct = system_truth == shadow_prediction
    aliasing_rate = 1 - np.mean(correct)

    # Count teleportations: adjacent time points where shadow changes
    # but system doesn't (or vice versa)
    shadow_changes = np.diff(shadow_prediction.astype(int))
    system_changes = np.diff(system_truth.astype(int))
    teleportations = np.sum((shadow_changes != 0) & (system_changes == 0))

    print(f"\nRESULTS:")
    print(f"  Points analyzed: {len(x):,}")
    print(f"  Shadow test: 'z > 25'")
    print(f"  System truth: 'x > 0' (right lobe)")
    print(f"\n  Aliasing rate: {aliasing_rate:.1%}")
    print(f"  → The shadow is WRONG {aliasing_rate:.0%} of the time")
    print(f"\n  Teleportations: {teleportations}")
    print(f"  → Times the shadow 'jumped' while the system flowed continuously")

    # Plotting
    fig = plt.figure(figsize=(16, 10))

    # Panel A: The full 3D system
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    colors = np.where(system_truth, '#E63946', '#457B9D')
    ax1.scatter(x[::10], y[::10], z[::10], c=colors[::10], s=1, alpha=0.5)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('A. The System (3D Lorenz Attractor)\nRed = Right Lobe (x>0), Blue = Left Lobe',
                  fontweight='bold')

    # Panel B: The shadow (y, z projection)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(shadow_y[::10], shadow_z[::10], c=colors[::10], s=1, alpha=0.5)
    ax2.axhline(y=25, color='black', linestyle='--', linewidth=2, label='Shadow cut: z=25')
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.set_title('B. The Shadow (2D Projection)\nWe can only see Y and Z',
                  fontweight='bold')
    ax2.legend()

    # Panel C: Where the shadow lies
    ax3 = fig.add_subplot(2, 2, 3)
    # Points where shadow is WRONG
    wrong_mask = ~correct
    ax3.scatter(shadow_y[::10], shadow_z[::10], c='lightgray', s=1, alpha=0.3)
    ax3.scatter(shadow_y[wrong_mask][::5], shadow_z[wrong_mask][::5],
                c='magenta', s=3, alpha=0.7, label=f'Shadow lies here ({aliasing_rate:.0%})')
    ax3.axhline(y=25, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Y')
    ax3.set_ylabel('Z')
    ax3.set_title(f'C. Where The Shadow Lies\n{aliasing_rate:.1%} of points misclassified',
                  fontweight='bold')
    ax3.legend()

    # Panel D: Summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    summary_text = f"""
    THE SHADOW BOX DEMONSTRATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    SYSTEM (The Territory):
      Lorenz attractor in 3D
      D_sys ≈ 2.06 (fractal dimension)

    SHADOW (The Map):
      Projection to (y, z) plane
      D_obs = 2

    BINARY TEST:
      Shadow says: "z > 25 means right lobe"
      System truth: "x > 0 means right lobe"

    RESULT:
      Aliasing = {aliasing_rate:.1%}

      The shadow is WRONG about the
      true system state {aliasing_rate:.0%} of the time.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    IMPLICATION FOR BIOLOGY:

    If a clean binary cut in a simple 3D→2D
    projection fails {aliasing_rate:.0%} of the time,
    imagine projecting 14D gene expression
    into a 2D t-SNE plot.

    The shadow MUST lie. The question is:
    how much, and about what?
    """
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax4.set_title('D. Summary', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig_lorenz_shadow.pdf', dpi=150, bbox_inches='tight')
    print("\nSaved: fig_lorenz_shadow.pdf")

    plt.show()

    return {
        'aliasing_rate': aliasing_rate,
        'teleportations': teleportations,
        'n_points': len(x)
    }


if __name__ == '__main__':
    results = run_shadow_box_analysis()
