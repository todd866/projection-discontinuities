#!/usr/bin/env python3
"""
The Shadow Box: System vs Observation
======================================

Demonstrates the ontological distinction between:
- D_sys: The intrinsic dimensionality of the system (the reality)
- D_obs: The dimensionality of our observations (the shadow)

Shows that a binary cut that looks clean in the 2D projection
can alias completely different causal states together.

This is Plato's cave for falsifiability: we can only falsify
the shadow, but the shadow does not obey the same laws as the system.

Paper: "The Limits of Falsifiability" (BioSystems 258, 2025)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def lorenz_system(state, t, sigma=10, rho=28, beta=8/3):
    """
    The Lorenz system - a classic chaotic attractor.

    This is our "system" with D_sys ≈ 2.06 (fractal dimension).
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def integrate_lorenz(initial, T=50, dt=0.01):
    """Integrate the Lorenz system using Euler method."""
    n_steps = int(T / dt)
    trajectory = np.zeros((n_steps, 3))
    trajectory[0] = initial

    for i in range(1, n_steps):
        state = trajectory[i-1]
        trajectory[i] = state + dt * lorenz_system(state, i*dt)

    return trajectory

def label_by_lobe(trajectory):
    """
    Label points by which lobe of the Lorenz attractor they're in.

    This represents two genuinely different dynamical states.
    """
    # The Lorenz attractor has two lobes, roughly separated by x > 0 vs x < 0
    # But in the full 3D space, they're connected by smooth transitions
    return (trajectory[:, 0] > 0).astype(int)

def run_simulation():
    """Main simulation demonstrating the shadow box concept."""

    np.random.seed(42)

    print("=" * 60)
    print("THE SHADOW BOX: SYSTEM VS OBSERVATION")
    print("=" * 60)
    print("\nGenerating Lorenz attractor trajectories...")

    # Generate trajectory on the Lorenz attractor
    # Start from two different initial conditions
    traj1 = integrate_lorenz(np.array([1.0, 1.0, 1.0]), T=100)
    traj2 = integrate_lorenz(np.array([1.0, 1.0, 1.0001]), T=100)  # Tiny perturbation

    # Subsample for visualization
    traj1 = traj1[::10]
    traj2 = traj2[::10]
    trajectory = traj1  # Use first trajectory

    # Label by lobe (the "true" causal state)
    labels = label_by_lobe(trajectory)

    print(f"Generated {len(trajectory)} points on the Lorenz attractor")
    print(f"Points in left lobe (x < 0): {np.sum(labels == 0)}")
    print(f"Points in right lobe (x > 0): {np.sum(labels == 1)}")

    # Create the shadow: project onto y-z plane
    shadow = trajectory[:, 1:]  # Drop x coordinate

    # Find a "good" binary cut in the shadow
    # This cut looks like it separates the data nicely
    cut_threshold = 25  # z > 25 vs z < 25

    shadow_prediction = (shadow[:, 1] > cut_threshold).astype(int)

    # How well does the shadow cut match the true labels?
    accuracy = np.mean(shadow_prediction == labels)
    print(f"\nBinary cut in shadow (z > {cut_threshold}):")
    print(f"  Accuracy vs true lobe: {accuracy:.1%}")

    # The aliasing problem: count how many states are misclassified
    aliased_left = np.sum((shadow_prediction == 1) & (labels == 0))
    aliased_right = np.sum((shadow_prediction == 0) & (labels == 1))
    print(f"  States from LEFT lobe classified as RIGHT: {aliased_left}")
    print(f"  States from RIGHT lobe classified as LEFT: {aliased_right}")

    # Type I/II error framing (speaks to falsificationists)
    total_left = np.sum(labels == 0)
    total_right = np.sum(labels == 1)
    print(f"\n  Type II Error Rate (False Falsification):")
    print(f"    Left lobe misclassified: {aliased_left/total_left:.1%}")
    print(f"    Right lobe misclassified: {aliased_right/total_right:.1%}")

    # Plotting
    fig = plt.figure(figsize=(16, 10))

    # Panel A: The full 3D system
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    colors = ['#E63946' if l == 0 else '#2A9D8F' for l in labels]
    ax1.scatter(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                c=colors, s=2, alpha=0.5)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    ax1.set_title('A. THE SYSTEM (3D Reality)\nColored by true dynamical state',
                  fontsize=12, fontweight='bold')
    ax1.view_init(elev=20, azim=45)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E63946', label='Left lobe (x < 0)'),
                       Patch(facecolor='#2A9D8F', label='Right lobe (x > 0)')]
    ax1.legend(handles=legend_elements, loc='upper left')

    # Panel B: The shadow (2D projection)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(shadow[:, 0], shadow[:, 1], c=colors, s=10, alpha=0.5)

    # Draw the binary cut
    y_range = np.array([shadow[:, 0].min() - 5, shadow[:, 0].max() + 5])
    ax2.plot(y_range, [cut_threshold, cut_threshold], 'k--', linewidth=2,
             label=f'Binary cut: z = {cut_threshold}')
    ax2.fill_between(y_range, cut_threshold, shadow[:, 1].max() + 5,
                     alpha=0.1, color='blue', label='Predicted: RIGHT')
    ax2.fill_between(y_range, shadow[:, 1].min() - 5, cut_threshold,
                     alpha=0.1, color='orange', label='Predicted: LEFT')

    ax2.set_xlabel('Y (observed)', fontsize=11)
    ax2.set_ylabel('Z (observed)', fontsize=11)
    ax2.set_title('B. THE SHADOW (2D Projection)\nX dimension is hidden',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel C: The aliasing problem - show misclassified points
    ax3 = fig.add_subplot(2, 2, 3)

    # Correctly classified
    correct = shadow_prediction == labels
    ax3.scatter(shadow[correct, 0], shadow[correct, 1],
                c=[colors[i] for i in np.where(correct)[0]],
                s=10, alpha=0.3, label='Correctly classified')

    # Aliased (misclassified)
    aliased = ~correct
    ax3.scatter(shadow[aliased, 0], shadow[aliased, 1],
                c='black', s=30, marker='x', linewidths=1.5,
                label=f'ALIASED ({np.sum(aliased)} points)')

    ax3.plot(y_range, [cut_threshold, cut_threshold], 'k--', linewidth=2)
    ax3.set_xlabel('Y (observed)', fontsize=11)
    ax3.set_ylabel('Z (observed)', fontsize=11)
    ax3.set_title(f'C. THE ALIASING PROBLEM\n{np.sum(aliased)} states ({100*np.mean(aliased):.0f}%) misclassified',
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Panel D: The philosophical point
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    concept_text = """
    PLATO'S CAVE FOR FALSIFIABILITY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    THE SYSTEM (3D Lorenz attractor):
    • Two genuinely different dynamical states
      (left lobe vs right lobe)
    • Smooth, continuous flow between them
    • D_sys ≈ 2.06 (fractal dimension)

    THE SHADOW (2D projection):
    • We only observe Y and Z
    • X is hidden (sub-threshold, unmeasured, etc.)
    • The projection ALIASES distinct states together

    THE "FALSIFICATION":
    • We draw a line in the shadow: z = 25
    • It looks clean! Two separated clusters!
    • But it's WRONG about the actual system

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    THE EPISTEMOLOGICAL POINT:

    In the shadow, points "teleport" across the cut
    when the hidden X coordinate changes sign.

    A falsificationist seeing only the shadow would say:
    "Theory X is false—the system violated continuity!"

    But in the full system, nothing discontinuous happened.
    The particle smoothly traversed a dimension orthogonal
    to the sensor.

    → We can only falsify the SHADOW.
    → The shadow does not obey the same laws as the SYSTEM.
    → Falsifying the shadow tells us nothing about the system
       when D_sys >> D_obs.
    """

    ax4.text(0.05, 0.98, concept_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax4.set_title('D. The Ontological Point', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    for path in ['figures/fig_shadow_box.pdf', '../figures/fig_shadow_box.pdf']:
        try:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to {path}")
            break
        except:
            continue

    plt.show()

    # Detailed analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS: WHERE DOES ALIASING OCCUR?")
    print("=" * 60)

    # Find transition points (where trajectory crosses x=0)
    transitions = np.where(np.diff(labels) != 0)[0]
    print(f"\nNumber of lobe transitions: {len(transitions)}")

    # Show that aliasing happens near transitions
    near_transition = np.zeros(len(trajectory), dtype=bool)
    for t in transitions:
        near_transition[max(0, t-5):min(len(trajectory), t+5)] = True

    aliased_near_transition = np.sum(aliased & near_transition)
    aliased_far_from_transition = np.sum(aliased & ~near_transition)

    print(f"Aliased points near transitions (±5 steps): {aliased_near_transition}")
    print(f"Aliased points far from transitions: {aliased_far_from_transition}")

    # Topological violation: count how many times trajectory "jumps" across the cut
    # in the shadow while the true state remains continuous
    shadow_jumps = np.sum(np.abs(np.diff(shadow_prediction)) == 1)
    true_jumps = np.sum(np.abs(np.diff(labels)) == 1)
    false_jumps = shadow_jumps - true_jumps  # Apparent discontinuities that aren't real

    print(f"\nTopological analysis:")
    print(f"  Apparent jumps in shadow (z crosses {cut_threshold}): {shadow_jumps}")
    print(f"  True lobe transitions in system: {true_jumps}")
    print(f"  TOPOLOGICAL VIOLATIONS: {abs(false_jumps)} false discontinuities")
    print(f"  (Times the shadow 'teleports' while the system flows continuously)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"""
The simulation demonstrates the "shadow box" concept:

1. THE SYSTEM: A 3D Lorenz attractor with two distinct dynamical
   states (lobes). Transitions between lobes are smooth and continuous.

2. THE SHADOW: A 2D projection (Y, Z only). The X dimension is hidden.

3. THE CUT: A binary classifier (z > {cut_threshold}) that looks clean
   in the shadow but achieves only {accuracy:.0%} accuracy on the true states.

4. THE ALIASING: {np.sum(aliased)} points ({100*np.mean(aliased):.0f}%) are misclassified.
   These are states where the hidden X coordinate places them in
   one lobe, but the visible (Y, Z) coordinates suggest the other.

ONTOLOGICAL IMPLICATION:

The arguments in "The Limits of Falsifiability" are not about
"big data" per se. They are about the GEOMETRY OF THE UNDERLYING
SYSTEM. The time series, images, and spike trains we analyze are
shadows—low-dimensional projections of a much higher-dimensional
phase space.

Falsifiability is a claim about REALITY ("the world cannot behave
this way"), but the only thing we ever falsify or confirm are
models of the SHADOW. When D_sys >> D_obs, and when much of the
causal structure lives in hidden dimensions, any low-D binary
projection necessarily discards most of that structure.

Popper's picture of decisive experiments carving away hypothesis
space is geometrically incoherent in this regime.
""")

if __name__ == '__main__':
    run_simulation()
