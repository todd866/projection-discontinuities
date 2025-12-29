#!/usr/bin/env python3
"""
Framework as Projection
========================

Demonstrates the key v2.0 insight: different scientific frameworks
are different projections of the same high-dimensional reality.

Like a 3D object casting different shadows on different walls:
- One observer sees a circle
- Another sees a rectangle
- Both are "correct" projections, but they generate incompatible claims

Paper: "The Limits of Falsifiability" v2.0 (2025)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
import os

os.makedirs('.', exist_ok=True)


def generate_cylinder(n_points=1000, radius=1, height=2):
    """Generate points on a cylinder surface."""
    theta = np.random.uniform(0, 2*np.pi, n_points)
    z = np.random.uniform(-height/2, height/2, n_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack([x, y, z])


def run_simulation():
    """Create the framework projection visualization."""

    np.random.seed(42)

    # Generate cylinder (our 3D "reality")
    cylinder = generate_cylinder(2000)

    # Create figure
    fig = plt.figure(figsize=(16, 12))

    # Panel A: The 3D object (Reality)
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(cylinder[:, 0], cylinder[:, 1], cylinder[:, 2],
                c=cylinder[:, 2], cmap='viridis', s=1, alpha=0.5)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z', fontsize=10)
    ax1.set_title('A. REALITY\n(The 3D System)', fontsize=14, fontweight='bold')
    ax1.view_init(elev=20, azim=45)
    ax1.set_box_aspect([1, 1, 1])

    # Panel B: Projection onto XY plane (sees circle)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.scatter(cylinder[:, 0], cylinder[:, 1], c='#2A9D8F', s=1, alpha=0.3)
    circle = plt.Circle((0, 0), 1, fill=False, color='#E63946', linewidth=2)
    ax2.add_patch(circle)
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Y', fontsize=11)
    ax2.set_title('B. FRAMEWORK 1: View from Above\n"The object is CIRCULAR"',
                  fontsize=12, fontweight='bold', color='#2A9D8F')
    ax2.grid(True, alpha=0.3)

    # Panel C: Projection onto XZ plane (sees rectangle)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(cylinder[:, 0], cylinder[:, 2], c='#E76F51', s=1, alpha=0.3)
    rect = plt.Rectangle((-1, -1), 2, 2, fill=False, color='#E63946', linewidth=2)
    ax3.add_patch(rect)
    ax3.set_xlim(-1.5, 1.5)
    ax3.set_ylim(-1.5, 1.5)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X', fontsize=11)
    ax3.set_ylabel('Z', fontsize=11)
    ax3.set_title('C. FRAMEWORK 2: View from Side\n"The object is RECTANGULAR"',
                  fontsize=12, fontweight='bold', color='#E76F51')
    ax3.grid(True, alpha=0.3)

    # Panel D: The epistemological point
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    concept_text = """
    FRAMEWORK DEPENDENCE OF FALSIFIABILITY
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    THE REALITY: A 3D cylinder

    FRAMEWORK 1 (XY projection):
    • Hypothesis: "The object is circular"
    • This is UNFALSIFIABLE from Framework 2's perspective
    • Every test in Framework 2 shows a rectangle

    FRAMEWORK 2 (XZ projection):
    • Hypothesis: "The object is rectangular"
    • This is UNFALSIFIABLE from Framework 1's perspective
    • Every test in Framework 1 shows a circle

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    THE DEEPER POINT:

    Before any measurement occurs, the choice of what
    counts as a test, what counts as evidence, and how
    the question is structured has already made a
    DIMENSIONAL REDUCTION.

    Different researchers asking "different questions"
    are often occupying different projections of the
    same underlying reality.

    Their disagreements may not be resolvable by evidence
    because they are not making claims in the same framework.

    This is not relativism—the cylinder exists.
    But falsification is FRAMEWORK-RELATIVE.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    IMPLICATIONS:

    • Physics "works" because its framework assumptions
      are unusually stable and widely shared
    • Biology struggles because researchers occupy
      genuinely different frameworks
    • The "unreasonable effectiveness of mathematics"
      reflects selection bias toward domains where
      projection loss is small
    """

    ax4.text(0.02, 0.98, concept_text, transform=ax4.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax4.set_title('D. The Epistemological Point', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    plt.savefig('fig_framework_projection.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig_framework_projection.png', dpi=150, bbox_inches='tight')
    print("Saved: fig_framework_projection.pdf/png")

    plt.close()


def create_three_levels_figure():
    """Create figure showing the three levels of limitation."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Level 1: Physical measurement limits
    ax1 = axes[0]
    x = np.linspace(0, 5, 100)
    landauer = np.ones_like(x) * 3e-21  # Landauer limit
    atp = np.ones_like(x) * 5e-20  # ATP energy

    ax1.fill_between(x, 0, landauer, alpha=0.3, color='#E63946',
                     label='Sub-Landauer Domain')
    ax1.axhline(y=3e-21, color='#E63946', linestyle='--', linewidth=2,
                label=f'Landauer limit: {3e-21:.1e} J')
    ax1.axhline(y=5e-20, color='#2A9D8F', linestyle='-', linewidth=2,
                label=f'ATP hydrolysis: {5e-20:.1e} J')

    # Example biological signals
    signals = {
        'Ephaptic\ncoupling': 1e-21,
        'Protein\nvibrations': 2e-21,
        'Weak\nsignals': 2.5e-21,
        'Action\npotential': 1e-19,
    }
    for i, (name, energy) in enumerate(signals.items()):
        color = '#E63946' if energy < 3e-21 else '#2A9D8F'
        ax1.scatter(i+0.5, energy, s=100, c=color, zorder=5)
        ax1.annotate(name, (i+0.5, energy), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=8)

    ax1.set_yscale('log')
    ax1.set_ylim(1e-22, 1e-18)
    ax1.set_xlim(0, 5)
    ax1.set_xticks([])
    ax1.set_ylabel('Energy (J)', fontsize=11)
    ax1.set_title('Level 1: PHYSICAL LIMITS\nSub-Landauer patterns cannot\nbe measured as bits',
                  fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)

    # Level 2: Dimensional projection
    ax2 = axes[1]
    dims = np.array([1, 10, 100, 1000, 10000])
    preserved = 1.0 / (10 ** dims)  # Information preserved by binary projection

    ax2.bar(range(len(dims)), -np.log10(1/preserved), color='#457B9D', alpha=0.7)
    ax2.set_xticks(range(len(dims)))
    ax2.set_xticklabels([f'n={d}' for d in dims])
    ax2.set_ylabel('Information lost\n(orders of magnitude)', fontsize=11)
    ax2.set_xlabel('System dimension (k=10 states each)', fontsize=10)
    ax2.set_title('Level 2: DIMENSIONAL PROJECTION\nBinary tests preserve ~nothing\nof high-D systems',
                  fontsize=11, fontweight='bold')

    # Add annotation for 100 neurons
    ax2.annotate('100 neurons:\n~$10^{-100}$ preserved',
                 xy=(2, 100), xytext=(3.5, 50),
                 arrowprops=dict(arrowstyle='->', color='#E63946'),
                 fontsize=9, color='#E63946')

    # Level 3: Framework dependence
    ax3 = axes[2]
    ax3.axis('off')

    # Draw nested circles representing frameworks
    from matplotlib.patches import Circle as MplCircle, FancyArrowPatch

    circle1 = MplCircle((0.3, 0.6), 0.2, fill=False, color='#2A9D8F', linewidth=2)
    circle2 = MplCircle((0.7, 0.6), 0.2, fill=False, color='#E76F51', linewidth=2)

    ax3.add_patch(circle1)
    ax3.add_patch(circle2)

    ax3.text(0.3, 0.6, 'Framework\nA', ha='center', va='center', fontsize=10)
    ax3.text(0.7, 0.6, 'Framework\nB', ha='center', va='center', fontsize=10)
    ax3.text(0.5, 0.85, 'REALITY', ha='center', va='center', fontsize=12,
             fontweight='bold', style='italic')

    # Question marks between circles
    ax3.text(0.5, 0.6, '?', ha='center', va='center', fontsize=20, color='#E63946')

    ax3.text(0.5, 0.25, 'Different frameworks project\nreality differently.\n\n'
             'Falsification is always relative\nto unstated assumptions.\n\n'
             'The framework itself is a\ndimensional reduction.',
             ha='center', va='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Level 3: FRAMEWORK DEPENDENCE\nAxiomatic choices precede\nall measurement',
                  fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig_three_levels.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig_three_levels.png', dpi=150, bbox_inches='tight')
    print("Saved: fig_three_levels.pdf/png")

    plt.close()


def create_wigner_selection_figure():
    """Create figure illustrating Wigner selection bias."""

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a scatter of "domains" with varying projection loss
    np.random.seed(42)
    n_domains = 200

    # Projection loss (how much information is lost in mathematical description)
    projection_loss = np.random.exponential(1, n_domains)

    # "Studied by physics" probability decreases with projection loss
    physics_prob = np.exp(-projection_loss * 2)
    studied_by_physics = np.random.random(n_domains) < physics_prob

    # Success of mathematical description (inverse of projection loss)
    math_success = 1 / (1 + projection_loss)

    # Plot
    ax.scatter(projection_loss[~studied_by_physics],
               math_success[~studied_by_physics],
               c='#CCCCCC', s=50, alpha=0.5, label='Not studied by physics')
    ax.scatter(projection_loss[studied_by_physics],
               math_success[studied_by_physics],
               c='#2A9D8F', s=100, alpha=0.8, label='Studied by physics')

    # Add domain labels
    domain_labels = {
        (0.1, 0.9): 'Particle\nphysics',
        (0.2, 0.85): 'Classical\nmechanics',
        (0.3, 0.75): 'Electro-\nmagnetism',
        (0.8, 0.55): 'Fluid\ndynamics',
        (1.5, 0.35): 'Ecology',
        (2.0, 0.3): 'Consciousness',
        (2.5, 0.25): 'Social\nsystems',
        (1.2, 0.45): 'Molecular\nbiology',
        (3.0, 0.2): 'Evolution',
    }

    for (x, y), label in domain_labels.items():
        color = '#2A9D8F' if x < 1 else '#666666'
        ax.annotate(label, (x, y), fontsize=8, ha='center', color=color)

    # Selection threshold
    ax.axvline(x=0.8, color='#E63946', linestyle='--', linewidth=2,
               label='Selection threshold')
    ax.fill_betweenx([0, 1], 0, 0.8, alpha=0.1, color='#2A9D8F')

    ax.set_xlabel('Projection Loss (D_sys >> D_math)', fontsize=12)
    ax.set_ylabel('Success of Mathematical Description', fontsize=12)
    ax.set_title("Wigner's 'Unreasonable Effectiveness' as Selection Bias\n"
                 "Physics studies domains where math works; biology is where it doesn't",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)

    # Add annotation
    ax.text(0.4, 0.15, 'Physics selects for\nlow projection loss',
            fontsize=10, ha='center', color='#2A9D8F',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2.5, 0.5, 'Biology lives here:\nhigh D_sys, low D_obs',
            fontsize=10, ha='center', color='#666666',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('fig_wigner_selection.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig_wigner_selection.png', dpi=150, bbox_inches='tight')
    print("Saved: fig_wigner_selection.pdf/png")

    plt.close()


if __name__ == '__main__':
    print("Generating v2.0 figures...")
    run_simulation()
    create_three_levels_figure()
    create_wigner_selection_figure()
    print("\nAll figures generated successfully!")
