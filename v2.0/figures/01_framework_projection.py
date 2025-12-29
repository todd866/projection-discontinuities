#!/usr/bin/env python3
"""
Framework as Projection
========================

Demonstrates the key v2.0 insight: different scientific frameworks
are different projections of the same high-dimensional reality.

Paper: "The Limits of Falsifiability" v2.0 (2025)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
    cylinder = generate_cylinder(2000)

    # Larger figure with better spacing
    fig = plt.figure(figsize=(14, 10))

    # Use gridspec for better control
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    # Panel A: The 3D object
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.scatter(cylinder[:, 0], cylinder[:, 1], cylinder[:, 2],
                c=cylinder[:, 2], cmap='viridis', s=2, alpha=0.6)
    ax1.set_xlabel('X', fontsize=11, labelpad=5)
    ax1.set_ylabel('Y', fontsize=11, labelpad=5)
    ax1.set_zlabel('Z', fontsize=11, labelpad=5)
    ax1.set_title('A. REALITY\n(The 3D System)', fontsize=13, fontweight='bold', pad=10)
    ax1.view_init(elev=20, azim=45)

    # Panel B: XY projection (circle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(cylinder[:, 0], cylinder[:, 1], c='#2A9D8F', s=2, alpha=0.4)
    circle = plt.Circle((0, 0), 1, fill=False, color='#E63946', linewidth=2.5)
    ax2.add_patch(circle)
    ax2.set_xlim(-1.6, 1.6)
    ax2.set_ylim(-1.6, 1.6)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X', fontsize=11)
    ax2.set_ylabel('Y', fontsize=11)
    ax2.set_title('B. FRAMEWORK 1: Top View\n"The object is CIRCULAR"',
                  fontsize=12, fontweight='bold', color='#2A9D8F', pad=10)
    ax2.grid(True, alpha=0.3)

    # Panel C: XZ projection (rectangle)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(cylinder[:, 0], cylinder[:, 2], c='#E76F51', s=2, alpha=0.4)
    rect = plt.Rectangle((-1, -1), 2, 2, fill=False, color='#E63946', linewidth=2.5)
    ax3.add_patch(rect)
    ax3.set_xlim(-1.6, 1.6)
    ax3.set_ylim(-1.6, 1.6)
    ax3.set_aspect('equal')
    ax3.set_xlabel('X', fontsize=11)
    ax3.set_ylabel('Z', fontsize=11)
    ax3.set_title('C. FRAMEWORK 2: Side View\n"The object is RECTANGULAR"',
                  fontsize=12, fontweight='bold', color='#E76F51', pad=10)
    ax3.grid(True, alpha=0.3)

    # Panel D: Key insight (simplified text)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Use structured text blocks instead of one dense block
    ax4.text(0.5, 0.95, 'FRAMEWORK DEPENDENCE',
             ha='center', va='top', fontsize=14, fontweight='bold',
             transform=ax4.transAxes)

    ax4.text(0.5, 0.85, 'The Reality: A 3D cylinder',
             ha='center', va='top', fontsize=11, style='italic',
             transform=ax4.transAxes)

    key_points = [
        'Each framework sees a different shadow',
        'Framework 1 claims: "circular"',
        'Framework 2 claims: "rectangular"',
        'Neither can falsify the other',
        'Both are projections of the same truth'
    ]

    for i, point in enumerate(key_points):
        ax4.text(0.1, 0.72 - i*0.1, f'• {point}',
                 ha='left', va='top', fontsize=10,
                 transform=ax4.transAxes)

    # Bottom box with key insight
    insight_text = ('Falsification is framework-relative.\n'
                   'The framework is itself a projection.\n'
                   'Different questions = different shadows.')

    ax4.text(0.5, 0.15, insight_text,
             ha='center', va='center', fontsize=10,
             transform=ax4.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                      edgecolor='#999999', alpha=0.9))

    ax4.set_title('D. The Key Insight', fontsize=12, fontweight='bold', pad=10)

    plt.savefig('fig_framework_projection.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig_framework_projection.png', dpi=150, bbox_inches='tight')
    print("Saved: fig_framework_projection.pdf/png")
    plt.close()


def create_three_levels_figure():
    """Create figure showing the three levels of limitation."""

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    plt.subplots_adjust(wspace=0.3)

    # Level 1: Physical measurement limits
    ax1 = axes[0]

    ax1.axhspan(0, 3e-21, alpha=0.25, color='#E63946', label='Sub-Landauer')
    ax1.axhline(y=3e-21, color='#E63946', linestyle='--', linewidth=2.5,
                label='Landauer limit')
    ax1.axhline(y=5e-20, color='#2A9D8F', linestyle='-', linewidth=2,
                label='ATP hydrolysis')

    # Cleaner signal placement
    signals = [
        ('Ephaptic', 1e-21, '#E63946'),
        ('Protein vib.', 2e-21, '#E63946'),
        ('Action pot.', 1e-19, '#2A9D8F'),
    ]

    for i, (name, energy, color) in enumerate(signals):
        ax1.scatter(i+1, energy, s=120, c=color, zorder=5, edgecolor='white', linewidth=1)
        # Offset annotations to avoid overlap
        offset = 15 if energy < 3e-21 else 12
        ax1.annotate(name, (i+1, energy), textcoords="offset points",
                    xytext=(0, offset), ha='center', fontsize=9, fontweight='bold')

    ax1.set_yscale('log')
    ax1.set_ylim(5e-22, 5e-19)
    ax1.set_xlim(0, 4)
    ax1.set_xticks([])
    ax1.set_ylabel('Energy (J)', fontsize=12)
    ax1.set_title('Level 1: PHYSICAL LIMITS\nSub-Landauer patterns exist\nbut cannot be measured as bits',
                  fontsize=11, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Level 2: Dimensional projection
    ax2 = axes[1]
    dims = np.array([1, 10, 100, 1000])
    info_lost = dims  # Orders of magnitude lost

    bars = ax2.bar(range(len(dims)), info_lost, color='#457B9D', alpha=0.8, width=0.6)
    ax2.set_xticks(range(len(dims)))
    ax2.set_xticklabels([f'n={d}' for d in dims], fontsize=10)
    ax2.set_ylabel('Information lost\n(orders of magnitude)', fontsize=11)
    ax2.set_xlabel('System dimension', fontsize=11)
    ax2.set_title('Level 2: DIMENSIONAL PROJECTION\nBinary tests destroy information',
                  fontsize=11, fontweight='bold', pad=15)

    # Cleaner annotation
    ax2.annotate('100 neurons:\n$10^{-100}$ preserved',
                 xy=(2, 100), xytext=(2.8, 500),
                 arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5),
                 fontsize=10, color='#E63946', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E63946'))

    ax2.set_ylim(0, 1100)

    # Level 3: Framework dependence
    ax3 = axes[2]
    ax3.axis('off')

    # Draw two circles representing frameworks
    from matplotlib.patches import Circle as MplCircle

    circle1 = MplCircle((0.3, 0.55), 0.18, fill=True, facecolor='#2A9D8F',
                         alpha=0.3, edgecolor='#2A9D8F', linewidth=2)
    circle2 = MplCircle((0.7, 0.55), 0.18, fill=True, facecolor='#E76F51',
                         alpha=0.3, edgecolor='#E76F51', linewidth=2)

    ax3.add_patch(circle1)
    ax3.add_patch(circle2)

    ax3.text(0.3, 0.55, 'Frame\nA', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(0.7, 0.55, 'Frame\nB', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(0.5, 0.85, 'REALITY', ha='center', va='center', fontsize=13,
             fontweight='bold', style='italic')

    # Question mark
    ax3.text(0.5, 0.55, '?', ha='center', va='center', fontsize=28,
             color='#E63946', fontweight='bold')

    # Simplified text
    ax3.text(0.5, 0.18,
             'Frameworks project reality differently\n'
             'Falsification is framework-relative\n'
             'Axioms cannot test themselves',
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                      edgecolor='#999999', alpha=0.9))

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('Level 3: FRAMEWORK DEPENDENCE\nAxiomatic choices precede\nall measurement',
                  fontsize=11, fontweight='bold', pad=15)

    plt.savefig('fig_three_levels.pdf', dpi=150, bbox_inches='tight')
    plt.savefig('fig_three_levels.png', dpi=150, bbox_inches='tight')
    print("Saved: fig_three_levels.pdf/png")
    plt.close()


def create_wigner_selection_figure():
    """Create figure illustrating Wigner selection bias."""

    fig, ax = plt.subplots(figsize=(11, 8))

    np.random.seed(42)
    n_domains = 150

    projection_loss = np.random.exponential(1, n_domains)
    physics_prob = np.exp(-projection_loss * 2)
    studied_by_physics = np.random.random(n_domains) < physics_prob
    math_success = 1 / (1 + projection_loss)

    # Plot scatter
    ax.scatter(projection_loss[~studied_by_physics],
               math_success[~studied_by_physics],
               c='#CCCCCC', s=40, alpha=0.4, label='Not studied by physics')
    ax.scatter(projection_loss[studied_by_physics],
               math_success[studied_by_physics],
               c='#2A9D8F', s=80, alpha=0.7, label='Studied by physics')

    # Cleaner domain labels with better positioning
    domains_physics = [
        (0.1, 0.92, 'Particle physics'),
        (0.25, 0.82, 'Mechanics'),
        (0.4, 0.72, 'E&M'),
    ]

    domains_other = [
        (0.9, 0.52, 'Fluids'),
        (1.4, 0.42, 'Mol. biology'),
        (2.0, 0.33, 'Ecology'),
        (2.5, 0.28, 'Consciousness'),
        (3.2, 0.22, 'Social systems'),
    ]

    for x, y, label in domains_physics:
        ax.annotate(label, (x, y), fontsize=9, ha='center', color='#2A9D8F',
                   fontweight='bold')

    for x, y, label in domains_other:
        ax.annotate(label, (x, y), fontsize=9, ha='center', color='#666666')

    # Selection zone
    ax.axvline(x=0.7, color='#E63946', linestyle='--', linewidth=2)
    ax.fill_betweenx([0, 1], 0, 0.7, alpha=0.08, color='#2A9D8F')

    ax.set_xlabel('Projection Loss (D_sys >> D_math)', fontsize=12)
    ax.set_ylabel('Success of Mathematical Description', fontsize=12)
    ax.set_title("Wigner's 'Unreasonable Effectiveness' as Selection Bias",
                 fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_xlim(-0.1, 4)
    ax.set_ylim(0, 1.05)

    # Two annotation boxes with clear separation
    ax.text(0.35, 0.18, 'Physics selects for\nlow projection loss',
            fontsize=10, ha='center', color='#2A9D8F', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#2A9D8F', alpha=0.9))

    ax.text(2.8, 0.55, 'Biology lives here:\nhigh D_sys, low D_obs',
            fontsize=10, ha='center', color='#666666',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#666666', alpha=0.9))

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
