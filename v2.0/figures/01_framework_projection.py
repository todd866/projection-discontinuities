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

    # Level 2: Dimensional projection (CORRECTED MATH)
    ax2 = axes[1]
    dims = np.array([10, 50, 100, 500, 1000])
    k = 10  # states per degree of freedom
    # Information preserved = 1 / (n * log2(k))
    info_preserved_pct = 100 / (dims * np.log2(k))  # as percentage

    bars = ax2.bar(range(len(dims)), info_preserved_pct, color='#457B9D', alpha=0.8, width=0.6)
    ax2.set_xticks(range(len(dims)))
    ax2.set_xticklabels([f'n={d}' for d in dims], fontsize=10)
    ax2.set_ylabel('Information preserved\nby binary test (%)', fontsize=11)
    ax2.set_xlabel('System dimension (k=10 states each)', fontsize=11)
    ax2.set_title('Level 2: DIMENSIONAL PROJECTION\nBinary tests preserve <1% of information',
                  fontsize=11, fontweight='bold', pad=15)

    # Annotation showing the math
    ax2.annotate('100 neurons:\n~0.3% preserved\n(99.7% lost)',
                 xy=(2, info_preserved_pct[2]), xytext=(3.2, 1.5),
                 arrowprops=dict(arrowstyle='->', color='#E63946', lw=1.5),
                 fontsize=9, color='#E63946', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#E63946'))

    ax2.set_ylim(0, 4)
    ax2.axhline(y=1, color='#E63946', linestyle='--', alpha=0.5, label='1% threshold')

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
    """Create figure illustrating Wigner selection bias.

    NOTE: This is a SCHEMATIC/HEURISTIC visualization. The key domain
    positions are hard-coded to ensure they always appear where the
    theory predicts, regardless of random background scatter.
    """

    fig, ax = plt.subplots(figsize=(12, 8))

    np.random.seed(42)

    # HARD-CODED key domain positions (these anchor the argument)
    # Format: (projection_loss, math_success, label, is_physics)
    key_domains = [
        (0.15, 0.87, 'Particle\nphysics', True),
        (0.25, 0.80, 'Classical\nmechanics', True),
        (0.35, 0.74, 'Electro-\nmagnetism', True),
        (0.55, 0.65, 'Thermo-\ndynamics', True),
        (1.0, 0.50, 'Fluid\ndynamics', True),
        (1.5, 0.40, 'Molecular\nbiology', False),
        (2.2, 0.31, 'Ecology', False),
        (2.8, 0.26, 'Consciousness', False),
        (3.4, 0.23, 'Social\nsystems', False),
    ]

    # Background scatter (illustrative noise, not specific domains)
    n_background = 60
    bg_loss = np.random.exponential(1.5, n_background)
    bg_success = 1 / (1 + bg_loss) + np.random.normal(0, 0.03, n_background)
    bg_success = np.clip(bg_success, 0.1, 0.95)

    # Plot background scatter
    ax.scatter(bg_loss, bg_success, c='#DDDDDD', s=30, alpha=0.3,
               label='Other domains (illustrative)')

    # Plot and label key domains
    for pl, ms, label, is_physics in key_domains:
        color = '#2A9D8F' if is_physics else '#888888'
        size = 120 if is_physics else 80
        ax.scatter(pl, ms, c=color, s=size, alpha=0.9, zorder=5,
                  edgecolor='white', linewidth=1)
        ax.annotate(label, (pl, ms), fontsize=8, ha='center', va='bottom',
                   color=color, fontweight='bold' if is_physics else 'normal',
                   xytext=(0, 8), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=color, alpha=0.85) if is_physics else None)

    # Selection zone
    ax.axvline(x=0.8, color='#E63946', linestyle='--', linewidth=2,
               label='Informal "physics" boundary')
    ax.fill_betweenx([0, 1.0], 0, 0.8, alpha=0.06, color='#2A9D8F')

    ax.set_xlabel('Projection Loss (D_sys >> D_math)', fontsize=12)
    ax.set_ylabel('Success of Mathematical Description', fontsize=12)
    ax.set_title("Wigner's 'Unreasonable Effectiveness' as Selection Bias\n"
                 "(Schematic Representation)",
                 fontsize=13, fontweight='bold', pad=10)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(0, 4.0)
    ax.set_ylim(0.15, 0.95)

    # Annotation boxes
    ax.text(0.4, 0.35, 'Physics selects for\nlow projection loss',
            fontsize=10, ha='center', color='#2A9D8F', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#2A9D8F', alpha=0.9))

    ax.text(2.8, 0.50, 'Biology lives here:\nhigh D_sys, low D_obs',
            fontsize=10, ha='center', color='#555555',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                     edgecolor='#555555', alpha=0.9))

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
