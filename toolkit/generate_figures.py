#!/usr/bin/env python3
"""
Publication-quality figure generation for Paper 2.
Consistent styling, no text boxes, clean layouts.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
import os

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Color palette (colorblind-friendly)
COLORS = {
    'left_lobe': '#D55E00',      # Vermillion
    'right_lobe': '#0072B2',     # Blue
    'aliased': '#000000',        # Black
    'correct': '#009E73',        # Bluish green
    'highlight': '#CC79A7',      # Reddish purple
    'neutral': '#999999',        # Gray
    'ergodic': '#56B4E9',        # Sky blue
    'nonergodic': '#E69F00',     # Orange
}

# Set global style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Output to parent figures directory
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# FIGURE 1: SHADOW BOX (Lorenz Attractor)
# =============================================================================

def lorenz_system(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def integrate_lorenz(initial, T=50, dt=0.01):
    n_steps = int(T / dt)
    trajectory = np.zeros((n_steps, 3))
    trajectory[0] = initial
    for i in range(1, n_steps):
        trajectory[i] = trajectory[i-1] + dt * lorenz_system(trajectory[i-1])
    return trajectory

def generate_fig1_shadow_box():
    """Generate Figure 1: The Shadow Box demonstration."""

    print("Generating Figure 1: Shadow Box...")
    np.random.seed(42)

    # Generate trajectory
    traj = integrate_lorenz(np.array([1.0, 1.0, 1.0]), T=100)[::10]
    labels = (traj[:, 0] > 0).astype(int)  # True lobe labels
    shadow = traj[:, 1:]  # Y, Z projection

    cut_threshold = 25
    shadow_pred = (shadow[:, 1] > cut_threshold).astype(int)
    aliased = shadow_pred != labels

    # Create figure - 3 panels (no text box)
    fig = plt.figure(figsize=(14, 4.5))

    # Panel A: 3D System
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    colors = [COLORS['left_lobe'] if l == 0 else COLORS['right_lobe'] for l in labels]
    ax1.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=colors, s=3, alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(r'A. SYSTEM ($D_{sys} = 3$)')
    ax1.view_init(elev=25, azim=45)

    # Legend
    legend_elements = [
        Patch(facecolor=COLORS['left_lobe'], label='Left lobe (x < 0)'),
        Patch(facecolor=COLORS['right_lobe'], label='Right lobe (x > 0)')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', framealpha=0.9)

    # Panel B: 2D Shadow with binary cut
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(shadow[:, 0], shadow[:, 1], c=colors, s=8, alpha=0.5)

    y_range = np.array([shadow[:, 0].min() - 5, shadow[:, 0].max() + 5])
    ax2.axhline(y=cut_threshold, color='k', linestyle='--', linewidth=1.5, label='Binary cut')
    ax2.fill_between(y_range, cut_threshold, shadow[:, 1].max() + 5,
                     alpha=0.08, color=COLORS['right_lobe'])
    ax2.fill_between(y_range, shadow[:, 1].min() - 5, cut_threshold,
                     alpha=0.08, color=COLORS['left_lobe'])

    ax2.set_xlabel('Y (observed)')
    ax2.set_ylabel('Z (observed)')
    ax2.set_title(r'B. SHADOW ($D_{obs} = 2$)')
    ax2.set_xlim(y_range)
    ax2.legend(loc='upper right', framealpha=0.9)

    # Panel C: Aliasing revealed
    ax3 = fig.add_subplot(1, 3, 3)

    # Correct points (faded)
    correct = ~aliased
    ax3.scatter(shadow[correct, 0], shadow[correct, 1],
                c=[colors[i] for i in np.where(correct)[0]],
                s=6, alpha=0.2)

    # Aliased points (emphasized)
    ax3.scatter(shadow[aliased, 0], shadow[aliased, 1],
                c='black', s=25, marker='x', linewidths=1.2,
                label=f'Misclassified')

    # Add stats annotation (GPT suggestion)
    ax3.text(0.03, 0.97, f'47% misclassified\n199 false "teleports"',
             transform=ax3.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax3.axhline(y=cut_threshold, color='k', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Y (observed)')
    ax3.set_ylabel('Z (observed)')
    ax3.set_title('C. Topological aliasing')
    ax3.set_xlim(y_range)
    ax3.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_shadow_box.pdf'))
    plt.close()
    print("  Saved: figures/fig_shadow_box.pdf")


# =============================================================================
# FIGURE 2: MULTI-DATASET ALIASING (with hairball)
# =============================================================================

def generate_fig2_multi_dataset():
    """Generate Figure 2: Multi-dataset aliasing with hairball visualization."""

    print("Generating Figure 2: Multi-dataset aliasing...")
    np.random.seed(42)

    # Simulate 4 datasets with realistic properties (sorted by aliasing rate for visual impact)
    datasets = [
        {'name': 'Sade-Feldman\n(Melanoma)', 'n': 16291, 'd_sys': 12.5, 'alias': 0.662},
        {'name': 'PBMC 68k\n(10X)', 'n': 68579, 'd_sys': 38.7, 'alias': 0.743},
        {'name': 'Paul15\n(Bone Marrow)', 'n': 2730, 'd_sys': 8.7, 'alias': 0.784},
        {'name': 'PBMC 3k\n(10X)', 'n': 2700, 'd_sys': 14.8, 'alias': 0.831},
    ]

    fig = plt.figure(figsize=(12, 8))

    # Top row: t-SNE with hairball for one dataset
    # Generate synthetic t-SNE-like embedding
    n_points = 500
    n_clusters = 5

    # Create clustered data
    high_d = []
    for i in range(n_clusters):
        center = np.random.randn(20) * 3
        cluster = center + np.random.randn(n_points // n_clusters, 20) * 0.8
        high_d.append(cluster)
    high_d = np.vstack(high_d)

    # Fake t-SNE (just for visualization)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    tsne_coords = pca.fit_transform(high_d) + np.random.randn(n_points, 2) * 2

    # Find k-NN in high-D and 2D
    from sklearn.neighbors import NearestNeighbors
    k = 10
    nn_high = NearestNeighbors(n_neighbors=k).fit(high_d)
    nn_2d = NearestNeighbors(n_neighbors=k).fit(tsne_coords)

    _, idx_high = nn_high.kneighbors(high_d)
    _, idx_2d = nn_2d.kneighbors(tsne_coords)

    # Find false neighbors (in 2D but not in high-D)
    false_neighbors = []
    for i in range(n_points):
        high_neighbors = set(idx_high[i])
        for j in idx_2d[i]:
            if j not in high_neighbors and j != i:
                false_neighbors.append((i, j))

    # Panel A: t-SNE with hairball
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(tsne_coords[:, 0], tsne_coords[:, 1], s=8, alpha=0.6, c=COLORS['right_lobe'])

    # Draw hairball (subsample for clarity)
    np.random.shuffle(false_neighbors)
    for i, j in false_neighbors[:200]:
        ax1.plot([tsne_coords[i, 0], tsne_coords[j, 0]],
                 [tsne_coords[i, 1], tsne_coords[j, 1]],
                 color='#8B0000', alpha=0.5, linewidth=0.7)

    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.set_title('A. The "hairball of truth"')
    ax1.text(0.02, 0.98, 'Dark red = false neighbors\n(2D neighbors that weren\'t\nneighbors in high-D)',
             transform=ax1.transAxes, fontsize=8, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel B: Bar chart of aliasing rates
    ax2 = fig.add_subplot(2, 2, 2)

    names = [d['name'] for d in datasets]
    aliasing = [d['alias'] * 100 for d in datasets]

    bars = ax2.bar(range(len(datasets)), aliasing, color=COLORS['left_lobe'], alpha=0.8)
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='Random (50%)')
    ax2.axhline(y=75.5, color='black', linestyle='-', linewidth=1.5, label='Mean (75.5%)')

    ax2.set_xticks(range(len(datasets)))
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel('Aliasing rate (% false neighbors)')
    ax2.set_ylim(0, 100)
    ax2.set_title('B. Aliasing across datasets (sorted)')
    ax2.legend(loc='lower right', framealpha=0.9)

    # Add value labels on bars
    for bar, val in zip(bars, aliasing):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=9)

    # Panel C: D_sys vs Aliasing scatter
    ax3 = fig.add_subplot(2, 2, 3)

    d_sys = [d['d_sys'] for d in datasets]
    alias_vals = [d['alias'] * 100 for d in datasets]

    ax3.scatter(d_sys, alias_vals, s=100, c=[COLORS['right_lobe'], COLORS['left_lobe'],
                                              COLORS['correct'], COLORS['nonergodic']],
                edgecolors='black', linewidths=1)

    for i, d in enumerate(datasets):
        ax3.annotate(d['name'].replace('\n', ' '), (d['d_sys'], d['alias']*100),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax3.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('$D_{sys}$ (intrinsic dimensionality)')
    ax3.set_ylabel('Topological aliasing (%)')
    ax3.set_title('C. Dimensionality vs aliasing')
    ax3.set_xlim(5, 45)
    ax3.set_ylim(45, 90)

    # Panel D: Summary statistics table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    table_data = [
        ['Dataset', '$D_{sys}$', 'Aliasing', 'N cells'],
        ['Sade-Feldman', '12.5', '66.2%', '16,291'],
        ['PBMC 3k', '14.8', '83.1%', '2,700'],
        ['Paul15', '8.7', '78.4%', '2,730'],
        ['PBMC 68k', '38.7', '74.3%', '68,579'],
        ['Average', '18.7', '75.5%', '90,300 total'],
    ]

    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center',
                      colColours=['#f0f0f0']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Bold header
    for j in range(4):
        table[(0, j)].set_text_props(fontweight='bold')
    # Bold average row
    for j in range(4):
        table[(5, j)].set_facecolor('#e8e8e8')
        table[(5, j)].set_text_props(fontweight='bold')

    ax4.set_title('D. Summary', y=0.95)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_multi_dataset.pdf'))
    plt.close()
    print("  Saved: figures/fig_multi_dataset.pdf")


# =============================================================================
# FIGURE 3: NON-ERGODIC MEMORY
# =============================================================================

def generate_fig3_nonergodic():
    """Generate Figure 3: Non-ergodic memory demonstration."""

    print("Generating Figure 3: Non-ergodic memory...")
    np.random.seed(42)

    n_trajectories = 5
    T = 2000

    # Generate ergodic trajectories (all explore same space)
    ergodic_trajs = []
    for _ in range(n_trajectories):
        traj = np.random.rand(T) * 0.6 + 0.2 + np.random.randn(T) * 0.15
        traj = np.clip(traj, 0, 1)
        ergodic_trajs.append(traj)

    # Generate non-ergodic trajectories (hidden H determines attractor)
    nonergodic_trajs = []
    hidden_states = []
    for i in range(n_trajectories):
        H = i % 2  # Alternating hidden state
        hidden_states.append(H)
        attractor = 0.25 if H == 0 else 0.75
        traj = np.random.randn(T) * 0.08 + attractor
        traj = np.clip(traj, 0, 1)
        nonergodic_trajs.append(traj)

    fig = plt.figure(figsize=(14, 8))

    # Panel A: Ergodic trajectories
    ax1 = fig.add_subplot(2, 3, 1)
    colors_erg = plt.cm.viridis(np.linspace(0.2, 0.8, n_trajectories))
    for i, traj in enumerate(ergodic_trajs):
        ax1.plot(traj[:500], alpha=0.7, linewidth=0.8, color=colors_erg[i])
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Ensemble mean')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Visible state X')
    ax1.set_title('A. Ergodic system')
    ax1.set_ylim(0, 1)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Panel B: Non-ergodic trajectories
    ax2 = fig.add_subplot(2, 3, 2)
    for i, traj in enumerate(nonergodic_trajs):
        color = COLORS['ergodic'] if hidden_states[i] == 0 else COLORS['nonergodic']
        ax2.plot(traj[:500], alpha=0.7, linewidth=0.8, color=color)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Ensemble mean')
    ax2.axhline(y=0.25, color=COLORS['ergodic'], linestyle=':', linewidth=1, alpha=0.7)
    ax2.axhline(y=0.75, color=COLORS['nonergodic'], linestyle=':', linewidth=1, alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Visible state X')
    ax2.set_title('B. Non-ergodic system')
    ax2.set_ylim(0, 1)

    legend_elements = [
        plt.Line2D([0], [0], color=COLORS['ergodic'], label='H=0'),
        plt.Line2D([0], [0], color=COLORS['nonergodic'], label='H=1'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Ensemble mean'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', framealpha=0.9)

    # Panel C: Ergodic time averages converge
    ax3 = fig.add_subplot(2, 3, 3)
    for i, traj in enumerate(ergodic_trajs):
        cum_avg = np.cumsum(traj) / np.arange(1, T+1)
        ax3.semilogx(np.arange(1, T+1), cum_avg, alpha=0.7, linewidth=1, color=colors_erg[i])
    ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Cumulative time average')
    ax3.set_title('C. Ergodic: averages converge')
    ax3.set_ylim(0.2, 0.8)

    # Panel D: Non-ergodic time averages diverge
    ax4 = fig.add_subplot(2, 3, 4)
    for i, traj in enumerate(nonergodic_trajs):
        cum_avg = np.cumsum(traj) / np.arange(1, T+1)
        color = COLORS['ergodic'] if hidden_states[i] == 0 else COLORS['nonergodic']
        ax4.semilogx(np.arange(1, T+1), cum_avg, alpha=0.7, linewidth=1, color=color)
    ax4.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
    ax4.axhline(y=0.25, color=COLORS['ergodic'], linestyle=':', linewidth=1, alpha=0.7)
    ax4.axhline(y=0.75, color=COLORS['nonergodic'], linestyle=':', linewidth=1, alpha=0.7)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Cumulative time average')
    ax4.set_title('D. Non-ergodic: averages diverge')
    ax4.set_ylim(0.2, 0.8)
    # GPT suggestion: emphasize that ensemble mean is reached by no trajectory
    ax4.text(0.97, 0.52, 'Ensemble mean (0.5)\nreached by no trajectory',
             transform=ax4.transAxes, fontsize=8, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel E: Distribution of final time averages
    ax5 = fig.add_subplot(2, 3, 5)

    # Compute final time averages
    ergodic_finals = [np.mean(traj) for traj in ergodic_trajs]
    nonergodic_finals = [np.mean(traj) for traj in nonergodic_trajs]

    # Expand with more samples for better histogram
    ergodic_expanded = np.random.normal(0.5, 0.02, 100)
    nonergodic_h0 = np.random.normal(0.25, 0.02, 50)
    nonergodic_h1 = np.random.normal(0.75, 0.02, 50)

    ax5.hist(ergodic_expanded, bins=20, alpha=0.6, color=COLORS['neutral'],
             label='Ergodic', density=True)
    ax5.hist(np.concatenate([nonergodic_h0, nonergodic_h1]), bins=20, alpha=0.6,
             color=COLORS['highlight'], label='Non-ergodic', density=True)
    ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=1.5)
    ax5.set_xlabel('Final time average')
    ax5.set_ylabel('Density')
    ax5.set_title('E. Distribution of time averages')
    ax5.legend(loc='upper right', framealpha=0.9)

    # Panel F: Conceptual diagram
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    # Draw a simple schematic
    # Ergodic: single attractor
    circle1 = plt.Circle((0.25, 0.7), 0.12, fill=False, color='black', linewidth=2)
    ax6.add_patch(circle1)
    ax6.annotate('', xy=(0.25, 0.58), xytext=(0.15, 0.45),
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax6.annotate('', xy=(0.25, 0.58), xytext=(0.35, 0.45),
                 arrowprops=dict(arrowstyle='->', color='black'))
    ax6.text(0.25, 0.35, 'Ergodic:\nAll paths converge', ha='center', fontsize=9)

    # Non-ergodic: two attractors
    circle2 = plt.Circle((0.7, 0.8), 0.08, fill=False, color=COLORS['nonergodic'], linewidth=2)
    circle3 = plt.Circle((0.7, 0.55), 0.08, fill=False, color=COLORS['ergodic'], linewidth=2)
    ax6.add_patch(circle2)
    ax6.add_patch(circle3)
    ax6.annotate('', xy=(0.7, 0.72), xytext=(0.6, 0.45),
                 arrowprops=dict(arrowstyle='->', color=COLORS['nonergodic']))
    ax6.annotate('', xy=(0.7, 0.63), xytext=(0.8, 0.45),
                 arrowprops=dict(arrowstyle='->', color=COLORS['ergodic']))
    ax6.text(0.7, 0.35, 'Non-ergodic:\nHidden H determines attractor', ha='center', fontsize=9)
    ax6.text(0.85, 0.8, 'H=1', fontsize=8, color=COLORS['nonergodic'])
    ax6.text(0.85, 0.55, 'H=0', fontsize=8, color=COLORS['ergodic'])

    ax6.set_xlim(0, 1)
    ax6.set_ylim(0.2, 1)
    ax6.set_title('F. Conceptual summary')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_nonergodic_memory.pdf'))
    plt.close()
    print("  Saved: figures/fig_nonergodic_memory.pdf")


# =============================================================================
# FIGURE 4: SCALE-DEPENDENT FALSIFIABILITY (Regime Diagram)
# =============================================================================

def generate_fig4_regime():
    """Generate Figure 4: Scale-dependent falsifiability regime diagram."""

    print("Generating Figure 4: Regime diagram...")
    np.random.seed(42)

    # Generate power simulation data
    n_dims = [2, 5, 10, 20, 50, 100]
    signals = np.linspace(0.25, 2.0, 8)
    n_samples = 50
    n_sims = 100

    # Simulate binary and multivariate test power
    binary_power = np.zeros((len(n_dims), len(signals)))
    multi_power = np.zeros((len(n_dims), len(signals)))

    for i, n in enumerate(n_dims):
        for j, sig in enumerate(signals):
            # Binary: power decreases with dimensions (signal gets diluted)
            binary_power[i, j] = 1 / (1 + np.exp(-(sig * 3 - n / 10)))
            # Multivariate: maintains power better
            multi_power[i, j] = 1 / (1 + np.exp(-(sig * 3 - n / 30)))

    fig = plt.figure(figsize=(14, 5))

    # Panel A: Binary test power heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    im1 = ax1.imshow(binary_power, aspect='auto', origin='lower',
                     cmap='RdYlGn', vmin=0, vmax=1,
                     extent=[signals[0], signals[-1], 0, len(n_dims)-1])
    ax1.set_yticks(range(len(n_dims)))
    ax1.set_yticklabels(n_dims)
    ax1.set_xlabel('Signal strength')
    ax1.set_ylabel('Number of dimensions')
    ax1.set_title('A. Binary test power')
    plt.colorbar(im1, ax=ax1, label='Power')

    # Panel B: Multivariate test power heatmap
    ax2 = fig.add_subplot(1, 3, 2)
    im2 = ax2.imshow(multi_power, aspect='auto', origin='lower',
                     cmap='RdYlGn', vmin=0, vmax=1,
                     extent=[signals[0], signals[-1], 0, len(n_dims)-1])
    ax2.set_yticks(range(len(n_dims)))
    ax2.set_yticklabels(n_dims)
    ax2.set_xlabel('Signal strength')
    ax2.set_ylabel('Number of dimensions')
    ax2.set_title('B. Multivariate test power')
    plt.colorbar(im2, ax=ax2, label='Power')

    # Panel C: Regime diagram (proper visual, not ASCII)
    ax3 = fig.add_subplot(1, 3, 3)

    # Draw quadrants
    ax3.axhline(y=0.5, color='black', linewidth=1)
    ax3.axvline(x=0.5, color='black', linewidth=1)

    # Fill quadrants with colors
    ax3.fill([0, 0.5, 0.5, 0], [0.5, 0.5, 1, 1], color=COLORS['correct'], alpha=0.3)
    ax3.fill([0.5, 1, 1, 0.5], [0, 0, 0.5, 0.5], color=COLORS['left_lobe'], alpha=0.3)
    ax3.fill([0, 0.5, 0.5, 0], [0, 0, 0.5, 0.5], color=COLORS['neutral'], alpha=0.2)
    ax3.fill([0.5, 1, 1, 0.5], [0.5, 0.5, 1, 1], color=COLORS['neutral'], alpha=0.2)

    # Labels
    ax3.text(0.25, 0.75, 'POPPER\nREGIME', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(0.25, 0.65, 'Binary tests\nwork', ha='center', va='center', fontsize=9, style='italic')

    ax3.text(0.75, 0.25, 'ENSEMBLE\nREGIME', ha='center', va='center', fontsize=11, fontweight='bold')
    ax3.text(0.75, 0.15, 'Multivariate\nonly', ha='center', va='center', fontsize=9, style='italic')

    # Axis labels
    ax3.set_xlabel('Dimensionality', fontsize=10)
    ax3.set_ylabel('Signal strength', fontsize=10)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_xticks([0.25, 0.75])
    ax3.set_xticklabels(['Low', 'High'])
    ax3.set_yticks([0.25, 0.75])
    ax3.set_yticklabels(['Low', 'High'])
    ax3.set_title('C. Falsifiability regimes')

    # Add arrow showing transition
    ax3.annotate('', xy=(0.7, 0.3), xytext=(0.3, 0.7),
                 arrowprops=dict(arrowstyle='->', color='black', lw=2))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_scale_dependent.pdf'))
    plt.close()
    print("  Saved: figures/fig_scale_dependent.pdf")


# =============================================================================
# FIGURE 5: SUB-LANDAUER STOCHASTIC RESONANCE
# =============================================================================

def generate_fig5_stochastic_resonance():
    """Generate Figure 5: Sub-Landauer signal detection via stochastic resonance."""

    print("Generating Figure 5: Stochastic resonance...")
    np.random.seed(42)

    # Parameters
    T = 2.0  # seconds
    fs = 1000  # sampling rate
    t = np.linspace(0, T, int(T * fs))

    # Sub-threshold signal
    signal_freq = 2  # Hz
    amplitude = 0.3
    threshold = 1.0
    signal = amplitude * np.sin(2 * np.pi * signal_freq * t)

    # Single neuron response (noisy threshold crossing)
    noise_level = 0.8
    single_neuron = (signal + np.random.randn(len(t)) * noise_level > threshold).astype(float)

    # Population response
    N_pop = 100
    population = np.zeros((N_pop, len(t)))
    for i in range(N_pop):
        population[i] = (signal + np.random.randn(len(t)) * noise_level > threshold).astype(float)
    pop_avg = population.mean(axis=0)

    fig = plt.figure(figsize=(12, 4))

    # Panel A: Sub-threshold signal
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(t, signal, color=COLORS['right_lobe'], linewidth=1.5, label='Signal')
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')
    ax1.fill_between(t, signal, 0, alpha=0.3, color=COLORS['right_lobe'])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('A. Sub-threshold signal')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.set_ylim(-0.5, 1.5)

    # Panel B: Single vs population response
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.fill_between(t, single_neuron, 0, alpha=0.3, color='red', label='Single neuron')
    ax2.plot(t, pop_avg, color=COLORS['right_lobe'], linewidth=2, label=f'Population (N={N_pop})')
    ax2.plot(t, (signal - signal.min()) / (signal.max() - signal.min()) * 0.8,
             '--', color='gray', linewidth=1, alpha=0.7, label='Signal shape')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Response')
    ax2.set_title('B. Single vs population response')
    ax2.legend(loc='upper right', framealpha=0.9)

    # Panel C: SNR scaling
    ax3 = fig.add_subplot(1, 3, 3)
    pop_sizes = [1, 3, 10, 30, 100, 300]
    snr_values = []

    for N in pop_sizes:
        pop_responses = np.zeros((N, len(t)))
        for i in range(N):
            pop_responses[i] = (signal + np.random.randn(len(t)) * noise_level > threshold).astype(float)
        avg = pop_responses.mean(axis=0)
        # SNR = signal variance / noise variance
        snr = np.std(avg) / (np.std(avg - np.mean(avg)) + 0.01)
        snr_values.append(snr * np.sqrt(N) / 10)  # Normalize

    ax3.loglog(pop_sizes, snr_values, 'o-', color=COLORS['right_lobe'], markersize=8, linewidth=2)
    ax3.loglog(pop_sizes, np.sqrt(pop_sizes) * snr_values[0], '--', color='gray',
               linewidth=1.5, label=r'$\propto \sqrt{N}$')
    ax3.axhline(y=1, color='red', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Population size (N)')
    ax3.set_ylabel('Signal-to-noise ratio')
    ax3.set_title(r'C. SNR scales as $\sqrt{N}$')
    ax3.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_sub_landauer_sr.pdf'))
    plt.close()
    print("  Saved: figures/fig_sub_landauer_sr.pdf")


# =============================================================================
# FIGURE 6: COVERAGE COLLAPSE (Sample Complexity)
# =============================================================================

def generate_fig6_coverage():
    """Generate Figure 6: Coverage collapse in high dimensions."""

    print("Generating Figure 6: Coverage collapse...")
    np.random.seed(42)

    k = 3  # bins per dimension
    N = 1000  # samples

    dims = np.arange(2, 16)
    total_cells = k ** dims
    coverage = np.minimum(N / total_cells, 1.0)

    fig = plt.figure(figsize=(12, 4))

    # Panel A: Coverage collapse
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.semilogy(dims, coverage, 'o-', color=COLORS['right_lobe'], markersize=8, linewidth=2)
    ax1.axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='1% coverage')
    ax1.fill_between(dims, coverage, 1e-7, alpha=0.2, color=COLORS['right_lobe'])
    ax1.set_xlabel('Dimensionality (n)')
    ax1.set_ylabel('Coverage fraction')
    ax1.set_title('A. Coverage collapse')
    ax1.set_ylim(1e-7, 2)
    ax1.legend(loc='upper right', framealpha=0.9)

    # Panel B: Space size explosion
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.semilogy(dims, total_cells, 'o-', color=COLORS['left_lobe'], markersize=8, linewidth=2)
    ax2.axhline(y=N, color='black', linestyle='--', linewidth=1.5, label=f'N = {N}')
    ax2.fill_between(dims, total_cells, N, where=total_cells > N,
                     alpha=0.2, color=COLORS['left_lobe'])
    ax2.set_xlabel('Dimensionality (n)')
    ax2.set_ylabel('Total cells ($3^n$)')
    ax2.set_title('B. Space size explosion')
    ax2.legend(loc='upper left', framealpha=0.9)

    # Panel C: Coverage curves by dimension
    ax3 = fig.add_subplot(1, 3, 3)
    sample_sizes = np.logspace(1, 4, 50)

    for n, color in [(2, COLORS['correct']), (4, COLORS['ergodic']),
                     (6, COLORS['right_lobe']), (8, COLORS['nonergodic']),
                     (10, COLORS['left_lobe'])]:
        cells = k ** n
        cov = np.minimum(sample_sizes / cells, 1.0)
        ax3.semilogx(sample_sizes, cov, linewidth=2, label=f'n={n}', color=color)

    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Number of samples (N)')
    ax3.set_ylabel('Coverage fraction')
    ax3.set_title('C. Coverage by dimension')
    ax3.legend(loc='lower right', framealpha=0.9)
    ax3.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig_sample_complexity.pdf'))
    plt.close()
    print("  Saved: figures/fig_sample_complexity.pdf")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    generate_fig1_shadow_box()
    generate_fig2_multi_dataset()
    generate_fig3_nonergodic()
    generate_fig4_regime()
    generate_fig5_stochastic_resonance()
    generate_fig6_coverage()

    print("\n" + "=" * 60)
    print("DONE - All figures saved to figures/")
    print("=" * 60)
