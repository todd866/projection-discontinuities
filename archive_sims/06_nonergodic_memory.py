#!/usr/bin/env python3
"""
Non-Ergodic Memory: When Time Averages Fail
============================================

Demonstrates why time averaging fails when hidden dimensions carry memory.

THE CONCEPT:
- Ergodic systems: time average = ensemble average (sample long enough, you see everything)
- Non-ergodic systems: time average ≠ ensemble average (history constrains future)

THE SIMULATION:
We create a system with:
1. A VISIBLE state X (what we measure)
2. A HIDDEN state H (sub-Landauer memory - we can't see it)
3. Dynamics where H affects X, but H is stable (memory persists)

An observer seeing only X will find that:
- Different trajectories give different time averages
- No single trajectory converges to the true ensemble average
- The system "remembers" in dimensions we cannot see

THE FUTURE VERSION:
Replace this toy model with:
- Single-cell RNA-seq trajectories (hidden epigenetic state)
- Neural population recordings (hidden neuromodulatory state)
- Patient longitudinal data (hidden disease substates)

Paper: "The Geometry of Biological Shadows" (Paper 2)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)


def run_ergodic_system(n_steps=10000, seed=None):
    """
    An ergodic system: simple random walk that explores all states.
    Time average converges to ensemble average regardless of initial condition.
    """
    if seed is not None:
        np.random.seed(seed)

    # Simple symmetric random walk on [0, 1]
    x = np.zeros(n_steps)
    x[0] = np.random.rand()

    for t in range(1, n_steps):
        # Mean-reverting random walk (Ornstein-Uhlenbeck-like)
        # REDUCED NOISE for cleaner visualization
        x[t] = 0.98 * x[t-1] + 0.02 * 0.5 + 0.03 * np.random.randn()
        x[t] = np.clip(x[t], 0, 1)

    return x


def run_nonergodic_system(n_steps=10000, hidden_state=None, seed=None):
    """
    A non-ergodic system: dynamics depend on a HIDDEN state that persists.

    The hidden state H ∈ {0, 1} determines which attractor the visible
    state X is drawn toward. H is set at initialization and never changes
    (it's the "memory" - like an epigenetic mark or a developmental decision).

    An observer seeing only X cannot determine H directly, and their
    time average will depend on which H they happened to start with.
    """
    if seed is not None:
        np.random.seed(seed)

    # Hidden state: determines which attractor X is drawn to
    if hidden_state is None:
        H = np.random.choice([0, 1])
    else:
        H = hidden_state

    # Attractor locations depend on hidden state
    attractor = 0.25 if H == 0 else 0.75  # More separated

    # Visible state dynamics
    x = np.zeros(n_steps)
    x[0] = 0.5  # Start at center so divergence is clear

    for t in range(1, n_steps):
        # Pulled toward attractor determined by hidden state
        # REDUCED NOISE, STRONGER PULL for cleaner visualization
        x[t] = 0.98 * x[t-1] + 0.02 * attractor + 0.02 * np.random.randn()
        x[t] = np.clip(x[t], 0, 1)

    return x, H


def compute_running_average(trajectory):
    """Compute cumulative time average."""
    return np.cumsum(trajectory) / np.arange(1, len(trajectory) + 1)


def run_simulation():
    """Main simulation demonstrating non-ergodicity from hidden memory."""

    np.random.seed(42)
    n_steps = 10000
    n_trajectories = 20

    print("=" * 60)
    print("NON-ERGODIC MEMORY: WHEN TIME AVERAGES FAIL")
    print("=" * 60)

    # =========================================================================
    # Part 1: Ergodic system - time averages converge
    # =========================================================================
    print("\n[1] ERGODIC SYSTEM (no hidden memory)")
    print("-" * 40)

    ergodic_trajectories = []
    ergodic_time_averages = []

    for i in range(n_trajectories):
        traj = run_ergodic_system(n_steps, seed=i)
        ergodic_trajectories.append(traj)
        ergodic_time_averages.append(np.mean(traj))

    ergodic_ensemble_mean = np.mean([np.mean(t) for t in ergodic_trajectories])
    ergodic_spread = np.std(ergodic_time_averages)

    print(f"  Ensemble average: {ergodic_ensemble_mean:.4f}")
    print(f"  Spread of time averages: {ergodic_spread:.4f}")
    print(f"  All trajectories converge to similar values: ✓")

    # =========================================================================
    # Part 2: Non-ergodic system - time averages depend on hidden state
    # =========================================================================
    print("\n[2] NON-ERGODIC SYSTEM (hidden memory)")
    print("-" * 40)

    nonergodic_trajectories = []
    nonergodic_time_averages = []
    hidden_states = []

    for i in range(n_trajectories):
        traj, H = run_nonergodic_system(n_steps, seed=i)
        nonergodic_trajectories.append(traj)
        nonergodic_time_averages.append(np.mean(traj))
        hidden_states.append(H)

    # Separate by hidden state
    avg_H0 = np.mean([nonergodic_time_averages[i] for i in range(n_trajectories) if hidden_states[i] == 0])
    avg_H1 = np.mean([nonergodic_time_averages[i] for i in range(n_trajectories) if hidden_states[i] == 1])
    nonergodic_ensemble_mean = np.mean(nonergodic_time_averages)
    nonergodic_spread = np.std(nonergodic_time_averages)

    print(f"  Trajectories with H=0: time avg → {avg_H0:.4f}")
    print(f"  Trajectories with H=1: time avg → {avg_H1:.4f}")
    print(f"  Ensemble average (both): {nonergodic_ensemble_mean:.4f}")
    print(f"  Spread of time averages: {nonergodic_spread:.4f}")
    print(f"  Time averages DIVERGE based on hidden state: ✗")

    # =========================================================================
    # Part 3: The epistemological problem
    # =========================================================================
    print("\n[3] THE EPISTEMOLOGICAL PROBLEM")
    print("-" * 40)
    print("""
  An observer seeing only X faces a dilemma:

  - Their time average converges... but to WHICH value?
  - It depends on the hidden state H they cannot observe
  - The ensemble average (0.5) is not achieved by ANY trajectory
  - "Average over time" gives a biased answer that depends on history

  This is why non-ergodicity breaks statistical inference:
  You cannot substitute time averaging for ensemble averaging
  when the system remembers in dimensions you cannot see.
    """)

    # =========================================================================
    # Plotting
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel A: Ergodic trajectories - show 3 clear trajectories over longer time
    ax1 = axes[0, 0]
    for i, traj in enumerate(ergodic_trajectories[:3]):
        ax1.plot(traj[:2000], alpha=0.8, linewidth=1.2)
    ax1.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Ensemble mean')
    ax1.set_xlabel('Time', fontsize=11)
    ax1.set_ylabel('Visible state X', fontsize=11)
    ax1.set_title('A. Ergodic System\n(all trajectories explore same region)', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Panel B: Non-ergodic trajectories - show clear divergence
    ax2 = axes[0, 1]
    # Pick 2 H=0 and 2 H=1 trajectories explicitly
    h0_idx = [i for i in range(n_trajectories) if hidden_states[i] == 0][:2]
    h1_idx = [i for i in range(n_trajectories) if hidden_states[i] == 1][:2]
    for i in h0_idx:
        ax2.plot(nonergodic_trajectories[i][:2000], alpha=0.8, linewidth=1.2, color='#E63946')
    for i in h1_idx:
        ax2.plot(nonergodic_trajectories[i][:2000], alpha=0.8, linewidth=1.2, color='#2A9D8F')
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Ensemble mean')
    ax2.axhline(y=0.25, color='#E63946', linestyle=':', linewidth=2, label='H=0 attractor')
    ax2.axhline(y=0.75, color='#2A9D8F', linestyle=':', linewidth=2, label='H=1 attractor')
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel('Visible state X', fontsize=11)
    ax2.set_title('B. Non-Ergodic System\n(trajectories diverge based on hidden H)', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='right', fontsize=9)

    # Panel C: Convergence of time averages (ergodic)
    ax3 = axes[0, 2]
    for i, traj in enumerate(ergodic_trajectories[:5]):
        running_avg = compute_running_average(traj)
        ax3.plot(running_avg, alpha=0.7, linewidth=1)
    ax3.axhline(y=0.5, color='black', linestyle='--', linewidth=2)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel('Cumulative time average', fontsize=11)
    ax3.set_title('C. Ergodic: Time Averages Converge\n(all → ensemble mean)', fontsize=12, fontweight='bold')
    ax3.set_ylim(0.2, 0.8)
    ax3.set_xscale('log')

    # Panel D: Convergence of time averages (non-ergodic) - THE KEY FIGURE
    ax4 = axes[1, 0]
    colors = ['#E63946' if hidden_states[i] == 0 else '#2A9D8F' for i in range(n_trajectories)]
    for i, traj in enumerate(nonergodic_trajectories[:10]):
        running_avg = compute_running_average(traj)
        ax4.plot(running_avg, alpha=0.7, linewidth=1.2, color=colors[i])
    ax4.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Ensemble mean')
    ax4.axhline(y=0.25, color='#E63946', linestyle=':', linewidth=2)
    ax4.axhline(y=0.75, color='#2A9D8F', linestyle=':', linewidth=2)
    ax4.set_xlabel('Time', fontsize=11)
    ax4.set_ylabel('Cumulative time average', fontsize=11)
    ax4.set_title('D. Non-Ergodic: Time Averages DIVERGE\n(depend on hidden H)', fontsize=12, fontweight='bold')
    ax4.set_ylim(0.15, 0.85)
    ax4.set_xscale('log')

    # Panel E: Distribution of final time averages
    ax5 = axes[1, 1]
    ax5.hist(ergodic_time_averages, bins=15, alpha=0.5, label='Ergodic', color='gray', density=True)
    ax5.hist(nonergodic_time_averages, bins=15, alpha=0.5, label='Non-ergodic', color='purple', density=True)
    ax5.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    ax5.set_xlabel('Final time average', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('E. Distribution of Time Averages\n(ergodic=tight, non-ergodic=bimodal)', fontsize=12, fontweight='bold')
    ax5.legend()

    # Panel F: Conceptual diagram
    ax6 = axes[1, 2]
    ax6.axis('off')

    concept_text = """
    WHY TIME AVERAGING FAILS
    ━━━━━━━━━━━━━━━━━━━━━━━━

    ERGODIC SYSTEM:
    • No hidden memory
    • All trajectories explore same states
    • Time avg → Ensemble avg as T → ∞
    • Inference: just measure longer

    NON-ERGODIC SYSTEM:
    • Hidden state H carries memory
    • H constrains which states are visited
    • Time avg → H-dependent value
    • Inference: measuring longer doesn't help!

    ━━━━━━━━━━━━━━━━━━━━━━━━

    THE BIOLOGICAL CASE:

    If sub-Landauer structure (epigenetic marks,
    conformational states, developmental history)
    encodes memory that you cannot observe:

    → Your time averages are BIASED
    → The bias depends on HISTORY
    → You cannot detect the bias
    → "Just collect more data" doesn't fix it

    This is why non-ergodicity breaks
    the classical inference toolkit.
    """

    ax6.text(0.05, 0.95, concept_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax6.set_title('F. The Epistemological Point', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    for path in ['figures/fig_nonergodic_memory.pdf', '../figures/fig_nonergodic_memory.pdf']:
        try:
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\nFigure saved to {path}")
            break
        except:
            continue

    plt.show()

    # =========================================================================
    # Quantitative summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("QUANTITATIVE SUMMARY")
    print("=" * 60)

    # Statistical test: are the two groups of non-ergodic trajectories different?
    group_H0 = [nonergodic_time_averages[i] for i in range(n_trajectories) if hidden_states[i] == 0]
    group_H1 = [nonergodic_time_averages[i] for i in range(n_trajectories) if hidden_states[i] == 1]

    t_stat, p_value = stats.ttest_ind(group_H0, group_H1)

    print(f"""
    ERGODIC SYSTEM:
      Spread of time averages (σ): {ergodic_spread:.4f}
      All converge to: ~{ergodic_ensemble_mean:.3f}

    NON-ERGODIC SYSTEM:
      Spread of time averages (σ): {nonergodic_spread:.4f}
      H=0 trajectories converge to: ~{avg_H0:.3f}
      H=1 trajectories converge to: ~{avg_H1:.3f}
      Difference: {abs(avg_H0 - avg_H1):.3f}
      t-test p-value: {p_value:.2e}

    KEY INSIGHT:
      In the non-ergodic case, trajectories converge—but to DIFFERENT
      values depending on hidden state. An observer cannot know which
      value their trajectory is converging to, because they cannot
      observe the hidden state that determines it.

      This is not "noise" or "insufficient data." It is a structural
      feature of systems with memory in unobserved dimensions.
    """)

    # =========================================================================
    # Future directions
    # =========================================================================
    print("\n" + "=" * 60)
    print("FUTURE VERSIONS OF THIS SIMULATION")
    print("=" * 60)
    print("""
    This toy model demonstrates the principle. Future versions would:

    1. SINGLE-CELL TRAJECTORIES
       - Use real scRNA-seq time courses
       - Hidden state = epigenetic/chromatin configuration
       - Show that cells with different histories have different attractors

    2. NEURAL POPULATION DYNAMICS
       - Use Stringer et al. or Allen Institute recordings
       - Hidden state = neuromodulatory tone, synaptic weights
       - Show that "same stimulus" gives different responses based on history

    3. PATIENT LONGITUDINAL DATA
       - Use disease progression cohorts
       - Hidden state = unobserved disease subtype or comorbidity
       - Show that "same treatment" gives different outcomes based on history

    4. PROTEIN CONFORMATIONAL DYNAMICS
       - Use molecular dynamics trajectories
       - Hidden state = rare conformational substates
       - Show that time averages depend on which substate was sampled

    The principle is the same: when systems have memory in dimensions
    you cannot observe, time averaging gives biased, history-dependent
    results that you cannot detect or correct from the observed data alone.
    """)


if __name__ == '__main__':
    run_simulation()
