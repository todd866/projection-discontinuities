#!/usr/bin/env python3
"""
DEMO: NON-ERGODIC MEMORY
========================

THE QUESTION:
Can we escape the curse of dimensionality by measuring longer?
"Just collect more time points and average them."

THE ANSWER:
No. When systems have hidden memory (hidden states that influence
observable dynamics), time averages ≠ ensemble averages.

THE CONCEPT:
A system is ERGODIC if:
    lim(T→∞) [time average] = ensemble average

Non-ergodic systems violate this. Different initial conditions
lead to different long-term averages, even with infinite time.

THE DEMONSTRATION:
We simulate a system with a hidden state that determines which
of two dynamical regimes the observable variable follows.

- Hidden state = 0: Observable tends toward 0.25
- Hidden state = 1: Observable tends toward 0.75
- Ensemble mean = 0.5

No individual trajectory ever reaches 0.5. The "average" is achieved
by NO actual system.

THE IMPLICATION:
"Just measure longer" doesn't help when:
- Cells have epigenetic memory (hidden states)
- Organisms have developmental history
- Ecosystems have path dependence

This is the second leg of the Inference Trilemma:
Time averaging fails because of non-ergodicity.

Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_hidden_state_system(n_trajectories=20, n_steps=500, dt=0.1):
    """
    Simulate a system with hidden states that create non-ergodicity.

    The observable X follows different dynamics depending on
    a hidden state H that we cannot measure.

    H = 0: dX/dt = -k(X - 0.25) + noise
    H = 1: dX/dt = -k(X - 0.75) + noise

    Hidden states are assigned at t=0 and persist forever.
    """
    k = 2.0  # Relaxation rate
    noise_strength = 0.1

    # Assign hidden states (half 0, half 1)
    hidden_states = np.array([i % 2 for i in range(n_trajectories)])

    # Target values based on hidden state
    targets = np.where(hidden_states == 0, 0.25, 0.75)

    # Initialize observables randomly in [0.3, 0.7]
    X = np.random.uniform(0.3, 0.7, n_trajectories)

    # Store trajectories
    trajectories = np.zeros((n_trajectories, n_steps))
    trajectories[:, 0] = X

    # Simulate
    for t in range(1, n_steps):
        # Drift toward target
        dX = -k * (X - targets) * dt + noise_strength * np.sqrt(dt) * np.random.randn(n_trajectories)
        X = X + dX
        X = np.clip(X, 0, 1)  # Keep in [0, 1]
        trajectories[:, t] = X

    return trajectories, hidden_states, targets


def run_nonergodic_demo():
    """Demonstrate non-ergodicity with hidden states."""

    print("=" * 60)
    print("NON-ERGODIC MEMORY")
    print("Why 'just measure longer' doesn't work")
    print("=" * 60)

    np.random.seed(42)

    # Simulate
    print("\nSimulating system with hidden states...")
    trajectories, hidden_states, targets = simulate_hidden_state_system(
        n_trajectories=20, n_steps=500
    )

    n_traj, n_steps = trajectories.shape
    time = np.arange(n_steps) * 0.1

    # Compute time averages for each trajectory (last half only - after equilibration)
    equilibration = n_steps // 2
    time_averages = np.mean(trajectories[:, equilibration:], axis=1)

    # Compute ensemble average at each time point
    ensemble_average = np.mean(trajectories, axis=0)

    # Statistics
    h0_avg = np.mean(time_averages[hidden_states == 0])
    h1_avg = np.mean(time_averages[hidden_states == 1])
    true_ensemble = 0.5  # (0.25 + 0.75) / 2

    print(f"\nRESULTS:")
    print(f"  Trajectories with H=0: time average → {h0_avg:.3f} (target: 0.25)")
    print(f"  Trajectories with H=1: time average → {h1_avg:.3f} (target: 0.75)")
    print(f"  Ensemble average:                     {true_ensemble:.3f}")
    print(f"\n  Gap between trajectory groups: {abs(h1_avg - h0_avg):.3f}")

    # Check how many trajectories are close to the ensemble mean
    near_ensemble = np.sum(np.abs(time_averages - true_ensemble) < 0.1)
    print(f"  Trajectories with time avg ≈ 0.5:     {near_ensemble}/{n_traj}")

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Individual trajectories
    ax = axes[0, 0]
    for i in range(n_traj):
        color = '#E63946' if hidden_states[i] == 0 else '#457B9D'
        alpha = 0.5
        ax.plot(time, trajectories[i], color=color, alpha=alpha, linewidth=0.8)

    ax.axhline(y=0.25, color='#E63946', linestyle='--', linewidth=2, label='Target (H=0)')
    ax.axhline(y=0.75, color='#457B9D', linestyle='--', linewidth=2, label='Target (H=1)')
    ax.axhline(y=0.5, color='black', linestyle=':', linewidth=2, label='Ensemble mean')
    ax.set_xlabel('Time')
    ax.set_ylabel('Observable X')
    ax.set_title('A. Individual Trajectories\n(Color = hidden state)', fontweight='bold')
    ax.legend(loc='right')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Panel B: Time averages distribution
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 25)
    ax.hist(time_averages[hidden_states == 0], bins=bins, alpha=0.7,
            color='#E63946', label=f'H=0 (mean={h0_avg:.2f})', edgecolor='black')
    ax.hist(time_averages[hidden_states == 1], bins=bins, alpha=0.7,
            color='#457B9D', label=f'H=1 (mean={h1_avg:.2f})', edgecolor='black')
    ax.axvline(x=0.5, color='black', linestyle=':', linewidth=2, label='Ensemble mean')
    ax.set_xlabel('Time Average of X')
    ax.set_ylabel('Count')
    ax.set_title('B. Distribution of Time Averages\n(Two distinct peaks, NOT at ensemble mean)',
                 fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel C: Ensemble average over time
    ax = axes[1, 0]
    ax.plot(time, ensemble_average, color='black', linewidth=2, label='Ensemble average')
    ax.fill_between(time,
                    np.percentile(trajectories, 25, axis=0),
                    np.percentile(trajectories, 75, axis=0),
                    alpha=0.3, color='gray', label='IQR')
    ax.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='True ensemble mean')
    ax.set_xlabel('Time')
    ax.set_ylabel('Observable X')
    ax.set_title('C. Ensemble Average Over Time\n(Converges to 0.5, but NO trajectory does)',
                 fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Panel D: Summary
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
    NON-ERGODICITY DEMONSTRATION
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    THE SETUP:
    • Observable X follows different dynamics based on
      hidden state H (which we cannot measure)
    • H=0 → X tends toward 0.25
    • H=1 → X tends toward 0.75
    • Ensemble mean = 0.50

    THE RESULT:
    • Time avg for H=0 trajectories: {h0_avg:.3f}
    • Time avg for H=1 trajectories: {h1_avg:.3f}
    • Trajectories near ensemble mean: {near_ensemble}/{n_traj}

    THE PROBLEM:
    The ensemble mean (0.5) is achieved by
    NO ACTUAL TRAJECTORY.

    It is a statistical ghost—an average over
    systems that don't exist.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    IMPLICATION FOR BIOLOGY:

    When cells have epigenetic memory,
    organisms have developmental history,
    or ecosystems have path dependence:

    "Just measure longer" DOES NOT HELP.

    Time averaging assumes ergodicity.
    Non-ergodic systems break this assumption.

    This is the 2nd leg of the Inference Trilemma:
    Time averaging fails → Non-ergodicity
    """

    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.9))
    ax.set_title('D. Summary', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fig_nonergodic_memory.pdf', dpi=150, bbox_inches='tight')
    print("\nSaved: fig_nonergodic_memory.pdf")

    plt.show()

    return {
        'trajectories': trajectories,
        'hidden_states': hidden_states,
        'time_averages': time_averages,
        'h0_avg': h0_avg,
        'h1_avg': h1_avg
    }


if __name__ == '__main__':
    results = run_nonergodic_demo()
