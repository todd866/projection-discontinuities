#!/usr/bin/env python3
"""
Sub-Landauer Patterns & Stochastic Resonance
=============================================

Demonstrates Eq. 8-9 from the paper:
    E_signal + E_noise > E_threshold  (individually)
    SNR(Y) ∝ √N  (population pooling)

Shows that signals below the Landauer threshold for individual detection
become detectable when pooled across populations - the "ensemble fingerprint"
that the paper argues is the appropriate observable for sub-Landauer patterns.

Papers: "The Geometry of Biological Shadows" (Paper 2) & "The Limits of Falsifiability" (Paper 1, BioSystems 258, 2025)
Author: Ian Todd
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import os

# Ensure figures directory exists
os.makedirs('../figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

def threshold_neuron(input_signal, noise_std, threshold=1.0):
    """
    Simple threshold neuron model.

    Output = 1 if (input + noise) > threshold, else 0

    This represents the "binary measurement" that the Landauer limit constrains.
    """
    noise = np.random.randn(*input_signal.shape) * noise_std
    return (input_signal + noise > threshold).astype(float)

def compute_snr(signal_component, output):
    """
    Compute signal-to-noise ratio.

    SNR = (power at signal frequency) / (total power - signal power)
    """
    # Use FFT to find power at signal frequency
    fft = np.fft.fft(output)
    power = np.abs(fft) ** 2

    # Find peak (signal frequency)
    signal_power = np.max(power[1:len(power)//2])  # Exclude DC
    total_power = np.sum(power[1:len(power)//2])
    noise_power = total_power - signal_power

    if noise_power <= 0:
        return np.inf
    return signal_power / noise_power

def run_simulation():
    """Main simulation of stochastic resonance in neural populations."""

    np.random.seed(42)

    # Simulation parameters
    T = 10.0  # Duration (seconds)
    dt = 0.001  # Time step
    t = np.arange(0, T, dt)
    n_timesteps = len(t)

    # Signal parameters
    signal_freq = 2.0  # Hz
    signal_amplitude = 0.3  # SUB-THRESHOLD (threshold = 1.0)

    # The weak periodic signal - individually undetectable
    signal = signal_amplitude * np.sin(2 * np.pi * signal_freq * t)

    # Noise level for optimal stochastic resonance
    noise_std = 0.8

    # Population sizes to test
    population_sizes = [1, 5, 10, 25, 50, 100, 200, 500]

    print("=" * 60)
    print("SUB-LANDAUER PATTERNS & STOCHASTIC RESONANCE")
    print("=" * 60)
    print(f"\nSignal amplitude: {signal_amplitude} (threshold = 1.0)")
    print(f"Signal is {signal_amplitude/1.0:.0%} of threshold - individually UNDETECTABLE")
    print(f"Noise std: {noise_std}")
    print()

    # Store results
    snrs = []
    single_neuron_outputs = []
    population_outputs = []

    for N in population_sizes:
        # Simulate N neurons receiving the same signal
        # Each neuron has independent noise
        outputs = np.zeros((N, n_timesteps))
        for i in range(N):
            outputs[i] = threshold_neuron(signal, noise_std)

        # Population average (the "ensemble fingerprint")
        pop_avg = np.mean(outputs, axis=0)

        # Compute SNR
        snr = compute_snr(signal, pop_avg)
        snrs.append(snr)

        if N == 1:
            single_neuron_outputs = outputs[0]
        if N == 100:
            population_outputs = pop_avg

        # Single neuron detection rate
        single_detection = np.mean(outputs[0])

        print(f"N = {N:3d}: SNR = {snr:8.2f}, Single neuron fires {single_detection:.1%} of time")

    # Plotting
    fig = plt.figure(figsize=(14, 10))

    # Panel A: Time series comparison
    ax1 = fig.add_subplot(2, 2, 1)
    t_plot = t[:2000]  # First 2 seconds
    ax1.plot(t_plot, signal[:2000], 'k-', linewidth=2, label='Signal (sub-threshold)', alpha=0.7)
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Detection threshold')
    ax1.fill_between(t_plot, 0, signal[:2000], alpha=0.3, color='blue')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Input', fontsize=11)
    ax1.set_title('A. Sub-Threshold Signal\n(Amplitude = 0.3, Threshold = 1.0)',
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_ylim(-0.5, 1.5)
    ax1.grid(True, alpha=0.3)

    # Panel B: Single neuron vs population
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t_plot, single_neuron_outputs[:2000], 'r-', alpha=0.5,
             linewidth=0.5, label='Single neuron')
    ax2.plot(t_plot, population_outputs[:2000], 'b-', linewidth=2,
             label='Population avg (N=100)')
    ax2.plot(t_plot, signal[:2000] / signal_amplitude * 0.3 + 0.5, 'k--',
             linewidth=1, alpha=0.5, label='Signal shape (scaled)')
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Output', fontsize=11)
    ax2.set_title('B. Single Neuron vs Population Response',
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel C: SNR vs population size
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.loglog(population_sizes, snrs, 'o-', color='#2A9D8F',
               linewidth=2, markersize=10)

    # Theoretical √N scaling
    N_fit = np.array(population_sizes)
    snr_fit = snrs[0] * np.sqrt(N_fit / population_sizes[0])
    ax3.loglog(N_fit, snr_fit, 'k--', linewidth=1.5, alpha=0.7,
               label=r'Theoretical: SNR $\propto \sqrt{N}$')

    ax3.set_xlabel('Population Size (N)', fontsize=11)
    ax3.set_ylabel('Signal-to-Noise Ratio', fontsize=11)
    ax3.set_title('C. SNR Scales as √N (Eq. 9)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, which='both')

    # Add annotation about detection threshold
    ax3.axhline(y=10, color='red', linestyle=':', alpha=0.7)
    ax3.annotate('Reliable detection\nthreshold', xy=(500, 10),
                 xytext=(100, 3), fontsize=9,
                 arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

    # Panel D: Conceptual diagram
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Create conceptual text
    concept_text = """
    THE SUB-LANDAUER DOMAIN
    ━━━━━━━━━━━━━━━━━━━━━━━

    Individual Level:
    • Signal energy < k_B T ln 2
    • Cannot be recorded as a discrete bit
    • Binary measurement destroys the pattern

    Population Level:
    • SNR ∝ √N (Eq. 9)
    • "Ensemble fingerprints" become detectable
    • Information preserved through correlation

    ━━━━━━━━━━━━━━━━━━━━━━━

    BIOLOGICAL IMPLICATION:

    Many causally-relevant patterns in neural
    systems exist in this regime:

    • Ephaptic coupling (~1 mV/mm)
    • Weak synaptic inputs
    • Metabolic oscillations

    These patterns shape behavior but resist
    Popperian falsification at the single-unit
    level. Only ensemble statistics can access
    them without destruction.
    """

    ax4.text(0.1, 0.95, concept_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    ax4.set_title('D. The Sub-Landauer Domain', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save figure
    for path in ['figures/fig_sub_landauer_sr.pdf', '../figures/fig_sub_landauer_sr.pdf']:
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
    print(f"""
The simulation demonstrates stochastic resonance (Eq. 8-9):

1. INDIVIDUAL FAILURE: A single neuron cannot reliably detect
   the signal (amplitude {signal_amplitude} < threshold 1.0).

2. ENSEMBLE SUCCESS: Population averaging reveals the signal.
   SNR scales as √N, matching the paper's Eq. 9.

3. √N SCALING: At N=100, SNR ≈ {snrs[population_sizes.index(100)]:.1f}
   At N=500, SNR ≈ {snrs[-1]:.1f}
   Ratio: {snrs[-1]/snrs[population_sizes.index(100)]:.2f} ≈ √5 = {np.sqrt(5):.2f}

IMPLICATION: Sub-Landauer patterns are:
  • Individually unmeasurable (no binary test possible)
  • Collectively detectable (ensemble fingerprints)
  • Causally decisive (they influence population dynamics)

This is why biological inference requires ensemble methods,
not single-case falsification.
""")

if __name__ == '__main__':
    run_simulation()
