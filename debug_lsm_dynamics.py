"""
Debug LSM dynamics to understand why traces are so similar
"""
import numpy as np
from snnpy.snn import SNN, SimulationParams
from pathlib import Path

def load_spike_dataset():
    data = np.load("sentence_spike_trains_500.npz")
    return data['X_spikes'], data['y_labels']

def simulate_and_analyze(multiplier, num_samples=5):
    """Simulate LSM with different multiplier and analyze dynamics"""

    print(f"\n{'='*60}")
    print(f"Testing multiplier = {multiplier}")
    print(f"{'='*60}")

    # Load data
    X_spikes, y_labels = load_spike_dataset()

    # LSM parameters
    NUM_NEURONS = 2000
    NUM_OUTPUT_NEURONS = 700
    SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2)

    base_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0,
        weight_variance=0.0,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=2.0,
        leak_coefficient=0,  # ← THE PROBLEM
        refractory_period=2,
        small_world_graph_p=0.2,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_spikes[0]
    )

    # Calculate weight
    total_spikes = 0
    total_elements = 0
    for sample in X_spikes[:500]:
        total_spikes += np.sum(sample)
        total_elements += (sample.shape[0] * sample.shape[1])
    avg_I = total_spikes / total_elements
    beta = SMALL_WORLD_K / 2
    numerator = (2.0 - 2 * avg_I * 2)  # threshold - 2*I*refractory
    w_critico = numerator / beta

    final_weight = w_critico * multiplier

    print(f"  w_critico: {w_critico:.8f}")
    print(f"  final_weight: {final_weight:.8f}")

    # Create LSM
    base_params.mean_weight = final_weight
    base_params.weight_variance = final_weight * 0.1
    lsm = SNN(simulation_params=base_params)

    # Simulate multiple samples and track key metrics
    all_spike_counts = []
    all_mean_potentials = []
    all_max_potentials = []

    for i in range(num_samples):
        lsm.reset()
        lsm.set_input_spike_times(X_spikes[i])

        # Run simulation and track membrane potentials
        num_timesteps = X_spikes[i].shape[1]
        output_neurons = lsm.output_neurons

        mem_trace = np.zeros((num_timesteps, len(output_neurons)), dtype=np.float32)
        spike_counts = np.zeros(len(output_neurons), dtype=int)

        # Access internal state
        T = num_timesteps
        Nin = lsm.num_input_neurons
        inputs = X_spikes[i]
        mem = lsm.membrane_potentials
        refr = lsm.refractory_timer
        out_idx = output_neurons
        curr_amp = np.float32(lsm.current_amplitude)

        W = lsm.synaptic_weights.tocsr()
        indptr, indices, data = W.indptr, W.indices, W.data

        for t in range(T):
            # Decay refractory
            refr -= lsm.time_step
            np.clip(refr, 0, None, out=refr)

            # Apply input
            if t < inputs.shape[1]:
                spikes_t = inputs[:, t]
                mem[:Nin] += curr_amp * spikes_t

            # Check for spikes
            spiking_mask = (mem >= lsm.membrane_threshold) & (refr == 0)

            if spiking_mask.any():
                # Count spikes in output neurons
                for idx in out_idx:
                    if spiking_mask[idx]:
                        spike_counts[out_idx == idx] += 1

                # Reset
                mem[spiking_mask] = 0.0
                refr[spiking_mask] = lsm.refractory_period + 1

            # NO LEAK (this is the issue!)
            # mem *= 1.0  # leak_coefficient = 0

            # Propagate spikes
            if spiking_mask.any():
                for j in np.flatnonzero(spiking_mask):
                    start, end = indptr[j], indptr[j + 1]
                    if start != end:
                        cols = indices[start:end]
                        mem[cols] += data[start:end]

            # Record
            mem_trace[t, :] = mem[out_idx]

        # Analyze this sample
        all_spike_counts.append(spike_counts.sum())
        all_mean_potentials.append(mem_trace.mean())
        all_max_potentials.append(mem_trace.max())

    # Summary statistics
    print(f"\n  Output neuron spiking:")
    print(f"    Mean spikes per sample: {np.mean(all_spike_counts):.1f}")
    print(f"    Std: {np.std(all_spike_counts):.1f}")

    print(f"\n  Membrane potentials:")
    print(f"    Mean across samples: {np.mean(all_mean_potentials):.3f}")
    print(f"    Std: {np.std(all_mean_potentials):.3f}")
    print(f"    Max across samples: {np.mean(all_max_potentials):.3f}")

    # Check if potentials grow unbounded (sign of no leak)
    if np.mean(all_max_potentials) > 10.0:
        print(f"\n  ⚠️  WARNING: Potentials growing very large!")
        print(f"     This suggests unbounded accumulation (no leak)")

    # Check variation between samples
    if np.std(all_mean_potentials) < 0.05:
        print(f"\n  ⚠️  WARNING: Very low variation between samples!")
        print(f"     Std = {np.std(all_mean_potentials):.4f} (too low)")

def main():
    print("="*60)
    print("DEBUGGING LSM DYNAMICS")
    print("="*60)

    # Test different multipliers
    for mult in [0.5, 0.8, 1.0]:
        simulate_and_analyze(mult, num_samples=10)

    print("\n" + "="*60)
    print("DIAGNOSIS:")
    print("="*60)
    print("\nIf you see:")
    print("  1. Very low variation between samples → LSM not discriminative")
    print("  2. Similar statistics for all multipliers → Weight changes don't matter")
    print("  3. Very high max potentials → No leak causing unbounded growth")
    print("\nThe solution:")
    print("  → Add leak! Change LEAK_COEFFICIENT from 0 to ~0.1-0.3")
    print("="*60)

if __name__ == "__main__":
    main()
