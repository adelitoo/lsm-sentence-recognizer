"""
LSM Sequence Extraction (using TRACE)

This script runs the LSM on the spike-encoded audio and saves the
full, time-series ANALOG TRACE from the reservoir neurons.
"""

import numpy as np
from snn import SNN, SimulationParams # Assuming SNN is in snn.py
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path
import argparse

# --- Network Parameters ---
NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 2
MEMBRANE_THRESHOLD = 2.0
SMALL_WORLD_P = 0.1
SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2) # k=200
TRACE_TAU = 10 

np.random.seed(42)

# --- Theoretical w_critico Function ---
def calculate_theoretical_w_critico(lsm_params, input_data):
    num_samples = min(500, len(input_data))  
    total_spikes = 0
    total_elements = 0
    for sample in input_data[:num_samples]:
        total_spikes += np.sum(sample)
        total_elements += (sample.shape[0] * sample.shape[1])
    if total_elements == 0: return 0.007
    avg_I = total_spikes / total_elements
    beta = lsm_params.small_world_graph_k / 2
    if beta == 0: return 0.007
    numerator = (lsm_params.membrane_threshold - 2 * avg_I * (lsm_params.refractory_period))
    w_critico = numerator / beta
    print("\n--- Theoretical Calculation ---")
    print(f"  Avg Input Rate (I): {avg_I:.6f} (spikes/neuron/timestep)")
    print(f"  Connectivity (beta): {beta:.1f} (k/2)")
    print(f"  Calculated w_critico: {w_critico:.8f}")
    print("-------------------------------")
    return w_critico

# --- Data Loading Function ---
def load_spike_dataset(filename="sentence_spike_trains.npz"):
    print(f"Loading '{filename}'...")
    data_path = Path(filename)
    if not data_path.exists():
        print(f"❌ Error: Dataset '{filename}' not found.")
        print("Please run 'audio_encoding.py' first.")
        return None, None
    data = np.load(data_path)
    X_spikes = data['X_spikes']
    y_labels = data['y_labels']
    print(f"✅ Loaded {len(X_spikes)} samples, shape {X_spikes.shape}")
    return X_spikes, y_labels

# --- Main Execution ---
def main(multiplier: float):
    
    X_spikes, y_labels = load_spike_dataset(filename="sentence_spike_trains.npz")
    if X_spikes is None: return
    
    # Split data (NO STRATIFY for this tiny dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X_spikes, y_labels, test_size=0.2, random_state=42
    )
    
    base_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0, weight_variance=0.0,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_train[0]
    )
            
    w_critico_calculated = calculate_theoretical_w_critico(base_params, X_train)
    optimal_weight = w_critico_calculated * multiplier
            
    print(f"\nUsing weight multiplier: {multiplier:.2f}")
    print(f"  FINAL WEIGHT USED: {optimal_weight:.8f}")

    print(f"\nCreating LSM ({NUM_NEURONS} neurons, {NUM_OUTPUT_NEURONS} outputs)...")
    base_params.mean_weight = optimal_weight
    base_params.weight_variance = optimal_weight * 0.1
    lsm = SNN(simulation_params=base_params)
    
    print("\nExtracting full time-series TRACES...")
    
    all_train_sequences = []
    for sample in tqdm(X_train, desc="Training"):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate(trace_tau=TRACE_TAU, reset_trace=True)
        all_train_sequences.append(lsm.get_trace_history_output().copy())
        
    all_test_sequences = []
    for sample in tqdm(X_test, desc="Testing"):
        lsm.reset()
        lsm.set_input_spike_times(sample)
        lsm.simulate(trace_tau=TRACE_TAU, reset_trace=True)
        # !!! This was the bug - fixed to use the trace history !!!
        all_test_sequences.append(lsm.get_trace_history_output().copy())
        
    # Save as float32, not int8
    X_train_seq = np.array(all_train_sequences, dtype=np.float32)
    X_test_seq = np.array(all_test_sequences, dtype=np.float32)

    print(f"\nExtracted sequences: train_shape={X_train_seq.shape}, test_shape={X_test_seq.shape}")
    
    # Save to the new trace file
    output_file = f"lsm_trace_sequences.npz"
    print(f"\nSaving to '{output_file}'...")
    np.savez_compressed(
        output_file,
        X_train_sequences=X_train_seq,
        y_train=y_train,
        X_test_sequences=X_test_seq,
        y_test=y_test,
        final_weight=optimal_weight,
    )
    
    print("\n" + "="*60)
    print("✅ SEQUENCE EXTRACTION COMPLETE!")
    print(f"Saved to: {output_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract full time-series sequences from an LSM."
    )
    parser.add_argument(
        "--multiplier", 
        type=float, 
        default=0.6, # You will tune this value
        help="Multiplier for w_critico (e.g., 0.6 for 60%%)"
    )
    args = parser.parse_args()
    main(multiplier=args.multiplier)