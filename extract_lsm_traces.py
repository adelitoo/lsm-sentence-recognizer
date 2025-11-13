"""
LSM Trace Extraction with SENTENCE-LEVEL SPLIT (500 Sentences)

This version extracts MEMBRANE POTENTIAL TRACES (not spike-based features) from LSM output neurons.
Uses sentence-level split for true generalization testing.

Key difference from feature extraction:
- Records membrane voltages over time (continuous analog signal)
- No windowing, no aggregation
- Direct input to CTC/GRU for temporal learning

*** MODIFIED: Includes a cosine similarity test for trace separability ***
"""

import numpy as np
from snnpy.snn import SNN, SimulationParams
from tqdm import tqdm
from pathlib import Path
import argparse

# --- NEW IMPORTS ---
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# -------------------

# --- Network Parameters (matching the spike feature version) ---
NUM_NEURONS = 1000
NUM_OUTPUT_NEURONS = 400
LEAK_COEFFICIENT = 0
REFRACTORY_PERIOD = 2
MEMBRANE_THRESHOLD = 2.0
SMALL_WORLD_P = 0.1
SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2)

np.random.seed(42)


def calculate_theoretical_w_critico(lsm_params, input_data):
    """Calculate theoretical critical weight"""
    num_samples = min(500, len(input_data))
    total_spikes = 0
    total_elements = 0
    for sample in input_data[:num_samples]:
        total_spikes += np.sum(sample)
        total_elements += (sample.shape[0] * sample.shape[1])
    if total_elements == 0:
        return 0.007
    avg_I = total_spikes / total_elements
    beta = lsm_params.small_world_graph_k / 2
    if beta == 0:
        return 0.007
    numerator = (lsm_params.membrane_threshold - 2 * avg_I * lsm_params.refractory_period)
    w_critico = numerator / beta
    print("\n--- Theoretical Calculation ---")
    print(f"  Avg Input Rate (I): {avg_I:.6f} (spikes/neuron/timestep)")
    print(f"  Connectivity (beta): {beta:.1f} (k/2)")
    print(f"  Calculated w_critico: {w_critico:.8f}")
    print("-------------------------------")
    return w_critico


def load_spike_dataset(filename="sentence_spike_trains.npz"):
    """Load spike train dataset"""
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

# --- NEW HELPER FUNCTION ---
def load_label_map(filepath="sentence_label_map_500.txt"):
    """Loads the 'sentence_label_map.txt' file into a dictionary."""
    label_map_path = Path(filepath)
    if not label_map_path.exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        print("   Cannot perform similarity test without it.")
        return None
    try:
        df = pd.read_csv(label_map_path)
        return df.set_index('label_id')['label_text'].to_dict()
    except Exception as e:
        print(f"Error loading label map: {e}")
        return None
# ---------------------------

def split_by_sentence(X_spikes, y_labels, test_size=0.2, random_state=42):
    """
    Split dataset by sentence, not by sample.
    Ensures all augmented versions of a sentence stay together in train or test.
    """
    print("\n" + "="*60)
    print("SENTENCE-LEVEL SPLIT (TRUE GENERALIZATION)")
    print("="*60)

    unique_sentence_ids = np.unique(y_labels)
    num_sentences = len(unique_sentence_ids)

    print(f"\nTotal unique sentences: {num_sentences}")
    print(f"Total samples: {len(X_spikes)}")

    # Load BALANCED split sentence IDs (ensures better word coverage)
    balanced_split_file = "balanced_sentence_split.npz"
    if Path(balanced_split_file).exists():
        print(f"\n✅ Loading BALANCED split from '{balanced_split_file}'")
        print(f"   (Maximizes word coverage for better generalization)")
        split_data = np.load(balanced_split_file)
        train_sentence_ids = split_data['train_sentence_ids']
        test_sentence_ids = split_data['test_sentence_ids']
    else:
        print(f"\n⚠️  Balanced split file not found, using random split")
        from sklearn.model_selection import train_test_split
        train_sentence_ids, test_sentence_ids = train_test_split(
            unique_sentence_ids,
            test_size=test_size,
            random_state=random_state
        )

    print(f"\nSplit configuration:")
    print(f"  Train sentences: {len(train_sentence_ids)} ({(1-test_size)*100:.0f}%)")
    print(f"  Test sentences:  {len(test_sentence_ids)} ({test_size*100:.0f}%)")

    # Create masks for train and test
    train_mask = np.isin(y_labels, train_sentence_ids)
    test_mask = np.isin(y_labels, test_sentence_ids)

    # Split data
    X_train = X_spikes[train_mask]
    y_train = y_labels[train_mask]
    X_test = X_spikes[test_mask]
    y_test = y_labels[test_mask]

    print(f"\nResulting split:")
    print(f"  Train samples: {len(X_train)} ({len(X_train)/len(X_spikes)*100:.1f}%)")
    print(f"  Test samples:  {len(X_test)} ({len(X_test)/len(X_spikes)*100:.1f}%)")

    # Verify no sentence overlap
    train_sentences_set = set(train_sentence_ids)
    test_sentences_set = set(test_sentence_ids)
    overlap = train_sentences_set & test_sentences_set

    if overlap:
        print(f"\n⚠️  WARNING: {len(overlap)} sentences appear in both train and test!")
    else:
        print(f"\n✅ VERIFIED: No sentence overlap between train and test!")
        print(f"   Test set contains COMPLETELY UNSEEN sentences.")

    print("="*60 + "\n")

    return X_train, X_test, y_train, y_test


def extract_membrane_traces(lsm, spike_sample, output_neurons):
    """
    Extract membrane potential traces from LSM output neurons over time.

    This requires modifying the simulation to record membrane potentials
    at each timestep for the output neurons.

    Returns: (timesteps, num_output_neurons) array of membrane voltages
    """
    lsm.reset()
    lsm.set_input_spike_times(spike_sample)

    # We need to manually run the simulation and record membrane potentials
    # Since snnpy doesn't have built-in trace recording, we'll extract it ourselves

    num_timesteps = spike_sample.shape[1]
    num_output = len(output_neurons)
    membrane_trace = np.zeros((num_timesteps, num_output), dtype=np.float32)

    # Access internal state during simulation
    T = num_timesteps
    N = lsm.num_neurons
    Nin = lsm.num_input_neurons
    inputs = spike_sample
    mem = lsm.membrane_potentials
    refr = lsm.refractory_timer

    # CRITICAL FIX: Zero out membrane potentials and refractory timers
    mem[:] = 0.0
    refr[:] = 0

    out_idx = output_neurons
    leak_factor = np.float32(1.0 - lsm.leak_coefficient)
    curr_amp = np.float32(lsm.current_amplitude)

    W = lsm.synaptic_weights.tocsr()
    indptr, indices, data = W.indptr, W.indices, W.data

    for t in range(T):
        # Decay refractory timers
        refr -= lsm.time_step
        np.clip(refr, 0, None, out=refr)

        # Apply input spikes
        if t < inputs.shape[1]:
            spikes_t = inputs[:, t]
            mem[:Nin] += curr_amp * spikes_t

        # Check for spiking neurons
        spiking_mask = (mem >= lsm.membrane_threshold) & (refr == 0)

        if spiking_mask.any():
            spk_idx = np.flatnonzero(spiking_mask)

            # Reset spiking neurons
            mem[spiking_mask] = 0.0
            refr[spiking_mask] = lsm.refractory_period + 1

        # Apply leak
        mem *= leak_factor

        # Propagate spikes through network
        if spiking_mask.any():
            for j in np.flatnonzero(spiking_mask):
                start, end = indptr[j], indptr[j + 1]
                if start != end:
                    cols = indices[start:end]
                    mem[cols] += data[start:end]

        # Record membrane potentials of output neurons
        membrane_trace[t, :] = mem[out_idx]

    return membrane_trace


def extract_dataset_traces(lsm, spike_data, desc=""):
    """Extract membrane traces for entire dataset"""
    all_traces = []
    for sample in tqdm(spike_data, desc=desc):
        traces = extract_membrane_traces(lsm, sample, lsm.output_neurons)
        all_traces.append(traces)
    return np.array(all_traces, dtype=np.float32)


def main(multiplier: float):

    # Load spike dataset
    X_spikes, y_labels = load_spike_dataset(filename="sentence_spike_trains.npz")
    if X_spikes is None:
        return

    # Split data BY SENTENCE (not by sample!)
    X_train, X_test, y_train, y_test = split_by_sentence(
        X_spikes, y_labels, test_size=0.2, random_state=42
    )

    # Create LSM parameters
    base_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0,
        weight_variance=0.0,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=LEAK_COEFFICIENT,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_train[0]
    )

    # Calculate critical weight
    w_critico_calculated = calculate_theoretical_w_critico(base_params, X_train)
    optimal_weight = w_critico_calculated * multiplier

    print(f"\nUsing weight multiplier: {multiplier:.2f}")
    print(f"  FINAL WEIGHT USED: {optimal_weight:.8f}")

    # Create LSM
    print(f"\nCreating LSM ({NUM_NEURONS} neurons, {NUM_OUTPUT_NEURONS} outputs)...")
    base_params.mean_weight = optimal_weight
    base_params.weight_variance = optimal_weight * 0.1
    lsm = SNN(simulation_params=base_params)

    # ####################################################################
    # ### START: LSM TRACE SEPARABILITY DEBUGGING BLOCK ###
    # ####################################################################
    print("\n" + "="*60)
    print("🔬 RUNNING LSM TRACE SEPARABILITY TEST")
    print("="*60)

    label_map = load_label_map()
    if label_map is None:
        print("Skipping similarity test as label map could not be loaded.")
    else:
        try:
            # --- 1. Find samples to compare ---
            
            # Sample A: The first training sample
            idx_A = 0
            label_A = y_train[idx_A]
            text_A = label_map[label_A]
            
            # Sample B: The first sample that is a DIFFERENT sentence
            idx_B = np.where(y_train != label_A)[0][0]
            label_B = y_train[idx_B]
            text_B = label_map[label_B]
            
            # Sample C: A different sample of the SAME sentence as A (an augmentation)
            # Find indices that match label A but are not index 0
            same_label_indices = np.where((y_train == label_A) & (np.arange(len(y_train)) != idx_A))[0]
            
            idx_C = None
            if len(same_label_indices) > 0:
                idx_C = same_label_indices[0]
                label_C = y_train[idx_C]
                text_C = label_map[label_C]

            print(f"Comparing the following samples:")
            print(f"  Sample A (idx {idx_A}): '{text_A}'")
            print(f"  Sample B (idx {idx_B}): '{text_B}'")
            if idx_C is not None:
                print(f"  Sample C (idx {idx_C}): '{text_C}' (Augmentation of A)")
            print("\nExtracting traces for comparison...")

            # --- 2. Extract traces ---
            trace_A = extract_membrane_traces(lsm, X_train[idx_A], lsm.output_neurons)
            trace_B = extract_membrane_traces(lsm, X_train[idx_B], lsm.output_neurons)
            if idx_C is not None:
                trace_C = extract_membrane_traces(lsm, X_train[idx_C], lsm.output_neurons)

            # --- 3. Flatten traces for similarity ---
            # To compare two time-series, we flatten them into one long vector
            vec_A = trace_A.flatten().reshape(1, -1)
            vec_B = trace_B.flatten().reshape(1, -1)
            if idx_C is not None:
                vec_C = trace_C.flatten().reshape(1, -1)

            # --- 4. Calculate and report cosine similarity ---
            sim_AB = cosine_similarity(vec_A, vec_B)[0][0]
            
            print("\n--- Cosine Similarity Results ---")
            print(f"  🎯 A vs B (DIFFERENT sentences): {sim_AB:.4f}")
            
            if idx_C is not None:
                sim_AC = cosine_similarity(vec_A, vec_C)[0][0]
                print(f"  🎯 A vs C (SAME sentence): {sim_AC:.4f}")

            print("\n--- Interpretation ---")
            if sim_AB > 0.6:
                print(f"  ⚠️  WARNING: Similarity between DIFFERENT sentences is HIGH ({sim_AB:.4f}).")
                print("      This suggests the LSM is not separating inputs well.")
                print("      Consider lowering the weight multiplier (e.g., --multiplier 0.6) or increasing it.")
            else:
                print(f"  ✅ GOOD: Similarity between DIFFERENT sentences is LOW ({sim_AB:.4f}).")
                print("      This suggests the LSM is separating inputs.")
            
            if idx_C is not None and sim_AC < 0.5:
                print(f"  ⚠️  WARNING: Similarity between SAME sentence is LOW ({sim_AC:.4f}).")
                print("      This suggests the LSM is too chaotic or sensitive.")
            elif idx_C is not None:
                print(f"  ✅ GOOD: Similarity between SAME sentence is HIGH ({sim_AC:.4f}).")

        except Exception as e:
            print(f"\n--- Error during similarity test ---")
            print(f"  {e}")
            print("  This might happen if the dataset is too small to find")
            print("  both different sentences and augmentations in the first few samples.")
            print("  Continuing with main trace extraction...")
            
    print("="*60 + "\n")
    # ####################################################################
    # ### END: LSM TRACE SEPARABILITY DEBUGGING BLOCK ###
    # ####################################################################


    # Extract membrane potential traces
    print("\nExtracting membrane potential traces (full dataset)...")
    X_train_traces = extract_dataset_traces(lsm, X_train, "Training")
    X_test_traces = extract_dataset_traces(lsm, X_test, "Testing")

    print(f"\nExtracted traces:")
    print(f"  Train shape: {X_train_traces.shape}")  # (samples, timesteps, neurons)
    print(f"  Test shape: {X_test_traces.shape}")

    # Calculate trace statistics
    print(f"\n--- Trace Statistics ---")
    print(f"  Trace value range: [{X_train_traces.min():.3f}, {X_train_traces.max():.3f}]")
    print(f"  Mean trace value: {X_train_traces.mean():.3f}")
    print(f"  Std trace value: {X_train_traces.std():.3f}")
    print("-------------------------")

    # Save with sentence-level split marker
    output_file = "lsm_trace_sequences.npz"
    print(f"\nSaving to '{output_file}'...")
    np.savez_compressed(
        output_file,
        X_train_sequences=X_train_traces,
        y_train=y_train,
        X_test_sequences=X_test_traces,
        y_test=y_test,
        final_weight=optimal_weight,
        split_type='sentence_level_500_traces'
    )

    print("\n" + "="*60)
    print("✅ SENTENCE-LEVEL TRACE EXTRACTION COMPLETE! (500 SENTENCES)")
    print(f"Saved to: {output_file}")
    print("\n🎯 IMPORTANT: These are MEMBRANE POTENTIAL TRACES!")
    print("   Continuous voltage signals (not spike features)")
    print("   Test set contains COMPLETELY UNSEEN sentences!")
    print("   Dataset: 400 train sentences, 100 test sentences")
    print("\nNext step:")
    print("  python train_ctc.py  # Train on trace data (modify to load trace file)")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract membrane potential traces with SENTENCE-LEVEL split for true generalization testing."
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="Multiplier for w_critico (try 0.7-0.9)"
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("SENTENCE-LEVEL TRACE EXTRACTION")
    print("Testing TRUE generalization to unseen sentences!")
    print("="*60)

    main(multiplier=args.multiplier)