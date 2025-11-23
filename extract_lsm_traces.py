import numpy as np
from snnpy.snn import Reservoir, SimulationParams
from tqdm import tqdm
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# --- NEW IMPORTS ---
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# -------------------

# --- Network Parameters (matching the spike feature version) ---
NUM_NEURONS = 2000
NUM_OUTPUT_NEURONS = 700
# LEAK_COEFFICIENT = 0.02  # <-- MODIFIED: This is now set by --leak arg
REFRACTORY_PERIOD = 2
MEMBRANE_THRESHOLD = 2
SMALL_WORLD_P = 0.3
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


# --- 📊 NEW PLOTTING FUNCTION ---
def generate_raster_plot(lsm, spike_sample, multiplier, plot_filename="lsm_raster_plot.png"):
    """
    Runs a single simulation to generate and save a raster plot
    of input, reservoir (non-output), and output neurons.
    """
    print(f"\n--- 📊 Generating Raster Plot for first sample ---")
    
    # --- 1. Select Neuron Indices ---
    Nin = lsm.num_input_neurons
    N = lsm.num_neurons
    T = spike_sample.shape[1]

    # Select up to 50 distributed input neurons
    num_inputs_to_plot = min(50, Nin)
    input_indices = np.linspace(0, Nin - 1, num_inputs_to_plot, dtype=int)

    # Select up to 50 distributed non-output reservoir neurons
    output_set = set(lsm.output_neurons)
    reservoir_indices_all = np.array([i for i in np.arange(Nin, N) if i not in output_set])
    num_reservoir_to_plot = min(50, len(reservoir_indices_all))
    
    if num_reservoir_to_plot > 0:
        reservoir_indices = reservoir_indices_all[np.linspace(0, len(reservoir_indices_all) - 1, num_reservoir_to_plot, dtype=int)]
    else:
        reservoir_indices = np.array([], dtype=int) # Handle case with no non-output neurons

    # Select up to 50 distributed output neurons
    num_outputs_to_plot = min(50, len(lsm.output_neurons))
    if num_outputs_to_plot > 0:
        output_indices = lsm.output_neurons[np.linspace(0, len(lsm.output_neurons) - 1, num_outputs_to_plot, dtype=int)]
    else:
        output_indices = np.array([], dtype=int)
    
    print(f"  Plotting {len(input_indices)} inputs, {len(reservoir_indices)} reservoir, {len(output_indices)} outputs.")

    # --- 2. Run Simulation to get all spikes ---
    # (This is a copy of the loop from extract_membrane_traces, but stores all spikes)
    lsm.reset()
    lsm.set_input_spike_times(spike_sample)

    inputs = spike_sample
    mem = lsm.membrane_potentials
    refr = lsm.refractory_timer
    mem[:] = 0.0
    refr[:] = 0

    # Use heterogeneous or uniform leak factor
    if lsm.heterogeneous_leak:
        leak_factor = lsm.leak_factors  # Per-neuron leak factors (N,)
    else:
        leak_factor = np.float32(1.0 - lsm.leak_coefficient)
    curr_amp = np.float32(lsm.current_amplitude)
    W = lsm.synaptic_weights.tocsr()
    indptr, indices, data = W.indptr, W.indices, W.data

    # We need the full spike matrix to select from
    full_spike_matrix = np.zeros((T, N), dtype=np.uint8)

    for t in range(T):
        refr -= lsm.time_step
        np.clip(refr, 0, None, out=refr)

        if t < inputs.shape[1]:
            spikes_t = inputs[:, t]
            mem[:Nin] += curr_amp * spikes_t
            full_spike_matrix[t, :Nin] = spikes_t # Record external input spikes

        spiking_mask = (mem >= lsm.membrane_threshold) & (refr == 0)
        # Record only reservoir/output spikes (not input neurons) to avoid overwriting
        reservoir_spiking_mask = spiking_mask.copy()
        reservoir_spiking_mask[:Nin] = False  # Exclude input neurons
        full_spike_matrix[t, reservoir_spiking_mask] = 1 # Record reservoir/output spikes

        spk_idx = np.flatnonzero(spiking_mask) # Get indices *before* reset

        if spiking_mask.any():
            mem[spiking_mask] = 0.0
            refr[spiking_mask] = lsm.refractory_period + 1

        mem *= leak_factor

        if spiking_mask.any():
            for j in spk_idx:
                start, end = indptr[j], indptr[j + 1]
                if start != end:
                    cols = indices[start:end]
                    mem[cols] += data[start:end]

    # --- 3. Extract spike data for plotting ---
    input_spikes = full_spike_matrix[:, input_indices]
    reservoir_spikes = full_spike_matrix[:, reservoir_indices]
    output_spikes = full_spike_matrix[:, output_indices]

    # --- 4. Generate Plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # Helper for eventplot
    def plot_raster(ax, spikes, title, color):
        if spikes.shape[1] == 0: # Handle case with 0 neurons
             ax.set_title(f"{title} (0 neurons selected)")
             ax.set_yticks([])
             ax.set_ylabel("Neuron Index")
             return
        
        spike_times_list = []
        for i in range(spikes.shape[1]):
            spike_times = np.where(spikes[:, i] == 1)[0]
            spike_times_list.append(spike_times)
        
        ax.eventplot(spike_times_list, colors=color, linelengths=0.8)
        ax.set_ylabel("Neuron Index")
        ax.set_title(title)
        ax.set_yticks([0, max(0, spikes.shape[1] - 1)]) # Handle case with 1 neuron
        ax.set_ylim(-1, spikes.shape[1])

    plot_raster(ax1, input_spikes, f"Input Neurons ({num_inputs_to_plot} / {Nin})", "blue")
    plot_raster(ax2, reservoir_spikes, f"Reservoir Neurons ({num_reservoir_to_plot} / {len(reservoir_indices_all)})", "green")
    plot_raster(ax3, output_spikes, f"Output Neurons ({num_outputs_to_plot} / {len(lsm.output_neurons)})", "red")

    ax3.set_xlabel("Time Step")
    fig.suptitle(f"LSM Raster Plot (w_mult: {multiplier:.2f}, leak: {lsm.leak_coefficient:.3f})", fontsize=16, y=1.02) # <-- MODIFIED
    fig.tight_layout()

    try:
        plt.savefig(plot_filename, bbox_inches="tight")
        print(f"  ✅ Raster plot saved to '{plot_filename}'")
    except Exception as e:
        print(f"  ❌ Error saving plot: {e}")
    
    plt.close(fig) # Close the figure to save memory
# --- ---------------------- ---


def extract_membrane_traces(lsm, spike_sample, output_neurons):
    """
    Extract membrane potential traces from LSM output neurons over time.
    
    *** MODIFIED: Now also returns debugging info (total_reservoir_spikes, num_active_reservoir_neurons) ***
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

    # Zero out membrane potentials and refractory timers
    mem[:] = 0.0
    refr[:] = 0

    out_idx = output_neurons
    # Use heterogeneous or uniform leak factor
    if lsm.heterogeneous_leak:
        leak_factor = lsm.leak_factors  # Per-neuron leak factors (N,)
    else:
        leak_factor = np.float32(1.0 - lsm.leak_coefficient)
    curr_amp = np.float32(lsm.current_amplitude)

    W = lsm.synaptic_weights.tocsr()
    indptr, indices, data = W.indptr, W.indices, W.data
    
    # --- 🧠 DEBUGGING VARS ---
    total_reservoir_spikes = 0
    active_reservoir_neurons = set()
    # --- END DEBUGGING VARS ---

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
        
        spk_idx = np.flatnonzero(spiking_mask) # Get indices *before* reset

        if spiking_mask.any():
            # --- 🧠 DEBUGGING LOGIC ---
            # Find which of these are in the reservoir (not input)
            reservoir_spikes_mask = spk_idx >= Nin
            if reservoir_spikes_mask.any():
                num_spikes_this_step = np.sum(reservoir_spikes_mask)
                total_reservoir_spikes += num_spikes_this_step
                
                # Add the *neuron indices* to the set
                active_reservoir_neurons.update(spk_idx[reservoir_spikes_mask])
            # --- END DEBUGGING LOGIC ---

            # Reset spiking neurons
            mem[spiking_mask] = 0.0
            refr[spiking_mask] = lsm.refractory_period + 1

        # Apply leak
        mem *= leak_factor

        # Propagate spikes through network
        if spiking_mask.any():
            # NOTE: We use spk_idx captured before leak/reset
            for j in spk_idx: 
                start, end = indptr[j], indptr[j + 1]
                if start != end:
                    cols = indices[start:end]
                    mem[cols] += data[start:end]

        # Record membrane potentials of output neurons
        membrane_trace[t, :] = mem[out_idx]

    # --- MODIFIED: Return new metrics ---
    return membrane_trace, total_reservoir_spikes, len(active_reservoir_neurons)


def extract_dataset_traces(lsm, spike_data, desc=""):
    """
    Extract membrane traces for entire dataset
    
    *** MODIFIED: Now also returns debugging info (spike counts, active counts) ***
    """
    all_traces = []
    
    # --- 🧠 DEBUGGING ---
    all_reservoir_spikes = []
    all_active_counts = []
    # --- END DEBUGGING ---

    for sample in tqdm(spike_data, desc=desc):
        # --- MODIFIED: Get new metrics ---
        traces, total_spikes, active_count = extract_membrane_traces(lsm, sample, lsm.output_neurons)
        
        all_traces.append(traces)

        # --- DEBUGGING ---
        all_reservoir_spikes.append(total_spikes)
        all_active_counts.append(active_count)
        # --- END DEBUGGING ---

    # --- MODIFIED: Return new metrics ---
    return np.array(all_traces, dtype=np.float32), np.array(all_reservoir_spikes), np.array(all_active_counts)


def main(multiplier: float, leak: float, leak_variance_divisor: float = None):

    # 1. Load spike dataset
    X_spikes, y_labels = load_spike_dataset(filename="sentence_spike_trains.npz")
    if X_spikes is None:
        return

    # 2. Split data BY SENTENCE
    X_train, X_test, y_train, y_test = split_by_sentence(
        X_spikes, y_labels, test_size=0.2, random_state=42
    )

    # 3. Calculate w_critico (Using Temporary Params)
    # We need a dummy object just to calculate the critical weight scaling
    temp_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=0.0,
        weight_variance=5.0,
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=leak,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_train[0],
        mean_distance=0.0, # Dummy value required for class validation
        leak_variance_divisor=leak_variance_divisor
    )

    w_critico_calculated = calculate_theoretical_w_critico(temp_params, X_train)
    
    # 4. Calculate Optimal Weight and Distance
    optimal_weight = w_critico_calculated * multiplier
    
    # Use the Professor's Ratio: Distance = 15 * Weight
    # This ensures the excitatory/inhibitory clusters separate as weights get stronger
    distance_coeff = 15.0 
    optimal_distance = distance_coeff * optimal_weight

    print(f"\nUsing weight multiplier: {multiplier:.2f}")
    print(f"  FINAL WEIGHT USED: {optimal_weight:.8f}")
    print(f"  MEAN DISTANCE: {optimal_distance:.8f}")
    print(f"  LEAK COEFFICIENT: {leak:.4f}")
    
    if leak_variance_divisor is not None:
        print(f"  HETEROGENEOUS LEAK: Enabled (Divisor {leak_variance_divisor})")
    else:
        print(f"  HETEROGENEOUS LEAK: Disabled (Uniform)")

    # 5. Create the Real Reservoir
    print(f"\nCreating Reservoir ({NUM_NEURONS} neurons, {NUM_OUTPUT_NEURONS} outputs)...")
    
    base_params = SimulationParams(
        num_neurons=NUM_NEURONS,
        mean_weight=optimal_weight,
        
        # FIXED VARIANCE: Use 5.0 (High Variance/Hubs) instead of scaling by weight
        weight_variance=5.0, 
        
        num_output_neurons=NUM_OUTPUT_NEURONS,
        is_random_uniform=False,
        membrane_threshold=MEMBRANE_THRESHOLD,
        leak_coefficient=leak,
        refractory_period=REFRACTORY_PERIOD,
        small_world_graph_p=SMALL_WORLD_P,
        small_world_graph_k=SMALL_WORLD_K,
        input_spike_times=X_train[0],
        
        # NEW MANDATORY PARAMS
        mean_distance=optimal_distance, 
        leak_variance_divisor=leak_variance_divisor
    )
    
    # Use Reservoir class (not SNN)
    lsm = Reservoir(simulation_params=base_params)
    
    # --- 🧠 DEBUGGING: Get reservoir size ---
    num_reservoir_neurons = lsm.num_neurons - lsm.num_input_neurons
    print(f"  Input Neurons: {lsm.num_input_neurons}")
    print(f"  Reservoir Neurons: {num_reservoir_neurons}")
    
    # --- 📊 Generate Raster Plot for first training sample ---
    generate_raster_plot(lsm, X_train[0], multiplier, "lsm_raster_plot.png")

    # 6. Extract membrane potential traces
    print("\nExtracting membrane potential traces (full dataset)...")
    
    X_train_traces, train_spikes, train_active = extract_dataset_traces(lsm, X_train, "Training")
    X_test_traces, test_spikes, test_active = extract_dataset_traces(lsm, X_test, "Testing")

    print(f"\nExtracted traces:")
    print(f"  Train shape: {X_train_traces.shape}") 
    print(f"  Test shape: {X_test_traces.shape}")

    # --- 🧠 LSM Activity Debugging ---
    print(f"\n--- 🧠 LSM Activity Debugging ---")
    
    # Train stats
    avg_train_active = np.mean(train_active)
    avg_train_active_pct = (avg_train_active / num_reservoir_neurons) * 100
    
    print(f"  [TRAIN] Avg. Active Reservoir Neurons: {avg_train_active:.1f} ({avg_train_active_pct:.2f}%)")

    # Test stats
    avg_test_active = np.mean(test_active)
    avg_test_active_pct = (avg_test_active / num_reservoir_neurons) * 100

    print(f"  [TEST] Avg. Active Reservoir Neurons: {avg_test_active:.1f} ({avg_test_active_pct:.2f}%)")
    
    print("\n  ℹ️  Interpretation:")
    if avg_train_active_pct < 1.0:
        print(f"  ❌ WARNING: Activity is < 1%. Network is 'dead'. Increase Multiplier.")
    elif avg_train_active_pct > 50.0:
        print(f"  ⚠️  WARNING: Activity is > 50%. Network is 'saturated'. Decrease Multiplier.")
    else:
        print(f"  ✅ Network activity is in a plausible range.")

    # 7. Save Data
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
    print("✅ SENTENCE-LEVEL TRACE EXTRACTION COMPLETE!")
    print(f"Saved to: {output_file}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract membrane potential traces with SENTENCE-LEVEL split for true generalization testing."
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=0.4,
        help="Multiplier for w_critico (try 0.7-0.9)"
    )
    # --- MODIFIED: Added leak argument ---
    parser.add_argument(
        "--leak",
        type=float,
        default=0.001,
        help="Leak coefficient (e.g., 0.01 to 0.1). Higher = faster leak / shorter memory."
    )
    parser.add_argument(
        "--leak-variance-divisor",
        type=float,
        default=20,
        help="Divisor for leak variance (e.g., 20 means std = leak/20). If not set, all neurons use same leak."
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("SENTENCE-LEVEL TRACE EXTRACTION")
    print("Testing TRUE generalization to unseen sentences!")
    print("="*60)

    # --- MODIFIED: Pass leak to main ---
    main(multiplier=args.multiplier, leak=args.leak, leak_variance_divisor=args.leak_variance_divisor)