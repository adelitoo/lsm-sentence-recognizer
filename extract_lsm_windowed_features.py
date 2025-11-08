"""
LSM Windowed Feature Extraction for CTC

This script extracts features from the LSM in sliding time windows,
combining the discriminative power of aggregated features (used for single words)
with the temporal structure needed for CTC (used for sentences).

Key differences from extract_lsm_sequences.py:
- Uses extract_features_from_spikes() instead of traces
- Extracts features in time windows
- Produces (samples, time_windows, features) for CTC
"""

import numpy as np
from snnpy.snn import SNN, SimulationParams
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
SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2)  # k=200

# --- Windowing Parameters ---
WINDOW_SIZE = 40  # Number of time steps per window
STRIDE = 20  # Overlap between windows (stride < window_size = overlap)

# --- Feature Configuration ---
# Use the same features that worked for single words
FEATURE_KEYS = ['spike_counts', 'spike_variances', 'mean_spike_times',
                'mean_isi', 'isi_variances']

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


def extract_windowed_features(lsm, spike_sample, window_size, stride, feature_keys):
    """
    Extract features from LSM in sliding time windows.

    Args:
        lsm: The SNN/LSM object
        spike_sample: Spike train of shape (input_neurons, time_bins)
        window_size: Number of time steps per window
        stride: Number of steps between windows
        feature_keys: List of feature names to extract

    Returns:
        Array of shape (num_windows, feature_dim)
    """
    num_input_neurons, total_time_bins = spike_sample.shape

    # Calculate number of windows
    num_windows = (total_time_bins - window_size) // stride + 1

    window_features = []

    for window_idx in range(num_windows):
        start_time = window_idx * stride
        end_time = start_time + window_size

        # Extract the spike window
        spike_window = spike_sample[:, start_time:end_time]

        # Reset LSM and simulate on this window
        lsm.reset()
        lsm.set_input_spike_times(spike_window)
        lsm.simulate()

        # Extract features
        feature_dict = lsm.extract_features_from_spikes()

        # Combine selected features
        parts = []
        for key in feature_keys:
            if key in feature_dict:
                vec = np.nan_to_num(
                    feature_dict[key].copy(),
                    nan=0.0, posinf=0.0, neginf=0.0
                )
                vec[vec < 0] = 0
                parts.append(vec)

        combined_features = np.concatenate(parts)
        window_features.append(combined_features)

    return np.array(window_features)


def extract_dataset_windowed_features(lsm, spike_data, window_size, stride, feature_keys, desc=""):
    """Extract windowed features for entire dataset"""
    all_windowed_features = []

    for sample in tqdm(spike_data, desc=desc):
        windowed_features = extract_windowed_features(
            lsm, sample, window_size, stride, feature_keys
        )
        all_windowed_features.append(windowed_features)

    return np.array(all_windowed_features, dtype=np.float32)


def main(multiplier: float, window_size: int, stride: int):

    # Load spike dataset
    X_spikes, y_labels = load_spike_dataset(filename="sentence_spike_trains.npz")
    if X_spikes is None:
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
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

    # Calculate number of windows
    _, total_time_bins = X_train[0].shape
    num_windows = (total_time_bins - window_size) // stride + 1

    print(f"\n--- Windowing Configuration ---")
    print(f"  Window size: {window_size} time steps")
    print(f"  Stride: {stride} time steps")
    print(f"  Total time bins: {total_time_bins}")
    print(f"  Number of windows: {num_windows}")
    print(f"  Feature keys: {FEATURE_KEYS}")
    print("-------------------------------")

    # Extract windowed features
    print("\nExtracting windowed features...")
    X_train_windowed = extract_dataset_windowed_features(
        lsm, X_train, window_size, stride, FEATURE_KEYS, "Training"
    )
    X_test_windowed = extract_dataset_windowed_features(
        lsm, X_test, window_size, stride, FEATURE_KEYS, "Testing"
    )

    print(f"\nExtracted features:")
    print(f"  Train shape: {X_train_windowed.shape}")
    print(f"  Test shape: {X_test_windowed.shape}")
    print(f"  Format: (samples, time_windows, feature_dim)")

    # Calculate feature statistics
    feature_dim = X_train_windowed.shape[2]
    print(f"\n--- Feature Statistics ---")
    print(f"  Feature dimension per window: {feature_dim}")
    print(f"  ({NUM_OUTPUT_NEURONS} neurons × {len(FEATURE_KEYS)} feature types)")
    print(f"  Feature value range: [{X_train_windowed.min():.3f}, {X_train_windowed.max():.3f}]")
    print(f"  Mean feature value: {X_train_windowed.mean():.3f}")
    print("-------------------------")

    # Save
    output_file = "lsm_windowed_features.npz"
    print(f"\nSaving to '{output_file}'...")
    np.savez_compressed(
        output_file,
        X_train_sequences=X_train_windowed,
        y_train=y_train,
        X_test_sequences=X_test_windowed,
        y_test=y_test,
        final_weight=optimal_weight,
        window_size=window_size,
        stride=stride,
        feature_keys=FEATURE_KEYS
    )

    print("\n" + "="*60)
    print("✅ WINDOWED FEATURE EXTRACTION COMPLETE!")
    print(f"Saved to: {output_file}")
    print("\nNext step: Run train_ctc.py with this feature file")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract windowed features from LSM for CTC training."
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=0.6,
        help="Multiplier for w_critico (e.g., 0.6 for 60%%)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"Size of each time window (default: {WINDOW_SIZE})"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help=f"Stride between windows (default: {STRIDE})"
    )
    args = parser.parse_args()

    main(multiplier=args.multiplier, window_size=args.window_size, stride=args.stride)
