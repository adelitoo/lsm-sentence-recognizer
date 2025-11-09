"""
LSM Windowed Feature Extraction with Feature Selection

This version filters out zero and low-variance features to reduce
dimensionality and improve CTC learning with small datasets.
"""

import numpy as np
from snnpy.snn import SNN, SimulationParams
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
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
SMALL_WORLD_K = int(0.10 * NUM_NEURONS * 2)

# --- Windowing Parameters ---
WINDOW_SIZE = 40
STRIDE = 20

# --- Feature Configuration ---
FEATURE_KEYS = ['spike_counts', 'spike_variances', 'mean_spike_times',
                'mean_isi', 'isi_variances']

# --- Feature Selection ---
VARIANCE_THRESHOLD = 0.01  # Filter features with variance < this

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
    """Extract features from LSM in sliding time windows"""
    num_input_neurons, total_time_bins = spike_sample.shape
    num_windows = (total_time_bins - window_size) // stride + 1
    window_features = []

    for window_idx in range(num_windows):
        start_time = window_idx * stride
        end_time = start_time + window_size
        spike_window = spike_sample[:, start_time:end_time]

        lsm.reset()
        lsm.set_input_spike_times(spike_window)
        lsm.simulate()

        feature_dict = lsm.extract_features_from_spikes()
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


def filter_low_variance_features(X_train, X_test, threshold=0.01):
    """
    Remove features with low variance across time and samples

    Args:
        X_train: Shape (samples, time_windows, features)
        X_test: Shape (samples, time_windows, features)
        threshold: Variance threshold

    Returns:
        Filtered X_train, X_test, and feature selector
    """
    print(f"\n--- Feature Filtering (variance threshold: {threshold}) ---")

    # Reshape to (samples * time, features) for variance calculation
    n_samples, n_windows, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)

    # Calculate feature statistics before filtering
    feature_vars = np.var(X_train_flat, axis=0)
    zero_features = np.sum(feature_vars == 0)
    low_var_features = np.sum(feature_vars < threshold)

    print(f"Original features: {n_features}")
    print(f"  Zero variance features: {zero_features}")
    print(f"  Low variance features (< {threshold}): {low_var_features}")

    # Filter using VarianceThreshold
    selector = VarianceThreshold(threshold=threshold)
    X_train_filtered_flat = selector.fit_transform(X_train_flat)
    X_test_filtered_flat = selector.transform(X_test_flat)

    # Reshape back to (samples, time_windows, features)
    n_features_kept = X_train_filtered_flat.shape[1]
    X_train_filtered = X_train_filtered_flat.reshape(n_samples, n_windows, n_features_kept)

    n_test_samples = X_test.shape[0]
    X_test_filtered = X_test_filtered_flat.reshape(n_test_samples, n_windows, n_features_kept)

    print(f"Features after filtering: {n_features_kept}")
    print(f"Reduction: {n_features - n_features_kept} features removed ({(n_features - n_features_kept) / n_features * 100:.1f}%)")
    print("-----------------------------------------------------------")

    return X_train_filtered, X_test_filtered, selector


def main(multiplier: float, window_size: int, stride: int, variance_threshold: float):

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

    print(f"\nExtracted features (before filtering):")
    print(f"  Train shape: {X_train_windowed.shape}")
    print(f"  Test shape: {X_test_windowed.shape}")

    # Filter low-variance features
    X_train_filtered, X_test_filtered, selector = filter_low_variance_features(
        X_train_windowed, X_test_windowed, threshold=variance_threshold
    )

    print(f"\nExtracted features (after filtering):")
    print(f"  Train shape: {X_train_filtered.shape}")
    print(f"  Test shape: {X_test_filtered.shape}")

    # Calculate feature statistics
    feature_dim = X_train_filtered.shape[2]
    print(f"\n--- Feature Statistics ---")
    print(f"  Feature dimension per window: {feature_dim}")
    print(f"  Feature value range: [{X_train_filtered.min():.3f}, {X_train_filtered.max():.3f}]")
    print(f"  Mean feature value: {X_train_filtered.mean():.3f}")
    print("-------------------------")

    # Save
    output_file = "lsm_windowed_features_filtered.npz"
    print(f"\nSaving to '{output_file}'...")
    np.savez_compressed(
        output_file,
        X_train_sequences=X_train_filtered,
        y_train=y_train,
        X_test_sequences=X_test_filtered,
        y_test=y_test,
        final_weight=optimal_weight,
        window_size=window_size,
        stride=stride,
        feature_keys=FEATURE_KEYS,
        variance_threshold=variance_threshold,
        feature_mask=selector.get_support()
    )

    print("\n" + "="*60)
    print("✅ FILTERED WINDOWED FEATURE EXTRACTION COMPLETE!")
    print(f"Saved to: {output_file}")
    print("\nNext step: Run train_ctc.py (it will auto-detect the filtered features)")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract filtered windowed features from LSM for CTC training."
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=0.8,
        help="Multiplier for w_critico (try 0.7-0.9 for more activity)"
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
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=VARIANCE_THRESHOLD,
        help=f"Variance threshold for feature filtering (default: {VARIANCE_THRESHOLD})"
    )
    args = parser.parse_args()

    main(
        multiplier=args.multiplier,
        window_size=args.window_size,
        stride=args.stride,
        variance_threshold=args.variance_threshold
    )
