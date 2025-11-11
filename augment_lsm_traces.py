"""
Augment LSM traces to increase training data

Augmentation strategies:
1. Add Gaussian noise to membrane potentials
2. Time stretching (interpolation)
3. Amplitude scaling
4. Temporal jittering
"""
import numpy as np
from scipy import interpolate
from pathlib import Path
import argparse

def add_gaussian_noise(traces, noise_level=0.05):
    """Add Gaussian noise to traces"""
    noise = np.random.randn(*traces.shape) * noise_level * traces.std()
    return traces + noise

def time_stretch(traces, factor=1.1):
    """
    Time stretch by interpolation
    factor > 1.0: slower (stretch)
    factor < 1.0: faster (compress)
    """
    num_timesteps = traces.shape[1]
    num_neurons = traces.shape[2]

    # Original time indices
    original_indices = np.arange(num_timesteps)
    # Stretched time indices
    stretched_length = int(num_timesteps * factor)
    stretched_indices = np.linspace(0, num_timesteps - 1, stretched_length)

    stretched_traces = np.zeros((traces.shape[0], num_timesteps, num_neurons), dtype=np.float32)

    for i in range(traces.shape[0]):
        for j in range(num_neurons):
            # Interpolate each neuron's trace
            f = interpolate.interp1d(original_indices, traces[i, :, j], kind='linear', fill_value='extrapolate')
            stretched = f(stretched_indices)

            # Crop or pad to original length
            if len(stretched) > num_timesteps:
                stretched_traces[i, :, j] = stretched[:num_timesteps]
            else:
                stretched_traces[i, :len(stretched), j] = stretched
                # Pad with last value
                stretched_traces[i, len(stretched):, j] = stretched[-1]

    return stretched_traces

def amplitude_scale(traces, scale_factor=1.1):
    """Scale amplitude of traces"""
    return traces * scale_factor

def temporal_jitter(traces, jitter_std=5):
    """
    Apply random temporal jittering by shifting each neuron's trace slightly
    """
    jittered = traces.copy()
    num_timesteps = traces.shape[1]

    for i in range(traces.shape[0]):
        for j in range(traces.shape[2]):
            # Random shift in range [-jitter_std, +jitter_std]
            shift = int(np.random.randn() * jitter_std)
            if shift > 0:
                jittered[i, shift:, j] = traces[i, :-shift, j]
                jittered[i, :shift, j] = traces[i, 0, j]  # Pad with first value
            elif shift < 0:
                jittered[i, :shift, j] = traces[i, -shift:, j]
                jittered[i, shift:, j] = traces[i, -1, j]  # Pad with last value

    return jittered

def augment_dataset(X, y, num_augmentations=4):
    """
    Create augmented versions of the dataset

    Args:
        X: Original traces (samples, timesteps, neurons)
        y: Original labels
        num_augmentations: Number of augmented copies per sample

    Returns:
        X_aug: Augmented traces
        y_aug: Corresponding labels
    """
    X_list = [X]  # Start with original
    y_list = [y]

    print(f"Creating {num_augmentations} augmented versions...")

    for aug_idx in range(num_augmentations):
        print(f"  Augmentation {aug_idx + 1}/{num_augmentations}...")

        # Apply random combination of augmentations
        X_aug = X.copy()

        # 1. Add noise (always)
        noise_level = np.random.uniform(0.03, 0.08)
        X_aug = add_gaussian_noise(X_aug, noise_level)

        # 2. Time stretch (50% chance)
        if np.random.rand() > 0.5:
            stretch_factor = np.random.uniform(0.95, 1.05)
            X_aug = time_stretch(X_aug, stretch_factor)

        # 3. Amplitude scale (50% chance)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.95, 1.05)
            X_aug = amplitude_scale(X_aug, scale)

        # 4. Temporal jitter (30% chance)
        if np.random.rand() > 0.7:
            jitter = int(np.random.uniform(3, 8))
            X_aug = temporal_jitter(X_aug, jitter)

        # Clip to valid range [0, max]
        X_aug = np.clip(X_aug, 0, X.max() * 1.2)

        X_list.append(X_aug)
        y_list.append(y)

    # Concatenate all
    X_augmented = np.concatenate(X_list, axis=0)
    y_augmented = np.concatenate(y_list, axis=0)

    return X_augmented, y_augmented

def main(num_augmentations):
    print("="*60)
    print("AUGMENTING LSM TRACES")
    print("="*60)

    # Load original data
    trace_file = "lsm_trace_sequences_sentence_split_500.npz"
    if not Path(trace_file).exists():
        print(f"❌ Error: Trace file not found")
        return

    dataset = np.load(trace_file, allow_pickle=True)
    X_train = dataset["X_train_sequences"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test_sequences"]
    y_test = dataset["y_test"]
    final_weight = dataset["final_weight"]

    print(f"\n✅ Loaded original data")
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Augment training data ONLY
    print(f"\nAugmenting training data...")
    X_train_aug, y_train_aug = augment_dataset(X_train, y_train, num_augmentations)

    print(f"\n✅ Augmentation complete!")
    print(f"   Original train: {X_train.shape[0]} samples")
    print(f"   Augmented train: {X_train_aug.shape[0]} samples ({num_augmentations + 1}x)")

    # Save augmented dataset
    output_file = "lsm_trace_sequences_sentence_split_500_augmented.npz"
    print(f"\nSaving to '{output_file}'...")
    np.savez_compressed(
        output_file,
        X_train_sequences=X_train_aug,
        y_train=y_train_aug,
        X_test_sequences=X_test,  # Keep test unchanged
        y_test=y_test,
        final_weight=final_weight,
        split_type='sentence_level_500_traces_augmented',
        num_augmentations=num_augmentations
    )

    print(f"\n✅ Saved augmented dataset to '{output_file}'")
    print(f"\nYou can now train with more data:")
    print(f"  - Modify train_ctc_traces.py to load '{output_file}'")
    print(f"  - Or create a new training script")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment LSM traces")
    parser.add_argument(
        "--num-augmentations",
        type=int,
        default=4,
        help="Number of augmented copies per sample (default: 4)"
    )
    args = parser.parse_args()

    print(f"\nCreating {args.num_augmentations} augmented versions per sample")
    print(f"Total training data will be {args.num_augmentations + 1}x original\n")

    main(args.num_augmentations)
