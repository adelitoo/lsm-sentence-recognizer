"""
Diagnostic script to analyze LSM feature quality for CTC

This script helps diagnose why CTC might not be learning by:
1. Checking feature separability between different sentences
2. Visualizing temporal patterns
3. Computing feature statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pathlib import Path

def load_features():
    """Load windowed features or traces"""
    feature_file = "lsm_windowed_features.npz"
    trace_file = "lsm_trace_sequences.npz"

    if Path(feature_file).exists():
        print(f"✅ Loading windowed features from '{feature_file}'")
        data = np.load(feature_file)
        feature_type = "windowed"
    elif Path(trace_file).exists():
        print(f"⚠️  Loading traces from '{trace_file}'")
        data = np.load(trace_file)
        feature_type = "traces"
    else:
        print("❌ No feature file found!")
        return None, None, None

    X_train = data['X_train_sequences']
    y_train = data['y_train']

    return X_train, y_train, feature_type


def load_label_map():
    """Load label map"""
    label_map = {}
    with open("sentence_label_map.txt", "r") as f:
        next(f)  # Skip header
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text
    return label_map


def analyze_temporal_structure(X_train, y_train, label_map):
    """Analyze temporal variation in features"""
    print("\n=== Temporal Structure Analysis ===")

    # Pick first 3 samples from different classes
    unique_labels = np.unique(y_train)[:3]

    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    for idx, label in enumerate(unique_labels):
        # Get first sample of this class
        sample_idx = np.where(y_train == label)[0][0]
        sample_features = X_train[sample_idx]  # Shape: (time, features)

        # Plot temporal variation
        axes[idx].imshow(sample_features.T, aspect='auto', cmap='viridis')
        axes[idx].set_title(f"Label {label}: '{label_map[label]}'")
        axes[idx].set_xlabel("Time Steps")
        axes[idx].set_ylabel("Feature Dimensions")

    plt.tight_layout()
    plt.savefig("temporal_structure_analysis.png")
    print("✅ Saved temporal structure plot to 'temporal_structure_analysis.png'")

    # Compute temporal variation statistics
    temporal_stds = np.std(X_train, axis=1)  # Std across time for each sample
    mean_temporal_std = np.mean(temporal_stds)
    print(f"\nTemporal variation (mean std across time): {mean_temporal_std:.4f}")

    if mean_temporal_std < 0.01:
        print("⚠️  WARNING: Very low temporal variation!")
        print("   Features might not be changing enough over time for CTC")


def analyze_separability(X_train, y_train, label_map):
    """Analyze feature separability using PCA"""
    print("\n=== Feature Separability Analysis ===")

    # Flatten temporal dimension for overall separability
    # Average over time to get per-sample representation
    X_avg = np.mean(X_train, axis=1)  # Shape: (samples, features)

    print(f"Feature shape after averaging over time: {X_avg.shape}")

    # Compute PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_avg)

    print(f"PCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

    # Plot PCA
    plt.figure(figsize=(12, 8))

    # Plot each class with different color
    unique_labels = np.unique(y_train)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for idx, label in enumerate(unique_labels):
        mask = y_train == label
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[idx]], label=f"{label}: {label_map[label][:20]}...",
                   alpha=0.6, s=50)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    plt.title("PCA: Feature Separability (averaged over time)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig("feature_separability_pca.png")
    print("✅ Saved PCA plot to 'feature_separability_pca.png'")

    # Check if classes are well-separated
    if pca.explained_variance_ratio_[0] < 0.1:
        print("\n⚠️  WARNING: Low variance explained by PC1!")
        print("   Features may not be discriminative enough")


def analyze_feature_statistics(X_train):
    """Analyze basic feature statistics"""
    print("\n=== Feature Statistics ===")

    print(f"Feature shape: {X_train.shape}")
    print(f"  (samples, time_steps, feature_dim)")

    print(f"\nValue range: [{X_train.min():.4f}, {X_train.max():.4f}]")
    print(f"Mean: {X_train.mean():.4f}")
    print(f"Std: {X_train.std():.4f}")

    # Check for NaN or Inf
    if np.isnan(X_train).any():
        print("⚠️  WARNING: NaN values detected!")

    if np.isinf(X_train).any():
        print("⚠️  WARNING: Inf values detected!")

    # Check for zero features
    zero_features = np.all(X_train == 0, axis=(0, 1))
    num_zero_features = np.sum(zero_features)
    if num_zero_features > 0:
        print(f"⚠️  WARNING: {num_zero_features} features are always zero!")

    # Check variance across samples
    feature_vars = np.var(X_train, axis=(0, 1))
    low_var_features = np.sum(feature_vars < 0.0001)
    if low_var_features > 0:
        print(f"⚠️  WARNING: {low_var_features} features have very low variance!")
        print("   These features won't help with discrimination")


def main():
    print("="*60)
    print("LSM Feature Quality Analysis")
    print("="*60)

    # Load data
    X_train, y_train, feature_type = load_features()
    if X_train is None:
        return

    label_map = load_label_map()

    print(f"\nFeature type: {feature_type}")
    print(f"Number of samples: {len(X_train)}")
    print(f"Number of classes: {len(np.unique(y_train))}")

    # Run analyses
    analyze_feature_statistics(X_train)
    analyze_temporal_structure(X_train, y_train, label_map)
    analyze_separability(X_train, y_train, label_map)

    print("\n" + "="*60)
    print("Analysis complete! Check the generated plots:")
    print("  - temporal_structure_analysis.png")
    print("  - feature_separability_pca.png")
    print("="*60)


if __name__ == "__main__":
    main()
