"""
Check if LSM traces are discriminative for different sentences
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_label_map(filepath="sentence_label_map_500.txt"):
    """Loads the sentence_label_map.txt file"""
    if not Path(filepath).exists():
        return None

    label_map = {}
    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    return label_map

def compute_trace_features(traces):
    """
    Compute simple features from traces to check discriminability

    Features:
    - Mean activity per neuron
    - Std activity per neuron
    - Max activity per neuron
    """
    # traces shape: (samples, timesteps, neurons)
    mean_activity = traces.mean(axis=1)  # (samples, neurons)
    std_activity = traces.std(axis=1)    # (samples, neurons)
    max_activity = traces.max(axis=1)    # (samples, neurons)

    # Concatenate all features
    features = np.concatenate([mean_activity, std_activity, max_activity], axis=1)
    return features

def main():
    print("="*60)
    print("CHECKING LSM TRACE DISCRIMINABILITY")
    print("="*60)

    # Load data
    label_map = load_label_map()
    trace_file = "lsm_trace_sequences_sentence_split_500.npz"

    if not Path(trace_file).exists():
        print(f"❌ Error: Trace file not found")
        return

    dataset = np.load(trace_file, allow_pickle=True)
    X_train = dataset["X_train_sequences"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test_sequences"]
    y_test = dataset["y_test"]

    print(f"\n✅ Loaded data")
    print(f"   Train shape: {X_train.shape}")
    print(f"   Test shape: {X_test.shape}")

    # Compute statistics
    print(f"\n--- Trace Statistics ---")
    print(f"  Train trace range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  Train mean: {X_train.mean():.3f}, std: {X_train.std():.3f}")
    print(f"  Test trace range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    print(f"  Test mean: {X_test.mean():.3f}, std: {X_test.std():.3f}")

    # Check for dead neurons (always zero)
    train_max_per_neuron = X_train.max(axis=(0, 1))  # Max across samples and time
    dead_neurons = np.sum(train_max_per_neuron == 0)
    print(f"\n  Dead neurons (never active): {dead_neurons}/{X_train.shape[2]}")

    # Check for saturation
    threshold = 2.0  # From MEMBRANE_THRESHOLD
    saturated_ratio = np.mean(X_train >= threshold)
    print(f"  Saturation (>= threshold): {saturated_ratio*100:.2f}%")

    # Compute simple features
    print(f"\n--- Computing discriminability features ---")
    train_features = compute_trace_features(X_train)
    test_features = compute_trace_features(X_test)

    print(f"  Feature shape: {train_features.shape}")

    # Check if traces are too similar (compute pairwise distances)
    print(f"\n--- Pairwise Similarity Check ---")
    # Sample 50 random pairs
    np.random.seed(42)
    num_pairs = 50
    indices = np.random.choice(len(train_features), size=(num_pairs, 2), replace=False)

    similarities = []
    for i, j in indices:
        # Cosine similarity
        feat_i = train_features[i]
        feat_j = train_features[j]
        similarity = np.dot(feat_i, feat_j) / (np.linalg.norm(feat_i) * np.linalg.norm(feat_j))
        similarities.append(similarity)

    avg_similarity = np.mean(similarities)
    print(f"  Average cosine similarity: {avg_similarity:.3f}")
    print(f"  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")

    if avg_similarity > 0.95:
        print(f"  ⚠️  WARNING: Traces are very similar! LSM may not be discriminative.")
    elif avg_similarity > 0.80:
        print(f"  ⚠️  CAUTION: Traces have high similarity.")
    else:
        print(f"  ✅ Traces show good variation.")

    # Visualize using PCA
    print(f"\n--- Running PCA for visualization ---")
    pca = PCA(n_components=2)
    train_pca = pca.fit_transform(train_features)

    explained_var = pca.explained_variance_ratio_
    print(f"  Explained variance: PC1={explained_var[0]:.3f}, PC2={explained_var[1]:.3f}")
    print(f"  Total explained: {explained_var.sum():.3f}")

    if explained_var[0] > 0.99:
        print(f"  ⚠️  WARNING: First PC explains >99% variance!")
        print(f"     This suggests traces are nearly identical!")

    # Plot PCA
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    # Color by first 10 sentences
    colors = y_train[:50]  # First 50 samples
    scatter = plt.scatter(train_pca[:50, 0], train_pca[:50, 1],
                         c=colors, cmap='tab20', alpha=0.6, s=50)
    plt.xlabel(f'PC1 ({explained_var[0]:.1%} var)')
    plt.ylabel(f'PC2 ({explained_var[1]:.1%} var)')
    plt.title('PCA of LSM Trace Features (First 50 samples)')
    plt.colorbar(scatter, label='Sentence ID')
    plt.grid(True, alpha=0.3)

    # Plot first 3 trace examples
    plt.subplot(1, 2, 2)
    for i in range(3):
        text = label_map[y_train[i]] if label_map else f"Sentence {y_train[i]}"
        plt.plot(X_train[i, :, 0], label=f"S{i}: {text[:20]}...", alpha=0.7)
    plt.xlabel('Timestep')
    plt.ylabel('Membrane Potential (Neuron 0)')
    plt.title('Example Traces (First Neuron Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('lsm_discriminability_check.png', dpi=150)
    print(f"\n✅ Visualization saved to 'lsm_discriminability_check.png'")

    # Check if different sentences produce different mean activities
    print(f"\n--- Checking between-sentence vs within-sentence variation ---")
    # Get first 10 unique sentences
    unique_sentences = np.unique(y_train)[:10]

    sentence_mean_features = []
    for sent_id in unique_sentences:
        mask = y_train == sent_id
        if np.sum(mask) > 0:
            mean_feat = train_features[mask].mean(axis=0)
            sentence_mean_features.append(mean_feat)

    sentence_mean_features = np.array(sentence_mean_features)

    # Compute variation between sentences
    between_var = sentence_mean_features.std(axis=0).mean()
    # Compute variation within first sentence
    mask = y_train == unique_sentences[0]
    within_var = train_features[mask].std(axis=0).mean() if np.sum(mask) > 1 else 0

    print(f"  Between-sentence variation: {between_var:.4f}")
    print(f"  Within-sentence variation: {within_var:.4f}")

    if within_var > 0:
        ratio = between_var / within_var
        print(f"  Between/Within ratio: {ratio:.2f}")

        if ratio < 1.5:
            print(f"  ⚠️  WARNING: Low discriminability! Ratio should be >> 1")
        elif ratio < 3.0:
            print(f"  ⚠️  CAUTION: Moderate discriminability")
        else:
            print(f"  ✅ Good discriminability")

    print("\n" + "="*60)

if __name__ == "__main__":
    main()
