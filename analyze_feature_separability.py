"""
Analyze LSM Feature Separability

This script checks if LSM features can distinguish between similar words/sentences.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def load_label_map(filepath="sentence_label_map_500.txt"):
    """Load sentence labels"""
    label_map = {}
    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    return label_map

def analyze_separability():
    """Analyze how well LSM features separate similar sentences"""

    # Load data
    print("Loading LSM features...")
    data = np.load("lsm_windowed_features_filtered_sentence_split_500.npz")
    X_train = data['X_train_sequences']  # (400, 99, 577)
    y_train = data['y_train']

    label_map = load_label_map()

    # Average over time to get sentence-level features
    X_train_avg = X_train.mean(axis=1)  # (400, 577)

    print(f"Feature shape: {X_train_avg.shape}")

    # Find pairs of similar sentences
    similar_pairs = [
        ("a bird sits", "a bird rests"),
        ("the cat sits", "the dog sits"),
        ("the wolf sits", "the fox sits"),
        ("in the park", "on the mat"),
    ]

    print("\n" + "="*60)
    print("FEATURE SEPARABILITY ANALYSIS")
    print("="*60)

    for phrase1, phrase2 in similar_pairs:
        # Find sentences containing these phrases
        idx1 = [i for i, idx in enumerate(y_train)
                if phrase1 in label_map[idx]]
        idx2 = [i for i, idx in enumerate(y_train)
                if phrase2 in label_map[idx]]

        if len(idx1) > 0 and len(idx2) > 0:
            # Get features
            feat1 = X_train_avg[idx1[0]]
            feat2 = X_train_avg[idx2[0]]

            # Calculate similarity
            similarity = cosine_similarity([feat1], [feat2])[0, 0]
            distance = np.linalg.norm(feat1 - feat2)

            sent1 = label_map[y_train[idx1[0]]]
            sent2 = label_map[y_train[idx2[0]]]

            print(f"\nComparing:")
            print(f"  1: '{sent1}'")
            print(f"  2: '{sent2}'")
            print(f"  Cosine similarity: {similarity:.4f}")
            print(f"  Euclidean distance: {distance:.2f}")

            if similarity > 0.95:
                print("  ⚠️  VERY SIMILAR - hard to distinguish!")
            elif similarity > 0.85:
                print("  ⚠️  Similar - may cause confusion")
            else:
                print("  ✓ Sufficiently different")

    # Calculate overall feature statistics
    print("\n" + "="*60)
    print("OVERALL FEATURE STATISTICS")
    print("="*60)

    # Pairwise distances between all training samples
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(X_train_avg)

    # Get distances for samples with same vs different labels
    same_class_dists = []
    diff_class_dists = []

    for i in range(len(y_train)):
        for j in range(i+1, len(y_train)):
            if y_train[i] == y_train[j]:
                same_class_dists.append(distances[i, j])
            else:
                diff_class_dists.append(distances[i, j])

    print(f"Within-class distance (same sentence): {np.mean(same_class_dists):.2f} ± {np.std(same_class_dists):.2f}")
    print(f"Between-class distance (diff sentences): {np.mean(diff_class_dists):.2f} ± {np.std(diff_class_dists):.2f}")

    ratio = np.mean(diff_class_dists) / np.mean(same_class_dists) if same_class_dists else 0
    print(f"Separation ratio (higher is better): {ratio:.2f}")

    if ratio < 1.5:
        print("⚠️  WARNING: Poor separation - features are not discriminative enough!")
    elif ratio < 3.0:
        print("⚠️  Moderate separation - features somewhat discriminative")
    else:
        print("✓ Good separation - features are discriminative")

    # Visualize with PCA
    print("\n" + "="*60)
    print("Generating PCA visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train_avg)

    plt.figure(figsize=(10, 8))

    # Color by first 10 unique labels
    unique_labels = np.unique(y_train)[:10]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, label_id in enumerate(unique_labels):
        mask = y_train == label_id
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[colors[i]], label=f"Sent {label_id}", alpha=0.6, s=50)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    plt.title("LSM Feature Separability (PCA)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("lsm_feature_separability.png", dpi=150)
    print("✅ Saved visualization to 'lsm_feature_separability.png'")

    print("\n" + "="*60)
    print(f"Total variance explained by PC1+PC2: {sum(pca.explained_variance_ratio_[:2])*100:.1f}%")
    print("="*60)

if __name__ == "__main__":
    analyze_separability()
