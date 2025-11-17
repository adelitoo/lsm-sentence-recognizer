# Save this as debug_similarity.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

def load_data(npz_file="lsm_trace_sequences.npz", map_file="sentence_label_map.txt"):
    """Loads traces and the label map."""
    
    if not Path(npz_file).exists():
        print(f"❌ Error: NPZ file not found at '{npz_file}'")
        return None, None, None
    if not Path(map_file).exists():
        print(f"❌ Error: Label map not found at '{map_file}'")
        return None, None, None

    print(f"✅ Loading traces from '{npz_file}'...")
    data = np.load(npz_file)
    X_train_traces = data['X_train_sequences']
    y_train_labels = data['y_train']

    print(f"✅ Loading label map from '{map_file}'...")
    label_map_df = pd.read_csv(map_file)
    label_map = label_map_df.set_index('label_id')['label_text'].to_dict()
    
    print(f"  Loaded {len(X_train_traces)} training traces.")
    return X_train_traces, y_train_labels, label_map

def create_fingerprints(traces):
    """
    Creates a single "fingerprint" vector for each sample by
    averaging the traces over the time dimension.
    """
    print("  Creating 'fingerprint' vectors by averaging over time...")
    # Input shape: (Samples, Time, Channels) -> (800, 2000, 700)
    # Output shape: (Samples, Channels) -> (800, 700)
    fingerprints = np.mean(traces, axis=1)
    
    # Normalize the fingerprints (good practice for similarity)
    norm = np.linalg.norm(fingerprints, axis=1, keepdims=True)
    norm[norm == 0] = 1 # Avoid division by zero
    return fingerprints / norm

def run_similarity_test(traces, labels, label_map):
    """
    Runs a test to see if similar sentences have similar traces.
    """
    
    # --- 1. Create Fingerprints ---
    fingerprints = create_fingerprints(traces)
    
    # --- 2. Find Specific Sentences for a Targeted Test ---
    # We need to find the *index* of specific sentences
    
    target_1_text = "the big dog sleeps on quiz" # This is our test sample 0
    target_2_text = "the quick brown fox jumps"   # A very different sentence
    target_3_text = "the big cat sleeps on quiz"   # A very similar sentence

    idx_1, idx_2, idx_3 = None, None, None

    # Find the indices for our target sentences
    text_to_id = {text: idx for idx, text in label_map.items()}
    id_to_array_idx = {label: i for i, label in enumerate(labels)}

    if target_1_text in text_to_id:
        idx_1 = id_to_array_idx.get(text_to_id[target_1_text])
    
    if target_2_text in text_to_id:
        idx_2 = id_to_array_idx.get(text_to_id[target_2_text])
        
    if target_3_text in text_to_id:
        idx_3 = id_to_array_idx.get(text_to_id[target_3_text])

    print("\n--- 🎯 Targeted Similarity Test ---")
    
    if idx_1 is None or idx_2 is None:
        print("⚠️  Warning: Could not find 'the big dog' or 'the quick brown fox' in the dataset.")
        print("   Skipping targeted test.")
    else:
        # Get the fingerprint vectors
        vec_1 = fingerprints[idx_1].reshape(1, -1)
        vec_2 = fingerprints[idx_2].reshape(1, -1)
        
        sim_1_2 = cosine_similarity(vec_1, vec_2)[0][0]
        
        print(f"  A: '{target_1_text}'")
        print(f"  B: '{target_2_text}'")
        print(f"  Similarity(A, B): {sim_1_2:.4f}  <-- We want this to be LOW")

    if idx_1 is not None and idx_3 is not None:
        vec_1 = fingerprints[idx_1].reshape(1, -1)
        vec_3 = fingerprints[idx_3].reshape(1, -1)
        
        sim_1_3 = cosine_similarity(vec_1, vec_3)[0][0]
        
        print(f"\n  A: '{target_1_text}'")
        print(f"  C: '{target_3_text}'")
        print(f"  Similarity(A, C): {sim_1_3:.4f}  <-- We want this to be HIGH")
        
    elif idx_1 is not None:
        print("\n  Note: 'the big cat' not in dataset, skipping similar-pair test.")

    # --- 3. General Separation Test ---
    print("\n--- 🌐 General Separation Test ---")
    print("  Calculating average similarity across 100 random pairs...")
    
    num_samples_to_test = min(100, len(fingerprints))
    # Get 100 random indices
    indices = np.random.choice(len(fingerprints), num_samples_to_test, replace=False)
    
    # Get the 100x700 matrix
    subset_fingerprints = fingerprints[indices]
    
    # Compute the 100x100 similarity matrix
    similarity_matrix = cosine_similarity(subset_fingerprints)
    
    # We only want the off-diagonal elements
    # (np.triu_indices gives upper triangle, k=1 ignores the diagonal)
    upper_triangle_indices = np.triu_indices(num_samples_to_test, k=1)
    all_pairs_similarity = similarity_matrix[upper_triangle_indices]
    
    avg_sim = np.mean(all_pairs_similarity)
    
    print(f"  Average similarity between {len(all_pairs_similarity)} pairs:")
    print(f"  Avg. Similarity: {avg_sim:.4f}")
    
    print("\n  ℹ️  Interpretation:")
    if avg_sim > 0.5:
        print(f"  ❌ PROBLEM: Avg. similarity is > 0.5. Traces are too similar.")
        print(f"     This proves the LSM is failing to separate inputs.")
    elif avg_sim < 0.2:
        print(f"  ✅ EXCELLENT: Avg. similarity is < 0.2. Traces are very distinct.")
        print(f"     This suggests the LSM is working well!")
    else:
        print(f"  🆗 MEH: Avg. similarity is {avg_sim:.4f}. Separation is mediocre.")

def main():
    traces, labels, label_map = load_data()
    if traces is not None:
        run_similarity_test(traces, labels, label_map)

if __name__ == "__main__":
    main()