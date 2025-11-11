"""
Create a Balanced Sentence-Level Split

Strategy:
- Ensure test sentences are unseen during training (TRUE generalization)
- But maximize word overlap to make the task feasible
- Aim for ~70-80% of test words appearing in training vocabulary
"""

import numpy as np
from collections import Counter

def load_label_map(filepath="sentence_label_map.txt"):
    """Load sentence labels"""
    label_map = {}
    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    return label_map

def get_sentence_words(sentence):
    """Get set of words in a sentence"""
    return set(sentence.split())

def calculate_word_coverage(train_sentences, test_sentences):
    """Calculate what % of test words appear in training"""
    train_words = set()
    for sent in train_sentences:
        train_words.update(sent.split())

    test_words = set()
    for sent in test_sentences:
        test_words.update(sent.split())

    covered_words = train_words & test_words
    coverage = len(covered_words) / len(test_words) if test_words else 0

    return coverage, train_words, test_words, covered_words

def create_balanced_split(label_map, test_size=0.2, target_coverage=0.75, random_state=42):
    """
    Create sentence-level split that maximizes word coverage in test set.

    Algorithm:
    1. Start with all sentences as candidates for training
    2. Iteratively select test sentences that maximize word overlap with remaining training candidates
    3. Stop when we reach target test size
    """
    np.random.seed(random_state)

    all_sentence_ids = list(label_map.keys())
    all_sentences = {idx: label_map[idx] for idx in all_sentence_ids}

    num_test = int(len(all_sentence_ids) * test_size)

    print("="*80)
    print("CREATING BALANCED SENTENCE-LEVEL SPLIT")
    print("="*80)
    print(f"Total sentences: {len(all_sentence_ids)}")
    print(f"Target test size: {num_test} ({test_size*100:.0f}%)")
    print(f"Target word coverage: {target_coverage*100:.0f}%")
    print()

    # Try multiple random orderings and pick the best one
    best_split = None
    best_coverage = 0

    for attempt in range(100):  # Try 100 different orderings
        np.random.seed(random_state + attempt)
        shuffled_ids = all_sentence_ids.copy()
        np.random.shuffle(shuffled_ids)

        # Greedy selection: pick test sentences that have maximum word overlap with training
        test_ids = []
        train_ids = shuffled_ids.copy()

        for _ in range(num_test):
            if not train_ids:
                break

            # Get current training vocabulary
            train_vocab = set()
            for tid in train_ids:
                train_vocab.update(get_sentence_words(all_sentences[tid]))

            # Find candidate test sentence with maximum word overlap
            best_candidate = None
            best_overlap = -1

            for candidate_id in train_ids:
                candidate_words = get_sentence_words(all_sentences[candidate_id])
                # Calculate how many of its words would be covered by remaining training
                remaining_train = [tid for tid in train_ids if tid != candidate_id]
                remaining_vocab = set()
                for tid in remaining_train:
                    remaining_vocab.update(get_sentence_words(all_sentences[tid]))

                overlap = len(candidate_words & remaining_vocab)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_candidate = candidate_id

            if best_candidate is not None:
                test_ids.append(best_candidate)
                train_ids.remove(best_candidate)

        # Calculate coverage for this split
        train_sentences = [all_sentences[i] for i in train_ids]
        test_sentences = [all_sentences[i] for i in test_ids]
        coverage, _, _, _ = calculate_word_coverage(train_sentences, test_sentences)

        if coverage > best_coverage:
            best_coverage = coverage
            best_split = (train_ids, test_ids)

        if coverage >= target_coverage:
            break  # Found good enough split

    train_ids, test_ids = best_split
    train_sentences = [all_sentences[i] for i in train_ids]
    test_sentences = [all_sentences[i] for i in test_ids]

    # Calculate final statistics
    coverage, train_words, test_words, covered_words = calculate_word_coverage(
        train_sentences, test_sentences
    )

    uncovered_words = test_words - train_words

    print(f"✅ Best split found (attempt with {best_coverage*100:.1f}% coverage):")
    print(f"\n  Train sentences: {len(train_ids)}")
    print(f"  Test sentences: {len(test_ids)}")
    print(f"\n  Train vocabulary: {len(train_words)} words")
    print(f"  Test vocabulary: {len(test_words)} words")
    print(f"  Word coverage: {len(covered_words)}/{len(test_words)} = {coverage*100:.1f}%")
    print(f"\n  Uncovered test words ({len(uncovered_words)}): {sorted(uncovered_words)}")

    # Verify no sentence overlap
    assert len(set(train_ids) & set(test_ids)) == 0, "Sentence overlap detected!"
    print(f"\n✅ VERIFIED: No sentence overlap between train and test!")

    # Show some example test sentences
    print(f"\n📋 Test sentences (first 10):")
    for i, idx in enumerate(test_ids[:10], 1):
        sent = all_sentences[idx]
        words = get_sentence_words(sent)
        covered = words & train_words
        uncovered = words - train_words
        print(f"  {i:2d}. {sent}")
        if uncovered:
            print(f"      ⚠️  Uncovered words: {sorted(uncovered)}")

    return np.array(train_ids), np.array(test_ids)

if __name__ == "__main__":
    # Load labels
    label_map = load_label_map()

    # Create balanced split
    train_ids, test_ids = create_balanced_split(
        label_map,
        test_size=0.2,
        target_coverage=0.75,  # Aim for 75% word coverage
        random_state=42
    )

    # Save the split for use in feature extraction
    np.savez(
        "balanced_sentence_split.npz",
        train_sentence_ids=train_ids,
        test_sentence_ids=test_ids
    )

    print("\n" + "="*80)
    print("✅ Balanced split saved to 'balanced_sentence_split.npz'")
    print("="*80)
