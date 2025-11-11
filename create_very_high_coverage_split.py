"""
Create sentence-level split with VERY HIGH word coverage (80-90%)

Strategy: Select only a few test sentences with maximum word overlap
"""

import numpy as np
from collections import Counter

def load_label_map(filepath="sentence_label_map.txt"):
    label_map = {}
    with open(filepath, "r") as f:
        next(f)
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    return label_map

def get_sentence_words(sentence):
    return set(sentence.split())

def calculate_word_coverage(train_sentences, test_sentences):
    train_words = set()
    for sent in train_sentences:
        train_words.update(sent.split())

    test_words = set()
    for sent in test_sentences:
        test_words.update(sent.split())

    covered_words = train_words & test_words
    coverage = len(covered_words) / len(test_words) if test_words else 0

    return coverage, train_words, test_words, covered_words

def create_high_coverage_split(label_map, num_test=10, random_state=42):
    """
    Create split with very high word coverage by carefully selecting test sentences.

    Uses only 10 test sentences (10%) to maximize word overlap.
    """
    np.random.seed(random_state)

    all_sentence_ids = list(label_map.keys())
    all_sentences = {idx: label_map[idx] for idx in all_sentence_ids}

    print("="*80)
    print("CREATING VERY HIGH COVERAGE SPLIT")
    print("="*80)
    print(f"Total sentences: {len(all_sentence_ids)}")
    print(f"Test sentences: {num_test} ({num_test/len(all_sentence_ids)*100:.0f}%)")
    print()

    # Try many random combinations and pick the best
    best_split = None
    best_coverage = 0

    for attempt in range(500):  # Try 500 combinations
        np.random.seed(random_state + attempt)
        shuffled_ids = all_sentence_ids.copy()
        np.random.shuffle(shuffled_ids)

        # Greedy selection
        test_ids = []
        train_ids = shuffled_ids.copy()

        for _ in range(num_test):
            if not train_ids:
                break

            # Get current training vocabulary
            train_vocab = set()
            for tid in train_ids:
                train_vocab.update(get_sentence_words(all_sentences[tid]))

            # Find test candidate with maximum overlap
            best_candidate = None
            best_overlap = -1

            for candidate_id in train_ids:
                candidate_words = get_sentence_words(all_sentences[candidate_id])
                # Remaining training after removing this sentence
                remaining_train = [tid for tid in train_ids if tid != candidate_id]
                remaining_vocab = set()
                for tid in remaining_train:
                    remaining_vocab.update(get_sentence_words(all_sentences[tid]))

                # How many test words would be covered?
                overlap = len(candidate_words & remaining_vocab)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_candidate = candidate_id

            if best_candidate is not None:
                test_ids.append(best_candidate)
                train_ids.remove(best_candidate)

        # Calculate coverage
        train_sentences = [all_sentences[i] for i in train_ids]
        test_sentences = [all_sentences[i] for i in test_ids]
        coverage, _, _, _ = calculate_word_coverage(train_sentences, test_sentences)

        if coverage > best_coverage:
            best_coverage = coverage
            best_split = (train_ids, test_ids)

        if coverage >= 0.85:  # Found good enough split
            break

    train_ids, test_ids = best_split
    train_sentences = [all_sentences[i] for i in train_ids]
    test_sentences = [all_sentences[i] for i in test_ids]

    # Final statistics
    coverage, train_words, test_words, covered_words = calculate_word_coverage(
        train_sentences, test_sentences
    )
    uncovered_words = test_words - train_words

    print(f"✅ Best split found (coverage: {coverage*100:.1f}%):")
    print(f"\n  Train sentences: {len(train_ids)} ({len(train_ids)/len(all_sentence_ids)*100:.0f}%)")
    print(f"  Test sentences: {len(test_ids)} ({len(test_ids)/len(all_sentence_ids)*100:.0f}%)")
    print(f"\n  Train vocabulary: {len(train_words)} words")
    print(f"  Test vocabulary: {len(test_words)} words")
    print(f"  Word coverage: {len(covered_words)}/{len(test_words)} = {coverage*100:.1f}%")
    print(f"\n  Uncovered test words ({len(uncovered_words)}): {sorted(uncovered_words)}")

    # Show test sentences
    print(f"\n📋 Test sentences:")
    for i, idx in enumerate(test_ids, 1):
        sent = all_sentences[idx]
        words = get_sentence_words(sent)
        uncov = words - train_words
        print(f"  {i:2d}. {sent}")
        if uncov:
            print(f"      ⚠️  Uncovered: {sorted(uncov)}")

    assert len(set(train_ids) & set(test_ids)) == 0, "Overlap detected!"
    print(f"\n✅ VERIFIED: No sentence overlap!")

    return np.array(train_ids), np.array(test_ids)

if __name__ == "__main__":
    label_map = load_label_map()

    train_ids, test_ids = create_high_coverage_split(
        label_map,
        num_test=10,  # Only 10 test sentences for very high coverage
        random_state=42
    )

    np.savez(
        "very_high_coverage_split.npz",
        train_sentence_ids=train_ids,
        test_sentence_ids=test_ids
    )

    print("\n" + "="*80)
    print("✅ Very high coverage split saved to 'very_high_coverage_split.npz'")
    print("="*80)
