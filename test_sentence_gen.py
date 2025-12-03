import random
random.seed(42)

def generate_meaningful_sentences():
    """
    Generate semantically meaningful 3-4 word sentences using the 35-word vocabulary.
    All sentences make sense in context and have consistent length for easier training.
    """

    sentences = []

    # --- Category 1: Simple Animal Actions (3 words) ---
    # Pattern: [article] [animal] [verb]
    animals = ["cat", "dog", "fox", "bird", "pig"]
    simple_actions = ["runs", "jumps", "walks", "sleeps", "eats", "plays", "waits", "thinks"]

    for article in ["the", "a", "my", "your"]:
        for animal in animals:
            for verb in simple_actions:
                sentences.append(f"{article} {animal} {verb}")

    # --- Category 2: Animal Actions with Adjectives (4 words) ---
    # Pattern: [article] [adjective] [animal] [verb]
    for article in ["the", "a", "my", "your"]:
        for adj in ["big", "red", "quick", "lazy", "young"]:
            for animal in animals:
                for verb in ["runs", "jumps", "walks", "sleeps", "plays", "eats", "waits"]:
                    sentences.append(f"{article} {adj} {animal} {verb}")

    # --- Category 3: Object Descriptions (3 words) ---
    # Pattern: [article] [adjective] [object]
    objects = ["box", "cup", "jar", "pen", "quiz"]

    for article in ["the", "a", "my", "your"]:
        for adj in ["big", "red", "quick", "lazy", "young", "next"]:
            for obj in objects:
                sentences.append(f"{article} {adj} {obj}")

    # --- Category 4: Simple Object References (3 words) ---
    # Pattern: [article] [noun] [verb]
    for article in ["the", "a", "my", "your"]:
        for obj in objects:
            for verb in ["waits"]:  # Personification - valid in storytelling
                sentences.append(f"{article} {obj} {verb}")

    # --- Category 5: Very + Adjective Constructions (4 words) ---
    # Pattern: [article] very [adjective] [noun]
    for article in ["the", "a"]:
        for adj in ["big", "quick", "lazy", "young", "red"]:
            for noun in animals:
                sentences.append(f"{article} very {adj} {noun}")

    # Add objects too
    for article in ["the", "a"]:
        for adj in ["big", "red", "next"]:
            for obj in ["box", "cup", "jar"]:
                sentences.append(f"{article} very {adj} {obj}")

    # --- Category 6: Sequential/Next constructions (3 words) ---
    # Pattern: the next [noun]
    for noun in animals + objects:
        sentences.append(f"the next {noun}")

    # --- Category 7: Zero constructions (2 words) ---
    # Pattern: zero [noun]
    for noun in animals + objects:
        sentences.append(f"zero {noun}")  # 2 words, needed for 'z' coverage

    # --- Category 8: Possessive Simple (3 words) ---
    # Pattern: [my/your] [adjective] [noun]
    for owner in ["my", "your"]:
        for adj in ["big", "red", "quick", "lazy", "young", "next"]:
            for noun in animals:
                sentences.append(f"{owner} {adj} {noun}")

    # --- Category 9: Animals "from" places (4 words) ---
    # Pattern: [article] [animal] from [location]
    for article in ["the", "a"]:
        for animal in animals:
            for location in ["box", "cup", "jar"]:
                sentences.append(f"{article} {animal} from {location}")

    # --- Category 10: Animals "in/on/by" places (4 words) ---
    # Pattern: [article] [animal] [preposition] [object]
    for article in ["the", "a"]:
        for animal in animals:
            for prep in ["in", "on", "by", "with"]:
                for obj in ["box", "cup", "jar", "pen"]:
                    sentences.append(f"{article} {animal} {prep} {obj}")

    return sentences

# Generate sentences
sentences = generate_meaningful_sentences()

# Filter to only 3-4 word sentences
sentences_3_4 = [s for s in sentences if 3 <= len(s.split()) <= 4]

print(f'Total sentences generated: {len(sentences)}')
print(f'Sentences with 3-4 words: {len(sentences_3_4)}')
print(f'Sentences with 2 words: {len([s for s in sentences if len(s.split()) == 2])}')

# Count by length
from collections import Counter
length_counts = Counter(len(s.split()) for s in sentences)
print(f'\nBreakdown by word count:')
for length in sorted(length_counts.keys()):
    print(f'  {length} words: {length_counts[length]} sentences')

# Show unique samples
print(f'\n{"="*60}')
print('Sample 3-word sentences:')
samples_3 = [s for s in sentences_3_4 if len(s.split()) == 3]
random.shuffle(samples_3)
for s in samples_3[:10]:
    print(f'  • {s}')

print(f'\nSample 4-word sentences:')
samples_4 = [s for s in sentences_3_4 if len(s.split()) == 4]
random.shuffle(samples_4)
for s in samples_4[:10]:
    print(f'  • {s}')

# Deduplicate
unique_sentences = list(set(sentences_3_4))
print(f'\n{"="*60}')
print(f'After deduplication: {len(unique_sentences)} unique 3-4 word sentences')
