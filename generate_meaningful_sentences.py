"""
Generate Meaningful Sentences for Training
- Using 35 words selected for maximum phonetic coverage
- Generating MEANINGFUL sentences (~5 words each)
- Using 'chatterbox-tts' for local, free audio generation
- Full alphabet coverage (a-z) for CTC training
"""

import csv
from pathlib import Path
import random
import torch
import torchaudio as ta

# --- Check for chatterbox-tts installation ---
try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    print("Error: 'chatterbox-tts' library not found.")
    print("Please install it by running: pip install chatterbox-tts")
    print("You also need: pip install torch torchaudio")
    exit()


# ===================================================
# === MINIMAL VOCABULARY (35 words - Full Phonetic Coverage) ===
# ===================================================

NOUNS = [
    "cat", "dog", "fox", "bird", "pig",      # animals
    "box", "cup", "jar", "quiz", "pen"       # objects
]

VERBS = [
    "runs", "jumps", "walks", "sleeps",      # actions
    "eats", "plays", "waits", "thinks"       # more actions
]

ADJECTIVES = [
    "big", "red", "quick", "lazy",           # descriptors
    "young", "very", "next"                  # more descriptors
]

ARTICLES = ["the", "a", "my", "your"]

PREPOSITIONS = ["in", "on", "with", "by", "from"]

MISC = ["zero"]

ALL_WORDS = NOUNS + VERBS + ADJECTIVES + ARTICLES + PREPOSITIONS + MISC

print(f"Total vocabulary: {len(ALL_WORDS)} words")
print(f"Vocabulary: {sorted(ALL_WORDS)}")

# Check alphabet coverage
all_chars = set()
for word in ALL_WORDS:
    all_chars.update(word.lower())
missing_chars = set('abcdefghijklmnopqrstuvwxyz') - all_chars
print(f"\nAlphabet coverage: {len(all_chars)}/26 letters")
if missing_chars:
    print(f"Missing letters: {sorted(missing_chars)}")
else:
    print("✓ Full alphabet coverage!")
print()


# ===================================================
# === MEANINGFUL SENTENCE TEMPLATES ===
# ===================================================

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

    # --- Category 7: Zero constructions (2-3 words) ---
    # Pattern: zero [noun]
    for noun in animals + objects:
        sentences.append(f"zero {noun}")  # 2 words, but needed for 'z'

    # Make some 3-word with zero
    for article in ["the"]:
        sentences.append(f"{article} zero {noun[:4]}")  # "the zero cat" etc (whimsical but valid)

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


# ===================================================
# === GENERATE DATASET ===
# ===================================================

random.seed(42)  # For reproducibility

print("\n" + "="*80)
print("Generating meaningful 3-4 word sentences...")
print("="*80)

all_sentences = generate_meaningful_sentences()
print(f"\nGenerated {len(all_sentences)} total sentences")

# Filter to only 3-4 word sentences
sentences_3_4 = [s for s in all_sentences if 3 <= len(s.split()) <= 4]
print(f"Filtered to 3-4 word sentences: {len(sentences_3_4)}")

# Remove duplicates
unique_sentences = list(set(sentences_3_4))
print(f"After deduplication: {len(unique_sentences)} unique sentences")

# Count by word length
from collections import Counter
length_counts = Counter(len(s.split()) for s in unique_sentences)
print(f"\nBreakdown:")
print(f"  3-word sentences: {length_counts[3]}")
print(f"  4-word sentences: {length_counts[4]}")

# Shuffle
random.shuffle(unique_sentences)

# Take target amount
TARGET_SENTENCES = 1000
if len(unique_sentences) < TARGET_SENTENCES:
    print(f"\n⚠️  Warning: Only generated {len(unique_sentences)} unique sentences")
    print(f"   Target was {TARGET_SENTENCES}")
    print(f"   Using all {len(unique_sentences)} sentences")
    sentences = unique_sentences
else:
    sentences = unique_sentences[:TARGET_SENTENCES]

print(f"\nFinal dataset: {len(sentences)} sentences (3-4 words each)")

# Show samples by category
print(f"\n{'='*80}")
print("Sample sentences by length:")
print("="*80)

by_length = {}
for s in sentences[:100]:  # Sample first 100
    length = len(s.split())
    by_length.setdefault(length, []).append(s)

for length in sorted(by_length.keys()):
    print(f"\n{length}-word sentences:")
    for s in by_length[length][:5]:  # Show 5 examples
        print(f"  • {s}")


# ===================================================
# === AUDIO GENERATION (using Chatterbox) ===
# ===================================================

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"\n{'='*80}")
print(f"Initializing ChatterboxTTS on device: '{device}'")
if device == "cpu":
    print("WARNING: Running on CPU. Generation will be significantly slower.")
print("This may take a while on the first run to download the model...")

try:
    model = ChatterboxTTS.from_pretrained(device=device)
    model_sr = model.sr
except Exception as e:
    print(f"\nCritical Error loading model: {e}")
    print("Please ensure you have a working internet connection for the first download.")
    exit()

output_dir = Path("sentences")
output_dir.mkdir(exist_ok=True)
metadata_file = output_dir / "sentences.csv"

print(f"\nStarting audio generation for {len(sentences)} sentences...")
print(f"Audio files will be saved in: '{output_dir.resolve()}'")
print(f"{'='*80}\n")

with open(metadata_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label_text"])

    for i, text in enumerate(sentences):
        print(f"Progress: {i + 1}/{len(sentences)} - '{text}'", end='\r')

        filename = f"sentence_{i + 1}.wav"
        filepath = output_dir / filename

        try:
            wav = model.generate(text)
            ta.save(filepath, wav, model_sr)
            writer.writerow([filename, text])

        except Exception as e:
            print(f"\n--- Error generating audio for: '{text}' ---")
            print(f"Error: {e}")
            print("Skipping this sentence.")
            print("-" * (len(text) + 34))


print(f"\n\n{'='*80}")
print(f"✅ All processing complete!")
print(f"✅ Metadata saved to '{metadata_file.resolve()}'")
print(f"✅ Audio files saved to '{output_dir.resolve()}/'")
print(f"{'='*80}")
