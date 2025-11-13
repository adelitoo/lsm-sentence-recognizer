
"""
Generate 500 Diverse Sentences for Training
- Using 35 words selected for maximum phonetic coverage
- Generating meaningful sentences (~5 words each) by rearranging words
- Using 'chatterbox-tts' for local, free audio generation.
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

# Carefully selected for:
# 1. Full alphabet coverage (a-z)
# 2. Maximum phonetic diversity
# 3. Common consonant clusters (st, br, ch, sh, qu, etc.)

NOUNS = [
    "cat", "dog", "fox", "bird", "pig",      # animals
    "box", "cup", "jar", "quiz", "pen"       # objects (includes q, z, j)
]  # 10 words

VERBS = [
    "runs", "jumps", "walks", "sleeps",      # actions (includes j)
    "eats", "plays", "waits", "thinks"       # more actions
]  # 8 verbs

ADJECTIVES = [
    "big", "red", "quick", "lazy",           # descriptors (includes q, z)
    "young", "very", "next"                  # more descriptors (includes y, v, x via next)
]  # 7 adjectives

ARTICLES = ["the", "a", "my", "your"]  # 4 words

PREPOSITIONS = ["in", "on", "with", "by", "from"]  # 5 words

MISC = ["zero"]  # 1 word - ensures 'z' is covered

# Total: 35 words
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
# === SENTENCE GENERATION (~5 words each) ===
# ===================================================

def generate_sentence():
    """Generate meaningful ~5 word sentences using 35 words"""

    sentence_patterns = [
        # Pattern 1: Article + Noun + Verb
        lambda: f"{random.choice(ARTICLES)} {random.choice(NOUNS)} {random.choice(VERBS)}",

        # Pattern 2: Article + Adjective + Noun + Verb
        lambda: f"{random.choice(ARTICLES)} {random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.choice(VERBS)}",

        # Pattern 3: Article + Noun + Verb + Preposition + Article + Noun
        lambda: f"{random.choice(ARTICLES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(PREPOSITIONS)} {random.choice(ARTICLES)} {random.choice(NOUNS)}",

        # Pattern 4: Article + Noun + Verb + Article + Noun
        lambda: f"{random.choice(ARTICLES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(ARTICLES)} {random.choice(NOUNS)}",

        # Pattern 5: Adjective + Noun + Verb + Preposition + Noun
        lambda: f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(PREPOSITIONS)} {random.choice(NOUNS)}",

        # Pattern 6: Article + Noun + Verb + Adjective + Noun
        lambda: f"{random.choice(ARTICLES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(ADJECTIVES)} {random.choice(NOUNS)}",

        # Pattern 7: Article + Adjective + Noun + Verb + Preposition + Noun
        lambda: f"{random.choice(ARTICLES)} {random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(PREPOSITIONS)} {random.choice(NOUNS)}",

        # Pattern 8: Article + Noun + Verb + Preposition + Adjective + Noun
        lambda: f"{random.choice(ARTICLES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(PREPOSITIONS)} {random.choice(ADJECTIVES)} {random.choice(NOUNS)}",

        # Pattern 9: Noun + Verb + Article + Noun
        lambda: f"{random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(ARTICLES)} {random.choice(NOUNS)}",

        # Pattern 10: Article + Noun + Verb + Noun
        lambda: f"{random.choice(ARTICLES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(NOUNS)}",

        # Pattern 11: Adjective + Noun + Verb
        lambda: f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)} {random.choice(VERBS)}",

        # Pattern 12: Article + Noun + Verb + Preposition + Noun
        lambda: f"{random.choice(ARTICLES)} {random.choice(NOUNS)} {random.choice(VERBS)} {random.choice(PREPOSITIONS)} {random.choice(NOUNS)}",
    ]

    return random.choice(sentence_patterns)()


# ===================================================
# === GENERATE 500 SENTENCES WITH DATA AUGMENTATION ===
# ===================================================

random.seed(42)  # For reproducibility
base_sentences = []
attempts = 0
max_attempts = 10000

# Configuration: Change this to scale dataset
TARGET_SENTENCES = 1000  # <-- CHANGE THIS (was 500)

# First, generate as many unique base sentences as possible
print("Generating base sentences...")
while len(base_sentences) < TARGET_SENTENCES and attempts < max_attempts:
    sentence = generate_sentence()
    if sentence not in base_sentences:
        base_sentences.append(sentence)
    attempts += 1

print(f"Generated {len(base_sentences)} unique base sentences")

# If we need more sentences, use data augmentation
sentences = base_sentences.copy()

if len(sentences) < TARGET_SENTENCES:
    print(f"\nNeed {TARGET_SENTENCES - len(sentences)} more sentences. Applying data augmentation...")

    augmentation_methods = [
        # Method 1: Add number at beginning (zero, one, two, etc.)
        lambda s: f"zero {s}" if "zero" not in s else s,

        # Method 2: Add "very" before adjectives if sentence contains adjective
        lambda s: s.replace(f" {random.choice(ADJECTIVES)} ", f" very {random.choice(ADJECTIVES)} ", 1)
                  if any(adj in s for adj in ADJECTIVES) else s,

        # Method 3: Duplicate with different article (the <-> a)
        lambda s: s.replace(" the ", " a ", 1) if " the " in s else s.replace(" a ", " the ", 1),

        # Method 4: Add preposition phrase at end
        lambda s: f"{s} {random.choice(PREPOSITIONS)} {random.choice(NOUNS)}",
    ]

    augmented = 0
    for base in base_sentences:
        if len(sentences) >= 500:
            break

        for method in augmentation_methods:
            if len(sentences) >= 500:
                break

            aug_sentence = method(base)
            if aug_sentence != base and aug_sentence not in sentences:
                sentences.append(aug_sentence)
                augmented += 1

    print(f"Added {augmented} augmented sentences")

# Ensure we have exactly TARGET_SENTENCES
sentences = sentences[:TARGET_SENTENCES]

print(f"\nFinal count: {len(sentences)} sentences")

print(f"Generated {len(sentences)} unique sentences")
print(f"\nSample sentences:")
for i in range(min(10, len(sentences))): # Show up to 10 samples
    print(f"  {i+1}. {sentences[i]}")


# ===================================================
# === AUDIO GENERATION (using Chatterbox) ===
# ===================================================

# 1. Automatically detect the best device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available(): # For Apple Silicon Macs
    device = "mps"
else:
    device = "cpu"

print(f"\n{'='*80}")
print(f"Initializing ChatterboxTTS on device: '{device}'")
if device == "cpu":
    print("WARNING: Running on CPU. Generation will be significantly slower.")
print("This may take a while on the first run to download the model...")

# 2. Load the model
try:
    model = ChatterboxTTS.from_pretrained(device=device)
    model_sr = model.sr  # Get the model's sample rate
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

# 3. Process and Save
with open(metadata_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label_text"])

    for i, text in enumerate(sentences):
        # Progress indicator
        print(f"Progress: {i + 1}/{len(sentences)}", end='\r')

        # Note: Chatterbox saves as .wav
        filename = f"sentence_{i + 1}.wav"
        filepath = output_dir / filename

        try:
            # 4. Generate the audio waveform
            wav = model.generate(text)
            
            # 5. Save the audio file
            ta.save(filepath, wav, model_sr)

            # 6. Write to CSV
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
