"""
Generate 200 Diverse Sentences for Training

Optimized for quick testing with good vocabulary coverage
"""

from elevenlabs.client import ElevenLabs
import csv
from pathlib import Path
import random

client = ElevenLabs(api_key="sk_a3614def3d14243a3bebfbb352bd93fdbc30d120d0f9571e")

# Sentence templates with word banks
templates = []

# === TEMPLATE 1: Subject + Verb + Location ===
subjects1 = ["The cat", "The dog", "A bird", "The rabbit", "A mouse", "The horse",
             "A squirrel", "The owl", "A deer", "The fox", "A bear", "The wolf"]
verbs1 = ["sleeps", "sits", "rests", "hides", "waits", "stands", "stays", "lies"]
locations1 = ["on the mat", "in the park", "by the tree", "under the bush",
              "near the house", "in the garden", "on the hill", "by the river",
              "in the forest", "under the bridge", "on the porch", "in the barn"]

for s in subjects1:
    for v in verbs1[:4]:
        for l in locations1[:3]:  # Reduced to 3
            templates.append(f"{s} {v} {l}")

# === TEMPLATE 2: Subject + Verb + Adverb ===
subjects2 = ["She", "He", "The child", "The boy", "The girl", "The man",
             "The woman", "I", "We", "They", "My friend", "The student"]
verbs2 = ["walks", "runs", "moves", "dances", "sings", "talks", "works",
          "reads", "writes", "plays", "thinks", "laughs"]
adverbs = ["quickly", "slowly", "carefully", "happily", "quietly", "loudly"]

for s in subjects2[:6]:  # First 6 subjects
    for v in verbs2[:4]:
        for a in adverbs[:2]:
            templates.append(f"{s} {v} {a}")

# === TEMPLATE 3: The + Noun + Verb + Object ===
nouns3 = ["teacher", "doctor", "farmer", "baker", "driver", "painter",
          "singer", "dancer", "writer", "player"]
verbs3 = ["helps", "teaches", "makes", "builds", "creates", "fixes"]
objects3 = ["the students", "the people", "a plan", "something new", "the work"]

for n in nouns3[:5]:
    for v in verbs3[:3]:
        for o in objects3[:2]:
            templates.append(f"The {n} {v} {o}")

# === TEMPLATE 4: Weather and Nature ===
weather = ["The rain", "Snow", "The wind", "Thunder", "Lightning", "The sun",
           "The moon", "Clouds"]
weather_verbs = ["falls", "blows", "shines", "moves", "drifts"]
weather_loc = ["on the ground", "through the trees", "in the sky", "across the field"]

for w in weather:
    for v in weather_verbs[:2]:
        for l in weather_loc[:2]:
            templates.append(f"{w} {v} {l}")

# === TEMPLATE 5: Questions ===
question_starts = ["Where", "When", "Why", "How", "What", "Who"]
question_middles = ["did you go", "is it", "can we do", "should I say"]
question_ends = ["today", "now", "yesterday", "tomorrow"]

for q in question_starts:
    for m in question_middles[:2]:
        for e in question_ends[:2]:
            templates.append(f"{q} {m} {e}")

# === TEMPLATE 6: Common Phrases ===
phrases = [
    "Thank you very much",
    "You are welcome",
    "Have a nice day",
    "See you later",
    "Good morning",
    "Good evening",
    "Good night",
    "How are you today",
    "I am fine thanks",
    "Nice to meet you",
    "Please come in",
    "Take a seat",
    "Can I help you",
    "Of course you can",
    "Let me see",
    "Wait a moment",
    "Just a minute",
    "No problem at all",
    "That sounds good",
    "I think so too",
]

templates.extend(phrases)

# === TEMPLATE 7: Time-based ===
subjects7 = ["I", "We", "They", "She", "He"]
time_verbs = ["arrive", "leave", "wake up", "go to sleep"]
times = ["at six", "at seven", "in the morning", "in the evening"]

for s in subjects7:
    for v in time_verbs[:2]:
        for t in times[:2]:
            templates.append(f"{s} {v} {t}")

# === TEMPLATE 8: Numbers ===
subjects8 = ["There are", "I see", "We have"]
numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight"]
items = ["birds", "books", "chairs", "trees", "cars", "people"]

for s in subjects8:
    for n in numbers[:4]:
        for i in items[:2]:
            templates.append(f"{s} {n} {i}")

# Remove duplicates and shuffle
templates = list(set(templates))
random.seed(42)
random.shuffle(templates)

# Select 200 sentences
sentences = templates[:200]

print(f"Generated {len(sentences)} unique sentences")
print(f"\nSample sentences (first 10):")
for i in range(10):
    print(f"  {i+1:3d}. {sentences[i]}")

# Calculate vocabulary
all_words = set()
for sent in sentences:
    all_words.update(sent.lower().split())

print(f"\nVocabulary statistics:")
print(f"  Total unique words: {len(all_words)}")
print(f"  Average sentence length: {sum(len(s.split()) for s in sentences) / len(sentences):.1f} words")

# Generate audio
voices = ["JBFqnCBsd6RMkjVDRZzb"]

output_dir = Path("sentences_200")
output_dir.mkdir(exist_ok=True)
metadata_file = output_dir / "sentences.csv"

print(f"\n{'='*80}")
print(f"Starting audio generation for {len(sentences)} sentences...")
print(f"Estimated time: ~{len(sentences) * 2 / 60:.0f} minutes")
print(f"{'='*80}\n")

with open(metadata_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label_text"])

    for i, text in enumerate(sentences):
        voice_id = voices[i % len(voices)]

        # Progress indicator every 20 sentences
        if (i + 1) % 20 == 0 or i == 0:
            elapsed_mins = (i + 1) * 2 / 60
            total_mins = len(sentences) * 2 / 60
            print(f"Progress: {i + 1}/{len(sentences)} ({(i+1)/len(sentences)*100:.1f}%) - {elapsed_mins:.1f}/{total_mins:.0f} min")

        try:
            audio = client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128",
            )

            filename = f"sentence_{i + 1}.mp3"
            filepath = output_dir / filename

            with open(filepath, "wb") as audio_f:
                for chunk in audio:
                    audio_f.write(chunk)

            writer.writerow([filename, text])

        except Exception as e:
            print(f"  ⚠️  Error generating sentence {i+1}: {e}")
            continue

print(f"\n{'='*80}")
print(f"✅ Audio generation complete!")
print(f"✅ Generated: {len(sentences)} audio files")
print(f"✅ Metadata saved to: {metadata_file}")
print(f"✅ Audio files saved to: {output_dir}/")
print(f"{'='*80}")
