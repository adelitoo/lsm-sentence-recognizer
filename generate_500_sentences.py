
"""
Generate 500 Diverse Sentences for Training
- Using 'chatterbox-tts' for local, free audio generation.
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
# === TEMPLATE AND SENTENCE GENERATION ===
# ===================================================

templates = []

# === TEMPLATE 1: Subject + Verb + Location ===
subjects1 = ["The cat", "The dog", "A bird", "The rabbit", "A mouse", "The horse",
             "A squirrel", "The owl", "A deer", "The fox", "A bear", "The wolf"]
verbs1 = ["sleeps", "sits", "rests", "hides", "waits", "stands", "stays", "lies"]
locations1 = ["on the mat", "in the park", "by the tree", "under the bush",
              "near the house", "in the garden", "on the hill", "by the river",
              "in the forest", "under the bridge", "on the porch", "in the barn"]

for s in subjects1:
    for v in verbs1[:4]:  # Use first 4 verbs
        for l in locations1[:4]: # Use first 4 locations
            templates.append(f"{s} {v} {l}")

# === TEMPLATE 2: Subject + Verb + Adverb ===
subjects2 = ["She", "He", "The child", "The boy", "The girl", "The man",
             "The woman", "I", "We", "They", "My friend", "The student"]
verbs2 = ["walks", "runs", "moves", "dances", "sings", "talks", "works",
          "reads", "writes", "plays", "thinks", "laughs"]
adverbs = ["quickly", "slowly", "carefully", "happily", "quietly", "loudly",
           "softly", "gently", "rapidly", "calmly", "eagerly", "proudly"]

for s in subjects2:
    for v in verbs2[:3]:
        for a in adverbs[:3]:
            templates.append(f"{s} {v} {a}")

# === TEMPLATE 3: The + Noun + Verb + Object ===
nouns3 = ["teacher", "doctor", "farmer", "baker", "driver", "painter",
          "singer", "dancer", "writer", "player", "worker", "builder"]
verbs3 = ["helps", "teaches", "makes", "builds", "creates", "fixes",
          "writes", "draws", "paints", "drives", "carries", "finds"]
objects3 = ["the students", "the people", "a plan", "something new", "a tool",
            "the problem", "a story", "a picture", "the work", "a solution"]

for n in nouns3:
    for v in verbs3[:3]:
        for o in objects3[:3]:
            templates.append(f"The {n} {v} {o}")

# === TEMPLATE 4: Weather and Nature ===
weather = ["The rain", "Snow", "The wind", "Thunder", "Lightning", "The sun",
           "The moon", "Clouds", "The storm", "Fog", "Mist"]
weather_verbs = ["falls", "blows", "shines", "moves", "drifts", "rises",
                 "sets", "appears", "comes", "passes"]
weather_loc = ["on the ground", "through the trees", "in the sky", "across the field",
               "over the mountain", "from the west", "at dawn", "during the night"]

for w in weather:
    for v in weather_verbs[:3]:
        for l in weather_loc[:3]:
            templates.append(f"{w} {v} {l}")

# === TEMPLATE 5: Questions ===
question_starts = ["Where", "When", "Why", "How", "What", "Who", "Which"]
question_middles = ["did you go", "is it", "are they", "can we do", "should I say",
                    "will they come", "have you been", "does it work"]
question_ends = ["today", "now", "yesterday", "tomorrow", "this morning",
                 "last night", "next week", "right now"]

for q in question_starts:
    for m in question_middles[:2]:
        for e in question_ends[:2]:
            templates.append(f"{q} {m} {e}")

# === TEMPLATE 6: Action + Object Sentences ===
actions = ["Open", "Close", "Take", "Give", "Bring", "Send", "Show", "Tell",
           "Make", "Find", "Keep", "Hold"]
objects6 = ["the door", "the window", "your book", "me that", "this here",
            "the message", "some water", "the truth", "a copy", "your hand"]

for a in actions:
    for o in objects6[:3]:
        templates.append(f"{a} {o}")

# === TEMPLATE 7: Time-based Sentences ===
subjects7 = ["I", "We", "They", "She", "He", "The family", "My friends"]
time_verbs = ["arrive", "leave", "start", "finish", "wake up", "go to sleep",
              "eat breakfast", "have dinner"]
times = ["at six", "at seven", "at noon", "in the morning", "in the evening",
         "at night", "on Monday", "every day"]

for s in subjects7:
    for v in time_verbs[:3]:
        for t in times[:3]:
            templates.append(f"{s} {v} {t}")

# === TEMPLATE 8: Possessive Sentences ===
possessors = ["My", "Your", "His", "Her", "Our", "Their", "The child's"]
possessions = ["book", "house", "car", "phone", "computer", "bag", "pen", "key"]
descriptors = ["is on the table", "is very old", "is brand new", "works well",
               "needs fixing", "looks great", "is missing", "is here"]

for p in possessors:
    for pos in possessions[:3]:
        for d in descriptors[:2]:
            templates.append(f"{p} {pos} {d}")

# === TEMPLATE 9: Comparative Sentences ===
subjects9 = ["This", "That", "It", "Everything", "Nothing", "Something"]
comparatives = ["is better than", "is worse than", "is bigger than", "is smaller than",
                "looks like", "seems like", "feels like", "sounds like"]
objects9 = ["before", "the other one", "I thought", "it should be", "yesterday"]

for s in subjects9:
    for c in comparatives[:2]:
        for o in objects9[:2]:
            templates.append(f"{s} {c} {o}")

# === TEMPLATE 10: Common Phrases ===
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
    "Maybe you are right",
    "I agree with you",
    "That makes sense",
    "Tell me more",
    "What do you mean",
    "I understand now",
    "That is correct",
    "You did great",
    "Well done indeed",
    "Keep up the work",
]

templates.extend(phrases)

# === TEMPLATE 11: Numbers and Counting ===
subjects11 = ["There are", "I see", "We have", "They found", "She counted"]
numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
items = ["birds", "books", "chairs", "trees", "cars", "people", "houses", "dogs", "cats"]
locations11 = ["here", "there", "in the room", "outside", "at home", "in the box"]

for s in subjects11:
    for n in numbers[:5]:
        for i in items[:3]:
            templates.append(f"{s} {n} {i}")

# === TEMPLATE 12: Modal Verbs ===
subjects12 = ["I", "You", "We", "They", "She", "He"]
modals = ["can", "could", "should", "would", "may", "might", "must"]
actions12 = ["go now", "do this", "try harder", "wait here", "come back",
             "help them", "finish soon"]

for s in subjects12:
    for m in modals[:3]:
        for a in actions12[:2]:
            templates.append(f"{s} {m} {a}")

# === Finalize Sentence List ===
templates = list(set(templates)) # Remove duplicates
random.seed(42)  # For reproducibility
random.shuffle(templates)

# Select 500 sentences
sentences = templates[:500]

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

output_dir = Path("sentences_500")
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
