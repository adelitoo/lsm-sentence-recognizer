from elevenlabs.client import ElevenLabs
import csv  # <-- 1. Import the CSV library
from pathlib import Path # <-- Good practice for handling paths

client = ElevenLabs(api_key="sk_a3614def3d14243a3bebfbb352bd93fdbc30d120d0f9571e") 

sentences = [
    # Original 10 sentences (keep for comparison)
    "The cat sleeps on the mat",
    "The dog runs in the park",
    "The bird sings on the tree",
    "The sun shines over the hill",
    "The rain falls on the ground",
    "The child plays with a ball",
    "The teacher writes on the board",
    "The car stops at the light",
    "The train moves along the track",
    "The clock ticks on the wall",

    # Animals (varied structures)
    "A horse gallops across the field",
    "The elephant trumpets loudly",
    "Fish swim in the deep ocean",
    "The owl hoots at night",
    "Bees buzz around the flowers",
    "A squirrel climbs the tall tree",
    "The rabbit hops through the grass",
    "Dolphins jump over the waves",
    "A tiger roars in the jungle",
    "The bear catches a salmon",

    # People and actions (different subjects)
    "She reads a book by the window",
    "He walks home from work",
    "They dance together at the party",
    "We cook dinner every evening",
    "I write letters to my friends",
    "The man fixes the broken chair",
    "The woman paints a beautiful picture",
    "Children laugh and play outside",
    "The boy rides his new bicycle",
    "The girl sings a happy song",

    # Weather and nature (varied vocabulary)
    "Snow falls gently on the ground",
    "The wind blows through the trees",
    "Thunder rumbles in the distance",
    "Lightning flashes across the sky",
    "Clouds drift slowly overhead",
    "The moon shines bright tonight",
    "Stars twinkle in the dark sky",
    "Waves crash against the rocks",
    "The river flows toward the sea",
    "Leaves rustle in the breeze",

    # Daily activities (different patterns)
    "I drink coffee in the morning",
    "She eats breakfast at seven",
    "He brushes his teeth carefully",
    "We watch movies on weekends",
    "They play cards every night",
    "The baby sleeps in the crib",
    "The phone rings three times",
    "Water boils in the kettle",
    "The door opens and closes",
    "Music plays softly in the room",

    # Questions and varied sentence types
    "Where did you go today",
    "What time is it now",
    "How are you feeling",
    "Can you help me please",
    "Do you like chocolate cake",
    "Is this your first visit",
    "Will it rain tomorrow",
    "Should we leave now",
    "May I ask a question",
    "Did you see that bird",

    # Short sentences (different from complex ones)
    "Hello there",
    "Good morning",
    "Thank you very much",
    "See you soon",
    "Have a nice day",
    "Please come in",
    "Wait a moment",
    "Look over here",
    "Listen carefully now",
    "Think about it",

    # Longer complex sentences
    "The old man walks slowly down the street",
    "My sister baked chocolate cookies yesterday afternoon",
    "The students study hard for their final exams",
    "A red car drives quickly past the house",
    "The young girl reads her favorite story again",
    "We enjoy spending time together as a family",
    "The flowers bloom beautifully in the spring garden",
    "He builds a wooden table in his workshop",
    "They travel to different countries every summer",
    "The doctor examines the patient very carefully",

    # Numbers and time (adds numeric diversity)
    "I wake up at six in the morning",
    "There are seven days in a week",
    "She counts to ten very slowly",
    "The store opens at nine o'clock",
    "We have three cats and two dogs",

    # Common phrases and expressions
    "Nice to meet you today",
    "How do you do this",
    "Let me think about that",
    "I understand what you mean",
    "That sounds like a plan",
    "Everything will be alright soon",
    "Take your time with it",
    "No problem at all",
    "You are very welcome",
    "Have fun and be safe",

    # Varied grammatical structures
    "Running is good for health",
    "To learn is to grow",
    "Happiness comes from within",
    "Knowledge brings great power",
    "Time heals all wounds slowly",
]

voices = [
    "JBFqnCBsd6RMkjVDRZzb",    
]

output_dir = Path("sentences")
output_dir.mkdir(exist_ok=True) # Create the directory if it doesn't exist
metadata_file = output_dir / "sentences.csv"

# 2. Open the CSV file *before* the loop
with open(metadata_file, "w", newline="", encoding="utf-8") as f:
    # 3. Create a writer and add the header row
    writer = csv.writer(f)
    writer.writerow(["filename", "label_text"])

    print("Generating audio and metadata...")
    for i, text in enumerate(sentences):
        voice_id = voices[i % len(voices)]
        
        audio = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )

        filename = f"sentence_{i + 1}.mp3"
        filepath = output_dir / filename
        
        # Save the audio file
        with open(filepath, "wb") as audio_f:
            for chunk in audio:
                audio_f.write(chunk)
        
        # 4. Write the filename and text to your CSV "answer key"
        writer.writerow([filename, text])

print(f"✅ All {len(sentences)} audio sentences generated successfully!")
print(f"✅ Metadata saved to '{metadata_file}'")