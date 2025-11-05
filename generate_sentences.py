from elevenlabs.client import ElevenLabs
import csv  # <-- 1. Import the CSV library
from pathlib import Path # <-- Good practice for handling paths

client = ElevenLabs(api_key="sk_a3614def3d14243a3bebfbb352bd93fdbc30d120d0f9571e") # REMINDER: Be careful sharing API keys!

sentences = [
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
]

voices = [
    "JBFqnCBsd6RMkjVDRZzb",  
    "TxGEqnHWrfWFTfGW9XjX",  
    "pNInz6obpgDQGcFmaJgB",  
    "EXAVITQu4vr4xnSDxMaL",  
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