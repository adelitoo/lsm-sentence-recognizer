# Meaningful Sentence Generation - Summary

## What Changed

### Old Approach (`generate_sentences.py`)
- **Random word combinations**: Grammatically valid but semantically nonsensical
- **Examples**:
  - ❌ "the big dog sleeps on quiz" (quiz is not a location)
  - ❌ "your red pig walks on quiz" (nonsensical)
  - ❌ "my box sleeps with the pen" (objects don't sleep)

### New Approach (`generate_meaningful_sentences.py`)
- **Meaningful templates**: Semantically coherent sentences
- **Examples**:
  - ✅ "the big dog sleeps" (natural)
  - ✅ "my cat in box" (plausible)
  - ✅ "the quick fox runs" (makes sense)
  - ✅ "a bird from jar" (believable scenario)

## Dataset Specifications

- **Total unique sentences**: 1,323
- **Target dataset size**: 1,000 sentences
- **Sentence length**: 3-4 words ONLY
  - 3-word sentences: ~370
  - 4-word sentences: ~958
- **Vocabulary**: Same 35 words (full alphabet coverage a-z)
- **Patterns**: 10 semantic categories

## Sentence Categories

### 1. Simple Animal Actions (3 words)
Pattern: `[article] [animal] [verb]`
- Examples: "the cat runs", "my dog sleeps", "a fox jumps"

### 2. Animal Actions with Adjectives (4 words)
Pattern: `[article] [adjective] [animal] [verb]`
- Examples: "the big cat runs", "my lazy dog sleeps", "a quick fox jumps"

### 3. Object Descriptions (3 words)
Pattern: `[article] [adjective] [object]`
- Examples: "the big box", "my red cup", "a next pen"

### 4. Very + Adjective (4 words)
Pattern: `[article] very [adjective] [noun]`
- Examples: "the very big cat", "a very quick dog"

### 5. Animals in Locations (4 words)
Pattern: `[article] [animal] [preposition] [object]`
- Examples: "the cat in box", "a dog on cup", "my bird with pen"

### 6. Animals from Places (4 words)
Pattern: `[article] [animal] from [location]`
- Examples: "the cat from box", "a dog from jar"

### 7-10. Additional patterns for vocabulary coverage
- Sequential: "the next cat"
- Zero constructions: "zero dog"
- Possessives: "my big cat"
- Object personification: "the box waits"

## Why This Improves Training

### 1. Better Acoustic Patterns
- Natural sentences → natural prosody from TTS
- Consistent rhythm and stress
- More predictable phonetic patterns

### 2. Easier Debugging
- "the cat runs" vs "a dog sleeps" → you can reason about expected differences
- Clear semantic meaning helps identify LSM encoding issues

### 3. More Realistic
- Closer to real speech recognition scenarios
- Better generalization to natural language

### 4. Better TTS Quality
- Text-to-speech models produce better intonation for natural sentences
- Less likely to have odd prosody or unnatural pauses

## How to Use

### Generate new dataset:
```bash
python generate_meaningful_sentences.py
```

This will:
1. Generate 1,323 unique meaningful sentences
2. Select 1,000 for the dataset
3. Create audio files in `sentences/` directory
4. Save metadata to `sentences/sentences.csv`

### Then run the full pipeline:
```bash
# Re-encode audio to spike trains
python audio_encoding.py

# Create balanced split
python create_balanced_sentence_split.py

# Extract LSM traces with improved parameters
python extract_lsm_traces.py --multiplier 8.0 --leak 0.01

# Train the model
python train_ctc_traces_linear.py
```

## Expected Impact

With meaningful sentences AND fixed LSM parameters:
- LSM separability should improve (gap > 0.3)
- Fewer hallucinated words in predictions
- Better word-level accuracy
- More consistent temporal patterns
