#!/bin/bash

# Script to Generate Rich Dataset for LSM Sentence Recognition
# This creates 100 unique sentences → 600 total samples (100 orig + 500 augmented)

set -e

echo "============================================================"
echo "🎯 Generating Rich Dataset for LSM Sentence Recognition"
echo "============================================================"
echo ""
echo "Dataset Overview:"
echo "  - 100 unique sentences (10x original)"
echo "  - Multiple categories:"
echo "    • Animals (10 sentences)"
echo "    • People & actions (10 sentences)"
echo "    • Weather & nature (10 sentences)"
echo "    • Daily activities (10 sentences)"
echo "    • Questions (10 sentences)"
echo "    • Short phrases (10 sentences)"
echo "    • Long complex sentences (10 sentences)"
echo "    • Numbers & time (5 sentences)"
echo "    • Common expressions (10 sentences)"
echo "    • Varied grammar (5 sentences)"
echo "    • Original sentences (10 sentences)"
echo ""
echo "  - 5 augmentations per sentence"
echo "  - Total: 600 samples (100 × 6)"
echo "============================================================"
echo ""

# Step 1: Generate audio for 100 sentences
echo "📢 Step 1/5: Generating audio for 100 unique sentences..."
echo "   (This will take ~3-5 minutes with ElevenLabs API)"
python generate_sentences.py

echo ""
echo "✅ Generated 100 audio files!"
echo ""

# Step 2: Augment to 600 samples
echo "📢 Step 2/5: Augmenting dataset (100 → 600 samples)..."
python augment_audio_dataset.py

echo ""
echo "✅ Created 600 total samples!"
echo ""

# Step 3: Encode audio to spikes
echo "📢 Step 3/5: Encoding audio to spike trains..."
# Update audio_encoding.py to use augmented directory
sed -i '' 's|Path("sentences")|Path("sentences_augmented")|g' audio_encoding.py
python audio_encoding.py --n-filters 128 --filterbank gammatone

echo ""
echo "✅ Encoded 600 samples to spike trains!"
echo ""

# Step 4: Extract filtered LSM features
echo "📢 Step 4/5: Extracting filtered LSM features..."
python extract_lsm_windowed_features_filtered.py --multiplier 0.8

echo ""
echo "✅ Extracted features!"
echo ""

# Step 5: Analyze quality
echo "📢 Step 5/5: Analyzing feature quality..."
python analyze_lsm_features.py

echo ""
echo "============================================================"
echo "✅ RICH DATASET GENERATION COMPLETE!"
echo "============================================================"
echo ""
echo "Dataset Statistics:"
echo "  - Unique sentences: 100"
echo "  - Total samples: 600"
echo "  - Training samples: ~480 (80%)"
echo "  - Test samples: ~120 (20%)"
echo "  - Feature dimensions: ~700-800 (after filtering)"
echo ""
echo "Next step: Train the CTC model"
echo "  Run: python train_ctc.py"
echo ""
echo "Expected improvements:"
echo "  - Much better sentence discrimination"
echo "  - Higher character accuracy"
echo "  - More robust to variations"
echo "  - Proper generalization to test set"
echo "============================================================"
