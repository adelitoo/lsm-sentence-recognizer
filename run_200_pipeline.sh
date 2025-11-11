#!/bin/bash
# Complete Pipeline for 200-Sentence Dataset
# This script runs the full pipeline after sentence generation completes

set -e  # Exit on error

echo "============================================================"
echo "200-SENTENCE DATASET - COMPLETE PIPELINE"
echo "============================================================"
echo ""
echo "This will run:"
echo "  1. Audio augmentation (200 → 1200 samples)"
echo "  2. Spike train encoding"
echo "  3. LSM feature extraction"
echo "  4. CTC model training"
echo "  5. Evaluation"
echo ""
echo "Estimated time: ~30-40 minutes total"
echo "============================================================"
echo ""

# Check if generation is complete
if [ ! -f "sentences_200/sentences.csv" ]; then
    echo "❌ Error: sentences_200/sentences.csv not found"
    echo "   Please wait for generate_200_sentences.py to complete"
    exit 1
fi

# Count generated files
NUM_FILES=$(ls sentences_200/*.mp3 2>/dev/null | wc -l | tr -d ' ')
if [ "$NUM_FILES" -lt 200 ]; then
    echo "❌ Error: Only $NUM_FILES/200 files generated"
    echo "   Please wait for generation to complete"
    exit 1
fi

echo "✅ Found $NUM_FILES/200 audio files"
echo ""

# STEP 1: Augmentation
echo "============================================================"
echo "STEP 1: Audio Augmentation"
echo "============================================================"
python augment_200_sentences.py
if [ $? -ne 0 ]; then
    echo "❌ Augmentation failed"
    exit 1
fi
echo ""

# STEP 2: Spike Train Encoding
echo "============================================================"
echo "STEP 2: Spike Train Encoding"
echo "============================================================"
python audio_encoding_200.py
if [ $? -ne 0 ]; then
    echo "❌ Encoding failed"
    exit 1
fi
echo ""

# STEP 3: LSM Feature Extraction
echo "============================================================"
echo "STEP 3: LSM Feature Extraction"
echo "============================================================"
python extract_lsm_windowed_features_filtered_sentence_split.py --multiplier 1.0
if [ $? -ne 0 ]; then
    echo "❌ Feature extraction failed"
    exit 1
fi
echo ""

# STEP 4: Model Training
echo "============================================================"
echo "STEP 4: CTC Model Training (5000 epochs)"
echo "============================================================"
python train_ctc.py
if [ $? -ne 0 ]; then
    echo "❌ Training failed"
    exit 1
fi
echo ""

# STEP 5: Results Summary
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Model saved to: ctc_model_sentence_split.pt"
echo ""
echo "Next steps:"
echo "  - Check training output above for final accuracy"
echo "  - Run: python evaluate_model.py (for detailed metrics)"
echo "  - Run: python compare_splits.py (compare with baseline)"
echo ""
echo "============================================================"
