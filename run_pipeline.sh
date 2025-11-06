#!/bin/bash

set -e

PREPARED_DATASET_FILE="sentences/sentences.csv"

if [ ! -f "$PREPARED_DATASET_FILE" ]; then
    echo "Dataset file not found at '$PREPARED_DATASET_FILE'."
    echo "Running dataset preparation script..."
    echo "========================================================"
    
    python prepare_librispeech.py
    
    echo "========================================================"
    echo "Dataset preparation complete."
else
    echo "Dataset already prepared. Skipping preparation step."
fi

echo ""
echo "Running audio encoding..."
python audio_encoding.py

echo ""
echo "Extracting LSM sequences..."
python extract_lsm_sequences.py

echo ""
echo "Training CTC model..."
python train_ctc.py

echo ""
echo "✅ Pipeline finished."
