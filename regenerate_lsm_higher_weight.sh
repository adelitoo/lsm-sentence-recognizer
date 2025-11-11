#!/bin/bash
# Regenerate LSM traces with higher weight multipliers to increase discriminability

echo "============================================"
echo "Regenerating LSM traces with higher weights"
echo "============================================"

# Try multiplier 1.0 (at critical weight)
echo ""
echo "Testing multiplier 1.0 (critical weight)..."
python extract_lsm_traces_sentence_split_500.py --multiplier 1.0

# Check discriminability
echo ""
echo "Checking discriminability..."
python check_lsm_discriminability.py

# Rename to avoid overwriting
mv lsm_trace_sequences_sentence_split_500.npz lsm_trace_sequences_sentence_split_500_m1.0.npz
mv lsm_discriminability_check.png lsm_discriminability_check_m1.0.png

# Try multiplier 1.2 (supercritical - more chaotic)
echo ""
echo "Testing multiplier 1.2 (supercritical)..."
python extract_lsm_traces_sentence_split_500.py --multiplier 1.2

# Check discriminability
echo ""
echo "Checking discriminability..."
python check_lsm_discriminability.py

# Rename
mv lsm_trace_sequences_sentence_split_500.npz lsm_trace_sequences_sentence_split_500_m1.2.npz
mv lsm_discriminability_check.png lsm_discriminability_check_m1.2.png

echo ""
echo "✅ Done! Compare discriminability results:"
echo "   - Original (m=0.8): Already tested"
echo "   - m=1.0: Check lsm_discriminability_check_m1.0.png"
echo "   - m=1.2: Check lsm_discriminability_check_m1.2.png"
echo ""
echo "Look for cosine similarity < 0.90 for better discriminability"
