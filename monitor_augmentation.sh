#!/bin/bash
# Monitor audio augmentation progress

echo "🔍 Monitoring audio augmentation progress..."
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    # Check if the process is still running
    if pgrep -f "audio_encoding.py" > /dev/null; then
        # Get the last progress line
        PROGRESS=$(ps aux | grep "audio_encoding.py" | grep -v grep | head -1)

        # Clear the line and show status
        echo -ne "\r⏳ Still processing... (check output for details)                    "

    else
        echo ""
        echo "✅ Process completed!"

        # Check if output file was created
        if [ -f "sentence_spike_trains.npz" ]; then
            FILE_SIZE=$(du -h sentence_spike_trains.npz | cut -f1)
            echo "📦 Output file created: sentence_spike_trains.npz ($FILE_SIZE)"

            # Show quick statistics
            python3 << 'EOF'
import numpy as np
try:
    data = np.load('sentence_spike_trains.npz')
    X = data['X_spikes']
    y = data['y_labels']

    unique_sentences = len(np.unique(y))
    total_samples = len(X)
    augmentation_factor = total_samples / unique_sentences

    print(f"\n📊 Dataset Statistics:")
    print(f"   Total samples: {total_samples:,}")
    print(f"   Unique sentences: {unique_sentences}")
    print(f"   Augmentation factor: {augmentation_factor:.1f}x")
    print(f"   Shape: {X.shape}")
    print(f"   Size: {X.nbytes / (1024**2):.1f} MB")
except Exception as e:
    print(f"Could not load statistics: {e}")
EOF
        else
            echo "⚠️  Output file not found"
        fi

        break
    fi

    sleep 5
done

echo ""
echo "🎯 Next steps:"
echo "   1. Run: python create_balanced_sentence_split.py"
echo "   2. Run: python extract_lsm_traces.py --multiplier 0.8"
echo "   3. Run: python diagnose_lsm_separability.py"
