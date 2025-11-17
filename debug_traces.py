# Save this as debug_traces.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_trace_voltages(npz_file="lsm_trace_sequences.npz", num_channels=10, sample_idx=0):
    """
    Loads the saved trace file and plots the voltage traces
    for a few channels to check their dynamics.
    """
    print(f"--- 📈 Plotting Traces from '{npz_file}' ---")
    
    if not Path(npz_file).exists():
        print(f"❌ Error: File not found. Run extraction first.")
        return

    try:
        data = np.load(npz_file)
        traces = data['X_train_sequences']
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    print(f"  Loaded traces with shape: {traces.shape}")
    
    # Get the first sample
    sample_traces = traces[sample_idx] # Shape: (timesteps, channels)
    
    # Select channels to plot
    num_total_channels = sample_traces.shape[1]
    channels_to_plot = np.linspace(0, num_total_channels - 1, min(num_channels, num_total_channels), dtype=int)
    
    plt.figure(figsize=(15, 8))
    
    for i, chan_idx in enumerate(channels_to_plot):
        # We add 'i' to each trace to stack them vertically
        plt.plot(sample_traces[:, chan_idx] + i, label=f"Channel {chan_idx}")
        
    plt.title(f"Membrane Potential Traces (Sample {sample_idx}, {len(channels_to_plot)} channels)")
    plt.xlabel("Time Step")
    plt.ylabel("Neuron Channel (stacked)")
    plt.yticks([]) # Disable y-ticks as they are just for separation
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    plot_filename = "lsm_trace_voltages.png"
    plt.savefig(plot_filename)
    print(f"  ✅ Voltage plot saved to '{plot_filename}'")
    plt.close()

if __name__ == "__main__":
    plot_trace_voltages()