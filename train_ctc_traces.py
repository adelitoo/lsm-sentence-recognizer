"""
CTC Training on Membrane Potential Traces (500 Sentences)

This version trains on CONTINUOUS MEMBRANE VOLTAGES instead of windowed spike features.

Key differences from regular CTC training:
- Loads lsm_trace_sequences_sentence_split_500.npz (membrane potentials)
- Input: (samples, 2000 timesteps, 700 neurons) continuous voltages
- No windowing - full temporal resolution
- Tests true generalization to unseen sentences
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import itertools

# --- 1. Define the Character Set (Vocabulary) ---
BLANK_TOKEN = 0
CHAR_LIST = " " + "abcdefghijklmnopqrstuvwxyz'"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHAR_LIST)}
INDEX_MAP = {i + 1: char for i, char in enumerate(CHAR_LIST)}
INDEX_MAP[BLANK_TOKEN] = "<b>"  # Representation for "blank"


def load_label_map(filepath="sentence_label_map_500.txt"):
    """Loads the 'sentence_label_map.txt' file into a dictionary."""
    if not Path(filepath).exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        return None

    label_map = {}
    with open(filepath, "r") as f:
        next(f)  # Skip header
        for line in f:
            idx, text = line.strip().split(",", 1)
            label_map[int(idx)] = text.lower()
    print(f"✅ Loaded label map with {len(label_map)} entries.")
    return label_map


def encode_text(text: str) -> torch.LongTensor:
    """Encodes a text string into a tensor of integer labels."""
    encoded = [CHAR_MAP[char] for char in text if char in CHAR_MAP]
    return torch.LongTensor(encoded)


def greedy_decoder(log_probs: torch.Tensor) -> str:
    """Decodes the output of the CTC model into text."""
    # Find the most likely character index at each time step
    # Shape of log_probs: (Time, Num_Classes)
    indices = torch.argmax(log_probs, dim=1)

    # Use itertools.groupby to collapse repeats and remove blanks
    deduped = [key for key, _ in itertools.groupby(indices)]

    decoded_text = ""
    for idx in deduped:
        i = idx.item()
        if i != BLANK_TOKEN:
            decoded_text += INDEX_MAP[i]

    return decoded_text.strip()


# --- 2. Define the PyTorch Readout Model ---
class CTCReadout(nn.Module):
    """
    A powerful readout model using a GRU layer to learn
    temporal patterns from membrane potential traces.
    """

    def __init__(self, input_features, num_classes):
        super().__init__()
        # Increased capacity for 700 membrane potential channels
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        # The GRU output will have 2 * hidden_size features because it's bidirectional
        self.linear = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        # Input x shape: (Batch_Size, Time_Steps, Membrane_Channels)

        # Pass through the GRU
        gru_out, _ = self.gru(x)

        # Pass the GRU output through the linear layer
        x = self.linear(gru_out)

        # Apply Log Softmax for CTC Loss
        return F.log_softmax(x, dim=2)


# --- 3. The Main Training Function ---
def train():
    print("=" * 60)
    print("🧠 Starting CTC Training on MEMBRANE POTENTIAL TRACES...")
    print("=" * 60)

    # --- Load Trace Data ---
    print("\nLoading trace data...")
    trace_file = "lsm_trace_sequences_sentence_split_500.npz"

    if not Path(trace_file).exists():
        print(f"❌ Error: Trace file not found at '{trace_file}'")
        print("Please run: python extract_lsm_traces_sentence_split_500.py first")
        return

    print(f"✅ Loading MEMBRANE POTENTIAL TRACES from '{trace_file}'")
    print(f"   🎯 Testing TRUE GENERALIZATION to unseen sentences!")
    dataset = np.load(trace_file, allow_pickle=True)    

    X_train = dataset["X_train_sequences"]
    y_train_indices = dataset["y_train"]
    X_test = dataset["X_test_sequences"]
    y_test_indices = dataset["y_test"]

    label_map = load_label_map()
    if label_map is None:
        return

    # Get the text for each label index
    y_train_text = [label_map[idx] for idx in y_train_indices]
    y_test_text = [label_map[idx] for idx in y_test_indices]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sample train label 0: {y_train_indices[0]} -> '{y_train_text[0]}'")

    # --- Visualize the first training sample ---
    print("\nVisualizing the first training sample (membrane potentials)...")
    plt.figure(figsize=(15, 5))
    plt.imshow(X_train[0].T, aspect="auto", cmap="viridis")
    plt.title(f"Membrane Potential Traces for Sample 0: '{y_train_text[0]}'")
    plt.xlabel("Time Steps")
    plt.ylabel("LSM Output Neurons")
    plt.colorbar(label="Membrane Voltage")
    plt.savefig("lsm_trace_visualization.png")
    print("✅ Visualization saved to lsm_trace_visualization.png")

    # --- Normalize Features ---
    print("\nNormalizing membrane potential traces...")
    # Flatten for normalization
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    # Compute mean and std from training set
    feature_mean = X_train_flat.mean(axis=0)
    feature_std = (
        X_train_flat.std(axis=0) + 1e-8
    )  # Add epsilon to avoid division by zero

    # Reshape for proper broadcasting with 3D arrays (batch, time, features)
    feature_mean = feature_mean.reshape(1, 1, -1)
    feature_std = feature_std.reshape(1, 1, -1)

    # Normalize
    X_train_normalized = (X_train - feature_mean) / feature_std
    X_test_normalized = (X_test - feature_mean) / feature_std

    print(f"  Original range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(
        f"  Normalized range: [{X_train_normalized.min():.3f}, {X_train_normalized.max():.3f}]"
    )

    # --- Convert to PyTorch Tensors ---
    X_train_tensor = torch.FloatTensor(X_train_normalized)
    X_test_tensor = torch.FloatTensor(X_test_normalized)

    # --- Parameters for Model & CTC ---
    num_samples, num_timesteps, num_membrane_channels = X_train_tensor.shape
    num_classes = len(CHAR_MAP) + 1  # +1 for the BLANK token

    print("\n--- Model Configuration ---")
    print(f"Membrane Potential Channels: {num_membrane_channels}")
    print(f"Time Steps: {num_timesteps}")
    print(f"Num Classes (w/ blank): {num_classes}")
    print(f"Data type: Continuous membrane voltages (not spike features)")

    # --- Initialize Model, Loss, and Optimizer ---
    model = CTCReadout(input_features=num_membrane_channels, num_classes=num_classes)

    loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, reduction="mean", zero_infinity=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-7
    )

    # --- Prepare CTC Targets ---
    y_train_encoded = [encode_text(text) for text in y_train_text]
    y_train_targets = torch.cat(y_train_encoded)
    y_train_target_lengths = torch.LongTensor([len(seq) for seq in y_train_encoded])
    X_train_input_lengths = torch.LongTensor([num_timesteps] * num_samples)

    # --- The Training Loop ---
    num_epochs = 5000
    print(f"\nStarting training for {num_epochs} epochs...")

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        optimizer.zero_grad()  # Reset gradients

        # --- Forward Pass ---
        log_probs = model(X_train_tensor)

        # --- Prepare for CTCLoss ---
        # CTCLoss expects (Time_Steps, Batch_Size, Num_Classes)
        log_probs_for_loss = log_probs.permute(1, 0, 2)

        # --- Calculate Loss ---
        loss = loss_fn(
            log_probs_for_loss,
            y_train_targets,
            X_train_input_lengths,
            y_train_target_lengths,
        )

        # --- Backward Pass ---
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update learning rate based on loss
        scheduler.step(loss)

        # Track best loss
        if loss.item() < best_loss:
            best_loss = loss.item()

        # --- Print Progress ---
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}, LR: {current_lr:.6f}"
            )

            # --- Check a prediction (Greedy Decode) ---
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # Get prediction for the first test sample
                test_sample_log_probs = model(X_test_tensor[0].unsqueeze(0))

                # Squeeze to (Time_Steps, Num_Classes)
                test_sample_log_probs = test_sample_log_probs.squeeze(0)

                decoded_text = greedy_decoder(test_sample_log_probs)

                print(f"  Test Sample 0 Target: '{y_test_text[0]}'")
                print(f"  Test Sample 0 Decoded: '{decoded_text}'\n")

    print("✅ Training complete.")

    # Save the trained model
    model_path = "ctc_model_traces.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to '{model_path}'")

    # Evaluate on all test samples
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON ALL TEST SAMPLES")
    print("=" * 60)

    model.eval()
    correct = 0
    total = len(X_test_tensor)

    with torch.no_grad():
        for i in range(total):
            test_sample_log_probs = model(X_test_tensor[i].unsqueeze(0))
            test_sample_log_probs = test_sample_log_probs.squeeze(0)
            decoded_text = greedy_decoder(test_sample_log_probs)

            if decoded_text == y_test_text[i]:
                correct += 1

            # Show first 5 and last 5 examples
            if i < 5 or i >= total - 5:
                match_symbol = "✅" if decoded_text == y_test_text[i] else "❌"
                print(f"{match_symbol} Sample {i}:")
                print(f"  Target:     '{y_test_text[i]}'")
                print(f"  Prediction: '{decoded_text}'")

    accuracy = (correct / total) * 100
    print(f"\n{'=' * 60}")
    print(f"Test Set Accuracy: {correct}/{total} = {accuracy:.2f}%")
    print(f"Split Type: SENTENCE-LEVEL (membrane potential traces)")
    print(f"🎯 This tests GENERALIZATION to completely unseen sentences!")
    print(f"Data type: Continuous membrane voltages (not windowed spike features)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train()
