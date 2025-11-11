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
from torch.utils.data import Dataset, DataLoader # <-- ADDED

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


# --- 1.5. Define Custom Dataset & Collate Function ---
class TraceDataset(Dataset):
    """Custom Dataset to handle (trace, encoded_text) pairs."""
    def __init__(self, x_data, y_encoded_list):
        self.x_data = x_data
        self.y_data = y_encoded_list
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        # Data is still on CPU, which is fine for __getitem__
        return self.x_data[idx], self.y_data[idx]

def ctc_collate_fn(batch):
    """
    Custom collate function to batch variable-length CTC data.
    batch is a list of tuples: [(x1, y1), (x2, y2), ...]
    """
    # --- Process X (Features) ---
    # Stack all x samples into a single batch tensor
    x_batch = torch.stack([item[0] for item in batch])
    
    # --- Process Y (Targets) ---
    y_batch_list = [item[1] for item in batch]
    # Concatenate all target sequences into one long tensor
    y_targets_concat = torch.cat(y_batch_list)
    # Create a tensor of the length of each target sequence
    y_target_lengths = torch.LongTensor([len(seq) for seq in y_batch_list])
    
    # --- Get Input Lengths ---
    # All our inputs have the same number of timesteps
    num_timesteps = x_batch.shape[1]
    input_lengths = torch.LongTensor([num_timesteps] * len(batch))
    
    return x_batch, y_targets_concat, input_lengths, y_target_lengths


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
    print(f"    🎯 Testing TRUE GENERALIZATION to unseen sentences!")
    dataset = np.load(trace_file, allow_pickle=True)

    X_train = dataset["X_train_sequences"]
    y_train = dataset["y_train"]
    X_test = dataset["X_test_sequences"]
    y_test = dataset["y_test"]

    label_map = load_label_map()
    if label_map is None:
        return

    # Get the text for each label index
    y_train_text = [label_map[idx] for idx in y_train]
    y_test_text = [label_map[idx] for idx in y_test]

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Sample train label 0: {y_train[0]} -> '{y_train_text[0]}'")

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

    # --- Setup Device (GPU/CPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # --- Convert to PyTorch Tensors (Keep on CPU for DataLoader) ---
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
    # Move model to GPU
    model = CTCReadout(input_features=num_membrane_channels, num_classes=num_classes).to(device)

    loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, reduction="mean", zero_infinity=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-7
    )

    # --- Prepare DataLoader (Replaces old CTC target prep) ---
    BATCH_SIZE = 16  # You can tune this (e.g., 8, 16, 32)
    y_train_encoded = [encode_text(text) for text in y_train_text]

    train_dataset = TraceDataset(X_train_tensor, y_train_encoded)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=2,  # Speeds up data loading
        pin_memory=True # Helps speed up CPU to GPU transfer
    )


    # --- The Training Loop (MODIFIED FOR BATCHING) ---
    num_epochs = 5000
    print(f"\nStarting training for {num_epochs} epochs... (Batch Size: {BATCH_SIZE})")

    best_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0

        # --- Batch Loop ---
        for X_batch, y_targets, X_input_lengths, y_target_lengths in train_loader:
            
            # Move current batch to GPU
            X_batch = X_batch.to(device)
            y_targets = y_targets.to(device)
            X_input_lengths = X_input_lengths.to(device)
            y_target_lengths = y_target_lengths.to(device)

            optimizer.zero_grad()  # Reset gradients for this batch

            # --- Forward Pass (on the batch) ---
            log_probs = model(X_batch)

            # --- Prepare for CTCLoss ---
            # CTCLoss expects (Time_Steps, Batch_Size, Num_Classes)
            log_probs_for_loss = log_probs.permute(1, 0, 2)

            # --- Calculate Loss ---
            loss = loss_fn(
                log_probs_for_loss,
                y_targets,
                X_input_lengths,
                y_target_lengths,
            )

            # --- Backward Pass ---
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # --- End of Epoch ---
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        # Update learning rate based on average epoch loss
        scheduler.step(avg_epoch_loss)

        # Track best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss

        # --- Print Progress ---
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Best: {best_loss:.4f}, LR: {current_lr:.6f}"
            )

            # --- Check a prediction (Greedy Decode) ---
            model.eval()  # Set model to evaluation mode
            with torch.no_grad():
                # Get prediction for the first test sample
                # Move ONE sample to the GPU for this test
                test_sample = X_test_tensor[0].unsqueeze(0).to(device)
                test_sample_log_probs = model(test_sample)

                # Squeeze to (Time_Steps, Num_Classes)
                test_sample_log_probs = test_sample_log_probs.squeeze(0)

                # Greedy decoder works on CPU
                decoded_text = greedy_decoder(test_sample_log_probs.cpu())

                print(f"  Test Sample 0 Target: '{y_test_text[0]}'")
                print(f"  Test Sample 0 Decoded: '{decoded_text}'\n")

    print("✅ Training complete.")

    # Save the trained model
    model_path = "ctc_model_traces.pt"
    # Save model state dict (move to CPU first)
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
            # Move ONE sample at a time to the GPU
            test_sample = X_test_tensor[i].unsqueeze(0).to(device)
            test_sample_log_probs = model(test_sample)
            
            test_sample_log_probs = test_sample_log_probs.squeeze(0)
            
            # Decode on CPU
            decoded_text = greedy_decoder(test_sample_log_probs.cpu())

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