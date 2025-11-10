import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import itertools

# --- 1. Define the Character Set (Vocabulary) ---
# The CTCLoss "blank" token is conventionally 0
BLANK_TOKEN = 0
# Create a mapping from integers to characters
# We add 1 because 0 is reserved for BLANK
CHAR_LIST = " " + "abcdefghijklmnopqrstuvwxyz'"
CHAR_MAP = {char: i + 1 for i, char in enumerate(CHAR_LIST)}
INDEX_MAP = {i + 1: char for i, char in enumerate(CHAR_LIST)}
INDEX_MAP[BLANK_TOKEN] = "<b>" # Representation for "blank"


def load_label_map(filepath="sentence_label_map.txt"):
    """Loads the 'sentence_label_map.txt' file into a dictionary."""
    if not Path(filepath).exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        return None
    
    label_map = {}
    with open(filepath, "r") as f:
        next(f) # Skip header
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
    # 1. Collapse repeats: [a, a, a, b, b] -> [a, b]
    # 2. Remove blanks: [a, <b>, <b>, b] -> [a, b]
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
    A more powerful readout model using a GRU layer to better learn
    temporal patterns from the LSM state.
    """
    def __init__(self, input_features, num_classes):
        super().__init__()
        # Increased capacity for high-dimensional features (1211 dims)
        self.gru = nn.GRU(
            input_size=input_features,
            hidden_size=128,   # Increased from 32 to 128 (4x more capacity!)
            num_layers=3,      # Increased from 2 to 3 layers
            batch_first=True,  # Input shape is (batch, seq, feature)
            bidirectional=True, # Look at past and future
            dropout=0.2        # Add dropout for regularization
        )
        # The GRU output will have 2 * hidden_size features because it's bidirectional
        self.linear = nn.Linear(128 * 2, num_classes)

    def forward(self, x):
        # Input x shape: (Batch_Size, Time_Steps, LSM_Features)
        
        # Pass through the GRU
        gru_out, _ = self.gru(x)
        
        # Pass the GRU output through the linear layer
        x = self.linear(gru_out)
        
        # Apply Log Softmax for CTC Loss
        return F.log_softmax(x, dim=2)


# --- 3. The Main Training Function ---
def train():
    print("="*60)
    print("🧠 Starting CTC Readout Layer Training...")
    print("="*60)

    # --- Load Data ---
    print("Loading data...")
    # Try to load features in order of preference: filtered > windowed > traces
    filtered_file = "lsm_windowed_features_filtered.npz"
    feature_file = "lsm_windowed_features.npz"
    trace_file = "lsm_trace_sequences.npz"

    if Path(filtered_file).exists():
        print(f"✅ Loading FILTERED windowed features from '{filtered_file}'")
        dataset = np.load(filtered_file)
    elif Path(feature_file).exists():
        print(f"✅ Loading windowed features from '{feature_file}'")
        dataset = np.load(feature_file)
    elif Path(trace_file).exists():
        print(f"⚠️  Windowed features not found, using traces from '{trace_file}'")
        print(f"   For better results, run: python extract_lsm_windowed_features_filtered.py")
        dataset = np.load(trace_file)
    else:
        print(f"❌ Error: No feature file found.")
        print("Please run feature extraction first.")
        return

    X_train = dataset['X_train_sequences']
    y_train_indices = dataset['y_train']
    X_test = dataset['X_test_sequences']
    y_test_indices = dataset['y_test']
    
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
    print("\nVisualizing the first training sample...")
    plt.figure(figsize=(15, 5))
    plt.imshow(X_train[0].T, aspect='auto', cmap='viridis')
    plt.title(f"LSM Trace for Sample 0: '{y_train_text[0]}'")
    plt.xlabel("Time Steps")
    plt.ylabel("LSM Output Neurons")
    plt.colorbar(label="Trace Value")
    plt.savefig("lsm_trace_visualization.png")
    print("✅ Visualization saved to lsm_trace_visualization.png")

    # --- Normalize Features (important for high-dimensional inputs!) ---
    print("\nNormalizing features...")
    # Flatten for normalization
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])

    # Compute mean and std from training set
    feature_mean = X_train_flat.mean(axis=0)
    feature_std = X_train_flat.std(axis=0) + 1e-8  # Add epsilon to avoid division by zero

    # Normalize
    X_train_normalized = (X_train - feature_mean) / feature_std
    X_test_normalized = (X_test - feature_mean) / feature_std

    print(f"  Original range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    print(f"  Normalized range: [{X_train_normalized.min():.2f}, {X_train_normalized.max():.2f}]")

    # --- Convert to PyTorch Tensors ---
    X_train_tensor = torch.FloatTensor(X_train_normalized)
    X_test_tensor = torch.FloatTensor(X_test_normalized)
    
    # --- Parameters for Model & CTC ---
    # From your lsm_sequences.npz output
    num_samples, num_timesteps, num_lsm_neurons = X_train_tensor.shape
    num_classes = len(CHAR_MAP) + 1 # +1 for the BLANK token
    
    print("\n--- Model Configuration ---")
    print(f"LSM Output Features: {num_lsm_neurons}")
    print(f"Time Steps: {num_timesteps}")
    print(f"Num Classes (w/ blank): {num_classes}")
    
    # --- Initialize Model, Loss, and Optimizer ---
    model = CTCReadout(input_features=num_lsm_neurons, num_classes=num_classes)

    # CTCLoss(blank=0) tells the loss function that index 0 is the blank token
    # reduction='mean' averages the loss over the batch
    loss_fn = nn.CTCLoss(blank=BLANK_TOKEN, reduction='mean', zero_infinity=True)

    # Use a learning rate scheduler for better convergence
    # Lower initial LR for larger model with more parameters
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )

    # --- Prepare CTC Targets ---
    # CTCLoss needs the labels in a specific format
    
    # 1. Encode all target text into integer sequences
    y_train_encoded = [encode_text(text) for text in y_train_text]
    
    # 2. Concatenate all sequences into one long tensor
    # e.g., [2,5,3] and [1,4] becomes [2,5,3,1,4]
    y_train_targets = torch.cat(y_train_encoded)
    
    # 3. Create a tensor of the *length* of each text label
    # e.g., [3, 2]
    y_train_target_lengths = torch.LongTensor([len(seq) for seq in y_train_encoded])
    
    # 4. Create a tensor of the *length* of each input sequence
    # For us, they are all the same length (400)
    X_train_input_lengths = torch.LongTensor([num_timesteps] * num_samples)

    # --- The Training Loop ---
    num_epochs = 5000
    print(f"Starting training for {num_epochs} epochs...")

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        optimizer.zero_grad() # Reset gradients

        # --- Forward Pass ---
        # Get log probabilities from the model
        # Shape: (Batch_Size, Time_Steps, Num_Classes)
        log_probs = model(X_train_tensor)

        # --- Prepare for CTCLoss ---
        # CTCLoss expects (Time_Steps, Batch_Size, Num_Classes)
        # So we must permute (swap) the first two dimensions
        log_probs_for_loss = log_probs.permute(1, 0, 2)

        # --- Calculate Loss ---
        loss = loss_fn(
            log_probs_for_loss,  # Model output
            y_train_targets,       # Concatenated text labels
            X_train_input_lengths, # Length of each LSM sequence (all 400)
            y_train_target_lengths # Length of each text label
        )

        # --- Backward Pass ---
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update learning rate based on loss
        scheduler.step(loss)

        # Track best loss
        if loss.item() < best_loss:
            best_loss = loss.item()

        # --- Print Progress ---
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}, LR: {current_lr:.6f}")

            # --- Check a prediction (Greedy Decode) ---
            model.eval() # Set model to evaluation mode
            with torch.no_grad():
                # Get prediction for the first test sample
                test_sample_log_probs = model(X_test_tensor[0].unsqueeze(0))

                # Squeeze to (Time_Steps, Num_Classes)
                test_sample_log_probs = test_sample_log_probs.squeeze(0)

                decoded_text = greedy_decoder(test_sample_log_probs)

                print(f"  Test Sample 0 Target: '{y_test_text[0]}'")
                print(f"  Test Sample 0 Decoded: '{decoded_text}'\n")

    print("✅ Training complete.")

if __name__ == "__main__":
    train()