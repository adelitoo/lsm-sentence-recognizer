import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
import itertools
from torch.utils.data import Dataset, DataLoader
import warnings
import argparse
import os

# --- TOKENIZER IMPORTS ---
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ==========================================
# === CONFIGURATION ===
# ==========================================
VOCAB_SIZE = 100   
CTC_BLANK_TOKEN = 0
STRIDE = 10  # CRITICAL: Compresses 2000 steps -> 200 steps

# ==========================================
# === 1. THE "DUMBEST" MODEL (Purist Readout) ===
# ==========================================
class PuristLSMReadout(nn.Module):
    """
    The Scientific Control Model.
    It has NO memory, NO learned temporal filters, and NO hidden layers.
    It simply averages the liquid state over 10ms and applies a linear classifier.
    
    If this works, the LSM is provably working.
    """
    def __init__(self, input_features, num_classes):
        super().__init__()
        
        # 1. FIXED Downsampling (Mathematical, not learned)
        # Reduces 2000 steps to 200 steps so CTC can function.
        self.pool = nn.AvgPool1d(kernel_size=STRIDE, stride=STRIDE)
        
        # 2. Direct Linear Projection
        # Maps 700 neurons -> 100 classes directly.
        self.fc = nn.Linear(input_features, num_classes)

    def forward(self, x):
        # Input x shape: (Batch, Time=2000, Features=700)
        
        # Permute for Pooling: PyTorch expects (Batch, Channels, Time)
        x = x.permute(0, 2, 1) 
        
        # Apply Fixed Averaging
        x = self.pool(x)
        
        # Permute back: (Batch, Time_Reduced=200, Features=700)
        x = x.permute(0, 2, 1)
        
        # Linear Classification
        x = self.fc(x)
        
        # Log Softmax for CTC
        return F.log_softmax(x, dim=2)

# ==========================================
# === HELPER FUNCTIONS ===
# ==========================================

def load_label_map(filepath="sentence_label_map.txt"):
    if not Path(filepath).exists():
        print(f"❌ Error: Label map not found at '{filepath}'")
        return None
    label_map = {}
    try:
        with open(filepath, "r") as f:
            next(f) # skip header
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    label_map[int(parts[0])] = parts[1].lower()
    except:
        pass 
    return label_map

def train_custom_tokenizer(label_map_dict, vocab_size=100):
    print(f"\nConstructing Tokenizer (Vocab: {vocab_size})...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["[UNK]"], vocab_size=vocab_size, min_frequency=1, show_progress=False)
    corpus = list(label_map_dict.values())
    tokenizer.train_from_iterator(corpus, trainer)
    return tokenizer

def encode_text_tokens(tokenizer, text: str):
    # IDs + 1 because 0 is reserved for CTC Blank
    return torch.LongTensor([i + 1 for i in tokenizer.encode(text).ids])

def decode_tokens(tokenizer, log_probs: torch.Tensor) -> str:
    """
    Advanced CTC Decoder with 'Echo Cancellation'.
    """
    if log_probs.dim() == 3:
        log_probs = log_probs.squeeze(0) 

    # 1. Get raw indices
    indices = torch.argmax(log_probs, dim=1).tolist()
    
    # 2. Filter out "blips" (tokens < 3 frames duration)
    # This removes the random 'runs' or 'jumps' that appear for 10ms inside a stable word
    stable_indices = []
    current_token = indices[0]
    count = 1
    
    for i in range(1, len(indices)):
        if indices[i] == current_token:
            count += 1
        else:
            # KEEP if: It's a blank, OR it lasted >= 3 frames (30ms)
            if current_token == 0 or count >= 3:
                stable_indices.extend([current_token] * count)
            current_token = indices[i]
            count = 1
    # Flush last token
    if current_token == 0 or count >= 3:
        stable_indices.extend([current_token] * count)

    # 3. Standard CTC Collapse (Merge adjacent duplicates)
    collapsed_ids = []
    prev_idx = -1
    for idx in stable_indices:
        if idx != prev_idx:
            if idx != 0: # Skip blanks
                collapsed_ids.append(idx - 1)
        prev_idx = idx
        
    # 4. Loop Removal (The "Jar Runs Jar" Fix)
    # If we see [A, B, A] and B is very short/common error, we might want to collapse.
    # For now, let's trust the "Blip Filter" (Step 2) to fix this.
    
    return tokenizer.decode(collapsed_ids).strip()
            
def calculate_edit_distance(predicted, target):
    """
    Returns (edit_distance, target_length)
    """
    r = target.split()
    h = predicted.split()
    d = np.zeros((len(r)+1, len(h)+1), dtype=np.uint8)
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]: d[i][j] = d[i-1][j-1]
            else: d[i][j] = min(d[i-1][j], d[i][j-1], d[i-1][j-1]) + 1
            
    return d[len(r)][len(h)], len(r)

# --- DATASET & COLLATE ---
class TraceDataset(Dataset):
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

def ctc_collate_fn(batch):
    x = torch.stack([b[0] for b in batch])
    y = torch.cat([b[1] for b in batch])
    
    # --- CRITICAL: MATCH THE STRIDE OF THE READOUT ---
    output_seq_len = x.shape[1] // STRIDE
    lx = torch.LongTensor([output_seq_len] * len(batch))
    
    ly = torch.LongTensor([len(b[1]) for b in batch])
    return x, y, lx, ly

# ==========================================
# === MAIN TRAINING LOOP ===
# ==========================================
def train(force_cpu=False):
    print("=" * 60)
    print("🧪 EXPERIMENT: The Purist Readout (Average + Linear)")
    print("   Goal: Prove the LSM is doing the work.")
    print("=" * 60)

    # 1. Load Data
    trace_file = "lsm_trace_sequences.npz"
    if not Path(trace_file).exists():
        print(f"❌ Error: '{trace_file}' not found.")
        print("   Run 'traces_information_extraction.py' first.")
        return
        
    print(f"Loading '{trace_file}'...")
    dataset = np.load(trace_file, allow_pickle=True)
    X_train, y_train = dataset["X_train_sequences"], dataset["y_train"]
    X_test, y_test = dataset["X_test_sequences"], dataset["y_test"]
    
    # 2. Tokenizer
    label_map = load_label_map()
    tokenizer = train_custom_tokenizer(label_map, vocab_size=VOCAB_SIZE)
    y_train_text = [label_map[i] for i in y_train]
    y_test_text = [label_map[i] for i in y_test]

    # 3. Normalize & Tensor (MEMORY OPTIMIZED)
    print("Normalizing data...")
    print(f"  Data shape: Train={X_train.shape}, Test={X_test.shape}")

    # Calculate statistics in chunks to avoid memory spike
    print("  Computing mean and std...")
    num_samples, num_timesteps, num_features = X_train.shape

    # Use online algorithm for mean/std (Welford's method - memory efficient)
    mean = np.zeros(num_features, dtype=np.float64)
    m2 = np.zeros(num_features, dtype=np.float64)
    n = 0

    chunk_size = 50  # Process 50 samples at a time
    from tqdm import tqdm
    for i in tqdm(range(0, num_samples, chunk_size), desc="  Computing stats"):
        chunk = X_train[i:i+chunk_size].reshape(-1, num_features)
        for row in chunk:
            n += 1
            delta = row - mean
            mean += delta / n
            delta2 = row - mean
            m2 += delta * delta2

    std = np.sqrt(m2 / n) + 1e-8

    print("  Converting to tensors (this may take a moment)...")
    # Normalize in-place to avoid creating extra copies
    X_train -= mean
    X_train /= std
    X_test -= mean
    X_test /= std

    # Convert to tensors (views, not copies)
    X_train_tensor = torch.from_numpy(X_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()

    # Free original numpy arrays to save RAM
    del X_train, X_test, mean, std, m2
    print("  ✓ Normalization complete")

    # Device selection with memory awareness
    if force_cpu:
        device = torch.device("cpu")
        print(f"Running on device: CPU (forced)")
        print("⚠️  Training will be slower but use less memory")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {device}")

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"Features: {X_train_tensor.shape[2]} neurons (increased from 700)")
            print("⚠️  Note: Keeping data on CPU, moving batches to GPU as needed")
            print("⚠️  If you get OOM errors, run with --cpu flag")

    y_train_encoded = [encode_text_tokens(tokenizer, t) for t in y_train_text]
    
    # 4. Initialize The PURIST Model
    num_classes = tokenizer.get_vocab_size() + 1
    
    model = PuristLSMReadout(
        input_features=X_train_tensor.shape[2],
        num_classes=num_classes
    ).to(device)

    # Optimizer (Linear models can take higher LR)
    optimizer = optim.Adam(model.parameters(), lr=0.01) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=20, verbose=True)
    loss_fn = nn.CTCLoss(blank=CTC_BLANK_TOKEN, zero_infinity=True)

    # MEMORY OPTIMIZATION: Reduce batch size for larger models
    # Original batch_size=32 works for 700 neurons, but 1872 neurons needs smaller batches
    batch_size = 16 if X_train_tensor.shape[2] > 1000 else 32
    print(f"Using batch size: {batch_size} (adjusted for {X_train_tensor.shape[2]} features)")

    train_loader = DataLoader(
        TraceDataset(X_train_tensor, y_train_encoded),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        num_workers=0,  # Keep 0 to avoid memory duplication
        pin_memory=False  # Disable to save memory
    )

    # 5. Training
    print("\n🚀 Starting Training (Target: 1000 Epochs)...")
    try:
        for epoch in range(1000):
            model.train()
            epoch_loss = 0.0

            for x, y, lx, ly in train_loader:
                x, y, lx, ly = x.to(device), y.to(device), lx.to(device), ly.to(device)
                optimizer.zero_grad()
                out = model(x).permute(1, 0, 2)
                loss = loss_fn(out, y, lx, ly)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # MEMORY OPTIMIZATION: Clear GPU cache periodically
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            avg_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # --- Reporting & X-Ray ---
            if (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1:4d} | Loss: {avg_loss:.4f}")
                model.eval()
                
                # --- NEW: Get Target ---
                target_text = y_test_text[0] 
                
                # Smart X-Ray
                x_ray_sample = X_test_tensor[0].unsqueeze(0).to(device)
                with torch.no_grad():
                    log_probs = model(x_ray_sample).squeeze(0)
                    probs = torch.exp(log_probs)
                
                # Find active steps (where Blank < 90%)
                blank_probs = probs[:, 0]
                active_indices = torch.where(blank_probs < 0.90)[0]
                
                print("-" * 50)
                print(f"   🎯 TARGET:    '{target_text}'")
                print(f"   🤖 PREDICTED: '{decode_tokens(tokenizer, log_probs.cpu())}'")
                print("-" * 50)
                
                if len(active_indices) > 0:
                    print("   🔍 X-RAY (Active Steps):")
                    # Show up to 5 evenly spaced active moments
                    steps_to_show = active_indices[::max(1, len(active_indices)//5)][:5]
                    for t in steps_to_show:
                        t = t.item()
                        top_probs, top_ids = torch.topk(probs[t], k=2)
                        
                        if top_ids[0] == 0:
                            cand_id, cand_p = top_ids[1].item(), top_probs[1].item()
                        else:
                            cand_id, cand_p = top_ids[0].item(), top_probs[0].item()
                            
                        word = tokenizer.decode([cand_id - 1]) if cand_id > 0 else "UNK"
                        print(f"     Step {t:3d}: Blank {probs[t][0]:.0%} | '{word}' ({cand_p:.0%})")
                print("-" * 50)

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")

    # 6. Final Evaluation
    print("\n" + "="*60)
    print("FINAL PER-SENTENCE BREAKDOWN")
    print("="*60)
    model.eval()
    
    total_dist = 0
    total_ref_words = 0
    perfect_sentences = 0
    
    per_sentence_accuracies = []

    with torch.no_grad():
        for i in range(len(X_test_tensor)):
            # MEMORY OPTIMIZATION: Process one sample at a time and clear cache
            out = model(X_test_tensor[i].unsqueeze(0).to(device))
            pred = decode_tokens(tokenizer, out.cpu())

            # Clear GPU cache every 10 samples
            if device.type == 'cuda' and i % 10 == 0:
                torch.cuda.empty_cache()
            target = y_test_text[i]
            
            # Word-level stats
            dist, ref_len = calculate_edit_distance(pred, target)
            
            # Approximate "Correct Words" count
            correct_approx = max(0, ref_len - dist)
            accuracy_percent = (correct_approx / ref_len) * 100 if ref_len > 0 else 0
            
            # Update Globals
            total_dist += dist
            total_ref_words += ref_len
            per_sentence_accuracies.append(accuracy_percent)
            
            if pred == target:
                perfect_sentences += 1
                status = "🏆 PERFECT"
            else:
                status = "⚠️  ERROR"
                
            # Print report for first 20 sentences
            if i < 20: 
                print(f"Sample {i+1}: {status}")
                print(f"  Target:    '{target}'")
                print(f"  Predicted: '{pred}'")
                print(f"  Result:    {correct_approx}/{ref_len} words correct ({accuracy_percent:.0f}%)")
                print("-" * 40)

    # --- CALCULATE METRICS ---
    # Global Word Error Rate
    global_wer = total_dist / max(1, total_ref_words)
    global_word_acc = max(0.0, 1.0 - global_wer)
    
    # Sentence Accuracy
    sent_acc = perfect_sentences / len(X_test_tensor)
    
    # Mean Per-Sentence Accuracy
    mean_sentence_acc = sum(per_sentence_accuracies) / len(per_sentence_accuracies)

    print("\n" + "="*60)
    print(f"📊 FINAL METRICS")
    print("-" * 40)
    print(f"  🏆 Perfect Sentences:      {sent_acc*100:.2f}%")
    print(f"  📉 Global Word Error Rate: {global_wer:.4f}")
    print("-" * 40)
    print(f"  ✅ Average Sentence Score: {mean_sentence_acc:.2f}%")
    print(f"     (On average, the model gets {mean_sentence_acc:.0f}% of the words in a sentence right)")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CTC model on LSM traces")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training (slower but less memory)")
    args = parser.parse_args()

    train(force_cpu=args.cpu)