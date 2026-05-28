# ============================================================
# V2C TRAINING — Colab Version (Key-Weighted Augmentation)
# ============================================================
# Run each section as a separate cell in Colab.
# Expected runtime: ~45-60 min on T4, ~15-20 min on A100.
# ============================================================


# ── CELL 1: Install dependencies ────────────────────────────
# !pip install music21 torch torchvision --quiet


# ── CELL 2: Upload files ─────────────────────────────────────
# Run this cell to upload the required files from your Mac.
# Upload all 6 files when prompted.
#
# from google.colab import files
# uploaded = files.upload()
# # Upload these files:
# #   model.py
# #   dataset.py
# #   tokenizer_v2.py
# #   v2_tokenizer.json
# #   v2_sequences_cache.pkl
# #   v2_splits.json


# ── CELL 3: Training script ──────────────────────────────────
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Files uploaded to Colab are placed in /content/
import sys
sys.path.insert(0, '/content')

from tokenizer_v2 import BachTokenizerV2
from dataset import ChoraleDataset
from model import ChoraleTransformer, count_parameters

CHECKPOINT_DIR = Path('/content')  # save checkpoint here
IMAGES_DIR = Path('/content')


def transpose_sequence(seq, tokenizer, semitones):
    transposed = []
    for tid in seq:
        tok = tokenizer.id_to_token.get(tid, "")
        if tok in ("<START>", "<END>", "<BAR>", "<SUSTAIN>") or tok.endswith("REST"):
            transposed.append(tid)
        elif tok.startswith("V") and "P" in tok and "D" in tok:
            try:
                after_v = tok[1:]
                v_part, rest = after_v.split("P", 1)
                midi_str, dur_str = rest.split("D", 1)
                new_midi = int(midi_str) + semitones
                if 21 <= new_midi <= 108:
                    new_tok = f"V{v_part}P{new_midi}D{dur_str}"
                    new_tid = tokenizer.token_to_id.get(new_tok)
                    transposed.append(new_tid if new_tid is not None else tid)
                else:
                    transposed.append(tid)
            except (ValueError, IndexError):
                transposed.append(tid)
        else:
            transposed.append(tid)
    return transposed


def main():
    tokenizer = BachTokenizerV2.load('/content/v2_tokenizer.json')
    with open('/content/v2_sequences_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    with open('/content/v2_splits.json') as f:
        splits = json.load(f)

    sequences = cache["sequences"]
    train_indices = splits["train"]
    val_indices = splits["val"]
    print(f"Vocab size: {tokenizer.vocab_size} | Train: {len(train_indices)} | Val: {len(val_indices)}")

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]

    # Key-weighted augmentation: favour small shifts over large ones
    shift_repeats = {-5: 1, -4: 1, -3: 1, -2: 2, -1: 3, 1: 3, 2: 2, 3: 1, 4: 1, 5: 1}
    augmented_train = list(train_sequences)
    for shift, repeats in shift_repeats.items():
        for _ in range(repeats):
            for seq in train_sequences:
                augmented_train.append(transpose_sequence(seq, tokenizer, shift))

    total_factor = 1 + sum(shift_repeats.values())
    print(f"Augmented: {len(augmented_train)} sequences ({total_factor}x)")

    train_dataset = ChoraleDataset(augmented_train, context_len=512, stride=64)
    val_dataset   = ChoraleDataset(val_sequences,   context_len=512, stride=32)
    print(f"Train windows: {len(train_dataset)} | Val windows: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = ChoraleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_heads=8, n_layers=4,
        context_len=512, dropout=0.15,
    )
    model.output_proj.weight = model.token_embedding.weight
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    num_epochs   = 40
    warmup_steps = 1000
    optimizer    = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    total_steps  = num_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_path = CHECKPOINT_DIR / 'v2c_transformer_best.pt'

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            n_batches  += 1
            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1)).item()
                n_val += 1
        val_loss = val_loss / n_val if n_val > 0 else 0
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.7f}")

    torch.save(model.state_dict(), CHECKPOINT_DIR / 'v2c_transformer_final.pt')
    with open(CHECKPOINT_DIR / 'v2c_losses.json', 'w') as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses, "best_val_loss": best_val_loss}, f)
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {best_path}")


main()


# ── CELL 4: Download the checkpoint ─────────────────────────
# from google.colab import files
# files.download('/content/v2c_transformer_best.pt')
# files.download('/content/v2c_losses.json')
