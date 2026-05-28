"""
Training script for ChoraleTransformer v2c.

Identical to v2b except for KEY-WEIGHTED augmentation instead of uniform ±5 semitones.
Instead of treating all transpositions equally, we weight shifts toward ±1-2 semitones
to better reflect Bach's actual key distribution (peaks near C, G, D, F major).

Augmentation scheme (repeats per training chorale):
  0 semitones:  1x  (original)
  ±1 semitones: 3x each  → 6 total  (nearby keys, most Bach-like)
  ±2 semitones: 2x each  → 4 total
  ±3 semitones: 1x each  → 2 total
  ±4 semitones: 1x each  → 2 total
  ±5 semitones: 1x each  → 2 total
  Total: 17x per chorale (vs 11x in v2b)

Why: uniform ±5 transposition teaches the model all 12 keys are equally likely,
flattening Bach's key preference → higher pitch class KL divergence. Weighted
augmentation restores the key distribution bias without losing harmony benefits
of interleaved tokenization.
"""

import json
import math
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tokenizer_v2 import BachTokenizerV2
from dataset import ChoraleDataset
from model import ChoraleTransformer, count_parameters


def transpose_sequence(seq, tokenizer, semitones):
    """Transpose a token sequence by shifting all MIDI pitches by semitones."""
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
                    if new_tid is not None:
                        transposed.append(new_tid)
                    else:
                        transposed.append(tid)
                else:
                    transposed.append(tid)
            except (ValueError, IndexError):
                transposed.append(tid)
        else:
            transposed.append(tid)
    return transposed


def main():
    checkpoint_dir = Path("/Users/kilehsu/153Assignment2/modeling/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    images_dir = Path("/Users/kilehsu/153Assignment2/images")
    images_dir.mkdir(exist_ok=True, parents=True)

    sequences_cache = checkpoint_dir / "v2_sequences_cache.pkl"
    tokenizer_path = checkpoint_dir / "v2_tokenizer.json"
    splits_path = checkpoint_dir / "v2_splits.json"

    print("Loading cached sequences and tokenizer...")
    if not sequences_cache.exists():
        print(f"ERROR: {sequences_cache} not found. Run train_v2.py first.")
        return
    tokenizer = BachTokenizerV2.load(str(tokenizer_path))
    with open(sequences_cache, 'rb') as f:
        cache = pickle.load(f)
    sequences = cache["sequences"]
    n_chorales = cache["n_chorales"]
    print(f"Loaded {n_chorales} cached sequences. Vocab size: {tokenizer.vocab_size}")

    print("\nLoading data splits...")
    with open(splits_path) as f:
        splits = json.load(f)
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]
    print(f"\nOriginal training set: {len(train_sequences)} sequences")

    # Key-weighted augmentation: more repeats for small shifts
    # shift -> number of copies to add (on top of the original which is kept once)
    shift_repeats = {
        -5: 1, -4: 1, -3: 1, -2: 2, -1: 3,
         1: 3,  2: 2,  3: 1,  4: 1,  5: 1,
    }

    print("Augmenting training data with key-weighted transpositions...")
    augmented_train = list(train_sequences)  # original copies (shift=0, 1x)
    for shift, repeats in shift_repeats.items():
        for _ in range(repeats):
            for seq in train_sequences:
                augmented_train.append(transpose_sequence(seq, tokenizer, shift))

    total_factor = 1 + sum(shift_repeats.values())
    print(f"Augmented training set: {len(augmented_train)} sequences ({total_factor}x original)")
    train_sequences = augmented_train

    print("\nCreating datasets...")
    train_dataset = ChoraleDataset(train_sequences, context_len=512, stride=64)
    val_dataset = ChoraleDataset(val_sequences, context_len=512, stride=32)
    print(f"Train dataset size: {len(train_dataset)} windows")
    print(f"Val dataset size: {len(val_dataset)} windows")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    print("\nInstantiating model...")
    model = ChoraleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=4,
        context_len=512,
        dropout=0.15,
    )
    model.output_proj.weight = model.token_embedding.weight
    param_count = count_parameters(model)
    print(f"Model parameter count: {param_count:,}")

    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    except Exception:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    num_epochs = 12

    warmup_steps = 1000
    total_steps = num_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    current_step = 0

    print("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = checkpoint_dir / "v2c_transformer_best.pt"

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                target_ids.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            current_step += 1

            epoch_train_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 200 == 0:
                print(f"  Epoch {epoch+1} batch {batch_idx+1}/{len(train_loader)} loss={loss.item():.4f}", flush=True)

        epoch_train_loss /= num_batches
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, tokenizer.vocab_size),
                    target_ids.view(-1),
                )
                epoch_val_loss += loss.item()
                num_val_batches += 1

        if num_val_batches > 0:
            epoch_val_loss /= num_val_batches
        val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f} | LR: {current_lr:.7f}")

    print("\nSaving final model...")
    torch.save(model.state_dict(), checkpoint_dir / "v2c_transformer_final.pt")

    losses_dict = {"train_losses": train_losses, "val_losses": val_losses, "best_val_loss": best_val_loss}
    with open(checkpoint_dir / "v2c_losses.json", 'w') as f:
        json.dump(losses_dict, f)

    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        plt.plot(val_losses, label="Val Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curves (V2c - Key-Weighted Augmentation)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = images_dir / "09e_training_curves_v2c.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_path}")
    except Exception as e:
        print(f"Error plotting: {e}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE (V2c)")
    print("=" * 50)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {param_count:,}")
    print(f"Train/Val/Test split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    print(f"Augmented training set: {len(augmented_train)} sequences ({total_factor}x original)")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
