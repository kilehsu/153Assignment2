"""
Training script for ChoraleTransformer v2b with data augmentation via transposition.

This script addresses overfitting in v2 by:
1. Using a smaller model (d_model=128, n_layers=4, n_heads=8)
2. Augmenting training data via transposition (±5 semitones → 11x more data)
3. Higher dropout (0.15)
4. Lower peak learning rate (5e-4)
5. More epochs (40) since augmentation makes each epoch harder

The transposition augmentation creates 11 variants (original + 10 transposed) per training chorale,
resulting in 294 * 11 = 3234 augmented sequences.
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
    """
    Transpose a token sequence by shifting all MIDI pitches by semitones.

    Args:
        seq: List of token IDs
        tokenizer: BachTokenizerV2 instance
        semitones: Number of semitones to transpose (negative for down)

    Returns:
        List of transposed token IDs
    """
    transposed = []
    for tid in seq:
        tok = tokenizer.id_to_token.get(tid, "")

        # Special tokens pass through unchanged
        if tok in ("<START>", "<END>", "<BAR>", "<SUSTAIN>") or tok.endswith("REST"):
            transposed.append(tid)
        # Note tokens: V{v}P{midi}D{dur}
        elif tok.startswith("V") and "P" in tok and "D" in tok:
            try:
                # Parse V{v}P{midi}D{dur}
                after_v = tok[1:]
                v_part, rest = after_v.split("P", 1)
                midi_str, dur_str = rest.split("D", 1)

                new_midi = int(midi_str) + semitones

                # Clamp to valid MIDI range 21-108
                if 21 <= new_midi <= 108:
                    new_tok = f"V{v_part}P{new_midi}D{dur_str}"
                    new_tid = tokenizer.token_to_id.get(new_tok)
                    if new_tid is not None:
                        transposed.append(new_tid)
                    else:
                        # Fallback: keep original if transposed token not in vocab
                        transposed.append(tid)
                else:
                    # Out of range, keep original
                    transposed.append(tid)
            except (ValueError, IndexError):
                # Parse error, keep original
                transposed.append(tid)
        else:
            # Unknown token, keep original
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

    # ========== Step 1-3: Load cached sequences and tokenizer ==========
    print("Loading cached sequences and tokenizer...")
    if not sequences_cache.exists():
        print(f"ERROR: {sequences_cache} not found. Run train_v2.py first to generate cache.")
        return
    if not tokenizer_path.exists():
        print(f"ERROR: {tokenizer_path} not found. Run train_v2.py first to generate tokenizer.")
        return

    tokenizer = BachTokenizerV2.load(str(tokenizer_path))
    with open(sequences_cache, 'rb') as f:
        cache = pickle.load(f)
    sequences = cache["sequences"]
    n_chorales = cache["n_chorales"]
    print(f"Loaded {n_chorales} cached sequences. Vocab size: {tokenizer.vocab_size}")

    # ========== Step 4: Load existing splits ==========
    print("\nLoading data splits...")
    if not splits_path.exists():
        print(f"ERROR: {splits_path} not found. Run train_v2.py first to generate splits.")
        return

    with open(splits_path) as f:
        splits = json.load(f)
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # ========== Step 5: Extract training sequences and augment via transposition ==========
    print("\nExtracting training sequences...")
    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]

    print(f"Original training set: {len(train_sequences)} sequences")

    print("Augmenting training data with transpositions (±5 semitones)...")
    augmented_train = list(train_sequences)  # Keep originals

    # Add transposed variants: ±5, ±4, ±3, ±2, ±1 semitones
    for semitones in [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]:
        for seq in train_sequences:
            transposed = transpose_sequence(seq, tokenizer, semitones)
            augmented_train.append(transposed)

    print(f"Augmented training set: {len(augmented_train)} sequences (was {len(train_sequences)})")
    train_sequences = augmented_train

    # ========== Step 6: Create datasets and loaders ==========
    print("\nCreating datasets...")
    train_dataset = ChoraleDataset(train_sequences, context_len=512, stride=32)
    val_dataset = ChoraleDataset(val_sequences, context_len=512, stride=16)

    print(f"Train dataset size: {len(train_dataset)} windows")
    print(f"Val dataset size: {len(val_dataset)} windows")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # ========== Step 7: Instantiate model ==========
    print("\nInstantiating model...")
    model = ChoraleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=4,
        context_len=512,
        dropout=0.15,
    )

    # Weight tying: tie output projection to token embedding
    model.output_proj.weight = model.token_embedding.weight

    param_count = count_parameters(model)
    print(f"Model parameter count: {param_count:,}")

    # ========== Step 8: Set device ==========
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    except Exception:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # ========== Step 9: Optimizer and scheduler ==========
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    num_epochs = 40

    # Linear warmup + cosine decay scheduler
    warmup_steps = 1000
    total_steps = num_epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    current_step = 0

    # ========== Step 10: Training loop ==========
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = checkpoint_dir / "v2b_transformer_best.pt"

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        num_batches = 0

        for input_ids, target_ids in train_loader:
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

        epoch_train_loss /= num_batches
        train_losses.append(epoch_train_loss)

        # Validation
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

        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | LR: {current_lr:.7f}")

    # ========== Step 11: Save final model ==========
    print("\nSaving models...")
    final_model_path = checkpoint_dir / "v2b_transformer_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # ========== Step 12: Save losses ==========
    losses_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
    losses_path = checkpoint_dir / "v2b_losses.json"
    with open(losses_path, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {losses_path}")

    # ========== Step 13: Plot training curves ==========
    print("\nPlotting training curves...")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        plt.plot(val_losses, label="Val Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curves (V2b - with Transposition Augmentation)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = images_dir / "09d_training_curves_v2b.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Error plotting: {e}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE (V2b)")
    print("=" * 50)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {param_count:,}")
    print(f"Original train/val/test split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    print(f"Augmented training set: {len(augmented_train)} sequences (11x original)")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
