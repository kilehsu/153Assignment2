"""
Main training script for ChoraleTransformer.
Loads Bach chorales, builds tokenizer, trains the model, and saves checkpoints.
"""

import json
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from music21 import corpus

from tokenizer import BachTokenizer
from dataset import ChoraleDataset
from model import ChoraleTransformer, count_parameters


def load_bach_chorales():
    """Load all 4-part Bach chorales from music21 corpus."""
    print("Loading Bach chorales from music21 corpus...")
    chorales = []

    bach_files = corpus.getComposer('bach')
    score_files = []
    for f in bach_files:
        f_str = str(f).lower()
        if any(ext in f_str for ext in ['.musicxml', '.xml', '.mxl', '.krn']):
            if 'analyses' not in f_str:
                score_files.append(str(f))

    print(f"Found {len(score_files)} Bach score files, filtering for 4-part chorales...")

    for filepath in score_files:
        try:
            s = corpus.parse(filepath)
            if len(s.parts) == 4:
                chorales.append(s)
        except Exception:
            continue

    print(f"Loaded {len(chorales)} 4-part Bach chorales")
    return chorales


def main():
    checkpoint_dir = Path("/Users/kilehsu/153Assignment2/modeling/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    images_dir = Path("/Users/kilehsu/153Assignment2/images")
    images_dir.mkdir(exist_ok=True, parents=True)

    sequences_cache = checkpoint_dir / "sequences_cache.pkl"

    # ========== Step 1-3: Load chorales and encode (with cache) ==========
    if sequences_cache.exists() and (checkpoint_dir / "tokenizer.json").exists():
        print("Loading cached encoded sequences...")
        tokenizer = BachTokenizer.load(str(checkpoint_dir / "tokenizer.json"))
        with open(sequences_cache, 'rb') as f:
            cache = pickle.load(f)
        sequences = cache["sequences"]
        n_chorales = cache["n_chorales"]
        print(f"Loaded {n_chorales} cached sequences. Vocab size: {tokenizer.vocab_size}")
    else:
        chorales = load_bach_chorales()
        n_chorales = len(chorales)

        print("\nBuilding tokenizer...")
        tokenizer = BachTokenizer()
        tokenizer.build_vocab(chorales)
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        tokenizer.save(str(checkpoint_dir / "tokenizer.json"))

        print("\nEncoding all chorales...")
        sequences = []
        for i, chorale in enumerate(chorales):
            seq = tokenizer.encode(chorale)
            sequences.append(seq)
            if (i + 1) % 50 == 0:
                print(f"  Encoded {i + 1}/{n_chorales}")

        with open(sequences_cache, 'wb') as f:
            pickle.dump({"sequences": sequences, "n_chorales": n_chorales}, f)
        print(f"Sequences cached to {sequences_cache}")

    # ========== Step 4: Split data 80/10/10 ==========
    splits_path = checkpoint_dir / "splits.json"
    if splits_path.exists():
        print("\nLoading existing splits...")
        with open(splits_path) as f:
            splits = json.load(f)
        train_indices = splits["train"]
        val_indices = splits["val"]
        test_indices = splits["test"]
    else:
        print("\nSplitting data (80/10/10 train/val/test)...")
        n_train = int(0.8 * n_chorales)
        n_val = int(0.1 * n_chorales)
        indices = np.random.permutation(n_chorales)
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()
        splits = {"train": train_indices, "val": val_indices, "test": test_indices}
        with open(splits_path, 'w') as f:
            json.dump(splits, f)

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # ========== Step 5: Create datasets and loaders ==========
    print("\nCreating datasets...")
    train_sequences = [sequences[i] for i in train_indices]
    val_sequences = [sequences[i] for i in val_indices]

    train_dataset = ChoraleDataset(train_sequences, context_len=512, stride=16)
    val_dataset   = ChoraleDataset(val_sequences,   context_len=512, stride=16)

    print(f"Train dataset size: {len(train_dataset)} windows")
    print(f"Val dataset size: {len(val_dataset)} windows")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=0)

    # ========== Step 6-8: Instantiate model ==========
    print("\nInstantiating model...")
    model = ChoraleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=4,
        context_len=512,
        dropout=0.1,
    )

    param_count = count_parameters(model)
    print(f"Model parameter count: {param_count:,}")

    # ========== Step 9: Set device ==========
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    except Exception:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # ========== Step 10: Optimizer and scheduler ==========
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    num_epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ========== Step 12: Training loop ==========
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_path = checkpoint_dir / "transformer_best.pt"

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

            epoch_train_loss += loss.item()
            num_batches += 1

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

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # ========== Step 13: Save final model ==========
    print("\nSaving models...")
    final_model_path = checkpoint_dir / "transformer_final.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # ========== Step 14: Save losses ==========
    losses_dict = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
    losses_path = checkpoint_dir / "losses.json"
    with open(losses_path, 'w') as f:
        json.dump(losses_dict, f)
    print(f"Losses saved to {losses_path}")

    # ========== Step 15: Plot training curves ==========
    print("\nPlotting training curves...")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss", linewidth=2)
        plt.plot(val_losses, label="Val Loss", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curves")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = images_dir / "09_training_curves.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Error plotting: {e}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Model parameters: {param_count:,}")
    print(f"Train/Val/Test split: {len(train_indices)}/{len(val_indices)}/{len(test_indices)}")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
