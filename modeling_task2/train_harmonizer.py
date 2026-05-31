"""
Training script for POP909 harmonizer seq2seq model.
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add modeling folder to path
sys.path.insert(0, os.path.dirname(__file__))

from pop909_dataset import POP909Dataset
from harmonizer_model import HarmonizerSeq2Seq


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for melody, piano in train_loader:
        melody = melody.to(device)
        piano = piano.to(device)

        # Forward pass
        logits, targets = model(melody, piano, teacher_forcing_ratio=0.5)

        # Loss: 4 separate cross-entropy losses (one per voice)
        logits_reshaped = logits.view(-1, 4, 129)
        targets_reshaped = targets.view(-1, 4)

        loss = 0
        for voice in range(4):
            voice_loss = nn.CrossEntropyLoss(ignore_index=0)(
                logits_reshaped[:, voice, :],
                targets_reshaped[:, voice]
            )
            loss = loss + voice_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 100 == 0:
            print(f"  Batch {num_batches}/{len(train_loader)}, loss={loss.item():.4f}")

    return total_loss / num_batches


def validate(model, val_loader, device):
    """Compute validation loss."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for melody, piano in val_loader:
            melody = melody.to(device)
            piano = piano.to(device)

            logits, targets = model(melody, piano, teacher_forcing_ratio=1.0)

            logits_reshaped = logits.view(-1, 4, 129)
            targets_reshaped = targets.view(-1, 4)

            loss = 0
            for voice in range(4):
                voice_loss = nn.CrossEntropyLoss(ignore_index=0)(
                    logits_reshaped[:, voice, :],
                    targets_reshaped[:, voice]
                )
                loss = loss + voice_loss

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    # Hyperparameters
    embed_dim = 128
    hidden_dim = 256
    num_layers = 2
    dropout = 0.3
    batch_size = 64
    lr = 1e-3
    epochs = 15  # val loss historically bottoms ~epoch 6; 15 gives margin and saves CPU hours
    seed = 42

    # Device: CUDA if present, else CPU. MPS is intentionally skipped — for this
    # small per-timestep LSTM decoder the GPU gives no speedup over CPU (kernel-launch
    # overhead dominates the 64-step Python loop). Set HARMONIZER_DEVICE to override.
    forced = os.environ.get('HARMONIZER_DEVICE')
    if forced:
        device = torch.device(forced)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(seed)

    # Data
    print("Loading datasets...")
    data_dir = 'data/POP909/POP909'
    train_ds = POP909Dataset(data_dir, split='train', val_ratio=0.1, seed=seed)
    val_ds = POP909Dataset(data_dir, split='val', val_ratio=0.1, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} windows, Val: {len(val_ds)} windows")

    # Model
    print("Creating model...")
    model = HarmonizerSeq2Seq(embed_dim, hidden_dim, num_layers, dropout)
    model = model.to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Training loop
    losses = {'train': [], 'val': []}
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        losses['train'].append(train_loss)
        losses['val'].append(val_loss)

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save checkpoint
            checkpoint_path = os.path.join(os.path.dirname(__file__), 'harmonizer_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'hyperparams': {
                    'embed_dim': embed_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                }
            }, checkpoint_path)
            print(f"  Saved best checkpoint to {checkpoint_path}")

    # Save loss curve
    loss_file = os.path.join(os.path.dirname(__file__), 'harmonizer_losses.json')
    with open(loss_file, 'w') as f:
        json.dump(losses, f, indent=2)
    print(f"\nSaved loss curve to {loss_file}")
    print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    main()
