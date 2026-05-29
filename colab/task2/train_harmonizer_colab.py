"""
Colab training script for POP909 harmonizer.
Run this in a Colab notebook with cells as marked below.
"""

# ============================================================================
# CELL 1: Install dependencies
# ============================================================================

# !pip install torch pretty_midi music21 --quiet

# ============================================================================
# CELL 2: Define dataset class
# ============================================================================

import os
import torch
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class POP909Dataset(Dataset):
    """Load pre-built dataset cache from pickle file."""

    def __init__(self, cache_file, split='train'):
        """
        Args:
            cache_file: path to pop909_cache.pkl
            split: 'train' or 'val'
        """
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
        self.windows = cache[split]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        melody_seq, piano_seq = self.windows[idx]
        melody = torch.LongTensor(melody_seq)
        piano = torch.LongTensor(piano_seq)
        return melody, piano


# ============================================================================
# CELL 3: Define model architecture
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F


class MelodyEncoder(nn.Module):
    """Bidirectional LSTM encoder for melody."""

    def __init__(self, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(130, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

    def forward(self, melody_seq):
        embedded = self.embedding(melody_seq)
        encoder_outputs, (h_n, c_n) = self.lstm(embedded)
        return encoder_outputs, (h_n, c_n)


class BahdanauAttention(nn.Module):
    """Bahdanau-style additive attention."""

    def __init__(self, hidden_dim, encoder_hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, encoder_hidden_dim)
        self.key_proj = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        self.v = nn.Linear(encoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        query = self.query_proj(decoder_hidden).unsqueeze(1)
        keys = self.key_proj(encoder_outputs)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)
        attention_weights = F.softmax(scores, dim=1)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)
        return context, attention_weights


class ChordDecoder(nn.Module):
    """LSTM decoder for generating 4-note chords."""

    def __init__(self, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(130, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim * 4, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.attention = BahdanauAttention(hidden_dim, 2 * hidden_dim)
        self.fc_out = nn.Linear(hidden_dim + 2 * hidden_dim, 4 * 129)

    def forward(self, piano_seq, encoder_outputs, encoder_hidden, teacher_forcing_ratio=0.5):
        batch_size, seq_len, _ = piano_seq.shape

        h_n, c_n = encoder_hidden
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_dim)
        h_n = h_n.mean(dim=1)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_dim)
        c_n = c_n.mean(dim=1)

        decoder_hidden = h_n
        decoder_cell = c_n

        all_logits = []
        all_targets = []
        prev_chord = torch.zeros(batch_size, 4, dtype=torch.long, device=piano_seq.device)

        for t in range(seq_len):
            embedded = self.embedding(prev_chord)
            embedded = embedded.reshape(batch_size, -1).unsqueeze(1)

            _, (decoder_hidden, decoder_cell) = self.lstm(
                embedded, (decoder_hidden, decoder_cell)
            )

            context, _ = self.attention(decoder_hidden[-1], encoder_outputs)
            combined = torch.cat([decoder_hidden[-1], context], dim=1)
            logits = self.fc_out(combined)

            all_logits.append(logits)
            all_targets.append(piano_seq[:, t, :])

            if torch.rand(1).item() < teacher_forcing_ratio:
                prev_chord = piano_seq[:, t, :]
            else:
                logits_reshaped = logits.view(batch_size, 4, 129)
                prev_chord = torch.argmax(logits_reshaped, dim=2)

        all_logits = torch.stack(all_logits, dim=1)
        all_logits = all_logits.reshape(batch_size * seq_len, 4 * 129)
        all_targets = torch.stack(all_targets, dim=1)
        all_targets = all_targets.reshape(batch_size * seq_len, 4)

        return all_logits, all_targets

    def generate(self, encoder_outputs, encoder_hidden, max_len=64, greedy=True):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        h_n, c_n = encoder_hidden
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_dim)
        h_n = h_n.mean(dim=1)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_dim)
        c_n = c_n.mean(dim=1)

        decoder_hidden = h_n
        decoder_cell = c_n

        generated = []
        prev_chord = torch.zeros(batch_size, 4, dtype=torch.long, device=device)

        for t in range(max_len):
            embedded = self.embedding(prev_chord)
            embedded = embedded.reshape(batch_size, -1).unsqueeze(1)

            _, (decoder_hidden, decoder_cell) = self.lstm(
                embedded, (decoder_hidden, decoder_cell)
            )

            context, _ = self.attention(decoder_hidden[-1], encoder_outputs)
            combined = torch.cat([decoder_hidden[-1], context], dim=1)
            logits = self.fc_out(combined)

            logits_reshaped = logits.view(batch_size, 4, 129)

            if greedy:
                prev_chord = torch.argmax(logits_reshaped, dim=2)
            else:
                probs = F.softmax(logits_reshaped, dim=2)
                prev_chord = torch.multinomial(
                    probs.reshape(batch_size * 4, 129), 1
                ).reshape(batch_size, 4)

            generated.append(prev_chord.detach())

        return torch.stack(generated, dim=1)


class HarmonizerSeq2Seq(nn.Module):
    """Complete seq2seq model."""

    def __init__(self, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = MelodyEncoder(embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = ChordDecoder(embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, melody_seq, piano_seq, teacher_forcing_ratio=0.5):
        encoder_outputs, encoder_hidden = self.encoder(melody_seq)
        logits, targets = self.decoder(
            piano_seq, encoder_outputs, encoder_hidden, teacher_forcing_ratio
        )
        return logits, targets

    def generate(self, melody_seq, max_len=64, greedy=True):
        encoder_outputs, encoder_hidden = self.encoder(melody_seq)
        generated = self.decoder.generate(
            encoder_outputs, encoder_hidden, max_len, greedy
        )
        return generated


# ============================================================================
# CELL 4: Load data and create dataloaders
# ============================================================================

# Assuming pop909_cache.pkl has been uploaded to /content/
cache_file = '/content/pop909_cache.pkl'

train_ds = POP909Dataset(cache_file, split='train')
val_ds = POP909Dataset(cache_file, split='val')

batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds)} windows")
print(f"Val: {len(val_ds)} windows")

# ============================================================================
# CELL 5: Create model and optimizer
# ============================================================================

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

embed_dim = 128
hidden_dim = 256
num_layers = 2
dropout = 0.3
lr = 1e-3

model = HarmonizerSeq2Seq(embed_dim, hidden_dim, num_layers, dropout)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

print("Model created")

# ============================================================================
# CELL 6: Training loop
# ============================================================================

import json

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0

    for melody, piano in train_loader:
        melody = melody.to(device)
        piano = piano.to(device)

        logits, targets = model(melody, piano, teacher_forcing_ratio=0.5)
        logits_reshaped = logits.view(-1, 4, 129)
        targets_reshaped = targets.view(-1, 4)

        loss = 0
        for voice in range(4):
            voice_loss = nn.CrossEntropyLoss(ignore_index=0)(
                logits_reshaped[:, voice, :],
                targets_reshaped[:, voice]
            )
            loss = loss + voice_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, val_loader, device):
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


epochs = 30
losses = {'train': [], 'val': []}
best_val_loss = float('inf')
best_epoch = 0

print(f"Training for {epochs} epochs (estimated 20-30 min on T4)...")

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
        checkpoint_path = '/content/harmonizer_best.pt'
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
        print(f"  Saved best checkpoint")

print(f"\n\nTraining complete!")
print(f"Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

# Save loss curve
with open('/content/harmonizer_losses.json', 'w') as f:
    json.dump(losses, f, indent=2)

# ============================================================================
# CELL 7: Download checkpoint
# ============================================================================

# After training, download /content/harmonizer_best.pt and /content/harmonizer_losses.json
# Use Files pane in Colab to download

print("Download harmonizer_best.pt and harmonizer_losses.json from the Files pane")
