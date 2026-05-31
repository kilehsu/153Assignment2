"""
Seq2Seq harmonizer: Encode melody, decode piano accompaniment with attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MelodyEncoder(nn.Module):
    """
    Bidirectional LSTM encoder for melody sequences.

    Input: (batch_size, seq_len) of pitch values 0-128
    Output: (batch_size, seq_len, 2*hidden_dim) of encoder states
    """

    def __init__(self, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding layer for melody pitches
        self.embedding = nn.Embedding(130, embed_dim)  # 0-128 pitches + padding

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

    def forward(self, melody_seq):
        """
        Args:
            melody_seq: (batch_size, seq_len) LongTensor

        Returns:
            encoder_outputs: (batch_size, seq_len, 2*hidden_dim)
            (h_n, c_n): final hidden and cell states
        """
        embedded = self.embedding(melody_seq)  # (batch_size, seq_len, embed_dim)
        encoder_outputs, (h_n, c_n) = self.lstm(embedded)
        return encoder_outputs, (h_n, c_n)


class BahdanauAttention(nn.Module):
    """Bahdanau-style additive attention."""

    def __init__(self, hidden_dim, encoder_hidden_dim):
        super().__init__()
        self.q = nn.Linear(hidden_dim, encoder_hidden_dim)
        self.k = nn.Linear(encoder_hidden_dim, encoder_hidden_dim)
        self.v = nn.Linear(encoder_hidden_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: (batch_size, hidden_dim)
            encoder_outputs: (batch_size, seq_len, encoder_hidden_dim)

        Returns:
            context: (batch_size, encoder_hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        query = self.q(decoder_hidden).unsqueeze(1)  # (batch, 1, encoder_hidden)
        keys = self.k(encoder_outputs)               # (batch, seq_len, encoder_hidden)
        scores = self.v(torch.tanh(query + keys)).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(scores, dim=1)
        context = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)  # (batch, encoder_hidden_dim)
        return context, attention_weights


class ChordDecoder(nn.Module):
    """
    LSTM decoder that generates 4 simultaneous pitches per step.

    Inputs at each step:
    - Previous chord (4 pitches) embedded
    - Attention context from encoder
    - LSTM hidden state

    Output: 4 logits for pitches 0-128 (128=REST/pad)
    """

    def __init__(self, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Embedding for chord pitches
        self.embedding = nn.Embedding(130, embed_dim)

        # LSTM decoder
        self.lstm = nn.LSTM(
            embed_dim * 4,  # 4 pitches per step
            hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )

        # Attention (encoder outputs are bidirectional: 2*hidden_dim)
        self.attention = BahdanauAttention(hidden_dim, 2 * hidden_dim)

        # Output projection: decoder_hidden (hidden_dim) + context (2*hidden_dim) -> 4 * (vocab=129)
        self.fc_out = nn.Linear(hidden_dim + 2 * hidden_dim, 4 * 129)

    def forward(self, piano_seq, encoder_outputs, encoder_hidden, teacher_forcing_ratio=0.5):
        """
        Args:
            piano_seq: (batch_size, seq_len, 4) target chords
            encoder_outputs: (batch_size, seq_len, 2*hidden_dim)
            encoder_hidden: tuple of (h_n, c_n)
            teacher_forcing_ratio: use ground truth as input probability

        Returns:
            logits: (batch_size * seq_len, 4 * 129) raw predictions
            piano_seq_true: (batch_size * seq_len, 4) target indices
        """
        batch_size, seq_len, _ = piano_seq.shape

        # Initialize decoder hidden state from encoder
        # BiLSTM encoder output dim: 2*hidden_dim
        # We need to project it back to hidden_dim for decoder
        h_n, c_n = encoder_hidden

        # Average bidirectional states
        h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_dim)
        h_n = h_n.mean(dim=1)  # (num_layers, batch, hidden_dim)
        c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_dim)
        c_n = c_n.mean(dim=1)  # (num_layers, batch, hidden_dim)

        decoder_hidden = h_n
        decoder_cell = c_n

        all_logits = []
        all_targets = []

        # Decode step by step
        prev_chord = torch.zeros(batch_size, 4, dtype=torch.long, device=piano_seq.device)

        for t in range(seq_len):
            # Embed previous chord (4 pitches)
            embedded = self.embedding(prev_chord)  # (batch, 4, embed_dim)
            embedded = embedded.reshape(batch_size, -1).unsqueeze(1)  # (batch, 1, 4*embed_dim)

            # LSTM step
            _, (decoder_hidden, decoder_cell) = self.lstm(
                embedded, (decoder_hidden, decoder_cell)
            )

            # Attention over encoder outputs
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs)

            # Combine hidden state and context
            combined = torch.cat([decoder_hidden[-1], context], dim=1)  # (batch, 2*hidden_dim)
            logits = self.fc_out(combined)  # (batch, 4*129)

            all_logits.append(logits)
            all_targets.append(piano_seq[:, t, :])  # (batch, 4)

            # Teacher forcing or greedy decoding
            if torch.rand(1).item() < teacher_forcing_ratio:
                prev_chord = piano_seq[:, t, :]
            else:
                # Greedy: select argmax for each of 4 voices
                logits_reshaped = logits.view(batch_size, 4, 129)
                prev_chord = torch.argmax(logits_reshaped, dim=2)

        # Stack all timesteps
        all_logits = torch.stack(all_logits, dim=1)  # (batch, seq_len, 4*129)
        all_logits = all_logits.reshape(batch_size * seq_len, 4 * 129)
        all_targets = torch.stack(all_targets, dim=1)  # (batch, seq_len, 4)
        all_targets = all_targets.reshape(batch_size * seq_len, 4)

        return all_logits, all_targets

    def generate(self, encoder_outputs, encoder_hidden, max_len=64, greedy=True):
        """
        Generate a piano sequence given encoder outputs.

        Args:
            encoder_outputs: (batch_size, seq_len, 2*hidden_dim)
            encoder_hidden: tuple of (h_n, c_n)
            max_len: maximum length to generate
            greedy: if True, use argmax; else sample

        Returns:
            generated: (batch_size, max_len, 4) LongTensor
        """
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device

        # Initialize decoder hidden state
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
            # Embed previous chord
            embedded = self.embedding(prev_chord)  # (batch, 4, embed_dim)
            embedded = embedded.reshape(batch_size, -1).unsqueeze(1)  # (batch, 1, 4*embed)

            # LSTM step
            _, (decoder_hidden, decoder_cell) = self.lstm(
                embedded, (decoder_hidden, decoder_cell)
            )

            # Attention
            context, _ = self.attention(decoder_hidden[-1], encoder_outputs)

            # Output
            combined = torch.cat([decoder_hidden[-1], context], dim=1)
            logits = self.fc_out(combined)  # (batch, 4*129)

            logits_reshaped = logits.view(batch_size, 4, 129)

            if greedy:
                prev_chord = torch.argmax(logits_reshaped, dim=2)
            else:
                # Sample from distribution
                probs = F.softmax(logits_reshaped, dim=2)
                prev_chord = torch.multinomial(
                    probs.reshape(batch_size * 4, 129), 1
                ).reshape(batch_size, 4)

            generated.append(prev_chord.detach())

        return torch.stack(generated, dim=1)  # (batch, max_len, 4)


class HarmonizerSeq2Seq(nn.Module):
    """
    Complete seq2seq model: MelodyEncoder + ChordDecoder with attention.
    """

    def __init__(self, embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = MelodyEncoder(embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = ChordDecoder(embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, melody_seq, piano_seq, teacher_forcing_ratio=0.5):
        """
        Args:
            melody_seq: (batch_size, seq_len)
            piano_seq: (batch_size, seq_len, 4)
            teacher_forcing_ratio: probability of using ground truth

        Returns:
            logits: (batch_size * seq_len, 4 * 129)
            targets: (batch_size * seq_len, 4)
        """
        encoder_outputs, encoder_hidden = self.encoder(melody_seq)
        logits, targets = self.decoder(
            piano_seq, encoder_outputs, encoder_hidden, teacher_forcing_ratio
        )
        return logits, targets

    def generate(self, melody_seq, max_len=64, greedy=True):
        """
        Generate piano accompaniment for a given melody.

        Args:
            melody_seq: (batch_size, seq_len)
            max_len: length of sequence to generate
            greedy: greedy decoding if True, else sample

        Returns:
            generated: (batch_size, max_len, 4)
        """
        encoder_outputs, encoder_hidden = self.encoder(melody_seq)
        generated = self.decoder.generate(
            encoder_outputs, encoder_hidden, max_len, greedy
        )
        return generated
