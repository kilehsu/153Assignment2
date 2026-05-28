"""
ChoraleTransformer: GPT-style decoder-only Transformer for symbolic music generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    """Single Transformer block with self-attention and FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: (seq_len, seq_len) boolean mask for causal attention

        Returns:
            (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            attn_mask=causal_mask,
            need_weights=False
        )
        x = x + self.dropout(attn_out)

        # FFN with residual connection
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x


class ChoraleTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        context_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.context_len = context_len

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(context_len, d_model)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (batch_size, seq_len) of token IDs

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Embeddings
        token_emb = self.token_embedding(x)  # (B, T, d_model)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_embedding(pos_ids)  # (1, T, d_model)
        x_emb = token_emb + pos_emb  # (B, T, d_model)

        # Causal mask: tokens can only attend to previous tokens
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        # Pass through transformer blocks
        x_out = x_emb
        for block in self.transformer_blocks:
            x_out = block(x_out, causal_mask=causal_mask)

        # Final layer norm and output projection
        x_out = self.ln_final(x_out)  # (B, T, d_model)
        logits = self.output_proj(x_out)  # (B, T, vocab_size)

        return logits

    def generate(
        self,
        start_tokens: List[int],
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        device: torch.device = None,
    ) -> List[int]:
        """
        Autoregressive generation.

        Args:
            start_tokens: List of initial token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            device: Device to run on

        Returns:
            List of generated token IDs (including start tokens)
        """
        if device is None:
            device = next(self.parameters()).device

        # Start with initial tokens
        tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)  # (1, len(start_tokens))

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get last context_len tokens
                seq_len = tokens.shape[1]
                if seq_len > self.context_len:
                    tokens_in = tokens[:, -self.context_len:]
                else:
                    tokens_in = tokens

                # Forward pass
                logits = self.forward(tokens_in)  # (1, seq_len, vocab_size)
                logits = logits[:, -1, :]  # (1, vocab_size) - last token logits

                # Temperature scaling
                logits = logits / temperature

                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                # Append to sequence
                tokens = torch.cat([tokens, next_token], dim=1)

        return tokens[0].cpu().tolist()
