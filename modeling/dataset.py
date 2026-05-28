"""
ChoraleDataset: PyTorch Dataset for tokenized Bach chorales.
Concatenates all sequences into one stream and chunks with a sliding window.
This ensures every token contributes to training regardless of sequence length,
and gives the model the full context_len look-back at every position.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class ChoraleDataset(Dataset):
    def __init__(self, sequences: List[List[int]], context_len: int = 512, stride: int = 128):
        """
        Initialize dataset.

        Args:
            sequences:   List of encoded token ID lists (one per chorale)
            context_len: Number of tokens the model sees as input (context window)
            stride:      Step size between consecutive windows. Default 128 gives
                         ~4x overlap, producing enough training examples even for
                         short sequences while avoiding the stride=1 redundancy.
        """
        self.context_len = context_len
        self.windows = []

        # Concatenate all sequences into one long token stream.
        # This is the standard GPT pre-training approach: sequence boundaries
        # are preserved by the <START>/<END> tokens already embedded in each
        # sequence, so the model learns to handle them naturally.
        all_tokens: List[int] = []
        for seq in sequences:
            all_tokens.extend(seq)

        # Sliding window over the full stream
        for i in range(0, len(all_tokens) - context_len, stride):
            input_tokens  = all_tokens[i:i + context_len]
            target_tokens = all_tokens[i + 1:i + context_len + 1]
            if len(target_tokens) == context_len:          # skip incomplete final window
                self.windows.append((input_tokens, target_tokens))

    def __len__(self) -> int:
        """Return total number of windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single window.

        Returns:
            (input_tensor, target_tensor) both as LongTensors of shape (context_len,)
        """
        input_tokens, target_tokens = self.windows[idx]
        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        target_tensor = torch.tensor(target_tokens, dtype=torch.long)
        return input_tensor, target_tensor
