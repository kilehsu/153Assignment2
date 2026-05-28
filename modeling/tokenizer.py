"""
BachTokenizer: Converts music21 Score objects to/from token sequences.
Token format: P{midi}_D{16ths} for notes, <START>, <END>, <BAR> for special tokens.
"""

import json
from typing import List, Dict, Tuple
from music21 import note, chord, stream


class BachTokenizer:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self._special_tokens = ["<START>", "<END>", "<BAR>"]

    def build_vocab(self, chorales: List):
        """
        Build vocabulary from a list of music21 Score objects.
        Flattens all 4 voices into one sequence.
        """
        # Add special tokens first
        token_id = 0
        for token in self._special_tokens:
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            token_id += 1

        # Collect all unique tokens from chorales
        unique_tokens = set()
        for score in chorales:
            # Flatten all parts (voices)
            all_notes = []
            for part in score.parts:
                for el in part.flatten().notesAndRests:
                    if isinstance(el, note.Note):
                        midi = el.pitch.midi
                        duration = int(round(el.quarterLength * 4))  # 16ths
                        if duration > 0:
                            token = f"P{midi}_D{duration}"
                            unique_tokens.add(token)
                    elif isinstance(el, chord.Chord):
                        # Take highest pitch
                        highest_pitch = el.sortAscending().pitches[-1]
                        midi = highest_pitch.midi
                        duration = int(round(el.quarterLength * 4))  # 16ths
                        if duration > 0:
                            token = f"P{midi}_D{duration}"
                            unique_tokens.add(token)

        # Add all unique tokens to vocabulary
        for token in sorted(unique_tokens):
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
            token_id += 1

    def encode(self, score) -> List[int]:
        """
        Encode a music21 Score object into a list of token IDs.
        Flattens all 4 voices in sequence: voice 0, then voice 1, etc.
        """
        tokens = [self.token_to_id["<START>"]]

        # Process each part sequentially
        for part in score.parts:
            for el in part.flatten().notesAndRests:
                if isinstance(el, note.Note):
                    midi = el.pitch.midi
                    duration = int(round(el.quarterLength * 4))  # 16ths
                    if duration > 0:
                        token_str = f"P{midi}_D{duration}"
                        if token_str in self.token_to_id:
                            tokens.append(self.token_to_id[token_str])
                elif isinstance(el, chord.Chord):
                    # Take highest pitch
                    highest_pitch = el.sortAscending().pitches[-1]
                    midi = highest_pitch.midi
                    duration = int(round(el.quarterLength * 4))  # 16ths
                    if duration > 0:
                        token_str = f"P{midi}_D{duration}"
                        if token_str in self.token_to_id:
                            tokens.append(self.token_to_id[token_str])

        tokens.append(self.token_to_id["<END>"])
        return tokens

    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Decode a list of token IDs into a list of token strings.
        """
        tokens = []
        for tid in token_ids:
            if tid in self.id_to_token:
                tokens.append(self.id_to_token[tid])
        return tokens

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        vocab_dict = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()}
        }
        with open(path, 'w') as f:
            json.dump(vocab_dict, f)

    @classmethod
    def load(cls, path: str) -> 'BachTokenizer':
        """Load vocabulary from JSON file."""
        tokenizer = cls()
        with open(path, 'r') as f:
            vocab_dict = json.load(f)
        tokenizer.token_to_id = vocab_dict["token_to_id"]
        tokenizer.id_to_token = {int(k): v for k, v in vocab_dict["id_to_token"].items()}
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)
