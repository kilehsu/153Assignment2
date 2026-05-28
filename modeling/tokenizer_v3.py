"""
BachTokenizerV3: Chord-state tokenizer for Bach chorales.
Token format: C{s}_{a}_{t}_{b}_D{dur} where s/a/t/b are MIDI pitches for soprano/alto/tenor/bass,
dur is duration in 16th notes. One token per harmonic change.
Special tokens: <START>, <END>, <BAR>
"""

import json
from typing import List, Tuple
from music21 import note, chord


class BachTokenizerV3:
    """
    Chord-state tokenizer: one token per harmonic change.
    Each token encodes the complete 4-voice state as C{s}_{a}_{t}_{b}_D{dur}.
    """

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self._special_tokens = ["<START>", "<END>", "<BAR>"]

    def _get_voice_at_time(self, events, t):
        """
        Get the MIDI pitch of a voice at time t.
        events: sorted list of (onset_16th, end_16th, midi).
        Returns MIDI at time t or None (for rests).
        """
        for (onset, end, midi) in events:
            if onset <= t < end:
                return midi
        return None

    def _extract_events(self, part):
        """
        Extract note events from a single voice.
        Returns sorted list of (onset_16th, end_16th, midi).
        """
        events = []
        for el in part.flatten().notesAndRests:
            onset = int(round(el.offset * 4))  # Convert to 16ths
            dur = max(1, int(round(el.quarterLength * 4)))
            end = onset + dur
            if isinstance(el, note.Note):
                events.append((onset, end, el.pitch.midi))
            elif isinstance(el, chord.Chord):
                midi = el.sortAscending().pitches[-1].midi
                events.append((onset, end, midi))
            # rests: no entry (returns None)
        return sorted(events)

    def _score_to_chord_sequence(self, score):
        """
        Convert score to list of (chord_tuple, duration_16ths) pairs.
        chord_tuple is (soprano_midi, alto_midi, tenor_midi, bass_midi) or None values for rests.
        """
        parts = list(score.parts)
        n_voices = min(4, len(parts))

        # Extract events for each voice
        voice_events = [self._extract_events(parts[v]) for v in range(n_voices)]

        # Pad to 4 voices
        while len(voice_events) < 4:
            voice_events.append([])

        # Find total duration
        max_time = 0
        for evs in voice_events:
            for (_, end, _) in evs:
                max_time = max(max_time, end)

        if max_time == 0:
            return []

        # For each 16th note, get the chord state
        chord_seq = []
        prev_state = None
        chord_start = 0

        for t in range(max_time + 1):
            state = tuple(self._get_voice_at_time(voice_events[v], t) for v in range(4))

            if t == max_time:
                # Finalize last chord
                if prev_state is not None:
                    chord_seq.append((prev_state, t - chord_start))
                break

            if state != prev_state:
                if prev_state is not None:
                    chord_seq.append((prev_state, t - chord_start))
                prev_state = state
                chord_start = t

        return chord_seq

    def _state_to_token(self, state, dur):
        """Convert a chord state tuple and duration to a token string."""
        parts = "_".join(str(m) if m is not None else "R" for m in state)
        return f"C{parts}_D{dur}"

    def build_vocab(self, chorales):
        """
        Build vocabulary from a list of music21 Score objects.
        """
        # Add special tokens first
        token_id = 0
        for t in self._special_tokens:
            self.token_to_id[t] = token_id
            self.id_to_token[token_id] = t
            token_id += 1

        # Collect all unique chord tokens
        unique_tokens = set()
        for score in chorales:
            for (state, dur) in self._score_to_chord_sequence(score):
                unique_tokens.add(self._state_to_token(state, dur))

        # Add all unique tokens to vocabulary
        for t in sorted(unique_tokens):
            self.token_to_id[t] = token_id
            self.id_to_token[token_id] = t
            token_id += 1

    def encode(self, score) -> List[int]:
        """
        Encode a music21 Score object into a list of token IDs.
        Each token represents one harmonic state change.
        """
        tokens = [self.token_to_id["<START>"]]
        for (state, dur) in self._score_to_chord_sequence(score):
            tok = self._state_to_token(state, dur)
            if tok in self.token_to_id:
                tokens.append(self.token_to_id[tok])
        tokens.append(self.token_to_id["<END>"])
        return tokens

    def decode(self, token_ids: List[int]) -> List[str]:
        """
        Decode a list of token IDs into a list of token strings.
        """
        return [self.id_to_token.get(tid, "<UNK>") for tid in token_ids]

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        vocab_dict = {
            "token_to_id": self.token_to_id,
            "id_to_token": {int(k): v for k, v in self.id_to_token.items()}
        }
        with open(path, 'w') as f:
            json.dump(vocab_dict, f)

    @classmethod
    def load(cls, path: str) -> 'BachTokenizerV3':
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
