"""
BachTokenizerV2: Converts music21 Score objects to/from token sequences with INTERLEAVED voices.

At each 16th-note grid position, emits tokens for all 4 voices in order.
Token format:
- Special: <START>, <END>, <BAR>, <SUSTAIN>
- Note tokens: V{v}P{midi}D{dur} e.g. V0P72D4 = voice 0, MIDI 72, 4 sixteenth notes
- REST tokens: V{v}REST e.g. V1REST
- At each position, exactly 4 tokens (one per voice): note start, SUSTAIN, or REST
"""

import json
from typing import List, Dict
from music21 import note, chord, stream


class BachTokenizerV2:
    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self._special_tokens = ["<START>", "<END>", "<BAR>", "<SUSTAIN>"]

    def _extract_voice_events(self, part, voice_idx):
        """
        Extract note events from a part.
        Returns list of (onset_16th, end_16th, midi_pitch) for each note/chord.
        """
        events = []
        for el in part.flatten().notesAndRests:
            onset = int(round(el.offset * 4))  # Convert to 16th-note grid
            dur = max(1, int(round(el.quarterLength * 4)))
            end = onset + dur
            if isinstance(el, note.Note):
                events.append((onset, end, el.pitch.midi))
            elif isinstance(el, chord.Chord):
                # Take highest pitch for chords
                midi = el.sortAscending().pitches[-1].midi
                events.append((onset, end, midi))
            # Rests: no entry (voice is silent)
        return events

    def build_vocab(self, chorales):
        """Build vocabulary from a list of chorales with interleaved voice structure."""
        token_id = 0

        # Add special tokens first
        for t in self._special_tokens:
            self.token_to_id[t] = token_id
            self.id_to_token[token_id] = t
            token_id += 1

        # Collect all unique tokens from chorales
        unique_tokens = set()
        for score in chorales:
            n_voices = min(4, len(score.parts))
            for v in range(n_voices):
                part = score.parts[v]
                for el in part.flatten().notesAndRests:
                    if isinstance(el, note.Note):
                        midi = el.pitch.midi
                        dur = max(1, int(round(el.quarterLength * 4)))
                        unique_tokens.add(f"V{v}P{midi}D{dur}")
                    elif isinstance(el, chord.Chord):
                        midi = el.sortAscending().pitches[-1].midi
                        dur = max(1, int(round(el.quarterLength * 4)))
                        unique_tokens.add(f"V{v}P{midi}D{dur}")
                    # REST tokens per voice
                    unique_tokens.add(f"V{v}REST")

        # Add all unique tokens to vocabulary
        for t in sorted(unique_tokens):
            self.token_to_id[t] = token_id
            self.id_to_token[token_id] = t
            token_id += 1

    def encode(self, score) -> List[int]:
        """
        Encode a score into interleaved voice tokens.

        At each 16th-note position, emit exactly 4 tokens (one per voice):
        - If a note starts: V{v}P{midi}D{dur}
        - If a note is sustaining: <SUSTAIN>
        - If voice is resting: V{v}REST
        """
        tokens = [self.token_to_id["<START>"]]

        parts = list(score.parts)
        n_voices = min(4, len(parts))

        # Extract events for each voice
        voice_events = []
        for v in range(n_voices):
            voice_events.append(self._extract_voice_events(parts[v], v))

        # Find total duration in 16th notes
        max_time = 0
        for events in voice_events:
            for (onset, end, _) in events:
                max_time = max(max_time, end)

        # At each time step, emit tokens for all 4 voices
        for t in range(max_time):
            for v in range(n_voices):
                # Check if a note starts at this time step
                note_start = None
                sustaining = False

                for (onset, end, midi) in voice_events[v]:
                    if onset == t:
                        note_start = (onset, end, midi)
                        break
                    elif onset < t < end:
                        sustaining = True

                # Emit appropriate token
                if note_start is not None:
                    # Note starts at this position
                    tok_str = f"V{v}P{note_start[2]}D{note_start[1] - note_start[0]}"
                    tid = self.token_to_id.get(tok_str)
                elif sustaining:
                    # Note is sustaining
                    tid = self.token_to_id["<SUSTAIN>"]
                else:
                    # Voice is resting
                    rest_tok = f"V{v}REST"
                    tid = self.token_to_id.get(rest_tok)

                if tid is not None:
                    tokens.append(tid)

        tokens.append(self.token_to_id["<END>"])
        return tokens

    def decode(self, token_ids: List[int]) -> List[str]:
        """Decode token IDs back to token strings."""
        return [self.id_to_token.get(tid, "<UNK>") for tid in token_ids]

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "id_to_token": {int(k): v for k, v in self.id_to_token.items()}
            }, f)

    @classmethod
    def load(cls, path: str) -> 'BachTokenizerV2':
        """Load vocabulary from JSON file."""
        tok = cls()
        with open(path) as f:
            d = json.load(f)
        tok.token_to_id = d["token_to_id"]
        tok.id_to_token = {int(k): v for k, v in d["id_to_token"].items()}
        return tok

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.token_to_id)
