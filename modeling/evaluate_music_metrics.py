"""
Comprehensive music evaluation module for Bach chorale generation.

Evaluates generated music quality across multiple metrics beyond perplexity:
- Pitch metrics: class distribution, KL divergence, scale consistency, range
- Rhythm metrics: note density, duration distribution, KL divergence
- Voice-leading metrics (v2/v3): smoothness, leap ratio, voice crossing
- Harmony metrics (v2/v3): consonance score, parallel fifths
- Sequence-level metrics: length, early termination, diversity

Compares Transformer, Markov baseline, and real Bach test set chorales.
"""

import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from tokenizer import BachTokenizer
from tokenizer_v2 import BachTokenizerV2
from tokenizer_v3 import BachTokenizerV3
from model import ChoraleTransformer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """Compute KL divergence KL(p || q)."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def get_scale_notes(root: int, scale_type: str = "major") -> set:
    """
    Get notes in a major or minor scale.
    root: MIDI note class (0-11)
    scale_type: "major" or "minor"

    Returns set of pitch classes (0-11) in the scale.
    """
    if scale_type == "major":
        intervals = [0, 2, 4, 5, 7, 9, 11]  # major scale intervals
    else:  # natural minor
        intervals = [0, 2, 3, 5, 7, 8, 10]  # natural minor intervals

    return {(root + i) % 12 for i in intervals}


def scale_consistency(notes: List[int]) -> float:
    """
    Compute scale consistency: fraction of notes fitting best-matching scale.

    Checks all 24 major/minor scales and returns the fraction of notes
    that fit the best-matching scale.
    """
    if not notes:
        return 0.0

    best_fit = 0.0

    # Try all 12 roots
    for root in range(12):
        # Try major
        scale = get_scale_notes(root, "major")
        fit = sum(1 for n in notes if (n % 12) in scale) / len(notes)
        best_fit = max(best_fit, fit)

        # Try minor
        scale = get_scale_notes(root, "minor")
        fit = sum(1 for n in notes if (n % 12) in scale) / len(notes)
        best_fit = max(best_fit, fit)

    return best_fit


def is_consonant_interval(semitones: int) -> bool:
    """Check if interval (mod 12) is consonant in Bach style."""
    interval_mod12 = semitones % 12
    # Consonant intervals: unison, minor/major third, perfect fourth, perfect fifth, minor/major sixth
    consonant_mod12 = {0, 3, 4, 5, 7, 8, 9}
    return interval_mod12 in consonant_mod12


# ============================================================================
# CONSTRAINED GENERATION
# ============================================================================

VOICE_RANGES = [
    (60, 81),   # Soprano
    (53, 74),   # Alto
    (48, 69),   # Tenor
    (40, 62),   # Bass
]

def generate_constrained(model, start_tokens, tokenizer, max_new_tokens=200,
                         temperature=0.9, nucleus_p=0.9, device=None):
    """Constrained generation with voice range masking and nucleus sampling."""
    if device is None:
        device = next(model.parameters()).device

    # Build token info cache
    token_to_info = {}
    for tid, tok_str in tokenizer.id_to_token.items():
        if tok_str in ('<START>', '<END>', '<BAR>', '<SUSTAIN>'):
            token_to_info[tid] = ('special', None, None)
        elif 'REST' in tok_str:
            try:
                v = int(tok_str[1])
                token_to_info[tid] = ('rest', v, None)
            except (ValueError, IndexError):
                token_to_info[tid] = ('unknown', None, None)
        elif tok_str.startswith('V') and 'P' in tok_str and 'D' in tok_str:
            try:
                after_v = tok_str[1:]
                v = int(after_v.split('P')[0])
                rest = after_v.split('P')[1]
                midi_str, dur_str = rest.split('D')
                token_to_info[tid] = ('note', v, int(midi_str))
            except (ValueError, IndexError):
                token_to_info[tid] = ('unknown', None, None)
        else:
            token_to_info[tid] = ('unknown', None, None)

    tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for gen_step in range(max_new_tokens):
            num_generated = tokens.shape[1] - len(start_tokens)
            voice_idx = num_generated % 4
            low_midi, high_midi = VOICE_RANGES[voice_idx]

            seq_len = tokens.shape[1]
            tokens_in = tokens[:, -model.context_len:] if seq_len > model.context_len else tokens

            logits = model.forward(tokens_in)[:, -1, :]
            logits_masked = logits.clone()

            for tid in range(tokenizer.vocab_size):
                tok_type, v, midi = token_to_info.get(tid, ('unknown', None, None))
                if tok_type == 'special':
                    continue
                if tok_type == 'rest':
                    continue
                if tok_type == 'note':
                    if v != voice_idx or midi < low_midi or midi > high_midi:
                        logits_masked[0, tid] = float('-inf')
                else:
                    logits_masked[0, tid] = float('-inf')

            if torch.all(torch.isinf(logits_masked)):
                logits_masked = logits.clone()
                for tid in range(tokenizer.vocab_size):
                    tok_type, _, _ = token_to_info.get(tid, ('unknown', None, None))
                    if tok_type not in ('special', 'rest'):
                        logits_masked[0, tid] = float('-inf')

            logits_masked = logits_masked / temperature
            probs = F.softmax(logits_masked, dim=-1)
            probs_sorted, indices = torch.sort(probs[0], descending=True)
            cumsum = torch.cumsum(probs_sorted, dim=0)
            mask = cumsum <= nucleus_p
            mask[0] = True
            probs_nucleus = probs_sorted.clone()
            probs_nucleus[~mask] = 0
            probs_nucleus = probs_nucleus / probs_nucleus.sum()

            sampled_idx = torch.multinomial(probs_nucleus, num_samples=1)
            next_token_id = indices[sampled_idx.item()]
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token_id.item() == tokenizer.token_to_id.get('<END>', -1):
                break

    return tokens[0].cpu().tolist()


# ============================================================================
# TOKEN DECODING: Version-specific
# ============================================================================

def decode_to_notes_v1(token_ids: List[int], tokenizer: BachTokenizer) -> Tuple[List[Dict], int]:
    """
    Decode v1 tokens to notes: P{midi}_D{dur}.

    Returns (list of notes, total_duration_16ths) where each note is:
    {voice: 0, midi: int, dur: int, onset_16th: int}
    """
    tokens = tokenizer.decode(token_ids)
    notes = []
    onset_16th = 0

    for token in tokens:
        if token in ["<START>", "<END>", "<BAR>"]:
            continue

        # Parse P{midi}_D{dur}
        if token.startswith("P") and "_D" in token:
            try:
                # Split on "_D"
                parts = token[1:].split("_D")  # Remove leading P first
                midi = int(parts[0])
                dur = int(parts[1])
                notes.append({
                    "voice": 0,
                    "midi": midi,
                    "dur": dur,
                    "onset_16th": onset_16th
                })
                onset_16th += dur
            except (ValueError, IndexError):
                pass

    return notes, onset_16th


def decode_to_notes_v2(token_ids: List[int], tokenizer: BachTokenizerV2) -> Tuple[List[Dict], int]:
    """
    Decode v2 tokens with interleaved voices.

    At each harmonic change, 4 tokens (one per voice): V{v}P{midi}D{dur} or V{v}REST.
    Then sustain tokens until next harmonic change.
    Returns (list of notes, total_duration_16ths).
    """
    tokens = tokenizer.decode(token_ids)

    notes = []
    onset_16th = 0

    # Skip START token
    i = 1
    while i < len(tokens) and tokens[i] not in ["<END>"]:
        token = tokens[i]

        if token == "<START>":
            i += 1
            continue

        if token == "<BAR>":
            i += 1
            continue

        if token == "<SUSTAIN>":
            # These are just placeholders, skip them
            onset_16th += 1
            i += 1
            continue

        # Check if this is a V note token: V{v}P{midi}D{dur}
        if token.startswith("V") and "P" in token and "D" in token:
            # Parse V{v}P{midi}D{dur}
            try:
                # Extract voice number
                v_part = token.split("P")[0]  # "V0", "V1", etc.
                voice_num = int(v_part[1])

                # Extract midi and duration: split on D, not _D
                parts = token.split("D")
                midi_str = parts[0].split("P")[1]  # Get number between P and D
                midi = int(midi_str)
                dur = int(parts[1])

                notes.append({
                    "voice": voice_num,
                    "midi": midi,
                    "dur": dur,
                    "onset_16th": onset_16th
                })

                # Skip sustain tokens until next V token (or end of sequence)
                i += 1
                while i < len(tokens) and tokens[i] == "<SUSTAIN>":
                    onset_16th += 1
                    i += 1

                continue
            except (ValueError, IndexError):
                pass

        if token.startswith("V") and token.endswith("REST"):
            # Rest token - skip it
            i += 1
            continue

        i += 1

    return notes, onset_16th


def decode_to_notes_v3(token_ids: List[int], tokenizer: BachTokenizerV3) -> Tuple[List[Dict], int]:
    """
    Decode v3 tokens: C{pitches}_D{dur}.
    Each token is a chord state: C{s}_{a}_{t}_{b}_D{dur}.

    Pitches may be "R" or "None" for rests. Valid pitches become notes.
    Returns (list of notes, total_duration_16ths).
    """
    tokens = tokenizer.decode(token_ids)
    notes = []
    onset_16th = 0

    for token in tokens:
        if token in ["<START>", "<END>", "<BAR>"]:
            continue

        # Parse C{pitches}_D{dur}
        if token.startswith("C") and "_D" in token:
            try:
                # Split on last "_D" to separate pitches from duration
                parts = token.rsplit("_D", 1)
                dur = int(parts[1])

                # Parse chord: C{s}_{a}_{t}_{b}
                pitch_str = parts[0][1:]  # Remove 'C'
                pitch_parts = pitch_str.split("_")

                # Extract pitches for 4 voices
                for voice, pitch_str_part in enumerate(pitch_parts[:4]):
                    if pitch_str_part not in ["R", "None", ""]:
                        try:
                            midi = int(pitch_str_part)
                            notes.append({
                                "voice": voice,
                                "midi": midi,
                                "dur": dur,
                                "onset_16th": onset_16th
                            })
                        except ValueError:
                            pass

                onset_16th += dur
            except (ValueError, IndexError):
                pass

    return notes, onset_16th


def decode_to_notes(token_ids: List[int], tokenizer, version: str) -> Tuple[List[Dict], int]:
    """Dispatch to version-specific decoder."""
    if version == "v1":
        return decode_to_notes_v1(token_ids, tokenizer)
    elif version == "v2":
        return decode_to_notes_v2(token_ids, tokenizer)
    elif version == "v3":
        return decode_to_notes_v3(token_ids, tokenizer)
    elif version == "v2b":
        return decode_to_notes_v2(token_ids, tokenizer)
    elif version == "v2b_constrained":
        return decode_to_notes_v2(token_ids, tokenizer)
    elif version == "v2c":
        return decode_to_notes_v2(token_ids, tokenizer)
    else:
        raise ValueError(f"Unknown version: {version}")


# ============================================================================
# MARKOV GENERATION
# ============================================================================

def generate_markov_samples(n_samples: int, max_length: int, transitions_path: Path) -> List[List[int]]:
    """
    Generate samples from a Markov model (bigram).

    Returns list of token ID sequences.
    """
    with open(transitions_path) as f:
        transitions = json.load(f)

    # Convert string keys back to ints
    transitions_int = {int(k): {int(k2): v for k2, v in v2.items()} for k, v2 in transitions.items()}

    samples = []
    for _ in range(n_samples):
        seq = [0]  # Assume 0 is START token

        for _ in range(max_length):
            prev_token = seq[-1]

            if prev_token not in transitions_int:
                break

            # Sample next token from distribution
            candidates = list(transitions_int[prev_token].keys())
            probs = np.array([transitions_int[prev_token][c] for c in candidates])
            probs /= probs.sum()

            next_token = np.random.choice(candidates, p=probs)
            seq.append(next_token)

            if next_token == 1:  # Assume 1 is END token
                break

        samples.append(seq)

    return samples


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def compute_pitch_metrics(notes_list: List[List[Dict]]) -> Dict:
    """
    Compute pitch-related metrics for a list of samples.

    Args:
        notes_list: List of note lists (one per sample)

    Returns:
        Dict with: pitch_class_histogram, scale_consistency, unique_pitches_per_sample, pitch_range
    """
    all_pitches = []
    all_pitch_classes = []
    scale_consistencies = []
    unique_pitches_per_sample = []
    pitch_ranges = []

    for notes in notes_list:
        if not notes:
            continue

        pitches = [n["midi"] for n in notes]
        all_pitches.extend(pitches)
        all_pitch_classes.extend([p % 12 for p in pitches])

        unique_pitches_per_sample.append(len(set(pitches)))
        pitch_ranges.append(max(pitches) - min(pitches))
        scale_consistencies.append(scale_consistency(pitches))

    # Pitch class histogram (normalized)
    pitch_class_hist = np.zeros(12)
    for pc in all_pitch_classes:
        pitch_class_hist[pc] += 1
    if pitch_class_hist.sum() > 0:
        pitch_class_hist /= pitch_class_hist.sum()

    return {
        "pitch_class_histogram": pitch_class_hist.tolist(),
        "scale_consistency": float(np.mean(scale_consistencies)) if scale_consistencies else 0.0,
        "unique_pitches_per_sample": float(np.mean(unique_pitches_per_sample)) if unique_pitches_per_sample else 0.0,
        "pitch_range": float(np.mean(pitch_ranges)) if pitch_ranges else 0.0,
    }


def compute_rhythm_metrics(notes_list: List[List[Dict]]) -> Dict:
    """Compute rhythm-related metrics."""
    note_densities = []
    avg_durations = []
    all_durations = []

    for notes in notes_list:
        if not notes:
            continue

        # Note density: notes per quarter note
        total_dur_16ths = max((n["onset_16th"] + n["dur"] for n in notes), default=1)
        total_dur_quarters = total_dur_16ths / 4
        if total_dur_quarters > 0:
            note_density = len(notes) / total_dur_quarters
        else:
            note_density = 0.0
        note_densities.append(note_density)

        # Average duration in quarter notes
        durations = [n["dur"] for n in notes]
        all_durations.extend(durations)
        if durations:
            avg_dur = np.mean(durations) / 4
            avg_durations.append(avg_dur)

    # Duration histogram (16th-note bins: 1, 2, 4, 8, 16, 32)
    duration_bins = [1, 2, 4, 8, 16, 32]
    duration_hist = np.zeros(len(duration_bins))
    for dur in all_durations:
        for i, bin_size in enumerate(duration_bins):
            if dur <= bin_size:
                duration_hist[i] += 1
                break
    if duration_hist.sum() > 0:
        duration_hist /= duration_hist.sum()

    return {
        "note_density": float(np.mean(note_densities)) if note_densities else 0.0,
        "avg_note_duration": float(np.mean(avg_durations)) if avg_durations else 0.0,
        "duration_histogram": duration_hist.tolist(),
    }


def compute_voiceleading_metrics(notes_list: List[List[Dict]]) -> Dict:
    """
    Compute voice-leading metrics (for v2/v3 with 4 voices).

    Returns: avg_voice_step, leap_ratio, voice_crossing_ratio (or None if not applicable).
    """
    avg_voice_steps = []
    leap_ratios = []
    voice_crossing_ratios = []

    for notes in notes_list:
        if not notes:
            continue

        # Group notes by voice and sort by onset
        voices = defaultdict(list)
        for note in notes:
            voices[note["voice"]].append(note)

        # Compute voice steps (consecutive notes in same voice)
        voice_steps_all = []
        for voice_id in voices:
            voice_notes = sorted(voices[voice_id], key=lambda x: x["onset_16th"])
            steps = []
            for i in range(len(voice_notes) - 1):
                midi1 = voice_notes[i]["midi"]
                midi2 = voice_notes[i + 1]["midi"]
                step = abs(midi2 - midi1)
                steps.append(step)
            voice_steps_all.extend(steps)

        if voice_steps_all:
            avg_voice_steps.append(np.mean(voice_steps_all))
            leap_ratio = sum(1 for s in voice_steps_all if s > 4) / len(voice_steps_all)
            leap_ratios.append(leap_ratio)

        # Voice crossing ratio: check at each timestep if lower voice has higher pitch than higher voice
        crossings = 0
        total_timesteps = 0

        # Group notes by onset to get chords
        onsets = defaultdict(lambda: {})
        for note in notes:
            onsets[note["onset_16th"]][note["voice"]] = note["midi"]

        if len(onsets) > 0:
            for onset in sorted(onsets.keys()):
                chord = onsets[onset]
                total_timesteps += 1

                # Check all pairs of voices
                for v1 in chord:
                    for v2 in chord:
                        if v1 < v2 and chord[v1] < chord[v2]:
                            # Lower-indexed voice has lower pitch = good
                            pass
                        elif v1 < v2 and chord[v1] > chord[v2]:
                            # Lower-indexed voice has higher pitch = crossing
                            crossings += 1

            if total_timesteps > 0:
                voice_crossing_ratios.append(crossings / total_timesteps)

    return {
        "avg_voice_step": float(np.mean(avg_voice_steps)) if avg_voice_steps else None,
        "leap_ratio": float(np.mean(leap_ratios)) if leap_ratios else None,
        "voice_crossing_ratio": float(np.mean(voice_crossing_ratios)) if voice_crossing_ratios else None,
    }


def compute_harmony_metrics(notes_list: List[List[Dict]]) -> Dict:
    """
    Compute harmony metrics (for v2/v3 with 4 voices).

    Returns: consonance_score, parallel_fifth_ratio.
    """
    consonance_scores = []
    parallel_fifth_ratios = []

    for notes in notes_list:
        if not notes:
            continue

        # Group notes by onset to get chords
        onsets = defaultdict(lambda: {})
        for note in notes:
            onsets[note["onset_16th"]][note["voice"]] = note["midi"]

        if len(onsets) < 1:
            continue

        # Consonance: fraction of simultaneous voice pairs with consonant interval
        consonant_pairs = 0
        total_pairs = 0

        for onset in sorted(onsets.keys()):
            chord = onsets[onset]
            voices_present = sorted(chord.keys())

            for i in range(len(voices_present)):
                for j in range(i + 1, len(voices_present)):
                    v1, v2 = voices_present[i], voices_present[j]
                    interval = chord[v2] - chord[v1]
                    if is_consonant_interval(interval):
                        consonant_pairs += 1
                    total_pairs += 1

        if total_pairs > 0:
            consonance_scores.append(consonant_pairs / total_pairs)

        # Parallel fifths: check consecutive chords
        sorted_onsets = sorted(onsets.keys())
        parallel_fifths = 0
        total_transitions = 0

        for idx in range(len(sorted_onsets) - 1):
            onset1 = sorted_onsets[idx]
            onset2 = sorted_onsets[idx + 1]
            chord1 = onsets[onset1]
            chord2 = onsets[onset2]

            # Check all voice pairs for parallel perfect fifths
            for v1 in chord1:
                for v2 in chord1:
                    if v1 < v2 and v1 in chord2 and v2 in chord2:
                        midi1_start = chord1[v1]
                        midi2_start = chord1[v2]
                        midi1_end = chord2[v1]
                        midi2_end = chord2[v2]

                        interval_start = (midi2_start - midi1_start) % 12
                        interval_end = (midi2_end - midi1_end) % 12

                        # Check for parallel perfect fifths (both 7 semitones = perfect fifth)
                        if interval_start == 7 and interval_end == 7:
                            parallel_fifths += 1
                        total_transitions += 1

        if total_transitions > 0:
            parallel_fifth_ratios.append(parallel_fifths / total_transitions)

    return {
        "consonance_score": float(np.mean(consonance_scores)) if consonance_scores else None,
        "parallel_fifth_ratio": float(np.mean(parallel_fifth_ratios)) if parallel_fifth_ratios else None,
    }


def compute_sequence_metrics(sequences: List[List[int]]) -> Dict:
    """Compute sequence-level metrics."""
    lengths = [len(seq) for seq in sequences]

    # Early termination: fraction ending with token 1 (END) before position 150
    early_ends = sum(1 for seq in sequences if len(seq) < 150 and len(seq) > 0 and seq[-1] == 1)
    early_end_ratio = early_ends / len(sequences) if sequences else 0.0

    # Unique token ratio: mean unique_tokens / total_tokens per sample
    unique_ratios = []
    for seq in sequences:
        if len(seq) > 0:
            unique_tokens = len(set(seq))
            unique_ratios.append(unique_tokens / len(seq))

    return {
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "early_end_ratio": float(early_end_ratio),
        "unique_token_ratio": float(np.mean(unique_ratios)) if unique_ratios else 0.0,
    }


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def main(version: str = "v1"):
    checkpoint_dir = Path("/Users/kilehsu/153Assignment2/modeling/checkpoints")
    images_dir = Path("/Users/kilehsu/153Assignment2/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    # Version-specific config
    if version == "v1":
        tokenizer_path = checkpoint_dir / "tokenizer.json"
        transformer_path = checkpoint_dir / "transformer_best.pt"
        sequences_cache_path = checkpoint_dir / "sequences_cache.pkl"
        splits_path = checkpoint_dir / "splits.json"
        d_model, n_heads, n_layers, context_len = 128, 8, 4, 512
        tokenizer_cls = BachTokenizer
    elif version == "v2":
        tokenizer_path = checkpoint_dir / "v2_tokenizer.json"
        transformer_path = checkpoint_dir / "v2_transformer_best.pt"
        sequences_cache_path = checkpoint_dir / "v2_sequences_cache.pkl"
        splits_path = checkpoint_dir / "v2_splits.json"
        d_model, n_heads, n_layers, context_len = 256, 8, 6, 512
        tokenizer_cls = BachTokenizerV2
    elif version == "v3":
        tokenizer_path = checkpoint_dir / "v3_tokenizer.json"
        transformer_path = checkpoint_dir / "v3_transformer_best.pt"
        sequences_cache_path = checkpoint_dir / "v3_sequences_cache.pkl"
        splits_path = checkpoint_dir / "v3_splits.json"
        d_model, n_heads, n_layers, context_len = 256, 8, 6, 256
        tokenizer_cls = BachTokenizerV3
    elif version == "v2b":
        # Same tokenizer/data as v2, smaller model (d_model=128, 4 layers)
        tokenizer_path = checkpoint_dir / "v2_tokenizer.json"
        transformer_path = checkpoint_dir / "v2b_transformer_best.pt"
        sequences_cache_path = checkpoint_dir / "v2_sequences_cache.pkl"
        splits_path = checkpoint_dir / "v2_splits.json"
        d_model, n_heads, n_layers, context_len = 128, 8, 4, 512
        tokenizer_cls = BachTokenizerV2
    elif version == "v2b_constrained":
        tokenizer_path = checkpoint_dir / "v2_tokenizer.json"
        transformer_path = checkpoint_dir / "v2b_transformer_best.pt"
        sequences_cache_path = checkpoint_dir / "v2_sequences_cache.pkl"
        splits_path = checkpoint_dir / "v2_splits.json"
        d_model, n_heads, n_layers, context_len = 128, 8, 4, 512
        tokenizer_cls = BachTokenizerV2
    elif version == "v2c":
        tokenizer_path = checkpoint_dir / "v2_tokenizer.json"
        transformer_path = checkpoint_dir / "v2c_transformer_best.pt"
        sequences_cache_path = checkpoint_dir / "v2_sequences_cache.pkl"
        splits_path = checkpoint_dir / "v2_splits.json"
        d_model, n_heads, n_layers, context_len = 128, 8, 4, 512
        tokenizer_cls = BachTokenizerV2
    else:
        raise ValueError(f"Unknown version: {version}")

    # Check if checkpoint exists
    if not transformer_path.exists():
        print(f"Checkpoint not found for {version}")
        exit(0)

    # ========== SETUP ==========
    print(f"\n{'='*70}")
    print(f"MUSIC METRICS EVALUATION: {version.upper()}")
    print(f"{'='*70}\n")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    print(f"Loading tokenizer ({version})...")
    tokenizer = tokenizer_cls.load(str(tokenizer_path))
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Load splits and sequences
    print("Loading splits and sequences...")
    with open(splits_path) as f:
        splits = json.load(f)
    with open(sequences_cache_path, 'rb') as f:
        cache = pickle.load(f)
    all_sequences = cache["sequences"]

    test_indices = splits["test"]
    test_sequences = [all_sequences[i] for i in test_indices if i < len(all_sequences)]
    print(f"Test sequences: {len(test_sequences)}")

    # Load Transformer model
    print(f"Loading Transformer model ({version})...")
    model = ChoraleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        context_len=context_len,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(transformer_path, map_location=device, weights_only=True))
    model.to(device)
    print("Model loaded")

    # ========== GENERATE SAMPLES ==========
    print("\n" + "="*70)
    print("GENERATING SAMPLES")
    print("="*70 + "\n")

    start_token_id = tokenizer.token_to_id["<START>"]

    # Generate 20 Transformer samples
    print("Generating 20 samples from Transformer...")
    transformer_samples = []
    model.eval()
    with torch.no_grad():
        for i in range(20):
            if version == "v2b_constrained":
                token_ids = generate_constrained(model, [start_token_id], tokenizer,
                                                 max_new_tokens=200, temperature=0.9,
                                                 nucleus_p=0.9, device=device)
            else:
                token_ids = model.generate([start_token_id], max_new_tokens=200, temperature=1.0, device=device)
            transformer_samples.append(token_ids)
    print(f"Generated {len(transformer_samples)} Transformer samples")

    # Generate 20 Markov samples
    print("Generating 20 samples from Markov...")
    markov_transitions_path = checkpoint_dir / "markov_transitions.json"
    if markov_transitions_path.exists():
        markov_samples = generate_markov_samples(20, 200, markov_transitions_path)
        print(f"Generated {len(markov_samples)} Markov samples")
    else:
        markov_samples = []
        print("Markov transitions file not found, skipping Markov evaluation")

    # Real Bach test set (first 20)
    real_samples = test_sequences[:20]
    print(f"Using {len(real_samples)} real Bach test sequences")

    # ========== DECODE TO NOTES ==========
    print("\n" + "="*70)
    print("DECODING SEQUENCES TO NOTES")
    print("="*70 + "\n")

    transformer_notes = [decode_to_notes(seq, tokenizer, version)[0] for seq in transformer_samples]
    print(f"Decoded Transformer: {len(transformer_notes)} samples")

    markov_notes = []
    if markov_samples:
        markov_notes = [decode_to_notes(seq, tokenizer, version)[0] for seq in markov_samples]
        print(f"Decoded Markov: {len(markov_notes)} samples")

    real_notes = [decode_to_notes(seq, tokenizer, version)[0] for seq in real_samples]
    print(f"Decoded Real Bach: {len(real_notes)} samples")

    # ========== COMPUTE METRICS ==========
    print("\n" + "="*70)
    print("COMPUTING METRICS")
    print("="*70 + "\n")

    metrics_transformer = {}
    metrics_markov = {}
    metrics_real = {}

    # Pitch metrics
    print("Computing pitch metrics...")
    metrics_transformer.update(compute_pitch_metrics(transformer_notes))
    if markov_notes:
        metrics_markov.update(compute_pitch_metrics(markov_notes))
    metrics_real.update(compute_pitch_metrics(real_notes))

    # Rhythm metrics
    print("Computing rhythm metrics...")
    metrics_transformer.update(compute_rhythm_metrics(transformer_notes))
    if markov_notes:
        metrics_markov.update(compute_rhythm_metrics(markov_notes))
    metrics_real.update(compute_rhythm_metrics(real_notes))

    # Voice-leading metrics (v2/v3 and v2b/v2b_constrained)
    if version in ["v2", "v3", "v2b", "v2b_constrained", "v2c"]:
        print("Computing voice-leading metrics...")
        metrics_transformer.update(compute_voiceleading_metrics(transformer_notes))
        if markov_notes:
            metrics_markov.update(compute_voiceleading_metrics(markov_notes))
        metrics_real.update(compute_voiceleading_metrics(real_notes))

        # Harmony metrics
        print("Computing harmony metrics...")
        metrics_transformer.update(compute_harmony_metrics(transformer_notes))
        if markov_notes:
            metrics_markov.update(compute_harmony_metrics(markov_notes))
        metrics_real.update(compute_harmony_metrics(real_notes))
    else:
        # v1: no voice-leading or harmony metrics
        metrics_transformer.update({"avg_voice_step": None, "leap_ratio": None, "voice_crossing_ratio": None})
        metrics_transformer.update({"consonance_score": None, "parallel_fifth_ratio": None})
        if markov_notes:
            metrics_markov.update({"avg_voice_step": None, "leap_ratio": None, "voice_crossing_ratio": None})
            metrics_markov.update({"consonance_score": None, "parallel_fifth_ratio": None})
        metrics_real.update({"avg_voice_step": None, "leap_ratio": None, "voice_crossing_ratio": None})
        metrics_real.update({"consonance_score": None, "parallel_fifth_ratio": None})

    # Sequence metrics
    print("Computing sequence-level metrics...")
    metrics_transformer.update(compute_sequence_metrics(transformer_samples))
    if markov_notes:
        metrics_markov.update(compute_sequence_metrics(markov_samples))
    metrics_real.update(compute_sequence_metrics(real_samples))

    # KL divergences (Transformer vs Real, Markov vs Real)
    print("Computing KL divergences...")
    try:
        transformer_kl_pitch = kl_div(
            metrics_transformer["pitch_class_histogram"],
            metrics_real["pitch_class_histogram"]
        )
        metrics_transformer["pitch_class_kl_div"] = transformer_kl_pitch
    except:
        metrics_transformer["pitch_class_kl_div"] = None

    if markov_notes:
        try:
            markov_kl_pitch = kl_div(
                metrics_markov["pitch_class_histogram"],
                metrics_real["pitch_class_histogram"]
            )
            metrics_markov["pitch_class_kl_div"] = markov_kl_pitch
        except:
            metrics_markov["pitch_class_kl_div"] = None

    metrics_real["pitch_class_kl_div"] = None  # Real is reference

    # Duration KL divergence
    try:
        transformer_kl_dur = kl_div(
            metrics_transformer["duration_histogram"],
            metrics_real["duration_histogram"]
        )
        metrics_transformer["duration_kl_div"] = transformer_kl_dur
    except:
        metrics_transformer["duration_kl_div"] = None

    if markov_notes:
        try:
            markov_kl_dur = kl_div(
                metrics_markov["duration_histogram"],
                metrics_real["duration_histogram"]
            )
            metrics_markov["duration_kl_div"] = markov_kl_dur
        except:
            metrics_markov["duration_kl_div"] = None

    metrics_real["duration_kl_div"] = None

    # ========== COMPARISONS ==========
    print("\nComputing comparison ratios...")

    comparisons = {}

    # Transformer vs Real
    comparisons["transformer_vs_real"] = {
        "pitch_class_kl_div": metrics_transformer.get("pitch_class_kl_div"),
        "duration_kl_div": metrics_transformer.get("duration_kl_div"),
        "scale_consistency_ratio": metrics_transformer.get("scale_consistency", 0) / max(metrics_real.get("scale_consistency", 1), 1e-6) if metrics_real.get("scale_consistency") else None,
        "consonance_ratio": (metrics_transformer.get("consonance_score") or 0.0) / max(metrics_real.get("consonance_score", 1), 1e-6) if (metrics_real.get("consonance_score") and metrics_transformer.get("consonance_score") is not None) else None,
        "avg_voice_step_ratio": (metrics_transformer.get("avg_voice_step") or 0.0) / max(metrics_real.get("avg_voice_step", 1), 1e-6) if (metrics_real.get("avg_voice_step") and metrics_transformer.get("avg_voice_step") is not None) else None,
    }

    # Markov vs Real
    if markov_notes:
        comparisons["markov_vs_real"] = {
            "pitch_class_kl_div": metrics_markov.get("pitch_class_kl_div"),
            "duration_kl_div": metrics_markov.get("duration_kl_div"),
            "scale_consistency_ratio": metrics_markov.get("scale_consistency", 0) / max(metrics_real.get("scale_consistency", 1), 1e-6) if metrics_real.get("scale_consistency") else None,
            "consonance_ratio": (metrics_markov.get("consonance_score") or 0.0) / max(metrics_real.get("consonance_score", 1), 1e-6) if (metrics_real.get("consonance_score") and metrics_markov.get("consonance_score") is not None) else None,
            "avg_voice_step_ratio": (metrics_markov.get("avg_voice_step") or 0.0) / max(metrics_real.get("avg_voice_step", 1), 1e-6) if (metrics_real.get("avg_voice_step") and metrics_markov.get("avg_voice_step") is not None) else None,
        }

    # ========== SAVE RESULTS ==========
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70 + "\n")

    results = {
        "version": version,
        "sources": {
            "transformer": metrics_transformer,
            "markov": metrics_markov,
            "real_bach": metrics_real,
        },
        "comparisons": comparisons,
    }

    output_json = checkpoint_dir / f"{version}_music_metrics.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics to {output_json}")

    # ========== PRINT COMPARISON TABLE ==========
    print("\n" + "="*70)
    print("METRIC COMPARISON TABLE")
    print("="*70 + "\n")

    print(f"{'Metric':<30} {'Transformer':<15} {'Markov':<15} {'Real Bach':<15}")
    print("-" * 75)

    # Scale consistency
    t_sc = metrics_transformer.get("scale_consistency")
    m_sc = metrics_markov.get("scale_consistency") if markov_notes else None
    r_sc = metrics_real.get("scale_consistency")
    t_sc_str = f"{t_sc:.3f}" if t_sc is not None else "N/A"
    m_sc_str = f"{m_sc:.3f}" if m_sc is not None else "N/A"
    r_sc_str = f"{r_sc:.3f}" if r_sc is not None else "N/A"
    print(f"{'Scale consistency':<30} {t_sc_str:<15} {m_sc_str:<15} {r_sc_str:<15}")

    # Note density
    t_nd = metrics_transformer.get("note_density")
    m_nd = metrics_markov.get("note_density") if markov_notes else None
    r_nd = metrics_real.get("note_density")
    t_nd_str = f"{t_nd:.3f}" if t_nd is not None else "N/A"
    m_nd_str = f"{m_nd:.3f}" if m_nd is not None else "N/A"
    r_nd_str = f"{r_nd:.3f}" if r_nd is not None else "N/A"
    print(f"{'Note density (notes/beat)':<30} {t_nd_str:<15} {m_nd_str:<15} {r_nd_str:<15}")

    # Pitch class KL div
    t_kl_p = metrics_transformer.get("pitch_class_kl_div")
    m_kl_p = metrics_markov.get("pitch_class_kl_div") if markov_notes else None
    t_kl_p_str = f"{t_kl_p:.3f}" if t_kl_p is not None else "---"
    m_kl_p_str = f"{m_kl_p:.3f}" if m_kl_p is not None else "---"
    print(f"{'Pitch class KL div':<30} {t_kl_p_str:<15} {m_kl_p_str:<15} {'---':<15}")

    # Duration KL div
    t_kl_d = metrics_transformer.get("duration_kl_div")
    m_kl_d = metrics_markov.get("duration_kl_div") if markov_notes else None
    t_kl_d_str = f"{t_kl_d:.3f}" if t_kl_d is not None else "---"
    m_kl_d_str = f"{m_kl_d:.3f}" if m_kl_d is not None else "---"
    print(f"{'Duration KL div':<30} {t_kl_d_str:<15} {m_kl_d_str:<15} {'---':<15}")

    # Avg voice step (lower is better)
    if version in ["v2", "v3", "v2b", "v2b_constrained", "v2c"]:
        t_avs = metrics_transformer.get("avg_voice_step")
        m_avs = metrics_markov.get("avg_voice_step") if markov_notes else None
        r_avs = metrics_real.get("avg_voice_step")
        t_avs_str = f"{t_avs:.2f}" if t_avs is not None else "N/A"
        m_avs_str = f"{m_avs:.2f}" if m_avs is not None else "N/A"
        r_avs_str = f"{r_avs:.2f}" if r_avs is not None else "N/A"
        print(f"{'Avg voice step (semitones)':<30} {t_avs_str:<15} {m_avs_str:<15} {r_avs_str:<15}")

        # Leap ratio
        t_lr = metrics_transformer.get("leap_ratio")
        m_lr = metrics_markov.get("leap_ratio") if markov_notes else None
        r_lr = metrics_real.get("leap_ratio")
        t_lr_str = f"{t_lr:.3f}" if t_lr is not None else "N/A"
        m_lr_str = f"{m_lr:.3f}" if m_lr is not None else "N/A"
        r_lr_str = f"{r_lr:.3f}" if r_lr is not None else "N/A"
        print(f"{'Leap ratio':<30} {t_lr_str:<15} {m_lr_str:<15} {r_lr_str:<15}")

        # Consonance score
        t_cs = metrics_transformer.get("consonance_score")
        m_cs = metrics_markov.get("consonance_score") if markov_notes else None
        r_cs = metrics_real.get("consonance_score")
        t_cs_str = f"{t_cs:.3f}" if t_cs is not None else "N/A"
        m_cs_str = f"{m_cs:.3f}" if m_cs is not None else "N/A"
        r_cs_str = f"{r_cs:.3f}" if r_cs is not None else "N/A"
        print(f"{'Consonance score':<30} {t_cs_str:<15} {m_cs_str:<15} {r_cs_str:<15}")

    # Unique token ratio
    t_utr = metrics_transformer.get("unique_token_ratio")
    m_utr = metrics_markov.get("unique_token_ratio") if markov_notes else None
    r_utr = metrics_real.get("unique_token_ratio")
    t_utr_str = f"{t_utr:.3f}" if t_utr is not None else "N/A"
    m_utr_str = f"{m_utr:.3f}" if m_utr is not None else "N/A"
    r_utr_str = f"{r_utr:.3f}" if r_utr is not None else "N/A"
    print(f"{'Unique token ratio':<30} {t_utr_str:<15} {m_utr_str:<15} {r_utr_str:<15}")

    # ========== GENERATE PLOTS ==========
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70 + "\n")

    # Plot 1: Pitch class distribution
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(12)
    width = 0.25

    t_pc = metrics_transformer.get("pitch_class_histogram", [0]*12)
    r_pc = metrics_real.get("pitch_class_histogram", [0]*12)

    ax.bar(x - width, t_pc, width, label="Transformer", alpha=0.8)
    if markov_notes:
        m_pc = metrics_markov.get("pitch_class_histogram", [0]*12)
        ax.bar(x, m_pc, width, label="Markov", alpha=0.8)
        ax.bar(x + width, r_pc, width, label="Real Bach", alpha=0.8)
    else:
        ax.bar(x + width/2, r_pc, width, label="Real Bach", alpha=0.8)

    ax.set_xlabel("Pitch Class")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Pitch Class Distribution Comparison ({version.upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels(["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plot_path = images_dir / f"metrics_pitch_class_{version}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Saved pitch class plot to {plot_path}")

    # Plot 2: Duration distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(6)
    width = 0.25

    t_dur = metrics_transformer.get("duration_histogram", [0]*6)
    r_dur = metrics_real.get("duration_histogram", [0]*6)

    ax.bar(x - width, t_dur, width, label="Transformer", alpha=0.8)
    if markov_notes:
        m_dur = metrics_markov.get("duration_histogram", [0]*6)
        ax.bar(x, m_dur, width, label="Markov", alpha=0.8)
        ax.bar(x + width, r_dur, width, label="Real Bach", alpha=0.8)
    else:
        ax.bar(x + width/2, r_dur, width, label="Real Bach", alpha=0.8)

    ax.set_xlabel("Duration (16th-note bins)")
    ax.set_ylabel("Proportion")
    ax.set_title(f"Duration Distribution Comparison ({version.upper()})")
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32"])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plot_path = images_dir / f"metrics_duration_{version}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    plt.close()
    print(f"Saved duration plot to {plot_path}")

    # Plot 3: Summary metrics (normalized bar chart)
    fig, ax = plt.subplots(figsize=(12, 6))

    metrics_to_plot = [
        ("Scale Consistency", metrics_transformer.get("scale_consistency"), metrics_real.get("scale_consistency")),
        ("Note Density", metrics_transformer.get("note_density"), metrics_real.get("note_density")),
        ("Consonance Score", metrics_transformer.get("consonance_score"), metrics_real.get("consonance_score")),
        ("Unique Token Ratio", metrics_transformer.get("unique_token_ratio"), metrics_real.get("unique_token_ratio")),
    ]

    labels = []
    transformer_vals = []
    real_vals = []

    for label, t_val, r_val in metrics_to_plot:
        if t_val is not None and r_val is not None and r_val > 0:
            labels.append(label)
            transformer_vals.append(t_val / r_val)  # Ratio to real
            real_vals.append(1.0)  # Reference

    if labels:
        x = np.arange(len(labels))
        width = 0.35

        ax.barh(x - width/2, transformer_vals, width, label="Transformer / Real", alpha=0.8)
        ax.barh(x + width/2, real_vals, width, label="Real Bach (normalized)", alpha=0.8)

        ax.set_ylabel("Metric")
        ax.set_xlabel("Ratio (Transformer) / Absolute (Real)")
        ax.set_title(f"Summary Metrics Comparison ({version.upper()})")
        ax.set_yticks(x)
        ax.set_yticklabels(labels)
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label="Equal to Real")
        ax.legend()
        ax.grid(axis='x', alpha=0.3)

        plot_path = images_dir / f"metrics_summary_{version}.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=100)
        plt.close()
        print(f"Saved summary plot to {plot_path}")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_json}")
    print(f"Plots saved to: {images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate music metrics for Bach chorale generation")
    parser.add_argument("--version", choices=["v1", "v2", "v3", "v2b", "v2b_constrained", "v2c"], default="v1",
                        help="Model version to evaluate")
    args = parser.parse_args()

    import os
    os.chdir(Path(__file__).parent)

    main(args.version)
