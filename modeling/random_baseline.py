"""
Random sampling baseline for Bach chorale generation.

This baseline generates random token sequences by uniformly sampling from the vocabulary.
For perplexity: uniform distribution gives perplexity = vocab_size (all tokens equally likely).

For music quality metrics, we generate random sequences and measure their actual quality.
"""

import sys
sys.path.insert(0, ".")

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

from tokenizer_v2 import BachTokenizerV2
from evaluate_music_metrics import (
    decode_to_notes_v2,
    compute_pitch_metrics,
    compute_rhythm_metrics,
    compute_voiceleading_metrics,
    compute_harmony_metrics,
    kl_div,
)


def main():
    checkpoint_dir = Path("modeling/checkpoints")

    # ========== Step 1: Load tokenizer ==========
    print("Loading tokenizer...")
    tokenizer = BachTokenizerV2.load(str(checkpoint_dir / "v2_tokenizer.json"))
    vocab_size = tokenizer.vocab_size
    print(f"Loaded tokenizer with vocab size: {vocab_size}")

    # ========== Step 2: Load splits and sequences ==========
    print("Loading splits...")
    with open(checkpoint_dir / "v2_splits.json", "r") as f:
        splits = json.load(f)

    val_indices = splits["val"]
    test_indices = splits["test"]

    print(f"Val set: {len(val_indices)} samples")
    print(f"Test set: {len(test_indices)} samples")

    # Load cached sequences
    print("Loading sequences...")
    with open(checkpoint_dir / "v2_sequences_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    sequences = cache["sequences"]
    n_chorales = cache["n_chorales"]

    print(f"Loaded {len(sequences)} sequences from {n_chorales} chorales")

    # Extract validation and test sequences
    val_sequences = [sequences[i] for i in val_indices if i < len(sequences)]
    test_sequences = [sequences[i] for i in test_indices if i < len(sequences)]

    print(f"Val sequences: {len(val_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")

    # ========== Step 3: Compute perplexity for random baseline ==========
    # For uniform distribution over vocab_size tokens, perplexity = vocab_size
    # This is because for each position, all tokens have probability 1/vocab_size
    # log-likelihood = -log(1/vocab_size) = log(vocab_size)
    # perplexity = exp(log(vocab_size)) = vocab_size

    val_perplexity = float(vocab_size)
    test_perplexity = float(vocab_size)

    print(f"\nRandom baseline perplexity (uniform distribution):")
    print(f"  Val Perplexity: {val_perplexity:.1f}")
    print(f"  Test Perplexity: {test_perplexity:.1f}")

    # ========== Step 4: Generate random sequences for music metrics ==========
    print("\nGenerating random sequences for music quality evaluation...")
    n_random_samples = 10
    max_new_tokens = 480

    random_sequences = []
    np.random.seed(42)

    for i in range(n_random_samples):
        # Generate random token sequence
        # Start with START token (id=0)
        seq = [0]

        # Generate max_new_tokens random tokens
        for _ in range(max_new_tokens):
            next_token = np.random.randint(0, vocab_size)
            seq.append(next_token)

            # Stop early if we hit END token (id=1)
            if next_token == 1:
                break

        random_sequences.append(seq)

    print(f"Generated {len(random_sequences)} random sequences")

    # ========== Step 5: Decode random sequences to notes ==========
    print("Decoding random sequences to notes...")
    random_notes_list = []

    for seq in random_sequences:
        try:
            notes, total_dur = decode_to_notes_v2(seq, tokenizer)
            if notes:
                random_notes_list.append(notes)
        except Exception as e:
            print(f"Warning: Failed to decode sequence: {e}")
            continue

    print(f"Successfully decoded {len(random_notes_list)} sequences to notes")

    # ========== Step 6: Extract real Bach music from val/test sets ==========
    print("\nExtracting real Bach sequences for reference...")

    real_bach_notes_list = []
    for seq in val_sequences + test_sequences:
        try:
            notes, total_dur = decode_to_notes_v2(seq, tokenizer)
            if notes:
                real_bach_notes_list.append(notes)
        except Exception as e:
            pass

    print(f"Extracted {len(real_bach_notes_list)} real Bach sequences")

    # ========== Step 7: Compute music quality metrics ==========
    print("\nComputing music quality metrics...")

    # Pitch metrics
    pitch_metrics = compute_pitch_metrics(random_notes_list)

    # Rhythm metrics
    rhythm_metrics = compute_rhythm_metrics(random_notes_list)

    # Voice-leading metrics
    voiceleading_metrics = compute_voiceleading_metrics(random_notes_list)

    # Harmony metrics
    harmony_metrics = compute_harmony_metrics(random_notes_list)

    # Real Bach reference for comparison
    real_pitch_metrics = compute_pitch_metrics(real_bach_notes_list)

    # ========== Step 8: Compute KL divergences vs real Bach ==========
    real_pitch_hist = real_pitch_metrics["pitch_class_histogram"]
    random_pitch_hist = pitch_metrics["pitch_class_histogram"]

    pitch_class_kl_div = kl_div(random_pitch_hist, real_pitch_hist)

    # Duration KL divergence
    real_rhythm_metrics = compute_rhythm_metrics(real_bach_notes_list)
    real_duration_hist = real_rhythm_metrics["duration_histogram"]
    random_duration_hist = rhythm_metrics["duration_histogram"]

    duration_kl_div = kl_div(random_duration_hist, real_duration_hist)

    # ========== Step 9: Compile all metrics ==========
    random_metrics = {
        "pitch_class_histogram": pitch_metrics["pitch_class_histogram"],
        "scale_consistency": pitch_metrics["scale_consistency"],
        "unique_pitches_per_sample": pitch_metrics["unique_pitches_per_sample"],
        "pitch_range": pitch_metrics["pitch_range"],
        "note_density": rhythm_metrics["note_density"],
        "avg_note_duration": rhythm_metrics["avg_note_duration"],
        "duration_histogram": rhythm_metrics["duration_histogram"],
        "avg_voice_step": voiceleading_metrics["avg_voice_step"],
        "leap_ratio": voiceleading_metrics["leap_ratio"],
        "voice_crossing_ratio": voiceleading_metrics["voice_crossing_ratio"],
        "consonance_score": harmony_metrics["consonance_score"],
        "parallel_fifth_ratio": harmony_metrics["parallel_fifth_ratio"],
        "pitch_class_kl_div": pitch_class_kl_div,
        "duration_kl_div": duration_kl_div,
    }

    # ========== Step 10: Save results ==========
    print("\nSaving results...")
    results = {
        "val_perplexity": val_perplexity,
        "test_perplexity": test_perplexity,
        "metrics": random_metrics,
        "n_samples": n_random_samples,
    }

    with open(checkpoint_dir / "random_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to modeling/checkpoints/random_baseline_results.json")

    # ========== Step 11: Print summary ==========
    print("\n" + "=" * 70)
    print("RANDOM BASELINE RESULTS")
    print("=" * 70)
    print(f"Val Perplexity: {val_perplexity:.1f}")
    print(f"Test Perplexity: {test_perplexity:.1f}")
    print(f"\nMusic Quality Metrics (from {n_random_samples} random samples):")
    print(f"  Scale Consistency: {random_metrics['scale_consistency']:.4f}")
    print(f"  Note Density: {random_metrics['note_density']:.4f}")
    print(f"  Pitch Class KL Div (vs real Bach): {random_metrics['pitch_class_kl_div']:.4f}")
    print(f"  Duration KL Div (vs real Bach): {random_metrics['duration_kl_div']:.4f}")
    print(f"  Avg Voice Step: {random_metrics['avg_voice_step']:.4f}")
    print(f"  Leap Ratio: {random_metrics['leap_ratio']:.4f}")
    print(f"  Consonance Score: {random_metrics['consonance_score']:.4f}")
    print(f"  Parallel Fifth Ratio: {random_metrics['parallel_fifth_ratio']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
