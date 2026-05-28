"""
Create comparison plots of real Bach chorales vs generated sequences.
Saves plots to images/:
  10_pitch_distribution_comparison.png
  11_interval_distribution_comparison.png
  12_duration_distribution_comparison.png
"""

import sys
import os
import json
import pickle
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '../modeling')
from tokenizer import BachTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

EVAL_RESULTS = os.path.join(PROJECT_ROOT, 'modeling', 'checkpoints', 'eval_results.json')
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'modeling', 'checkpoints', 'tokenizer.json')
SEQUENCES_CACHE = os.path.join(PROJECT_ROOT, 'modeling', 'checkpoints', 'sequences_cache.pkl')
SPLITS_PATH = os.path.join(PROJECT_ROOT, 'modeling', 'checkpoints', 'splits.json')
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_token(token_str):
    """Return (midi, dur_16ths) for a P{midi}_D{dur} token, else None."""
    if not token_str.startswith('P'):
        return None
    try:
        mid_part, dur_part = token_str.split('_')
        return int(mid_part[1:]), int(dur_part[1:])
    except (ValueError, IndexError):
        return None


def extract_stats(token_id_lists, tokenizer):
    """
    Given a list of token-ID sequences, decode and return:
      pitches   : list of all MIDI pitches
      intervals : list of semitone intervals between consecutive pitches
      durations : list of all durations (in 16th notes)
    """
    pitches, intervals, durations = [], [], []
    for token_ids in token_id_lists:
        token_strings = tokenizer.decode(token_ids)
        prev_pitch = None
        for tok in token_strings:
            parsed = parse_token(tok)
            if parsed is None:
                continue
            midi, dur = parsed
            pitches.append(midi)
            durations.append(dur)
            if prev_pitch is not None:
                intervals.append(midi - prev_pitch)
            prev_pitch = midi
    return pitches, intervals, durations


def normalized_bar(values, bins, label, color, ax, alpha=0.7):
    """Plot a normalized bar chart over specified bins."""
    counts = Counter(values)
    total = max(sum(counts.values()), 1)
    heights = [counts.get(b, 0) / total for b in bins]
    ax.bar(bins, heights, label=label, color=color, alpha=alpha, width=0.8)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Load data
    with open(EVAL_RESULTS, 'r') as f:
        eval_data = json.load(f)
    generated_samples = eval_data['generated_samples']

    with open(SEQUENCES_CACHE, 'rb') as f:
        cache = pickle.load(f)
    all_sequences = cache['sequences']

    with open(SPLITS_PATH, 'r') as f:
        splits = json.load(f)
    test_indices = splits['test']

    tokenizer = BachTokenizer.load(TOKENIZER_PATH)

    # Real Bach test sequences
    real_sequences = [all_sequences[i] for i in test_indices]

    print(f"Real Bach test sequences : {len(real_sequences)}")
    print(f"Generated samples        : {len(generated_samples)}")

    real_pitches, real_intervals, real_durations = extract_stats(real_sequences, tokenizer)
    gen_pitches,  gen_intervals,  gen_durations  = extract_stats(generated_samples, tokenizer)

    print(f"Real notes: {len(real_pitches)}, Gen notes: {len(gen_pitches)}")

    # ── Plot 1: Pitch distribution ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    pitch_min = min(min(real_pitches, default=36), min(gen_pitches, default=36))
    pitch_max = max(max(real_pitches, default=84), max(gen_pitches, default=84))
    pitch_bins = list(range(pitch_min, pitch_max + 1))

    normalized_bar(real_pitches, pitch_bins, 'Real Bach (test)', 'steelblue', ax, alpha=0.6)
    normalized_bar(gen_pitches,  pitch_bins, 'Generated',        'darkorange', ax, alpha=0.6)

    ax.set_xlabel('MIDI Pitch', fontsize=13)
    ax.set_ylabel('Normalized Frequency', fontsize=13)
    ax.set_title('Pitch Distribution: Real Bach vs Generated', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(pitch_min - 1, pitch_max + 1)
    plt.tight_layout()
    out = os.path.join(IMAGES_DIR, '10_pitch_distribution_comparison.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Plot 2: Interval distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    interval_bins = list(range(-12, 13))

    # Clip intervals to [-12, 12] for display
    real_iv_clipped = [max(-12, min(12, iv)) for iv in real_intervals]
    gen_iv_clipped  = [max(-12, min(12, iv)) for iv in gen_intervals]

    normalized_bar(real_iv_clipped, interval_bins, 'Real Bach (test)', 'steelblue',  ax, alpha=0.6)
    normalized_bar(gen_iv_clipped,  interval_bins, 'Generated',        'darkorange', ax, alpha=0.6)

    ax.set_xlabel('Interval (semitones)', fontsize=13)
    ax.set_ylabel('Normalized Frequency', fontsize=13)
    ax.set_title('Interval Distribution: Real Bach vs Generated', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(interval_bins)
    plt.tight_layout()
    out = os.path.join(IMAGES_DIR, '11_interval_distribution_comparison.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Plot 3: Duration distribution ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))

    all_durs = real_durations + gen_durations
    dur_bins = sorted(set(all_durs)) if all_durs else list(range(1, 17))

    normalized_bar(real_durations, dur_bins, 'Real Bach (test)', 'steelblue',  ax, alpha=0.6)
    normalized_bar(gen_durations,  dur_bins, 'Generated',        'darkorange', ax, alpha=0.6)

    ax.set_xlabel('Duration (16th notes)', fontsize=13)
    ax.set_ylabel('Normalized Frequency', fontsize=13)
    ax.set_title('Duration Distribution: Real Bach vs Generated', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Label common durations
    dur_labels = {1: '16th', 2: '8th', 4: 'quarter', 8: 'half', 16: 'whole'}
    ax.set_xticks(dur_bins)
    ax.set_xticklabels(
        [f"{b}\n({dur_labels[b]})" if b in dur_labels else str(b) for b in dur_bins],
        fontsize=9
    )
    plt.tight_layout()
    out = os.path.join(IMAGES_DIR, '12_duration_distribution_comparison.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")

    print("\nAll comparison plots saved to images/")


if __name__ == '__main__':
    main()
