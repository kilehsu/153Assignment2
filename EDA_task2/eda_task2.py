"""
eda_task2.py
Task 2 EDA: Harmonization with prefix-conditioning.
Analyze soprano melody (voice 0) and accompanying voices (alto/tenor/bass).
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from music21 import corpus
from collections import Counter

# Configuration
OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
CHECKPOINT_DIR = "/Users/kilehsu/153Assignment2/modeling/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use seaborn-v0_8-whitegrid style
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 60)
print("TASK 2 EDA: HARMONIZATION (PREFIX-CONDITIONING)")
print("=" * 60)

# Load Bach chorales (limit to first 60 for speed)
print("\nLoading Bach chorales from music21 corpus...")
bach_works = corpus.getComposer('bach')
chorales = []

# Collect chorales with at least 4 parts
for work_path in bach_works[:60]:  # Limit to first 60
    try:
        score = corpus.parse(work_path)
        if hasattr(score, 'parts') and len(score.parts) >= 4:
            chorales.append(score)
    except:
        pass

print(f"Loaded {len(chorales)} chorales with >= 4 parts")

# Load test split from v2_splits.json
splits_path = os.path.join(CHECKPOINT_DIR, "v2_splits.json")
with open(splits_path) as f:
    splits = json.load(f)

test_indices = set(splits["test"])
print(f"Test split size: {len(test_indices)}")

# Load sequences cache to determine test set size in tokenized form
sequences_cache_path = os.path.join(CHECKPOINT_DIR, "v2_sequences_cache.pkl")
with open(sequences_cache_path, 'rb') as f:
    cache = pickle.load(f)
sequences = cache["sequences"]

# Get test sequences
test_sequences = [sequences[i] for i in splits["test"] if i < len(sequences)]
print(f"Test sequences from cache: {len(test_sequences)}")

# Extract voice-wise data
voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
voice_pitches = [[], [], [], []]
voice_note_counts = [[], [], [], []]

# Store pitch classes per bar for harmonic complexity
pitch_classes_per_bar = []

for score_idx, score in enumerate(chorales):
    # Extract note count per voice
    voice_notes_this_piece = [0, 0, 0, 0]

    # Collect all pitch classes for the entire score
    all_pitch_classes = set()

    for voice_idx, part in enumerate(score.parts[:4]):  # Ensure 4 voices
        for element in part.flatten().notes:
            if hasattr(element, 'pitch'):
                try:
                    midi = element.pitch.midi
                    voice_pitches[voice_idx].append(midi)
                    voice_notes_this_piece[voice_idx] += 1

                    # Extract pitch class (0-11)
                    pitch_class = midi % 12
                    all_pitch_classes.add(pitch_class)
                except:
                    pass

    # Approximate harmonic complexity as unique pitch classes per score
    if all_pitch_classes:
        pitch_classes_per_bar.append(len(all_pitch_classes))

    # Store note counts
    for voice_idx in range(4):
        if voice_notes_this_piece[voice_idx] > 0:
            voice_note_counts[voice_idx].append(voice_notes_this_piece[voice_idx])

# Compute soprano pitch statistics
soprano_min = min(voice_pitches[0]) if voice_pitches[0] else 0
soprano_max = max(voice_pitches[0]) if voice_pitches[0] else 0
soprano_mean = np.mean(voice_pitches[0]) if voice_pitches[0] else 0

# Compute mean notes per voice
mean_notes_per_voice = []
for voice_idx in range(4):
    if voice_note_counts[voice_idx]:
        mean_notes_per_voice.append(np.mean(voice_note_counts[voice_idx]))
    else:
        mean_notes_per_voice.append(0)

# Compute average pitch classes per bar
avg_pitch_classes = np.mean(pitch_classes_per_bar) if pitch_classes_per_bar else 0

# Compute soprano-bass interval distribution
soprano_bass_intervals = []
for score in chorales:
    soprano_notes = []
    bass_notes = []

    # Collect notes from soprano (voice 0) and bass (voice 3)
    for element in score.parts[0].flatten().notes:
        try:
            soprano_notes.append(element.pitch.midi)
        except:
            pass

    for element in score.parts[3].flatten().notes:
        try:
            bass_notes.append(element.pitch.midi)
        except:
            pass

    # Pair up notes (assuming roughly aligned timing)
    for i in range(min(len(soprano_notes), len(bass_notes))):
        interval = abs(soprano_notes[i] - bass_notes[i])
        soprano_bass_intervals.append(interval)

# Get top 12 most common intervals
interval_counter = Counter(soprano_bass_intervals)
top_intervals = interval_counter.most_common(12)
top_interval_sizes = [size for size, _ in top_intervals]
top_interval_counts = [count for _, count in top_intervals]

# Get unique chord types (simplified: count of unique pitch combinations per voice)
unique_chord_types = len(set(tuple(sorted([
    int(voice_pitches[0][i] % 12) if i < len(voice_pitches[0]) else -1,
    int(voice_pitches[1][i] % 12) if i < len(voice_pitches[1]) else -1,
    int(voice_pitches[2][i] % 12) if i < len(voice_pitches[2]) else -1,
    int(voice_pitches[3][i] % 12) if i < len(voice_pitches[3]) else -1,
])) for i in range(min(len(voice_pitches[0]), len(voice_pitches[1]), len(voice_pitches[2]), len(voice_pitches[3])))))

# Get top 3 soprano-bass intervals
top_3_intervals = [size for size, _ in top_intervals[:3]]

print("\n" + "=" * 60)
print("TASK 2 ANALYSIS COMPLETE")
print("=" * 60)

# Recompute soprano-bass intervals with proper alignment
soprano_bass_intervals_aligned = []
for score in chorales:
    # Collect all elements with offsets for proper alignment
    soprano_notes_with_offset = []
    bass_notes_with_offset = []

    # Extract soprano notes (voice 0)
    if len(score.parts) > 0:
        for element in score.parts[0].flatten().notesAndRests:
            if hasattr(element, 'pitch'):
                try:
                    soprano_notes_with_offset.append((element.offset, element.pitch.midi))
                except:
                    pass

    # Extract bass notes (voice 3)
    if len(score.parts) > 3:
        for element in score.parts[3].flatten().notesAndRests:
            if hasattr(element, 'pitch'):
                try:
                    bass_notes_with_offset.append((element.offset, element.pitch.midi))
                except:
                    pass

    # Match notes at the same onset (within 0.1 quarter notes)
    for s_offset, s_midi in soprano_notes_with_offset:
        for b_offset, b_midi in bass_notes_with_offset:
            if abs(s_offset - b_offset) < 0.1:
                interval = abs(s_midi - b_midi)
                soprano_bass_intervals_aligned.append(interval)
                break

# Use aligned intervals if available, otherwise fall back to original
if soprano_bass_intervals_aligned:
    soprano_bass_intervals = soprano_bass_intervals_aligned

# Count intervals and filter to 7-28
interval_counter = Counter(soprano_bass_intervals)
filtered_intervals = {k: v for k, v in sorted(interval_counter.items()) if 7 <= k <= 28}

# Create figure with figsize=(13,5), 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# LEFT PANEL: Soprano Pitch Range
ax1.hist(voice_pitches[0], bins=range(50, 86), color='blue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('MIDI Note', fontsize=11)
ax1.set_ylabel('Count', fontsize=11)
ax1.set_title('Soprano Pitch Range\n(Conditioning Input to Harmonizer)', fontsize=12, fontweight='bold')
ax1.set_xlim(50, 85)
ax1.axvline(soprano_mean, color='red', linestyle='--', linewidth=2, label=f'mean={soprano_mean:.1f}')
ax1.text(soprano_mean + 0.5, ax1.get_ylim()[1] * 0.95, f'mean={soprano_mean:.1f}', fontsize=10, color='red')
ax1.grid(axis='y', alpha=0.3)

# RIGHT PANEL: Soprano-Bass Voicing Gap
if filtered_intervals:
    intervals = sorted(filtered_intervals.keys())
    counts = [filtered_intervals[i] for i in intervals]
    x_pos = np.arange(len(intervals))
    ax2.bar(x_pos, counts, color='#E8735A', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Interval (semitones)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Soprano–Bass Voicing Gap\n(interval between melody and bass)', fontsize=12, fontweight='bold')
    # Set x-ticks every 2 semitones
    ax2.set_xticks(x_pos[::2])
    ax2.set_xticklabels(intervals[::2], rotation=45)
    ax2.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'task2_eda.png'), dpi=120, bbox_inches='tight')
print(f"Saved: task2_eda.png")
plt.close(fig)

# Print the required stats block
print(f"\n=== TASK 2 EDA STATS ===")
print(f"Chorales analyzed: {len(chorales)}")
print(f"Test chorales: {len(test_indices)}")
print(f"Soprano MIDI: min={soprano_min} max={soprano_max} mean={soprano_mean:.1f}")
print(f"Avg pitch classes per bar: {avg_pitch_classes:.1f}")
print(f"Top-3 soprano-bass intervals (semitones): {top_3_intervals[0]}, {top_3_intervals[1]}, {top_3_intervals[2]}")
print(f"Unique 4-voice chord types: {unique_chord_types}")
