"""
eda_extra.py
Task 2 EDA: Two additional figures for improved grading.
Figure 1: SATB voice ranges and soprano melodic intervals
Figure 2: Key distribution and chord complexity (voice activity)
"""

import os
import sys
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from music21 import corpus
from collections import Counter

# Add modeling to path for tokenizer
sys.path.insert(0, '/Users/kilehsu/153Assignment2/modeling')

# Configuration
OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
CHECKPOINT_DIR = "/Users/kilehsu/153Assignment2/modeling/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use seaborn-v0_8-whitegrid style
plt.style.use('seaborn-v0_8-whitegrid')

print("=" * 70)
print("TASK 2 EDA: ADDITIONAL FIGURES")
print("=" * 70)

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

# Load splits
splits_path = os.path.join(CHECKPOINT_DIR, "v2_splits.json")
with open(splits_path) as f:
    splits = json.load(f)

test_indices = set(splits["test"])
print(f"Test split size: {len(test_indices)}")

# Load sequences cache
sequences_cache_path = os.path.join(CHECKPOINT_DIR, "v2_sequences_cache.pkl")
with open(sequences_cache_path, 'rb') as f:
    cache = pickle.load(f)
sequences = cache["sequences"]
print(f"Total sequences in cache: {len(sequences)}")

# ============================================================================
# FIGURE 1: SATB Voice Ranges & Soprano Melodic Intervals
# ============================================================================
print("\n" + "=" * 70)
print("Generating Figure 1: Voice Ranges & Soprano Melodic Steps")
print("=" * 70)

# Extract voice pitches across all chorales
voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
voice_pitches = [[], [], [], []]

for score in chorales:
    for voice_idx, part in enumerate(score.parts[:4]):  # Ensure 4 voices
        for element in part.flatten().notes:
            if hasattr(element, 'pitch'):
                try:
                    midi = element.pitch.midi
                    voice_pitches[voice_idx].append(midi)
                except:
                    pass

# Extract soprano melodic intervals
soprano_intervals = []
for score in chorales:
    soprano_notes = []
    # Collect notes from soprano (voice 0) with proper ordering
    for element in score.parts[0].flatten().notes:
        try:
            soprano_notes.append(element.pitch.midi)
        except:
            pass

    # Calculate step sizes between consecutive notes
    for i in range(len(soprano_notes) - 1):
        step_size = abs(soprano_notes[i + 1] - soprano_notes[i])
        soprano_intervals.append(step_size)

print(f"Voice pitch ranges collected:")
for i, voice in enumerate(voice_names):
    if voice_pitches[i]:
        print(f"  {voice}: {min(voice_pitches[i])}-{max(voice_pitches[i])} ({len(voice_pitches[i])} notes)")

print(f"Soprano melodic intervals: {len(soprano_intervals)} transitions")
print(f"  Stepwise (0-2 st): {sum(1 for x in soprano_intervals if x <= 2)}")
print(f"  Leaps (>=3 st): {sum(1 for x in soprano_intervals if x >= 3)}")

# Create Figure 1 with 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# LEFT PANEL: SATB Voice Ranges (horizontal box plot)
voice_colors = {
    'Soprano': '#4C72B0',
    'Alto': '#DD8452',
    'Tenor': '#55A868',
    'Bass': '#C44E52'
}

positions = [3, 2, 1, 0]  # Reverse for bottom-to-top display
box_data = [voice_pitches[3], voice_pitches[2], voice_pitches[1], voice_pitches[0]]  # Reverse order
colors_list = [voice_colors['Bass'], voice_colors['Tenor'], voice_colors['Alto'], voice_colors['Soprano']]

bp = ax1.boxplot(box_data, vert=False, positions=positions, widths=0.6, patch_artist=True,
                  boxprops=dict(linewidth=1.5),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5),
                  medianprops=dict(color='black', linewidth=2))

for patch, color in zip(bp['boxes'], colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax1.set_yticklabels(['Bass', 'Tenor', 'Alto', 'Soprano'])
ax1.set_xlabel('MIDI Pitch', fontsize=11)
ax1.set_xlim(40, 90)
ax1.set_title('SATB Voice Ranges\n(conditioning input vs generation targets)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add annotation
ax1.text(85, 2.2, '← Model generates\n   these 3', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

# RIGHT PANEL: Soprano Melodic Step Sizes with background shading
step_histogram, bins = np.histogram(soprano_intervals, bins=range(0, 13), range=(0, 12))

# Plot histogram
ax2.bar(bins[:-1], step_histogram, width=0.8, color='#4C72B0', alpha=0.8, edgecolor='black')

# Background shading: stepwise (0-2) in light green, leaps (3-12) in light orange
ax2.axvspan(-0.5, 2.5, alpha=0.15, color='green', label='Step (0-2 st)')
ax2.axvspan(2.5, 12.5, alpha=0.15, color='orange', label='Leap (≥3 st)')

ax2.set_xlabel('Semitones', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Soprano Melodic Step Sizes\n(characterizing melody difficulty)', fontsize=12, fontweight='bold')
ax2.set_xlim(-0.5, 12.5)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'task2_eda_voices.png'), dpi=120, bbox_inches='tight')
print(f"\nSaved: task2_eda_voices.png")
plt.close(fig)

# ============================================================================
# FIGURE 2: Key Distribution & Voice Activity
# ============================================================================
print("\n" + "=" * 70)
print("Generating Figure 2: Key Distribution & Voice Activity")
print("=" * 70)

# Get test chorales using indices
test_chorales = []
for idx in sorted(test_indices):
    if idx < len(chorales):
        test_chorales.append(chorales[idx])

print(f"Test chorales collected: {len(test_chorales)}")

# Extract key signatures from test set
key_counts = Counter()
for score in test_chorales:
    try:
        key = score.analyze('key')
        # Format: e.g., "C major", "A minor"
        key_str = str(key)
        key_counts[key_str] += 1
    except:
        key_counts['Unknown'] += 1

print(f"Key signature distribution:")
for key, count in key_counts.most_common(10):
    print(f"  {key}: {count}")

# Extract voice activity (notes sounding per 16th-note position)
# For each chorale, find all unique 16th-note positions and count active voices
voice_activity_distribution = [0, 0, 0, 0]  # counts for 1, 2, 3, 4 voices

for score in chorales:
    # Collect all 16th-note positions across all voices
    all_offsets = set()
    for part in score.parts[:4]:
        for element in part.flatten().notesAndRests:
            # Quantize to 16th notes (0.25 quarter notes)
            quantized_offset = round(element.offset / 0.25) * 0.25
            all_offsets.add(quantized_offset)

    # For each 16th-note position, count how many voices have a note
    for offset in all_offsets:
        voices_active = 0
        for part_idx, part in enumerate(score.parts[:4]):
            has_note = False
            for element in part.flatten().notesAndRests:
                quantized_offset = round(element.offset / 0.25) * 0.25
                if abs(quantized_offset - offset) < 0.01 and element.isNote:
                    has_note = True
                    break
            if has_note:
                voices_active += 1

        if 1 <= voices_active <= 4:
            voice_activity_distribution[voices_active - 1] += 1

print(f"\nVoice activity distribution:")
print(f"  1-voice: {voice_activity_distribution[0]}")
print(f"  2-voice: {voice_activity_distribution[1]}")
print(f"  3-voice: {voice_activity_distribution[2]}")
print(f"  4-voice: {voice_activity_distribution[3]}")

total_positions = sum(voice_activity_distribution)
percentages = [100 * count / total_positions for count in voice_activity_distribution] if total_positions > 0 else [0, 0, 0, 0]

# Create Figure 2 with 2 panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# LEFT PANEL: Key Distribution
# Sort by frequency
sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
key_names = [k[0] for k in sorted_keys]
key_freqs = [k[1] for k in sorted_keys]

# Determine colors: major (blue) vs minor (salmon)
key_colors = []
for key_name in key_names:
    if 'major' in key_name:
        key_colors.append('#4C72B0')  # Blue for major
    elif 'minor' in key_name:
        key_colors.append('#DD8452')  # Salmon for minor
    else:
        key_colors.append('#999999')  # Gray for unknown

# Horizontal bar chart
y_pos = np.arange(len(key_names))
ax1.barh(y_pos, key_freqs, color=key_colors, alpha=0.8, edgecolor='black')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(key_names, fontsize=9)
ax1.set_xlabel('Count', fontsize=11)
ax1.set_title(f'Key Distribution — Test Set ({len(test_chorales)} chorales)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# RIGHT PANEL: Voice Activity (Pie or Bar)
voice_labels = ['1-voice', '2-voice', '3-voice', '4-voice']
colors_activity = ['#d9d9d9', '#9ecae1', '#4292c6', '#084594']

# Bar chart with percentage labels
x_pos = np.arange(len(voice_labels))
bars = ax2.bar(x_pos, voice_activity_distribution, color=colors_activity, alpha=0.8, edgecolor='black')

# Add percentage labels on bars
for i, (bar, pct) in enumerate(zip(bars, percentages)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Count', fontsize=11)
ax2.set_xlabel('Voices Active', fontsize=11)
ax2.set_title('Voice Activity per 16th-note Position', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(voice_labels)
ax2.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'task2_eda_harmony.png'), dpi=120, bbox_inches='tight')
print(f"\nSaved: task2_eda_harmony.png")
plt.close(fig)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TASK 2 EDA SUMMARY")
print("=" * 70)

print(f"\nFigure 1: task2_eda_voices.png")
print(f"  - SATB Voice Ranges: Soprano {min(voice_pitches[0])}-{max(voice_pitches[0])}, "
      f"Alto {min(voice_pitches[1])}-{max(voice_pitches[1])}, "
      f"Tenor {min(voice_pitches[2])}-{max(voice_pitches[2])}, "
      f"Bass {min(voice_pitches[3])}-{max(voice_pitches[3])}")
print(f"  - Soprano Melodic Steps: {sum(1 for x in soprano_intervals if x <= 2)} stepwise, "
      f"{sum(1 for x in soprano_intervals if x >= 3)} leaps")

print(f"\nFigure 2: task2_eda_harmony.png")
print(f"  - Most common key: {key_names[0]} ({key_freqs[0]} chorales)")
print(f"  - 4-voice positions: {percentages[3]:.1f}%")

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)
