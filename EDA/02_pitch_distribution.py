"""
02_pitch_distribution.py
Analyze pitch distributions across all chorales and per voice.
"""

import os
from music21 import corpus
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("PITCH DISTRIBUTION ANALYSIS")
print("=" * 60)

# Set color palette
sns.set_palette("Set2")

# Load chorales
bach_works = corpus.getComposer('bach')
chorales = []

for work_path in bach_works:
    try:
        score = corpus.parse(work_path)
        if hasattr(score, 'parts') and len(score.parts) == 4:
            chorales.append(score)
    except:
        pass

print(f"Loaded {len(chorales)} chorales")

# Extract pitch information
all_pitches = []  # MIDI note numbers
all_pitch_classes = []  # 0-11 (C, C#, D, etc.)
voice_pitches = [[], [], [], []]  # Separate lists for S/A/T/B
voice_pitch_classes = [[], [], [], []]

for score in chorales:
    for voice_idx, part in enumerate(score.parts):
        for element in part.flatten().notes:
            try:
                midi = element.pitch.midi
                pc = element.pitch.pitchClass  # 0-11

                all_pitches.append(midi)
                all_pitch_classes.append(pc)
                voice_pitches[voice_idx].append(midi)
                voice_pitch_classes[voice_idx].append(pc)
            except:
                pass

print(f"Total notes extracted: {len(all_pitches)}")
print(f"MIDI range: {min(all_pitches)} to {max(all_pitches)}")
print(f"Pitch class distribution: {len(all_pitch_classes)} total")

# Pitch class names
pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Plot 1: Overall Pitch Class Distribution
fig, ax = plt.subplots(figsize=(12, 6))
pitch_class_counts = np.bincount(all_pitch_classes, minlength=12)
colors = sns.color_palette("Set2", 12)
ax.bar(range(12), pitch_class_counts, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Pitch Class', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Overall Pitch Class Distribution (All Voices)', fontsize=14, fontweight='bold')
ax.set_xticks(range(12))
ax.set_xticklabels(pitch_names)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02a_pitch_class_overall.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 02a_pitch_class_overall.png")
plt.close(fig)

# Plot 2: Pitch Class Distribution by Voice
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
axes = axes.flatten()

for voice_idx, ax in enumerate(axes):
    pc_counts = np.bincount(voice_pitch_classes[voice_idx], minlength=12)
    ax.bar(range(12), pc_counts, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Pitch Class', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{voice_names[voice_idx]} - Pitch Class Distribution', fontsize=11, fontweight='bold')
    ax.set_xticks(range(12))
    ax.set_xticklabels(pitch_names, fontsize=9)
    ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02b_pitch_class_by_voice.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 02b_pitch_class_by_voice.png")
plt.close(fig)

# Plot 3: MIDI Note Distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(all_pitches, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('MIDI Note Number', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of MIDI Note Numbers (All Voices)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02c_midi_note_distribution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 02c_midi_note_distribution.png")
plt.close(fig)

# Print statistics
print("\n" + "=" * 60)
print("PITCH DISTRIBUTION STATISTICS")
print("=" * 60)
print(f"\nOverall Pitch Class Distribution:")
for i, name in enumerate(pitch_names):
    count = pitch_class_counts[i]
    pct = 100 * count / len(all_pitch_classes)
    print(f"  {name:3s}: {count:6d} ({pct:5.2f}%)")

print(f"\nVoice Statistics:")
for voice_idx, voice_name in enumerate(voice_names):
    pitches = voice_pitches[voice_idx]
    print(f"\n{voice_name}:")
    print(f"  Notes: {len(pitches)}")
    print(f"  MIDI range: {min(pitches)} to {max(pitches)}")
    print(f"  Mean MIDI: {np.mean(pitches):.2f}")
    print(f"  Std Dev: {np.std(pitches):.2f}")

print("\nPitch Distribution Analysis Complete!")
