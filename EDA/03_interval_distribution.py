"""
03_interval_distribution.py
Analyze melodic intervals between consecutive notes for each voice.
"""

import os
from music21 import corpus
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("INTERVAL DISTRIBUTION ANALYSIS")
print("=" * 60)

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

# Extract intervals
all_intervals = []
voice_intervals = [[], [], [], []]  # S/A/T/B

voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

for score in chorales:
    for voice_idx, part in enumerate(score.parts):
        notes_in_voice = []
        for element in part.flatten().notes:
            try:
                midi = element.pitch.midi
                notes_in_voice.append(midi)
            except:
                pass

        # Compute intervals between consecutive notes
        for i in range(len(notes_in_voice) - 1):
            interval = notes_in_voice[i + 1] - notes_in_voice[i]
            all_intervals.append(interval)
            voice_intervals[voice_idx].append(interval)

print(f"Total intervals extracted: {len(all_intervals)}")
print(f"Interval range: {min(all_intervals)} to {max(all_intervals)} semitones")

# Plot 1: Overall Interval Distribution
fig, ax = plt.subplots(figsize=(12, 6))
bins = range(-20, 21)
ax.hist(all_intervals, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Melodic Interval (semitones)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Overall Melodic Interval Distribution (All Voices)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='No change')
ax.grid(axis='y', alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '03a_interval_overall.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 03a_interval_overall.png")
plt.close(fig)

# Plot 2: Interval Distribution by Voice
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for voice_idx, ax in enumerate(axes):
    ax.hist(voice_intervals[voice_idx], bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Melodic Interval (semitones)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'{voice_names[voice_idx]} - Melodic Intervals', fontsize=11, fontweight='bold')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '03b_interval_by_voice.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 03b_interval_by_voice.png")
plt.close(fig)

# Statistics
print("\n" + "=" * 60)
print("INTERVAL STATISTICS")
print("=" * 60)

print(f"\nOverall Intervals:")
print(f"  Total intervals: {len(all_intervals)}")
print(f"  Mean interval: {np.mean(all_intervals):.3f} semitones")
print(f"  Std Dev: {np.std(all_intervals):.3f}")
print(f"  Median: {np.median(all_intervals):.1f}")

# Unison (0 semitones)
unison = sum(1 for i in all_intervals if i == 0)
stepwise = sum(1 for i in all_intervals if abs(i) <= 2)
leap = sum(1 for i in all_intervals if abs(i) > 2)
print(f"\n  Unison (0): {unison} ({100*unison/len(all_intervals):.1f}%)")
print(f"  Stepwise (±1-2): {stepwise} ({100*stepwise/len(all_intervals):.1f}%)")
print(f"  Leaps (>2): {leap} ({100*leap/len(all_intervals):.1f}%)")

print(f"\nPer Voice Intervals:")
for voice_idx, voice_name in enumerate(voice_names):
    intervals = voice_intervals[voice_idx]
    print(f"\n{voice_name}:")
    print(f"  Total intervals: {len(intervals)}")
    print(f"  Mean: {np.mean(intervals):.3f}")
    print(f"  Std Dev: {np.std(intervals):.3f}")
    print(f"  Min: {min(intervals)}, Max: {max(intervals)}")

print("\nInterval Analysis Complete!")
