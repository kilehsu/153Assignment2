"""
07_voice_ranges.py
Analyze pitch ranges for each voice (Soprano, Alto, Tenor, Bass).
"""

import os
from music21 import corpus
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("VOICE RANGE ANALYSIS")
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

# Extract pitch data per voice
voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
voice_pitches = [[], [], [], []]

for score in chorales:
    for voice_idx, part in enumerate(score.parts):
        for element in part.flatten().notes:
            try:
                midi = element.pitch.midi
                voice_pitches[voice_idx].append(midi)
            except:
                pass

# Compute statistics
voice_stats = {}
for voice_idx, voice_name in enumerate(voice_names):
    pitches = voice_pitches[voice_idx]
    voice_stats[voice_name] = {
        'min': min(pitches),
        'max': max(pitches),
        'mean': np.mean(pitches),
        'median': np.median(pitches),
        'std': np.std(pitches),
        'q1': np.percentile(pitches, 25),
        'q3': np.percentile(pitches, 75),
    }

# Plot 1: Box Plots of Voice Ranges
fig, ax = plt.subplots(figsize=(12, 7))
box_data = [voice_pitches[i] for i in range(4)]
bp = ax.boxplot(box_data, labels=voice_names, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5))

# Color the boxes differently
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('MIDI Note Number', fontsize=12)
ax.set_title('Pitch Range Distribution by Voice', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '07a_voice_ranges_boxplot.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 07a_voice_ranges_boxplot.png")
plt.close(fig)

# Plot 2: Violin Plots for Distribution Shape
fig, ax = plt.subplots(figsize=(12, 7))
parts = ax.violinplot(box_data, positions=range(4), showmeans=True, showmedians=True)

# Style the violin plot
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
for pc, color in zip(parts['bodies'], colors):
    pc.set_facecolor(color)
    pc.set_alpha(0.7)

ax.set_ylabel('MIDI Note Number', fontsize=12)
ax.set_title('Pitch Distribution Shape by Voice (Violin Plot)', fontsize=14, fontweight='bold')
ax.set_xticks(range(4))
ax.set_xticklabels(voice_names)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '07b_voice_ranges_violin.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 07b_voice_ranges_violin.png")
plt.close(fig)

# Plot 3: Range Visualization (min-max bar chart)
fig, ax = plt.subplots(figsize=(12, 6))
mins = [voice_stats[v]['min'] for v in voice_names]
maxs = [voice_stats[v]['max'] for v in voice_names]
means = [voice_stats[v]['mean'] for v in voice_names]

x = np.arange(len(voice_names))
width = 0.25

bars1 = ax.bar(x - width, mins, width, label='Min', color='#FF6B6B', alpha=0.8, edgecolor='black')
bars2 = ax.bar(x, means, width, label='Mean', color='#4ECDC4', alpha=0.8, edgecolor='black')
bars3 = ax.bar(x + width, maxs, width, label='Max', color='#45B7D1', alpha=0.8, edgecolor='black')

ax.set_ylabel('MIDI Note Number', fontsize=12)
ax.set_title('Voice Pitch Statistics (Min, Mean, Max)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(voice_names)
ax.legend()
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '07c_voice_ranges_stats.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 07c_voice_ranges_stats.png")
plt.close(fig)

# Statistics
print("\n" + "=" * 60)
print("VOICE RANGE STATISTICS")
print("=" * 60)

for voice_name in voice_names:
    stats = voice_stats[voice_name]
    print(f"\n{voice_name}:")
    print(f"  Min MIDI:    {stats['min']}")
    print(f"  Max MIDI:    {stats['max']}")
    print(f"  Range:       {stats['max'] - stats['min']} semitones")
    print(f"  Mean MIDI:   {stats['mean']:.2f}")
    print(f"  Median MIDI: {stats['median']:.1f}")
    print(f"  Std Dev:     {stats['std']:.2f}")
    print(f"  Q1 (25%):    {stats['q1']:.1f}")
    print(f"  Q3 (75%):    {stats['q3']:.1f}")

print("\nVoice Range Analysis Complete!")
