"""
04_duration_analysis.py
Analyze note durations and rhythm patterns.
"""

import os
from music21 import corpus
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("DURATION AND RHYTHM ANALYSIS")
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

# Extract durations
all_durations = []  # In quarter lengths
duration_counts = Counter()

# Also track notes per measure (note density)
note_densities = []

for score in chorales:
    for part in score.parts:
        measure_notes = 0
        for element in part.flatten():
            # Count measures for density
            if hasattr(element, 'measureNumber'):
                # This would be more complex, skip for now
                pass
            try:
                if hasattr(element, 'duration'):
                    duration_ql = element.duration.quarterLength
                    if duration_ql > 0:
                        all_durations.append(duration_ql)
                        # Round to nearest sensible duration value
                        rounded = round(duration_ql * 4) / 4  # To nearest 16th
                        duration_counts[rounded] += 1
            except:
                pass

print(f"Total notes with duration info: {len(all_durations)}")
print(f"Duration range: {min(all_durations):.3f} to {max(all_durations):.1f} quarter lengths")

# Map common durations
duration_labels = {
    0.25: '16th',
    0.5: '8th',
    1.0: 'Quarter',
    1.5: 'Dotted Quarter',
    2.0: 'Half',
    3.0: 'Dotted Half',
    4.0: 'Whole'
}

# Plot 1: Duration Distribution
fig, ax = plt.subplots(figsize=(12, 6))
sorted_durations = sorted(duration_counts.items())
dur_vals = [d[0] for d in sorted_durations]
dur_counts = [d[1] for d in sorted_durations]
dur_labels_plot = [duration_labels.get(d, f'{d}QL') for d in dur_vals]

colors = sns.color_palette("Set2", len(dur_vals))
bars = ax.bar(range(len(dur_vals)), dur_counts, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Note Duration', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Note Durations', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(dur_vals)))
ax.set_xticklabels(dur_labels_plot, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '04a_duration_distribution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 04a_duration_distribution.png")
plt.close(fig)

# Plot 2: Duration Histogram (log scale for better visibility)
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(all_durations, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Note Duration (Quarter Lengths)', fontsize=12)
ax.set_ylabel('Frequency (log scale)', fontsize=12)
ax.set_title('Distribution of Note Durations (Histogram)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '04b_duration_histogram.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 04b_duration_histogram.png")
plt.close(fig)

# Statistics
print("\n" + "=" * 60)
print("DURATION STATISTICS")
print("=" * 60)

print(f"\nOverall Duration Distribution:")
print(f"  Mean duration: {np.mean(all_durations):.3f} quarter lengths")
print(f"  Median duration: {np.median(all_durations):.3f}")
print(f"  Std Dev: {np.std(all_durations):.3f}")

print(f"\nMost Common Durations:")
for duration in sorted(duration_counts.keys()):
    count = duration_counts[duration]
    pct = 100 * count / len(all_durations)
    label = duration_labels.get(duration, f'{duration}')
    print(f"  {label:20s}: {count:6d} ({pct:5.2f}%)")

print("\nDuration Analysis Complete!")
