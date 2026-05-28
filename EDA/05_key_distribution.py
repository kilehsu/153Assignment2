"""
05_key_distribution.py
Analyze key signatures and major/minor modes.
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
print("KEY AND MODE DISTRIBUTION ANALYSIS")
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

# Extract key signatures
key_counts = Counter()
mode_counts = Counter()

for score in chorales:
    try:
        # Analyze the key
        key = score.analyze('key')
        key_str = str(key)
        key_counts[key_str] += 1

        # Extract mode
        mode = 'Major' if key.mode == 'major' else 'Minor'
        mode_counts[mode] += 1
    except Exception as e:
        # Skip if key analysis fails
        pass

print(f"Successfully analyzed keys for {sum(key_counts.values())} chorales")
print(f"Unique keys found: {len(key_counts)}")

# Plot 1: Key Distribution
fig, ax = plt.subplots(figsize=(14, 6))
sorted_keys = sorted(key_counts.items(), key=lambda x: -x[1])
keys = [k[0] for k in sorted_keys[:20]]  # Top 20 keys
counts = [k[1] for k in sorted_keys[:20]]

colors = sns.color_palette("Set2", len(keys))
bars = ax.bar(range(len(keys)), counts, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Key Signature', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Top 20 Key Signatures in Bach Chorales', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(keys)))
ax.set_xticklabels(keys, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05a_key_distribution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 05a_key_distribution.png")
plt.close(fig)

# Plot 2: Major vs Minor Pie Chart
fig, ax = plt.subplots(figsize=(8, 6))
mode_vals = list(mode_counts.values())
mode_labels = list(mode_counts.keys())
colors_pie = ['#FF6B6B', '#4ECDC4']
explode = (0.05, 0.05)
wedges, texts, autotexts = ax.pie(mode_vals, labels=mode_labels, autopct='%1.1f%%',
                                     colors=colors_pie, explode=explode, startangle=90,
                                     textprops={'fontsize': 12, 'weight': 'bold'})
ax.set_title('Major vs Minor Modes in Bach Chorales', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05b_major_vs_minor.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 05b_major_vs_minor.png")
plt.close(fig)

# Plot 3: Full Key Distribution (bar chart)
fig, ax = plt.subplots(figsize=(16, 8))
sorted_all_keys = sorted(key_counts.items(), key=lambda x: -x[1])
all_keys = [k[0] for k in sorted_all_keys]
all_counts = [k[1] for k in sorted_all_keys]

colors = sns.color_palette("husl", len(all_keys))
bars = ax.bar(range(len(all_keys)), all_counts, color=colors, edgecolor='black', alpha=0.8)
ax.set_xlabel('Key Signature', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('All Key Signatures in Bach Chorales', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(all_keys)))
ax.set_xticklabels(all_keys, rotation=90, fontsize=9)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05c_all_keys_distribution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 05c_all_keys_distribution.png")
plt.close(fig)

# Statistics
print("\n" + "=" * 60)
print("KEY STATISTICS")
print("=" * 60)

print(f"\nMode Distribution:")
for mode, count in mode_counts.items():
    pct = 100 * count / sum(mode_counts.values())
    print(f"  {mode}: {count} ({pct:.1f}%)")

print(f"\nTop 15 Most Common Keys:")
for i, (key, count) in enumerate(sorted_keys[:15], 1):
    pct = 100 * count / sum(key_counts.values())
    print(f"  {i:2d}. {key:10s}: {count:3d} ({pct:5.1f}%)")

print("\nKey Analysis Complete!")
