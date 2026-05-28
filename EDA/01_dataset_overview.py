"""
01_dataset_overview.py
Load all Bach chorales and provide basic statistics.
"""

import os
import sys
from music21 import corpus
import matplotlib.pyplot as plt
import numpy as np

# Configuration
OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
EDA_DIR = "/Users/kilehsu/153Assignment2/EDA"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

print("=" * 60)
print("DATASET OVERVIEW: Bach Chorales")
print("=" * 60)

# Load all Bach chorales
print("\nLoading Bach chorales from music21 corpus...")
try:
    bach_works = corpus.getComposer('bach')
    print(f"Found {len(bach_works)} works by Bach")
except Exception as e:
    print(f"Error loading Bach works: {e}")
    sys.exit(1)

# Filter for chorales (typically 'bwv' with 4-part structure)
chorales = []
piece_lengths = []  # Number of notes in each chorale
total_notes = 0

for work_path in bach_works:
    try:
        score = corpus.parse(work_path)
        # Check if it's a chorale (usually 4 parts)
        num_parts = len(score.parts) if hasattr(score, 'parts') else 0
        if num_parts == 4:
            chorales.append(score)

            # Count notes in this chorale
            note_count = 0
            for part in score.parts:
                for element in part.flatten().notes:
                    note_count += 1
            piece_lengths.append(note_count)
            total_notes += note_count
    except Exception as e:
        # Skip pieces with parsing errors
        pass

print(f"\nSuccessfully loaded {len(chorales)} 4-part chorales")
print(f"Total notes across all chorales: {total_notes}")
print(f"Average notes per chorale: {total_notes / len(chorales):.1f}" if chorales else "N/A")
print(f"Min notes per chorale: {min(piece_lengths) if piece_lengths else 'N/A'}")
print(f"Max notes per chorale: {max(piece_lengths) if piece_lengths else 'N/A'}")

# Plot: Histogram of piece lengths
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(piece_lengths, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of Notes', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Chorale Lengths (Number of Notes)', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01_piece_length_distribution.png'), dpi=300, bbox_inches='tight')
print(f"\nSaved: 01_piece_length_distribution.png")
plt.close(fig)

# Save summary statistics
summary_file = os.path.join(EDA_DIR, 'dataset_summary.txt')
with open(summary_file, 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("BACH CHORALES DATASET SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total Chorales: {len(chorales)}\n")
    f.write(f"Total Notes: {total_notes}\n")
    f.write(f"Average Notes per Chorale: {total_notes / len(chorales):.1f}\n")
    f.write(f"Min Notes per Chorale: {min(piece_lengths)}\n")
    f.write(f"Max Notes per Chorale: {max(piece_lengths)}\n")
    f.write(f"Median Notes per Chorale: {np.median(piece_lengths):.1f}\n")
    f.write(f"Std Dev Notes per Chorale: {np.std(piece_lengths):.1f}\n")

print(f"Saved: dataset_summary.txt")
print("\nDataset Overview Complete!")
