"""
06_tokenization_preview.py
Implement a simple tokenizer and analyze the vocabulary.
"""

import os
from music21 import corpus
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import seaborn as sns

OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
EDA_DIR = "/Users/kilehsu/153Assignment2/EDA"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

print("=" * 60)
print("TOKENIZATION PREVIEW")
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

# Simple tokenizer: P{midi}_D{duration_in_16ths}
# Duration mapping: 0.125 -> 2 sixteenths, 0.25 -> 4, 0.5 -> 8, 1.0 -> 16, 2.0 -> 32, etc.

def duration_to_16ths(ql):
    """Convert quarter length to number of sixteenths"""
    return int(round(ql * 4))

token_counts = Counter()
all_tokens = []
sequence_lengths = []

special_tokens = {'<START>', '<END>', '<BAR>'}

for score in chorales:
    score_tokens = ['<START>']

    # Get all notes from all 4 voices, sorted by time
    # For simplicity, we'll just concatenate voices and add <BAR> at measure boundaries
    try:
        measure_num = None
        for part_idx, part in enumerate(score.parts):
            for element in part.flatten().notes:
                try:
                    midi = element.pitch.midi
                    ql = element.duration.quarterLength
                    dur_16ths = duration_to_16ths(ql)

                    # Create token
                    token = f"P{midi}_D{dur_16ths}"
                    score_tokens.append(token)
                    token_counts[token] += 1

                    # Track measure changes (simple heuristic)
                    if hasattr(element, 'measureNumber'):
                        if measure_num is None:
                            measure_num = element.measureNumber
                        elif element.measureNumber != measure_num and token not in ['<BAR>']:
                            score_tokens.append('<BAR>')
                            measure_num = element.measureNumber
                except:
                    pass
    except:
        pass

    score_tokens.append('<END>')
    all_tokens.extend(score_tokens)
    sequence_lengths.append(len(score_tokens))

# Add special tokens to vocabulary
for token in special_tokens:
    if token not in token_counts:
        token_counts[token] = 0

vocab_size = len(token_counts)
total_tokens = len(all_tokens)

print(f"\nTokenization Results:")
print(f"  Vocabulary size: {vocab_size}")
print(f"  Total tokens: {total_tokens}")
print(f"  Average tokens per piece: {total_tokens / len(chorales):.1f}")
print(f"  Min tokens per piece: {min(sequence_lengths)}")
print(f"  Max tokens per piece: {max(sequence_lengths)}")
print(f"  Median tokens per piece: {np.median(sequence_lengths):.1f}")

# Plot 1: Token Frequency Distribution (Top 50)
fig, ax = plt.subplots(figsize=(14, 7))
top_tokens = token_counts.most_common(50)
tokens = [t[0] for t in top_tokens]
freqs = [t[1] for t in top_tokens]

colors = sns.color_palette("Set2", 50)
bars = ax.barh(range(len(tokens)), freqs, color=colors, edgecolor='black', alpha=0.8)
ax.set_ylabel('Token', fontsize=11)
ax.set_xlabel('Frequency', fontsize=12)
ax.set_title('Top 50 Most Frequent Tokens', fontsize=14, fontweight='bold')
ax.set_yticks(range(len(tokens)))
ax.set_yticklabels(tokens, fontsize=8)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06a_token_frequency.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 06a_token_frequency.png")
plt.close(fig)

# Plot 2: Token Sequence Length Distribution
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(sequence_lengths, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Tokenized Sequence Lengths', fontsize=14, fontweight='bold')
ax.axvline(np.mean(sequence_lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sequence_lengths):.1f}')
ax.grid(axis='y', alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06b_sequence_length_distribution.png'), dpi=300, bbox_inches='tight')
print(f"Saved: 06b_sequence_length_distribution.png")
plt.close(fig)

# Save vocabulary to file
vocab_file = os.path.join(EDA_DIR, 'vocabulary.txt')
with open(vocab_file, 'w') as f:
    f.write("# Vocabulary for Bach Chorales Tokenization\n")
    f.write(f"# Total vocabulary size: {vocab_size}\n")
    f.write(f"# Total tokens: {total_tokens}\n")
    f.write("# Format: P{MIDI_note}_D{duration_in_16ths}\n\n")

    # Write special tokens first
    for token in sorted(special_tokens):
        f.write(f"{token}\n")

    f.write("\n# Music tokens (sorted by frequency)\n")
    for token, count in token_counts.most_common():
        if token not in special_tokens:
            f.write(f"{token} {count}\n")

print(f"Saved: vocabulary.txt")

# Statistics
print("\n" + "=" * 60)
print("TOKENIZATION STATISTICS")
print("=" * 60)

print(f"\nVocabulary Size: {vocab_size}")
print(f"Total Tokens: {total_tokens}")
print(f"Unique Token Types: {vocab_size}")

print(f"\nSequence Length Statistics:")
print(f"  Mean: {np.mean(sequence_lengths):.1f}")
print(f"  Median: {np.median(sequence_lengths):.1f}")
print(f"  Std Dev: {np.std(sequence_lengths):.1f}")
print(f"  Min: {min(sequence_lengths)}")
print(f"  Max: {max(sequence_lengths)}")

print(f"\nTop 10 Most Frequent Tokens:")
for i, (token, count) in enumerate(token_counts.most_common(10), 1):
    pct = 100 * count / total_tokens
    print(f"  {i:2d}. {token:20s}: {count:7d} ({pct:5.2f}%)")

print("\nTokenization Preview Complete!")
