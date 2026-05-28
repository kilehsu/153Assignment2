"""
Markov chain baseline for Bach chorale generation.
Computes perplexity on validation and test sets.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

from tokenizer import BachTokenizer


def main():
    checkpoint_dir = Path("/Users/kilehsu/153Assignment2/modeling/checkpoints")

    # ========== Step 1: Load tokenizer ==========
    print("Loading tokenizer...")
    tokenizer = BachTokenizer.load(str(checkpoint_dir / "tokenizer.json"))
    print(f"Loaded tokenizer with vocab size: {tokenizer.vocab_size}")

    # ========== Step 2: Load splits ==========
    print("Loading splits...")
    with open(checkpoint_dir / "splits.json", 'r') as f:
        splits = json.load(f)
    train_indices = splits["train"]
    val_indices = splits["val"]
    test_indices = splits["test"]
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")

    # ========== Step 3: Load and encode train chorales ==========
    print("Loading and encoding train chorales...")
    from music21 import corpus

    bach_files = corpus.getComposer('bach')
    score_files = []
    for f in bach_files:
        f_str = str(f).lower()
        if any(ext in f_str for ext in ['.musicxml', '.xml', '.mxl', '.krn']):
            if 'analyses' not in f_str:
                score_files.append(str(f))

    # Load all chorales
    chorales = []
    for filepath in score_files:
        try:
            s = corpus.parse(filepath)
            if len(s.parts) == 4:
                chorales.append(s)
        except:
            continue

    # Encode train set
    train_sequences = []
    for idx in train_indices:
        if idx < len(chorales):
            seq = tokenizer.encode(chorales[idx])
            train_sequences.append(seq)

    print(f"Encoded {len(train_sequences)} training sequences")

    # ========== Step 4: Build bigram transition counts ==========
    print("Building bigram transitions...")
    bigram_counts = defaultdict(lambda: defaultdict(int))

    for seq in train_sequences:
        for i in range(len(seq) - 1):
            prev_token = seq[i]
            next_token = seq[i + 1]
            bigram_counts[prev_token][next_token] += 1

    print(f"Found {len(bigram_counts)} unique previous tokens")

    # ========== Step 5: Compute transition probabilities ==========
    print("Computing transition probabilities...")
    bigram_probs = {}
    for prev_token in bigram_counts:
        total = sum(bigram_counts[prev_token].values())
        bigram_probs[prev_token] = {}
        for next_token in bigram_counts[prev_token]:
            bigram_probs[prev_token][next_token] = bigram_counts[prev_token][next_token] / total

    # ========== Step 6: Save transition matrix ==========
    print("Saving transition matrix...")
    # Convert to JSON-serializable format
    transitions_json = {}
    for prev_token in bigram_probs:
        transitions_json[str(prev_token)] = {
            str(next_token): prob for next_token, prob in bigram_probs[prev_token].items()
        }
    with open(checkpoint_dir / "markov_transitions.json", 'w') as f:
        json.dump(transitions_json, f)

    # ========== Step 7-9: Compute perplexity ==========
    print("\nEvaluating on validation set...")
    val_sequences = []
    for idx in val_indices:
        if idx < len(chorales):
            seq = tokenizer.encode(chorales[idx])
            val_sequences.append(seq)

    epsilon = 1e-8
    val_nll = 0.0
    val_total_tokens = 0

    for seq in val_sequences:
        for i in range(len(seq) - 1):
            prev_token = seq[i]
            next_token = seq[i + 1]
            if prev_token in bigram_probs and next_token in bigram_probs[prev_token]:
                prob = bigram_probs[prev_token][next_token]
            else:
                prob = epsilon
            val_nll += -np.log(prob)
            val_total_tokens += 1

    val_perplexity = np.exp(val_nll / val_total_tokens) if val_total_tokens > 0 else float('inf')
    print(f"Markov Val Perplexity: {val_perplexity:.4f}")

    print("\nEvaluating on test set...")
    test_sequences = []
    for idx in test_indices:
        if idx < len(chorales):
            seq = tokenizer.encode(chorales[idx])
            test_sequences.append(seq)

    test_nll = 0.0
    test_total_tokens = 0

    for seq in test_sequences:
        for i in range(len(seq) - 1):
            prev_token = seq[i]
            next_token = seq[i + 1]
            if prev_token in bigram_probs and next_token in bigram_probs[prev_token]:
                prob = bigram_probs[prev_token][next_token]
            else:
                prob = epsilon
            test_nll += -np.log(prob)
            test_total_tokens += 1

    test_perplexity = np.exp(test_nll / test_total_tokens) if test_total_tokens > 0 else float('inf')
    print(f"Markov Test Perplexity: {test_perplexity:.4f}")

    # ========== Step 10: Save results ==========
    print("\nSaving results...")
    results = {
        "val_perplexity": float(val_perplexity),
        "test_perplexity": float(test_perplexity),
        "val_nll": float(val_nll),
        "test_nll": float(test_nll),
        "val_total_tokens": int(val_total_tokens),
        "test_total_tokens": int(test_total_tokens),
    }
    with open(checkpoint_dir / "markov_results.json", 'w') as f:
        json.dump(results, f)

    print("=" * 50)
    print("MARKOV BASELINE COMPLETE")
    print("=" * 50)
    print(f"Val Perplexity: {val_perplexity:.4f}")
    print(f"Test Perplexity: {test_perplexity:.4f}")


if __name__ == "__main__":
    main()
