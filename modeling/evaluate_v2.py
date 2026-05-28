"""
Evaluate the trained V2 Transformer model and compare against Markov baseline and V1.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer


def compute_perplexity(model, sequences, context_len=512, device='cpu'):
    """Compute perplexity using chunked forward passes (one per context_len block)."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    with torch.no_grad():
        for seq in sequences:
            for start in range(0, len(seq) - 1, context_len):
                chunk = seq[start:start + context_len + 1]
                if len(chunk) < 2:
                    continue
                valid_len = len(chunk) - 1

                input_ids = chunk[:-1]
                if len(input_ids) < context_len:
                    input_ids = [0] * (context_len - len(input_ids)) + input_ids

                input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
                logits = model(input_tensor)  # (1, context_len, vocab_size)

                log_probs = F.log_softmax(logits[0, -valid_len:, :], dim=-1)
                targets = torch.tensor(chunk[1:], dtype=torch.long, device=device)
                nll = F.nll_loss(log_probs, targets, reduction='sum')

                total_nll += nll.item()
                total_tokens += valid_len

    return np.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')


def main():
    checkpoint_dir = Path("/Users/kilehsu/153Assignment2/modeling/checkpoints")

    # ========== Step 1: Load tokenizer ==========
    print("Loading tokenizer V2...")
    tokenizer = BachTokenizerV2.load(str(checkpoint_dir / "v2_tokenizer.json"))
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ========== Step 2: Load splits and cached sequences ==========
    print("Loading splits and cached sequences...")

    # Try v2 splits first, fall back to v1
    splits_path = checkpoint_dir / "v2_splits.json"
    if not splits_path.exists():
        splits_path = checkpoint_dir / "splits.json"

    with open(splits_path) as f:
        splits = json.load(f)

    # Try v2 cache first, fall back to v1
    sequences_cache_path = checkpoint_dir / "v2_sequences_cache.pkl"
    if not sequences_cache_path.exists():
        sequences_cache_path = checkpoint_dir / "sequences_cache.pkl"

    with open(sequences_cache_path, 'rb') as f:
        cache = pickle.load(f)
    sequences = cache["sequences"]

    val_sequences = [sequences[i] for i in splits["val"] if i < len(sequences)]
    test_sequences = [sequences[i] for i in splits["test"] if i < len(sequences)]
    print(f"Val sequences: {len(val_sequences)}, Test sequences: {len(test_sequences)}")

    # ========== Step 3: Load best V2 transformer checkpoint ==========
    print("Loading best V2 transformer model...")
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    except Exception:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = ChoraleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        context_len=512,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(checkpoint_dir / "v2_transformer_best.pt", map_location=device, weights_only=True))
    model.to(device)
    print("Model loaded")

    # ========== Step 4-5: Compute perplexity ==========
    print("\nComputing V2 Transformer perplexity on validation set...")
    val_perplexity = compute_perplexity(model, val_sequences, context_len=512, device=device)
    print(f"V2 Transformer Val Perplexity: {val_perplexity:.4f}")

    print("Computing V2 Transformer perplexity on test set...")
    test_perplexity = compute_perplexity(model, test_sequences, context_len=512, device=device)
    print(f"V2 Transformer Test Perplexity: {test_perplexity:.4f}")

    # ========== Step 6: Load Markov baseline results ==========
    print("\n" + "=" * 50)
    print("COMPARISON WITH BASELINES")
    print("=" * 50)

    with open(checkpoint_dir / "markov_results.json") as f:
        markov_results = json.load(f)

    markov_val_ppl = markov_results["val_perplexity"]
    markov_test_ppl = markov_results["test_perplexity"]

    print(f"\n{'Model':<20} {'Val Perplexity':<20} {'Test Perplexity':<20}")
    print("-" * 60)
    print(f"{'Markov Baseline':<20} {markov_val_ppl:<20.4f} {markov_test_ppl:<20.4f}")

    # Try to load V1 results for comparison
    try:
        with open(checkpoint_dir / "eval_results.json") as f:
            v1_results = json.load(f)
        v1_val_ppl = v1_results["transformer_val_perplexity"]
        v1_test_ppl = v1_results["transformer_test_perplexity"]
        print(f"{'V1 Transformer':<20} {v1_val_ppl:<20.4f} {v1_test_ppl:<20.4f}")
    except FileNotFoundError:
        v1_val_ppl = None
        v1_test_ppl = None
        print(f"{'V1 Transformer':<20} {'N/A':<20} {'N/A':<20}")

    print(f"{'V2 Transformer':<20} {val_perplexity:<20.4f} {test_perplexity:<20.4f}")
    print("-" * 60)

    improvement_val_markov = (markov_val_ppl - val_perplexity) / markov_val_ppl * 100
    improvement_test_markov = (markov_test_ppl - test_perplexity) / markov_test_ppl * 100

    print(f"{'Markov Improvement':<20} {improvement_val_markov:>18.2f}% {improvement_test_markov:>18.2f}%")

    if v1_val_ppl is not None:
        improvement_val_v1 = (v1_val_ppl - val_perplexity) / v1_val_ppl * 100
        improvement_test_v1 = (v1_test_ppl - test_perplexity) / v1_test_ppl * 100
        print(f"{'V1 Improvement':<20} {improvement_val_v1:>18.2f}% {improvement_test_v1:>18.2f}%")
    else:
        improvement_val_v1 = None
        improvement_test_v1 = None

    # ========== Step 7: Generate samples ==========
    print("\n" + "=" * 50)
    print("GENERATING SAMPLES")
    print("=" * 50)

    start_token_id = tokenizer.token_to_id["<START>"]
    generated_samples = []

    for sample_idx in range(3):
        print(f"\nGenerating sample {sample_idx + 1}...")
        token_ids = model.generate([start_token_id], max_new_tokens=200, temperature=1.0, device=device)
        generated_samples.append(token_ids)
        token_strings = tokenizer.decode(token_ids)
        print(f"Tokens ({len(token_strings)}): {token_strings[:20]}...")

    # ========== Step 8: Save results ==========
    print("\nSaving evaluation results...")
    eval_results = {
        "val_perplexity": float(val_perplexity),
        "test_perplexity": float(test_perplexity),
        "markov_val_perplexity": float(markov_val_ppl),
        "markov_test_perplexity": float(markov_test_ppl),
        "improvement_val_percent": float(improvement_val_markov),
        "improvement_test_percent": float(improvement_test_markov),
    }

    if v1_val_ppl is not None:
        eval_results["v1_val_perplexity"] = float(v1_val_ppl)
        eval_results["v1_test_perplexity"] = float(v1_test_ppl)
        eval_results["improvement_val_over_v1_percent"] = float(improvement_val_v1)
        eval_results["improvement_test_over_v1_percent"] = float(improvement_test_v1)

    eval_results["generated_samples"] = generated_samples

    with open(checkpoint_dir / "v2_eval_results.json", 'w') as f:
        json.dump(eval_results, f)

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE (V2)")
    print("=" * 50)
    print(f"V2 Transformer Val Perplexity:  {val_perplexity:.4f}")
    print(f"V2 Transformer Test Perplexity: {test_perplexity:.4f}")
    print(f"Markov Val Perplexity:          {markov_val_ppl:.4f}")
    print(f"Markov Test Perplexity:         {markov_test_ppl:.4f}")
    print(f"Val improvement (vs Markov): {improvement_val_markov:.2f}%")
    print(f"Test improvement (vs Markov): {improvement_test_markov:.2f}%")

    if v1_val_ppl is not None:
        print(f"\nV1 Val Perplexity:              {v1_val_ppl:.4f}")
        print(f"Val improvement (vs V1): {improvement_val_v1:.2f}%")


if __name__ == "__main__":
    main()
