"""
Evaluate the trained Transformer model and compare against Markov baseline.
Compute perplexity and generate sample sequences.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from tokenizer import BachTokenizer
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
    print("Loading tokenizer...")
    tokenizer = BachTokenizer.load(str(checkpoint_dir / "tokenizer.json"))
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ========== Step 2: Load splits and cached sequences ==========
    print("Loading splits and cached sequences...")
    with open(checkpoint_dir / "splits.json") as f:
        splits = json.load(f)

    with open(checkpoint_dir / "sequences_cache.pkl", 'rb') as f:
        cache = pickle.load(f)
    sequences = cache["sequences"]

    val_sequences  = [sequences[i] for i in splits["val"]  if i < len(sequences)]
    test_sequences = [sequences[i] for i in splits["test"] if i < len(sequences)]
    print(f"Val sequences: {len(val_sequences)}, Test sequences: {len(test_sequences)}")

    # ========== Step 3: Load best transformer checkpoint ==========
    print("Loading best transformer model...")
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    except Exception:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    model = ChoraleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=8,
        n_layers=4,
        context_len=512,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(checkpoint_dir / "transformer_best.pt", map_location=device, weights_only=True))
    model.to(device)
    print("Model loaded")

    # ========== Step 4-5: Compute perplexity ==========
    print("\nComputing Transformer perplexity on validation set...")
    val_perplexity = compute_perplexity(model, val_sequences, context_len=512, device=device)
    print(f"Transformer Val Perplexity: {val_perplexity:.4f}")

    print("Computing Transformer perplexity on test set...")
    test_perplexity = compute_perplexity(model, test_sequences, context_len=512, device=device)
    print(f"Transformer Test Perplexity: {test_perplexity:.4f}")

    # ========== Step 6: Compare against Markov baseline ==========
    print("\n" + "=" * 50)
    print("COMPARISON WITH MARKOV BASELINE")
    print("=" * 50)

    with open(checkpoint_dir / "markov_results.json") as f:
        markov_results = json.load(f)

    markov_val_ppl  = markov_results["val_perplexity"]
    markov_test_ppl = markov_results["test_perplexity"]

    print(f"\n{'Model':<20} {'Val Perplexity':<20} {'Test Perplexity':<20}")
    print("-" * 60)
    print(f"{'Markov Baseline':<20} {markov_val_ppl:<20.4f} {markov_test_ppl:<20.4f}")
    print(f"{'Transformer':<20} {val_perplexity:<20.4f} {test_perplexity:<20.4f}")
    print("-" * 60)
    improvement_val  = (markov_val_ppl  - val_perplexity)  / markov_val_ppl  * 100
    improvement_test = (markov_test_ppl - test_perplexity) / markov_test_ppl * 100
    print(f"{'Improvement':<20} {improvement_val:>18.2f}% {improvement_test:>18.2f}%")

    # ========== Step 7: Generate 3 sample sequences ==========
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
        "transformer_val_perplexity": float(val_perplexity),
        "transformer_test_perplexity": float(test_perplexity),
        "markov_val_perplexity": float(markov_val_ppl),
        "markov_test_perplexity": float(markov_test_ppl),
        "improvement_val_percent": float(improvement_val),
        "improvement_test_percent": float(improvement_test),
        "generated_samples": generated_samples,
    }

    with open(checkpoint_dir / "eval_results.json", 'w') as f:
        json.dump(eval_results, f)

    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Transformer Val Perplexity:  {val_perplexity:.4f}")
    print(f"Transformer Test Perplexity: {test_perplexity:.4f}")
    print(f"Markov Val Perplexity:       {markov_val_ppl:.4f}")
    print(f"Markov Test Perplexity:      {markov_test_ppl:.4f}")
    print(f"Val improvement:  {improvement_val:.2f}%")
    print(f"Test improvement: {improvement_test:.2f}%")


if __name__ == "__main__":
    main()
