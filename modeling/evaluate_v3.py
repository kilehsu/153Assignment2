"""
Evaluate the trained Transformer V3 model (chord-token encoding).
Compute perplexity and compare against Markov baseline and previous approaches.
"""

import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from tokenizer_v3 import BachTokenizerV3
from model import ChoraleTransformer


def compute_perplexity(model, sequences, context_len=256, device='cpu'):
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
    print("Loading tokenizer (V3)...")
    tokenizer = BachTokenizerV3.load(str(checkpoint_dir / "v3_tokenizer.json"))
    print(f"Vocab size: {tokenizer.vocab_size}")

    # ========== Step 2: Load splits and cached sequences ==========
    print("Loading splits and cached sequences...")
    splits_path = checkpoint_dir / "v3_splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            splits = json.load(f)
    else:
        with open(checkpoint_dir / "splits.json") as f:
            splits = json.load(f)

    with open(checkpoint_dir / "v3_sequences_cache.pkl", 'rb') as f:
        cache = pickle.load(f)
    sequences = cache["sequences"]

    val_sequences  = [sequences[i] for i in splits["val"]  if i < len(sequences)]
    test_sequences = [sequences[i] for i in splits["test"] if i < len(sequences)]
    print(f"Val sequences: {len(val_sequences)}, Test sequences: {len(test_sequences)}")

    # ========== Step 3: Load best transformer checkpoint ==========
    print("Loading best transformer model (V3)...")
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
        context_len=256,
        dropout=0.1,
    )
    model.load_state_dict(torch.load(checkpoint_dir / "v3_transformer_best.pt", map_location=device, weights_only=True))
    model.to(device)
    print("Model loaded")

    # ========== Step 4-5: Compute perplexity ==========
    print("\nComputing Transformer V3 perplexity on validation set...")
    val_perplexity = compute_perplexity(model, val_sequences, context_len=256, device=device)
    print(f"Transformer V3 Val Perplexity: {val_perplexity:.4f}")

    print("Computing Transformer V3 perplexity on test set...")
    test_perplexity = compute_perplexity(model, test_sequences, context_len=256, device=device)
    print(f"Transformer V3 Test Perplexity: {test_perplexity:.4f}")

    # ========== Step 6: Compare against baselines ==========
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS APPROACHES")
    print("=" * 70)

    markov_results = None
    eval_results_v1 = None
    eval_results_v2 = None

    if (checkpoint_dir / "markov_results.json").exists():
        with open(checkpoint_dir / "markov_results.json") as f:
            markov_results = json.load(f)

    if (checkpoint_dir / "eval_results.json").exists():
        with open(checkpoint_dir / "eval_results.json") as f:
            eval_results_v1 = json.load(f)

    if (checkpoint_dir / "v2_eval_results.json").exists():
        with open(checkpoint_dir / "v2_eval_results.json") as f:
            eval_results_v2 = json.load(f)

    print(f"\n{'Model':<25} {'Val Perplexity':<20} {'Test Perplexity':<20}")
    print("-" * 65)

    if markov_results:
        markov_val = markov_results["val_perplexity"]
        markov_test = markov_results["test_perplexity"]
        print(f"{'Markov Baseline':<25} {markov_val:<20.4f} {markov_test:<20.4f}")

    if eval_results_v1:
        v1_val = eval_results_v1.get("transformer_val_perplexity", float('nan'))
        v1_test = eval_results_v1.get("transformer_test_perplexity", float('nan'))
        print(f"{'V1 (Sequential)':<25} {v1_val:<20.4f} {v1_test:<20.4f}")

    if eval_results_v2:
        v2_val = eval_results_v2.get("val_perplexity", float('nan'))
        v2_test = eval_results_v2.get("test_perplexity", float('nan'))
        print(f"{'V2 (Parallel Voices)':<25} {v2_val:<20.4f} {v2_test:<20.4f}")

    print(f"{'V3 (Chord-Token)':<25} {val_perplexity:<20.4f} {test_perplexity:<20.4f}")
    print("-" * 65)

    # Compute improvements if baseline exists
    if markov_results:
        improvement_val = (markov_results["val_perplexity"] - val_perplexity) / markov_results["val_perplexity"] * 100
        improvement_test = (markov_results["test_perplexity"] - test_perplexity) / markov_results["test_perplexity"] * 100
        print(f"{'Improvement vs Markov':<25} {improvement_val:>18.2f}% {improvement_test:>18.2f}%")

    # ========== Step 7: Generate 3 sample sequences ==========
    print("\n" + "=" * 70)
    print("GENERATING SAMPLES")
    print("=" * 70)

    start_token_id = tokenizer.token_to_id["<START>"]
    generated_samples = []

    for sample_idx in range(3):
        print(f"\nGenerating sample {sample_idx + 1}...")
        token_ids = model.generate([start_token_id], max_new_tokens=100, temperature=1.0, device=device)
        generated_samples.append(token_ids)
        token_strings = tokenizer.decode(token_ids)
        print(f"Tokens ({len(token_strings)}): {token_strings[:15]}...")

    # ========== Step 8: Save results ==========
    print("\nSaving evaluation results...")
    v3_eval_results = {
        "val_perplexity": float(val_perplexity),
        "test_perplexity": float(test_perplexity),
        "generated_samples": generated_samples,
    }

    if markov_results:
        v3_eval_results["improvement_val_percent"] = float((markov_results["val_perplexity"] - val_perplexity) / markov_results["val_perplexity"] * 100)
        v3_eval_results["improvement_test_percent"] = float((markov_results["test_perplexity"] - test_perplexity) / markov_results["test_perplexity"] * 100)

    with open(checkpoint_dir / "v3_eval_results.json", 'w') as f:
        json.dump(v3_eval_results, f)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE (V3)")
    print("=" * 70)
    print(f"Transformer V3 Val Perplexity:  {val_perplexity:.4f}")
    print(f"Transformer V3 Test Perplexity: {test_perplexity:.4f}")
    if markov_results:
        print(f"Val improvement vs Markov:       {v3_eval_results.get('improvement_val_percent', 0):.2f}%")
        print(f"Test improvement vs Markov:      {v3_eval_results.get('improvement_test_percent', 0):.2f}%")


if __name__ == "__main__":
    main()
