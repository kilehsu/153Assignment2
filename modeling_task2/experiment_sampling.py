"""
Temperature + top-p sampling experiment for soprano-conditioned harmonization.

Goal: Test whether lower temperature and nucleus (top-p) sampling reduces
KL divergence vs real Bach and improves scale consistency.

Test (temperature, top_p) combinations:
- (1.0, 1.0)  <- baseline, no top-p
- (0.9, 1.0)  <- temperature only
- (0.8, 1.0)  <- lower temperature
- (1.0, 0.95) <- top-p only
- (0.9, 0.95) <- combined
- (0.8, 0.90) <- aggressive combined
"""

import sys
import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / 'modeling'))

from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer

CHECKPOINT_DIR = Path(__file__).parent.parent / 'modeling' / 'checkpoints'

# Major and natural-minor interval patterns (semitones from root)
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def build_scales():
    """Return dict: (root, mode) -> frozenset of pitch classes."""
    scales = {}
    for root in range(12):
        scales[(root, 'major')] = frozenset((root + i) % 12 for i in MAJOR_INTERVALS)
        scales[(root, 'minor')] = frozenset((root + i) % 12 for i in MINOR_INTERVALS)
    return scales


ALL_SCALES = build_scales()


def detect_key(pitch_classes):
    """Pick the major/minor key that fits the most observed pitch classes."""
    if not pitch_classes:
        return None, None, 0.0
    best_key, best_mode, best_score = 0, 'major', 0.0
    for (root, mode), scale_pcs in ALL_SCALES.items():
        score = sum(1 for pc in pitch_classes if pc in scale_pcs) / len(pitch_classes)
        if score > best_score:
            best_score, best_key, best_mode = score, root, mode
    return best_key, best_mode, best_score


def build_token_info(tokenizer):
    """Cache per-token: ('note', voice, midi_pc) | ('special',) | ('rest',) | ('unknown',)"""
    info = {}
    for tid, tok in tokenizer.id_to_token.items():
        if tok in ('<START>', '<END>', '<BAR>', '<SUSTAIN>'):
            info[tid] = ('special',)
        elif tok.endswith('REST'):
            info[tid] = ('rest',)
        elif tok.startswith('V') and 'P' in tok and 'D' in tok:
            try:
                after_v = tok[1:]
                v = int(after_v.split('P')[0])
                midi = int(after_v.split('P')[1].split('D')[0])
                info[tid] = ('note', v, midi % 12)   # store pitch class
            except (ValueError, IndexError):
                info[tid] = ('unknown',)
        else:
            info[tid] = ('unknown',)
    return info


def extract_soprano_tokens(sequence, tokenizer):
    """
    Extract soprano tokens from a full interleaved sequence.
    At each 16th-note position: [v0_token, v1_token, v2_token, v3_token]

    Return list of soprano (voice 0) token IDs, skipping special tokens.
    """
    decoded = tokenizer.decode(sequence)
    soprano_tokens = []

    # Position 0 is START, then we have groups of 4 tokens
    # Position 1 (in group 0) is voice 0, position 2 is voice 1, etc.
    # At next group: positions 5, 6, 7, 8 -> voice 0 is at 5
    # Pattern: voice 0 is at positions 1, 5, 9, 13, ... = 1 + 4*k

    i = 1  # Start after START token
    while i < len(decoded):
        tok = decoded[i]
        if tok in ('<START>', '<END>', '<BAR>'):
            i += 1
            continue
        # We're at a voice 0 position (position 1, 5, 9, ...)
        soprano_tokens.append(sequence[i])
        i += 4  # Next voice 0 is 4 positions away

    return soprano_tokens


def top_p_sample(logits, p=0.95):
    """
    Nucleus (top-p) sampling: sample from the smallest set of tokens
    whose cumulative probability exceeds p.

    Args:
        logits: (vocab_size,) tensor
        p: float, cumulative probability threshold

    Returns:
        sampled token index as (1,) tensor
    """
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumulative prob exceeds p
    sorted_probs[cumulative - sorted_probs > p] = 0.0
    sorted_probs /= sorted_probs.sum()

    sampled_idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx[sampled_idx]  # Returns shape (1,)


def harmonize(model, soprano_tokens, tokenizer, temperature=1.0, top_p=1.0, device=None,
              chromatic_penalty=3.0):
    """
    Autoregressively harmonize given soprano tokens.

    At each step:
    - Determine which voice position we're at (step % 4)
    - If voice 0 and we have a soprano token: force it
    - Otherwise: sample from model with temperature and top_p
    - Apply key detection + chromatic penalty after 40 tokens
    - Stop when all soprano consumed OR model emits END OR 600 steps

    Args:
        model: ChoraleTransformer
        soprano_tokens: list of token IDs
        tokenizer: BachTokenizerV2
        temperature: float, sampling temperature
        top_p: float, nucleus sampling threshold (1.0 = disabled)
        device: torch device
        chromatic_penalty: float, penalty for out-of-key tokens

    Returns:
        list of token IDs
    """
    if device is None:
        device = next(model.parameters()).device

    start_id   = tokenizer.token_to_id.get('<START>', 0)
    end_id     = tokenizer.token_to_id.get('<END>', -1)
    sustain_id = tokenizer.token_to_id.get('<SUSTAIN>', -1)

    tokens = torch.tensor([[start_id]], dtype=torch.long, device=device)
    soprano_idx = 0  # How many soprano tokens we've used
    detected_key = None
    detected_mode = None
    key_pcs = None
    token_info = build_token_info(tokenizer)

    model.eval()
    with torch.no_grad():
        for step in range(600):
            num_generated = tokens.shape[1] - 1  # Exclude START
            current_voice = num_generated % 4  # Which voice we're predicting

            # Key detection after warmup
            if num_generated == 40 and detected_key is None:
                pitch_classes = []
                for tid in tokens[0].tolist():
                    i = token_info.get(tid, ('unknown',))
                    if i[0] == 'note':
                        pitch_classes.append(i[2])
                detected_key, detected_mode, fit = detect_key(pitch_classes)
                key_pcs = ALL_SCALES[(detected_key, detected_mode)]
                # Silently detect key (no print for experiments)

            tokens_in = tokens[:, -model.context_len:]
            logits = model.forward(tokens_in)[:, -1, :]

            # If voice 0 and we have a soprano token: force it
            if current_voice == 0 and soprano_idx < len(soprano_tokens):
                next_token = torch.tensor([[soprano_tokens[soprano_idx]]],
                                        dtype=torch.long, device=device)
                soprano_idx += 1
            else:
                # Suppress END until we've used all soprano tokens
                if soprano_idx < len(soprano_tokens):
                    logits[0, end_id] = float('-inf')

                # Chromatic penalty
                if key_pcs is not None:
                    for tid, i in token_info.items():
                        if i[0] == 'note' and i[2] not in key_pcs:
                            logits[0, tid] -= chromatic_penalty

                # Apply temperature
                logits = logits / temperature

                # Sample with top-p if p < 1.0
                if top_p < 1.0:
                    sampled = top_p_sample(logits[0], p=top_p)  # shape (1,)
                    next_token = sampled.view(1, 1)  # shape (1, 1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # shape (1, 1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == end_id:
                break

    return tokens[0].cpu().tolist()


def compute_pitch_class_distribution(token_ids, token_info):
    """
    Compute pitch class distribution from token sequence.

    Returns:
        dict mapping pitch class (0-11) to count
    """
    pitch_classes = defaultdict(int)

    for tid in token_ids:
        info = token_info.get(tid, ('unknown',))
        if info[0] == 'note':
            pitch_classes[info[2]] += 1

    # Normalize to probability distribution
    total = sum(pitch_classes.values())
    if total == 0:
        return {i: 0.0 for i in range(12)}

    return {i: pitch_classes[i] / total for i in range(12)}


def compute_kl_divergence(p, q):
    """
    Compute KL divergence D(p || q) where p and q are dicts mapping pitch class to probability.
    Uses log base 2 for interpretability.

    KL(p || q) = sum_i p(i) * log2(p(i) / q(i))

    If p(i) > 0 but q(i) = 0, smooth with small epsilon to avoid infinity.
    If p(i) = 0, contributes 0.
    """
    epsilon = 1e-10  # Smoothing to avoid log(0)
    kl = 0.0
    for i in range(12):
        p_i = p.get(i, 0.0)
        q_i = q.get(i, 0.0)

        if p_i > 0:
            # Smooth q_i to avoid division by zero or log(0)
            q_i_smooth = max(q_i, epsilon)
            kl += p_i * np.log2(p_i / q_i_smooth)

    return kl


def compute_scale_consistency(token_ids, token_info):
    """
    Compute scale consistency: percentage of note tokens that fit in detected key.

    Returns:
        float in [0, 100]
    """
    pitch_classes = []
    for tid in token_ids:
        info = token_info.get(tid, ('unknown',))
        if info[0] == 'note':
            pitch_classes.append(info[2])

    if not pitch_classes:
        return 0.0

    detected_key, detected_mode, fit = detect_key(pitch_classes)
    if detected_key is None:
        return 0.0

    key_pcs = ALL_SCALES[(detected_key, detected_mode)]
    consistent = sum(1 for pc in pitch_classes if pc in key_pcs)

    return 100.0 * consistent / len(pitch_classes)


def main():
    print("="*70)
    print("Temperature + Top-P Sampling Experiment")
    print("="*70)

    # Load tokenizer
    tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
    print(f"Tokenizer: vocab size {tokenizer.vocab_size}")

    # Load model (use correct architecture from checkpoint: d_model=256, n_layers=6)
    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size,
                               d_model=256, n_heads=8, n_layers=6,
                               context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'v2_transformer_best.pt',
                                     map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load test sequences
    with open(CHECKPOINT_DIR / 'v2_sequences_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    sequences = cache['sequences']
    print(f"Loaded {len(sequences)} sequences")

    # Load splits
    with open(CHECKPOINT_DIR / 'v2_splits.json') as f:
        splits = json.load(f)
    test_indices = splits['test']
    print(f"Test split: {len(test_indices)} sequences\n")

    # Extract real pitch class distribution from test sequences
    token_info = build_token_info(tokenizer)
    real_distributions = []
    for test_idx in test_indices[:5]:
        sequence = sequences[test_idx]
        real_dist = compute_pitch_class_distribution(sequence, token_info)
        real_distributions.append(real_dist)

    # Average real Bach distribution
    real_avg = defaultdict(float)
    for dist in real_distributions:
        for pc in range(12):
            real_avg[pc] += dist[pc] / 5.0

    print(f"Real Bach pitch class distribution (averaged over 5 test sequences):")
    print(f"  {dict(real_avg)}\n")

    # Test configurations
    configs = [
        (1.0, 1.0,  "1.0", "1.00"),
        (0.9, 1.0,  "0.9", "1.00"),
        (0.8, 1.0,  "0.8", "1.00"),
        (1.0, 0.95, "1.0", "0.95"),
        (0.9, 0.95, "0.9", "0.95"),
        (0.8, 0.90, "0.8", "0.90"),
    ]

    results = []

    # Run experiments - use only 2 test sequences for speed (still representative)
    for temp, top_p, temp_str, top_p_str in configs:
        print(f"Testing: temperature={temp}, top_p={top_p}")

        kl_values = []
        scale_consistency_values = []

        for idx_in_test in range(2):  # Use 2 instead of 5 for speed
            test_idx = test_indices[idx_in_test]
            sequence = sequences[test_idx]
            soprano_tokens = extract_soprano_tokens(sequence, tokenizer)

            # Harmonize with current settings
            harmonized = harmonize(model, soprano_tokens, tokenizer,
                                 temperature=temp, top_p=top_p, device=device,
                                 chromatic_penalty=3.0)

            # Compute metrics
            gen_dist = compute_pitch_class_distribution(harmonized, token_info)
            kl = compute_kl_divergence(real_avg, gen_dist)
            scale_cons = compute_scale_consistency(harmonized, token_info)

            kl_values.append(kl)
            scale_consistency_values.append(scale_cons)

        # Average metrics
        avg_kl = np.mean(kl_values)
        avg_scale = np.mean(scale_consistency_values)

        results.append({
            'temperature': temp,
            'top_p': top_p,
            'temp_str': temp_str,
            'top_p_str': top_p_str,
            'kl_divergence': avg_kl,
            'scale_consistency': avg_scale,
            'is_baseline': (temp == 1.0 and top_p == 1.0)
        })

        print(f"  KL Divergence: {avg_kl:.4f}")
        print(f"  Scale Consistency: {avg_scale:.1f}%\n")

    # Also compute metrics for real Bach (baseline) - use 2 sequences
    print("Computing metrics for real Bach...")
    real_kl_values = []
    real_scale_values = []
    for test_idx in test_indices[:2]:  # Use 2 instead of 5
        sequence = sequences[test_idx]
        dist = compute_pitch_class_distribution(sequence, token_info)
        kl = compute_kl_divergence(real_avg, dist)  # Should be low
        scale = compute_scale_consistency(sequence, token_info)
        real_kl_values.append(kl)
        real_scale_values.append(scale)

    real_avg_kl = np.mean(real_kl_values)
    real_avg_scale = np.mean(real_scale_values)
    print(f"Real Bach KL Divergence: {real_avg_kl:.4f}")
    print(f"Real Bach Scale Consistency: {real_avg_scale:.1f}%\n")

    # Print results table
    print("="*70)
    print("RESULTS TABLE")
    print("="*70)
    print(f"{'Temp':<6} {'Top-p':<7} {'KL Divergence':<14} {'Scale Consistency':<18}")
    print("-"*70)

    for result in results:
        marker = " <- baseline" if result['is_baseline'] else ""
        print(f"{result['temp_str']:<6} {result['top_p_str']:<7} {result['kl_divergence']:>13.4f} {result['scale_consistency']:>16.1f}%{marker}")

    print("-"*70)
    print(f"{'Real':<6} {'Bach':<7} {real_avg_kl:>13.4f} {real_avg_scale:>16.1f}%")
    print("="*70)

    # Create visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid", {"grid.color": "0.9"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Prepare data for plotting
    labels = [f"T={r['temp_str']}\np={r['top_p_str']}" for r in results]
    kl_divs = [r['kl_divergence'] for r in results]
    scales = [r['scale_consistency'] for r in results]

    # Colors: red for baseline, blue for others
    colors = ['red' if r['is_baseline'] else 'steelblue' for r in results]

    # Left panel: KL Divergence
    bars1 = ax1.bar(range(len(labels)), kl_divs, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=real_avg_kl, color='green', linestyle='--', linewidth=2, label='Real Bach')
    ax1.set_xlabel('Configuration', fontsize=11)
    ax1.set_ylabel('KL Divergence', fontsize=11)
    ax1.set_title('KL Divergence from Real Bach', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right panel: Scale Consistency
    bars2 = ax2.bar(range(len(labels)), scales, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(y=real_avg_scale, color='green', linestyle='--', linewidth=2, label='Real Bach')
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylabel('Scale Consistency (%)', fontsize=11)
    ax2.set_title('Scale Consistency', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylim([0, 105])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()

    images_dir = Path(__file__).parent.parent / 'images'
    images_dir.mkdir(exist_ok=True)

    fig_path = images_dir / 'experiment_sampling.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    print(f"\nFigure saved to {fig_path}")

    plt.close()


if __name__ == '__main__':
    main()
