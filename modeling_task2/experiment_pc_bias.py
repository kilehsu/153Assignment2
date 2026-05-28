"""
Bach pitch class bias experiment.

Test whether adding a log-prob bonus proportional to Bach's pitch class distribution
reduces KL divergence while preserving harmonic quality.

Compares bias weights [0.0, 0.5, 1.0, 2.0, 3.0] with chromatic_penalty=3.0.
"""

import sys, json, pickle, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, 'modeling')
from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer

CHECKPOINT_DIR = Path('modeling/checkpoints')


def kl_div(p: np.ndarray, q: np.ndarray, eps: float = 1e-8) -> float:
    """Compute KL divergence KL(p || q)."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def get_scale_notes(root: int, scale_type: str = "major") -> set:
    """Get notes in a major or minor scale."""
    if scale_type == "major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    else:  # natural minor
        intervals = [0, 2, 3, 5, 7, 8, 10]
    return {(root + i) % 12 for i in intervals}


def scale_consistency(notes: list) -> float:
    """Compute scale consistency: fraction of notes fitting best-matching scale."""
    if not notes:
        return 0.0
    best_fit = 0.0
    for root in range(12):
        for scale_type in ["major", "minor"]:
            scale = get_scale_notes(root, scale_type)
            fit = sum(1 for n in notes if (n % 12) in scale) / len(notes)
            best_fit = max(best_fit, fit)
    return best_fit


def build_scales():
    """Return dict: (root, mode) -> frozenset of pitch classes."""
    scales = {}
    for root in range(12):
        major_intervals = [0, 2, 4, 5, 7, 9, 11]
        minor_intervals = [0, 2, 3, 5, 7, 8, 10]
        scales[(root, 'major')] = frozenset((root + i) % 12 for i in major_intervals)
        scales[(root, 'minor')] = frozenset((root + i) % 12 for i in minor_intervals)
    return scales


def detect_key(pitch_classes):
    """Pick the major/minor key that fits the most observed pitch classes."""
    if not pitch_classes:
        return None, None, 0.0
    all_scales = build_scales()
    best_key, best_mode, best_score = 0, 'major', 0.0
    for (root, mode), scale_pcs in all_scales.items():
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
    """Extract soprano tokens from full interleaved sequence."""
    decoded = tokenizer.decode(sequence)
    soprano_tokens = []
    i = 1  # Start after START token
    while i < len(decoded):
        tok = decoded[i]
        if tok in ('<START>', '<END>', '<BAR>'):
            i += 1
            continue
        soprano_tokens.append(sequence[i])
        i += 4  # Next voice 0 is 4 positions away
    return soprano_tokens


def harmonize_with_pc_bias(model, soprano_tokens, tokenizer, temperature=1.0, device=None,
                           chromatic_penalty=3.0, bias_weight=0.0, bach_pc_hist=None):
    """
    Autoregressively harmonize with optional pitch class bias.

    Adds log-prob bonus proportional to Bach's pitch class distribution.
    """
    if device is None:
        device = next(model.parameters()).device

    start_id   = tokenizer.token_to_id.get('<START>', 0)
    end_id     = tokenizer.token_to_id.get('<END>', -1)
    sustain_id = tokenizer.token_to_id.get('<SUSTAIN>', -1)

    tokens = torch.tensor([[start_id]], dtype=torch.long, device=device)
    soprano_idx = 0
    detected_key = None
    detected_mode = None
    key_pcs = None
    token_info = build_token_info(tokenizer)
    all_scales = build_scales()

    model.eval()
    with torch.no_grad():
        for step in range(600):
            num_generated = tokens.shape[1] - 1
            current_voice = num_generated % 4

            # Key detection after warmup
            if num_generated == 40 and detected_key is None:
                pitch_classes = []
                for tid in tokens[0].tolist():
                    i = token_info.get(tid, ('unknown',))
                    if i[0] == 'note':
                        pitch_classes.append(i[2])
                detected_key, detected_mode, fit = detect_key(pitch_classes)
                key_pcs = all_scales[(detected_key, detected_mode)]

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

                # PC bias: add log-prob bonus proportional to Bach distribution
                if bias_weight > 0.0 and bach_pc_hist is not None:
                    for tid, i in token_info.items():
                        if i[0] == 'note':
                            pc = i[2]
                            logits[0, tid] += bias_weight * np.log(bach_pc_hist[pc] + 1e-8)

                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == end_id:
                break

    return tokens[0].cpu().tolist()


def extract_pitch_classes(token_ids, tokenizer, token_info):
    """Extract pitch class sequence from token IDs."""
    pcs = []
    for tid in token_ids:
        info = token_info.get(tid, ('unknown',))
        if info[0] == 'note':
            pcs.append(info[2])
    return pcs


def main():
    print("=" * 70)
    print("BACH PITCH CLASS BIAS EXPERIMENT")
    print("=" * 70)

    # Load tokenizer
    tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
    print(f'\nTokenizer: vocab size {tokenizer.vocab_size}')

    # Load model
    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size,
                               d_model=128, n_heads=8, n_layers=4,
                               context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'v2c_transformer_best.pt',
                                     map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f'Model loaded on {device}')

    # Load test sequences and splits
    with open(CHECKPOINT_DIR / 'v2_sequences_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    sequences = cache['sequences']
    print(f'Loaded {len(sequences)} sequences')

    with open(CHECKPOINT_DIR / 'v2_splits.json') as f:
        splits = json.load(f)
    test_indices = splits['test']
    print(f'Test split: {len(test_indices)} sequences')

    # Load Bach reference pitch class histogram
    with open(CHECKPOINT_DIR / 'v2c_music_metrics.json') as f:
        metrics = json.load(f)
    bach_pc_hist_raw = metrics['sources']['real_bach']['pitch_class_histogram']
    bach_pc_hist = np.array(bach_pc_hist_raw, dtype=float)
    print(f'\nBach pitch class histogram loaded (12 classes)')
    print(f'  Min: {bach_pc_hist.min():.4f}, Max: {bach_pc_hist.max():.4f}')

    # Build token info
    token_info = build_token_info(tokenizer)

    # Test chorale indices (0-4)
    test_chorale_indices = test_indices[:5]
    print(f'\nTesting on {len(test_chorale_indices)} chorales')

    # Bias weights to test
    bias_weights = [0.0, 0.5, 1.0, 2.0, 3.0]
    chromatic_penalty = 3.0

    results = {w: {'kl_divs': [], 'scale_consistencies': []} for w in bias_weights}

    print("\n" + "=" * 70)
    print("HARMONIZING TEST CHORALES")
    print("=" * 70)

    for chorale_idx, test_idx in enumerate(test_chorale_indices):
        print(f'\nChorale {chorale_idx} (global index {test_idx}):')
        sequence = sequences[test_idx]
        soprano_tokens = extract_soprano_tokens(sequence, tokenizer)
        print(f'  Soprano tokens: {len(soprano_tokens)}')

        # Test each bias weight
        for bias_weight in bias_weights:
            print(f'  Bias weight {bias_weight}: ', end='', flush=True)

            harmonized = harmonize_with_pc_bias(
                model, soprano_tokens, tokenizer,
                temperature=1.0, device=device,
                chromatic_penalty=chromatic_penalty,
                bias_weight=bias_weight,
                bach_pc_hist=bach_pc_hist
            )

            # Extract pitch classes from generated sequence
            pcs = extract_pitch_classes(harmonized, tokenizer, token_info)

            if len(pcs) > 0:
                # Compute pitch class histogram
                pc_hist = np.zeros(12, dtype=float)
                for pc in pcs:
                    pc_hist[pc] += 1
                pc_hist /= len(pcs)

                # Compute KL divergence
                kl = kl_div(bach_pc_hist, pc_hist)

                # Compute scale consistency
                scale_cons = scale_consistency(pcs)

                results[bias_weight]['kl_divs'].append(kl)
                results[bias_weight]['scale_consistencies'].append(scale_cons)

                print(f'KL={kl:.4f}, SC={scale_cons:.1%}')
            else:
                print('ERROR: No pitch classes extracted')

    # Compute averages and print results table
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    print(f"\n{'Bias Weight':<15} | {'KL Divergence':<15} | {'Scale Consistency':<20}")
    print("-" * 52)

    for bias_weight in bias_weights:
        if results[bias_weight]['kl_divs']:
            avg_kl = np.mean(results[bias_weight]['kl_divs'])
            avg_sc = np.mean(results[bias_weight]['scale_consistencies'])
            marker = " <- baseline (no bias)" if bias_weight == 0.0 else ""
            print(f"{bias_weight:<15.1f} | {avg_kl:<15.4f} | {avg_sc*100:<19.1f}%{marker}")

    # Real Bach reference (KL = 0 by definition)
    real_sc = metrics['sources']['real_bach']['scale_consistency']
    print("-" * 52)
    print(f"{'Real Bach':<15} | {0.0:<15.4f} | {real_sc*100:<19.1f}%")

    # Save figure
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid", rc={'figure.facecolor':'white'})
        plt.rcParams['figure.figsize'] = (10, 4)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Left panel: KL divergence vs bias weight
        kl_means = [np.mean(results[w]['kl_divs']) for w in bias_weights]
        kl_stds = [np.std(results[w]['kl_divs']) for w in bias_weights]
        ax1.errorbar(bias_weights, kl_means, yerr=kl_stds, marker='o', capsize=5,
                     linewidth=2, markersize=8, label='Model')
        ax1.axhline(y=0.0, color='red', linestyle='--', linewidth=2, label='Real Bach')
        ax1.set_xlabel('Bias Weight', fontsize=12)
        ax1.set_ylabel('KL Divergence', fontsize=12)
        ax1.set_title('Pitch Class KL Divergence vs Bias Weight', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Right panel: Scale consistency vs bias weight
        sc_means = [np.mean(results[w]['scale_consistencies']) * 100 for w in bias_weights]
        sc_stds = [np.std(results[w]['scale_consistencies']) * 100 for w in bias_weights]
        ax2.errorbar(bias_weights, sc_means, yerr=sc_stds, marker='o', capsize=5,
                     linewidth=2, markersize=8, label='Model')
        ax2.axhline(y=real_sc*100, color='red', linestyle='--', linewidth=2, label='Real Bach')
        ax2.set_xlabel('Bias Weight', fontsize=12)
        ax2.set_ylabel('Scale Consistency (%)', fontsize=12)
        ax2.set_title('Scale Consistency vs Bias Weight', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = Path('images/experiment_pc_bias.png')
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(fig_path), dpi=120)
        print(f'\n✓ Figure saved to {fig_path}')
        plt.close()

    except Exception as e:
        print(f'\n✗ Error saving figure: {e}')

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
