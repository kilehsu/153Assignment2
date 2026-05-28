"""
Combination experiment: test 5 configurations blending penalty, PC bias, and top-p sampling.

Configurations:
1. Baseline: penalty=3.0, bias=0.0, top_p=1.0
2. S1+S2: penalty=1.5, bias=2.0, top_p=1.0
3. S2+S3: penalty=3.0, bias=2.0, top_p=0.95
4. S1+S3: penalty=1.5, bias=0.0, top_p=0.95
5. All three: penalty=1.5, bias=2.0, top_p=0.95

Test on first 5 test chorales (indices 0-4), compute KL divergence and scale consistency.
"""

import sys
import json
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'modeling'))

from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer

CHECKPOINT_DIR = Path(__file__).parent.parent / 'modeling' / 'checkpoints'
IMAGES_DIR = Path(__file__).parent.parent / 'images'
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Scale definitions
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


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
    """
    decoded = tokenizer.decode(sequence)
    soprano_tokens = []

    i = 1  # Start after START token
    while i < len(decoded):
        tok = decoded[i]
        if tok in ('<START>', '<END>', '<BAR>'):
            i += 1
            continue
        # We're at a voice 0 position
        soprano_tokens.append(sequence[i])
        i += 4  # Next voice 0 is 4 positions away

    return soprano_tokens


def top_p_sample(logits, p=0.95):
    """Nucleus (top-p) sampling."""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumulative prob exceeds p
    sorted_probs[cumulative - sorted_probs > p] = 0.0
    sorted_probs /= sorted_probs.sum()

    sampled_idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx[sampled_idx]


def harmonize(model, soprano_tokens, tokenizer, temperature=1.0, top_p=1.0, device=None,
              chromatic_penalty=3.0, bias_weight=0.0, bach_pc_hist=None):
    """
    Autoregressively harmonize with optional penalty, PC bias, and top-p sampling.
    """
    if device is None:
        device = next(model.parameters()).device

    start_id   = tokenizer.token_to_id.get('<START>', 0)
    end_id     = tokenizer.token_to_id.get('<END>', -1)

    tokens = torch.tensor([[start_id]], dtype=torch.long, device=device)
    soprano_idx = 0
    detected_key = None
    detected_mode = None
    key_pcs = None
    token_info = build_token_info(tokenizer)

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
                key_pcs = ALL_SCALES[(detected_key, detected_mode)]

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

                # Apply temperature
                logits = logits / temperature

                # Sample with top-p if p < 1.0
                if top_p < 1.0:
                    sampled = top_p_sample(logits[0], p=top_p)
                    next_token = sampled.view(1, 1)
                else:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == end_id:
                break

    return tokens[0].cpu().tolist()


def kl_div(p, q, eps=1e-8):
    """Compute KL divergence between two probability distributions."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * (np.log(p) - np.log(q)))


def pitch_class_histogram(notes, normalize=True):
    """Compute pitch class histogram from MIDI notes."""
    hist = np.zeros(12)
    for midi in notes:
        pc = midi % 12
        hist[pc] += 1

    if normalize and np.sum(hist) > 0:
        hist = hist / np.sum(hist)

    return hist


def scale_consistency(midi_notes):
    """Compute scale consistency: % of notes fitting the best-fitting major/minor scale."""
    if not midi_notes:
        return 0.0

    pitch_classes = [m % 12 for m in midi_notes]
    best_key, best_mode, best_score = detect_key(pitch_classes)

    return best_score * 100.0


def parse_notes_from_tokens(token_ids, tokenizer):
    """Extract (voice, midi, dur) tuples from token IDs."""
    decoded = tokenizer.decode(token_ids)
    notes = []

    for tok in decoded:
        if tok.startswith('V') and 'P' in tok and 'D' in tok:
            try:
                after_v = tok[1:]
                v = int(after_v.split('P')[0])
                midi_str, dur_str = after_v.split('P')[1].split('D')
                midi = int(midi_str)
                dur = int(dur_str)
                notes.append((v, midi, dur))
            except (ValueError, IndexError):
                pass

    return notes


def main():
    print("="*70)
    print("COMBINATION EXPERIMENT: Penalty + PC Bias + Top-P Sampling")
    print("="*70)

    # Load tokenizer
    tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
    print(f"Tokenizer: vocab size {tokenizer.vocab_size}")

    # Load model
    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size,
                               d_model=128, n_heads=8, n_layers=4,
                               context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'v2c_transformer_best.pt',
                                     map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model loaded on {device}")

    # Load sequences and splits
    with open(CHECKPOINT_DIR / 'v2_sequences_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    sequences = cache['sequences']

    with open(CHECKPOINT_DIR / 'v2_splits.json') as f:
        splits = json.load(f)
    test_indices = splits['test']

    print(f"Loaded {len(sequences)} sequences, test split has {len(test_indices)} sequences")

    # Use first 5 test chorales
    test_indices = test_indices[:5]
    print(f"Using test indices: {test_indices}")

    # Load Bach PC histogram for bias
    with open(CHECKPOINT_DIR / 'v2c_music_metrics.json') as f:
        metrics = json.load(f)
    bach_pc_hist_raw = metrics['sources']['real_bach']['pitch_class_histogram']
    bach_pc_hist = np.array(bach_pc_hist_raw, dtype=float)

    # Compute real Bach reference pitch class histogram
    print("\nComputing real Bach reference histogram from 5 test chorales...")
    real_bach_notes = []
    for test_idx in test_indices:
        sequence = sequences[test_idx]
        notes = parse_notes_from_tokens(sequence, tokenizer)
        real_bach_notes.extend([midi for _, midi, _ in notes])

    real_bach_hist = pitch_class_histogram(real_bach_notes, normalize=True)
    real_bach_consistency = scale_consistency(real_bach_notes)

    print(f"Real Bach: {len(real_bach_notes)} total notes, scale consistency {real_bach_consistency:.1f}%")

    # Define configurations
    configs = [
        ('Baseline',     3.0, 0.0, 1.00),
        ('S1+S2',        1.5, 2.0, 1.00),
        ('S2+S3',        3.0, 2.0, 0.95),
        ('S1+S3',        1.5, 0.0, 0.95),
        ('All Three',    1.5, 2.0, 0.95),
    ]

    results = {}

    for config_name, penalty, bias, top_p in configs:
        print(f"\n{'='*70}")
        print(f"Testing: {config_name} (penalty={penalty}, bias={bias}, top_p={top_p})")
        print(f"{'='*70}")

        all_generated_notes = []

        for idx_in_test, test_idx in enumerate(test_indices):
            print(f"  Harmonizing test chorale {idx_in_test} (index {test_idx})...")

            sequence = sequences[test_idx]
            soprano_tokens = extract_soprano_tokens(sequence, tokenizer)
            print(f"    Soprano tokens: {len(soprano_tokens)}")

            harmonized = harmonize(model, soprano_tokens, tokenizer,
                                 temperature=1.0, top_p=top_p, device=device,
                                 chromatic_penalty=penalty, bias_weight=bias,
                                 bach_pc_hist=bach_pc_hist)
            print(f"    Generated: {len(harmonized)} tokens")

            # Parse notes from generated sequence
            notes = parse_notes_from_tokens(harmonized, tokenizer)
            midi_notes = [midi for _, midi, _ in notes]
            all_generated_notes.extend(midi_notes)
            print(f"    Extracted {len(midi_notes)} notes")

        # Compute metrics
        gen_hist = pitch_class_histogram(all_generated_notes, normalize=True)
        kl = kl_div(real_bach_hist, gen_hist)
        consistency = scale_consistency(all_generated_notes)

        results[config_name] = {
            'penalty': penalty,
            'bias': bias,
            'top_p': top_p,
            'kl_divergence': kl,
            'scale_consistency': consistency,
            'notes_count': len(all_generated_notes)
        }

        print(f"\n  Results for {config_name}:")
        print(f"    KL divergence vs real Bach: {kl:.4f}")
        print(f"    Scale consistency: {consistency:.1f}%")
        print(f"    Total notes: {len(all_generated_notes)}")

    # Print results table
    print(f"\n{'='*70}")
    print("RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Config':<12} | {'Penalty':>7} | {'Bias':>5} | {'Top-p':>6} | {'KL Div':>8} | {'Scale %':>7}")
    print(f"{'-'*70}")

    for config_name, penalty, bias, top_p in configs:
        kl = results[config_name]['kl_divergence']
        consistency = results[config_name]['scale_consistency']
        baseline_marker = "   <- baseline" if config_name == 'Baseline' else ""
        print(f"{config_name:<12} | {penalty:>7.1f} | {bias:>5.1f} | {top_p:>6.2f} | {kl:>8.4f} | {consistency:>7.1f}%{baseline_marker}")

    # Real Bach row
    print(f"{'-'*70}")
    print(f"{'Real Bach':<12} |   -    |  -    |   -   | {'0.0000':>8} | {real_bach_consistency:>7.1f}%")

    # Create figure
    print(f"\nCreating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    sns.set_style("whitegrid", {"grid.color": "0.9"})

    # Prepare data for plotting
    config_names = [name for name, _, _, _ in configs]
    kl_list = [results[name]['kl_divergence'] for name in config_names]
    consistency_list = [results[name]['scale_consistency'] for name in config_names]

    # Colors: red for baseline, dark blue for best, light blue for others
    best_kl_idx = np.argmin(kl_list)
    colors_kl = ['red' if name == 'Baseline' else ('darkblue' if i == best_kl_idx else 'lightblue')
                 for i, name in enumerate(config_names)]
    colors_sc = ['red' if name == 'Baseline' else 'lightblue' for name in config_names]

    # Left panel: KL divergence
    y_pos = np.arange(len(config_names))
    ax1.barh(y_pos, kl_list, color=colors_kl, alpha=0.8, edgecolor='black', linewidth=0.7)
    ax1.axvline(x=0.0, color='green', linestyle='--', linewidth=2, label='Real Bach')
    ax1.set_xlabel('KL Divergence', fontsize=11)
    ax1.set_title('Pitch Class KL Divergence (lower is better)', fontsize=12, fontweight='bold')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(config_names, fontsize=10)
    ax1.invert_yaxis()
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (kl, name) in enumerate(zip(kl_list, config_names)):
        ax1.text(kl + 0.01, i, f'{kl:.4f}', va='center', fontsize=9)

    # Right panel: Scale consistency
    ax2.barh(y_pos, consistency_list, color=colors_sc, alpha=0.8, edgecolor='black', linewidth=0.7)
    ax2.axvline(x=real_bach_consistency, color='green', linestyle='--', linewidth=2, label='Real Bach')
    ax2.set_xlabel('Scale Consistency (%)', fontsize=11)
    ax2.set_title('Scale Consistency (higher is better)', fontsize=12, fontweight='bold')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(config_names, fontsize=10)
    ax2.set_xlim([0, 105])
    ax2.invert_yaxis()
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (sc, name) in enumerate(zip(consistency_list, config_names)):
        ax2.text(sc + 1, i, f'{sc:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    # Save figure
    fig_path = IMAGES_DIR / 'experiment_combined.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")
    plt.close()

    # Find best config
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")

    # Filter configs with scale consistency >= 90%
    valid_configs = [(name, results[name]) for name in config_names
                     if results[name]['scale_consistency'] >= 90.0]

    if valid_configs:
        best_name, best_result = min(valid_configs, key=lambda x: x[1]['kl_divergence'])
        print(f"\nBest config (KL-minimizing with Scale % >= 90%):")
        print(f"  Name: {best_name}")
        print(f"  Penalty: {best_result['penalty']}")
        print(f"  Bias: {best_result['bias']}")
        print(f"  Top-p: {best_result['top_p']}")
        print(f"  KL Divergence: {best_result['kl_divergence']:.4f}")
        print(f"  Scale Consistency: {best_result['scale_consistency']:.1f}%")

        # Save best config for later use
        best_config_file = Path(__file__).parent / 'best_config.json'
        with open(best_config_file, 'w') as f:
            json.dump({
                'name': best_name,
                'penalty': best_result['penalty'],
                'bias': best_result['bias'],
                'top_p': best_result['top_p'],
            }, f, indent=2)
        print(f"\nBest config saved to: {best_config_file}")
    else:
        print("\nWarning: No configs meet scale consistency >= 90% threshold!")
        print("Using all configs to find best by KL divergence.")
        best_name, best_result = min([(name, results[name]) for name in config_names],
                                     key=lambda x: x[1]['kl_divergence'])
        print(f"Best overall: {best_name} with KL={best_result['kl_divergence']:.4f}")

    print("\nExperiment complete.")


if __name__ == '__main__':
    main()
