"""
Chromatic penalty tuning experiment for Task 2.

Tests penalty values [0.5, 1.0, 1.5, 2.0, 3.0] on 5 test chorales,
measuring KL divergence vs real Bach and scale consistency.
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
from collections import defaultdict

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
    At each 16th-note position: [v0_token, v1_token, v2_token, v3_token]

    Return list of soprano (voice 0) token IDs, skipping special tokens.
    """
    decoded = tokenizer.decode(sequence)
    soprano_tokens = []

    # Position 0 is START, then we have groups of 4 tokens
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


def harmonize(model, soprano_tokens, tokenizer, temperature=1.0, device=None,
              chromatic_penalty=3.0):
    """
    Autoregressively harmonize given soprano tokens.

    At each step:
    - Determine which voice position we're at (step % 4)
    - If voice 0 and we have a soprano token: force it
    - Otherwise: sample from model with temperature
    - Apply key detection + chromatic penalty after 40 tokens
    - Stop when all soprano consumed OR model emits END OR 600 steps

    Returns list of token IDs.
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

                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == end_id:
                break

    return tokens[0].cpu().tolist()


def parse_notes_from_tokens(token_ids, tokenizer):
    """
    Extract (voice, midi, dur) tuples from token IDs.
    """
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


def kl_div(p, q, eps=1e-8):
    """Compute KL divergence between two probability distributions."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    return np.sum(p * (np.log(p) - np.log(q)))


def pitch_class_histogram(notes, normalize=True):
    """
    Compute pitch class histogram from MIDI notes.

    Args:
        notes: list of MIDI note numbers
        normalize: if True, return normalized histogram (probabilities)

    Returns:
        numpy array of length 12 (pitch classes 0-11)
    """
    hist = np.zeros(12)
    for midi in notes:
        pc = midi % 12
        hist[pc] += 1

    if normalize and np.sum(hist) > 0:
        hist = hist / np.sum(hist)

    return hist


def scale_consistency(midi_notes):
    """
    Compute scale consistency: % of notes fitting the best-fitting major/minor scale.

    Args:
        midi_notes: list of MIDI note numbers

    Returns:
        percentage (0-100) of notes in best-fitting scale
    """
    if not midi_notes:
        return 0.0

    pitch_classes = [m % 12 for m in midi_notes]
    best_key, best_mode, best_score = detect_key(pitch_classes)

    return best_score * 100.0


def main():
    print("Loading checkpoint files...")

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

    # Compute real Bach reference pitch class histogram
    print("\nComputing real Bach reference histogram from 5 test chorales...")
    real_bach_notes = []
    for test_idx in test_indices:
        sequence = sequences[test_idx]
        notes = parse_notes_from_tokens(sequence, tokenizer)
        real_bach_notes.extend([midi for _, midi, _ in notes])

    real_bach_hist = pitch_class_histogram(real_bach_notes, normalize=True)
    real_bach_pc_dist = [real_bach_notes[i % 12] % 12 for i in range(len(real_bach_notes))]
    real_bach_consistency = scale_consistency(real_bach_notes)

    print(f"Real Bach: {len(real_bach_notes)} total notes, scale consistency {real_bach_consistency:.1f}%")

    # Test penalties
    penalties = [0.5, 1.0, 1.5, 2.0, 3.0]
    results = {}

    for penalty in penalties:
        print(f"\n{'='*60}")
        print(f"Testing chromatic_penalty = {penalty}")
        print(f"{'='*60}")

        all_generated_notes = []

        for idx_in_test, test_idx in enumerate(test_indices):
            print(f"\n  Harmonizing test chorale {idx_in_test} (index {test_idx})...")

            sequence = sequences[test_idx]
            soprano_tokens = extract_soprano_tokens(sequence, tokenizer)
            print(f"    Soprano tokens: {len(soprano_tokens)}")

            harmonized = harmonize(model, soprano_tokens, tokenizer,
                                 temperature=1.0, device=device,
                                 chromatic_penalty=penalty)
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

        results[penalty] = {
            'kl_divergence': kl,
            'scale_consistency': consistency,
            'notes_count': len(all_generated_notes)
        }

        print(f"\n  Results for penalty={penalty}:")
        print(f"    KL divergence vs real Bach: {kl:.4f}")
        print(f"    Scale consistency: {consistency:.1f}%")
        print(f"    Total notes: {len(all_generated_notes)}")

    # Print results table
    print(f"\n{'='*60}")
    print("RESULTS TABLE")
    print(f"{'='*60}")
    print(f"{'Penalty':<8} | {'KL Divergence':<15} | {'Scale Consistency':<18}")
    print(f"{'-'*8}+{'-'*15}+{'-'*18}")

    for penalty in penalties:
        kl = results[penalty]['kl_divergence']
        consistency = results[penalty]['scale_consistency']
        baseline_marker = "   <- baseline" if penalty == 3.0 else ""
        print(f"{penalty:<8.1f} | {kl:<15.4f} | {consistency:<17.1f}%{baseline_marker}")

    # Real Bach row
    print(f"{'-'*8}+{'-'*15}+{'-'*18}")
    print(f"{'Real':<8}  | {0.0:<15.4f} | {real_bach_consistency:<17.1f}%")

    # Create figure
    print(f"\nCreating figure...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    sns.set_style("whitegrid", {"grid.color": "0.9"})

    # Left panel: KL divergence vs penalty
    penalty_list = list(results.keys())
    kl_list = [results[p]['kl_divergence'] for p in penalty_list]

    ax1.plot(penalty_list, kl_list, 'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.plot(3.0, results[3.0]['kl_divergence'], 'o', markersize=12, color='red',
             label='Baseline (3.0)', zorder=5)
    ax1.set_xlabel('Chromatic Penalty', fontsize=11)
    ax1.set_ylabel('KL Divergence', fontsize=11)
    ax1.set_title('KL Divergence vs Penalty', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right panel: Scale consistency vs penalty
    consistency_list = [results[p]['scale_consistency'] for p in penalty_list]

    ax2.plot(penalty_list, consistency_list, 's-', linewidth=2, markersize=8, color='forestgreen')
    ax2.plot(3.0, results[3.0]['scale_consistency'], 's', markersize=12, color='red',
             label='Baseline (3.0)', zorder=5)
    ax2.set_xlabel('Chromatic Penalty', fontsize=11)
    ax2.set_ylabel('Scale Consistency (%)', fontsize=11)
    ax2.set_title('Scale Consistency vs Penalty', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Save figure
    fig_path = IMAGES_DIR / 'experiment_penalty.png'
    plt.savefig(fig_path, dpi=120, bbox_inches='tight')
    print(f"Figure saved to: {fig_path}")

    print("\nExperiment complete.")


if __name__ == '__main__':
    main()
