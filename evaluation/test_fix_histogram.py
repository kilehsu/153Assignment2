"""
Test pitch class histogram correction fix for v2c model.
Tests histogram_correction_strength parameter from 0.0 to 5.0.
"""
import sys, os, json, pickle, torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
MODELING_DIR = PROJECT_ROOT / 'modeling'

sys.path.insert(0, str(MODELING_DIR))

from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer

CHECKPOINT_DIR = MODELING_DIR / 'checkpoints'
OUTPUT_DIR = EVAL_DIR
IMAGES_DIR = PROJECT_ROOT / 'images'

# Major and natural-minor interval patterns
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


def generate_with_histogram_correction(model, start_tokens, tokenizer, token_info,
                                       bach_ref_histogram,
                                       max_new_tokens=480, min_new_tokens=240,
                                       temperature=1.0, warmup_tokens=60,
                                       chromatic_penalty=4.0, sustain_penalty=2.5,
                                       max_voice_sustains=16,
                                       histogram_correction_strength=0.0,
                                       device=None):
    """
    Autoregressive generation with:
      - key-detection + chromatic soft masking
      - per-voice sustain penalty to keep all 4 voices active
      - histogram correction to balance pitch class distribution
      - min_new_tokens to prevent early termination
    """
    if device is None:
        device = next(model.parameters()).device

    sustain_id = tokenizer.token_to_id.get('<SUSTAIN>', -1)
    end_id = tokenizer.token_to_id.get('<END>', -1)

    tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)
    detected_key = detected_mode = None
    key_pcs = None
    # count consecutive SUSTAIN tokens per voice (reset on any non-SUSTAIN token)
    voice_sustain_counts = [0, 0, 0, 0]

    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            num_generated = tokens.shape[1] - len(start_tokens)
            current_voice = num_generated % 4  # which voice we're predicting now

            if num_generated == warmup_tokens and detected_key is None:
                pitch_classes = []
                for tid in tokens[0].tolist():
                    i = token_info.get(tid, ('unknown',))
                    if i[0] == 'note':
                        pitch_classes.append(i[2])
                detected_key, detected_mode, fit = detect_key(pitch_classes)
                key_pcs = ALL_SCALES[(detected_key, detected_mode)]

            tokens_in = tokens[:, -model.context_len:]
            logits = model.forward(tokens_in)[:, -1, :]

            # Suppress <END> until min length reached
            if num_generated < min_new_tokens:
                logits[0, end_id] = float('-inf')

            # Chromatic penalty
            if key_pcs is not None:
                for tid, i in token_info.items():
                    if i[0] == 'note' and i[2] not in key_pcs:
                        logits[0, tid] -= chromatic_penalty

            # Histogram correction: boost underrepresented, penalize overrepresented
            if histogram_correction_strength > 0.0:
                # Compute current pitch class histogram
                pitch_classes = []
                for tid in tokens[0].tolist():
                    i = token_info.get(tid, ('unknown',))
                    if i[0] == 'note':
                        pitch_classes.append(i[2])

                # Count current distribution
                if pitch_classes:
                    pc_counter = Counter(pitch_classes)
                    current_hist = np.array([pc_counter.get(pc, 0) for pc in range(12)], dtype=float)
                    current_hist = current_hist / current_hist.sum() if current_hist.sum() > 0 else current_hist
                else:
                    current_hist = np.zeros(12)

                # Apply correction to each note token
                bach_ref = np.array(bach_ref_histogram)
                for tid, i in token_info.items():
                    if i[0] == 'note':
                        pc = i[2]
                        # boost underrepresented, penalize overrepresented
                        correction = histogram_correction_strength * (bach_ref[pc] - current_hist[pc])
                        logits[0, tid] += correction

            # Voice sustain penalty: discourage this voice from sustaining too long
            if voice_sustain_counts[current_voice] >= max_voice_sustains:
                logits[0, sustain_id] -= sustain_penalty

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            # Update sustain counter for this voice
            if next_token.item() == sustain_id:
                voice_sustain_counts[current_voice] += 1
            else:
                voice_sustain_counts[current_voice] = 0

            if next_token.item() == end_id:
                break

    return tokens[0].cpu().tolist()


def compute_pitch_class_histogram(token_ids, token_info):
    """Extract pitch class distribution from token sequence."""
    pitch_classes = []
    for tid in token_ids:
        i = token_info.get(tid, ('unknown',))
        if i[0] == 'note':
            pitch_classes.append(i[2])

    if not pitch_classes:
        return np.zeros(12)

    hist = np.array([pitch_classes.count(pc) for pc in range(12)], dtype=float)
    return hist / hist.sum()


def kl_divergence(p, q, epsilon=1e-10):
    """Compute KL(p||q) where p is reference, q is model distribution."""
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)))


def compute_scale_consistency(token_ids, token_info):
    """Percentage of notes fitting best detected key."""
    pitch_classes = []
    for tid in token_ids:
        i = token_info.get(tid, ('unknown',))
        if i[0] == 'note':
            pitch_classes.append(i[2])

    if not pitch_classes:
        return 0.0

    _, _, fit = detect_key(pitch_classes)
    return fit * 100.0


def count_notes(token_ids, token_info):
    """Count total number of notes in sequence."""
    count = 0
    for tid in token_ids:
        i = token_info.get(tid, ('unknown',))
        if i[0] == 'note':
            count += 1
    return count


def main():
    print("Loading model and tokenizer...")
    tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
    print(f'Tokenizer: vocab size {tokenizer.vocab_size}')

    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size,
                               d_model=128, n_heads=8, n_layers=4,
                               context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'v2c_transformer_best.pt',
                                     map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f'Model loaded on {device}')

    # Load Bach reference histogram
    with open(CHECKPOINT_DIR / 'v2c_music_metrics.json', 'r') as f:
        metrics = json.load(f)
    bach_ref_histogram = metrics['sources']['real_bach']['pitch_class_histogram']
    print(f"\nBach reference histogram: {bach_ref_histogram}")

    token_info = build_token_info(tokenizer)
    start_id = tokenizer.token_to_id['<START>']

    # Test strengths
    strengths = [0.0, 1.0, 2.0, 3.0, 5.0]
    results = {
        'strength': [],
        'kl_div': [],
        'scale_pct': [],
        'note_count': []
    }

    print("\n" + "="*70)
    print("Testing histogram correction strengths")
    print("="*70)

    for strength in strengths:
        print(f"\nTesting strength = {strength}")
        kl_divs = []
        scale_pcts = []
        note_counts = []

        for sample_idx in range(1, 9):  # 8 samples
            print(f"  Sample {sample_idx}/8...", end='', flush=True)
            token_ids = generate_with_histogram_correction(
                model, [start_id], tokenizer, token_info,
                bach_ref_histogram,
                max_new_tokens=480, min_new_tokens=240,
                temperature=1.0, warmup_tokens=60,
                chromatic_penalty=4.0, sustain_penalty=2.5,
                histogram_correction_strength=strength,
                device=device
            )

            # Compute metrics
            gen_hist = compute_pitch_class_histogram(token_ids, token_info)
            kl = kl_divergence(np.array(bach_ref_histogram), gen_hist)
            scale_pct = compute_scale_consistency(token_ids, token_info)
            note_cnt = count_notes(token_ids, token_info)

            kl_divs.append(kl)
            scale_pcts.append(scale_pct)
            note_counts.append(note_cnt)
            print(f" KL={kl:.3f}")

        # Compute averages
        avg_kl = np.mean(kl_divs)
        avg_scale = np.mean(scale_pcts)
        avg_notes = np.mean(note_counts)

        results['strength'].append(strength)
        results['kl_div'].append(avg_kl)
        results['scale_pct'].append(avg_scale)
        results['note_count'].append(avg_notes)

        print(f"  -> Avg KL: {avg_kl:.3f}, Avg Scale: {avg_scale:.1f}%, Avg Notes: {avg_notes:.0f}")

    # Print results table
    print("\n" + "="*70)
    print("RESULTS TABLE")
    print("="*70)
    print(f"{'Strength':<10} | {'KL Div':<8} | {'Scale %':<8} | {'Notes':<6}")
    print("-" * 40)
    for i, strength in enumerate(strengths):
        kl = results['kl_div'][i]
        scale = results['scale_pct'][i]
        notes = results['note_count'][i]
        baseline_marker = " <- baseline" if strength == 0.0 else ""
        print(f"{strength:<10.1f} | {kl:<8.3f} | {scale:<8.1f} | {notes:<6.0f}{baseline_marker}")

    # Find best strength (lowest KL div)
    best_idx = np.argmin(results['kl_div'])
    best_strength = results['strength'][best_idx]
    best_kl = results['kl_div'][best_idx]
    baseline_kl = results['kl_div'][0]

    improvement = baseline_kl - best_kl
    improvement_pct = (improvement / baseline_kl) * 100 if baseline_kl > 0 else 0

    print("\n" + "="*70)
    print(f"BEST STRENGTH: {best_strength} (KL={best_kl:.3f})")
    print(f"Baseline KL (0.0): {baseline_kl:.3f}")
    print(f"Improvement: {improvement:.3f} ({improvement_pct:.1f}%)")
    print("="*70)

    # Save results to JSON
    result_data = {
        'strengths': results['strength'],
        'kl_divs': results['kl_div'],
        'scale_pcts': results['scale_pct'],
        'note_counts': results['note_count'],
        'best_strength': float(best_strength),
        'best_kl': float(best_kl),
        'baseline_kl': float(baseline_kl),
        'improvement_pct': float(improvement_pct)
    }
    with open(EVAL_DIR / 'histogram_results.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\nResults saved to histogram_results.json")

    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: KL divergence vs strength
    ax = axes[0]
    ax.plot(results['strength'], results['kl_div'], 'b-o', linewidth=2, markersize=8, label='Histogram Correction')
    ax.axhline(y=baseline_kl, color='r', linestyle='--', linewidth=2, label='Baseline (0.0)', alpha=0.7)
    ax.scatter([best_strength], [best_kl], color='g', s=200, marker='*', zorder=5, label=f'Best ({best_strength})')
    ax.set_xlabel('Histogram Correction Strength', fontsize=11)
    ax.set_ylabel('KL Divergence', fontsize=11)
    ax.set_title('Pitch Class KL Divergence vs Strength', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panel 2: Scale consistency % vs strength
    ax = axes[1]
    ax.plot(results['strength'], results['scale_pct'], 'g-s', linewidth=2, markersize=8, label='Scale Consistency')
    ax.axhline(y=results['scale_pct'][0], color='r', linestyle='--', linewidth=2, label='Baseline (0.0)', alpha=0.7)
    ax.set_xlabel('Histogram Correction Strength', fontsize=11)
    ax.set_ylabel('Scale Consistency (%)', fontsize=11)
    ax.set_title('Scale Consistency vs Strength', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_path = IMAGES_DIR / 'fix_histogram.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.close()


if __name__ == '__main__':
    main()
