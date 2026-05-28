"""Test rejection sampling fix: generate N candidates, pick best KL."""
import sys, json, pickle, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter

EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'modeling'))
from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer

CHECKPOINT_DIR = PROJECT_ROOT / 'modeling' / 'checkpoints'

# Major and natural-minor interval patterns (semitones from root)
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES      = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


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


def generate_keyed(model, start_tokens, tokenizer, token_info,
                   max_new_tokens=480, min_new_tokens=240, temperature=1.0,
                   warmup_tokens=60, chromatic_penalty=4.0,
                   sustain_penalty=2.5, max_voice_sustains=16, device=None):
    """
    Autoregressive generation with:
      - key-detection + chromatic soft masking
      - per-voice sustain penalty to keep all 4 voices active
      - min_new_tokens to prevent early termination
    """
    if device is None:
        device = next(model.parameters()).device

    sustain_id = tokenizer.token_to_id.get('<SUSTAIN>', -1)
    end_id     = tokenizer.token_to_id.get('<END>', -1)

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

            # Voice sustain penalty: discourage this voice from sustaining too long
            if voice_sustain_counts[current_voice] >= max_voice_sustains:
                logits[0, sustain_id] -= sustain_penalty

            logits = logits / temperature
            probs  = F.softmax(logits, dim=-1)
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


def compute_kl(token_ids, token_info, bach_ref):
    """Compute KL divergence of pitch class distribution from Bach reference."""
    pcs = [token_info[t][2] for t in token_ids if token_info.get(t, ('x',))[0]=='note']
    if not pcs:
        return 99.0
    hist = np.array([pcs.count(i) for i in range(12)], dtype=float)
    hist /= hist.sum()
    eps = 1e-10
    ref = np.clip(bach_ref, eps, 1.0)
    q = np.clip(hist, eps, 1.0)
    return float(np.sum(ref * (np.log(ref) - np.log(q))))


def main():
    print("=" * 70)
    print("Rejection Sampling Experiment: Pitch Class KL Divergence")
    print("=" * 70)

    # Load model, tokenizer, Bach reference
    print("\n[1] Loading model and tokenizer...")
    tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
    print(f"    Tokenizer: vocab size {tokenizer.vocab_size}")

    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size,
                               d_model=128, n_heads=8, n_layers=4,
                               context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'v2c_transformer_best.pt',
                                     map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()
    print(f"    Model loaded on {device}")

    # Load Bach reference histogram
    metrics_path = CHECKPOINT_DIR / 'v2_music_metrics.json'
    with open(metrics_path) as f:
        metrics = json.load(f)
    bach_ref = np.array(metrics['sources']['real_bach']['pitch_class_histogram'])
    print(f"    Bach reference KL (vs itself): 0.0")

    token_info = build_token_info(tokenizer)
    start_id   = tokenizer.token_to_id['<START>']

    # Test configuration
    N_VALUES = [1, 3, 5, 8]
    N_TRIALS = 5

    print(f"\n[2] Running rejection sampling experiment...")
    print(f"    N_VALUES: {N_VALUES}")
    print(f"    N_TRIALS: {N_TRIALS}")

    results = {
        "n_values": N_VALUES,
        "avg_kl": [],
        "best_kl": [],
        "all_trials": {}
    }

    baseline_kl = None

    for n_cand in N_VALUES:
        print(f"\n    Testing N={n_cand}...")
        trial_kls = []

        for trial in range(N_TRIALS):
            # Generate N candidates
            candidates_kl = []
            for cand_idx in range(n_cand):
                token_ids = generate_keyed(
                    model, [start_id], tokenizer, token_info,
                    max_new_tokens=480, min_new_tokens=240, temperature=1.0,
                    warmup_tokens=60, chromatic_penalty=4.0, device=device
                )
                kl_val = compute_kl(token_ids, token_info, bach_ref)
                candidates_kl.append(kl_val)

            # Pick best KL
            best_kl = min(candidates_kl)
            trial_kls.append(best_kl)

            if n_cand == 1 and baseline_kl is None:
                baseline_kl = best_kl

            print(f"      Trial {trial+1}: candidates KL = {[f'{k:.4f}' for k in candidates_kl]}, best = {best_kl:.4f}")

        avg_kl = np.mean(trial_kls)
        min_kl = np.min(trial_kls)
        max_kl = np.max(trial_kls)
        std_kl = np.std(trial_kls)

        results["avg_kl"].append(avg_kl)
        results["best_kl"].append(min_kl)
        results["all_trials"][f"n={n_cand}"] = trial_kls

        print(f"      Summary: avg={avg_kl:.4f}, best={min_kl:.4f}, worst={max_kl:.4f}, std={std_kl:.4f}")

    # Add baseline
    results["baseline_kl"] = baseline_kl

    # Save results
    output_path = EVAL_DIR / 'rejection_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[3] Results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline KL (N=1): {baseline_kl:.4f}")
    for n_cand, avg_kl, best_kl in zip(N_VALUES, results["avg_kl"], results["best_kl"]):
        improvement = (baseline_kl - avg_kl) / baseline_kl * 100 if baseline_kl else 0
        print(f"N={n_cand}: avg_kl={avg_kl:.4f} (improvement: {improvement:+.1f}%), best_kl={best_kl:.4f}")

    best_n = N_VALUES[np.argmin(results["avg_kl"])]
    print(f"\nBest N value: {best_n}")
    print("=" * 70)


if __name__ == '__main__':
    main()
