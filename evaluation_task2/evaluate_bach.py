"""
Task 2 evaluation — Bach SATB harmonization via prefix-conditioning.

Metrics (for 3 test chorales):
  - Scale consistency   : % generated notes in detected key
  - Voice crossing rate : % chords with ordering violations (e.g. Alto > Soprano)
  - Parallel 5ths rate  : % consecutive chord pairs with parallel perfect 5ths

Baselines:
  - Random in-key       : sample from key scale, in typical SATB ranges
  - Soprano-only        : just the soprano, no harmony

Outputs:
  evaluation_task2/bach_metrics.json
"""

import sys, os, json, pickle
import numpy as np
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'modeling'))
sys.path.insert(0, str(ROOT / 'modeling_task2'))

import torch
from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer
from harmonize import harmonize, extract_soprano_tokens, decode_to_4voice_score

CKPT     = ROOT / 'modeling' / 'checkpoints'
EVAL_DIR = ROOT / 'evaluation_task2'

MAJOR_INTERVALS = [0,2,4,5,7,9,11]
MINOR_INTERVALS = [0,2,3,5,7,8,10]
VOICE_RANGES    = [(60,81), (53,74), (48,69), (36,64)]  # S, A, T, B


def detect_key(pitches):
    best_root, best_mode, best_score = 0, 'major', -1
    for root in range(12):
        for mode, ivs in [('major', MAJOR_INTERVALS), ('minor', MINOR_INTERVALS)]:
            scale = frozenset((root+i)%12 for i in ivs)
            score = sum(1 for p in pitches if p%12 in scale)
            if score > best_score:
                best_score, best_root, best_mode = score, root, mode
    ivs = MAJOR_INTERVALS if best_mode == 'major' else MINOR_INTERVALS
    scale_pcs = frozenset((best_root+i)%12 for i in ivs)
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    print(f"    Detected key: {note_names[best_root]} {best_mode}  (fit={best_score}/{len(pitches)})")
    return scale_pcs


def compute_metrics(voice_notes):
    """voice_notes: list of 4 lists, each [(t, midi, dur), ...]"""
    all_pitches = [p for notes in voice_notes for _, p, _ in notes]
    if not all_pitches:
        return {}

    scale_pcs = detect_key(all_pitches)

    # Scale consistency
    in_scale = sum(1 for p in all_pitches if p % 12 in scale_pcs)
    sc = 100.0 * in_scale / len(all_pitches)

    # Build chord dict indexed by timestep
    chords = {}
    for v, notes in enumerate(voice_notes):
        for t, midi, _ in notes:
            chords.setdefault(t, [None]*4)[v] = midi

    # Voice crossing
    crossings, total = 0, 0
    for chord in chords.values():
        if all(c is not None for c in chord):
            total += 1
            # S > A, A > T, T > B expected
            if chord[1] > chord[0] or chord[2] > chord[1] or chord[3] > chord[2]:
                crossings += 1
    vc = 100.0 * crossings / total if total > 0 else 0.0

    # Parallel 5ths
    times = sorted(chords)
    p5_count, pair_count = 0, 0
    for i in range(len(times) - 1):
        c1, c2 = chords[times[i]], chords[times[i+1]]
        if all(x is not None for x in c1 + c2):
            pair_count += 1
            for va in range(4):
                for vb in range(va+1, 4):
                    if ((c1[va] - c1[vb]) % 12 == 7 and
                        (c2[va] - c2[vb]) % 12 == 7 and
                        c1[va] != c2[va]):
                        p5_count += 1
    p5r = 100.0 * p5_count / (pair_count * 6) if pair_count > 0 else 0.0

    return {
        'Scale Consistency (%)': round(sc, 1),
        'Voice Crossing (%)':    round(vc, 1),
        'Parallel 5ths (%)':     round(p5r, 2),
        'n_notes':               len(all_pitches),
    }


def random_baseline(voice_notes, scale_pcs):
    """Replace Alto/Tenor/Bass with random in-key pitches in correct ranges."""
    sop_notes = voice_notes[0]
    rand = [sop_notes, [], [], []]
    for t, sop_midi, dur in sop_notes:
        for v in range(1, 4):
            lo, hi = VOICE_RANGES[v]
            candidates = [p for p in range(lo, hi+1) if p % 12 in scale_pcs]
            pitch = int(np.random.choice(candidates)) if candidates else (lo+hi)//2
            rand[v].append((t, pitch, dur))
    return rand


def main():
    np.random.seed(42)

    print("Loading tokenizer and model...")
    tokenizer = BachTokenizerV2.load(str(CKPT / 'v2_tokenizer.json'))
    with open(CKPT / 'v2_sequences_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    seqs = cache['sequences']
    with open(CKPT / 'v2_splits.json') as f:
        splits = json.load(f)

    model = ChoraleTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=128, n_heads=8, n_layers=4,
        context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(
        str(CKPT / 'v2c_transformer_best.pt'),
        map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model on {device}")

    our_results, rand_results = [], []

    for i, test_idx in enumerate(splits['test'][:3]):
        print(f"\n--- Chorale {i+1} (seq {test_idx}) ---")

        seq = seqs[test_idx]
        soprano_toks = extract_soprano_tokens(seq, tokenizer)
        print(f"  Soprano tokens: {len(soprano_toks)}")

        harmonized = harmonize(model, soprano_toks, tokenizer,
                               temperature=1.0, device=device, chromatic_penalty=3.0)
        voice_notes = decode_to_4voice_score(harmonized, tokenizer)

        for v in range(4):
            print(f"  Voice {v}: {len(voice_notes[v])} notes")

        our = compute_metrics(voice_notes)
        our_results.append(our)
        print(f"  Our model  → scale={our['Scale Consistency (%)']:.1f}%  "
              f"vc={our['Voice Crossing (%)']:.1f}%  "
              f"p5={our['Parallel 5ths (%)']:.2f}%")

        all_pitches = [p for notes in voice_notes for _, p, _ in notes]
        scale_pcs = detect_key(all_pitches)
        rand_notes = random_baseline(voice_notes, scale_pcs)
        rand = compute_metrics(rand_notes)
        rand_results.append(rand)
        print(f"  Random     → scale={rand['Scale Consistency (%)']:.1f}%  "
              f"vc={rand['Voice Crossing (%)']:.1f}%  "
              f"p5={rand['Parallel 5ths (%)']:.2f}%")

    def avg(results):
        keys = [k for k in results[0] if k != 'n_notes']
        return {k: round(float(np.mean([r[k] for r in results])), 2) for k in keys}

    output = {
        'Our Model (prefix-conditioning)': avg(our_results),
        'Random In-Key Baseline':          avg(rand_results),
        '_meta': {'n_chorales': len(our_results), 'test_indices': splits['test'][:3]},
    }

    out_path = EVAL_DIR / 'bach_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*55)
    print("RESULTS (averaged over 3 test chorales)")
    print("="*55)
    print(f"{'Metric':<30} {'Our Model':>12} {'Random':>12}")
    print("-"*55)
    for k in avg(our_results):
        print(f"{k:<30} {avg(our_results)[k]:>12} {avg(rand_results)[k]:>12}")
    print("="*55)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    main()
