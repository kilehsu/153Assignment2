# ============================================================
# ARTIFACT COLLECTION FOR TASK 1 — Run on Colab A100
# ============================================================
# This script trains v2b (full 40 epochs) and v2c (12 epochs),
# evaluates music metrics for all models, then downloads a zip
# of every JSON and .pt file needed for the notebook.
#
# STEP 1: Upload files (run upload cell below)
# STEP 2: Run training cells (v2b then v2c)
# STEP 3: Run evaluation cell
# STEP 4: Run download cell
# ============================================================


# ── CELL 1: Install dependencies ────────────────────────────
# !pip install music21 pretty_midi soundfile --quiet


# ── CELL 2: Upload required files ───────────────────────────
# from google.colab import files
# print("Upload these 6 files from modeling/ and modeling/checkpoints/:")
# print("  model.py, dataset.py, tokenizer_v2.py")
# print("  v2_tokenizer.json, v2_sequences_cache.pkl, v2_splits.json")
# uploaded = files.upload()


# ── CELL 3: Shared setup ─────────────────────────────────────
import json, math, pickle, re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tokenizer_v2 import BachTokenizerV2
from dataset import ChoraleDataset
from model import ChoraleTransformer, count_parameters

CHECKPOINT_DIR = Path('/content')
print("Loading tokenizer and sequences cache...")
tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
with open(CHECKPOINT_DIR / 'v2_sequences_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
sequences = cache['sequences']
with open(CHECKPOINT_DIR / 'v2_splits.json') as f:
    splits = json.load(f)
train_seqs = [sequences[i] for i in splits['train']]
val_seqs   = [sequences[i] for i in splits['val']]
test_seqs  = [sequences[i] for i in splits['test']]
print(f"Vocab: {tokenizer.vocab_size} | Train: {len(train_seqs)} | Val: {len(val_seqs)} | Test: {len(test_seqs)}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


# ── CELL 4: Helper — transpose sequence ─────────────────────
def transpose_sequence(seq, tokenizer, semitones):
    transposed = []
    for tid in seq:
        tok = tokenizer.id_to_token.get(tid, '')
        if tok in ('<START>', '<END>', '<BAR>', '<SUSTAIN>') or tok.endswith('REST'):
            transposed.append(tid)
        elif tok.startswith('V') and 'P' in tok and 'D' in tok:
            try:
                after_v = tok[1:]
                v_part, rest = after_v.split('P', 1)
                midi_str, dur_str = rest.split('D', 1)
                new_midi = int(midi_str) + semitones
                if 21 <= new_midi <= 108:
                    new_tok = f'V{v_part}P{new_midi}D{dur_str}'
                    new_tid = tokenizer.token_to_id.get(new_tok)
                    if new_tid is not None:
                        transposed.append(new_tid)
                    else:
                        transposed.append(tid)
                else:
                    transposed.append(tid)
            except (ValueError, IndexError):
                transposed.append(tid)
        else:
            transposed.append(tid)
    return transposed


# ── CELL 5: TRAIN V2B (uniform ±5 augmentation, 40 epochs) ──
# Skip this cell if you already have v2b_transformer_best.pt uploaded.
def train_model(train_sequences, val_sequences, tokenizer, device,
                num_epochs, augment_scheme, save_path, losses_path):
    """Generic training loop with augmentation scheme dict {shift: repeats}."""
    augmented = list(train_sequences)
    for shift, repeats in augment_scheme.items():
        for _ in range(repeats):
            for seq in train_sequences:
                augmented.append(transpose_sequence(seq, tokenizer, shift))
    print(f"Augmented: {len(augmented)} sequences ({len(augmented)//len(train_sequences)}x)")

    train_ds = ChoraleDataset(augmented, context_len=512, stride=64)
    val_ds   = ChoraleDataset(val_sequences, context_len=512, stride=32)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=64, shuffle=False, num_workers=2)

    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size, d_model=128,
                               n_heads=8, n_layers=4, context_len=512, dropout=0.15)
    model.output_proj.weight = model.token_embedding.weight
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    total_steps = num_epochs * len(train_dl)
    warmup = 1000
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s / max(1, warmup)) if s < warmup
                  else max(0.0, 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))))
    )

    train_losses, val_losses, best_val, step = [], [], float('inf'), 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss, nb = 0.0, 0
        for inp, tgt in train_dl:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model(inp)
            loss = nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), tgt.view(-1))
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); step += 1
            epoch_loss += loss.item(); nb += 1

        model.eval(); vl, nvb = 0.0, 0
        with torch.no_grad():
            for inp, tgt in val_dl:
                inp, tgt = inp.to(device), tgt.to(device)
                logits = model(inp)
                vl += nn.functional.cross_entropy(logits.view(-1, tokenizer.vocab_size), tgt.view(-1)).item()
                nvb += 1
        tl = epoch_loss / nb; vl = vl / max(1, nvb)
        train_losses.append(tl); val_losses.append(vl)
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_path)
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {tl:.4f} | Val: {vl:.4f} | LR: {lr:.7f}")

    losses = {'train_losses': train_losses, 'val_losses': val_losses, 'best_val_loss': best_val}
    with open(losses_path, 'w') as f:
        json.dump(losses, f)
    print(f"Saved: {save_path} (best val {best_val:.4f})")
    print(f"Saved: {losses_path}")
    return model

# V2B — uniform ±5 (1 copy each)
v2b_scheme = {s: 1 for s in range(-5, 6) if s != 0}
print("\n=== Training V2B (40 epochs, uniform ±5 augmentation) ===")
train_model(train_seqs, val_seqs, tokenizer, device,
            num_epochs=40,
            augment_scheme=v2b_scheme,
            save_path=CHECKPOINT_DIR / 'v2b_transformer_best.pt',
            losses_path=CHECKPOINT_DIR / 'v2b_losses.json')


# ── CELL 6: TRAIN V2C (key-weighted augmentation, 12 epochs) ─
v2c_scheme = {-5:1, -4:1, -3:1, -2:2, -1:3, 1:3, 2:2, 3:1, 4:1, 5:1}
print("\n=== Training V2C (40 epochs, key-weighted augmentation) ===")
train_model(train_seqs, val_seqs, tokenizer, device,
            num_epochs=40,
            augment_scheme=v2c_scheme,
            save_path=CHECKPOINT_DIR / 'v2c_transformer_best.pt',
            losses_path=CHECKPOINT_DIR / 'v2c_losses.json')


# ── CELL 7: Music metrics evaluation ─────────────────────────
import torch.nn.functional as F
from collections import defaultdict

def kl_div(p, q, eps=1e-8):
    p = np.array(p, dtype=float) + eps; q = np.array(q, dtype=float) + eps
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))

def get_scale_notes(root, scale_type='major'):
    intervals = [0,2,4,5,7,9,11] if scale_type == 'major' else [0,2,3,5,7,8,10]
    return {(root + i) % 12 for i in intervals}

def scale_consistency(notes):
    if not notes: return 0.0
    pcs = [n % 12 for n in notes]
    best = 0.0
    for root in range(12):
        for stype in ('major', 'minor'):
            scale = get_scale_notes(root, stype)
            best = max(best, sum(1 for pc in pcs if pc in scale) / len(pcs))
    return best

def parse_v2_tokens(token_ids, tokenizer):
    """Return list of (voice, midi, dur_sixteenths) tuples."""
    notes = []
    for tid in token_ids:
        tok = tokenizer.id_to_token.get(tid, '')
        if tok.startswith('V') and 'P' in tok and 'D' in tok:
            try:
                after_v = tok[1:]
                v = int(after_v.split('P')[0])
                midi_str, dur_str = after_v.split('P')[1].split('D')
                notes.append((v, int(midi_str), int(dur_str)))
            except (ValueError, IndexError):
                pass
    return notes

def compute_metrics_v2(samples_tokens, tokenizer, ref_hist=None):
    all_midi, all_pcs, all_durs = [], [], []
    voice_steps = [[] for _ in range(4)]
    consecutive = [[] for _ in range(4)]
    leaps = crossings = parallel5 = total_pairs = 0
    consonant = total_ints = 0

    CONSONANT = {0,3,4,7,8,9}

    for token_ids in samples_tokens:
        notes = parse_v2_tokens(token_ids, tokenizer)
        by_voice = defaultdict(list)
        for v, midi, dur in notes:
            by_voice[v].append((midi, dur))
            all_midi.append(midi); all_pcs.append(midi % 12)
            all_durs.append(dur)

        for v in range(4):
            vn = by_voice[v]
            for i in range(1, len(vn)):
                step = abs(vn[i][0] - vn[i-1][0])
                voice_steps[v].append(step)
                if step > 2: leaps += 1
                total_pairs += 1

        # voice crossing (soprano < alto etc)
        for pos in range(min(len(by_voice[v]) for v in range(4)) if all(v in by_voice for v in range(4)) else 0):
            pitches = [by_voice[v][pos][0] for v in range(4) if pos < len(by_voice[v])]
            if len(pitches) == 4:
                for vi in range(3):
                    if pitches[vi] < pitches[vi+1]:
                        crossings += 1

        # consonance + parallel 5ths between soprano/bass
        sb_s = by_voice[0]; sb_b = by_voice[3]
        for i in range(min(len(sb_s), len(sb_b))):
            intv = abs(sb_s[i][0] - sb_b[i][0]) % 12
            if intv in CONSONANT: consonant += 1
            total_ints += 1
            if i > 0 and len(sb_s) > 1 and len(sb_b) > 1:
                prev_intv = abs(sb_s[i-1][0] - sb_b[i-1][0]) % 12
                if prev_intv == 7 and intv == 7:
                    parallel5 += 1

    pc_hist = [all_pcs.count(pc) / max(1, len(all_pcs)) for pc in range(12)]
    dur_hist_bins = [1, 2, 4, 8, 16, 32]
    dur_hist = [sum(1 for d in all_durs if d == b) / max(1, len(all_durs)) for b in dur_hist_bins]

    avg_step = np.mean([s for sv in voice_steps for s in sv]) if any(voice_steps) else 0.0
    leap_ratio = leaps / max(1, total_pairs)
    vc_ratio = crossings / max(1, len(samples_tokens))
    cons = consonant / max(1, total_ints)
    p5_ratio = parallel5 / max(1, total_ints)

    note_density = len(all_midi) / max(1, len(samples_tokens))
    avg_dur = np.mean(all_durs) * 0.25 if all_durs else 0.0  # quarter notes

    kl_pc = kl_div(pc_hist, ref_hist) if ref_hist is not None else None
    kl_dur = None  # skip for brevity

    return {
        'pitch_class_histogram': pc_hist,
        'scale_consistency': scale_consistency(all_midi),
        'unique_pitches_per_sample': len(set(all_midi)) / max(1, len(samples_tokens)),
        'pitch_range': (max(all_midi) - min(all_midi)) if all_midi else 0,
        'note_density': note_density,
        'avg_note_duration': avg_dur,
        'duration_histogram': dur_hist,
        'avg_voice_step': float(avg_step),
        'leap_ratio': float(leap_ratio),
        'voice_crossing_ratio': float(vc_ratio),
        'consonance_score': float(cons),
        'parallel_fifth_ratio': float(p5_ratio),
        'pitch_class_kl_div': float(kl_pc) if kl_pc is not None else None,
    }

def generate_samples(checkpoint_path, tokenizer, device, n=20, max_new=480):
    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size, d_model=128,
                               n_heads=8, n_layers=4, context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model = model.to(device).eval()
    start_id = tokenizer.token_to_id['<START>']
    samples = []
    for _ in range(n):
        ids = model.generate([start_id], max_new_tokens=max_new, temperature=1.0, device=device)
        samples.append(ids)
    return samples

print("\n=== Computing music metrics ===")

# Reference: real Bach test set tokens
ref_tokens = test_seqs
ref_metrics = compute_metrics_v2(ref_tokens, tokenizer)
ref_pc_hist = ref_metrics['pitch_class_histogram']

results = {}
for name, ckpt in [('v2b', 'v2b_transformer_best.pt'), ('v2c', 'v2c_transformer_best.pt')]:
    path = CHECKPOINT_DIR / ckpt
    if not path.exists():
        print(f"  Skipping {name} — {ckpt} not found")
        continue
    print(f"  Generating 20 samples from {name}...")
    samples = generate_samples(path, tokenizer, device, n=20)
    metrics = compute_metrics_v2(samples, tokenizer, ref_pc_hist)
    results[name] = metrics
    out = CHECKPOINT_DIR / f'{name}_music_metrics_colab.json'
    with open(out, 'w') as f:
        json.dump({'version': name, 'metrics': metrics, 'ref_metrics': ref_metrics}, f, indent=2)
    print(f"  Saved {out.name} | KL={metrics['pitch_class_kl_div']:.4f} | consonance={metrics['consonance_score']:.4f}")

print("Metrics done.")


# ── CELL 8: Package and download ─────────────────────────────
import zipfile, os
from google.colab import files

zip_path = '/content/task1_artifacts.zip'
to_collect = [
    'v2b_transformer_best.pt',
    'v2b_losses.json',
    'v2b_music_metrics_colab.json',
    'v2c_transformer_best.pt',
    'v2c_losses.json',
    'v2c_music_metrics_colab.json',
]

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in to_collect:
        fpath = CHECKPOINT_DIR / fname
        if fpath.exists():
            zf.write(fpath, fname)
            print(f"  + {fname} ({fpath.stat().st_size // 1024} KB)")
        else:
            print(f"  MISSING: {fname}")

print(f"\nZip size: {Path(zip_path).stat().st_size // 1024} KB")
files.download(zip_path)
print("Download started — check your browser.")
