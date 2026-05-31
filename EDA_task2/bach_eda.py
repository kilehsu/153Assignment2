"""
Task 2 EDA — Bach chorale dataset.
Produces figures saved to images/:
  - task2_soprano_range.png
  - task2_satb_ranges.png
  - task2_intervals.png
"""

import sys, os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / 'modeling'))
from tokenizer_v2 import BachTokenizerV2

CKPT   = Path(__file__).parent.parent / 'modeling' / 'checkpoints'
IMAGES = Path(__file__).parent.parent / 'images'
IMAGES.mkdir(exist_ok=True)

tokenizer = BachTokenizerV2.load(str(CKPT / 'v2_tokenizer.json'))
with open(CKPT / 'v2_sequences_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
seqs = cache['sequences']
with open(CKPT / 'v2_splits.json') as f:
    splits = json.load(f)

print(f"Vocab size    : {tokenizer.vocab_size}")
print(f"Sequences     : {len(seqs)}")
print(f"Train/Val/Test: {len(splits['train'])} / {len(splits['val'])} / {len(splits['test'])}")

# ── collect per-voice pitch data ──────────────────────────────────────────────
voice_pitches = {v: [] for v in range(4)}
voice_names   = ['Soprano', 'Alto', 'Tenor', 'Bass']

for idx in splits['train'] + splits['val'] + splits['test']:
    decoded = tokenizer.decode(seqs[idx])
    for tok in decoded:
        for v in range(4):
            prefix = f'V{v}P'
            if tok.startswith(prefix) and 'D' in tok:
                try:
                    midi = int(tok[len(prefix):].split('D')[0])
                    voice_pitches[v].append(midi)
                except ValueError:
                    pass

for v in range(4):
    p = voice_pitches[v]
    print(f"{voice_names[v]:8s}: n={len(p):6,}  range={min(p)}-{max(p)}  median={int(np.median(p))}")

# ── Figure 1: Soprano pitch histogram ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(voice_pitches[0], bins=range(55, 86), color='royalblue',
        edgecolor='white', lw=0.3)
ax.axvline(np.median(voice_pitches[0]), color='red', ls='--',
           label=f"Median: {int(np.median(voice_pitches[0]))}")
ax.set_xlabel('MIDI Pitch')
ax.set_ylabel('Note Count')
ax.set_title('Soprano Pitch Distribution (Bach Chorales)')
ax.legend()

note_labels = {60:'C4',62:'D4',64:'E4',65:'F4',67:'G4',69:'A4',71:'B4',
               72:'C5',74:'D5',76:'E5',77:'F5',79:'G5',81:'A5'}
ax2 = ax.secondary_xaxis('top')
ticks = [k for k in note_labels if 57 <= k <= 83]
ax2.set_xticks(ticks)
ax2.set_xticklabels([note_labels[k] for k in ticks], fontsize=7, rotation=45)

plt.tight_layout()
out = IMAGES / 'task2_soprano_range.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# ── Figure 2: SATB box plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
colors = ['royalblue', 'coral', 'mediumseagreen', 'orchid']
bp = ax.boxplot([voice_pitches[v] for v in range(4)], patch_artist=True,
                medianprops=dict(color='black', lw=2))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax.set_xticklabels(voice_names, fontsize=12)
ax.set_ylabel('MIDI Pitch')
ax.set_title('SATB Voice Pitch Ranges (Bach Chorales)')

# Add note name labels on y-axis
yticks = [36, 48, 52, 55, 60, 64, 67, 72, 76, 79, 84]
ylabels = ['C2','C3','E3','G3','C4','E4','G4','C5','E5','G5','C6']
ax.set_yticks(yticks)
ax.set_yticklabels(ylabels, fontsize=8)
ax.yaxis.grid(True, alpha=0.3)

plt.tight_layout()
out = IMAGES / 'task2_satb_ranges.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

# ── Figure 3: Soprano-Bass interval distribution ──────────────────────────────
intervals = []
for idx in splits['train'] + splits['test']:
    decoded = tokenizer.decode(seqs[idx])
    note_toks = [t for t in decoded if t not in ('<START>', '<END>', '<BAR>', '<SUSTAIN>')]
    for i in range(0, len(note_toks) - 3, 4):
        block = note_toks[i:i+4]
        sop, bass = None, None
        for tok in block:
            if tok.startswith('V0P') and 'D' in tok:
                try: sop = int(tok[3:].split('D')[0])
                except: pass
            if tok.startswith('V3P') and 'D' in tok:
                try: bass = int(tok[3:].split('D')[0])
                except: pass
        if sop and bass and sop > bass:
            intervals.append(sop - bass)

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(intervals, bins=range(0, 40), color='steelblue', edgecolor='white', lw=0.3)
ax.set_xlabel('Semitones (Soprano − Bass)')
ax.set_ylabel('Count')
ax.set_title('Soprano–Bass Interval Distribution (Bach Chorales)')
for interval, name in [(7,'P5'), (12,'P8'), (19,'P12')]:
    ax.axvline(interval, color='red', ls=':', alpha=0.7)
    ax.text(interval + 0.2, ax.get_ylim()[1] * 0.85, name, fontsize=9, color='red')
plt.tight_layout()
out = IMAGES / 'task2_intervals.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {out}")

print(f"\nMean S-B interval: {np.mean(intervals):.1f} semitones  (std {np.std(intervals):.1f})")
print("EDA complete.")
