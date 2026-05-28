# Overnight Session Handoff
**Date:** May 26, 2026 | **Status:** v2b trained, evaluated, MIDIs generated

---

## What Was Done

### 1. New Tokenizer: Interleaved 4-Voice (`tokenizer_v2.py`)
The original model encoded all voices **sequentially** (all soprano, then all alto, etc.) — meaning it never saw harmonic context. The new v2 tokenizer encodes **all 4 voices at every 16th-note timestep**, emitting tokens like `V0P72D4` (voice 0, MIDI 72, duration 4×16ths). This lets the model learn harmony directly.

### 2. Three Models Trained

| Model | Params | Tokenization | Epochs | Best Val Loss | Outcome |
|-------|--------|--------------|--------|---------------|---------|
| v1 (original) | 1.2M | Sequential | 20 | 2.57 | Baseline |
| v2 | 5M | Interleaved | 6 (stopped) | 0.604 | Overfit hard after epoch 4 — too big for 294 chorales |
| **v2b** | 933K | Interleaved | 25 (stopped) | **0.393** | ✅ No overfitting, best model |

v2b fixes overfitting via **11× data augmentation** (transpose each chorale ±5 semitones → 3,234 sequences from 294) plus smaller model (d_model=128, 4 layers) and higher dropout (0.15).

### 3. Comprehensive Music Metrics (`evaluate_music_metrics.py`)
New eval beyond perplexity. Compares Transformer vs Markov vs real Bach on:
- **Pitch:** pitch class KL divergence, scale consistency, pitch range, unique pitches
- **Rhythm:** note density, duration KL divergence, avg note duration
- **Voice-leading (v2/v2b only):** avg step size, leap ratio, voice crossing ratio
- **Harmony (v2/v2b only):** consonance score, parallel fifth ratio
- Saves per-model JSON + 3 plots to `images/`

### 4. Generated Files
| File | Description |
|------|-------------|
| `evaluation/generated_v2b_{1,2,3}.mid` | v2b 4-voice MIDI samples |
| `evaluation/generated_v2b_{1,2,3}.wav` | Rendered WAVs |
| `checkpoints/v1_music_metrics.json` | Full v1 metrics |
| `checkpoints/v2_music_metrics.json` | Full v2 metrics |
| `checkpoints/v2b_music_metrics.json` | Full v2b metrics |
| `images/metrics_pitch_class_v2b.png` | Pitch distribution comparison |
| `images/metrics_duration_v2b.png` | Duration distribution comparison |
| `images/metrics_summary_v2b.png` | Summary bar chart |
| `images/09d_training_curves_v2b.png` | v2b training curves |

---

## Current Metric Results

| Metric | V1 Transformer | **V2b Transformer** | Markov | Real Bach |
|--------|---------------|---------------------|--------|-----------|
| Scale consistency ↑ | 0.895 | **0.923** | 0.691 | 0.933 |
| Note density | 1.076 | 1.551 | 21.5 ⚠️ | 1.675 |
| Pitch class KL div ↓ | **0.023** | 0.096 | 0.206 | — |
| Duration KL div ↓ | **0.003** | 0.080 | 1.585 | — |
| Unique token ratio | 0.287 | 0.183 | 0.488 | 0.106 |
| Consonance score ↑ | N/A | 0.890 | N/A | — |
| Parallel 5ths ↓ | N/A | 0.019 | N/A | — |

**v2b wins:** scale consistency (nearest to real Bach), note density, harmony metrics now computable  
**v1 wins:** pitch and duration KL divergence (pitch distribution closer to real Bach)

---

## Why Pitch/Duration KL Is Worse in v2b

Three root causes:

1. **Uniform transposition kills key preference.** Real Bach strongly favors keys near C major. Training on ±5 semitones makes the model key-agnostic → pitch class histogram flattens out vs real Bach's distribution.

2. **Duration encoded inside tokens.** `V0P72D4` predicts pitch AND duration jointly as one token. This is harder than predicting them independently, and errors compound. v1's `P72_D4` format works the same way but the model had 20 clean epochs on un-augmented data.

3. **SUSTAIN token dominance.** Most tokens in the interleaved sequence are `<SUSTAIN>`. The model spends capacity learning "sustain follows sustain" rather than learning note distributions.

---

## Next Move Options (Ranked by Impact)

### Option A — Voice-range constrained sampling (no retraining, quick win)
During generation, mask out any token outside each voice's valid MIDI range:
- Soprano: 60–81 (C4–A5)
- Alto: 53–74 (F3–D5)  
- Tenor: 48–69 (C3–A4)
- Bass: 40–62 (E2–D4)

This would fix voice crossings and bring pitch distribution closer to real Bach with zero retraining. Implement in `model.py` `generate()` with a per-position voice mask.

### Option B — Nucleus/top-k sampling (no retraining, quick win)
Replace temperature=1.0 with top-p=0.9 or top-k=50 during generation. Reduces the model sampling from the long tail of unlikely pitches, improving pitch KL divergence. 5-line change in `evaluate_music_metrics.py` and generation scripts.

### Option C — Key-weighted augmentation (retrain ~1-2hrs)
Instead of uniform ±5 semitones, weight transpositions towards smaller shifts (±1-2 are 3× more likely than ±5). Bach's actual key distribution peaks around C, G, D, F major — the flat/uniform distribution hurts KL.

### Option D — Separate pitch+duration tokens (retrain, bigger change)
Split `V0P72D4` into `V0P72` then `D4` as separate tokens. Halves the effective vocab for pitch prediction, making it easier to learn. Would likely improve both pitch KL and duration KL. Requires new tokenizer_v2c.py.

**Recommended:** Do A + B first (30 min, no retraining), evaluate, then decide if C is worth it.

---

## Notebook Status

**Not updated yet** — waiting until results are more final. When ready to update, add:
1. Section: v2b model — interleaved tokenization motivation + architecture diagram
2. Training curves plot (`09d_training_curves_v2b.png`) + overfitting comparison (v2 vs v2b)
3. Metrics comparison table (all 3 models vs Markov vs real Bach)
4. Generated audio embeds (WAV files)
5. Discussion: what the harmony metrics tell us that perplexity doesn't

---

## Key Files Added Overnight

```
modeling/
  tokenizer_v2.py          # Interleaved 4-voice tokenizer
  tokenizer_v3.py          # Chord-state tokenizer (not used, vocab too large)
  train_v2.py              # v2 training (overfit, don't use)
  train_v2b.py             # v2b training (good, use this)
  evaluate_v2.py           # Perplexity eval for v2
  evaluate_music_metrics.py # Full music quality metrics (v1/v2/v2b/v3)
evaluation/
  generate_v2b.py          # Generates 3 MIDI samples from v2b
  generated_v2b_{1,2,3}.mid
  generated_v2b_{1,2,3}.wav
checkpoints/
  v2b_transformer_best.pt  # Best v2b checkpoint (epoch 24, val loss 0.393)
  v2b_losses.json
  v2_tokenizer.json        # Used by v2b too
  v2_sequences_cache.pkl
  v1_music_metrics.json
  v2_music_metrics.json
  v2b_music_metrics.json
```
