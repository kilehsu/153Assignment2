# CLAUDE.md — Project-wide instructions for Claude Code

---

## Project structure

```
153Assignment2/
├── CLAUDE.md                ← this file
├── instructions.txt         ← assignment rubric
├── notebook.ipynb           ← main deliverable notebook (Task 1 complete)
├── notebook_task2.ipynb     ← Task 2 notebook (in progress)
├── images/                  ← all saved figures (png)
│
├── EDA/                     ← Task 1 EDA scripts (complete)
├── modeling/                ← Task 1 model, checkpoints
├── evaluation/              ← Task 1 generation + WAV output
│
├── EDA_task2/               ← Task 2 EDA scripts + results JSON
├── modeling_task2/          ← Task 2 seq2seq model, training, generation
├── evaluation_task2/        ← Task 2 MIDI/WAV output, metrics
└── colab/                   ← Colab notebooks for GPU training
```

---

## Task 1 — Symbolic unconditioned generation (COMPLETE)

- Model: GPT-style ChoraleTransformer, interleaved 4-voice BachTokenizerV2
- Best checkpoint: `modeling/checkpoints/v2c_transformer_best.pt` (val loss 0.527, 12 epochs, Colab A100)
- Generation: `evaluation/generate_v2c_keyed.py` — key-detection + chromatic soft masking, `max_new_tokens=480`
- All audio: native `<audio controls>` WAV tags with base64-encoded WAV — never use html-midi-player
- Notebook cells 1–53 complete; do not restructure

---

## Task 2 — Symbolic conditioned generation: Harmonization

### Concept

Given a **melody** track (POP909 track 0), generate a **piano accompaniment** (4 simultaneous pitches per timestep, from POP909 track 2). This is a seq2seq melody-to-piano task.

### Architecture

**`modeling_task2/harmonizer_model.py`** — `HarmonizerSeq2Seq`
- **Encoder**: `MelodyEncoder` — bidirectional LSTM (embed_dim=128, hidden_dim=256, num_layers=2)
- **Decoder**: `ChordDecoder` — unidirectional LSTM with Bahdanau attention; generates 4 pitches per step
- **Vocab**: 130 tokens (MIDI pitches 0–128, 128=REST, 0=padding)
- **Input**: melody sequence (seq_len=64 16th-note windows)
- **Output**: piano sequence (seq_len=64, 4 voices per step)

### Data

- **Dataset**: POP909 (909 pop songs), cloned to `data/POP909/POP909`
- **Script**: `modeling_task2/pop909_dataset.py` — `POP909Dataset`
  - Quantizes to 16th notes, uses stride=32 windows of length 64
  - Discards windows with >80% REST melody
  - Cache: `modeling_task2/pop909_cache.pkl` (96 MB, already built)
- **Split**: 90% train / 10% val (seed=42)

### Training status — COMPLETE (CPU run killed; use Colab checkpoint)

- **Checkpoint**: `modeling_task2/harmonizer_best.pt` — epoch 7, val loss **3.4164**
- **Hyperparams**: embed_dim=128, hidden_dim=256, num_layers=2, dropout=0.3, batch_size=64, lr=1e-3
- **Loss history**: `modeling_task2/harmonizer_losses.json`
- **Colab notebook**: `colab/train_harmonizer_colab.ipynb` — GPU training (~15 min on T4)
  - After Colab run, download `harmonizer_best.pt` and drop into `modeling_task2/`

### Generation

- **Script**: `modeling_task2/generate_harmony.py`
- **Outputs** (already generated for songs 0, 1, 2):
  - `evaluation_task2/harmonized_N_piano.mid` — model output
  - `evaluation_task2/harmonized_N_satb.mid` — melody + harmony combined
- Run: `python3 modeling_task2/generate_harmony.py --song <folder_name>`

### Evaluation

- **Script**: `evaluation_task2/run_eval.py`
- **Metrics**: `evaluation_task2/harmony_metrics.json` (already computed)
- **Plot script**: `evaluation_task2/plot_metrics.py`
- Run: `cd evaluation_task2 && python3 run_eval.py`

### Generation config (best found)

`modeling_task2/best_config.json`:
- penalty=1.5, bias=2.0, top_p=0.95 ("All Three" config)

### Rubric checklist (2 pts each, 8 pts total)

- [ ] **EDA**: melody range histogram, chord vocabulary size, key/tempo coverage
- [ ] **Modeling**: seq2seq formulation, encoder-decoder diagram, code walkthrough
- [ ] **Evaluation**: baselines (random, unigram), metric table vs baseline
- [ ] **Related Work**: DeepBach (Hadjeres 2017), Coconet (Huang 2017), comparison to ours

---

## Notebook organisation

- **`notebook.ipynb` cells 1–53**: Task 1 (complete — do not restructure)
- **`notebook_task2.ipynb`**: Task 2 (merge into notebook.ipynb at cell 54+)
  - §6 Task 2 Introduction
  - §6.1 EDA (melody/piano distribution)
  - §6.2 Modeling (seq2seq, attention)
  - §6.3 Evaluation (metrics, baseline comparison)
  - §6.4 Related Work (DeepBach, Coconet)
  - §6.5 Generated samples (native `<audio>` WAV)

All audio cells must use native `<audio controls>` tags with base64-encoded WAV. Never use html-midi-player.

---

## WAV rendering — IMPORTANT

- **Soundfont**: `modeling/checkpoints/MuseScore_General.sf3`
- **fluidsynth binary**: `/usr/local/bin/fluidsynth` (x86_64 under Rosetta)
- **pyfluidsynth / pm.fluidsynth() will CRASH** — arm64 Python cannot dlopen the x86 libfluidsynth.dylib
- **Always use CLI subprocess** with this exact argument order:

```python
subprocess.run(
    ['/usr/local/bin/fluidsynth', '-ni', '-F', wav_path, '-r', '44100', SOUNDFONT, mid_path],
    capture_output=True, timeout=180
)
```

Use `shutil.which('fluidsynth')` in notebooks to auto-detect the path.

---

## Code conventions

- Figures saved to `images/` with descriptive names (e.g. `images/task2_melody_range.png`)
- Evaluation scripts are standalone, runnable from `evaluation_task2/` with `sys.path.insert(0, '../modeling_task2')`
- `max_new_tokens=480` for Task 1 generation (context_len=512, 1 START token)
