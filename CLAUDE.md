# CLAUDE.md — Project-wide instructions for Claude Code

## Agent policy
- **Always spawn Haiku subagents** (`model: "haiku"`) for any research, file exploration, code writing,
  or analysis task that can be delegated. Only the top-level orchestrator runs on Sonnet/Opus.
- Before spawning, check whether an existing agent in this session can be continued via SendMessage.
- Subagents write results back to files; the orchestrator summarises and reports to the user.

---

## Project structure

```
153Assignment2/
├── CLAUDE.md            ← this file
├── instructions.txt     ← assignment rubric
├── notebook.ipynb       ← single deliverable notebook (Task 1 + Task 2)
├── images/              ← all saved figures (png)
│
├── EDA/                 ← Task 1 EDA scripts (already complete)
├── modeling/            ← Task 1 model code, training scripts, checkpoints
├── evaluation/          ← Task 1 generation + WAV output
│
├── EDA_task2/           ← Task 2 EDA (soprano range, chord vocab, coverage stats)
├── modeling_task2/      ← Task 2 model / conditioned generation code
└── evaluation_task2/    ← Task 2 generated MIDI + WAV + metric outputs
```

Create the task2 folders if they don't exist before writing files into them.

---

## Task 1 — Symbolic unconditioned generation (DONE)

- Model: GPT-style ChoraleTransformer, interleaved 4-voice BachTokenizerV2
- Best checkpoint: `modeling/checkpoints/v2c_transformer_best.pt` (val loss 0.527, 12 epochs, Colab A100)
- Generation: `evaluation/generate_v2c_keyed.py` — key-detection + chromatic soft masking, `max_new_tokens=480`
- All audio: native `<audio>` WAV tags (no html-midi-player anywhere)
- Rubric sections covered: EDA, Modeling, Evaluation, Related Work

---

## Task 2 — Symbolic conditioned generation: Harmonization (NO RETRAINING NEEDED)

### Concept
Given a **soprano melody** (voice 0 tokens), generate the remaining three voices
(Alto, Tenor, Bass — voices 1-3) that harmonize it.

We reuse the existing `v2c_transformer_best.pt` via **prefix-conditioning + constrained decoding**:
- Extract soprano token sequences from Bach test chorales
- Feed them as a prompt; at each 4-token group, lock voice 0 to the known soprano token,
  freely sample voices 1-3 from the model's distribution
- No GPU training required

### Folder responsibilities

| Folder | Contents | Subagent task |
|---|---|---|
| `EDA_task2/` | `soprano_range.py`, `chord_vocab.py`, `harmony_stats.py` | Analyze soprano pitch range, common chord progressions in test set, interval distribution between melody and bass |
| `modeling_task2/` | `harmonize.py` — constrained generation loop | Lock voice 0 at each step; sample voices 1-3; enforce in-key constraint from v2c_keyed logic |
| `evaluation_task2/` | MIDI + WAV output, `evaluate_harmony.py` | Compute: voice leading errors, parallel 5ths/8ths rate, pitch class KL vs Bach reference, listen test |

### Rubric checklist (2 pts each, 8 pts total)
- [ ] **EDA**: soprano range histogram, chord vocabulary size, key coverage in test set
- [ ] **Modeling**: formulation (p(alto,tenor,bass | soprano)), prefix-conditioning diagram, code walkthrough
- [ ] **Evaluation**: baselines (random harmonization, unigram-sampled chords), metric table vs baseline + vs Bach original
- [ ] **Related Work**: DeepBach (Hadjeres 2017) harmonization results, Coconet (Huang 2017), compare to ours

### Key implementation note
In `BachTokenizerV2`, tokens at positions `[4k, 4k+1, 4k+2, 4k+3]` correspond to voices `[0, 1, 2, 3]`.
During harmonization generation, at position `4k` (voice 0), force the model's next token to the
known soprano token instead of sampling. At positions `4k+1`, `4k+2`, `4k+3`, sample normally.

---

## Notebook organisation (notebook.ipynb)

- **Cells 1–52**: Task 1 (complete — do not restructure)
- **Cell 53**: §5 Related Work (Task 1)
- **Cells 54+**: Task 2 begins here
  - §6 Task 2 Introduction
  - §6.1 EDA (soprano distribution, chord vocab)
  - §6.2 Modeling (prefix-conditioning approach)
  - §6.3 Evaluation (metrics, baseline comparison)
  - §6.4 Related Work (DeepBach, Coconet)
  - §6.5 Generated samples (native `<audio>` WAV)

All audio cells must use native `<audio controls>` tags with base64-encoded WAV. Never use html-midi-player.

---

## Code conventions

- Always use `max_new_tokens=480` for generation (context_len=512, 1 START token)
- Soundfont: `modeling/checkpoints/MuseScore_General.sf3` for all WAV rendering
- Save figures to `images/` with descriptive names (e.g. `images/task2_soprano_range.png`)
- Keep evaluation scripts standalone and runnable from `evaluation_task2/` with `sys.path.insert(0, '../modeling')`
