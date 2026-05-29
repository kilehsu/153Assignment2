# POP909 Harmonizer Training on Colab

## Overview

This directory contains everything needed to train a seq2seq harmonizer model on the POP909 dataset using Google Colab.

## What is the model?

- **Task**: Given a melody, generate piano accompaniment
- **Architecture**: Bidirectional LSTM encoder (melody) + LSTM decoder (piano) with Bahdanau attention
- **Input**: Melody sequence (sequence of pitches, 0-128)
- **Output**: Piano sequence (4 simultaneous pitches per timestep, 0-128 each)
- **Quantization**: 16th notes (64 16th-note slots ≈ 4 bars at 120 BPM)

## Setup

### Step 1: Prepare the data

On your local machine:

```bash
cd /Users/kilehsu/153Assignment2/modeling_task2
python pop909_dataset.py
```

This will create `pop909_cache.pkl` (a cached, pre-processed dataset).

### Step 2: Open Colab

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### Step 3: Upload cache file

1. In the Colab left sidebar, click the Files icon
2. Click "Upload to session storage"
3. Select `pop909_cache.pkl` from `modeling_task2/`

### Step 4: Run training

Copy the contents of `train_harmonizer_colab.py` into separate Colab cells (cells are marked with comments like `# ── CELL 1 ──`):

- **CELL 1**: Install dependencies
- **CELL 2**: Define dataset class
- **CELL 3**: Define model classes (MelodyEncoder, ChordDecoder, HarmonizerSeq2Seq)
- **CELL 4**: Load data and create dataloaders
- **CELL 5**: Create model and optimizer
- **CELL 6**: Run training loop (estimated 20-30 min on T4 GPU)
- **CELL 7**: Download checkpoint

Expected behavior:
```
Epoch 1/30
  Train loss: 3.4521
  Val loss: 3.2103
  Saved best checkpoint
...
Epoch 30/30
  Train loss: 1.2341
  Val loss: 1.1856

Training complete!
Best val loss: 0.9876 at epoch 27
```

### Step 5: Download checkpoint

In the Colab Files pane (left sidebar), download:
- `harmonizer_best.pt` (the trained model)
- `harmonizer_losses.json` (training curves)

Save these to `modeling_task2/`.

## Running inference locally

Once you have the checkpoint:

```bash
cd /Users/kilehsu/153Assignment2/modeling_task2

# Generate for song 001
python generate_harmony.py --song 001

# Generate for song 042 with custom output
python generate_harmony.py --song 042 --output /tmp/harmonies

# Generate dummy output if checkpoint doesn't exist yet
python generate_harmony.py --song 001 --dummy
```

Output:
- `evaluation_task2/harmony_001.mid` — generated piano + original melody
- `evaluation_task2/harmony_001_original.mid` — original piano for comparison
- `evaluation_task2/harmony_001.wav` — audio rendering (if fluidsynth installed)

## Expected results

- **Train dataset**: ~64k windows (from all 909 songs, 80/20 train/val split)
- **Val dataset**: ~7k windows
- **Training time**: ~25 min on T4 GPU
- **Final val loss**: ~0.9-1.2 (CE loss)

## Troubleshooting

### Checkpoint not found error

If `harmonizer_best.pt` doesn't exist and you run `generate_harmony.py`, either:
1. Train a model first (follow steps above)
2. Use `--dummy` flag to generate random output for testing: `python generate_harmony.py --song 001 --dummy`

### Out of memory on Colab

Reduce `batch_size` in CELL 5 from 64 to 32.

### WAV conversion fails

The generation script tries to use `fluidsynth` to convert MIDI to WAV. If it's not installed:
- On Colab: `!apt-get install fluidsynth` in a cell
- On Mac: `brew install fluidsynth`
- On Linux: `sudo apt-get install fluidsynth`

If `MuseScore_General.sf3` soundfont is not available, WAV conversion will be skipped (MIDI files will still be generated).

## Files included

- `train_harmonizer_colab.py` — Colab training script (copy cells to notebook)
- `README.md` — This file

## Local training (alternative)

If you have a local GPU, you can also train locally:

```bash
cd /Users/kilehsu/153Assignment2
python modeling_task2/train_harmonizer.py
```

This uses the same model and data pipeline but runs on your machine.
