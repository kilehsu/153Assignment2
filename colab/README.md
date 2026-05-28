# Running V2C Training on Google Colab

Expected time: ~45-60 min on T4 (free), ~15-20 min on A100 (Colab Pro)

---

## Step 1 — Open a Colab notebook

Go to https://colab.research.google.com → New notebook.
Make sure you have a GPU: Runtime → Change runtime type → T4 GPU.

---

## Step 2 — Install dependencies

Paste into a cell and run:

```python
!pip install music21 torch --quiet
```

---

## Step 3 — Upload these 6 files from your Mac

Paste into a cell and run:

```python
from google.colab import files
uploaded = files.upload()
```

When the file picker opens, select all 6 of these files at once:

| File | Location on your Mac |
|------|----------------------|
| `model.py` | `153Assignment2/modeling/model.py` |
| `dataset.py` | `153Assignment2/modeling/dataset.py` |
| `tokenizer_v2.py` | `153Assignment2/modeling/tokenizer_v2.py` |
| `v2_tokenizer.json` | `153Assignment2/modeling/checkpoints/v2_tokenizer.json` |
| `v2_sequences_cache.pkl` | `153Assignment2/modeling/checkpoints/v2_sequences_cache.pkl` |
| `v2_splits.json` | `153Assignment2/modeling/checkpoints/v2_splits.json` |

---

## Step 4 — Run the training script

Paste the entire contents of `train_v2c_colab.py` into a cell and run it.
You'll see batch-level progress every 200 batches and epoch summaries.

---

## Step 5 — Download the checkpoint

After training finishes, paste into a cell and run:

```python
from google.colab import files
files.download('/content/v2c_transformer_best.pt')
files.download('/content/v2c_losses.json')
```

Move the downloaded files into `153Assignment2/modeling/checkpoints/`.

---

## Step 6 — Run evaluation back on your Mac

```bash
cd /Users/kilehsu/153Assignment2/modeling
python evaluate_music_metrics.py --version v2c
```

Then compare `checkpoints/v2c_music_metrics.json` against `v2b_music_metrics.json`.
Key metrics to check: `pitch_class_kl_div` (target < 0.06) and `unique_token_ratio` (target < 0.15).
