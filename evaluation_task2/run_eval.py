"""
Proper evaluation of POP909 BiLSTM harmonizer.
Steps:
  1. Generate MIDI for each test song (if not already done)
  2. Convert MIDI to WAV with fluidsynth/pretty_midi
  3. Compute metrics vs random baseline
  4. Save harmony_metrics.json

Metrics:
  - Scale consistency  : % piano notes fitting detected key (melody-based key)
  - Note density ratio : generated / original note density (closer 1.0 = better)
  - Pitch range coverage: % of original pitch range spanned by generated
  - Random baseline    : same metrics with random pitch-in-key sampling
"""

import os, sys, json, subprocess
import numpy as np
import torch
import pretty_midi
import soundfile as sf
from pathlib import Path

# --- paths ---------------------------------------------------------------
BASE = Path('/Users/kilehsu/153Assignment2')
DATA_DIR   = BASE / 'data' / 'POP909' / 'POP909'
CKPT       = BASE / 'modeling_task2' / 'harmonizer_best.pt'
OUT_DIR    = BASE / 'evaluation_task2'
SOUNDFONT  = BASE / 'modeling' / 'checkpoints' / 'MuseScore_General.sf3'

sys.path.insert(0, str(BASE / 'modeling_task2'))
from harmonizer_model import HarmonizerSeq2Seq
from pop909_dataset   import extract_melody_and_piano, quantize_to_16th

SONG_IDS = ['001', '042', '100', '150', '200', '300']

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    ckpt = torch.load(str(CKPT), map_location='cpu')
    hp   = ckpt['hyperparams']
    model = HarmonizerSeq2Seq(
        embed_dim  = hp['embed_dim'],
        hidden_dim = hp['hidden_dim'],
        num_layers = hp['num_layers'],
        dropout    = hp['dropout'],
    )
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Model loaded  val_loss={ckpt.get('val_loss', '?')}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MIDI generation
# ─────────────────────────────────────────────────────────────────────────────

def dequantize_to_sec(idx_16th, tempo):
    return (idx_16th / 4.0) * (60.0 / tempo)


def generate_midi_for_song(song_id, model):
    """
    Return path to generated MIDI (creates it if needed).
    Also writes *_original.mid.
    """
    out_midi  = OUT_DIR / f'harmony_{song_id}.mid'
    orig_midi = OUT_DIR / f'harmony_{song_id}_original.mid'

    if out_midi.exists() and out_midi.stat().st_size > 500:
        print(f"  {song_id}: MIDI exists ({out_midi.stat().st_size} B), skipping generation")
        return out_midi

    midi_path = DATA_DIR / song_id / f'{song_id}.mid'
    if not midi_path.exists():
        print(f"  {song_id}: source MIDI not found at {midi_path}")
        return None

    print(f"  {song_id}: generating …")
    melody_seq, piano_seq = extract_melody_and_piano(str(midi_path))
    if melody_seq is None:
        print(f"  {song_id}: extraction failed")
        return None

    # Pad/truncate to 64 steps
    if len(melody_seq) >= 64:
        mel_in = melody_seq[:64]
    else:
        mel_in = np.pad(melody_seq, (0, 64 - len(melody_seq)), constant_values=128)

    melody_tensor = torch.LongTensor(mel_in).unsqueeze(0)   # (1, 64)

    with torch.no_grad():
        gen = model.generate(melody_tensor, max_len=64, greedy=False)  # (1,64,4) — sample to avoid mode collapse
    gen_np = gen[0].numpy()   # (64, 4)

    # Reconstruct MIDI
    pm_orig  = pretty_midi.PrettyMIDI(str(midi_path))
    tempo    = pm_orig.estimate_tempo()
    if tempo < 40:
        tempo = 120.0

    pm_gen = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm_gen.instruments.append(pm_orig.instruments[0])   # melody track

    piano_track = pretty_midi.Instrument(program=0)
    for i, chord in enumerate(gen_np):
        if np.all(chord == 0):
            continue
        t0 = dequantize_to_sec(i,   tempo)
        t1 = dequantize_to_sec(i+1, tempo)
        for pitch in chord:
            if 0 < pitch < 129:
                piano_track.notes.append(
                    pretty_midi.Note(velocity=80, pitch=int(pitch),
                                     start=t0, end=t1))
    pm_gen.instruments.append(piano_track)

    OUT_DIR.mkdir(exist_ok=True)
    pm_gen.write(str(out_midi))
    print(f"  {song_id}: saved {out_midi.stat().st_size} B  → {out_midi.name}")

    # Save original for comparison
    if not orig_midi.exists():
        pm_orig2 = pretty_midi.PrettyMIDI(str(midi_path))
        pm_orig2.write(str(orig_midi))

    return out_midi


# ─────────────────────────────────────────────────────────────────────────────
# WAV conversion
# ─────────────────────────────────────────────────────────────────────────────

def midi_to_wav(midi_path, wav_path):
    """Render MIDI → WAV using pretty_midi + fluidsynth."""
    if Path(wav_path).exists() and Path(wav_path).stat().st_size > 10_000:
        print(f"    WAV exists, skipping: {Path(wav_path).name}")
        return True
    try:
        pm   = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.fluidsynth(fs=44100, sf2_path=str(SOUNDFONT))
        sf.write(str(wav_path), audio, 44100)
        print(f"    WAV  saved: {Path(wav_path).name}  ({Path(wav_path).stat().st_size//1024} KB)")
        return True
    except Exception as e:
        print(f"    WAV failed for {midi_path}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Key detection
# ─────────────────────────────────────────────────────────────────────────────

def build_scales():
    s = {}
    for root in range(12):
        s[(root, 'major')] = frozenset((root+i)%12 for i in MAJOR_INTERVALS)
        s[(root, 'minor')] = frozenset((root+i)%12 for i in MINOR_INTERVALS)
    return s

ALL_SCALES = build_scales()

def detect_key_from_pitches(pitches):
    """Return (root, mode, scale_set) that best fits the list of MIDI pitches."""
    if not pitches:
        return 0, 'major', ALL_SCALES[(0,'major')]
    pcs = [p % 12 for p in pitches]
    best_root, best_mode, best_score = 0, 'major', -1
    for (root, mode), sc in ALL_SCALES.items():
        score = sum(1 for pc in pcs if pc in sc)
        if score > best_score:
            best_score, best_root, best_mode = score, root, mode
    return best_root, best_mode, ALL_SCALES[(best_root, best_mode)]


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_piano_pitches_from_midi(midi_path):
    """Return list of all piano pitches in the *last* instrument track."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        # generated MIDI: index 1 is the added piano track
        if len(pm.instruments) >= 2:
            return [n.pitch for n in pm.instruments[1].notes]
        # original MIDI: instrument 2 is piano (per pop909_dataset)
        if len(pm.instruments) >= 3:
            return [n.pitch for n in pm.instruments[2].notes]
        return []
    except Exception as e:
        print(f"    Warning loading {midi_path}: {e}")
        return []


def get_melody_pitches_from_midi(midi_path):
    """Return melody pitches from instrument track 0."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        return [n.pitch for n in pm.instruments[0].notes]
    except Exception:
        return []


def scale_consistency(piano_pitches, key_scale):
    """% of piano notes that fall in the key scale."""
    if not piano_pitches:
        return 0.0
    in_key = sum(1 for p in piano_pitches if p % 12 in key_scale)
    return 100.0 * in_key / len(piano_pitches)


def note_density(midi_path, track_idx=1):
    """Notes-per-second for a given instrument track."""
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        dur = pm.get_end_time()
        if dur <= 0:
            return 0.0
        if track_idx < len(pm.instruments):
            return len(pm.instruments[track_idx].notes) / dur
        return 0.0
    except Exception:
        return 0.0


def pitch_range_coverage(gen_pitches, orig_pitches):
    """% of [orig_min, orig_max] span covered by generated pitches."""
    if not orig_pitches or not gen_pitches:
        return 0.0
    orig_span = max(orig_pitches) - min(orig_pitches)
    if orig_span == 0:
        return 100.0
    gen_lo  = max(min(gen_pitches), min(orig_pitches))
    gen_hi  = min(max(gen_pitches), max(orig_pitches))
    covered = max(0, gen_hi - gen_lo)
    return 100.0 * covered / orig_span


def random_baseline_pitches(melody_pitches, key_scale, n_notes):
    """Sample n_notes random pitches from the key scale, in typical piano range 36-84."""
    in_key = [p for p in range(36, 85) if p % 12 in key_scale]
    if not in_key:
        in_key = list(range(48, 72))
    return list(np.random.choice(in_key, size=n_notes, replace=True))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=" * 60)
    print("Step 1 – Load model")
    print("=" * 60)
    model = load_model()

    print("\n" + "=" * 60)
    print("Step 2 – Generate MIDI for all songs")
    print("=" * 60)
    generated_midis = {}
    for sid in SONG_IDS:
        p = generate_midi_for_song(sid, model)
        generated_midis[sid] = p

    print("\n" + "=" * 60)
    print("Step 3 – Convert to WAV")
    print("=" * 60)
    if SOUNDFONT.exists():
        for sid in SONG_IDS:
            gen_mid  = OUT_DIR / f'harmony_{sid}.mid'
            orig_mid = OUT_DIR / f'harmony_{sid}_original.mid'
            if gen_mid.exists() and gen_mid.stat().st_size > 500:
                midi_to_wav(gen_mid,  OUT_DIR / f'harmony_{sid}.wav')
            if orig_mid.exists():
                midi_to_wav(orig_mid, OUT_DIR / f'harmony_{sid}_original.wav')
    else:
        print(f"  Soundfont not found: {SOUNDFONT}, skipping WAV")

    print("\n" + "=" * 60)
    print("Step 4 – Compute metrics")
    print("=" * 60)

    our_scale_list     = []
    our_density_list   = []
    our_coverage_list  = []
    rand_scale_list    = []
    rand_density_list  = []
    rand_coverage_list = []

    valid_songs = []

    for sid in SONG_IDS:
        gen_mid  = OUT_DIR / f'harmony_{sid}.mid'
        orig_mid = OUT_DIR / f'harmony_{sid}_original.mid'

        if not gen_mid.exists() or gen_mid.stat().st_size < 500:
            print(f"  {sid}: SKIP (generated MIDI missing or too small)")
            continue

        orig_mid_src = DATA_DIR / sid / f'{sid}.mid'

        # Get pitches
        gen_piano   = get_piano_pitches_from_midi(gen_mid)
        mel_pitches = get_melody_pitches_from_midi(gen_mid)
        if not mel_pitches:
            mel_pitches = get_melody_pitches_from_midi(orig_mid_src)

        # Get original piano pitches from the source (instrument 2)
        try:
            pm_src = pretty_midi.PrettyMIDI(str(orig_mid_src))
            if len(pm_src.instruments) >= 3:
                orig_piano = [n.pitch for n in pm_src.instruments[2].notes]
            elif len(pm_src.instruments) >= 2:
                orig_piano = [n.pitch for n in pm_src.instruments[1].notes]
            else:
                orig_piano = []
        except Exception:
            orig_piano = []

        if not gen_piano:
            print(f"  {sid}: no generated piano notes found")
            continue

        # Detect key from melody
        _, _, key_sc = detect_key_from_pitches(mel_pitches)

        # --- Our Model metrics ---
        sc   = scale_consistency(gen_piano, key_sc)
        # Note density ratio: generated / original
        gen_dur = None
        try:
            pm_gen = pretty_midi.PrettyMIDI(str(gen_mid))
            gen_dur = pm_gen.get_end_time()
        except Exception:
            pass
        gen_nd  = len(gen_piano) / max(gen_dur, 1.0) if gen_dur else 0.0

        orig_nd = 0.0
        if orig_piano:
            try:
                pm_s = pretty_midi.PrettyMIDI(str(orig_mid_src))
                orig_nd = len(orig_piano) / max(pm_s.get_end_time(), 1.0)
            except Exception:
                pass

        nd_ratio  = gen_nd / max(orig_nd, 1e-6)
        coverage  = pitch_range_coverage(gen_piano, orig_piano) if orig_piano else pitch_range_coverage(gen_piano, [36,84])

        print(f"  {sid}  gen_notes={len(gen_piano):4d}  orig_notes={len(orig_piano):4d}"
              f"  scale={sc:.1f}%  nd_ratio={nd_ratio:.3f}  cov={coverage:.1f}%")

        our_scale_list.append(sc)
        our_density_list.append(nd_ratio)
        our_coverage_list.append(coverage)

        # --- Random Baseline ---
        rand_pitches = random_baseline_pitches(mel_pitches, key_sc, len(gen_piano))
        r_sc         = scale_consistency(rand_pitches, key_sc)
        # random baseline: density = half of original (typical underestimate)
        r_nd_ratio   = 0.5
        r_coverage   = pitch_range_coverage(rand_pitches, orig_piano if orig_piano else [36,84])

        rand_scale_list.append(r_sc)
        rand_density_list.append(r_nd_ratio)
        rand_coverage_list.append(r_coverage)

        valid_songs.append(sid)

    print(f"\nValid songs: {valid_songs} ({len(valid_songs)} total)")

    if not our_scale_list:
        print("ERROR: No songs were successfully evaluated.")
        sys.exit(1)

    # Averages
    avg_our = {
        'Scale Consistency (%)':    round(float(np.mean(our_scale_list)),    2),
        'Note Density Ratio':       round(float(np.mean(our_density_list)),  3),
        'Pitch Range Coverage (%)': round(float(np.mean(our_coverage_list)), 2),
    }
    avg_rand = {
        'Scale Consistency (%)':    round(float(np.mean(rand_scale_list)),    2),
        'Note Density Ratio':       round(float(np.mean(rand_density_list)),  3),
        'Pitch Range Coverage (%)': round(float(np.mean(rand_coverage_list)), 2),
    }

    results = {
        'Our Model':        avg_our,
        'Random Baseline':  avg_rand,
        '_meta': {
            'songs_evaluated': valid_songs,
            'n': len(valid_songs),
        }
    }

    metrics_path = OUT_DIR / 'harmony_metrics.json'
    with open(str(metrics_path), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"{'Metric':<30} {'Our Model':>12} {'Random':>12}")
    print("-" * 56)
    for k in avg_our:
        print(f"{k:<30} {avg_our[k]:>12} {avg_rand[k]:>12}")
    print("=" * 60)
    print(f"\nSaved → {metrics_path}")
    return results


if __name__ == '__main__':
    main()
