"""
Generate piano accompaniment for a melody using trained harmonizer model.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pretty_midi
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from harmonizer_model import HarmonizerSeq2Seq
from pop909_dataset import extract_melody_and_piano, quantize_to_16th


def load_checkpoint(checkpoint_path):
    """Load trained model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    hyperparams = checkpoint['hyperparams']

    model = HarmonizerSeq2Seq(
        embed_dim=hyperparams['embed_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    return model


def get_song_path(song_id, data_dir):
    """Find song path by ID (e.g. '042')."""
    song_dir = os.path.join(data_dir, song_id.zfill(3))
    midi_path = os.path.join(song_dir, f"{song_id.zfill(3)}.mid")

    if not os.path.exists(midi_path):
        return None

    return midi_path


def generate_for_song(song_id, data_dir, checkpoint_path, output_dir, use_dummy=False):
    """
    Generate harmony for a song.

    Args:
        song_id: song identifier (e.g. '001', '042')
        data_dir: path to POP909 dataset
        checkpoint_path: path to trained model checkpoint
        output_dir: where to save generated MIDI
        use_dummy: if True and checkpoint missing, generate dummy output
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model = load_checkpoint(checkpoint_path)
    if model is None:
        if use_dummy:
            print(f"Checkpoint not found; generating dummy output for song {song_id}")
            model = None
        else:
            return False

    # Load original song
    midi_path = get_song_path(song_id, data_dir)
    if midi_path is None:
        print(f"Song {song_id} not found")
        return False

    print(f"Processing song {song_id}: {midi_path}")

    # Extract melody and piano
    melody_seq, piano_seq = extract_melody_and_piano(midi_path)
    if melody_seq is None:
        print(f"Failed to parse {midi_path}")
        return False

    print(f"Melody shape: {melody_seq.shape}, Piano shape: {piano_seq.shape}")

    # Find first window that has substantial melody content (not all REST=128)
    # REST tokens are 128; find first slot with a real note
    window_len = 64
    first_note_slot = 0
    for i, pitch in enumerate(melody_seq):
        if pitch != 128:
            first_note_slot = i
            break

    # Start the window at the first real melody note
    window_start = first_note_slot
    window_end = window_start + window_len

    if window_end > len(melody_seq):
        # Fall back: use the last available window
        window_start = max(0, len(melody_seq) - window_len)
        window_end = window_start + window_len

    print(f"Using window [{window_start}, {window_end}) — first_note_slot={first_note_slot}")

    # Convert to tensor
    if len(melody_seq) >= window_end:
        melody_window = melody_seq[window_start:window_end]
        melody_tensor = torch.LongTensor(melody_window).unsqueeze(0)  # (1, 64)
    else:
        # Pad
        melody_window = melody_seq[window_start:]
        pad_len = window_len - len(melody_window)
        padded = np.pad(melody_window, (0, pad_len), constant_values=128)
        melody_tensor = torch.LongTensor(padded).unsqueeze(0)

    # Load original to get timing info
    pm_original = pretty_midi.PrettyMIDI(midi_path)

    # Use actual MIDI tempo track — estimate_tempo() is unreliable for pop
    _, tempos = pm_original.get_tempo_changes()
    tempo = float(tempos[0]) if len(tempos) > 0 else 120.0
    if tempo < 40 or tempo > 300:
        tempo = 120.0
    slot_duration = 60.0 / (tempo * 4.0)
    print(f"Tempo: {tempo:.1f} BPM  →  slot: {slot_duration*1000:.1f} ms")

    # Generate piano for the FULL song in non-overlapping windows
    total_slots = len(melody_seq)
    all_generated = np.zeros((total_slots, 4), dtype=np.int64)
    if model is not None:
        for win_start in range(0, total_slots, window_len):
            chunk = melody_seq[win_start:win_start + window_len]
            actual_len = len(chunk)
            if actual_len < window_len:
                chunk = np.pad(chunk, (0, window_len - actual_len), constant_values=128)
            mel_t = torch.LongTensor(chunk).unsqueeze(0)
            with torch.no_grad():
                gen = model.generate(mel_t, max_len=window_len, greedy=False)
            all_generated[win_start:win_start + actual_len] = gen[0].numpy()[:actual_len]
    else:
        all_generated = np.random.randint(36, 84, size=(total_slots, 4))

    print(f"Generated {total_slots} slots → {total_slots * slot_duration:.1f}s")

    # Build output MIDI
    pm_generated = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    pm_generated.instruments.append(pm_original.instruments[0])  # copy melody

    piano_track = pretty_midi.Instrument(program=0, is_drum=False, name='Generated Piano')
    for slot_idx, chord in enumerate(all_generated):
        start_sec = slot_idx * slot_duration
        end_sec   = start_sec + slot_duration
        seen = set()
        for pitch in chord:
            pitch = int(pitch)
            if pitch <= 0 or pitch >= 128 or pitch in seen:
                continue
            seen.add(pitch)
            piano_track.notes.append(
                pretty_midi.Note(velocity=80, pitch=pitch, start=start_sec, end=end_sec))

    pm_generated.instruments.append(piano_track)

    # Save generated MIDI
    generated_midi_path = os.path.join(output_dir, f'harmony_{song_id.zfill(3)}.mid')
    pm_generated.write(generated_midi_path)
    print(f"Saved generated MIDI: {generated_midi_path}")

    # Save original piano for comparison
    pm_original_copy = pretty_midi.PrettyMIDI(midi_path)
    original_midi_path = os.path.join(output_dir, f'harmony_{song_id.zfill(3)}_original.mid')
    pm_original_copy.write(original_midi_path)
    print(f"Saved original MIDI: {original_midi_path}")

    # Convert to WAV using pretty_midi (no subprocess, no audio playback)
    soundfont_path = '/Users/kilehsu/153Assignment2/modeling/checkpoints/MuseScore_General.sf3'
    if os.path.exists(soundfont_path):
        try:
            import soundfile as sf_audio
            for mid_path, wav_label in [
                (generated_midi_path, f'harmony_{song_id.zfill(3)}.wav'),
                (original_midi_path,  f'harmony_{song_id.zfill(3)}_original.wav'),
            ]:
                wav_path = os.path.join(output_dir, wav_label)
                pm_tmp = pretty_midi.PrettyMIDI(mid_path)
                audio = pm_tmp.fluidsynth(fs=44100, sf2_path=soundfont_path)
                sf_audio.write(wav_path, audio, 44100)
                print(f"Saved WAV: {wav_path}")
        except Exception as e:
            print(f"WAV conversion skipped: {e}")
    else:
        print(f"Soundfont not found at {soundfont_path} — skipping WAV")

    return True


def main():
    parser = argparse.ArgumentParser(description='Generate harmony for POP909 song')
    parser.add_argument('--song', type=str, default='001', help='Song ID (e.g. 001, 042)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--dummy', action='store_true', help='Generate dummy output if no checkpoint')
    args = parser.parse_args()

    # Defaults
    if args.checkpoint is None:
        args.checkpoint = os.path.join(os.path.dirname(__file__), 'harmonizer_best.pt')
    if args.output is None:
        args.output = '/Users/kilehsu/153Assignment2/evaluation_task2'

    data_dir = '/Users/kilehsu/153Assignment2/data/POP909/POP909'

    success = generate_for_song(
        args.song, data_dir, args.checkpoint, args.output, use_dummy=args.dummy
    )

    if success:
        print("Done!")
    else:
        print("Generation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
