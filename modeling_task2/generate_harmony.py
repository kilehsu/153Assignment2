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

    # Convert to tensor
    if len(melody_seq) >= 64:
        melody_tensor = torch.LongTensor(melody_seq[:64]).unsqueeze(0)  # (1, 64)
    else:
        # Pad
        pad_len = 64 - len(melody_seq)
        padded = np.pad(melody_seq, (0, pad_len), constant_values=128)
        melody_tensor = torch.LongTensor(padded).unsqueeze(0)

    # Generate
    if model is not None:
        with torch.no_grad():
            generated_piano = model.generate(melody_tensor, max_len=64, greedy=True)
        generated_piano = generated_piano[0].numpy()  # (64, 4)
    else:
        # Dummy output: random chords
        generated_piano = np.random.randint(36, 84, size=(64, 4))

    print(f"Generated piano shape: {generated_piano.shape}")

    # Reconstruct MIDI
    # Load original to get timing info
    pm_original = pretty_midi.PrettyMIDI(midi_path)
    tempo = pm_original.estimate_tempo()
    if tempo < 40:
        tempo = 120.0

    # Create new MIDI with melody + generated piano
    pm_generated = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    # Add melody track (copy from original)
    melody_track = pm_original.instruments[0]
    pm_generated.instruments.append(melody_track)

    # Convert generated piano sequence back to MIDI notes
    def dequantize_to_sec(idx_16th, tempo):
        """Convert 16th-note index back to seconds."""
        beat_position = idx_16th / 4.0
        return beat_position * (60.0 / tempo)

    piano_track = pretty_midi.Instrument(program=0, is_drum=False)

    for idx_16th, chord in enumerate(generated_piano):
        # Skip if all zeros (rest)
        if np.all(chord == 0):
            continue

        start_sec = dequantize_to_sec(idx_16th, tempo)
        end_sec = dequantize_to_sec(idx_16th + 1, tempo)

        for pitch in chord:
            if pitch > 0 and pitch < 129:
                note = pretty_midi.Note(velocity=80, pitch=int(pitch), start=start_sec, end=end_sec)
                piano_track.notes.append(note)

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

    # Try to convert to WAV
    soundfont_path = '/Users/kilehsu/153Assignment2/modeling/checkpoints/MuseScore_General.sf3'
    if os.path.exists(soundfont_path):
        try:
            import subprocess
            # Use fluidsynth to convert
            wav_path = os.path.join(output_dir, f'harmony_{song_id.zfill(3)}.wav')
            subprocess.run([
                'fluidsynth', '-ni', soundfont_path,
                generated_midi_path, '-F', wav_path
            ], check=True, capture_output=True)
            print(f"Saved generated WAV: {wav_path}")

            wav_original_path = os.path.join(output_dir, f'harmony_{song_id.zfill(3)}_original.wav')
            subprocess.run([
                'fluidsynth', '-ni', soundfont_path,
                original_midi_path, '-F', wav_original_path
            ], check=True, capture_output=True)
            print(f"Saved original WAV: {wav_original_path}")
        except Exception as e:
            print(f"WAV conversion skipped: {e}")
    else:
        print(f"Soundfont not found: {soundfont_path}")

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
