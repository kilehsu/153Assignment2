"""
POP909 Dataset: Load melody and piano tracks, quantize to 16th notes, create fixed-length windows.
"""

import os
import torch
import numpy as np
import pretty_midi
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def quantize_to_16th(onset_time, tempo, resolution=480):
    """
    Convert onset time (seconds) to 16th-note index.

    Args:
        onset_time: time in seconds
        tempo: tempo in BPM
        resolution: MIDI resolution (PPQ), default 480

    Returns:
        16th-note index (one 16th note = one quarter note / 4)
    """
    # seconds to beats: time_sec * (tempo / 60)
    beat_position = onset_time * (tempo / 60.0)
    # beats to 16th notes: beat_position * 4
    sixteenth_index = int(round(beat_position * 4))
    return sixteenth_index


def extract_melody_and_piano(midi_path, tempo=120.0, window_len=64):
    """
    Extract melody (track 0) and piano (track 2) from a POP909 MIDI file.
    Quantize to 16th notes and return as sequences.

    Args:
        midi_path: path to MIDI file
        tempo: detected or default tempo in BPM
        window_len: length of sequences in 16th notes

    Returns:
        (melody_seq, piano_seq) as numpy arrays
        melody_seq: shape (n_windows, window_len), values 0-128 (128=REST)
        piano_seq: shape (n_windows, window_len, 4), values 0-128 per voice

        Returns (None, None) if extraction fails
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)

        # Estimate tempo if needed
        if tempo == 120.0:
            est_tempo = pm.estimate_tempo()
            if est_tempo > 40:  # sanity check
                tempo = est_tempo

        # Get duration in 16th notes
        duration_sec = pm.get_end_time()
        duration_16th = int(duration_sec * (tempo / 60.0) * 4) + 1

        # Initialize sequences (REST = 128)
        melody_seq = np.full(duration_16th, 128, dtype=np.int32)
        piano_seq = np.full((duration_16th, 4), 0, dtype=np.int32)

        # Extract melody (track 0)
        if len(pm.instruments) > 0:
            melody_track = pm.instruments[0]
            for note in melody_track.notes:
                start_idx = quantize_to_16th(note.start, tempo)
                end_idx = quantize_to_16th(note.end, tempo)
                # Clamp to valid range
                start_idx = min(max(0, start_idx), duration_16th - 1)
                end_idx = min(max(start_idx + 1, end_idx), duration_16th)
                # Fill with this pitch
                for idx in range(start_idx, end_idx):
                    melody_seq[idx] = note.pitch

        # Extract piano (track 2) - 4 simultaneous pitches
        if len(pm.instruments) > 2:
            piano_track = pm.instruments[2]
            # Build a dictionary of active notes per 16th-note slot
            note_times = {}  # slot -> list of pitches

            for note in piano_track.notes:
                start_idx = quantize_to_16th(note.start, tempo)
                end_idx = quantize_to_16th(note.end, tempo)
                start_idx = min(max(0, start_idx), duration_16th - 1)
                end_idx = min(max(start_idx + 1, end_idx), duration_16th)

                for idx in range(start_idx, end_idx):
                    if idx not in note_times:
                        note_times[idx] = []
                    note_times[idx].append(note.pitch)

            # Fill piano_seq with up to 4 pitches per slot, sorted descending
            for idx in range(duration_16th):
                if idx in note_times:
                    pitches = sorted(set(note_times[idx]), reverse=True)[:4]
                    for voice, pitch in enumerate(pitches):
                        piano_seq[idx, voice] = pitch

        return melody_seq, piano_seq
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return None, None


def create_windows(melody_seq, piano_seq, window_len=64, stride=32, rest_threshold=0.8):
    """
    Segment sequences into fixed-length windows with overlap.
    Discard windows where melody is >80% REST.

    Returns:
        list of (melody_window, piano_window) tuples
    """
    windows = []

    # Ensure sequences are long enough
    min_len = min(len(melody_seq), len(piano_seq))
    melody_seq = melody_seq[:min_len]
    piano_seq = piano_seq[:min_len]

    for start in range(0, len(melody_seq) - window_len + 1, stride):
        end = start + window_len

        mel_window = melody_seq[start:end]
        piano_window = piano_seq[start:end]

        # Check REST threshold
        rest_count = np.sum(mel_window == 128)
        if rest_count / len(mel_window) > rest_threshold:
            continue

        windows.append((mel_window.copy(), piano_window.copy()))

    return windows


class POP909Dataset(Dataset):
    """
    PyTorch Dataset for POP909 melody-to-piano task.
    """

    def __init__(self, data_dir, split='train', val_ratio=0.1, seed=42):
        """
        Args:
            data_dir: path to POP909-Dataset folder
            split: 'train' or 'val'
            val_ratio: fraction of songs reserved for validation
            seed: random seed for reproducible split
        """
        self.data_dir = data_dir
        self.split = split
        self.seed = seed

        # Discover all song folders
        song_folders = sorted([d for d in os.listdir(data_dir)
                              if os.path.isdir(os.path.join(data_dir, d))])

        # Split by song index
        np.random.seed(seed)
        n_songs = len(song_folders)
        n_val = max(1, int(n_songs * val_ratio))
        val_indices = set(np.random.choice(n_songs, n_val, replace=False))

        if split == 'train':
            self.song_folders = [song_folders[i] for i in range(n_songs) if i not in val_indices]
        else:  # val
            self.song_folders = [song_folders[i] for i in range(n_songs) if i in val_indices]

        # Load or create cache
        cache_file = os.path.join(os.path.dirname(__file__), 'pop909_cache.pkl')
        if os.path.exists(cache_file):
            print(f"Loading cache from {cache_file}")
            with open(cache_file, 'rb') as f:
                self.windows = pickle.load(f)[split]
        else:
            print(f"Building dataset cache...")
            self.windows = self._build_cache(cache_file)

    def _build_cache(self, cache_file):
        """Build windows from all songs and save cache."""
        all_windows = {'train': [], 'val': []}
        failed_songs = []

        np.random.seed(self.seed)
        n_songs = len(os.listdir(self.data_dir))
        val_indices = set(np.random.choice(n_songs, max(1, int(n_songs * 0.1)), replace=False))

        song_folders = sorted([d for d in os.listdir(self.data_dir)
                              if os.path.isdir(os.path.join(self.data_dir, d))])

        for song_idx, song_folder in enumerate(song_folders):
            song_path = os.path.join(self.data_dir, song_folder)
            midi_file = os.path.join(song_path, f"{song_folder}.mid")

            if not os.path.exists(midi_file):
                failed_songs.append(song_folder)
                continue

            # Extract sequences
            melody_seq, piano_seq = extract_melody_and_piano(midi_file)
            if melody_seq is None:
                failed_songs.append(song_folder)
                continue

            # Create windows
            windows = create_windows(melody_seq, piano_seq)

            # Assign to train/val
            split_key = 'val' if song_idx in val_indices else 'train'
            all_windows[split_key].extend(windows)

        print(f"Failed to parse: {len(failed_songs)} songs")
        if failed_songs:
            print(f"  Examples: {failed_songs[:5]}")

        print(f"Train windows: {len(all_windows['train'])}")
        print(f"Val windows: {len(all_windows['val'])}")

        # Save cache
        with open(cache_file, 'wb') as f:
            pickle.dump(all_windows, f)
        print(f"Saved cache to {cache_file}")

        return all_windows[self.split]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        melody_seq, piano_seq = self.windows[idx]

        # Convert to torch tensors
        melody = torch.LongTensor(melody_seq)  # shape (64,)
        piano = torch.LongTensor(piano_seq)     # shape (64, 4)

        return melody, piano


if __name__ == '__main__':
    # Test the dataset
    data_dir = '/Users/kilehsu/153Assignment2/data/POP909/POP909'

    print("Building POP909 dataset...")
    train_ds = POP909Dataset(data_dir, split='train', val_ratio=0.1)
    val_ds = POP909Dataset(data_dir, split='val', val_ratio=0.1)

    print(f"\nTrain dataset: {len(train_ds)} windows")
    print(f"Val dataset: {len(val_ds)} windows")

    # Sample a batch
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    melody, piano = next(iter(loader))

    print(f"\nSample batch:")
    print(f"  Melody shape: {melody.shape}")
    print(f"  Piano shape: {piano.shape}")
    print(f"  Melody sample: {melody[0][:10]}")
    print(f"  Piano sample:\n{piano[0][:5]}")
