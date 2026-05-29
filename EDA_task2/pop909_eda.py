#!/usr/bin/env python3
"""
POP909 Exploratory Data Analysis
Analyzes melody pitch distribution, chord vocabulary, song length, and note density
across all 909 songs in the POP909 dataset.
"""

import os
import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt

# Add venv packages
sys.path.insert(0, '/Users/kilehsu/153Assignment2/.venv/lib/python3.13/site-packages')
import pretty_midi

# Dataset path
DATASET_ROOT = Path('/Users/kilehsu/153Assignment2/data/POP909/POP909')
OUTPUT_DIR = Path('/Users/kilehsu/153Assignment2/EDA_task2')
IMAGES_DIR = Path('/Users/kilehsu/153Assignment2/images')

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def parse_chord_midi(chord_file):
    """
    Parse chord_midi.txt file
    Format: start_time(sec)  end_time(sec)  chord_name
    Returns: list of (start, end, chord) tuples and chord duration stats
    """
    chords = []
    durations = []

    try:
        with open(chord_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) >= 3:
                    start, end, chord = float(parts[0]), float(parts[1]), parts[2]
                    chords.append((start, end, chord))
                    durations.append(end - start)
    except Exception as e:
        print(f"  Error reading {chord_file}: {e}")

    return chords, durations


def get_chord_type(chord_name):
    """
    Extract base chord type (root + quality) from chord name
    Examples: 'B:maj' -> 'B:maj', 'F#:maj7/5' -> 'F#:maj7', 'N' -> 'N'
    """
    if chord_name == 'N':
        return 'N'

    # Remove any slash notation and complex extensions
    base = chord_name.split('/')[0]  # Remove slash inversions
    return base


def analyze_song(song_id, song_dir):
    """
    Analyze a single song and return statistics
    """
    stats = {
        'song_id': song_id,
        'midi_path': str(song_dir / f'{song_id}.mid'),
        'chord_path': str(song_dir / 'chord_midi.txt'),
    }

    # Analyze MIDI
    try:
        mid = pretty_midi.PrettyMIDI(stats['midi_path'])
        stats['duration_sec'] = mid.get_end_time()

        # Find MELODY track (typically first track with name 'MELODY')
        melody_inst = None
        for inst in mid.instruments:
            if inst.name == 'MELODY' and not inst.is_drum:
                melody_inst = inst
                break

        # If no MELODY track, use first non-drum track
        if melody_inst is None:
            for inst in mid.instruments:
                if not inst.is_drum:
                    melody_inst = inst
                    break

        if melody_inst is None:
            stats['error'] = 'No non-drum instrument found'
            return stats

        # Extract melody pitch stats
        notes = melody_inst.notes
        if not notes:
            stats['error'] = 'MELODY track has no notes'
            return stats

        pitches = [n.pitch for n in notes]
        stats['melody_pitch_min'] = int(min(pitches))
        stats['melody_pitch_max'] = int(max(pitches))
        stats['melody_pitch_mean'] = float(np.mean(pitches))
        stats['melody_pitch_std'] = float(np.std(pitches))
        stats['melody_note_count'] = len(notes)

        # Note density: notes per second
        stats['melody_note_density'] = len(notes) / max(stats['duration_sec'], 0.1)

        # Store pitches for histogram
        stats['melody_pitches'] = pitches

    except Exception as e:
        stats['error'] = f'MIDI read error: {e}'
        return stats

    # Analyze chords
    try:
        chords, durations = parse_chord_midi(stats['chord_path'])

        if not chords:
            stats['error'] = 'No chords found'
            return stats

        # Extract chord types
        chord_types = [get_chord_type(c[2]) for c in chords]
        stats['chord_types'] = chord_types
        stats['chord_count'] = len(chords)
        stats['unique_chord_types'] = len(set(chord_types))

        if durations:
            stats['chord_duration_mean'] = float(np.mean(durations))
            stats['chord_duration_std'] = float(np.std(durations))

        # Compute song length in beats (approximate using chord timing)
        if chords:
            stats['song_length_sec'] = chords[-1][1]  # Last chord end time

    except Exception as e:
        stats['chord_error'] = f'Chord read error: {e}'
        # Don't return early; we still want melody stats

    return stats


def main():
    """Main EDA pipeline"""
    print("Starting POP909 EDA...")
    print(f"Dataset root: {DATASET_ROOT}")

    # Collect all song results
    all_results = {
        'dataset_info': {
            'name': 'POP909',
            'root': str(DATASET_ROOT),
            'total_songs_attempted': 0,
            'successful_analyses': 0,
        },
        'melody_stats': {},
        'chord_stats': {},
        'song_stats': {},
        'all_pitches': [],
        'all_chord_types': Counter(),
        'song_lengths': [],
        'note_densities': [],
    }

    # Get all song directories
    song_dirs = sorted([d for d in DATASET_ROOT.iterdir() if d.is_dir() and d.name.isdigit()])

    print(f"Found {len(song_dirs)} song directories")

    for i, song_dir in enumerate(song_dirs):
        song_id = song_dir.name
        all_results['dataset_info']['total_songs_attempted'] += 1

        if i % 100 == 0:
            print(f"Processing song {i+1}/{len(song_dirs)}...")

        stats = analyze_song(song_id, song_dir)
        all_results['song_stats'][song_id] = stats

        # Aggregate successful analyses
        if 'error' not in stats and 'melody_pitches' in stats:
            all_results['dataset_info']['successful_analyses'] += 1

            # Aggregate pitches
            all_results['all_pitches'].extend(stats['melody_pitches'])

            # Aggregate song length and density
            if 'song_length_sec' in stats:
                all_results['song_lengths'].append(stats['song_length_sec'])
            if 'melody_note_density' in stats:
                all_results['note_densities'].append(stats['melody_note_density'])

            # Aggregate chord types
            if 'chord_types' in stats:
                for ct in stats['chord_types']:
                    all_results['all_chord_types'][ct] += 1

    # Compute aggregated statistics
    if all_results['all_pitches']:
        pitches = all_results['all_pitches']
        all_results['melody_stats'] = {
            'pitch_min': int(min(pitches)),
            'pitch_max': int(max(pitches)),
            'pitch_mean': float(np.mean(pitches)),
            'pitch_std': float(np.std(pitches)),
            'total_notes': len(pitches),
        }

    if all_results['song_lengths']:
        all_results['song_stats'] = {
            'length_min': float(min(all_results['song_lengths'])),
            'length_max': float(max(all_results['song_lengths'])),
            'length_mean': float(np.mean(all_results['song_lengths'])),
            'length_std': float(np.std(all_results['song_lengths'])),
        }

    if all_results['note_densities']:
        all_results['song_stats']['note_density_min'] = float(min(all_results['note_densities']))
        all_results['song_stats']['note_density_max'] = float(max(all_results['note_densities']))
        all_results['song_stats']['note_density_mean'] = float(np.mean(all_results['note_densities']))

    # Chord statistics
    total_chord_instances = sum(all_results['all_chord_types'].values())
    top_20_chords = dict(all_results['all_chord_types'].most_common(20))

    all_results['chord_stats'] = {
        'unique_chord_types': len(all_results['all_chord_types']),
        'total_chord_instances': total_chord_instances,
        'top_20_chords': top_20_chords,
    }

    # Save results to JSON
    results_file = OUTPUT_DIR / 'pop909_eda_results.json'

    # Convert Counter to dict for JSON serialization
    all_results['all_chord_types'] = dict(all_results['all_chord_types'])

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Generate figures
    print("Generating figures...")

    # Figure 1: Melody pitch distribution
    if all_results['all_pitches']:
        plt.figure(figsize=(12, 6))
        plt.hist(all_results['all_pitches'], bins=range(20, 90), edgecolor='black', alpha=0.7)
        plt.xlabel('MIDI Pitch Number')
        plt.ylabel('Frequency')
        plt.title('POP909: Melody Pitch Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = IMAGES_DIR / 'task2_melody_pitch_distribution.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")

    # Figure 2: Top 20 chord vocabulary
    if top_20_chords:
        plt.figure(figsize=(14, 6))
        chords = list(top_20_chords.keys())
        counts = list(top_20_chords.values())

        # Sort by count descending
        sorted_pairs = sorted(zip(chords, counts), key=lambda x: x[1], reverse=True)
        chords, counts = zip(*sorted_pairs)

        plt.bar(range(len(chords)), counts, edgecolor='black', alpha=0.7)
        plt.xticks(range(len(chords)), chords, rotation=45, ha='right')
        plt.ylabel('Frequency')
        plt.title(f'POP909: Top 20 Most Common Chord Types')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig_path = IMAGES_DIR / 'task2_chord_vocabulary.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")

    # Figure 3: Song length distribution
    if all_results['song_lengths']:
        plt.figure(figsize=(12, 6))
        plt.hist(all_results['song_lengths'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Song Length (seconds)')
        plt.ylabel('Frequency')
        plt.title('POP909: Song Length Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = IMAGES_DIR / 'task2_song_length.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")

    # Figure 4: Note density distribution
    if all_results['note_densities']:
        plt.figure(figsize=(12, 6))
        plt.hist(all_results['note_densities'], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Note Density (notes per second)')
        plt.ylabel('Frequency')
        plt.title('POP909: Melody Note Density Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = IMAGES_DIR / 'task2_note_density.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {fig_path}")

    # Print summary
    print("\n" + "="*60)
    print("POP909 EDA SUMMARY")
    print("="*60)
    print(f"Total songs analyzed: {all_results['dataset_info']['successful_analyses']}/{all_results['dataset_info']['total_songs_attempted']}")
    print()
    print("MELODY STATISTICS:")
    for key, val in all_results['melody_stats'].items():
        print(f"  {key}: {val}")
    print()
    print("SONG STATISTICS:")
    for key, val in all_results['song_stats'].items():
        if isinstance(val, float):
            print(f"  {key}: {val:.2f}")
        else:
            print(f"  {key}: {val}")
    print()
    print("CHORD STATISTICS:")
    for key, val in all_results['chord_stats'].items():
        if key == 'top_20_chords':
            print(f"  {key}:")
            for chord, count in sorted(val.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {chord}: {count}")
        else:
            print(f"  {key}: {val}")
    print("="*60)


if __name__ == '__main__':
    main()
