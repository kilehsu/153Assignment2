"""
08_temporal_structure.py
Create piano roll visualization of a sample chorale.
"""

import os
from music21 import corpus
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "/Users/kilehsu/153Assignment2/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("TEMPORAL / STRUCTURE ANALYSIS")
print("=" * 60)

# Load chorales
bach_works = corpus.getComposer('bach')
chorales = []

for work_path in bach_works:
    try:
        score = corpus.parse(work_path)
        if hasattr(score, 'parts') and len(score.parts) == 4:
            chorales.append((work_path, score))
    except:
        pass

print(f"Loaded {len(chorales)} chorales")

# Use the first chorale for visualization
if chorales:
    work_path, score = chorales[0]
    print(f"Visualizing: {work_path}")

    # Extract note information
    # We need: start time, duration, pitch, and voice
    notes_data = []  # List of (start_time, duration, midi_pitch, voice_idx)

    for voice_idx, part in enumerate(score.parts):
        for element in part.flatten().notes:
            try:
                start_time = element.quarterLength if hasattr(element, 'offset') else 0
                # Get actual offset
                if hasattr(element, 'offset'):
                    start_time = element.offset
                duration = element.duration.quarterLength
                midi = element.pitch.midi

                notes_data.append((start_time, duration, midi, voice_idx))
            except:
                pass

    if notes_data:
        # Create piano roll visualization
        fig, ax = plt.subplots(figsize=(16, 8))

        # Color palette for voices
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

        # Plot notes as rectangles
        max_time = max((start + duration for start, duration, _, _ in notes_data))
        min_midi = min((m for _, _, m, _ in notes_data))
        max_midi = max((m for _, _, m, _ in notes_data))

        for start_time, duration, midi, voice_idx in notes_data:
            color = colors[voice_idx]
            rect = plt.Rectangle((start_time, midi - 0.4), duration, 0.8,
                                 facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            ax.add_patch(rect)

        # Set axis properties
        ax.set_xlim(0, max_time + 1)
        ax.set_ylim(min_midi - 2, max_midi + 2)
        ax.set_xlabel('Time (Quarter Notes)', fontsize=12)
        ax.set_ylabel('MIDI Note Number', fontsize=12)
        ax.set_title('Piano Roll Visualization of a Bach Chorale', fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=voice_names[i])
                          for i in range(4)]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, '08_piano_roll_sample.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: 08_piano_roll_sample.png")
        plt.close(fig)

        # Additional: Stacked time series plot (pitch per voice over time)
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        voice_names_plot = ['Soprano', 'Alto', 'Tenor', 'Bass']

        for voice_idx in range(4):
            voice_notes = [(st, d, m) for st, d, m, v in notes_data if v == voice_idx]

            if voice_notes:
                # Sort by start time
                voice_notes.sort()
                start_times = [st for st, _, _ in voice_notes]
                pitches = [m for _, _, m in voice_notes]

                ax = axes[voice_idx]
                ax.scatter(start_times, pitches, s=20, alpha=0.6, color=colors[voice_idx])
                ax.plot(start_times, pitches, alpha=0.3, color=colors[voice_idx], linewidth=1)
                ax.set_ylabel(f'{voice_names_plot[voice_idx]}\nMIDI Pitch', fontsize=10)
                ax.set_ylim(min_midi - 2, max_midi + 2)
                ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (Quarter Notes)', fontsize=12)
        fig.suptitle('Pitch Trajectories by Voice', fontsize=14, fontweight='bold', y=0.995)
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, '08b_pitch_trajectories.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: 08b_pitch_trajectories.png")
        plt.close(fig)

        # Statistics about the sample
        print("\n" + "=" * 60)
        print("SAMPLE CHORALE STATISTICS")
        print("=" * 60)
        print(f"Total notes: {len(notes_data)}")
        print(f"Duration: {max_time:.1f} quarter notes ({max_time/4:.1f} measures)")
        print(f"MIDI range: {min_midi} to {max_midi}")

        for voice_idx in range(4):
            voice_notes = [(st, d, m) for st, d, m, v in notes_data if v == voice_idx]
            if voice_notes:
                pitches = [m for _, _, m in voice_notes]
                print(f"\n{voice_names_plot[voice_idx]}:")
                print(f"  Notes: {len(voice_notes)}")
                print(f"  Pitch range: {min(pitches)} to {max(pitches)}")
                print(f"  Mean pitch: {np.mean(pitches):.2f}")

print("\nTemporal Structure Analysis Complete!")
