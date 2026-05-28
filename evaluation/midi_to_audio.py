"""
Best-effort MIDI → WAV conversion for the 3 generated MIDI files.
Tries midi2audio (FluidSynth wrapper) first, then pretty_midi + fluidsynth,
then exits gracefully with a message if neither is available.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MIDI_FILES = [
    os.path.join(SCRIPT_DIR, f'generated_{i}.mid') for i in range(1, 4)
]
WAV_FILES = [
    os.path.join(SCRIPT_DIR, f'generated_{i}.wav') for i in range(1, 4)
]


def try_midi2audio(midi_path, wav_path):
    """Try conversion via the midi2audio package (wraps FluidSynth CLI)."""
    try:
        from midi2audio import FluidSynth
        FluidSynth().midi_to_audio(midi_path, wav_path)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  midi2audio failed for {os.path.basename(midi_path)}: {e}")
        return False


def try_pretty_midi(midi_path, wav_path):
    """Try conversion via pretty_midi.fluidsynth + soundfile."""
    try:
        import pretty_midi
        import soundfile as sf
        import numpy as np

        pm = pretty_midi.PrettyMIDI(midi_path)
        audio = pm.fluidsynth(fs=44100)
        # pretty_midi returns a float64 array; ensure it's in [-1, 1]
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        sf.write(wav_path, audio, 44100)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"  pretty_midi failed for {os.path.basename(midi_path)}: {e}")
        return False


def main():
    print("Attempting MIDI → WAV conversion...\n")

    any_success = False

    for midi_path, wav_path in zip(MIDI_FILES, WAV_FILES):
        name = os.path.basename(midi_path)
        if not os.path.exists(midi_path):
            print(f"  SKIP: {name} not found (run convert_to_midi.py first)")
            continue

        print(f"Converting {name}...")

        if try_midi2audio(midi_path, wav_path):
            print(f"  [midi2audio] Written: {wav_path}")
            any_success = True
            continue

        if try_pretty_midi(midi_path, wav_path):
            print(f"  [pretty_midi] Written: {wav_path}")
            any_success = True
            continue

        print(f"  Could not convert {name} — no working backend found.")

    if not any_success:
        print("\nAudio conversion not available — install FluidSynth for audio output.")
        print("  On macOS:  brew install fluidsynth && pip install midi2audio")
        print("  On Ubuntu: apt-get install fluidsynth && pip install midi2audio")
    else:
        print("\nAudio conversion complete.")


if __name__ == '__main__':
    main()
