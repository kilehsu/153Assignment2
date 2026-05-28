"""
Convert the 3 generated samples from eval_results.json to MIDI files.
The tokenizer encodes all 4 SATB voices sequentially (soprano → alto → tenor → bass),
so we split the generated token stream into 4 equal parts and assign each to its own
music21 Part so they play simultaneously as proper 4-voice polyphony.

Saves: evaluation/generated_1.mid, generated_2.mid, generated_3.mid
       evaluation/generated_1.wav, generated_2.wav, generated_3.wav
"""

import sys
import os
import json
import subprocess

sys.path.insert(0, '../modeling')
from tokenizer import BachTokenizer

import music21.stream as m21stream
import music21.note as m21note
import music21.instrument as m21instr

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

EVAL_RESULTS   = os.path.join(PROJECT_ROOT, 'modeling', 'checkpoints', 'eval_results.json')
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'modeling', 'checkpoints', 'tokenizer.json')
OUTPUT_DIR     = SCRIPT_DIR

# General MIDI soundfont — standard path from `brew install fluidsynth`
SOUNDFONT_PATHS = [
    '/usr/share/sounds/sf2/FluidR3_GM.sf2',
    '/opt/homebrew/share/soundfonts/default.sf2',
    os.path.expanduser('~/Library/Audio/Sounds/Banks/FluidR3_GM.sf2'),
]


def find_soundfont():
    for p in SOUNDFONT_PATHS:
        if os.path.exists(p):
            return p
    # Try to find any .sf2 on the system
    try:
        result = subprocess.run(['find', '/opt/homebrew', '/usr', '-name', '*.sf2', '-maxdepth', '8'],
                                capture_output=True, text=True, timeout=10)
        for line in result.stdout.strip().splitlines():
            if os.path.exists(line):
                return line
    except Exception:
        pass
    return None


def token_to_note(token_str):
    """Parse P{midi}_D{dur}. Returns (midi, quarterLength) or None."""
    if not token_str.startswith('P'):
        return None
    try:
        mid_part, dur_part = token_str.split('_')
        midi      = int(mid_part[1:])
        dur_16ths = int(dur_part[1:])
        return midi, dur_16ths / 4.0
    except (ValueError, IndexError):
        return None


# SATB MIDI program numbers (General MIDI)
VOICE_INSTRUMENTS = [
    (m21instr.Soprano,  'Soprano'),
    (m21instr.Alto,     'Alto'),
    (m21instr.Tenor,    'Tenor'),
    (m21instr.Bass,     'Bass'),
]


def tokens_to_midi_4voice(token_strings, output_path):
    """
    Split token stream into 4 equal sections (one per SATB voice) and
    stack as simultaneous parts so they play as chords/polyphony.
    """
    # Filter to just note tokens
    note_tokens = [t for t in token_strings if t.startswith('P')]

    n = len(note_tokens)
    # Split into 4 equal-ish chunks
    chunk_size = n // 4
    voice_tokens = [
        note_tokens[0 * chunk_size : 1 * chunk_size],
        note_tokens[1 * chunk_size : 2 * chunk_size],
        note_tokens[2 * chunk_size : 3 * chunk_size],
        note_tokens[3 * chunk_size :],               # remainder goes to Bass
    ]

    score = m21stream.Score()

    for voice_idx, (tokens, (instr_cls, voice_name)) in enumerate(zip(voice_tokens, VOICE_INSTRUMENTS)):
        part = m21stream.Part()
        part.insert(0, instr_cls())

        for tok in tokens:
            result = token_to_note(tok)
            if result is not None:
                midi_pitch, quarter_length = result
                n_obj = m21note.Note()
                n_obj.pitch.midi = midi_pitch
                n_obj.quarterLength = quarter_length
                part.append(n_obj)

        score.append(part)
        print(f'    {voice_name}: {len(tokens)} notes, '
              f'range MIDI {min(int(t.split("_")[0][1:]) for t in tokens if t.startswith("P") and "_" in t)}'
              f'–{max(int(t.split("_")[0][1:]) for t in tokens if t.startswith("P") and "_" in t)}')

    score.write('midi', fp=output_path)
    return len(note_tokens)


def midi_to_wav(midi_path, wav_path, soundfont):
    """Convert MIDI to WAV using FluidSynth via pretty_midi."""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        audio = pm.fluidsynth(fs=44100, sf2_path=soundfont)
        import soundfile as sf
        sf.write(wav_path, audio, 44100)
        print(f'    WAV written via pretty_midi: {wav_path}')
        return True
    except Exception as e:
        pass

    # Fallback: call fluidsynth CLI directly
    try:
        fluidsynth_bin = subprocess.run(['which', 'fluidsynth'], capture_output=True, text=True).stdout.strip()
        if fluidsynth_bin:
            result = subprocess.run(
                [fluidsynth_bin, '-ni', soundfont, midi_path, '-F', wav_path, '-r', '44100'],
                capture_output=True, text=True, timeout=30
            )
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                print(f'    WAV written via fluidsynth CLI: {wav_path}')
                return True
    except Exception as e:
        pass

    print(f'    WAV conversion failed (install soundfile: pip install soundfile)')
    return False


def main():
    with open(EVAL_RESULTS) as f:
        eval_data = json.load(f)

    generated_samples = eval_data['generated_samples']
    print(f'Loaded {len(generated_samples)} generated samples')

    tokenizer = BachTokenizer.load(TOKENIZER_PATH)
    print(f'Tokenizer vocab size: {tokenizer.vocab_size}')

    soundfont = find_soundfont()
    if soundfont:
        print(f'Soundfont found: {soundfont}')
    else:
        print('No soundfont found — MIDI only. Install with: brew install fluid-soundfont-gm')

    for i, token_ids in enumerate(generated_samples, start=1):
        print(f'\nSample {i}:')
        token_strings = tokenizer.decode(token_ids)
        midi_path = os.path.join(OUTPUT_DIR, f'generated_{i}.mid')
        note_count = tokens_to_midi_4voice(token_strings, midi_path)
        print(f'  MIDI: {midi_path}  ({note_count} total notes across 4 voices)')

        if soundfont:
            wav_path = os.path.join(OUTPUT_DIR, f'generated_{i}.wav')
            midi_to_wav(midi_path, wav_path, soundfont)

    print('\nDone.')


if __name__ == '__main__':
    main()
