"""Generate 3 MIDI samples from the v2b model (interleaved 4-voice tokenization)."""
import sys, os, subprocess
sys.path.insert(0, '../modeling')

import torch
from pathlib import Path
from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer
import music21.stream as m21stream
import music21.note as m21note
import music21.instrument as m21instr

CHECKPOINT_DIR = Path('../modeling/checkpoints')
OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SOUNDFONT_PATHS = [
    '/opt/homebrew/share/soundfonts/default.sf2',
    os.path.expanduser('~/Library/Audio/Sounds/Banks/FluidR3_GM.sf2'),
]

VOICE_INSTRUMENTS = [m21instr.Piano, m21instr.Piano, m21instr.Piano, m21instr.Piano]
VOICE_NAMES = ['Soprano', 'Alto', 'Tenor', 'Bass']


def find_soundfont():
    for p in SOUNDFONT_PATHS:
        if os.path.exists(p): return p
    try:
        r = subprocess.run(['find', '/opt/homebrew', '/usr', '-name', '*.sf2', '-maxdepth', '8'],
                           capture_output=True, text=True, timeout=10)
        for line in r.stdout.strip().splitlines():
            if os.path.exists(line): return line
    except Exception: pass
    return None


def decode_to_4voice_score(token_ids, tokenizer):
    """Parse interleaved V{v}P{midi}D{dur} tokens into 4 per-voice note lists."""
    strings = tokenizer.decode(token_ids)
    relevant = [t for t in strings if t not in ('<START>', '<END>', '<BAR>')]

    # Each group of 4 = one 16th-note timestep
    voice_notes = [[] for _ in range(4)]  # list of (onset_16th, midi, dur_16ths)
    for i in range(0, len(relevant) - 3, 4):
        group = relevant[i:i+4]
        t = i // 4
        for tok in group:
            if tok == '<SUSTAIN>' or 'REST' in tok:
                continue
            if tok.startswith('V') and 'P' in tok and 'D' in tok:
                try:
                    after_v = tok[1:]
                    v = int(after_v.split('P')[0])
                    rest = after_v.split('P')[1]
                    midi_str, dur_str = rest.split('D')
                    voice_notes[v].append((t, int(midi_str), int(dur_str)))
                except (ValueError, IndexError):
                    pass
    return voice_notes


def voice_notes_to_part(notes, instr_cls, voice_name):
    part = m21stream.Part()
    part.insert(0, instr_cls())
    if not notes:
        return part
    # Sort by onset
    notes = sorted(notes, key=lambda x: x[0])
    current_pos = 0.0
    for onset, midi, dur in notes:
        onset_q = onset / 4.0
        dur_q = max(0.25, dur / 4.0)
        if onset_q > current_pos:
            rest = m21note.Rest()
            rest.quarterLength = onset_q - current_pos
            part.append(rest)
        n = m21note.Note()
        n.pitch.midi = midi
        n.quarterLength = dur_q
        part.append(n)
        current_pos = onset_q + dur_q
    midi_vals = [m for _, m, _ in notes]
    print(f'    {voice_name}: {len(notes)} notes, MIDI range {min(midi_vals)}–{max(midi_vals)}')
    return part


def tokens_to_midi(token_ids, tokenizer, output_path):
    voice_notes = decode_to_4voice_score(token_ids, tokenizer)
    score = m21stream.Score()
    for v, (notes, instr_cls, name) in enumerate(zip(voice_notes, VOICE_INSTRUMENTS, VOICE_NAMES)):
        part = voice_notes_to_part(notes, instr_cls, name)
        score.append(part)
    score.write('midi', fp=str(output_path))
    total = sum(len(n) for n in voice_notes)
    print(f'    MIDI saved: {output_path} ({total} total notes)')
    return total


def midi_to_wav(midi_path, wav_path, soundfont):
    try:
        import pretty_midi, soundfile as sf
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.fluidsynth(fs=44100, sf2_path=soundfont)
        sf.write(str(wav_path), audio, 44100)
        print(f'    WAV: {wav_path}')
        return True
    except Exception:
        pass
    try:
        fs_bin = subprocess.run(['which', 'fluidsynth'], capture_output=True, text=True).stdout.strip()
        if fs_bin:
            subprocess.run([fs_bin, '-ni', soundfont, str(midi_path), '-F', str(wav_path), '-r', '44100'],
                           capture_output=True, timeout=30)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                print(f'    WAV: {wav_path}')
                return True
    except Exception: pass
    print('    WAV skipped (no soundfont/fluidsynth)')
    return False


def main():
    tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
    print(f'Tokenizer loaded. Vocab size: {tokenizer.vocab_size}')

    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size,
                               d_model=128, n_heads=8, n_layers=4,
                               context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'v2b_transformer_best.pt',
                                     map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f'Model loaded on {device}')

    soundfont = find_soundfont()
    start_id = tokenizer.token_to_id['<START>']

    for i in range(1, 4):
        print(f'\nGenerating sample {i}...')
        token_ids = model.generate([start_id], max_new_tokens=400, temperature=1.0, device=device)
        mid_path = OUTPUT_DIR / f'generated_v2b_{i}.mid'
        tokens_to_midi(token_ids, tokenizer, mid_path)
        if soundfont:
            wav_path = OUTPUT_DIR / f'generated_v2b_{i}.wav'
            midi_to_wav(mid_path, wav_path, soundfont)

    print('\nDone.')

if __name__ == '__main__':
    main()
