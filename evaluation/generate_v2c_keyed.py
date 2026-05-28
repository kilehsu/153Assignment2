"""
Generate MIDI samples from v2c with key-detection + chromatic soft masking.

After a warmup period, detects the key the model is settling into, then
softly penalizes out-of-key notes for the rest of generation. This directly
fixes the black-key overrepresentation without retraining.
"""
import sys, os, subprocess, json
sys.path.insert(0, '../modeling')

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import Counter
from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer
import music21.stream as m21stream
import music21.note as m21note
import music21.instrument as m21instr

CHECKPOINT_DIR = Path('../modeling/checkpoints')
OUTPUT_DIR     = Path(os.path.dirname(os.path.abspath(__file__)))
SOUNDFONT_PATHS = [
    str(CHECKPOINT_DIR / 'MuseScore_General.sf3'),
    '/opt/homebrew/share/soundfonts/default.sf2',
    os.path.expanduser('~/Library/Audio/Sounds/Banks/FluidR3_GM.sf2'),
]
VOICE_NAMES       = ['Soprano', 'Alto', 'Tenor', 'Bass']
PIANO_INSTRUMENTS = [m21instr.Piano]*4
SATB_INSTRUMENTS  = [m21instr.Soprano, m21instr.Alto, m21instr.Tenor, m21instr.Bass]

# Major and natural-minor interval patterns (semitones from root)
MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES      = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


def build_scales():
    """Return dict: (root, mode) -> frozenset of pitch classes."""
    scales = {}
    for root in range(12):
        scales[(root, 'major')] = frozenset((root + i) % 12 for i in MAJOR_INTERVALS)
        scales[(root, 'minor')] = frozenset((root + i) % 12 for i in MINOR_INTERVALS)
    return scales

ALL_SCALES = build_scales()


def detect_key(pitch_classes):
    """Pick the major/minor key that fits the most observed pitch classes."""
    if not pitch_classes:
        return None, None, 0.0
    best_key, best_mode, best_score = 0, 'major', 0.0
    for (root, mode), scale_pcs in ALL_SCALES.items():
        score = sum(1 for pc in pitch_classes if pc in scale_pcs) / len(pitch_classes)
        if score > best_score:
            best_score, best_key, best_mode = score, root, mode
    return best_key, best_mode, best_score


def build_token_info(tokenizer):
    """Cache per-token: ('note', voice, midi_pc) | ('special',) | ('rest',) | ('unknown',)"""
    info = {}
    for tid, tok in tokenizer.id_to_token.items():
        if tok in ('<START>', '<END>', '<BAR>', '<SUSTAIN>'):
            info[tid] = ('special',)
        elif tok.endswith('REST'):
            info[tid] = ('rest',)
        elif tok.startswith('V') and 'P' in tok and 'D' in tok:
            try:
                after_v = tok[1:]
                v = int(after_v.split('P')[0])
                midi = int(after_v.split('P')[1].split('D')[0])
                info[tid] = ('note', v, midi % 12)   # store pitch class
            except (ValueError, IndexError):
                info[tid] = ('unknown',)
        else:
            info[tid] = ('unknown',)
    return info


def generate_keyed(model, start_tokens, tokenizer, token_info,
                   max_new_tokens=480, min_new_tokens=240, temperature=1.0,
                   warmup_tokens=60, chromatic_penalty=4.0,
                   sustain_penalty=2.5, max_voice_sustains=16,
                   histogram_correction_strength=3.0, bach_ref_histogram=None,
                   device=None):
    """
    Autoregressive generation with:
      - key-detection + chromatic soft masking
      - per-voice sustain penalty to keep all 4 voices active
      - min_new_tokens to prevent early termination
      - histogram correction to balance pitch class distribution vs Bach reference
    """
    if device is None:
        device = next(model.parameters()).device

    sustain_id = tokenizer.token_to_id.get('<SUSTAIN>', -1)
    end_id     = tokenizer.token_to_id.get('<END>', -1)

    tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)
    detected_key = detected_mode = None
    key_pcs = None
    # count consecutive SUSTAIN tokens per voice (reset on any non-SUSTAIN token)
    voice_sustain_counts = [0, 0, 0, 0]

    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            num_generated = tokens.shape[1] - len(start_tokens)
            current_voice = num_generated % 4  # which voice we're predicting now

            if num_generated == warmup_tokens and detected_key is None:
                pitch_classes = []
                for tid in tokens[0].tolist():
                    i = token_info.get(tid, ('unknown',))
                    if i[0] == 'note':
                        pitch_classes.append(i[2])
                detected_key, detected_mode, fit = detect_key(pitch_classes)
                key_pcs = ALL_SCALES[(detected_key, detected_mode)]
                print(f'    Key detected: {NOTE_NAMES[detected_key]} {detected_mode} '
                      f'(fit={fit:.2%}, after {warmup_tokens} tokens)')

            tokens_in = tokens[:, -model.context_len:]
            logits = model.forward(tokens_in)[:, -1, :]

            # Suppress <END> until min length reached
            if num_generated < min_new_tokens:
                logits[0, end_id] = float('-inf')

            # Chromatic penalty
            if key_pcs is not None:
                for tid, i in token_info.items():
                    if i[0] == 'note' and i[2] not in key_pcs:
                        logits[0, tid] -= chromatic_penalty

            # Histogram correction: boost underrepresented pitch classes vs Bach reference
            if histogram_correction_strength > 0.0 and bach_ref_histogram is not None:
                pitch_classes = []
                for tid in tokens[0].tolist():
                    ti = token_info.get(tid, ('unknown',))
                    if ti[0] == 'note':
                        pitch_classes.append(ti[2])
                if pitch_classes:
                    pc_counter = Counter(pitch_classes)
                    cur = np.array([pc_counter.get(pc, 0) for pc in range(12)], dtype=float)
                    cur /= cur.sum()
                else:
                    cur = np.zeros(12)
                bach_ref = np.array(bach_ref_histogram)
                for tid, ti in token_info.items():
                    if ti[0] == 'note':
                        logits[0, tid] += histogram_correction_strength * (bach_ref[ti[2]] - cur[ti[2]])

            # Voice sustain penalty: discourage this voice from sustaining too long
            if voice_sustain_counts[current_voice] >= max_voice_sustains:
                logits[0, sustain_id] -= sustain_penalty

            logits = logits / temperature
            probs  = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            # Update sustain counter for this voice
            if next_token.item() == sustain_id:
                voice_sustain_counts[current_voice] += 1
            else:
                voice_sustain_counts[current_voice] = 0

            if next_token.item() == end_id:
                print(f'    <END> at token {num_generated}')
                break

    return tokens[0].cpu().tolist()


# ── MIDI / WAV helpers (same as generate_v2c.py) ─────────────────────────────

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
    strings    = tokenizer.decode(token_ids)
    relevant   = [t for t in strings if t not in ('<START>', '<END>', '<BAR>')]
    voice_notes = [[] for _ in range(4)]
    for i in range(0, len(relevant) - 3, 4):
        group = relevant[i:i+4]
        t = i // 4
        for tok in group:
            if tok == '<SUSTAIN>' or 'REST' in tok:
                continue
            if tok.startswith('V') and 'P' in tok and 'D' in tok:
                try:
                    after_v  = tok[1:]
                    v        = int(after_v.split('P')[0])
                    midi_str, dur_str = after_v.split('P')[1].split('D')
                    voice_notes[v].append((t, int(midi_str), int(dur_str)))
                except (ValueError, IndexError):
                    pass
    return voice_notes


def voice_notes_to_part(notes, instr_cls, voice_name):
    part = m21stream.Part()
    part.insert(0, instr_cls())
    if not notes:
        return part
    notes = sorted(notes, key=lambda x: x[0])
    cur = 0.0
    for onset, midi, dur in notes:
        oq = onset / 4.0
        dq = max(0.25, dur / 4.0)
        if oq > cur:
            rest = m21note.Rest()
            rest.quarterLength = oq - cur
            part.append(rest)
        n = m21note.Note()
        n.pitch.midi    = midi
        n.quarterLength = dq
        part.append(n)
        cur = oq + dq
    midi_vals = [m for _, m, _ in notes]
    print(f'      {voice_name}: {len(notes)} notes, MIDI {min(midi_vals)}-{max(midi_vals)}')
    return part


def tokens_to_midi(token_ids, tokenizer, output_path, instruments):
    voice_notes = decode_to_4voice_score(token_ids, tokenizer)
    score = m21stream.Score()
    for notes, instr_cls, name in zip(voice_notes, instruments, VOICE_NAMES):
        score.append(voice_notes_to_part(notes, instr_cls, name))
    score.write('midi', fp=str(output_path))
    total = sum(len(n) for n in voice_notes)
    print(f'    -> {output_path.name} ({total} notes)')
    return total


def midi_to_wav(midi_path, wav_path, soundfont):
    try:
        import pretty_midi, soundfile as sf
        pm    = pretty_midi.PrettyMIDI(str(midi_path))
        audio = pm.fluidsynth(fs=44100, sf2_path=soundfont)
        sf.write(str(wav_path), audio, 44100)
        print(f'    -> {wav_path.name}')
        return True
    except Exception: pass
    try:
        fs = subprocess.run(['which', 'fluidsynth'], capture_output=True, text=True).stdout.strip()
        if fs:
            subprocess.run([fs, '-ni', soundfont, str(midi_path), '-F', str(wav_path), '-r', '44100'],
                           capture_output=True, timeout=30)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                print(f'    -> {wav_path.name}')
                return True
    except Exception: pass
    print('    WAV skipped (no soundfont/fluidsynth)')
    return False


def main():
    tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
    print(f'Tokenizer: vocab size {tokenizer.vocab_size}')

    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size,
                               d_model=128, n_heads=8, n_layers=4,
                               context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'v2c_transformer_best.pt',
                                     map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model  = model.to(device)
    model.eval()
    print(f'Model loaded on {device}')

    token_info = build_token_info(tokenizer)
    soundfont  = find_soundfont()
    start_id   = tokenizer.token_to_id['<START>']

    # Load Bach reference histogram for histogram correction
    with open(CHECKPOINT_DIR / 'v2c_music_metrics.json') as f:
        _m = json.load(f)
    bach_ref = _m['sources']['real_bach']['pitch_class_histogram']
    print(f'Histogram correction: strength=3.0 (reduces pitch KL vs Bach)')

    for i in range(1, 4):
        print(f'\n-- Sample {i} --')
        token_ids = generate_keyed(
            model, [start_id], tokenizer, token_info,
            max_new_tokens=480, min_new_tokens=240, temperature=1.0,
            warmup_tokens=60, chromatic_penalty=4.0,
            histogram_correction_strength=3.0, bach_ref_histogram=bach_ref,
            device=device
        )
        for label, suffix, instrs in [
            ('piano', 'piano', PIANO_INSTRUMENTS),
            ('satb',  'satb',  SATB_INSTRUMENTS),
        ]:
            mid = OUTPUT_DIR / f'generated_v2c_keyed_{i}_{suffix}.mid'
            tokens_to_midi(token_ids, tokenizer, mid, instrs)
            if soundfont:
                midi_to_wav(mid, OUTPUT_DIR / f'generated_v2c_keyed_{i}_{suffix}.wav', soundfont)

    print('\nDone.')


if __name__ == '__main__':
    main()
