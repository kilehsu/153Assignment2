"""
Task 2: Soprano-conditioned harmonization via prefix-conditioning.

Given soprano tokens extracted from a real Bach chorale, generate
the remaining three voices (Alto, Tenor, Bass) autoregressively.
At each voice-0 position, force the known soprano token.
At voice-1/2/3 positions, sample from the model with temperature.
"""
import sys, os, pickle, json, subprocess
from pathlib import Path

# Import model components
modeling_path = Path(__file__).parent.parent / 'modeling'
sys.path.insert(0, str(modeling_path))

import torch
import torch.nn.functional as F
from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer
import music21.stream as m21stream
import music21.note as m21note
import music21.instrument as m21instr

script_dir = Path(__file__).parent
CHECKPOINT_DIR = script_dir.parent / 'modeling' / 'checkpoints'
OUTPUT_DIR     = script_dir.parent / 'evaluation_task2'
SOUNDFONT_PATHS = [
    '/opt/homebrew/share/soundfonts/default.sf2',
    os.path.expanduser('~/Library/Audio/Sounds/Banks/FluidR3_GM.sf2'),
    str(CHECKPOINT_DIR / 'MuseScore_General.sf3'),
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


def extract_soprano_tokens(sequence, tokenizer):
    """
    Extract soprano tokens from a full interleaved sequence.
    At each 16th-note position: [v0_token, v1_token, v2_token, v3_token]

    Return list of soprano (voice 0) token IDs, skipping special tokens.
    """
    decoded = tokenizer.decode(sequence)
    soprano_tokens = []

    # Position 0 is START, then we have groups of 4 tokens
    # Position 1 (in group 0) is voice 0, position 2 is voice 1, etc.
    # At next group: positions 5, 6, 7, 8 -> voice 0 is at 5
    # Pattern: voice 0 is at positions 1, 5, 9, 13, ... = 1 + 4*k

    i = 1  # Start after START token
    while i < len(decoded):
        tok = decoded[i]
        if tok in ('<START>', '<END>', '<BAR>'):
            i += 1
            continue
        # We're at a voice 0 position (position 1, 5, 9, ...)
        soprano_tokens.append(sequence[i])
        i += 4  # Next voice 0 is 4 positions away

    return soprano_tokens


def harmonize(model, soprano_tokens, tokenizer, temperature=1.0, device=None,
              chromatic_penalty=3.0):
    """
    Autoregressively harmonize given soprano tokens.

    At each step:
    - Determine which voice position we're at (step % 4)
    - If voice 0 and we have a soprano token: force it
    - Otherwise: sample from model with temperature
    - Apply key detection + chromatic penalty after 40 tokens
    - Stop when all soprano consumed OR model emits END OR 600 steps

    Returns list of token IDs.
    """
    if device is None:
        device = next(model.parameters()).device

    start_id   = tokenizer.token_to_id.get('<START>', 0)
    end_id     = tokenizer.token_to_id.get('<END>', -1)
    sustain_id = tokenizer.token_to_id.get('<SUSTAIN>', -1)

    tokens = torch.tensor([[start_id]], dtype=torch.long, device=device)
    soprano_idx = 0  # How many soprano tokens we've used
    detected_key = None
    detected_mode = None
    key_pcs = None
    token_info = build_token_info(tokenizer)

    model.eval()
    with torch.no_grad():
        for step in range(600):
            num_generated = tokens.shape[1] - 1  # Exclude START
            current_voice = num_generated % 4  # Which voice we're predicting

            # Key detection after warmup
            if num_generated == 40 and detected_key is None:
                pitch_classes = []
                for tid in tokens[0].tolist():
                    i = token_info.get(tid, ('unknown',))
                    if i[0] == 'note':
                        pitch_classes.append(i[2])
                detected_key, detected_mode, fit = detect_key(pitch_classes)
                key_pcs = ALL_SCALES[(detected_key, detected_mode)]
                print(f'    Key detected: {NOTE_NAMES[detected_key]} {detected_mode} '
                      f'(fit={fit:.2%}, after {40} tokens)')

            tokens_in = tokens[:, -model.context_len:]
            logits = model.forward(tokens_in)[:, -1, :]

            # If voice 0 and we have a soprano token: force it
            if current_voice == 0 and soprano_idx < len(soprano_tokens):
                next_token = torch.tensor([[soprano_tokens[soprano_idx]]],
                                        dtype=torch.long, device=device)
                soprano_idx += 1
            else:
                # Suppress END until we've used all soprano tokens
                if soprano_idx < len(soprano_tokens):
                    logits[0, end_id] = float('-inf')

                # Chromatic penalty
                if key_pcs is not None:
                    for tid, i in token_info.items():
                        if i[0] == 'note' and i[2] not in key_pcs:
                            logits[0, tid] -= chromatic_penalty

                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == end_id:
                print(f'    <END> at token {num_generated}')
                break

    return tokens[0].cpu().tolist()


# ── MIDI / WAV helpers (copied from generate_v2c_keyed.py) ──────────────────────


def find_soundfont():
    for p in SOUNDFONT_PATHS:
        if os.path.exists(p): return p
    try:
        r = subprocess.run(['find', '/opt/homebrew', '/usr', '-name', '*.sf*', '-maxdepth', '8'],
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = BachTokenizerV2.load(str(CHECKPOINT_DIR / 'v2_tokenizer.json'))
    print(f'Tokenizer: vocab size {tokenizer.vocab_size}')

    # Load model
    model = ChoraleTransformer(vocab_size=tokenizer.vocab_size,
                               d_model=128, n_heads=8, n_layers=4,
                               context_len=512, dropout=0.0)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / 'v2c_transformer_best.pt',
                                     map_location='cpu', weights_only=True))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    print(f'Model loaded on {device}')

    # Load test sequences
    with open(CHECKPOINT_DIR / 'v2_sequences_cache.pkl', 'rb') as f:
        cache = pickle.load(f)
    sequences = cache['sequences']
    print(f'Loaded {len(sequences)} sequences')

    with open(CHECKPOINT_DIR / 'v2_splits.json') as f:
        splits = json.load(f)
    test_indices = splits['test']
    print(f'Test split: {len(test_indices)} sequences')

    soundfont = find_soundfont()
    if soundfont:
        print(f'Soundfont found: {soundfont}')

    # Harmonize 3 test chorales
    for idx_in_test, test_idx in enumerate(test_indices[:3]):
        print(f'\n-- Harmonizing test chorale {idx_in_test} (global index {test_idx}) --')

        sequence = sequences[test_idx]
        soprano_tokens = extract_soprano_tokens(sequence, tokenizer)
        print(f'  Soprano tokens: {len(soprano_tokens)}')

        harmonized = harmonize(model, soprano_tokens, tokenizer,
                             temperature=1.0, device=device, chromatic_penalty=3.0)
        print(f'  Generated: {len(harmonized)} tokens')

        for label, suffix, instrs in [
            ('piano', 'piano', PIANO_INSTRUMENTS),
            ('satb',  'satb',  SATB_INSTRUMENTS),
        ]:
            mid = OUTPUT_DIR / f'harmonized_{idx_in_test}_{suffix}.mid'
            tokens_to_midi(harmonized, tokenizer, mid, instrs)
            if soundfont:
                midi_to_wav(mid, OUTPUT_DIR / f'harmonized_{idx_in_test}_{suffix}.wav', soundfont)

    print('\nDone.')


if __name__ == '__main__':
    main()
