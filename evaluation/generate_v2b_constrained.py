"""Generate 3 MIDI samples from the v2b model with voice range constraints and nucleus sampling."""
import sys, os, subprocess
sys.path.insert(0, '../modeling')

import torch
import torch.nn.functional as F
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

# Voice ranges (MIDI): Soprano, Alto, Tenor, Bass
VOICE_RANGES = [
    (60, 81),   # Soprano
    (53, 74),   # Alto
    (48, 69),   # Tenor
    (40, 62),   # Bass
]


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


def generate_constrained(model, start_tokens, tokenizer, max_new_tokens=400, temperature=0.9,
                        nucleus_p=0.9, device=None):
    """
    Autoregressive generation with voice range constraints and nucleus sampling.

    Args:
        model: ChoraleTransformer model
        start_tokens: List of initial token IDs
        tokenizer: BachTokenizerV2 instance
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        nucleus_p: Nucleus (top-p) sampling threshold
        device: Device to run on

    Returns:
        List of generated token IDs
    """
    if device is None:
        device = next(model.parameters()).device

    # Build a mapping from token id to (voice, midi, dur) for note tokens
    token_to_info = {}  # id -> (voice, midi, dur) or None for special/rest
    for tid, tok_str in tokenizer.id_to_token.items():
        if tok_str in ('<START>', '<END>', '<BAR>', '<SUSTAIN>'):
            token_to_info[tid] = ('special', None, None)
        elif 'REST' in tok_str:
            # V{v}REST
            try:
                v = int(tok_str[1])
                token_to_info[tid] = ('rest', v, None)
            except (ValueError, IndexError):
                token_to_info[tid] = ('unknown', None, None)
        elif tok_str.startswith('V') and 'P' in tok_str and 'D' in tok_str:
            # V{v}P{midi}D{dur}
            try:
                after_v = tok_str[1:]
                v = int(after_v.split('P')[0])
                rest = after_v.split('P')[1]
                midi_str, dur_str = rest.split('D')
                midi = int(midi_str)
                dur = int(dur_str)
                token_to_info[tid] = ('note', v, midi)
            except (ValueError, IndexError):
                token_to_info[tid] = ('unknown', None, None)
        else:
            token_to_info[tid] = ('unknown', None, None)

    # Start with initial tokens
    tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)  # (1, len(start_tokens))

    model.eval()
    with torch.no_grad():
        for gen_step in range(max_new_tokens):
            # Determine current voice being generated
            # Position 0 = <START>, position 1 = first generated token = voice 0
            # So num_tokens_generated = len(generated) counts from 0
            # voice_idx = num_tokens_generated % 4
            num_generated = tokens.shape[1] - len(start_tokens)
            voice_idx = num_generated % 4

            # Get last context_len tokens
            seq_len = tokens.shape[1]
            if seq_len > model.context_len:
                tokens_in = tokens[:, -model.context_len:]
            else:
                tokens_in = tokens

            # Forward pass
            logits = model.forward(tokens_in)  # (1, seq_len, vocab_size)
            logits = logits[:, -1, :]  # (1, vocab_size) - last token logits

            # Apply voice range mask
            logits_masked = logits.clone()
            low_midi, high_midi = VOICE_RANGES[voice_idx]

            for tid in range(tokenizer.vocab_size):
                tok_type, v, midi = token_to_info.get(tid, ('unknown', None, None))

                # Always allow special tokens, sustain
                if tok_type == 'special':
                    continue

                # Allow REST tokens for any voice
                if tok_type == 'rest':
                    continue

                # For note tokens: only allow if voice matches AND midi in range
                if tok_type == 'note':
                    if v != voice_idx or midi < low_midi or midi > high_midi:
                        logits_masked[0, tid] = float('-inf')
                else:
                    # Unknown tokens: mask out
                    logits_masked[0, tid] = float('-inf')

            # Check if all logits are -inf (fallback to allow sustain/rest)
            if torch.all(torch.isinf(logits_masked)):
                logits_masked = logits.clone()
                for tid in range(tokenizer.vocab_size):
                    tok_type, v, midi = token_to_info.get(tid, ('unknown', None, None))
                    if tok_type == 'special' or tok_type == 'rest':
                        continue
                    logits_masked[0, tid] = float('-inf')

            # Temperature scaling
            logits_masked = logits_masked / temperature

            # Nucleus (top-p) sampling
            probs = F.softmax(logits_masked, dim=-1)  # (1, vocab_size)
            probs_sorted, indices = torch.sort(probs[0], descending=True)

            # Find cumulative sum and cutoff
            cumsum = torch.cumsum(probs_sorted, dim=0)
            mask = cumsum <= nucleus_p
            # Always include at least the top token
            mask[0] = True

            # Zero out everything below cutoff
            probs_nucleus = probs_sorted.clone()
            probs_nucleus[~mask] = 0
            probs_nucleus = probs_nucleus / probs_nucleus.sum()

            # Sample from nucleus
            sampled_idx = torch.multinomial(probs_nucleus, num_samples=1)  # (1,)
            next_token_id = indices[sampled_idx.item()]
            next_token = torch.tensor([[next_token_id]], dtype=torch.long, device=device)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop if we hit <END>
            if next_token_id.item() == tokenizer.token_to_id.get('<END>', -1):
                break

    return tokens[0].cpu().tolist()


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
    return voice_notes, total


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


def compute_metrics(voice_notes_list, all_token_ids_list, tokenizer):
    """
    Compute quick metrics across samples.

    Args:
        voice_notes_list: List of 3 voice_notes results
        all_token_ids_list: List of 3 token sequences
        tokenizer: BachTokenizerV2
    """
    print("\n" + "="*70)
    print("CONSTRAINED GENERATION METRICS")
    print("="*70)

    for sample_idx, (voice_notes, token_ids) in enumerate(zip(voice_notes_list, all_token_ids_list), 1):
        print(f"\nSample {sample_idx}:")

        # Note density
        token_strings = tokenizer.decode(token_ids)
        relevant = [t for t in token_strings if t not in ('<START>', '<END>', '<BAR>')]
        n_timesteps = len(relevant) // 4 if len(relevant) > 0 else 1
        total_notes = sum(len(notes) for notes in voice_notes)
        density = total_notes / max(1, n_timesteps) if n_timesteps > 0 else 0
        print(f"  Note density: {density:.2f} notes/timestep")

        # Voice crossing count
        crossing_count = 0
        for v in range(3):
            notes_v = voice_notes[v]
            notes_v_plus = voice_notes[v + 1]
            # Check for intersecting ranges
            if notes_v and notes_v_plus:
                midi_v = [m for _, m, _ in notes_v]
                midi_v_plus = [m for _, m, _ in notes_v_plus]
                max_v = max(midi_v)
                min_v_plus = min(midi_v_plus)
                if max_v > min_v_plus:
                    # Count how many notes in v_plus are below some notes in v
                    for m_v in midi_v:
                        for m_vp in midi_v_plus:
                            if m_vp < m_v:
                                crossing_count += 1
        print(f"  Voice crossings: {crossing_count}")

        # MIDI range violations
        range_violations = 0
        for v in range(4):
            notes = voice_notes[v]
            low, high = VOICE_RANGES[v]
            for onset, midi, dur in notes:
                if midi < low or midi > high:
                    range_violations += 1
        print(f"  MIDI range violations: {range_violations}")


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
    print(f'Using constrained generation with voice ranges:')
    for v, name in enumerate(VOICE_NAMES):
        low, high = VOICE_RANGES[v]
        print(f'  {name} (V{v}): MIDI {low}–{high}')

    soundfont = find_soundfont()
    start_id = tokenizer.token_to_id['<START>']

    voice_notes_list = []
    all_token_ids_list = []

    for i in range(1, 4):
        print(f'\nGenerating sample {i}...')
        token_ids = generate_constrained(model, [start_id], tokenizer,
                                        max_new_tokens=400, temperature=0.9,
                                        nucleus_p=0.9, device=device)
        all_token_ids_list.append(token_ids)

        mid_path = OUTPUT_DIR / f'generated_v2b_constrained_{i}.mid'
        voice_notes, total = tokens_to_midi(token_ids, tokenizer, mid_path)
        voice_notes_list.append(voice_notes)

        if soundfont:
            wav_path = OUTPUT_DIR / f'generated_v2b_constrained_{i}.wav'
            midi_to_wav(mid_path, wav_path, soundfont)

    # Compute and print metrics
    compute_metrics(voice_notes_list, all_token_ids_list, tokenizer)

    print('\n' + "="*70)
    print("Done.")
    print("="*70)


if __name__ == '__main__':
    main()
