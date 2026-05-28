"""Evaluate harmonization quality: our model vs random baseline vs real Bach."""
import sys, os, pickle, json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import jensenshannon

try:
    import pretty_midi
except ImportError:
    pretty_midi = None

# Add modeling directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent / 'modeling'))
sys.path.insert(0, str(script_dir.parent / 'modeling_task2'))

from tokenizer_v2 import BachTokenizerV2
from model import ChoraleTransformer
import torch

CHECKPOINT_DIR = script_dir.parent / 'modeling' / 'checkpoints'
OUTPUT_DIR = script_dir
VOICE_NAMES = ['Soprano', 'Alto', 'Tenor', 'Bass']

MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


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
                info[tid] = ('note', v, midi % 12)
            except (ValueError, IndexError):
                info[tid] = ('unknown',)
        else:
            info[tid] = ('unknown',)
    return info


def decode_to_4voice_score(token_ids, tokenizer):
    """
    Decode token sequence to per-voice note lists.
    Format: (time_step, midi_pitch, duration)
    """
    strings = tokenizer.decode(token_ids)
    relevant = [t for t in strings if t not in ('<START>', '<END>', '<BAR>')]
    voice_notes = [[] for _ in range(4)]

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
                    midi_str, dur_str = after_v.split('P')[1].split('D')
                    voice_notes[v].append((t, int(midi_str), int(dur_str)))
                except (ValueError, IndexError):
                    pass
    return voice_notes


def compute_pitch_class_histogram(voice_notes_list):
    """
    Compute 12-bin histogram of pitch classes across all voices.
    Returns numpy array of shape (12,) with normalized counts.
    """
    counts = np.zeros(12)
    total = 0
    for voice_notes in voice_notes_list:
        for t, midi, dur in voice_notes:
            pc = midi % 12
            counts[pc] += 1
            total += 1
    if total > 0:
        counts = counts / total
    return counts


def compute_kl_divergence(generated_hist, reference_hist):
    """
    KL divergence from generated to reference.
    Uses Jensen-Shannon distance (symmetric) for stability.
    """
    # Smooth to avoid log(0)
    eps = 1e-10
    gen = generated_hist + eps
    ref = reference_hist + eps
    gen = gen / gen.sum()
    ref = ref / ref.sum()

    kl = np.sum(ref * (np.log(ref) - np.log(gen)))
    return float(kl)


def compute_scale_consistency(voice_notes_list):
    """
    Compute % of notes that fit the detected key.
    """
    pitch_classes = []
    for voice_notes in voice_notes_list:
        for t, midi, dur in voice_notes:
            pitch_classes.append(midi % 12)

    if not pitch_classes:
        return 0.0

    detected_key, detected_mode, fit = detect_key(pitch_classes)
    if detected_key is None:
        return 0.0

    return fit


def compute_parallel_5ths_rate(voice_notes_list):
    """
    Rate of consecutive soprano-bass interval pairs where both are 7 (mod 12).
    This detects forbidden parallel 5ths.
    """
    soprano = voice_notes_list[0]  # voice 0
    bass = voice_notes_list[3]     # voice 3

    if not soprano or not bass:
        return 0.0

    # Sort by time step
    soprano_sorted = sorted(soprano, key=lambda x: x[0])
    bass_sorted = sorted(bass, key=lambda x: x[0])

    # Extract just the MIDI pitches in order
    sop_pitches = [midi for t, midi, dur in soprano_sorted]
    bas_pitches = [midi for t, midi, dur in bass_sorted]

    if len(sop_pitches) < 2 or len(bas_pitches) < 2:
        return 0.0

    # Check consecutive intervals
    parallel_5ths_count = 0
    total_pairs = 0

    for i in range(min(len(sop_pitches), len(bas_pitches)) - 1):
        # Interval from note i to note i+1
        sop_interval = (sop_pitches[i+1] - sop_pitches[i]) % 12
        bas_interval = (bas_pitches[i+1] - bas_pitches[i]) % 12

        # Parallel 5ths: both intervals == 7
        if sop_interval == 7 and bas_interval == 7:
            parallel_5ths_count += 1

        total_pairs += 1

    if total_pairs == 0:
        return 0.0

    return parallel_5ths_count / total_pairs


def compute_voice_crossing_rate(voice_notes_list):
    """
    Rate of time steps where voice crossing occurs.
    Voice crossing = soprano < alto OR alto < tenor OR tenor < bass
    At each time step where all 4 voices have notes, check if soprano >= alto >= tenor >= bass.
    Rate = violations / total simultaneous positions checked.
    """
    soprano = voice_notes_list[0]
    alto = voice_notes_list[1]
    tenor = voice_notes_list[2]
    bass = voice_notes_list[3]

    # Create dictionaries: time_step -> midi pitch
    sop_dict = {t: midi for t, midi, dur in soprano}
    alt_dict = {t: midi for t, midi, dur in alto}
    ten_dict = {t: midi for t, midi, dur in tenor}
    bas_dict = {t: midi for t, midi, dur in bass}

    # Find all time steps where ALL 4 voices have notes (simultaneous position)
    all_times = set(sop_dict.keys()) & set(alt_dict.keys()) & set(ten_dict.keys()) & set(bas_dict.keys())

    if not all_times:
        return 0.0

    crossing_count = 0
    for t in all_times:
        sop_pitch = sop_dict[t]
        alt_pitch = alt_dict[t]
        ten_pitch = ten_dict[t]
        bas_pitch = bas_dict[t]

        # Check if soprano >= alto >= tenor >= bass
        if not (sop_pitch >= alt_pitch >= ten_pitch >= bas_pitch):
            crossing_count += 1

    return crossing_count / len(all_times)


def load_midi_to_voices(midi_path):
    """
    Load MIDI file and extract per-voice note lists.
    Returns list of 4 voice note lists: [(t, midi, dur), ...]
    """
    if pretty_midi is None:
        return None

    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"Warning: Could not load MIDI {midi_path}: {e}")
        return None

    voice_notes = [[] for _ in range(4)]

    # Try to get instruments in order (Soprano, Alto, Tenor, Bass)
    for voice_idx, instr in enumerate(pm.instruments[:4]):
        if voice_idx >= 4:
            break
        for note in instr.notes:
            # Convert time to nearest 16th note (quarter = 4 sixteenths)
            t = int(round(note.start * 4))
            dur = int(round((note.end - note.start) * 4))
            if dur < 1:
                dur = 1
            voice_notes[voice_idx].append((t, note.pitch, dur))

    return voice_notes


def generate_random_harmonization(soprano_notes, tokenizer, vocab_size=1000):
    """
    Generate random harmonization by sampling uniformly from vocab
    for each alto, tenor, bass position.
    """
    voice_notes = [[], [], [], []]

    # Copy soprano
    voice_notes[0] = list(soprano_notes)

    # Generate random for other voices
    # Use a simple strategy: for each soprano onset, add random notes at the same time
    for t, midi, dur in soprano_notes:
        # Alto, tenor, bass: random MIDI in typical ranges
        for v in [1, 2, 3]:
            # Random pitch between 36-96 (typical range)
            random_midi = np.random.randint(36, 97)
            voice_notes[v].append((t, random_midi, dur))

    return voice_notes


# ─────────────────────────────────────────────────────────────────────────────
# Inline harmonize functions (from modeling_task2/harmonize.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_soprano_tokens(sequence, tokenizer):
    """
    Extract soprano tokens from a full interleaved sequence.
    At each 16th-note position: [v0_token, v1_token, v2_token, v3_token]

    Return list of soprano (voice 0) token IDs, skipping special tokens.
    """
    decoded = tokenizer.decode(sequence)
    soprano_tokens = []

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


def top_p_sample(logits, p=0.95):
    """Nucleus (top-p) sampling."""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens where cumulative prob exceeds p
    sorted_probs[cumulative - sorted_probs > p] = 0.0
    sorted_probs /= sorted_probs.sum()

    sampled_idx = torch.multinomial(sorted_probs, 1)
    return sorted_idx[sampled_idx]


def harmonize(model, soprano_tokens, tokenizer, temperature=1.0, device=None,
              chromatic_penalty=3.0, bias_weight=0.0, bach_pc_hist=None, top_p=1.0):
    """
    Autoregressively harmonize given soprano tokens with optional penalty, PC bias, and top-p.

    At each step:
    - Determine which voice position we're at (step % 4)
    - If voice 0 and we have a soprano token: force it
    - Otherwise: sample from model with temperature
    - Apply key detection + chromatic penalty after 40 tokens
    - Apply PC bias if enabled
    - Apply top-p sampling if p < 1.0
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

                # PC bias: add log-prob bonus proportional to Bach distribution
                if bias_weight > 0.0 and bach_pc_hist is not None:
                    for tid, i in token_info.items():
                        if i[0] == 'note':
                            pc = i[2]
                            logits[0, tid] += bias_weight * np.log(bach_pc_hist[pc] + 1e-8)

                # Apply temperature
                logits = logits / temperature

                # Sample with top-p if p < 1.0
                if top_p < 1.0:
                    sampled = top_p_sample(logits[0], p=top_p)
                    next_token = sampled.view(1, 1)
                else:
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() == end_id:
                break

    return tokens[0].cpu().tolist()


def main():
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

    # Load sequences cache and splits
    with open(CHECKPOINT_DIR / 'v2_sequences_cache.pkl', 'rb') as f:
        cache_dict = pickle.load(f)
        sequences_cache = cache_dict['sequences']

    with open(CHECKPOINT_DIR / 'v2_splits.json', 'r') as f:
        splits = json.load(f)

    test_indices = splits['test'][:10]  # First 10 test indices
    print(f'Using {len(test_indices)} test chorales')

    # Load best config from combination experiment
    best_config_file = Path(__file__).parent.parent / 'modeling_task2' / 'best_config.json'
    if best_config_file.exists():
        with open(best_config_file, 'r') as f:
            best_config = json.load(f)
        chromatic_penalty = best_config['penalty']
        bias_weight = best_config['bias']
        top_p = best_config['top_p']
        print(f'Loaded best config: penalty={chromatic_penalty}, bias={bias_weight}, top_p={top_p}')
    else:
        # Fallback to defaults
        chromatic_penalty = 3.0
        bias_weight = 0.0
        top_p = 1.0
        print(f'Best config not found, using defaults: penalty={chromatic_penalty}, bias={bias_weight}, top_p={top_p}')

    # Load Bach PC histogram for bias
    with open(CHECKPOINT_DIR / 'v2c_music_metrics.json') as f:
        metrics = json.load(f)
    bach_pc_hist_raw = metrics['sources']['real_bach']['pitch_class_histogram']
    bach_pc_hist = np.array(bach_pc_hist_raw, dtype=float)

    # Extract real Bach sequences for test set
    real_bach_voices = []
    for idx in test_indices:
        token_ids = sequences_cache[idx]
        voice_notes = decode_to_4voice_score(token_ids, tokenizer)
        real_bach_voices.append(voice_notes)

    # Load our generated harmonizations from MIDI files (or generate if needed)
    our_model_voices = []
    for idx in range(len(test_indices)):
        midi_file = OUTPUT_DIR / f'harmonized_{idx}_satb.mid'
        if midi_file.exists():
            voices = load_midi_to_voices(midi_file)
            if voices:
                our_model_voices.append(voices)
            else:
                print(f"Warning: Could not load {midi_file}, generating...")
                # Generate using our model
                sequence = sequences_cache[test_indices[idx]]
                soprano_tokens = extract_soprano_tokens(sequence, tokenizer)
                harmonized = harmonize(model, soprano_tokens, tokenizer,
                                     temperature=1.0, device=device,
                                     chromatic_penalty=chromatic_penalty,
                                     bias_weight=bias_weight, bach_pc_hist=bach_pc_hist,
                                     top_p=top_p)
                voices = decode_to_4voice_score(harmonized, tokenizer)
                our_model_voices.append(voices)
        else:
            print(f"MIDI file not found: {midi_file}, generating...")
            # Generate using our model
            sequence = sequences_cache[test_indices[idx]]
            soprano_tokens = extract_soprano_tokens(sequence, tokenizer)
            harmonized = harmonize(model, soprano_tokens, tokenizer,
                                 temperature=1.0, device=device,
                                 chromatic_penalty=chromatic_penalty,
                                 bias_weight=bias_weight, bach_pc_hist=bach_pc_hist,
                                 top_p=top_p)
            voices = decode_to_4voice_score(harmonized, tokenizer)
            our_model_voices.append(voices)

    # Generate random baseline harmonizations
    random_baselines = []
    for idx, real_voices in enumerate(real_bach_voices):
        # Use soprano from real Bach, random alto/tenor/bass
        soprano_notes = real_voices[0]
        np.random.seed(42 + idx)  # For reproducibility
        random_voices = generate_random_harmonization(soprano_notes, tokenizer)
        random_baselines.append(random_voices)

    # Compute metrics
    results = {
        'our_model': {},
        'random_baseline': {},
        'real_bach': {}
    }

    # Reference: Real Bach pitch class histogram (average across test)
    real_bach_histograms = []
    for voices in real_bach_voices:
        hist = compute_pitch_class_histogram(voices)
        real_bach_histograms.append(hist)
    reference_histogram = np.mean(real_bach_histograms, axis=0)

    print("\n=== Extracting metrics for each test chorale ===\n")

    # Our Model
    our_kl_divs = []
    our_scales = []
    our_p5s = []
    our_vcross = []

    for idx, voices in enumerate(our_model_voices):
        kl = compute_kl_divergence(compute_pitch_class_histogram(voices), reference_histogram)
        scale = compute_scale_consistency(voices)
        p5s = compute_parallel_5ths_rate(voices)
        vcross = compute_voice_crossing_rate(voices)

        our_kl_divs.append(kl)
        our_scales.append(scale)
        our_p5s.append(p5s)
        our_vcross.append(vcross)

        print(f"  Our Model {idx}: KL={kl:.3f}, Scale={scale:.2%}, P5s={p5s:.2%}, Vcross={vcross:.2%}")

    # Random Baseline
    rand_kl_divs = []
    rand_scales = []
    rand_p5s = []
    rand_vcross = []

    for idx, voices in enumerate(random_baselines):
        kl = compute_kl_divergence(compute_pitch_class_histogram(voices), reference_histogram)
        scale = compute_scale_consistency(voices)
        p5s = compute_parallel_5ths_rate(voices)
        vcross = compute_voice_crossing_rate(voices)

        rand_kl_divs.append(kl)
        rand_scales.append(scale)
        rand_p5s.append(p5s)
        rand_vcross.append(vcross)

        print(f"  Random {idx}: KL={kl:.3f}, Scale={scale:.2%}, P5s={p5s:.2%}, Vcross={vcross:.2%}")

    # Real Bach
    real_kl_divs = []
    real_scales = []
    real_p5s = []
    real_vcross = []

    for idx, voices in enumerate(real_bach_voices):
        kl = compute_kl_divergence(compute_pitch_class_histogram(voices), reference_histogram)
        scale = compute_scale_consistency(voices)
        p5s = compute_parallel_5ths_rate(voices)
        vcross = compute_voice_crossing_rate(voices)

        real_kl_divs.append(kl)
        real_scales.append(scale)
        real_p5s.append(p5s)
        real_vcross.append(vcross)

        print(f"  Real Bach {idx}: KL={kl:.3f}, Scale={scale:.2%}, P5s={p5s:.2%}, Vcross={vcross:.2%}")

    # Average across the 10 chorales
    avg_our_kl = np.mean(our_kl_divs)
    avg_our_scale = np.mean(our_scales) * 100
    avg_our_p5s = np.mean(our_p5s) * 100
    avg_our_vcross = np.mean(our_vcross) * 100

    avg_rand_kl = np.mean(rand_kl_divs)
    avg_rand_scale = np.mean(rand_scales) * 100
    avg_rand_p5s = np.mean(rand_p5s) * 100
    avg_rand_vcross = np.mean(rand_vcross) * 100

    avg_real_kl = np.mean(real_kl_divs)
    avg_real_scale = np.mean(real_scales) * 100
    avg_real_p5s = np.mean(real_p5s) * 100
    avg_real_vcross = np.mean(real_vcross) * 100

    # Print results table
    print("\n" + "="*80)
    print("=== TASK 2 EVALUATION (n=10 test chorales) ===")
    print("="*80)
    print(f"{'Metric':<30} | {'Our Model':>12} | {'Random':>12} | {'Real Bach':>12}")
    print("-"*80)
    print(f"{'Scale consistency ↑':<30} | {avg_our_scale:>11.1f}% | {avg_rand_scale:>11.1f}% | {avg_real_scale:>11.1f}%")
    print(f"{'Voice crossing rate ↓':<30} | {avg_our_vcross:>11.1f}% | {avg_rand_vcross:>11.1f}% | {avg_real_vcross:>11.1f}%")
    print(f"{'Parallel 5ths rate ↓':<30} | {avg_our_p5s:>11.1f}% | {avg_rand_p5s:>11.1f}% | {avg_real_p5s:>11.1f}%")
    print(f"{'Pitch KL divergence ↓':<30} | {avg_our_kl:>12.3f} | {avg_rand_kl:>12.3f} | {avg_real_kl:>12.3f}")
    print("="*80)

    # Save metrics to JSON
    metrics_dict = {
        'num_test_chorales': len(test_indices),
        'our_model': {
            'scale_consistency': float(avg_our_scale),
            'voice_crossing_rate': float(avg_our_vcross),
            'parallel_5ths_rate': float(avg_our_p5s),
            'pitch_kl_divergence': float(avg_our_kl),
        },
        'random_baseline': {
            'scale_consistency': float(avg_rand_scale),
            'voice_crossing_rate': float(avg_rand_vcross),
            'parallel_5ths_rate': float(avg_rand_p5s),
            'pitch_kl_divergence': float(avg_rand_kl),
        },
        'real_bach': {
            'scale_consistency': float(avg_real_scale),
            'voice_crossing_rate': float(avg_real_vcross),
            'parallel_5ths_rate': float(avg_real_p5s),
            'pitch_kl_divergence': float(avg_real_kl),
        }
    }

    metrics_file = OUTPUT_DIR / 'harmony_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Create visualization
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid", {'grid.linestyle': '-', 'grid.linewidth': 0.5})
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sources = ['Our Model', 'Random', 'Real Bach']
        colors = {'Our Model': '#4C72B0', 'Random': '#DD8452', 'Real Bach': '#55A868'}
        color_list = [colors[s] for s in sources]

        x_pos = np.arange(len(sources))
        width = 0.25

        # Left panel: Scale Consistency and Voice Crossing Rate
        scale_vals = [avg_our_scale, avg_rand_scale, avg_real_scale]
        vcross_vals = [avg_our_vcross, avg_rand_vcross, avg_real_vcross]

        ax = axes[0]
        bars1 = ax.bar(x_pos - width/2, scale_vals, width, label='Scale Consistency (%)',
                       color=color_list, alpha=0.8, edgecolor='black', linewidth=0.7)
        bars2 = ax.bar(x_pos + width/2, vcross_vals, width, label='Voice Crossing Rate (%)',
                       color=color_list, alpha=0.5, edgecolor='black', linewidth=0.7)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title('Scale Consistency & Voice Crossing', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sources, fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        # Right panel: Pitch KL divergence
        kl_vals = [avg_our_kl, avg_rand_kl, avg_real_kl]

        ax = axes[1]
        bars = ax.bar(x_pos, kl_vals, width*2, color=color_list, alpha=0.8,
                      edgecolor='black', linewidth=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        ax.set_ylabel('KL Divergence (lower is better)', fontsize=10)
        ax.set_title('Pitch Class Distribution (KL)', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sources, fontsize=9)
        ax.set_ylim(0, max(kl_vals) * 1.15)
        ax.text(0.5, -0.25, '* Random KL is low by chance', transform=ax.transAxes,
               fontsize=8, ha='center', style='italic', color='gray')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        img_dir = OUTPUT_DIR.parent / 'images'
        img_dir.mkdir(exist_ok=True, parents=True)
        fig_path = img_dir / 'task2_evaluation.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
        plt.close()

    except ImportError as e:
        print(f"Warning: Could not create visualization (missing module: {e})")


if __name__ == '__main__':
    main()
