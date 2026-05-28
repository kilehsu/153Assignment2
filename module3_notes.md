# Module 3: Symbolic Music Generation
**Course:** Machine Learning for Music

---

## Table of Contents
1. [Overview & Four Paradigms](#overview)
2. [3.1 Historical Approaches](#31-historical-approaches)
3. [3.2 Next-Token Prediction: Markov Chains & HMMs](#32-next-token-prediction)
4. [3.3 Music Tokenization](#33-music-tokenization)
5. [3.4 Language Modeling Framework](#34-language-modeling-framework)
6. [3.5 Language Models for Symbolic Music Generation](#35-language-models-for-symbolic-music)
7. [3.6 2D Representations & GANs](#36-2d-representations--gans)
8. [Appendix: Datasets](#appendix-datasets)

---

## Overview

### Four Paradigms of Music Generation
```
text        → MIDI / symbolic tokens     → Symbolic music generation (Module 3)
image       → waveform / spectrogram     → Audio-domain generation (Module 4)
piano roll  → 2D image-like              → GAN-based approaches
```

### Two Main Symbolic Approaches
| Approach | Models | Description |
|---|---|---|
| **Text-based** | RNNs, LSTMs, Transformers | Treat music like NLP sequences |
| **Image-based** | GANs, VAEs, Diffusion | Treat piano roll like a 2D image |

### Possible Tasks
| Type | Examples |
|---|---|
| **Unconditional** | ∅ → melody, ∅ → chords, ∅ → lead sheet |
| **Conditional** | melody → lead sheet, melody → multitrack, solo → multitrack |
| **Multimodal** | text → music, video → music |

---

## 3.1 Historical Approaches

### Grammars
- A grammar = rules to expand high-level symbols into sequences
- Rules can be designed by hand, learned from corpus, or via evolutionary algorithms
- **Limitations:** fail to capture internal coherency; hard to manually define rules that produce good compositions

### L-Systems (Prusinkiewicz, 1986)
- Rewrite system where production rules are applied repeatedly
- **Example (Hilbert curve):**
  - Alphabet: A, B; Constants: F + −
  - Production: `A → +BF−AFA−FB+`, `B → −AF+BFB+FA−`
- Each horizontal line → pitch; line length → duration
- Advantages: successive notes are close; has repeated patterns
- Criticism: "cheats" by limiting to major scale notes

### Rule-Based Systems
- Encode music theory rules algorithmically
- Hierarchy of rules: harmony, melody, rhythm
- **Example: Vivace (Thomas, 1985)** — 7-module system for 4-part chorale harmonization
  - Modules: Melody Writer, Harmonic Rhythm Selector, Phrase Shaper, Chord Designator, etc.

### Evolutionary / Genetic Algorithms
- Steps: Initialization → Evaluation (fitness function) → Selection → Reproduction with variation
- **Example (Towsey et al., 2001):** fitness function = weighted sum of features:
  - Pitch variety, Key Centered, Contour Stability, Note Density
- Fitness functions survive as a means of offline evaluation of generated music

### Aleatoric (Random) Music
- "Depending on chance" — e.g., Mozart's Musikalisches Würfelspiel (1792)
- Goal: **not** true randomness, but a random process likely to generate musical output

### Early ANNs
- Todd (1989): 3-layer recurrent ANN for monophonic melody (absolute pitches)
- Duff (1989): Bach-style, relative pitch encoding
- Mozer (1991): RNN capturing local and global patterns

---

## 3.2 Next-Token Prediction

### Music as a Stochastic Process
- Model likelihood `p(x)` of musical events
- Goal: generate sequences that are **likely** under the model (i.e., musical)

### Markov Chains
**Core idea:** `P(next event | history) = P(next event | previous event)`

**Transition probabilities:**
```
P(Si → Sj) = count(Si → Sj) / count(Si → any)
```

**Generation steps:**
1. Pick starting state uniformly at random
2. Select subsequent states using transition probabilities

**From Twinkle Twinkle Little Star (C C G G A A G):**
- Context-free model: probability = 1/279,936
- Ideal Markov model: probability = 1/192

### N-grams (Generalization of Markov Chains)
| N | Name | Context |
|---|---|---|
| 1 | Unigram | No context |
| 2 | Bigram (= Markov chain) | 1 previous token |
| 3 | Trigram | 2 previous tokens |

- For N-gram: num transition probs = |states|^N

### Markov Chains as Matrices
```
        Next
Context  C    G    A
C       [½    ½    0  ]
G       [0    ½    ½  ]
A       [0    ½    ½  ]
```

### Practical Example: Jazz Chord Progressions
- Dataset: 456 jazz standards from [Jazzomat Research Project](https://jazzomat.hfm-weimar.de/)
- Parse chord progressions → extract bigrams → compute transition probs → sample
- **Key insight:** must limit vocabulary to a single key's chord set to avoid wandering

### Limitations of Markov Chains
- Low-order: strange, aimless compositions
- High-order: essentially rehashes corpus; computationally expensive
- Still useful as a pedagogical model for the "token generation" paradigm

### Evaluation: Perplexity
$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i}\log P(x_i | x_{<i})\right)$$

- Measures how "surprised" the model is by held-out sequences
- **Lower perplexity = better model** (assigns higher probability to test data)
- Used both for training (as loss) and evaluation

### Hidden Markov Models (HMMs)
**Extension:** model has internal "hidden" states; observations depend on hidden state

**Components:**
- **Transition probabilities:** `P(Si → Sj)`
- **Emission probabilities:** `P(Si → Oi)` (hidden state → observation)
- **Initial state** `S0`

**Musical hidden states:**
- Current key
- Underlying chord / chord function
- Section (A vs B)

### HMMs for Chorale Harmonization (Allan & Williams, 2004)
- **Task:** given melody (observed), estimate underlying chords (hidden)
- **Model:**
  - Observed states: absolute pitch of melody note
  - Hidden states: pitch intervals + harmonic labels (e.g., 'T' = tonic)
- **Learning:** transition & emission probs measured from training data
- **Inference:** Viterbi algorithm finds most likely hidden state sequence
- Separate models for major and minor keys
- **Assumptions:** harmonic state constant within a beat; first-order Markov

### Coconet & Bach Doodle (Huang et al., 2017)
- Recent harmonization using CNN (not HMM)
- [Interactive demo](https://doodles.google/doodle/celebrating-johann-sebastian-bach/)

---

## 3.3 Music Tokenization

### Core Challenges
Compared to text tokenization, music has:
- Events that occur **simultaneously and overlap**
- Significant **metadata** (instrumentation, dynamics, phrasing)
- Complex **timing information**
- Inherently hard-to-read notation for humans and machines

### ABC Notation (1997)
Simple text-based notation for monophonic tunes:
```
X:571
T:Ah! vous dirai-je, maman (Twinkle Twinkle Little Star)
M:C          ← meter
L:1/4        ← unit note length
Q:120        ← tempo
K:C          ← key
CCGG|AAG2|FFEE|DDC2:|
```
- Letters A–G: pitches; `^`=sharp, `_`=flat, `=`=natural
- Lower case = higher octave
- **Limitations:** monophonic only, limited expressiveness

**Extension: NotaGen (Wang et al., 2025)** — interleaved ABC for multi-track via `[V:]` voice indicators

### MIDI Representation
**MIDI messages:**
- `Note_on` / `Note_off`
- `Time_shift`
- `Program_change` (instrument, 0–127)
- `Control_change`, `Pitch_bend`

**MIDI note numbers:** 0–127; Middle C = 60

**MIDI-like token sequence:**
```
Note_on_67, Time_shift_quarter_note, Note_off_67,
Note_on_67, Time_shift_quarter_note, Note_off_67, ...
```

**Polyphonic (multiple simultaneous pitches):**
```
Note_on_65, Note_on_68, Time_shift_eighth_note, Note_on_77, ...
```

### Performance RNN / MIDI-like (Simon & Oore, 2017)
- Dataset: MAESTRO (Yamaha e-Piano Competition)
- Vocabulary: 128 Note-On + 128 Note-Off + 100 Time-Shift (10ms–1s) + 32 Set-Velocity

**Limitation:** no high-level semantic info (downbeat, tempo, chords) — must be learned implicitly

### REMI (Huang & Yang, 2020)
"Represent MIDI following the way humans read music"
- `Note-Off` → **Note Duration**
- `Time-Shift` → **Bar & Position** (beat-based, structured)
- Additional tokens: **Tempo**, **Chord**
- Outperforms MIDI-like for pop piano generation

### Compound Word (Hsiao et al., 2021)
- Groups tokens into "Compound Words": e.g., `(pitch, duration, velocity)` in one token
- Predicts multiple token types simultaneously → shorter sequences
- Reduces sequence length → better at capturing long-term structure

### Multitrack Music Machine / MMT (Ens & Pasquier, 2020 / Dong et al., 2023)
**MMM structure:**
```
PIECE_START
  TRACK_START [INST] [DENSITY] <bars> TRACK_END
  TRACK_START [INST] [DENSITY] <bars> TRACK_END
  ...
```
- Supports **inpainting** via `BAR-FILL` / `FILL_IN` placeholder tokens

**MMT (Dong et al., 2023):** Compound Word representation `(beat, position, pitch, duration, instrument)`

### Instrument Representation
- MIDI Program Change: 128 instruments in 16 families (program numbers 0–127)

### MidiTok Library
```python
from miditok import REMI, TokenizerConfig
from symusic import Score

config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=True)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=train_files)
tokenizer.save("tokenizer.json")
```

### Tokenization Summary
| Method | Granularity | Polyphonic | Meter-aware | Notes |
|---|---|---|---|---|
| ABC | Character-like | No | Bar-level | Simple; GPT-compatible |
| MIDI-like | Fine | Yes | No | Performance RNN |
| REMI | Medium | Yes | Yes | Beat/bar-explicit |
| Compound Word | Coarser | Yes | Yes | Fewer tokens |

---

## 3.4 Language Modeling Framework

### Language Models (General)
A LM defines a probability distribution over sequences:
$$P(x_i \mid x_1, x_2, \ldots, x_{i-1})$$

**Architectures:**
| Architecture | Objective | Examples |
|---|---|---|
| Masked LM (MLM) | Fill in the gaps | BERT, RoBERTa |
| Autoregressive LM | Predict next token | GPT, FolkRNN |
| Encoder-Decoder | Sequence-to-sequence | T5 |

### Autoregressive Generation
```
<SOT> → The → lively → jazz → tune → in → F → major → ... → <EOT>
```
Autoregressiveness enforced via causal masks (Transformers) or recurrence (RNNs).

### Conditioning
`P(output | condition)` — condition can be:
- Prefix tokens
- Encoder output (cross-attention)
- Feature concatenation
- Class label embedding

### Music as Language
Replace text tokens with music tokens → same framework applies:
- Unconditional: `P(music_tokens)`
- Conditional: `P(music_tokens | melody)`, `P(music_tokens | chord_sequence)`

---

## 3.5 Language Models for Symbolic Music

### RNNs and LSTMs

**Vanilla RNN:**
- Hidden state: `h_t = f(h_{t-1}, x_t)`
- Limited long-term memory

**LSTM:**
- Adds forget gate, input gate, output gate
- Better long-term memory via cell state
- Character-level or word-level variants

### Folk RNN (Sturm et al., 2015)
- Data: folk tunes collections
- Representation: ABC notation (without metadata)
- Model: character-level LSTM
- [folkrnn.org](https://folkrnn.org)

### RNN-based Models (Historical SOTA)
| Model | Year | Notes |
|---|---|---|
| Folk RNN | 2015 | ABC + char LSTM |
| Melody RNN | 2016 | Magenta |
| Performance RNN | 2017 | MIDI-like + LSTM |
| DeepBach | 2017 | Bach chorales |

**Limitation:** struggles with very long sequences

### Transformers
- Uses **self-attention mechanism** — learns what to attend to from data
- Better than RNNs for long sequences
- Decoder-only: for unconditional/prompt-based generation
- Encoder-decoder: for conditional tasks

**Self-attention:**
```
For each token: compute query (q), key (k), value (v)
Attention score = softmax(q · k^T / √d)
Output = weighted sum of values
```

### Music Transformer (Huang et al., 2019)
- Data: MAESTRO dataset
- Representation: MIDI-like
- Model: Transformer (decoder-only)
- Key finding: self-attention learns musically meaningful patterns (rhythm, harmony)
- [Demo](https://magenta.tensorflow.org/music-transformer)

### Transformer-based Models
| Model | Year | Representation | Notes |
|---|---|---|---|
| Music Transformer | 2019 | MIDI-like | Long-term structure |
| LakhNES | 2019 | MIDI-like | Cross-domain pretraining |
| Pop Music Transformer | 2020 | REMI | Beat-based |
| Multitrack Music Transformer | 2023 | Compound Word | Multi-instrument |

### Code Example (workbook3.ipynb)
Simple RNN-based model provided as starting point for Assignment 2.

---

## 3.6 2D Representations & GANs

### Piano Rolls (Recap)
- 2D matrix: time × pitch
- Brightness = MIDI velocity (dynamics)
- Simultaneous notes handled naturally
- Benefits:
  - Many musical patterns are **translation-invariant** in time and pitch axis
  - Easy to "see" repeated patterns
  - Can use CNNs: learn invariances across pitch and time

### CNNs for Music
- Convolutional layers learn reusable local pattern detectors (kernels)
- Applied to piano roll as 2D input

### Coconet (Huang et al., 2017) — Bach Doodle
- Task: complete partial musical scores (inpainting)
- Input: 3D feature map (time × pitch × voice: SATB)
- Output: probability distributions over erased note pitches
- **Sampling:** iterative rewriting — randomly erase notes, resample, repeat until coherent
- Can develop material in **any order** (not just left-to-right)
- [Google Doodle demo](https://doodles.google/doodle/celebrating-johann-sebastian-bach/)

### GANs (Goodfellow et al., 2014)
**Architecture:**
```
Random noise → Generator → Fake samples
                                ↓
Real samples  → Discriminator → Real/Fake prediction
```
- **Generator:** learns to fool discriminator
- **Discriminator:** learns to distinguish real vs. fake
- Adversarial training until equilibrium

### MidiNet (Yang et al., 2017)
- Convolutional GAN for **one-bar melody generation**
- Conditioned on: previous bar (2D) or chord progression (1D)
- Generates music measure-by-measure

### MuseGAN (Dong et al., 2018)
- Multi-track MIDI generation (piano, guitar, bass, strings, drums)
- Input tensor: `4 bars × 96 time steps × 84 pitches × 5 tracks`
- Three generator models:
  1. **Jamming:** each track has independent generator
  2. **Composer:** single generator for all tracks (knows harmonic structure)
  3. **Hybrid:** combination
- [Demo](https://hermandong.com/musegan/results)

### Multitrack Piano Roll
- Extend piano roll with a **track dimension**
- Tensor shape: `(bars, time_steps, pitches, tracks)`

---

## Appendix: Datasets

| Dataset | Size | Format | Content |
|---|---|---|---|
| **PDMX** | 254,077 scores | MusicXML | Public domain multi-track; genre/rating metadata |
| **Lakh MIDI** | 178,561 files | MIDI | Multi-track; 45k matched to Million Song Dataset |
| **HookTheory** | 26,175 songs | Aligned annotation | Melody + harmony for YouTube recordings |
| **POP909** | 909 songs | MIDI | Pop piano; tempo/beat/chord/key annotations |
| **Pop1k7** | 1,747 songs | MIDI | 4-min pop piano with transcribed MIDI |
| **MAESTRO** | 1,184 performances | MIDI + audio | Classical piano; aligned note labels |
| **EMOPIA** | 1,078 clips | MIDI | Pop piano; emotion labels |
| **MetaMIDI** | 436,631 files | MIDI | Multi-track; artist/genre metadata |
| **SymphonyNet** | 46,359 files | MIDI | Multi-track multi-instrument symphonies |
| **GiantMIDI-Piano** | 10,855 pieces | MIDI | Classical piano; transcribed from recordings |
| **JSB Chorales** | 382 chorales | Python lists | Bach 4-part harmonizations |
| **Nottingham** | ~1,000 tunes | ABC / MIDI | British & American folk tunes |

### Key Dataset Links
- MAESTRO: https://magenta.tensorflow.org/datasets/maestro
- Lakh MIDI: https://colinraffel.com/projects/lmd/
- JSB Chorales: https://github.com/czhuang/JSB-Chorales-dataset
- Nottingham: https://abc.sourceforge.net/NMD/

---

## Key Takeaways

### Conceptual
1. Symbolic music generation = next-token prediction (text paradigm) **or** image generation (2D paradigm)
2. Both paradigms are viable; many top results use text-based approaches
3. Tokenization is still an **open problem** — many competing schemes (ABC, MIDI-like, REMI, Compound Word)
4. Music tokenization is harder than text: simultaneous events, timing info, metadata

### Model Evolution
```
Markov Chains → RNN/LSTM → Transformer
(limited context)  (medium)    (long-term structure, SOTA)
```

### Evaluation
- **Perplexity:** measures how well model predicts held-out sequences (lower = better)
- **Fitness functions:** pitch variety, key centeredness, contour stability, note density
- **Musical quality:** subjective; perplexity doesn't fully capture it

### Assignment 2 Pointers
- **Workbook3.ipynb:** provided skeleton with Markov chains + simple RNN
- **MidiTok library:** supports REMI, Compound Word, MMT, and more
- **Good starting datasets for unconditioned generation:** Nottingham (small, ABC), JSB Chorales, POP909, MAESTRO
- Transformers (decoder-only GPT-style) are current SOTA for symbolic unconditioned generation
