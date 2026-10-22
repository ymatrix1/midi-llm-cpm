# PRODUCT REQUIREMENTS DOCUMENT (PRD) & MASTER SYSTEM SPECIFICATION
## Project: SOTA Symbolic Multitrack Text-to-MIDI & Stem-Infilling Engine (`midi-llm-cpm`)

**Version:** 3.1 (SOTA Open-Source Specification — Progressive `2k -> 4k` Context, REMI-CoT, LLM-CFG & GRPO/DPO)  
**Target Architecture:** Causal Transformer LLM (`MiniCPM-1B` / `Qwen2.5-1.5B`) + Bar-Interleaved REMI-CoT Symbolic Grammar + LLM-CFG + GRPO/DPO  
**Reference Baseline to Beat:** [`slseanwu/MIDI-LLM_Llama-3.2-1B`](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B) ([GitHub: `slSeanWU/MIDI-LLM`](https://github.com/slSeanWU/MIDI-LLM), ISMIR 2026 — Baseline: `FAD = 0.173`, `CLAP = 22.1`)  
**Hardware Envelope:** Local Tier: 1x NVIDIA RTX 4070 Super (12GB VRAM) | Cloud Scale Tier: 8x NVIDIA H100 (80GB)

---

## 1. EXECUTIVE SUMMARY & PRODUCT VISION
Design, architect, and implement a state-of-the-art, commercial-grade **Symbolic Multitrack Text-to-MIDI Generation and Accompaniment Infilling Engine** engineered to outperform `MIDI-LLM_Llama-3.2-1B` across both objective audio-text benchmarks (**FAD $\downarrow$**, **CLAP $\uparrow$**) and real-world DAW production standards.

The system ingests freeform natural language producer prompts (e.g., *"118 BPM neo-soul groove in F minor with lush Rhodes 9th voicings, syncopated slap bass, and swung boom-bap drums"*)—optionally conditioned on existing MIDI bars, chord progressions, or instrument stems—and autoregressively generates human-expressive, harmonically locked, multitrack **Standard MIDI Files (SMF Type 1, `.mid`)** ready for drag-and-drop import into Ableton Live, Logic Pro, FL Studio, or Reaper.

### 1.1 Core SOTA Differentiators Over `MIDI-LLM_Llama-3.2-1B`
| Dimension | `MIDI-LLM_Llama-3.2-1B` (Baseline) | `midi-llm-cpm` v3.1 (Our Architecture) | Why Ours Wins |
| :--- | :--- | :--- | :--- |
| **Tokenization & Vocab Overhead** | Stanford AMT (`+55,030` tokens; ~340 MB embedding weight overhead) | **Bar-Interleaved REMI-CoT (`+576` tokens, $9 \times 64$ Tensor-Core aligned)** + optional AMT wrapper | **99% smaller vocab expansion**; fits 12GB VRAM easily while retaining full AMT compatibility mode |
| **Velocity & Pedal Expressiveness** | **None** (flat per-instrument velocity, no `CC64` sustain pedal) | **32 Dynamic Velocity Bins + `[PEDAL_ON]`/`[PEDAL_OFF]` (`CC64`) + `NOMML` `[EXPR_HUMAN]` tags** | Drastically more natural FluidSynth piano/guitar/drum renders ($\rightarrow$ **lower FAD**) |
| **Harmonic & Structural Planning** | Unstructured 10ms arrival-time stream; chords only in text prompt | **Inline Bar-Level "Musical Chain-of-Thought" (`[BAR_k] [SEC_*] [CHORD_ROOT_*] [CHORD_QUAL_*]`)** | Model predicts each bar's chord *before* its notes; all tracks attend to a local chord token $\le 30$ tokens away |
| **Prompt Adherence Mechanism** | Standard causal sampling (`temp=1.0, top_p=0.98`); suffers prompt dilution | **10% Prompt Dropout + Batched Logit-Level Classifier-Free Guidance (LLM-CFG, $\gamma \in [1.5, 2.5]$)** | Prevents prompt forgetting over long sequences ($\rightarrow$ **+2 to +4 higher CLAP score**) |
| **Key & Harmony Generalization** | Trained on raw key distribution (~45% C Maj / A Min bias) | **12x Joint Text + Symbolic Pitch-Transposition Augmentation ($\Delta p \in [-5, +6]$ semitones)** | Uniform harmonic accuracy across all 24 keys, including rare keys (`F# min`, `Db maj`) |
| **Post-Training Alignment** | SFT only (relies on post-hoc rejection of burst failures) | **Deterministic Grammar FSM + Multi-Critic DPO / GRPO (Harmony + Groove + Prompt-Instrument + CLAP)** | **0.0% syntax failures** + explicit optimization for voice-leading, groove lock, and prompt faithfulness |
| **Context Length & Song Horizon** | Hard-capped at `2,048` tokens (~13–14 bars; cuts off before Chorus) | **Progressive `2,048` (80%) $\rightarrow$ `4,096` (20%) Context Curriculum + Sliding-Window Outpainting** | Natively models **28–32 bar full songs** (`Intro -> Verse -> Chorus -> Outro`) & **16-bar stem infilling** in one attention span; scales to 64+ bars via sliding window |

### 1.2 Target Use Cases
1. **Unconditional Text-to-Multitrack Generation:** Generate 4 to 64+ bars of multi-instrument arrangement from text prompts and structured metadata tags (`28–32` bars natively within a single `4,096`-token context window).
2. **Conditional Stem Infilling (Anticipatory / Accompaniment Generation):** Given an existing set of tracks (e.g., 16 bars of Drums + Chords) and a prompt, generate a missing 16-bar target stem (e.g., `[TRACK_BASS]` or `[TRACK_MELODY]`) locked to the existing groove and bar-level chords inside the `4,096`-token context window.
3. **Bar Continuation & Multi-Section Outpainting:** Given an initial 4-bar seed loop, generate a 32-to-64 bar arrangement transitioning across `[SEC_INTRO]`, `[SEC_VERSE]`, `[SEC_CHORUS]`, `[SEC_BRIDGE]`, and `[SEC_OUTRO]`.

### 1.3 Success Metrics & Acceptance Criteria (SLOs)
| Metric Category | Metric | Target Threshold |
| :--- | :--- | :--- |
| **Deterministic Grammar** | Invalid Token Transition / Hanging Note Rate | **0.0%** (enforced via FSM `LogitsProcessor` + `allowed_token_ids`) |
| **Round-Trip Fidelity** | Tokenizer Encode $\rightarrow$ Decode Timing Drift | **$\le$ 1 grid tick** ($\le 1/96\text{th}$ note); **0 pitch drift** |
| **Harmonic Coherence** | Scale Consistency / In-Key & Bar-Chord Pitch Ratio | **$\ge$ 94%** alignment with conditioned `[KEY_*]` & `[CHORD_*]` |
| **Groove & Humanization** | Velocity Entropy ($H_{\text{vel}}$) | **$H_{\text{vel}} \ge 2.2$ bits** (eliminates flat mechanical velocities) |
| **SOTA Benchmark Target** | 896-ID `MIDI-LLM` Test Split (`FAD` $\downarrow$ / `CLAP` $\uparrow$) | **FAD $\le 0.145$** (vs. `0.173`) / **CLAP $\ge 24.5$** (vs. `22.1`) |
| **Hardware Envelope (`2k` / `4k`)** | Peak Training VRAM (`2,048` `bs=2` / `4,096` `bs=1` + Chunked CE) | **$\le$ 10.8 GB** on RTX 4070 Super (12GB) |
| **Inference Latency** | 8-Bar Multitrack Generation (4 tracks, ~1,200 tokens) | **$< 4.5$ seconds** on single RTX 4070 Super (`bfloat16` + KV Cache / vLLM) |
| **Code Quality & Formatting** | `uv run ruff check .` & `uv run ruff format --check .` | **0 warnings / 0 errors** (100% clean) |
| **Automated Test Coverage** | `uv run pytest --cov=src --cov-branch --cov-report=term-missing` | **$\ge 90\%$ line & branch coverage** across core modules |

---

## 2. REUSABLE ASSETS FROM `MIDI-LLM`, `MidiCaps` & `GigaMIDI v2.0.0`

An audit of [`slseanwu/MIDI-LLM_Llama-3.2-1B`](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B), [`amaai-lab/MidiCaps`](https://huggingface.co/datasets/amaai-lab/MidiCaps), and [`Metacreation/GigaMIDI`](https://huggingface.co/datasets/Metacreation/GigaMIDI) provides our data and evaluation backbone:

1. **Self-Contained SFT Dataset (`amaai-lab/MidiCaps`):**
   * Contains both `midicaps.tar.gz` (all 168,385 LakhMIDI `.mid` files) AND `train.json` with **19 pre-computed columns**:
     * `location`, `caption`, `genre`, `genre_prob`, `mood`, `mood_prob`, `key`, `time_signature`, `tempo`, `tempo_word`, `duration`, `duration_word`, `chord_summary`, `chord_summary_occurence`, `instrument_summary`, `instrument_numbers_sorted`, `all_chords`, `all_chords_timestamps`, and `test_set`.
   * **Crucial SOTA Leverage:** We map `all_chords` + `all_chords_timestamps` directly onto bar boundaries to supervise inline `[CHORD_ROOT_*] [CHORD_QUAL_*]` **Musical Chain-of-Thought** tokens without running slow external chord-detection passes.
2. **Extended GigaMIDI v2 (`Metacreation/GigaMIDI`, `config_name="v2.0.0"`):**
   * **Mandatory HF Config Flag:** On Hugging Face, `Metacreation/GigaMIDI` sets `v1.0.0` as `default: true`. `downloader.py` **MUST explicitly pass `name="v2.0.0"`** (`load_dataset("Metacreation/GigaMIDI", "v2.0.0", streaming=True)`) using local Hugging Face CLI authentication.
   * **Zero-Disk In-Memory Parsing & 28 Pre-Computed Columns:** Each Parquet record in `v2.0.0` embeds raw MIDI bytes in `row["music"]` (loaded directly via `symusic.Score.from_midi(row["music"])`), file hash `row["md5"]`, plus:
     * **Expressive Performance & Loop Metadata (`NOMML`):** `NOMML` (Note Onset Median Metric Level per track), `MIDI_program_number__expressive_`, `instrument_group__expressive_`, `start_tick__expressive_`, `end_tick__expressive_`, `duration_beats__expressive_`, `note_density__expressive_`, `loopability__expressive_` (2.3M expressive loops + 9.2M non-expressive loops).
     * **Ensemble Category Filter:** `instrument_category__drums-only__0__all-instruments-with-drums__1_no-drums__2` (`1` = full multitrack with drums).
     * **Curated Style & Metadata Ground Truth:** `music_styles_curated`, `music_style_scraped`, `music_style_audio_text_Discogs`, `music_style_audio_text_Lastfm`, `music_style_audio_text_Tagtraum`, `tempo`, `TPQN`, `total_notes`, `avg_velocity`, `min_velocity`, `max_velocity`.
3. **Official 896-Track Evaluation Holdout (`assets/evaluation_set_lakh_ids.txt`):**
   * Committed in `assets/evaluation_set_lakh_ids.txt` (alongside `assets/example_prompts.txt`). Strictly excluded from all CPT (`GigaMIDI v2.0.0`) and SFT (`MidiCaps`) training splits to guarantee zero data leakage and apples-to-apples FAD/CLAP benchmarking.

---

## 3. REPOSITORY & DIRECTORY ARCHITECTURE

```text
midi-llm-cpm/
├── PRD.md                           # This Product Requirements Document & Specification (v3.1)
├── assets/
│   ├── evaluation_set_lakh_ids.txt  # 896 held-out LakhMIDI MD5 IDs from slSeanWU/MIDI-LLM
│   └── example_prompts.txt          # Canonical evaluation prompts from slSeanWU/MIDI-LLM
├── configs/
│   ├── tokenizer_config.yaml        # Vocab bounds (REMI-CoT 576 vs AMT 55028), 96-pos grid, 64 GM programs
│   ├── train_local_4070.yaml        # 12GB VRAM config (Progressive 2048->4096, Chunked CE, QLoRA/DoRA)
│   ├── train_cloud_h100.yaml        # Full CPT + SFT ZeRO-2 config (MusicPile + GigaMIDI v2 + MidiCaps)
│   ├── align_dpo_grpo.yaml          # Multi-Critic DPO & GRPO alignment hyperparameters (seq_len=4096)
│   └── deepspeed_zero2.json         # Multi-GPU DeepSpeed configuration
├── src/
│   ├── __init__.py
│   ├── tokenization/
│   │   ├── vocab.py                 # 576-token Tensor-Core-aligned REMI-CoT vocab + AMT 55028 constants
│   │   ├── midi_tokenizer.py        # Bidirectional Bar-Interleaved REMI-CoT & AMT wrapper
│   │   └── grammar_fsm.py           # Finite State Machine & HuggingFace LogitsProcessor + CFG combiner
│   ├── data/
│   │   ├── downloader.py            # HF streaming loader for MidiCaps, GigaMIDI (v2.0.0), and MusicPile
│   │   ├── cleaner.py               # Fast C++/symusic heuristic filters, 896-ID eval exclusion & MinHash dedup
│   │   ├── mir_extractor.py         # Vectorized Krumhansl-Schmuckler key, bar-chord & section detector
│   │   ├── transposition.py         # 12x Joint Text-Regex + Symbolic MIDI Pitch-Transposition Augmenter
│   │   ├── llm_captioner.py         # Converts MidiCaps 19-col & GigaMIDI v2 28-col into diverse prompts
│   │   └── dataset.py               # PyTorch Dataset, 2048->4096 curriculum sampler, 10% CFG dropout & infilling
│   ├── models/
│   │   ├── model_loader.py          # MiniCPM-1B / Qwen2.5-1.5B / Llama-3.2-1B loader, vocab init & Chunked CE
│   │   └── music_critics.py         # Multi-critic reward suite (Harmony, Scale, Groove, Prompt-Inst, CLAP)
│   ├── training/
│   │   ├── train_cpt.py             # Stage 1 Continued Pretraining (MusicPile text + GigaMIDI v2.0.0)
│   │   ├── train_sft.py             # Stage 2 Progressive SFT (Embed Warmup -> 2048 SFT -> 4096 Long-Horizon SFT)
│   │   └── train_dpo.py             # Stage 3 Offline DPO & Online GRPO trainer (TRL DPOTrainer / GRPOTrainer)
│   ├── inference/
│   │   ├── generator.py             # HF & vLLM generator with LLM-CFG, FSM grammar & sliding-window outpaint
│   │   └── exporter.py              # DAW-ready SMF Type 1 writer (64 GM Programs, CC64 Pedal, Ch 10 Drums)
│   └── utils/
│       ├── audio_synth.py           # FluidSynth + librosa silence trim + pyloudnorm (-18 LUFS) + MP3
│       ├── metrics.py               # Objective MIR metrics + excessive simultaneous note validator
│       └── benchmark_midillm.py     # Automated 896-track FAD & LAION-CLAP evaluation harness vs. MIDI-LLM
├── scripts/
│   ├── cli.py                       # Unified cross-platform CLI (prepare, cpt, sft, align, generate, benchmark)
│   ├── train_local_4070.ps1         # Windows PowerShell launcher for RTX 4070 Super
│   └── train_local_4070.sh          # Linux/WSL2 bash launcher
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures (synthetic multitrack MIDIs, mock tokenizers)
│   ├── test_vocab.py                # Verifies 576-token (9x64) alignment, bijective IDs, AMT constants
│   ├── test_tokenizer_roundtrip.py  # Zero-drift roundtrip, CC64 pedal, inline chord CoT & triplet tests
│   ├── test_grammar_fsm.py          # Verifies FSM LogitsProcessor & LLM-CFG logit combination
│   ├── test_transposition.py        # Verifies joint text-prompt regex + symbolic pitch/chord transposition
│   ├── test_cleaner_and_mir.py      # Tests anti-metronome filter, 896-ID holdout, Krumhansl key detector
│   ├── test_infilling_mask.py       # Verifies bar-interleaved infilling, 2k/4k packing, 10% CFG dropout & -100 mask
│   ├── test_music_critics.py        # Tests Harmony, Scale, Groove, and Prompt-Instrument faithfulness critics
│   └── test_midi_export.py          # Verifies SMF Type 1 validity, CC64 sustain, Ch 10 drums, zero hang
├── .python-version                  # Pinned Python 3.11 for uv
├── pyproject.toml                   # Unified uv, ruff, pytest, coverage, and build configuration
├── uv.lock                          # Deterministic lockfile managed by uv
└── README.md
```

---

## 4. DETAILED COMPONENT SPECIFICATIONS

### Component A: Bar-Interleaved REMI-CoT Vocabulary, Tokenizer & Grammar FSM (`src/tokenization/`)

#### A.1 Bar-Interleaved REMI-CoT Architecture (576-Token Mode — $9 \times 64$ Tensor-Core Aligned)
1. **Triplet & Swing Resolution (96 Positions / 4/4 Bar):**
   * Standardize on **24 ticks per quarter note** (`96` positions per 4/4 bar: `[POS_0]` to `[POS_95]`), cleanly dividing both 16th/32nd duplets AND 8th/16th triplets while capturing human micro-timing swing.
2. **Bar-Interleaved Serialization with Inline "Musical Chain-of-Thought" (CoT):**
   * At the start of every `[BAR_k]`, emit optional structural section markers (`[SEC_*]`), meter/tempo changes, and the bar's **Inline Chord Token Pair (`[CHORD_ROOT_*] [CHORD_QUAL_*]`)** before emitting active tracks:
   ```text
   [BAR_0] [SEC_INTRO] [TS_4_4] [BPM_120] [CHORD_ROOT_F] [CHORD_QUAL_MIN7]
     [TRACK_DRUMS]  [INST_DRUMS] [POS_0] [PITCH_36] [DUR_6] [VEL_26] ...
     [TRACK_BASS]   [INST_33]    [POS_0] [PITCH_41] [DUR_24] [VEL_24] ...
     [TRACK_CHORDS] [INST_4]     [POS_0] [PEDAL_ON] [PITCH_60] [DUR_48] [VEL_18] [PITCH_65] [DUR_48] [VEL_18] ...
   [BAR_1] [CHORD_ROOT_BB] [CHORD_QUAL_DOM7]
     [TRACK_DRUMS] ...
   ```
   * **Why Inline Chord CoT Works:** During generation, the LLM predicts `[CHORD_ROOT_*] [CHORD_QUAL_*]` at the start of each bar. Every subsequent track in that bar (`[TRACK_BASS]`, `[TRACK_CHORDS]`, `[TRACK_MELODY]`) attends directly to that explicit chord decision within a $\le 30$-token local window, preventing cross-track harmonic drift.
3. **Exact 576-Token Allocation ($9 \times 64 = 576$, Exact 64x Tensor-Core Alignment):**
   * **Structure, Section & Pedal Control (24 tokens):** `[MUSIC_BOS]`, `[MUSIC_EOS]`, `[MUSIC_PAD]`, `[PROMPT_START]`, `[PROMPT_END]`, `[INFILL_START]`, `[INFILL_END]`, `[MASK_TRACK]`, `[MASK_BAR]`, `[BAR_START]`, `[BAR_END]`, `[PEDAL_ON]`, `[PEDAL_OFF]`, `[EXPR_HUMAN]`, `[EXPR_QUANTIZED]`, `[SEC_INTRO]`, `[SEC_VERSE]`, `[SEC_CHORUS]`, `[SEC_BRIDGE]`, `[SEC_OUTRO]`, + 4 reserved.
   * **Bar Index (64 tokens):** `[BAR_0]` to `[BAR_63]`.
   * **Metric Position (96 tokens):** `[POS_0]` to `[POS_95]`.
   * **Time Signatures (8 tokens):** `[TS_4_4]`, `[TS_3_4]`, `[TS_6_8]`, `[TS_12_8]`, `[TS_2_4]`, `[TS_5_4]`, `[TS_7_8]`, `[TS_OTHER]`.
   * **Quantized Tempo Bins (32 tokens):** `[BPM_40]` to `[BPM_200]` in 5 BPM increments.
   * **Inline Bar-Level Chord CoT Tokens (24 tokens):** 12 pitch-class roots (`[CHORD_ROOT_C]` to `[CHORD_ROOT_B]`), 11 chord qualities (`[CHORD_QUAL_MAJ]`, `MIN`, `DOM7`, `MAJ7`, `MIN7`, `DIM`, `HDIM7`, `AUG`, `SUS2`, `SUS4`, `POW5`), and `[CHORD_NONE]`.
   * **Semantic Track Roles (10 tokens):** `[TRACK_DRUMS]`, `[TRACK_BASS]`, `[TRACK_CHORDS]`, `[TRACK_PAD]`, `[TRACK_MELODY]`, `[TRACK_LEAD]`, `[TRACK_ARP]`, `[TRACK_GUITAR]`, `[TRACK_STRINGS]`, `[TRACK_FX]`.
   * **High-Resolution General MIDI Instruments (66 tokens):** 64 distinct GM melodic instrument programs (`[INST_0]` to `[INST_63]`, preserving exact timbral distinctions such as Acoustic Grand vs. Rhodes vs. Church Organ vs. French Horn vs. Slap Bass for superior FluidSynth FAD/CLAP scores), `[INST_DRUMS]`, and `[INST_UNKNOWN]`.
   * **Pitch (128 tokens):** `[PITCH_0]` to `[PITCH_127]`.
   * **Duration (64 tokens):** `[DUR_1]` to `[DUR_64]` (1-tick resolution from 1 to 24 ticks; stepped logarithmic bins up to 4 bars / 384 ticks).
   * **Velocity (32 tokens):** `[VEL_0]` to `[VEL_31]` (MIDI velocities `4..127` in steps of 4).
   * **Global Harmonic & Groove Conditioning (28 tokens):** 24 keys (`[KEY_C_MAJ]` .. `[KEY_B_MIN]`), `[DENSITY_LOW]`, `[DENSITY_MED]`, `[DENSITY_HIGH]`, `[SWING_ON]`.
   * **Total Added Musical Vocabulary:** $24 + 64 + 96 + 8 + 32 + 24 + 10 + 66 + 128 + 64 + 32 + 28 = \mathbf{576\text{ tokens}}$ ($576 \pmod{64} = 0$).

#### A.2 Optional AMT Compatibility Mode (`AMTCompatTokenizer`)
* Directly wraps the Stanford `anticipation` library (`AMT_GPT2_BOS_ID = 55026`, arrival-time triplet grammar `[time, dur, note]`) shifted by `base_vocab_size`, allowing our pipeline to load, evaluate, or run DPO/GRPO directly on `slseanwu/MIDI-LLM_Llama-3.2-1B`.

#### A.3 Deterministic Grammar FSM & Classifier-Free Guidance Processor (`grammar_fsm.py`)
1. **`MusicGrammarLogitsProcessor`:** Enforces 100% valid state transitions at every decoding step:
   * `[BAR_k]` $\rightarrow$ `[SEC_*] | [TS_*] | [BPM_*] | [CHORD_ROOT_*] | [CHORD_NONE] | [TRACK_*] | [MUSIC_EOS]`
   * `[CHORD_ROOT_*]` $\rightarrow$ `[CHORD_QUAL_*]`
   * `[TRACK_*]` $\rightarrow$ `[INST_*]`
   * `[INST_*]` $\rightarrow$ `[POS_*]`
   * `[POS_p]` $\rightarrow$ `[PEDAL_ON] | [PEDAL_OFF] | [PITCH_*]`
   * `[PITCH_n]` $\rightarrow$ `[DUR_*]` $\rightarrow$ `[VEL_*]`
   * `[VEL_v]` $\rightarrow$ `[PITCH_{m>n}]` (ascending chord notes at same `POS_p`) $|$ `[POS_{q>p}]` (monotonic forward time within bar) $|$ `[TRACK_*]` (next unvisited track in bar) $|$ `[BAR_{k+1}]` $|$ `[INFILL_END]` $|$ `[MUSIC_EOS]`.
2. **`ClassifierFreeGuidanceLogitsProcessor` (`LLM-CFG`):**
   * Maintains paired conditional (full prompt) and unconditional (`[PROMPT_START] [PROMPT_END]`) batch branches during generation, combining logits prior to FSM masking:
     $$\ell_{\text{CFG}} = \ell_{\text{uncond}} + \gamma \cdot (\ell_{\text{cond}} - \ell_{\text{uncond}}), \quad \gamma \in [1.5, 2.5]$$

---

### Component B: Data Engineering, 12x Transposition & `MidiCaps` / `GigaMIDI v2.0.0` Pipeline (`src/data/`)

#### B.1 Dataset Sources & Zero-Leakage Holdout (`downloader.py`, `cleaner.py`)
1. **Strict 896-ID Evaluation Holdout:** Load `assets/evaluation_set_lakh_ids.txt` and block any matching MD5 hash from entering CPT or SFT datasets.
2. **SFT Dataset (`amaai-lab/MidiCaps`):**
   * Stream/extract `midicaps.tar.gz` paired with `train.json` (168,385 files).
   * Map `all_chords` and `all_chords_timestamps` onto `[BAR_k]` boundaries to populate `[CHORD_ROOT_*] [CHORD_QUAL_*]` tokens automatically.
3. **CPT Datasets (`m-a-p/MusicPile` + `Metacreation/GigaMIDI` `v2.0.0`):**
   * **`m-a-p/MusicPile` (~1.7B tokens):** Music theory, ABC notation, and musicology text.
   * **`Metacreation/GigaMIDI` (`name="v2.0.0"`, ~1.4B tokens):**
     * Explicitly load `load_dataset("Metacreation/GigaMIDI", "v2.0.0", streaming=True)`.
     * Exclude all `row["md5"]` hashes present in `MidiCaps` or `evaluation_set_lakh_ids.txt`.
     * Parse raw MIDI bytes in-memory via `symusic.Score.from_midi(row["music"])`.
     * Prioritize multitrack files (`instrument_category__drums-only__0__all-instruments-with-drums__1_no-drums__2 == 1`) and annotate `[EXPR_HUMAN]` vs. `[EXPR_QUANTIZED]` using `NOMML` and `loopability__expressive_`.
     * Synthesize weak natural-language captions for CPT from `music_styles_curated`, `music_style_audio_text_Discogs`, `tempo`, and `loop_instrument_type`.

#### B.2 Fast C++ MIDI Sanitization (`cleaner.py`)
1. Reject files with $< 32$ notes, duration $< 8\text{s}$ or $> 600\text{s}$, or corrupted delta-time headers.
2. **Anti-Metronome Filter:** Drop tracks where $> 90\%$ of notes share a single pitch AND single IOI with zero velocity variance.
3. **Simultaneous Note Burst Filter:** Reject corrupted MIDI segments with $> 64$ simultaneous note-ons (`has_excessive_notes_at_any_time`).
4. **Melodic N-Gram MinHash Deduplication:** Deduplicate near-identical arrangements (Jaccard threshold $\ge 0.85$).

#### B.3 12x Joint Text + Symbolic Pitch-Transposition Augmentation (`transposition.py`)
To eliminate the C-Major / A-Minor dataset bias and achieve SOTA key adherence across all 24 keys:
1. During SFT data loading, with probability $p_{\text{transpose}} = 0.75$, sample a semitone shift $\Delta p \in [-5, +6] \setminus \{0\}$.
2. Shift all `[PITCH_n]` tokens on non-drum tracks (`track != [TRACK_DRUMS]`) by $\Delta p$, clamping/octave-wrapping within playable instrument register bounds.
3. Circularly shift `[KEY_*]` and all inline `[CHORD_ROOT_*]` tokens by $\Delta p \pmod{12}$.
4. **Synchronized Prompt Rewriting:** Apply a music-aware regex parser to the text caption and metadata tags that transposes every mentioned key signature and chord symbol in the natural language prompt by $\Delta p$ semitones (using enharmonic spelling matched to the target key signature, e.g., `Eb Major` uses flats `Ab, Bb`, while `E Major` uses sharps `F#m, G#m`).

#### B.4 Prompt Formatting, 10% CFG Dropout & Stem Infilling (`llm_captioner.py`, `dataset.py`)
Use the `MIDI-LLM` canonical system prompt prefix:
```text
You are a world-class composer. Please compose some music according to the following description: 
```
During SFT (`seq_len = 2048` in Phase A $\rightarrow$ `seq_len = 4096` in Phase B):
1. **10% Unconditional Prompt Dropout (for LLM-CFG):** In 10% of samples, replace the user description with an empty conditioning span so the model learns unconditional priors for inference-time Classifier-Free Guidance.
2. **Prompt Style Mixing (on remaining 90%):**
   * **50% Rich `MidiCaps` Caption:** Original `caption` field (with synchronized transposition if $\Delta p \neq 0$).
   * **30% Structured Producer Prompt:** Synthesized from `MidiCaps` / `GigaMIDI v2.0.0` features (`BPM`, `Key`, `Time Signature`, `Instruments`, `Chord Progression`, `Expressiveness`).
   * **20% Short Creative Prompt:** Concise vibe/genre/scene prompt.
3. **Causal Task Mixing (70% Text-to-MIDI / 30% Stem & Bar Infilling):**
   * **70% Standard Text-to-Multitrack:** Prompt $\rightarrow$ `[MUSIC_BOS]` $\rightarrow$ Bar-Interleaved REMI-CoT Tokens $\rightarrow$ `[MUSIC_EOS]`.
   * **30% Anticipatory Stem Infilling:** Mask out a target stem (e.g., `[TRACK_BASS]` or `[TRACK_MELODY]`) from the bar-interleaved context and train the model to generate the missing stem between `[INFILL_START]` and `[INFILL_END]` locked to the unmasked tracks and inline bar chords.

---

### Component C: Model Adaptation, Chunked CE & Progressive `2,048 -> 4,096` Training on 12GB VRAM (`src/models/`, `src/training/`)

#### C.1 Foundation Models, Variance-Preserving Init & Chunked Cross-Entropy (`model_loader.py`)
1. Support `openbmb/MiniCPM-1B-sft-bf16`, `Qwen/Qwen2.5-1.5B`, and `meta-llama/Llama-3.2-1B` in `torch.bfloat16`.
2. Add the 576 musical tokens and call `model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)`.
3. Initialize new embedding and `lm_head` rows from $\mathcal{N}(\mu_{\text{old}}, \Sigma_{\text{diag}} \cdot 0.5)$ to prevent initial logit spikes.
4. **Chunked / Fused Linear Cross-Entropy (Eliminating the `4,096` `lm_head` VRAM Spike):**
   * At `seq_len = 4,096`, materializing a dense FP32 `[batch, 4096, vocab_size]` logits tensor + backward gradients consumes **~5.0 GB VRAM per sample** on Qwen2.5-1.5B (`vocab ≈ 152k`), which causes OOM on 12GB GPUs.
   * Implement **Chunked Cross-Entropy** (computing `lm_head(hidden_states[:, i:i+512])` + `F.cross_entropy` in 512-token slices, or using `liger-kernel`'s `FusedLinearCrossEntropyLoss`) so the full `[4096, vocab_size]` logits tensor is never materialized in VRAM. This cuts peak `lm_head` memory from **~5.0 GB to ~0.3 GB**, enabling `seq_len = 4,096` in **~8.8 GB VRAM** on an RTX 4070 Super.

#### C.2 Progressive Context Curriculum for 12GB VRAM (`train_sft.py`)
At ~140 tokens per 4-track bar, `2,048` tokens fits only ~13–14 bars (cutting off before a Chorus/B-section and limiting stem infilling to 7–8 bars), whereas `4,096` tokens fits **28–32 bars** (a full `Intro -> Verse -> Chorus -> Bridge -> Outro` arrangement or a full **16-bar multitrack context + 16-bar infilled stem**). To achieve `4,096`-token mastery without sacrificing throughput:
* **Phase 1 (Embedding & Head Alignment Warmup — 10% of steps, `seq_len = 2,048`):**
  * Freeze transformer layers; train only the newly added 576 rows of `embed_tokens` and `lm_head` (via a backward gradient hook zeroing rows $< V_{\text{orig}}$) with `lr = 5e-4`, `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 16`.
* **Phase 2A (High-Throughput Core SFT — 70% of steps, `seq_len = 2,048`):**
  * LoRA/DoRA (`r = 32, lora_alpha = 64`, targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) + row-masked trainable new embeddings.
  * Optimizer: `paged_adamw_8bit` (`bitsandbytes`), `lr = 2e-4`, `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 16` (effective batch = 32 sequences = 65,536 tokens/step).
* **Phase 2B (Long-Horizon Song Structure & 16-Bar Stem Infilling SFT — Final 20% of steps, `seq_len = 4,096`):**
  * Switch sequence packer to `max_seq_length = 4,096` (`per_device_train_batch_size = 1`, `gradient_accumulation_steps = 32`, `lr = 5e-5` cosine decay) with Chunked Cross-Entropy enabled, focusing on multi-section (`[SEC_VERSE] -> [SEC_CHORUS]`) arrangements and 16-bar stem infilling.
* **Loss Masking:** Compute Cross-Entropy loss strictly on musical tokens (`labels = -100` over the system prompt, text description, and unmasked infilling context bars).

---

### Component D: Multi-Critic Preference Alignment — DPO & Online GRPO (`src/models/music_critics.py`, `src/training/train_dpo.py`)

With syntax 100% enforced by `MusicGrammarLogitsProcessor`, Stage 3 aligns the model (`max_seq_length = 4,096`) using **Offline DPO** and **Online Group Relative Policy Optimization (GRPO, $G=4$ candidates per prompt)** driven by a 5-part composite reward:
$$R_{\text{total}}(x, y) = 0.25 R_{\text{harmony}} + 0.20 R_{\text{scale}} + 0.20 R_{\text{groove}} + 0.20 R_{\text{prompt\_inst}} + 0.15 R_{\text{dynamics}}$$

1. **Harmonic & Bar-Chord Consistency Critic ($R_{\text{harmony}}$):** Penalizes sustained minor-2nd / major-7th clashes between `[TRACK_BASS]` / `[TRACK_CHORDS]` / `[TRACK_MELODY]` and verifies pitched notes in `[BAR_k]` resolve against that bar's predicted `[CHORD_ROOT_*] [CHORD_QUAL_*]`.
2. **Scale Adherence Critic ($R_{\text{scale}}$):** Rewards in-key pitch adherence matching the prompted `[KEY_*]` token.
3. **Groove Lock & Polyphony Critic ($R_{\text{groove}}$):** Rewards Bass + Kick Drum (`PITCH_35`/`36`) onset synchronization ($\pm 2$ ticks) and penalizes polyphonic note bursts (`has_excessive_notes_at_any_time`).
4. **Prompt-Instrument & Tempo Faithfulness Critic ($R_{\text{prompt\_inst}}$):** Parses instruments and tempo mentioned in prompt $x$ and rewards exact recall in the generated `[INST_*]`, `[TRACK_*]`, and `[BPM_*]` tokens (directly maximizing **CLAP** alignment).
5. **Expressive Dynamics & Anti-Loop Critic ($R_{\text{dynamics}}$):** Rewards human velocity entropy ($H_{\text{vel}} \ge 2.2$ bits), natural sustain pedal usage (`[PEDAL_ON]`/`[PEDAL_OFF]`), and penalizes $> 2$ consecutive identical bars.

---

### Component E: Inference Backends, Sliding-Window Outpainting, DAW Exporter & Benchmark Suite (`src/inference/`, `src/utils/`)

1. **Dual Inference Engine with LLM-CFG & Sliding-Window Outpainting (`generator.py`):**
   * **HuggingFace Transformers & vLLM (`--fp8`) Backends:** Supports both single-prompt local generation and high-throughput batched generation up to `max_new_tokens = 4,090` natively.
   * **Classifier-Free Guidance (`--cfg_scale 1.8`):** Boosts prompt adherence and CLAP score out of the box.
   * **Sliding-Window Long-Song Outpainting (`--bars 64+`):** Generates arbitrarily long multi-section songs beyond 32 bars by retaining `[Prompt Prefix] + [Bar 0..1 Theme Anchor] + [Most Recent 12 Bars]` across `4,096`-token sliding generation windows.
2. **Studio-Grade SMF Type 1 Exporter (`exporter.py`):**
   * Outputs `mido.MidiFile(type=1, ticks_per_beat=480)` with Track 0 reserved for Tempo/Time-Signature/Section Marker meta-events, Channel 9 (DAW Channel 10) reserved for Drums, `CC64` sustain pedal events, 64 fine-grained GM `program_change` mappings, and automatic same-pitch overlap clipping (`note_off` inserted 1 tick prior to re-trigger).
3. **Normalized Audio Preview (`audio_synth.py`):**
   * Headless FluidSynth (`FluidR3_GM.sf2`) $\rightarrow$ `librosa.effects.trim(top_db=30)` $\rightarrow$ `pyloudnorm` loudness normalization (`-18.0` LUFS) $\rightarrow$ FFmpeg MP3 rendering.
4. **Official `MIDI-LLM` FAD & LAION-CLAP Benchmark Harness (`benchmark_midillm.py`):**
   * Evaluates any checkpoint on the 896 held-out Lakh IDs (`assets/evaluation_set_lakh_ids.txt`), synthesizes audio previews, and computes **Fréchet Audio Distance (FAD)** and **LAION-CLAP Audio-Text Cosine Similarity** with a comparison table against `slseanwu/MIDI-LLM_Llama-3.2-1B` (`FAD = 0.173`, `CLAP = 22.1`).

---

## 5. DEVELOPER TOOLCHAIN, LINTING & QUALITY GATES (`uv`, `ruff`, `pytest`, `coverage`)

### 5.1 Package & Environment Management (`uv`)
All dependency resolution, virtual environment management, and command execution MUST use **`uv`** (`pyproject.toml` + `uv.lock`, pinned to Python `3.11`):
* **Initialize & Sync:** `uv sync --all-extras --dev`
* **CLI Entrypoint (`[project.scripts]`):** `midi-llm = "src.scripts.cli:app"` (or `uv run midi-llm ...`)
* **Optional Dependency Groups:**
  * `core`: `torch`, `transformers`, `accelerate`, `peft`, `trl`, `datasets`, `mido`, `symusic`, `numpy`, `pyyaml`, `typer`, `rich`, `tqdm`
  * `quant`: `bitsandbytes`, `liger-kernel`, `vllm` (Linux/WSL2)
  * `audio`: `midi2audio`, `librosa`, `soundfile`, `pyloudnorm`
  * `eval`: `frechet_audio_distance`, `laion_clap`
  * `amt`: `anticipation` (Stanford AMT 55k compatibility)
  * `dev`: `pytest`, `pytest-cov`, `pytest-mock`, `ruff`, `mypy`

### 5.2 Linting & Formatting (`ruff`)
Enforce strict zero-warning linting and formatting via `ruff` in `pyproject.toml`:
* **Target Version:** `py311`, `line-length = 100`
* **Enabled Rule Sets:** `E`, `F`, `W` (pycodestyle/pyflakes), `I` (isort), `B` (flake8-bugbear), `UP` (pyupgrade), `SIM` (flake8-simplify), `N` (pep8-naming), `C4` (comprehensions), `PT` (pytest-style), `RUF` (Ruff-specific rules).
* **Gate Commands:**
  ```powershell
  uv run ruff check .
  uv run ruff format --check .
  ```

### 5.3 Automated Testing & Code Coverage (`pytest` + `pytest-cov`)
Configure `pytest` and `coverage` in `pyproject.toml` (`--cov=src --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=90` on core deterministic logic):
1. **`tests/test_vocab.py`:** Verifies exact 576-token ($9 \times 64$) Tensor-Core alignment (`len(vocab) % 64 == 0`), bijective token-to-ID mappings, inline Chord CoT tokens, 64 GM instruments, and AMT 55,028 offset constants.
2. **`tests/test_tokenizer_roundtrip.py`:** Generates synthetic 4-track MIDI files (Drums, Bass, Sustained Rhodes Chords with `CC64`, Triplet Lead), runs `encode -> decode`, and asserts exact pitch equality, $\le 1$ grid tick timing drift, `CC64` pedal preservation, and bar-interleaved CoT ordering.
3. **`tests/test_grammar_fsm.py`:** Runs 256 steps of random logits through `MusicGrammarLogitsProcessor` and `ClassifierFreeGuidanceLogitsProcessor` (in both REMI-CoT and AMT triplet modes) and asserts 100% syntactically valid sequences.
4. **`tests/test_transposition.py`:** Verifies that `transposition.py` shifts non-drum MIDI pitches, `[KEY_*]`, `[CHORD_ROOT_*]`, and natural language chord/key strings in the text prompt identically across $\Delta p \in [-5, +6]$ semitones without altering drum pitches.
5. **`tests/test_cleaner_and_mir.py`:** Tests the anti-metronome filter, `has_excessive_notes_at_any_time` burst detector, 896-ID `evaluation_set_lakh_ids.txt` holdout filter, and Krumhansl-Schmuckler key detector.
6. **`tests/test_infilling_mask.py`:** Verifies stem-infilling sequence construction across `2,048` and `4,096` context lengths, Chunked Cross-Entropy equivalence against standard `F.cross_entropy`, 10% CFG prompt dropout, and loss masking (`labels = -100` on prompt and conditioning context).
7. **`tests/test_music_critics.py`:** Verifies all 5 DPO/GRPO reward critics ($R_{\text{harmony}}, R_{\text{scale}}, R_{\text{groove}}, R_{\text{prompt\_inst}}, R_{\text{dynamics}}$) rank in-key, instrument-faithful grooves strictly higher than clashing or wrong-instrument sequences.
8. **`tests/test_midi_export.py`:** Verifies exported `.mid` files are valid SMF Type 1, assign `[TRACK_DRUMS]` to Channel 9 (DAW Ch 10), write `CC64` pedal events, clip overlapping same-pitch notes cleanly, and leave zero hanging notes.

```powershell
# Run full test suite with line & branch coverage report
uv run pytest --cov=src --cov-branch --cov-report=term-missing
```
