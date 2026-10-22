# PRODUCT REQUIREMENTS DOCUMENT (PRD) & MASTER SYSTEM SPECIFICATION
## Project: SOTA Symbolic Multitrack Text-to-MIDI & Stem-Infilling Engine (`midi-llm-cpm`)

**Version:** 2.2 (Production Specification — Incorporating `slseanwu/MIDI-LLM`, `MidiCaps`, & `GigaMIDI v2.0.0`)  
**Target Architecture:** Causal Transformer LLM (`MiniCPM-1B` / `Qwen2.5-1.5B`) + Symbolic Multitrack Grammar  
**Reference Baseline:** [`slseanwu/MIDI-LLM_Llama-3.2-1B`](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B) ([GitHub: `slSeanWU/MIDI-LLM`](https://github.com/slSeanWU/MIDI-LLM), ISMIR 2026)  
**Hardware Envelope:** Local Tier: 1x NVIDIA RTX 4070 Super (12GB VRAM) | Cloud Scale Tier: 8x NVIDIA H100 (80GB)

---

## 1. EXECUTIVE SUMMARY & PRODUCT VISION
Design, architect, and implement a state-of-the-art, commercial-grade **Symbolic Multitrack Text-to-MIDI Generation and Accompaniment Infilling Engine**.

The system accepts freeform natural language producer prompts (e.g., *"118 BPM neo-soul groove in F minor with lush Rhodes 9th voicings, syncopated slap bass, and swung boom-bap drums"*)—optionally conditioned on existing MIDI bars or tracks—and autoregressively generates human-expressive, harmonically coherent, multitrack **Standard MIDI Files (SMF Type 1, `.mid`)** ready for drag-and-drop import into Ableton Live, Logic Pro, FL Studio, or Reaper.

### 1.1 Target Use Cases
1. **Unconditional Text-to-Multitrack Generation:** Generate 4 to 64 bars of multi-instrument arrangement from text prompts and structured metadata tags.
2. **Conditional Stem Infilling (Anticipatory / Accompaniment Generation):** Given an existing set of tracks (e.g., Drums + Chords) and a prompt, generate a missing target stem (e.g., `[TRACK_BASS]` or `[TRACK_MELODY]`) locked to the existing groove and harmony.
3. **Bar Continuation / Outpainting:** Given an initial 4-bar seed loop, generate a 16-bar B-section or variation.

### 1.2 Success Metrics & Acceptance Criteria (SLOs)
| Metric Category | Metric | Target Threshold |
| :--- | :--- | :--- |
| **Deterministic Grammar** | Invalid Token Transition / Hanging Note Rate | **0.0%** (enforced via FSM `LogitsProcessor` + `allowed_token_ids`) |
| **Round-Trip Fidelity** | Tokenizer Encode $\rightarrow$ Decode Timing Drift | **$\le$ 1 grid tick** ($\le 1/96\text{th}$ note); **0 pitch drift** |
| **Harmonic Coherence** | Scale Consistency / In-Key Pitch Ratio | **$\ge$ 92%** alignment with conditioned `[KEY_*]` |
| **Groove & Humanization** | Velocity Entropy ($H_{\text{vel}}$) | **$H_{\text{vel}} \ge 2.1$ bits** (eliminates flat mechanical velocities) |
| **Benchmark Parity** | Evaluation on 896 MIDI-LLM Lakh IDs (`FAD` $\downarrow$ / `CLAP` $\uparrow$) | **FAD $\le 0.17$** / **CLAP $\ge 22.1$** (matching/beating MIDI-LLM 1.47B) |
| **Hardware Envelope** | Peak Training VRAM (Local Config, `seq_len=2048`) | **$\le$ 11.2 GB** on RTX 4070 Super (12GB) |
| **Inference Latency** | 8-Bar Multitrack Generation (4 tracks, ~1,200 tokens) | **$< 4.5$ seconds** on single RTX 4070 Super (`bfloat16` + KV Cache / vLLM) |
| **Code Quality & Formatting** | `uv run ruff check .` & `uv run ruff format --check .` | **0 warnings / 0 errors** (100% clean) |
| **Automated Test Coverage** | `uv run pytest --cov=src --cov-report=term-missing` | **$\ge 90\%$ line & branch coverage** across core modules |

---

## 2. LESSONS & REUSABLE ASSETS FROM `MIDI-LLM`, `MidiCaps` & `GigaMIDI v2.0.0`

An audit of [`slseanwu/MIDI-LLM_Llama-3.2-1B`](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B), [`amaai-lab/MidiCaps`](https://huggingface.co/datasets/amaai-lab/MidiCaps), and [`Metacreation/GigaMIDI`](https://huggingface.co/datasets/Metacreation/GigaMIDI) reveals critical architectural trade-offs and directly reusable assets:

### 2.1 What We Can Grab Directly
1. **Self-Contained SFT Dataset (`amaai-lab/MidiCaps`):**
   * We **do not** need to download raw LakhMIDI separately or re-run slow `music21` MIR extraction on those 168,385 files.
   * `amaai-lab/MidiCaps` provides both `midicaps.tar.gz` (all 168,385 LakhMIDI `.mid` files) AND `train.json` containing **19 pre-computed columns**:
     * `location`, `caption`, `genre`, `genre_prob`, `mood`, `mood_prob`, `key`, `time_signature`, `tempo`, `tempo_word`, `duration`, `duration_word`, `chord_summary`, `chord_summary_occurence`, `instrument_summary`, `instrument_numbers_sorted`, `all_chords`, `all_chords_timestamps`, and `test_set`.
2. **Extended GigaMIDI v2 (`Metacreation/GigaMIDI`, `config_name="v2.0.0"`):**
   * **Critical HF Config Caveat:** On Hugging Face, `Metacreation/GigaMIDI` sets `v1.0.0` as `default: true`. Our data pipeline (`downloader.py`) **MUST explicitly pass `name="v2.0.0"`** (`load_dataset("Metacreation/GigaMIDI", "v2.0.0", streaming=True)`) to load **Extended GigaMIDI v2.0.0** (2,136,218 unique MIDI files, 6.89M tracks, 2.3M expressive loops + 9.2M non-expressive loops).
   * **Zero-Disk In-Memory Parsing & 28 Pre-Computed Columns:** Each Parquet row in `v2.0.0` contains raw MIDI bytes in `row["music"]` (loaded directly in-memory via `symusic.Score.from_midi(row["music"])`), file hash `row["md5"]`, plus rich pre-computed columns:
     * **Expressive Performance & Loop Metadata (`NOMML`):** `NOMML` (Note Onset Median Metric Level per track), `MIDI_program_number__expressive_`, `instrument_group__expressive_`, `start_tick__expressive_`, `end_tick__expressive_`, `duration_beats__expressive_`, `note_density__expressive_`, `loopability__expressive_`, plus `loop_track_idx`, `loop_instrument_type`, `loop_start`, `loop_end`.
     * **Ensemble Category Filter:** `instrument_category__drums-only__0__all-instruments-with-drums__1_no-drums__2` (`0` = drums-only, `1` = all-instruments-with-drums, `2` = no-drums).
     * **Curated Genre/Style Ground Truth (Weak Captioning for CPT):** `music_styles_curated` (Dr. Cale Plut curated styles), `music_style_scraped`, `music_style_audio_text_Discogs`, `music_style_audio_text_Lastfm`, `music_style_audio_text_Tagtraum`, `tempo`, `TPQN`, `total_notes`, `avg_velocity`, `min_velocity`, `max_velocity`.
3. **Official 896-Track Evaluation Split (`assets/evaluation_set_lakh_ids.txt`):**
   * Grabbed directly from `slSeanWU/MIDI-LLM/blob/main/assets/evaluation_set_lakh_ids.txt` to ensure zero train/test contamination during CPT (`GigaMIDI v2.0.0` via `row["md5"]`) and SFT (`MidiCaps`), and enable direct apples-to-apples benchmarking against MIDI-LLM's published FAD (`0.173`) and CLAP (`22.1`) scores.
4. **System Prompt & Sequence Length Convention:**
   * System Prompt Prefix: `"You are a world-class composer. Please compose some music according to the following description: "` followed by `prompt + " "` and `[MUSIC_BOS]`.
   * Training sequence length `2048` (`max_new_tokens=2046` at inference) with `temperature=1.0, top_p=0.98`.
5. **Polyphonic Explosion Filter & Audio Preview Pipeline (`midi_llm/utils.py`):**
   * Port `has_excessive_notes_at_any_time(tokens, max_notes_per_time=64)` (using `torch.bincount` on onset times) as a post-generation sanity check.
   * Port `synthesize_midi_to_audio()` (FluidSynth `FluidR3_GM.sf2` $\rightarrow$ `librosa.effects.trim(top_db=30)` $\rightarrow$ `pyloudnorm` LUFS normalization to `-18.0` / `-14.0` LUFS $\rightarrow$ FFmpeg MP3 export).
6. **Dual-Mode Tokenization Support (Compact REMI+ vs. AMT 55k Compatibility):**
   * **Why MIDI-LLM used 55,028 tokens:** MIDI-LLM appends the full [Anticipatory Music Transformer (AMT)](https://github.com/jthickstun/anticipation) arrival-time triplet vocabulary (`55,028` tokens: `[time_offset, duration, (instrument, pitch)]` + `AMT_GPT2_BOS_ID=55026`) directly onto Llama-3.2-1B (`128,256` $\rightarrow$ `183,286` tokens).
   * **The 12GB VRAM Trade-off:** Adding 55,030 tokens adds $\sim 338\text{ MB}$ of parameters to `embed_tokens` + `lm_head` alone (and $> 2.7\text{ GB}$ of optimizer states + gradients in FP32/BF16), AND completely discards MIDI velocity (AMT uses flat per-instrument velocities) and bar/beat metric grids (using raw 10ms arrival-time bins up to 100 seconds).
   * **Our Solution:** Support **two pluggable tokenizer backends** in `src/tokenization/`:
     1. **`BarInterleavedREMITokenizer` (Default for 12GB 4070 Super & DAW Production):** Adds **512 tokens** (64-aligned), preserves **32 expressive velocity levels**, explicit **Bar/Beat/Triplet grid** (`96` pos/bar), and **chord conditioning**.
     2. **`AMTCompatTokenizer` (Optional Wrapper over `anticipation`):** Wraps `anticipation.convert.midi_to_events` / `events_to_midi` so our pipeline can also load, fine-tune, or run DPO directly on `slseanwu/MIDI-LLM_Llama-3.2-1B` checkpoints!

---

## 3. REPOSITORY & DIRECTORY ARCHITECTURE

```text
midi-llm-cpm/
├── PRD.md                           # This Product Requirements Document & Specification
├── assets/
│   ├── evaluation_set_lakh_ids.txt  # 896 held-out LakhMIDI MD5 IDs from slSeanWU/MIDI-LLM
│   └── example_prompts.txt          # Canonical evaluation prompts from slSeanWU/MIDI-LLM
├── configs/
│   ├── tokenizer_config.yaml        # Vocabulary bounds (REMI+ 512 vs AMT 55028), tick resolution
│   ├── train_local_4070.yaml        # 12GB VRAM config (QLoRA/DoRA + modules_to_save, seq_len=2048)
│   ├── train_cloud_h100.yaml        # Full CPT + SFT ZeRO-2 config (MusicPile + GigaMIDI v2 + MidiCaps)
│   ├── dpo_music_theory.yaml        # DPO alignment hyperparameters
│   └── deepspeed_zero2.json         # Multi-GPU DeepSpeed configuration
├── src/
│   ├── __init__.py
│   ├── tokenization/
│   │   ├── vocab.py                 # 512-token Tensor-Core-aligned REMI+ vocab + AMT 55028 constants
│   │   ├── midi_tokenizer.py        # Bidirectional Bar-Interleaved REMI+ & AMT wrapper
│   │   └── grammar_fsm.py           # Finite State Machine & HuggingFace LogitsProcessor
│   ├── data/
│   │   ├── downloader.py            # HF downloader for MidiCaps, GigaMIDI (v2.0.0), and MusicPile
│   │   ├── cleaner.py               # Fast C++/symusic heuristic filters, 896-ID eval exclusion & dedup
│   │   ├── mir_extractor.py         # Vectorized Krumhansl-Schmuckler fallback for uncaptioned GigaMIDI
│   │   ├── llm_captioner.py         # Converts MidiCaps 19-col & GigaMIDI v2 28-col into prompts
│   │   └── dataset.py               # PyTorch Dataset, AMT/Track-Infilling augmentation & packing
│   ├── models/
│   │   ├── model_loader.py          # MiniCPM-1B / Qwen2.5-1.5B / Llama-3.2-1B loader & vocab resize
│   │   └── music_critics.py         # Rule-based music theory & polyphony scorers for DPO
│   ├── training/
│   │   ├── train_cpt.py             # Stage 1 Continued Pretraining (MusicPile text + GigaMIDI v2.0.0)
│   │   ├── train_sft.py             # Stage 2 SFT & Infilling trainer (MidiCaps + LakhMIDI)
│   │   └── train_dpo.py             # Stage 3 Music-theoretic DPO trainer (TRL DPOTrainer)
│   ├── inference/
│   │   ├── generator.py             # HF Transformers & vLLM (FP8/BF16) generator with FSM grammar
│   │   └── exporter.py              # DAW-ready SMF Type 1 writer (GM Programs, Ch 10 Drums, zero hang)
│   └── utils/
│       ├── audio_synth.py           # FluidSynth + librosa silence trim + pyloudnorm (-18 LUFS) + MP3
│       └── metrics.py               # Objective MIR metrics + excessive simultaneous note validator
├── scripts/
│   ├── cli.py                       # Unified cross-platform CLI (prepare, cpt, sft, dpo, generate, eval)
│   ├── train_local_4070.ps1         # Windows PowerShell launcher for RTX 4070 Super
│   └── train_local_4070.sh          # Linux/WSL2 bash launcher
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures (synthetic multitrack MIDIs, mock tokenizers)
│   ├── test_vocab.py                # Verifies 512-token alignment, bijective ID mappings, AMT constants
│   ├── test_tokenizer_roundtrip.py  # Zero-drift roundtrip & triplet/swing preservation tests
│   ├── test_grammar_fsm.py          # Verifies FSM prevents out-of-order token generation
│   ├── test_cleaner_and_mir.py      # Tests anti-metronome filter, 896-ID holdout, Krumhansl key detector
│   ├── test_infilling_mask.py       # Verifies bar-interleaved & AMT infilling formatting + label=-100 mask
│   ├── test_music_critics.py        # Tests DPO harmonic clash, scale adherence, and groove lock scorers
│   └── test_midi_export.py          # Verifies SMF Type 1 validity, Ch 10 drums, zero hanging notes
├── .python-version                  # Pinned Python 3.11 for uv
├── pyproject.toml                   # Unified uv, ruff, pytest, coverage, and build configuration
├── uv.lock                          # Deterministic lockfile managed by uv
└── README.md
```

---

## 4. DETAILED COMPONENT SPECIFICATIONS

### Component A: Symbolic Vocabulary, Tokenizer & Grammar FSM (`src/tokenization/`)

#### A.1 Bar-Interleaved REMI+ (512-Token Mode — Default for 12GB VRAM & DAW Control)
1. **Triplet & Swing Resolution (96 Positions / 4/4 Bar):**
   * Standardize on **24 ticks per quarter note** (`96` positions per 4/4 bar: `[POS_0]` to `[POS_95]`). Unlike a 64-step power-of-2 grid, 96 cleanly divides both 16th/32nd duplets AND 8th/16th triplets.
2. **Bar-Interleaved Multitrack Serialization:**
   * Serialize **Bar-by-Bar across active tracks** so vertical harmony (Bass + Chords + Drums in the same bar) stays within a ~50–150 token local attention window:
   ```text
   [BAR_0] [TS_4_4] [BPM_120] [CHORD_F_MIN7]
     [TRACK_DRUMS]  [INST_DRUMS] [POS_0] [PITCH_36] [DUR_6] [VEL_26] ...
     [TRACK_BASS]   [INST_33]    [POS_0] [PITCH_41] [DUR_24] [VEL_24] ...
     [TRACK_CHORDS] [INST_4]     [POS_0] [PITCH_60] [DUR_48] [VEL_18] [PITCH_65] [DUR_48] [VEL_18] ...
   [BAR_1] ...
   ```
3. **Exact 512-Token Allocation (Tensor-Core 64x Aligned):**
   * **Structure & Control (16 tokens):** `[MUSIC_BOS]`, `[MUSIC_EOS]`, `[MUSIC_PAD]`, `[PROMPT_START]`, `[PROMPT_END]`, `[INFILL_START]`, `[INFILL_END]`, `[MASK_TRACK]`, `[MASK_BAR]`, `[BAR_START]`, `[BAR_END]`, + 5 reserved.
   * **Bar Index (64 tokens):** `[BAR_0]` to `[BAR_63]`.
   * **Metric Position (96 tokens):** `[POS_0]` to `[POS_95]`.
   * **Time Signatures (8 tokens):** `[TS_4_4]`, `[TS_3_4]`, `[TS_6_8]`, `[TS_12_8]`, `[TS_2_4]`, `[TS_5_4]`, `[TS_7_8]`, `[TS_OTHER]`.
   * **Quantized Tempo Bins (32 tokens):** `[BPM_40]` to `[BPM_200]` in 5 BPM increments.
   * **Semantic Track Roles (10 tokens):** `[TRACK_DRUMS]`, `[TRACK_BASS]`, `[TRACK_CHORDS]`, `[TRACK_PAD]`, `[TRACK_MELODY]`, `[TRACK_LEAD]`, `[TRACK_ARP]`, `[TRACK_GUITAR]`, `[TRACK_STRINGS]`, `[TRACK_FX]`.
   * **General MIDI Program Families (34 tokens):** 32 compressed GM instrument groups (`[INST_0]` to `[INST_31]`), `[INST_DRUMS]`, `[INST_UNKNOWN]`.
   * **Pitch (128 tokens):** `[PITCH_0]` to `[PITCH_127]`.
   * **Duration (64 tokens):** `[DUR_1]` to `[DUR_64]`.
   * **Velocity (32 tokens):** `[VEL_0]` to `[VEL_31]` (MIDI velocities `4..127` in steps of 4).
   * **Harmonic/Conditioning Hint Tokens (28 tokens):** `[KEY_C_MAJ]` .. `[KEY_B_MIN]` (24 keys), `[DENSITY_LOW]`, `[DENSITY_MED]`, `[DENSITY_HIGH]`, `[SWING_ON]`.

#### A.2 Optional AMT Compatibility Mode (`AMTCompatTokenizer`)
* Directly supports the `anticipation` library (`AMT_GPT2_BOS_ID = 55026`, triplet grammar `[time, dur, note]`) shifted by `base_vocab_size`, enabling seamless loading and DPO post-training of `slseanwu/MIDI-LLM_Llama-3.2-1B`.

#### A.3 Deterministic Grammar Finite State Machine (`grammar_fsm.py`)
Implement a `transformers.LogitsProcessor` (`MusicGrammarLogitsProcessor`) and vLLM `allowed_token_ids` mask enforcing valid transitions:
* **REMI+ Mode:** `[BAR_k]` $\rightarrow$ `[TS_*] | [BPM_*] | [TRACK_*] | [MUSIC_EOS]`; `[TRACK_*]` $\rightarrow$ `[INST_*] | [POS_*]`; `[POS_p]` $\rightarrow$ `[PITCH_n]`; `[PITCH_n]` $\rightarrow$ `[DUR_d]`; `[DUR_d]` $\rightarrow$ `[VEL_v]`; `[VEL_v]` $\rightarrow$ `[PITCH_{m>n}] | [POS_{q>p}] | [TRACK_*] | [BAR_{k+1}] | [MUSIC_EOS]`.
* **AMT Mode:** Enforce strict modulo-3 triplet cycle (`Step 0: Arrival Time` $\rightarrow$ `Step 1: Duration` $\rightarrow$ `Step 2: Instrument-Pitch`) and restrict logits exclusively to `ALLOWED_TOKEN_IDS`.

---

### Component B: Data Engineering & `MidiCaps` / `GigaMIDI v2.0.0` Pipeline (`src/data/`)

#### B.1 Dataset Sources & Strict Evaluation Split (`downloader.py`, `cleaner.py`)
1. **Evaluation Holdout Filtering:** Always load `assets/evaluation_set_lakh_ids.txt` (the 896 LakhMIDI MD5 hashes from `slSeanWU/MIDI-LLM`) and strictly exclude them from both CPT and SFT splits.
2. **SFT Dataset (`amaai-lab/MidiCaps`):**
   * Stream/extract `midicaps.tar.gz` and join with `train.json`.
   * Directly leverage the pre-extracted `MidiCaps` columns (`caption`, `key`, `time_signature`, `tempo`, `chord_summary`, `all_chords`, `all_chords_timestamps`, `instrument_numbers_sorted`, `genre`, `mood`) without redundant slow `music21` parsing.
3. **CPT Datasets (`m-a-p/MusicPile` + `Metacreation/GigaMIDI` `v2.0.0`):**
   * **`m-a-p/MusicPile` (~1.7B tokens):** Music theory, ABC notation, and musicology text.
   * **`Metacreation/GigaMIDI` (`name="v2.0.0"`, ~1.4B tokens):**
     * Load via `load_dataset("Metacreation/GigaMIDI", "v2.0.0", streaming=True)` using authenticated HuggingFace CLI credentials.
     * Exclude any `row["md5"]` present in `MidiCaps` (`train.json`) or `assets/evaluation_set_lakh_ids.txt`.
     * Parse raw MIDI bytes directly in-memory via `symusic.Score.from_midi(row["music"])`.
     * Filter/prioritize multitrack ensembles (`instrument_category__drums-only__0__all-instruments-with-drums__1_no-drums__2 == 1`), high expressive performance (`NOMML` and `loopability__expressive_`), and optionally synthesize weak metadata captions from `music_styles_curated`, `music_style_audio_text_Discogs`, `tempo`, and `loop_instrument_type`.

#### B.2 Fast C++ MIDI Sanitization (`cleaner.py`)
1. Reject files with $< 32$ notes, duration $< 8\text{s}$ or $> 600\text{s}$, or corrupted headers.
2. **Anti-Metronome Filter:** Drop tracks where $> 90\%$ of notes share a single pitch AND single IOI with zero velocity variance.
3. **Simultaneous Note Burst Filter:** Reject corrupted MIDI segments with $> 64$ simultaneous note-ons at a single timestamp (`has_excessive_notes_at_any_time`).
4. **Melodic N-Gram MinHash Deduplication:** Deduplicate near-identical arrangements (Jaccard threshold $\ge 0.85$).

#### B.3 Prompt Formatting & AMT / Stem Infilling Augmentation (`llm_captioner.py`, `dataset.py`)
Use the `MIDI-LLM` canonical system prompt prefix:
```text
You are a world-class composer. Please compose some music according to the following description: 
```
During SFT, mix three prompt styles and two causal task objectives (`seq_len = 2048`):
1. **Prompt Style Mixing:**
   * **50% Rich `MidiCaps` Caption:** Original `caption` field from `train.json`.
   * **30% Structured Producer Prompt:** Synthesized from `MidiCaps` / `GigaMIDI v2.0.0` features (e.g., `"{tempo} BPM {mood} {genre} track in {key} ({time_signature}) featuring {instrument_summary}. Chord progression: {chord_summary}."`).
   * **20% Short Creative Prompt:** High-level vibe/genre description.
2. **Causal Task Mixing (70% Text-to-MIDI / 30% Infilling):**
   * **70% Standard Text-to-Multitrack:** Prompt $\rightarrow$ `[MUSIC_BOS]` $\rightarrow$ Interleaved Multitrack Tokens $\rightarrow$ `[MUSIC_EOS]`.
   * **30% Anticipatory / Stem Infilling:** Mask out a target track (e.g., `[TRACK_BASS]` or `[TRACK_MELODY]`) or prefix context bars, placing the remaining tracks in the conditioning prefix and training the model to infill the missing stem between `[INFILL_START]` and `[INFILL_END]`.

---

### Component C: Model Adaptation & 12GB RTX 4070 Super Training (`src/models/`, `src/training/`)

#### C.1 Foundation Models & Smart Vocabulary Expansion (`model_loader.py`)
1. Support `openbmb/MiniCPM-1B-sft-bf16`, `Qwen/Qwen2.5-1.5B`, and `meta-llama/Llama-3.2-1B` in `torch.bfloat16`.
2. Resize token embeddings (`pad_to_multiple_of=64`) and initialize new musical token embeddings from $\mathcal{N}(\mu_{\text{old}}, \Sigma_{\text{diag}} \cdot 0.5)$ so newly added tokens start in-distribution with existing text tokens.

#### C.2 Two-Stage Local Training Recipe for 12GB VRAM (`train_sft.py`)
To train expanded embeddings + backbone inside 12GB VRAM on an RTX 4070 Super:
* **Stage 1 (Embedding & Head Alignment Warmup):** Freeze transformer layers; train only the newly added rows of `embed_tokens` and `lm_head` (via gradient row mask) with `lr=5e-4`.
* **Stage 2 (Joint LoRA/DoRA + Embeddings SFT):**
  * LoRA (`r=32, lora_alpha=64`, targets: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`) + masked trainable new embedding rows.
  * Optimizer: `paged_adamw_8bit` (`bitsandbytes`).
  * Attention: FlashAttention-2 / PyTorch SDPA (`attn_implementation="sdpa"` or `"flash_attention_2"`).
  * Memory: `gradient_checkpointing=True` (`use_reentrant=False`), `per_device_train_batch_size=2`, `gradient_accumulation_steps=16`, `max_seq_length=2048`.
* **Loss Masking:** Compute Cross-Entropy loss strictly on musical tokens (`labels = -100` over the system prompt and text description).

---

### Component D: Music-Theoretic Preference Optimization (DPO) (`src/models/music_critics.py`, `src/training/train_dpo.py`)

With syntax enforced by `MusicGrammarLogitsProcessor`, DPO aligns the model toward **musicality, harmonic resolution, and groove**:
1. **Harmonic Clash Critic ($R_{\text{harmony}}$):** Penalizes sustained minor-2nd / major-7th clashes against `[TRACK_BASS]` or the prompted `MidiCaps` `all_chords` progression on strong beats.
2. **Scale Adherence Critic ($R_{\text{scale}}$):** Rewards in-key pitch adherence matching the `MidiCaps` `key` field.
3. **Groove & Polyphony Sanity Critic ($R_{\text{groove}}$):** Rewards Bass + Kick Drum onset lock ($\pm 2$ ticks), velocity dynamic range ($H_{\text{vel}} \ge 2.1$ bits), and penalizes polyphonic note bursts (`has_excessive_notes_at_any_time`).

---

### Component E: Inference Backends, DAW Exporter & Audio Preview (`src/inference/`, `src/utils/`)

1. **Dual Inference Engine (`generator.py`):**
   * **HuggingFace Transformers Backend:** Fast startup for single-prompt generation and local testing with `MusicGrammarLogitsProcessor`.
   * **vLLM Backend:** High-throughput batch/interactive generation supporting `--fp8` quantization and `allowed_token_ids`.
2. **Studio-Grade SMF Type 1 Exporter (`exporter.py`):**
   * Outputs `mido.MidiFile(type=1, ticks_per_beat=480)` with Track 0 reserved for Tempo/Time-Signature meta-events, Channel 9 (DAW Channel 10) reserved for Drums, automatic same-pitch overlap clipping (`note_off` inserted 1 tick prior to re-trigger), and human-readable DAW track names.
3. **Normalized Audio Preview (`audio_synth.py`):**
   * Headless FluidSynth (`FluidR3_GM.sf2`) $\rightarrow$ `librosa.effects.trim(top_db=30)` $\rightarrow$ `pyloudnorm` loudness normalization (`-18.0` LUFS) $\rightarrow$ FFmpeg MP3 rendering.

---

## 5. DEVELOPER TOOLCHAIN, LINTING & QUALITY GATES (`uv`, `ruff`, `pytest`, `coverage`)

### 5.1 Package & Environment Management (`uv`)
All dependency resolution, virtual environment management, and command execution MUST use **`uv`** (`pyproject.toml` + `uv.lock`, pinned to Python `3.11`):
* **Initialize & Sync:** `uv sync --all-extras --dev`
* **CLI Entrypoint (`[project.scripts]`):** `midi-llm = "src.scripts.cli:app"` (or `uv run midi-llm ...`)
* **Optional Dependency Groups:**
  * `core`: `torch`, `transformers`, `accelerate`, `peft`, `trl`, `datasets`, `mido`, `symusic`, `numpy`, `pyyaml`, `typer`, `rich`, `tqdm`
  * `quant`: `bitsandbytes`, `vllm` (Linux/WSL2)
  * `audio`: `midi2audio`, `librosa`, `soundfile`, `pyloudnorm`
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
Configure `pytest` and `coverage` in `pyproject.toml` (`--cov=src --cov-report=term-missing --cov-report=xml --cov-fail-under=90` on core deterministic logic):
1. **`tests/test_vocab.py`:** Verifies exact 512-token Tensor-Core alignment (`len(vocab) % 64 == 0`), bijective token-to-ID mappings, and AMT 55,028 offset constants.
2. **`tests/test_tokenizer_roundtrip.py`:** Generates synthetic 4-track MIDI files (Drums, Bass, Chords, Triplet Lead), runs `encode -> decode`, and asserts exact pitch equality, $\le 1$ grid tick timing drift, and bar-interleaved ordering.
3. **`tests/test_grammar_fsm.py`:** Runs 256 steps of random logits through `MusicGrammarLogitsProcessor` (in both REMI+ and AMT triplet modes) and asserts 100% syntactically valid sequences.
4. **`tests/test_cleaner_and_mir.py`:** Tests the anti-metronome filter, `has_excessive_notes_at_any_time` burst detector, 896-ID `evaluation_set_lakh_ids.txt` holdout filter, and Krumhansl-Schmuckler key detection accuracy on synthetic major/minor scales.
5. **`tests/test_infilling_mask.py`:** Verifies stem-infilling sequence construction, `MidiCaps` prompt synthesis, and loss masking (`labels = -100` on prompt and conditioning context).
6. **`tests/test_music_critics.py`:** Verifies DPO reward critics rank consonant in-key grooves strictly higher than minor-2nd clashing or flat-velocity sequences ($R(x, y_{\text{chosen}}) > R(x, y_{\text{rejected}})$).
7. **`tests/test_midi_export.py`:** Verifies exported `.mid` files are valid SMF Type 1, assign `[TRACK_DRUMS]` to Channel 9 (DAW Ch 10), clip overlapping same-pitch notes cleanly, and leave zero hanging notes.

```powershell
# Run full test suite with line & branch coverage report
uv run pytest --cov=src --cov-branch --cov-report=term-missing
```
