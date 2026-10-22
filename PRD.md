# PRODUCT REQUIREMENTS DOCUMENT (PRD) & SYSTEM SPECIFICATION
## Project: Open, Reproducible Symbolic Multitrack Text-to-MIDI Research & Production Stack (`midi-llm-cpm`)

**Version:** 4.1 (Model Shortlist & Non-Reasoning Decoder Specification)
**Positioning:** An open, reproducible text-to-symbolic-MIDI research and production stack targeting state-of-the-art performance on a locked public benchmark.  
**Reference Baselines:**
* [`slseanwu/MIDI-LLM_Llama-3.2-1B`](https://huggingface.co/slseanwu/MIDI-LLM_Llama-3.2-1B) ([GitHub: `slSeanWU/MIDI-LLM`](https://github.com/slSeanWU/MIDI-LLM), ISMIR 2026 — `FAD = 0.173`, `CLAP = 22.1` on 896 MidiCaps eval IDs)
* [`amaai-lab/text2midi`](https://github.com/amaai-lab/text2midi) (Symbolic text-to-MIDI & human listening protocol)
* [`Agogic`](https://yisuanwang.github.io/Agogic/) & [`Moonbeam`](https://arxiv.org/abs/2505.15559) (Expressive performance timing & multitrack infilling representations)
* [`SAGE-Music`](https://arxiv.org/abs/2510.00395) (Low-latency symbolic music generation benchmarking)

---

## 1. EXECUTIVE SUMMARY & RESEARCH VISION
Design, implement, and benchmark an open-source, reproducible **Symbolic Multitrack Text-to-MIDI Generation and Stem-Infilling Stack**.

The system accepts natural language musical descriptions (e.g., *"118 BPM neo-soul groove in F minor with lush Rhodes 9th voicings, syncopated slap bass, and swung boom-bap drums"*)—optionally paired with an **Explicit Musical Plan** (form, harmonic rhythm, instrumentation) or existing MIDI stems—and autoregressively outputs expressive, multitrack **Standard MIDI Files (SMF Type 1, `.mid`)** verified for import into Ableton Live, Logic Pro, FL Studio, and Reaper.

### 1.1 Architectural Hypotheses & Differentiators to Validate via Ablation
Rather than assuming SOTA in advance, this stack is structured around five falsifiable architectural hypotheses evaluated via locked ablations (Component E.4):

| Hypothesis | `MIDI-LLM_Llama-3.2-1B` Baseline | `midi-llm-cpm` v4.1 Design | Ablation Metric |
| :--- | :--- | :--- | :--- |
| **H1: Compact Expressive Representation vs. 55k Arrival-Time Triplets** | Stanford AMT (`+55,030` tokens; flat velocity, no `CC64`, 10ms bins) | **Bar-Interleaved REMI-Plan (`+661` tokens; `pad_to_multiple_of=64` on total vocab)** with **Per-Note Microtiming Residuals (`[MICRO_*]`)**, **32 Velocity Bins**, **`CC64`/`CC1`/`CC11`/`PitchBend`**, and **130 Full GM Instrument Tokens (128 Melodic + Drums + Unknown)** | Microtiming error (ms), Velocity Entropy, Multi-SoundFont FAD |
| **H2: Explicit Musical Plan vs. Direct Note Emission** | Chords/structure only in freeform text prompt | **Controllable Plan Header & Beat-Level Harmonic Rhythm:** `Form -> Beat-Level Chords (Root/Quality/Bass) -> Instrumentation -> Notes` | Chord Adherence (%), Structural Recurrence, User Plan Editability |
| **H3: Classifier-Free Guidance (LLM-CFG)** | Standard causal sampling (`temp=1.0, top_p=0.98`) | **10% Unconditional Prompt Dropout + Batched Logit-Level CFG ($\gamma \in [1.5, 2.5]$)** | LAION-CLAP ($\uparrow$), Instrument/Tempo Recall ($\uparrow$) |
| **H4: Work-Deduplicated 12x Transposition Augmentation** | Raw corpus key distribution (~45% C Maj / A Min bias) | **Work-level deduplication prior to split + 12x Joint Text-Regex & Symbolic Pitch Transposition ($\Delta p \in [-5, +6]$)** | Rare-Key Scale Adherence (`F# min`, `Db maj`), Tonal Distance |
| **H5: Progressive `2,048 -> 4,096` Context Curriculum** | Fixed `2,048` context (~13–14 bars) | **Phase A (`2,048`) $\rightarrow$ Phase B (`4,096`) with Chunked Cross-Entropy + Unbounded `[BAR]` Relative Indexing** | 32-bar Structural Completeness, 16-bar Stem Infilling F1 |

### 1.2 Model Selection Policy
* **Primary decoder candidate:** [`openbmb/MiniCPM5-1B-Base`](https://huggingface.co/openbmb/MiniCPM5-1B-Base), fine-tuned as a causal MIDI-token decoder. Use a no-think task format for text-facing generation; serialized MIDI output must contain only the specified plan and music grammar, not free-form reasoning traces. The post-trained [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B) checkpoint supports both thinking and no-think modes and may be profiled as a separate candidate.
* **Comparison backbones:** [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) in non-thinking mode, and `meta-llama/Llama-3.2-1B` / the published MIDI-LLM checkpoint for direct baseline comparison. Include `Qwen2.5-1.5B` only as an optional capacity/VRAM ablation, not the default local model.
* The supplied general-capability charts are used only to prioritize this shortlist. They do not establish music-generation quality. Select the release decoder only after the locked text-to-MIDI bake-off in Component E, using the same data split, tokenization, training-token budget, decoding settings, and music-specific metrics for each candidate.
* A thinking model may be evaluated as an optional prompt-to-plan component. It is outside the critical path and must return a schema-validated plan; the MIDI decoder remains non-thinking. Keep this path only if it improves held-out prompt adherence or human preference enough to justify added latency.

### 1.3 Target Use Cases
1. **Unconditional Text-to-Multitrack Generation:** Generate 4 to 64+ bars of multi-instrument arrangement from text prompts.
2. **Plan-Conditioned Generation (Editable Blueprint):** Generate or edit an explicit symbolic plan (`Section -> Beat-Level Chord Progression -> Active GM Instruments`) before rendering notes.
3. **Conditional Stem Infilling (Accompaniment Generation):** Given existing tracks (e.g., 16 bars of Drums + Chords) and a prompt, infill a missing target stem (`[TRACK_BASS]`, `[TRACK_MELODY]`) locked to the context groove and harmonic rhythm.
4. **Bar Continuation & Outpainting:** Extend a 4-to-8 bar seed loop into a multi-section arrangement using unbounded `[BAR]` relative indexing.

### 1.4 Hardware Tiers & OS Support Matrix
To ensure reproducibility across academic labs and independent contributors, hardware targets are separated into explicit tiers and verified via a mandatory **Phase 0 Memory & Throughput Benchmark (`scripts/bench_hardware.py`)**:

| Tier | Hardware Envelope | Supported Workflows | OS Support |
| :--- | :--- | :--- | :--- |
| **Tier 0 (CPU-Only)** | Any modern CPU, 8 GB RAM (`midi-llm-cpm[cpu]`) | Tokenization, MIR, Explicit Plan parsing, SMF Type 1 export, unit tests | Windows, Linux, macOS |
| **Tier 1 (12 GB VRAM)** | Consumer GPU (12 GB VRAM) | Inference (`bf16`/`fp8`), QLoRA/DoRA SFT (`seq_len=2048` validated; `seq_len=4096` gated on Chunked CE benchmark $\le 11.2\text{ GB}$) | Windows Native (SDPA/HF), WSL2/Linux (Triton/`liger-kernel`/`vLLM`) |
| **Tier 2 (16–24 GB VRAM)** | Workstation GPU (16–24 GB VRAM, matching `MIDI-LLM` baseline rec) | Native `seq_len=4096` LoRA/DoRA SFT, Offline DPO, batched vLLM inference | Windows Native, WSL2/Linux |
| **Tier 3 (Multi-GPU Cloud)** | 4x–8x 80 GB GPUs (DeepSpeed ZeRO-2/3) | Full-parameter CPT (`MusicPile` + `GigaMIDI v2.0.0`), Full SFT, Online GRPO experiments | Linux |

---

## 2. DATA SOURCES, LICENSING & STRICT LEAKAGE CONTROL

### 2.1 Corpus Inventory & License Separation
Because symbolic music datasets carry heterogeneous licenses, the data pipeline enforces **strict license partitioning** so users can train either a **Research-Only (Non-Commercial)** checkpoint or a **Commercially Permissive** checkpoint:

| Dataset | Role | Size / Files | License | Usage Policy in `midi-llm-cpm` |
| :--- | :--- | :--- | :--- | :--- |
| **[`amaai-lab/MidiCaps`](https://huggingface.co/datasets/amaai-lab/MidiCaps)** | Primary SFT (Paired Text + LakhMIDI + 19 MIR columns) | 168,385 MIDIs + captions | **CC-BY-SA-4.0** | Default SFT corpus; directly provides `caption`, `key`, `tempo`, `all_chords`, `all_chords_timestamps`, `instrument_numbers_sorted`. |
| **[`Metacreation/GigaMIDI`](https://huggingface.co/datasets/Metacreation/GigaMIDI) (`name="v2.0.0"`)** | Primary CPT (Standalone Expressive Multitrack MIDIs + 28 columns) | 2,136,218 MIDIs (2.3M expressive loops) | **CC-BY-NC-4.0** (Gated) | **Non-Commercial Research Track Only.** Must pass `name="v2.0.0"` (`v1.0.0` is HF default). Disabled when `--commercial-only` flag is set. |
| **[`m-a-p/MusicPile`](https://huggingface.co/datasets/m-a-p/MusicPile)** | Text/Symbolic CPT | ~1.7B tokens | Mixed Open | Filtered to permissive music theory / ABC notation subsets. |
| **896 Evaluation Holdout (`assets/evaluation_set_lakh_ids.txt`)** | Locked Public Benchmark | 896 LakhMIDI IDs | CC-BY-SA-4.0 | Strictly isolated for zero-shot evaluation. |

### 2.2 Multi-Stage De-Duplication & Anti-Leakage Protocol (`src/data/dedup_leakage.py`)
File-level MD5 matching and single-track MinHash are insufficient to prevent train/test leakage across LakhMIDI, `MidiCaps`, and `GigaMIDI v2.0.0` (which aggregates Lakh, MetaMIDI, XMIDI, MAGD, and MASD, often containing multiple transcriptions or transpositions of the same song). We enforce a **4-layer leakage barrier BEFORE any pitch-transposition augmentation**:
1. **Layer 1 — Exact & Embedded Hash Exclusion:** Block all 896 evaluation MD5s (`assets/evaluation_set_lakh_ids.txt`) and `MidiCaps` test-split hashes across every corpus.
2. **Layer 2 — Canonical Work / Metadata Clustering:** Normalize `(title, artist)` strings and match external identifiers (`audio_text_matches_sid` Spotify IDs and `audio_text_matches_mbid` MusicBrainz IDs in `GigaMIDI v2.0.0`) so all arrangements of the same musical work are assigned to the same split. Any work matching an evaluation track is purged from training.
3. **Layer 3 — Transposition-Invariant Chromagram & Interval LSH:** Compute bar-level **12-D pitch-class chromagram vectors (circularly normalized to tonic = 0)** and **melodic interval $(\Delta \text{pitch}, \text{IOI ratio})$ 5-gram MinHash LSH**. Reject any training candidate with cosine similarity $\ge 0.92$ or Jaccard similarity $\ge 0.80$ to any evaluation track across all 12 possible transpositions.
4. **Layer 4 — Frozen Public Split Manifest:** Export `manifests/frozen_splits_v1.parquet` (recording file ID, work cluster ID, split assignment, and nearest-eval-neighbor distance) alongside an automated leakage audit report (`uv run midi-llm audit-leakage`).

---

## 3. REPOSITORY & DIRECTORY ARCHITECTURE

```text
midi-llm-cpm/
├── PRD.md                           # This Product Requirements Document & Specification (v4.1)
├── LICENSE                          # Apache-2.0 License for source code
├── MODEL_AND_DATA_LICENSES.md       # License compatibility matrix (Apache-2.0, Llama-3.2, CC-BY-SA, CC-BY-NC)
├── assets/
│   ├── evaluation_set_lakh_ids.txt  # 896 held-out LakhMIDI MD5 IDs from slSeanWU/MIDI-LLM
│   └── example_prompts.txt          # Canonical evaluation prompts from slSeanWU/MIDI-LLM
├── configs/
│   ├── tokenizer_config.yaml        # REMI-Plan vocab, 24-PPQ + microtiming bins, 130 GM tokens
│   ├── train_tier1_12gb.yaml        # Tier 1 (12 GB VRAM) config (QLoRA/DoRA, Chunked CE, 2k->4k)
│   ├── train_tier2_24gb.yaml        # Tier 2 (16-24 GB VRAM) config (Native 4k LoRA/DoRA + DPO)
│   ├── train_tier3_cloud.yaml       # Tier 3 (Multi-GPU ZeRO-2/3) config (Full CPT + SFT)
│   ├── align_dpo.yaml               # Validated Offline DPO & experimental GRPO hyperparameters
│   └── deepspeed_zero2.json         # Multi-GPU DeepSpeed configuration
├── src/
│   ├── __init__.py
│   ├── tokenization/
│   │   ├── vocab.py                 # REMI-Plan vocabulary (32 BPM bins, 130 GM, beat chords, microtiming)
│   │   ├── midi_tokenizer.py        # Bidirectional Bar-Interleaved REMI-Plan & AMT 55k wrapper
│   │   └── grammar_fsm.py           # Finite State Machine LogitsProcessor & LLM-CFG processor
│   ├── data/
│   │   ├── downloader.py            # HF loader for MidiCaps, GigaMIDI (v2.0.0), MusicPile (+ license gate)
│   │   ├── dedup_leakage.py         # 4-layer work-level & transposition-invariant leakage blocker
│   │   ├── cleaner.py               # C++/symusic filters, unquantized MIDI policy & anti-metronome check
│   │   ├── mir_extractor.py         # Krumhansl key, beat-level chord/inversion & section plan extractor
│   │   ├── transposition.py         # 12x Joint Text-Regex + Symbolic MIDI Pitch-Transposition Augmenter
│   │   ├── llm_captioner.py         # Converts MidiCaps & GigaMIDI v2 metadata into prompt/plan pairs
│   │   └── dataset.py               # PyTorch Dataset, 2048->4096 curriculum, 10% CFG dropout & infilling
│   ├── models/
│   │   ├── model_loader.py          # Model loader, pad_to_multiple_of=64 resize, variance init & Chunked CE
│   │   └── music_critics.py         # Calibrated preference critics + anti-reward-hacking diversity guards
│   ├── training/
│   │   ├── train_cpt.py             # Stage 1 Continued Pretraining (MusicPile + GigaMIDI v2.0.0)
│   │   ├── train_sft.py             # Stage 2 Progressive SFT (Embed Warmup -> 2k SFT -> 4k SFT)
│   │   └── train_dpo.py             # Stage 3 Validated Offline DPO (primary) & Online GRPO (experimental)
│   ├── inference/
│   │   ├── generator.py             # HF & Linux/WSL2 vLLM generator with LLM-CFG, Plan edit & outpainting
│   │   └── exporter.py              # DAW-ready SMF Type 1 writer (128 GM, CC1/11/64, PitchBend, Microtiming)
│   └── evaluation/
│       ├── audio_synth.py           # Multi-SoundFont FluidSynth renderer + LUFS normalization
│       ├── midi_metrics.py          # MIDI-native suite (microtiming ms, chord acc, tonal dist, recurrence)
│       ├── benchmark_midillm.py     # Locked 896-ID multi-seed FAD, CLAP & latency (TTFB/RTF) harness
│       └── human_eval_prep.py       # Generates blinded MUSHRA-style listening test bundles for producers
├── scripts/
│   ├── cli.py                       # Unified CLI (audit-leakage, bench-hw, prepare, train, generate, eval)
│   ├── bench_hardware.py            # Phase 0 VRAM & throughput profiler across sequence lengths & models
│   ├── train_tier1.ps1              # Windows PowerShell launcher (SDPA + HF backend)
│   └── train_tier1.sh               # Linux/WSL2 bash launcher (FlashAttn-2 / Liger / vLLM)
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── daw_fixtures/                # Golden SMF Type 1 regression files for Ableton/Logic/FL/Reaper
│   ├── test_vocab.py                # Verifies exact token counts (32 BPM, 130 GM, beat chords, microtiming)
│   ├── test_tokenizer_roundtrip.py  # Sub-millisecond roundtrip with microtiming residuals, CCs & pitch bend
│   ├── test_grammar_fsm.py          # Verifies FSM LogitsProcessor & LLM-CFG across Plan + Note states
│   ├── test_dedup_leakage.py        # Verifies transposition-invariant leakage blocker catches shifted clones
│   ├── test_transposition.py        # Verifies joint text-regex + symbolic pitch/chord/inversion shifts
│   ├── test_infilling_and_ce.py     # Verifies stem infilling, Chunked CE exactness & label=-100 masking
│   ├── test_music_critics.py        # Verifies critics + anti-reward-hacking penalties on degenerate loops
│   └── test_midi_export.py          # Verifies SMF Type 1 DAW compliance, 128 GM programs, zero hanging notes
├── .python-version                  # Pinned Python 3.11 for uv
├── pyproject.toml                   # Unified uv, ruff, pytest, coverage, and optional extras ([cpu], [gpu])
└── README.md
```

---

## 4. DETAILED TECHNICAL SPECIFICATIONS

### Component A: Expressive REMI-Plan Symbolic Representation & Grammar FSM (`src/tokenization/`)

#### A.1 Addressing Representation Bottlenecks (Grid + Microtiming + Continuous Controls)
A coarse 24-PPQ grid alone quantizes away human groove, while raw 10ms arrival-time triplets (AMT) lose bar/beat structure and chord alignment. Inspired by **Agogic** ([Wang et al.](https://yisuanwang.github.io/Agogic/)) and **Moonbeam** ([arXiv:2505.15559](https://arxiv.org/abs/2505.15559)), we combine a **96-position metric bar grid (`24` PPQ = 20 ticks per step at 480 TPQ)** with **Per-Note Microtiming Residuals (`[MICRO_-4]`..`[MICRO_+4]`)** and **Expressive Continuous Control Tokens**:
* **Exact Timing Reconstruction:** Each note's tick onset at `480` TPQ is reconstructed as $t_{\text{onset}} = t_{\text{bar}} + 20 \cdot \text{POS}_p + 2 \cdot \text{MICRO}_m$, where $t_{\text{bar}}$ is the bar start tick and $m \in [-4, +4]$ spans $[-8, +8]$ ticks ($\approx \pm 8.3\text{ ms}$ at 120 BPM, reducing quantization error to **$\le 1\text{ tick} \approx 1\text{ ms}$**).
* **Unbounded Bar Indexing:** Instead of a fixed `[BAR_0]..[BAR_63]` vocabulary (which overflows on songs $> 64$ bars), every bar begins with a universal **`[BAR]` boundary token** followed by a **Phrase-Relative Bar Index (`[BAR_REL_0]`..`[BAR_REL_15]`)** and optional **Section Countdown (`[BARS_LEFT_1]`..`[BARS_LEFT_16]`)**. This supports arbitrarily long arrangements without vocabulary overflow.
* **Beat-Level Harmonic Rhythm & Inversions:** Chords are not restricted to one per bar. `[CHORD_ROOT_*] [CHORD_QUAL_*] [CHORD_BASS_*]` can be emitted at **any `[POS_p]`** within a bar (e.g., `POS_0`, `POS_24`, `POS_48`, `POS_72`), capturing multi-chord bars, syncopated harmonic anticipations, and slash-chord bass inversions.

#### A.2 Explicit Controllable Musical Plan (`Form -> Harmonic Rhythm -> Instrumentation -> Notes`)
Before emitting bar-interleaved notes, each sequence includes an **Explicit Musical Plan** block between `[PLAN_START]` and `[PLAN_END]` that is **human-readable, user-editable, and independently evaluable**:
```text
[PROMPT_START] <System + User Prompt> [PROMPT_END]
[PLAN_START]
  [KEY_F_MIN] [TS_4_4] [BPM_120] [EXPR_HUMAN]
  [ACTIVE_TRACKS] [TRACK_DRUMS] [INST_DRUMS] [TRACK_BASS] [INST_33] [TRACK_CHORDS] [INST_4]
  [SEC_VERSE] [BAR_REL_0] [POS_0] [CHORD_ROOT_F] [CHORD_QUAL_MIN9] [CHORD_BASS_ROOT]
              [BAR_REL_1] [POS_0] [CHORD_ROOT_BB] [CHORD_QUAL_MIN7] [CHORD_BASS_ROOT] [POS_48] [CHORD_ROOT_C] [CHORD_QUAL_DOM7] [CHORD_BASS_3RD]
[PLAN_END]
[MUSIC_BOS]
  [BAR] [BAR_REL_0] [SEC_VERSE] [POS_0] [CHORD_ROOT_F] [CHORD_QUAL_MIN9] [CHORD_BASS_ROOT]
    [TRACK_DRUMS]  [INST_DRUMS] [POS_0] [MICRO_0] [PITCH_36] [DUR_6] [VEL_26] ...
    [TRACK_BASS]   [INST_33]    [POS_0] [MICRO_+1] [PITCH_41] [DUR_24] [VEL_24] ...
    [TRACK_CHORDS] [INST_4]     [POS_0] [PEDAL_ON] [MICRO_0] [PITCH_60] [DUR_48] [VEL_18] ...
[MUSIC_EOS]
```

#### A.3 Complete Symbolic Vocabulary Table & Total-Vocab Alignment
| Token Group | Count | Exact Specification |
| :--- | :--- | :--- |
| **Structure, Plan & Section Control** | **28** | `[MUSIC_BOS]`, `[MUSIC_EOS]`, `[MUSIC_PAD]`, `[PROMPT_START]`, `[PROMPT_END]`, `[PLAN_START]`, `[PLAN_END]`, `[ACTIVE_TRACKS]`, `[INFILL_START]`, `[INFILL_END]`, `[MASK_TRACK]`, `[MASK_BAR]`, `[BAR]`, `[EXPR_HUMAN]`, `[EXPR_QUANTIZED]`, `[SWING_ON]`, `[DENSITY_LOW]`, `[DENSITY_MED]`, `[DENSITY_HIGH]`, `[SEC_INTRO]`, `[SEC_VERSE]`, `[SEC_PRECHORUS]`, `[SEC_CHORUS]`, `[SEC_BRIDGE]`, `[SEC_SOLO]`, `[SEC_OUTRO]`, `[PEDAL_ON]`, `[PEDAL_OFF]` |
| **Phrase-Relative Bar & Countdown** | **32** | `[BAR_REL_0]` to `[BAR_REL_15]` (16 tokens) + `[BARS_LEFT_1]` to `[BARS_LEFT_16]` (16 tokens) |
| **Metric Position (`24` PPQ)** | **96** | `[POS_0]` to `[POS_95]` (in 4/4; scales with active time signature up to `12/8` / `5/4`) |
| **Per-Note Microtiming Residuals** | **9** | `[MICRO_-4]` to `[MICRO_+4]` ($\pm 8$ ticks in 2-tick steps around grid position) |
| **Time Signatures** | **8** | `[TS_4_4]`, `[TS_3_4]`, `[TS_6_8]`, `[TS_12_8]`, `[TS_2_4]`, `[TS_5_4]`, `[TS_7_8]`, `[TS_OTHER]` |
| **Quantized Tempo Bins & Curves** | **39** | **32 Absolute Bins:** `[BPM_45]` to `[BPM_200]` inclusive in 5-BPM steps ($\frac{200-45}{5} + 1 = 32$) + **7 Relative Tempo Curve Tokens:** `[TEMPO_HOLD]`, `[TEMPO_ACCEL_1..3]`, `[TEMPO_RIT_1..3]` |
| **Key Signatures** | **24** | `[KEY_C_MAJ]` .. `[KEY_B_MAJ]` (12) + `[KEY_C_MIN]` .. `[KEY_B_MIN]` (12) |
| **Beat-Level Chords & Inversions** | **36** | **12 Roots:** `[CHORD_ROOT_C..B]`; **16 Qualities:** `MAJ`, `MIN`, `DOM7`, `MAJ7`, `MIN7`, `MIN7B5`, `DIM7`, `AUG`, `SUS2`, `SUS4`, `ADD9`, `MIN9`, `MAJ9`, `DOM9`, `ALT`, `POW5`; **7 Bass Inversion Degrees:** `[CHORD_BASS_ROOT]`, `B3`, `3RD`, `4TH`, `5TH`, `B7`, `7TH`; + `[CHORD_NC]` |
| **Semantic Track Roles** | **10** | `[TRACK_DRUMS]`, `[TRACK_BASS]`, `[TRACK_CHORDS]`, `[TRACK_PAD]`, `[TRACK_MELODY]`, `[TRACK_LEAD]`, `[TRACK_ARP]`, `[TRACK_GUITAR]`, `[TRACK_STRINGS]`, `[TRACK_FX]` |
| **Full General MIDI Instruments** | **130** | **All 128 GM Melodic Programs:** `[INST_0]` to `[INST_127]` + `[INST_DRUMS]` + `[INST_UNKNOWN]` |
| **Pitch** | **128** | `[PITCH_0]` to `[PITCH_127]` |
| **Duration** | **64** | `[DUR_1]` to `[DUR_64]` (1-tick resolution up to 24 ticks; logarithmic steps up to 4 bars) |
| **Velocity** | **32** | `[VEL_0]` to `[VEL_31]` (MIDI velocity `4..127` in steps of 4) |
| **Continuous Controls (`PitchBend`/`CC`)** | **25** | **9 Pitch Bend Bins:** `[PBEND_-4]`..`[PBEND_+4]`; **8 Modulation (`CC1`) Bins:** `[CC1_0]`..`[CC1_7]`; **8 Expression (`CC11`) Bins:** `[CC11_0]`..`[CC11_7]` |
| **Total Symbolic Tokens ($V_{\text{music}}$)** | **661** | **Total Model Vocab Alignment:** We call `model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)` so the **final combined vocabulary size** $V_{\text{total}} = \lceil (V_{\text{base}} + 661) / 64 \rceil \times 64$ is strictly 64-aligned for GPU Tensor Cores regardless of the base LLM's starting vocabulary size. |

---

### Component B: Data Engineering, Unquantized MIDI Policy & Transposition (`src/data/`)

#### B.1 Explicit Policy for Unquantized vs. Quantized Source MIDI (`cleaner.py`)
Real-world corpora (`GigaMIDI v2.0.0`, `LakhMIDI`) mix DAW-quantized tracks with unquantized live performances lacking tempo maps:
1. **Beat-Grid Alignment Check:** Compute the **Onset Grid Alignment Ratio (OGAR)** against the MIDI file's embedded tempo/time-signature map (allowing $\pm 8$ ticks for `[MICRO_*]` human swing/rubato).
2. **Case 1 — Metric Expressive MIDI ($\text{OGAR} \ge 0.70$, valid tempo map):** Encode directly using `[POS_*]` + `[MICRO_-4..+4]` residuals and tag `[EXPR_HUMAN]` (if `NOMML > 0` or microtiming variance $> 2$ ticks) or `[EXPR_QUANTIZED]`.
3. **Case 2 — Unaligned Live Performance ($\text{OGAR} < 0.70$):** Do **not** force-quantize onto a corrupt 120 BPM grid (which destroys rhythm). Either re-estimate beat positions via dynamic programming beat tracking (if confidence $\ge 0.80$) or route to an isolated raw-time (`AMTCompatTokenizer`) auxiliary split.

#### B.2 12x Joint Text + Symbolic Pitch-Transposition Augmentation (`transposition.py`)
Applied **only after** work-level train/val/test splitting (Section 2.2):
1. Shift all non-drum `[PITCH_n]` tokens by $\Delta p \in [-5, +6]$ semitones (wrapping/clamping within instrument register bounds).
2. Circularly shift `[KEY_*]` and all `[CHORD_ROOT_*]` tokens in both the `[PLAN_START]..[PLAN_END]` block and `[BAR]` headers by $\Delta p \pmod{12}$.
3. Rewrite all key names and chord symbols in the natural language prompt via an enharmonic-aware parser matched to the target key signature.

---

### Component C: Model Adaptation, Phase 0 VRAM Gate & Progressive `2k -> 4k` Training (`src/models/`, `src/training/`)

#### C.1 Phase 0 Hardware & Memory Verification Gate (`scripts/bench_hardware.py`)
Before running multi-hour training, `uv run midi-llm bench-hw` profiles peak CUDA memory and tokens/sec across:
* Base Models: `openbmb/MiniCPM5-1B-Base` (primary candidate), `Qwen/Qwen3.5-0.8B` (non-thinking comparison), and `meta-llama/Llama-3.2-1B` (MIDI-LLM comparison). Optionally profile `Qwen2.5-1.5B` as a capacity/VRAM ablation.
* Sequence Lengths: `2,048` (`bs=2, grad_accum=16`) vs. `4,096` (`bs=1, grad_accum=32`)
* Loss Implementations: Standard `F.cross_entropy` vs. **512-Token Chunked Cross-Entropy** vs. `liger-kernel` (Linux/WSL2)
* **Acceptance Rule:** Tier 1 (`12 GB VRAM`) configs automatically enable `4,096` in Phase 2B only when `bench_hardware.py` confirms peak reserved VRAM $\le 11.2\text{ GB}$ on the active device; otherwise `4,096` is deferred to Tier 2 (`16–24 GB VRAM`).
* **Mode Rule:** Use a no-think task format for MIDI generation and supervised fine-tuning. Do not train or decode free-form reasoning traces as part of the music-token target. The base checkpoint is preferred for adaptation; the post-trained MiniCPM5-1B no-think mode may be included in the bake-off.

#### C.2 Progressive Context Curriculum (`train_sft.py`)
1. **Stage 1 (Embedding & Head Alignment Warmup — 10% steps, `seq_len=2,048`):** Freeze transformer blocks; train only newly added rows of `embed_tokens` and `lm_head` (`lr=5e-4`).
2. **Stage 2A (Core SFT + Plan Supervision — 70% steps, `seq_len=2,048`):** Train LoRA/DoRA (`r=32, alpha=64`) + new embeddings with 10% unconditional prompt dropout (for LLM-CFG) and 30% stem-infilling examples.
3. **Stage 2B (Long-Horizon Song & 16-Bar Stem Infilling SFT — Final 20% steps, `seq_len=4,096`):** Extend context window to `4,096` tokens (28–32 bars of 4-track music or full 16-bar context + 16-bar stem infilling) using Chunked Cross-Entropy (`lr=5e-5`).

---

### Component D: De-Risked Preference Alignment (`src/models/music_critics.py`, `src/training/train_dpo.py`)

Rule-based music critics can easily be **reward-hacked** by a policy collapsing into monotonous, zero-syncopation root-note loops. We therefore structure post-training alignment as a **strictly gated 3-step progression**:

1. **Milestone 1 (Primary Deliverable — Zero RL Dependency):**
   * SFT + Explicit Musical Plan + Deterministic `MusicGrammarLogitsProcessor` + `LLM-CFG`. Must independently beat or match baseline metrics before any preference optimization is applied.
2. **Milestone 2 (Calibrated Offline DPO with Anti-Collapse Guards):**
   * **Critic Calibration:** Validate composite critic scores ($R_{\text{harmony}}, R_{\text{scale}}, R_{\text{groove}}, R_{\text{prompt-inst}}, R_{\text{dynamics}}$) against a held-out set of **200 human/expert-ranked MIDI pairs**; tune critic weights so pairwise agreement with human raters is $\ge 75\%$.
   * **Anti-Reward-Hacking Guards:** Reject any $y_{\text{chosen}}$ candidate whose intra-song bar self-similarity exceeds `0.85` (anti-looping guard), whose note density falls outside the 10th–90th percentile of the training corpus, or whose velocity entropy $H_{\text{vel}} < 2.0$ bits.
   * Run conservative Offline `DPOTrainer` ($\beta = 0.2$, $\le 1$ epoch, low `lr = 5e-6`).
3. **Milestone 3 (Experimental Online GRPO — Research Extension Only):**
   * Online Group Relative Policy Optimization ($G=4$) is isolated as a non-blocking research experiment evaluated in the ablation table.

---

### Component E: Evaluation Protocol, Ablations & Human Listening Study (`src/evaluation/`)

To substantiate any SOTA claim, evaluation is conducted across **four orthogonal pillars** under a locked protocol (**3 random seeds: `42, 1337, 2026`**, reporting **mean $\pm$ 95% CI**):

#### E.1 Audio-Rendered Benchmarks (`benchmark_midillm.py`)
1. **Primary `MIDI-LLM` Comparability (896 Held-Out Lakh IDs):**
   * Render via pinned `FluidSynth 2.3.x` + `FluidR3_GM.sf2` + `-18.0 LUFS` normalization (`pyloudnorm`).
   * Compute **Fréchet Audio Distance (FAD $\downarrow$)** and **LAION-CLAP Score ($\uparrow$)** against `slseanwu/MIDI-LLM_Llama-3.2-1B` (`FAD = 0.173`, `CLAP = 22.1`).
2. **Multi-SoundFont Robustness Check:**
   * Re-evaluate FAD and CLAP using a second independent SoundFont (`GeneralUser_GS.sf2`) to verify improvements come from symbolic composition quality rather than `FluidR3_GM` instrument-patch artifacts.

#### E.2 MIDI-Native Symbolic Metrics (`midi_metrics.py`)
1. **Timing & Expressiveness (per [Agogic](https://yisuanwang.github.io/Agogic/)):** Round-trip onset/offset timing error (ms & ticks), Inter-Onset Interval (IOI) histogram KL-divergence, and Velocity Entropy ($H_{\text{vel}}$).
2. **Harmony & Tonal Structure:** In-Key Scale Consistency (%), Beat-Level Plan Chord Adherence (%), and Tonal Distance ([Harte et al.](https://dl.acm.org/doi/10.1145/1180639.1180665)).
3. **Arrangement & Form:** Polyphony distribution KL-divergence, Prompted GM Instrument Recall/Precision, Bar-to-Bar Structural Recurrence score, and Degenerate Repetition rate.
4. **Task-Specific Continuation & Stem Infilling Metrics:**
   * **Stem Infilling:** Bass-Kick Groove Lock F1, Harmonic Clash rate against context stems (< 1.5%), and held-out Ground-Truth Stem NLL / Chroma Cosine Similarity.
   * **Bar Continuation:** Boundary voice-leading continuity and motif similarity across the seed-to-continuation junction.
5. **Latency Benchmark Protocol (per [SAGE-Music](https://arxiv.org/abs/2510.00395)):**
   * Report **Time-to-First-Bar (TTFB, ms)** and **Tokens/sec** at `batch_size=1` and `batch_size=4` across HuggingFace (`bf16`) and Linux/WSL2 `vLLM` (`bf16` & `fp8`) on each hardware tier.

#### E.3 Blinded Producer / Musician Listening Study (`human_eval_prep.py`, per [`text2midi`](https://github.com/amaai-lab/text2midi))
Generate a randomized, double-blind MUSHRA-style evaluation pack (50 diverse prompts $\times$ 4 anonymized systems: *Ground Truth*, *`MIDI-LLM_Llama-3.2-1B`*, *`midi-llm-cpm` SFT*, *`midi-llm-cpm` + Plan + CFG + DPO*) rated on 5-point Likert / 100-point scales for:
1. **Overall Musical Quality**
2. **Text Prompt & Attribute Faithfulness**
3. **Harmonic & Voice-Leading Coherence**
4. **Rhythmic Groove & Expressive Naturalness**

#### E.4 Required Ablation Table (Published in `README.md`)
Every release checkpoint must populate the following ablation matrix on the 896-ID test set:
1. `Base SFT (REMI-Basic, No Plan, No Microtiming, No CFG, 2k)`
2. `+ Per-Note Microtiming & Continuous Controls (CC1/11/64, PitchBend)`
3. `+ Explicit Musical Plan (Form -> Beat Chords -> Instrumentation)`
4. `+ 12x Work-Deduplicated Pitch-Transposition Augmentation`
5. `+ Classifier-Free Guidance (LLM-CFG, gamma=1.8)`
6. `+ Progressive 2,048 -> 4,096 Context Curriculum`
7. `+ Calibrated Offline DPO (vs. Experimental Online GRPO)`

---

## 5. OPEN-SOURCE PRODUCT REQUIREMENTS, TOOLCHAIN & QUALITY GATES

### 5.1 Open-Source Deliverables & Community Standards
1. **Code & Weight Licensing (`LICENSE`, `MODEL_AND_DATA_LICENSES.md`):**
   * Source code licensed under **Apache-2.0**.
   * Clear documentation of downstream weight licenses (`MiniCPM5` / `Qwen3.5` / `Qwen2.5` Apache-2.0 vs. `Llama-3.2` Community License) and dataset tracks (`MidiCaps`-only CC-BY-SA-4.0 track vs. `+GigaMIDI v2.0.0` CC-BY-NC-4.0 research track).
2. **Model Cards, Dataset Cards & Copyright/Safety Statement:**
   * Publish standardized Hugging Face Model Cards, frozen split manifests, and an ethical copyright/attribution statement (including training-data memorization checks ensuring generated outputs do not regurgitate $> 4$-bar verbatim n-grams from training songs).
3. **Small Downloadable Demo Checkpoint & CPU-Only Install:**
   * Provide a lightweight demo checkpoint and a zero-CUDA installation path (`uv sync --extra cpu`) so users on laptops can run tokenization, explicit plan inspection, and DAW `.mid` export without a GPU.
4. **DAW Import Regression Fixtures (`tests/daw_fixtures/`):**
   * Maintain golden SMF Type 1 `.mid` files verified to load with correct Track Names, Tempo Maps, Time Signatures, Channel 10 Drum maps, `CC64` sustain pedals, and zero hanging notes in **Ableton Live, Logic Pro, FL Studio, and Reaper**.

### 5.2 Developer Toolchain (`uv`, `ruff`, `pytest`, `coverage`)
All dependency resolution, linting, and testing are managed deterministically via **`uv`** (`Python 3.11`):
```powershell
# 1. Zero-warning lint & format check
uv run ruff check .
uv run ruff format --check .

# 2. Full automated test suite (>= 90% line & branch coverage on core modules)
uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=90
```
