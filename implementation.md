# Implementation

How this edition of SpeechQM-Agent works inside: the run flow, how a prompt becomes a graph of checks, how each
check is implemented, the speech models, and how to extend it. For usage see `README.md`; for differences from the
original code see `changes.md`.

---

## 1. Files

| File | Role |
|---|---|
| `run_qc.py` | Entry point for users. Orchestrates sorting, token, model access, Ollama, downloads, per-language runs, summary |
| `run.sh`, `run.bat` | Start `run_qc.py` with the project's Python in UTF-8 mode from any directory |
| `setup.sh`, `setup.ps1` | One-time install: uv, Python 3.11 `.venv`, `requirements.txt` (PyTorch build picked by `--torch-backend=auto`), Ollama |
| `config.py` | Every model name, threshold and language table; values overridable by env vars |
| `prepare_runs.py` | Sorts a dataset into `<out>/<language>/audios` and converts transcript JSONs to `results/indicconf_hypothesis-gt.csv` |
| `llm.py` | The agent system: prompt parsing, task selection, prompt checker, DAG construction, the 24 check nodes, LangGraph execution, the code-writing agent |
| `utility_functions.py` | Speech and text tooling: audio loading, VAD, ASR, diarization, global speaker IDs, CTC forced alignment, upsampling detection, IndicLID, transliteration |
| `download_models.py` | Model access checks and downloads (Ollama pull over HTTP, Hugging Face snapshots, IndicLID from GitHub) |
| `summarize.py` | Merges the per-check CSVs into `qc_summary.csv` |
| `my_secrets.py` | Optional `PROMPT` for the original `python llm.py` entry point (no keys) |
| `models/indiclid/` | IndicLID models (downloaded on first run, ~1.5 GB) |
| `pipeline.log` | Detailed log of every run, including every script the code-writing agent wrote and its output |

## 2. Run flow (`run_qc.py`)

```
run.sh / run.bat <audio> [--language X] [--prompt "..."]
 │
 ├─ 1 prepare_runs.main(audio, out, language)      → {language: <out>/<language>}
 │     audio found recursively; language = nearest parent folder named after a language, else --language
 │     transcript JSON in the same folder whose stem contains the audio stem → reference row
 │
 ├─ 2 offline (huggingface.co unreachable or HF_HUB_OFFLINE=1)? → HF_HUB_OFFLINE=1, use cached models only,
 │     pyannote if it is in the cache, else the local backend
 │   token: --token | HF_TOKEN | saved login | ask (getpass) → whoami() → login() saves it
 │     check_gated(): auth_check() on pyannote repos + gated ASR candidates of the languages found
 │       pyannote blocked → SPEECH_BACKEND = "local"; ai4bharat blocked → its Vakyansh fallback is used later
 │
 ├─ 3 Ollama: GET /api/tags; not running → start it (Mac app / `ollama serve`); model missing → POST /api/pull
 │
 ├─ 4 ensure_hf(languages, backend): first accessible ASR model per language, CTC model, IndicBERT tokenizer,
 │     VAD + diarization models (by loading them once); ensure_indiclid()
 │
 ├─ 5 for each language:
 │     clear old outputs (transcription is skipped if indicconf_hypothesis.csv exists)
 │     prompt = default checks (+ reference checks if indicconf_hypothesis-gt.csv exists) or --prompt
 │     llm.main(prompt, audio_dir=<out>/<language>/audios, lang_code=<ISO code>)
 │     a failure is reported and the next language runs
 │
 └─ summarize.write_summary(out, runs) → <out>/qc_summary.csv
```

`config.SPEECH_BACKEND` is read when `config` is first imported (step 1), so `run_qc.py` sets both the env var and
the module attribute before `utility_functions` is imported in step 4.

On Windows `run_qc.py` re-executes itself with `-X utf8` if UTF-8 mode is off; otherwise reading/writing Indic
text and printing transcripts would fail under the cp1252 code page.

## 3. From prompt to graph (`llm.main`)

```
prompt ──► select_tasks (LLM) ──► prompt_checker_agent (LLM, ≤ 3 rounds) ──► task set
                                                                               │
          checker may only remove: task set = selection ∩ checker result
          keyword_tasks: regexes for each check's name add checks the LLM missed
          add_dependencies: TASK_DEPS closure (e.g. WER needs normalization needs ASR)
          REFERENCE_TASKS {9,16,18,23,24} dropped when no reference CSV exists
                                                                               │
          topological_sort_tasks (LLM) → layers; structure_is_valid? else layer_tasks (Kahn layering)
                                                                               │
          build_graph_from_structure → LangGraph StateGraph: START → layer 1 nodes (parallel) → layer 2 → … → END
          graph.invoke({audio_dir, ground_truth_csv, lang_code, user_prompt})
```

- **Planner prompts** list the 24 checks with keyword rules; the prompt checker compares the selection with the
  user's prompt and returns `Correct` / `Missing` / `Extra` until consistent.
- The small local LLM is not reliable at this, so the deterministic steps after it guarantee that (a) checks named
  in the prompt run, (b) dependencies run first, (c) the order is valid. Tasks the prompt checker *adds* are
  ignored: on an 18-query benchmark (the paper's hallucination rate: planned actions the query did not request)
  29 % of the planned actions were unrequested when its additions were kept, 6 % with selection ∩ checker, with no
  requested task missed either way.
- **State** (`CombinedStateDict`): inputs plus one output key per check (`A` = transcript CSV path, `D` = silence,
  `E` = speakers, `<name>_output` for the others). Nodes return only their own key, so parallel nodes never write
  the same key.
- **Language**: `language_of(state)` maps `lang_code` through `config.LANG_CODES`; default `DEFAULT_LANGUAGE`
  (Hindi). In the prompt-only mode, `parse_prompt` takes a language name (English only if not followed by "word")
  or ISO code from the prompt.

### Task dependencies (`TASK_DEPS`)

| Check | Needs |
|---|---|
| 3 | 1 (ASR transcript), 8 (silence, for the speech duration) |
| 4, 5, 6, 10, 13, 14, 15, 16, 17 | 1 (ASR transcript) |
| 12, 22 | 2 (speaker table) |
| 9, 23, 24 | 16 (normalized reference) |

## 4. The 24 checks

Implementation types: **model** = fixed code around a speech/text model; **code** = fixed Python;
**agent** = the code-writing agent (section 5) writes and runs a script; **judge** = the LLM answers per transcript.

| # | Check | Output (`results/`) | Type | Notes |
|---|---|---|---|---|
| 1 | ASR transcription | `indicconf_hypothesis.csv` | model | wav2vec2 CTC per language (section 6.2) |
| 2 | Speakers and durations | `num_speakers.csv` | model | pyannote or local diarization, global speaker IDs (6.3) |
| 3 | Transcript quality | `transcript_quality.csv` | code | fails: ≤ 3 words, > 20 % repeated 4-word phrases (loops), or < 5 letters per second of speech (ASR dropped speech) |
| 4 | Characters (graphemes) | `character_list.csv` | agent | |
| 5 | Vocabulary | `vocab_list.csv` | code | unique words, tags removed |
| 6 | Script / language verification | `language_verification.csv` | code | share of letters in the language's Unicode block (`config.SCRIPTS`); OK = 100 % |
| 7 | Audio length | `audio_length.csv` | agent | |
| 8 | Silence | `vad_silence_stats.csv` | model | duration − detected speech |
| 9 | English word count | `english_word_count.csv` | code | Latin-script words in the reference |
| 10 | CTC score | `ctc_scores.csv` | model | forced alignment, per file of the reference if it has one, else the ASR transcript (`Transcript_Source`) (6.4) |
| 11 | Upsampling | `upsampling_check.csv` | code | stored above 8 kHz with energy 4–8 kHz < 2 % of total, at the file's own rate; native 8 kHz files are not upsampled |
| 12 | New / old speaker | `valid_speaker.csv` | code | `Old` when any of the file's global speaker IDs (`num_speakers.csv`) appears in another file, listing those files |
| 13 | Domain | `domain_check.csv` | judge | one label per file; reply capped at 30 tokens and must name a listed domain, else `Unknown` |
| 14 | Forced alignment | `audio_transcript_matching.csv` | agent | script calls `force_alignment_and_ctc_score` with the language's model |
| 15 | Language identification | `indiclid_language_identification.csv` | model | IndicLID (6.5) |
| 16 | Normalization (remove tags) | `normalized_list.csv` | agent | removes HTML/XML-style tags from the reference |
| 17 | LLM score | `llm_scores.csv` | judge | 0–10 coherence/fluency + comment |
| 18 | Transliteration | `*_transliterated.csv` | model/judge | reference only; IndicXlit if installed, else the LLM maps the Roman words (JSON, batches of 40, answers outside the target script rejected) and only those words are replaced; `roman_words_left`; skipped for English |
| 19 | Corruption | `corruption_check.csv` | agent | empty, unreadable by soundfile, or `.mp3` → corrupt (SpeechQM rule) |
| 20 | File extension | `audio_format_check.csv` | agent | `.wav` valid |
| 21 | Sample rate | `sample_rate_check.csv` | agent | 16 kHz → Pass |
| 22 | Speaker durations (dataset) | `speaker_durations.csv` | code | total hours per global speaker |
| 23 | Duplicate utterances | `duplicate_utterances.csv` | code | identical normalized references |
| 24 | WER | `wer.csv` | code | `jiwer` per file paired by filename, punctuation and tags removed |

Text checks share `llm.words()`: removes `[...]` / `<...>` tags and splits on Unicode punctuation, symbols,
separators and controls, keeping combining marks. Python's `\w` and `str.isalpha()` treat Indic vowel signs
(category M) as non-letters, which is why model-written code got 5, 6 and 9 wrong and they are now fixed code.

## 5. The code-writing agent (`run_agent`)

```
task prompt + describe_inputs() (real CSV columns and first row of every CSV the prompt mentions)
   │
   ▼  up to 3 attempts, serialized by AGENT_LOCK
coder LLM ─► ```python``` block ─► extract_code ─► fix_paths ─► run_code (fresh namespace, stdout captured,
   ▲                                                 SystemExit caught, short traceback on error)
   │                                                           │
   └── "script failed / file missing / N rows but must be M" ◄─┘  output CSV validated:
                                                                    exists, non-empty, row count = expected
```

- **Expected rows**: one per file for file checks (19, 20 all files; 7, 21 audio files), one per input row for
  transcript checks. A script that silently skips files is sent back.
- On Windows, backslashes in the prompt become `/` so paths survive being pasted into Python string literals.
- Every script and its output is in `pipeline.log`.
- Generated code runs in-process with the user's permissions; it is only ever given the task prompts in `llm.py`.
- `fix_paths`: the 8B model mis-copies long paths (seen: `…fd5b0e03f1` for `…fd5b0e0303f1` in a 120-character
  path, three attempts in a row). Any path literal in the script that is ≥ 90 % similar to, but not exactly, a
  path given in the prompt is replaced by that path (logged as `fixed a miscopied path`).

## 6. Speech and language models (`utility_functions.py`)

### 6.1 Audio, device, concurrency

- `load_audio`: librosa, 16 kHz mono float32, any format libsndfile/audioread reads (`.wav`, `.mp3`, `.flac`).
- `DEVICE = config.torch_device()`: `cuda` → `mps` → `cpu`.
- `MODEL_LOCK` (re-entrant) wraps every model load and forward pass. LangGraph runs a layer's nodes in threads;
  concurrent calls into TorchScript VAD / MPS models corrupted the heap. Models are cached with `lru_cache`.

### 6.2 VAD and ASR

- `speech_regions(audio)`: pyannote `VoiceActivityDetection` on `segmentation-3.0` (audio passed as an in-memory
  `{"waveform", "sample_rate"}` dict, so FFmpeg/torchcodec is not needed), or Silero VAD for the local backend.
- `asr_chunks`: speech regions merged into chunks of ≤ `ASR_CHUNK_S` = 8 s (8 s gave the lowest WER of 30 / 8 / 5 s
  on Hindi and Telugu; Vakyansh models were trained on short utterances and emit few words on long windows).
- `asr_model_for(language)`: first candidate in `config.ASR_MODELS[language]` that loads (gated ai4bharat models
  raise until their terms are accepted), cached per language. `transcribe_audio` runs greedy CTC decoding per chunk.

### 6.3 Diarization and global speaker IDs

- **pyannote backend**: `speaker-diarization-3.1` (pyannote.audio 4) on the in-memory waveform; durations per
  speaker plus one embedding per speaker.
- **local backend**: Silero speech regions → 3 s windows every 1.5 s → SpeechBrain ECAPA embeddings → agglomerative
  clustering (cosine distance 0.7) → small clusters folded into their nearest neighbour.
- `save_num_speakers`: per-file speakers are matched to a folder-wide registry of speaker centroids; cosine
  similarity ≥ `SPEAKER_MATCH_THRESHOLD` (0.6) means the same person and reuses the ID, otherwise a new
  `SPEAKER_xx` is created. On test data the same voice across two files scored 0.76 and different people ≤ 0.21.
  Output: `File Name, Number of Speakers, Speaker Durations` (JSON, hours).

### 6.4 CTC score / forced alignment

`force_alignment_and_ctc_score(audio, transcript, model)`: log-softmax emissions in 30 s pieces, transcript
tokenized with the model's vocabulary (`|` = word boundary), trellis + backtracking (torchaudio tutorial
algorithm), token segments merged into words; the score is the mean frame probability along the path.
`align_model_for(language)`: Vakyansh Hindi for Hindi, the language's ASR model otherwise.

### 6.5 Language identification

IndicLID (AI4Bharat): FastText for native script (FTN) and romanized text (FTR); romanized input below the
confidence threshold goes to the IndicBERT classifier. Models in `models/indiclid/`; the BERT checkpoint is a
pickled full model, loaded with `weights_only=False` and the `ai4bharat/IndicBERTv2-MLM-only` tokenizer.
`fasttext-predict` (prediction-only build with wheels for every OS) replaces `fasttext`; texts are predicted one
at a time because its batch API differs.

### 6.6 LLM

`local_llm(model)` → `ChatOllama(base_url=OLLAMA_BASE_URL, num_ctx=8192, temperature=0)`, `reasoning=False` for
qwen3 so replies contain only the answer. The planner and coder are the same `qwen3:8b` by default.

## 7. Data contracts

```
<out>/
├── qc_summary.csv                         one row per audio file (utf-8-sig for Excel)
└── <language>/
    ├── audios/                            copies of the input audio (flat; file names unique per language)
    └── results/
        ├── indicconf_hypothesis-gt.csv    filename,ground_truth          (from transcript JSONs; optional)
        ├── indicconf_hypothesis.csv       Filename,Indiconformer_Hypothesis   (ASR)
        └── <one CSV per check>            see section 4
```

- Reference JSON: `{"segments": [{"text": str, "start_time": float, ...}]}`; segment texts are joined in time order.
  It belongs to the audio file in the same folder with the same name, else whose name contains the audio name as
  a whole token; several candidates → none is used (reported).
- `prepare_runs` re-syncs on every run: changed source files (size or modification time) are copied again,
  files no longer in the source are removed from `audios/`, the reference CSV is rebuilt; `._*` files are ignored.
- Results always go to `results/` next to the audio folder (`llm.results_dir`).
- `summarize.COLUMNS` maps each result file's columns (case-insensitive, `*` wildcards) to summary columns.

## 8. Configuration (`config.py`)

| Setting | Default | Env var |
|---|---|---|
| Planner / coder LLM | `qwen3:8b` | `PLANNER_MODEL`, `CODER_MODEL` |
| Ollama server | `http://localhost:11434` | `OLLAMA_BASE_URL` |
| LLM context | 8192 | `LLM_NUM_CTX` |
| Speech backend | `pyannote` | `SPEECH_BACKEND` (`local`) |
| Same-speaker threshold | 0.6 cosine | `SPEAKER_MATCH_THRESHOLD` |
| Language when none given | Hindi | `DEFAULT_LANGUAGE` |
| Hindi CTC model | Vakyansh hindi-him-4200 | `ALIGN_MODEL` |
| Local ECAPA model | `speechbrain/spkrec-ecapa-voxceleb` | `SPEAKER_MODEL` |
| Hugging Face token | saved login | `HF_TOKEN` |

`ASR_MODELS`, `LANG_CODES`, `SCRIPTS`, `PYANNOTE_REPOS` and the local-diarization window/threshold constants are
edited in the file.

## 9. Extending

**Add a language**: add `"Name": [candidate repos…]` to `ASR_MODELS` (wav2vec2 CTC models with a
`Wav2Vec2Processor`), its ISO code to `LANG_CODES`, its script's Unicode block to `SCRIPTS`, and spelling variants
of the folder name to `prepare_runs.ALIASES`.

**Add a check**:
1. write `def my_agent(state) -> {"my_output": ...}` in `llm.py` (fixed code, or `run_agent(prompt, output_path,
   expected_rows)` for a generated script);
2. add `my_output: str` to `CombinedStateDict`;
3. register `25: ("node_my_check", my_agent, "my_output")` in `node_map`, describe it in the task lists of
   `select_tasks` / `prompt_checker_agent`, add a keyword regex to `KEYWORD_TASKS`, its prerequisites to `TASK_DEPS`,
   the accepted task range in `main()` (`1 <= x <= 24`), and its inputs in `utility_functions.get_required_inputs`;
4. map its columns in `summarize.COLUMNS` to show it in `qc_summary.csv`.

## 10. Verification done

| Test | Result |
|---|---|
| Mac M3 Pro (MPS), real models, 3 Hindi test clips, QC1 + QC2 prompts | all requested checks produced CSVs; speakers matched the human annotation (e.g. 38.4 + 20.8 s vs 38.7 + 18.4 s) |
| Code-writing agent benchmark (9 tasks) | 9/9 with row-count validation |
| Full 24-file dataset sorted into 10 languages | 22 audio + 11 transcripts matched |
| Full check set on Tamil and Telugu folders (mp3 + wav, 8–48 kHz) | all checks completed; 2 Telugu calls correctly share a speaker |
| Clean Ubuntu 24.04 container (arm64, CPU only, no token): `setup.sh` then `run.sh` on Hindi + Telugu | completed: Vakyansh Hindi fallback, local speaker backend, IndicLID downloaded, summary written |
| Windows: dependency resolution for `x86_64-pc-windows-msvc` | all 150 packages resolve; every PyPI package has a prebuilt Windows wheel |
| Mac: `setup.sh` on the existing environment, full run, forced-offline run (`HF_HUB_OFFLINE=1`) | all completed; offline results identical |
| `prepare_runs` edge cases (hidden `._` file, `sample_1` vs `sample_10`, changed / deleted source files, removed transcripts) | all handled |
| `shellcheck` on `setup.sh` / `run.sh`; PowerShell 7.4 parser on `setup.ps1` | no findings / parses |
| Windows: actual run | **not tested** (no Windows machine available) |

## 11. Known limitations

- WER is high on code-mixed speech: references write English words in Latin script, the ASR in the native script.
- Vakyansh models for Malayalam (8 h), Assamese (8 h) and Punjabi (10 h) are weak; Telugu WER was ~0.93–0.96.
- pyannote sometimes over- or under-splits speakers (Telugu insurance call: 4 found vs 2 annotated; Tamil: second
  speaker 8 s vs 50 s).
- IndicLID labels noisy Hindi ASR output as Maithili now and then.
- The planner LLM may add checks that were not asked for (harmless; they just take time).
- `.mp3` is reported as corrupt and non-`.wav` as invalid by design of the SpeechQM spec.
