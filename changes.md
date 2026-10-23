# Changes from the original SpeechQM-Agent

Baseline: the original repository at commit `226bf45` ("ACL 2026 Paper"), which ran on a Linux server with an
NVIDIA GPU, used OpenAI `gpt-4.1-mini` (and Groq for benchmarking) as its LLM, transcribed Hindi only, and was
started by writing a prompt into `my_secrets.py`.

This edition runs fully locally on Mac, Windows and Linux, can be set up and run by someone else with two commands
and their own Hugging Face token, and supports 15 languages. The changes are grouped below by purpose; file names
point to where each change lives. 

---

## 1. Local models instead of cloud APIs

| Role | Before | Now | Why |
|---|---|---|---|
| Planner (task selection, prompt checker, DAG), domain classifier, LLM-as-a-judge | `gpt-4.1-mini` via OpenAI API key | `qwen3:8b` via Ollama (`config.PLANNER_MODEL`) | No API keys or per-call cost; best of qwen2.5 7B / 14B / qwen3 8B on task selection in our tests |
| Code-writing agent | same OpenAI model in a LangChain ReAct agent with a Python REPL tool | `qwen3:8b` (`config.CODER_MODEL`) in a write → run → repair loop | Small local models did not follow the ReAct format; qwen3:8b solved 9/9 code tasks vs 0/9 for qwen2.5-coder:7b |
| Transliteration | AI4Bharat IndicXlit (needs fairseq) | IndicXlit if installed, otherwise the local LLM | fairseq has no Apple Silicon / Windows build |

- `llm.py`: `ChatOllama` via `local_llm()`; qwen3's "thinking" mode is switched off (`reasoning=False`) so replies
  are just the answer.
- API keys in `my_secrets.py` are no longer used; the file now only holds the optional prompt and is optional itself.

## 2. Runs on Mac, Windows and Linux, CPU or GPU

- `config.torch_device()`: uses CUDA, else Apple MPS, else CPU (the original assumed CUDA).
- `utility_functions.py`: one lock (`MODEL_LOCK`) around all model loading and inference. LangGraph runs independent
  checks in parallel threads; concurrent calls into the VAD and MPS models corrupted memory and crashed the process
  (SIGTRAP) on the Mac.
- Long files: ASR runs on ≤ 8 s chunks built from detected speech; CTC emissions are computed in 30 s pieces. The
  original passed whole files to wav2vec2, which runs out of memory on long recordings.
- `.mp3` files are decoded directly (librosa) instead of converting to a temporary `.wav` written into the audio
  folder; parallel checks saw those temporary files and counted them.
- Windows: Python runs in UTF-8 mode (Indic text in files and console), paths given to the code-writing LLM use
  forward slashes (`C:\Users` inside a Python string is a syntax error), trailing `\` is handled like `/`.
- FFmpeg is not required (pyannote receives already-loaded audio).

## 3. One-command setup and run for other people

New files:

| File | Purpose |
|---|---|
| `setup.sh` / `setup.ps1` | one-time setup on Mac/Linux and Windows: uv, Python 3.11 `.venv`, packages (CUDA PyTorch when an NVIDIA GPU is present via `--torch-backend=auto`), Ollama install |
| `run.sh` / `run.bat` | start a run from any directory |
| `run_qc.py` | the run itself: sort the folder by language → ask for / validate / save the Hugging Face token → check access to gated models and fall back where missing → start Ollama and pull the LLM if needed → download models → run all checks per language → write `qc_summary.csv` |
| `prepare_runs.py` | splits a nested dataset (`<anything>/<language>/<audio + transcript json>`) into one folder per language, copies the audio, converts transcript JSONs to the reference CSV |
| `summarize.py` | merges every check's CSV into `qc_summary.csv`, one row per audio file |
| `download_models.py` | rewritten as reusable functions (`ensure_ollama`, `ensure_hf`, `ensure_indiclid`, `has_access`) |
| `requirements.txt` | cross-platform pinned packages (replaces the Linux-only conda `env/denv.yml` for this edition) |
| `config.py` | all model names, thresholds and paths in one place; each overridable by an env var |

- `llm.main()` accepts the audio folder and language directly (`main(prompt, audio_dir=..., lang_code=...)`);
  the original parsed a POSIX path out of the prompt, which cannot work for Windows paths.
- The user's audio folder is never modified: files are copied to `<audio>_qc/<language>/audios`.

- Offline runs: when huggingface.co is unreachable (or `HF_HUB_OFFLINE=1`), the token check is skipped and the
  models already in the local cache are used.
- `prepare_runs.py` keeps the copies in sync with the source folder (changed files re-copied, deleted files
  removed, stale references cleared), skips macOS `._*` metadata files, and matches a transcript JSON to its audio
  by exact name or by the audio name as a whole token (`sample_1` no longer also matches `sample_10_x.json`).
- IndicLID is downloaded with progress into a temporary folder and moved into place only when complete.
- `run.sh` keeps a Mac awake during the run (`caffeinate -i`): a full run with the Mac idle-sleeping 28 times took
  over 6 hours instead of about 2.
- Git: `setup.sh` / `run.sh` are stored executable; `.gitattributes` keeps `.sh`/`.py` LF and `.bat`/`.ps1` CRLF
  regardless of the user's `core.autocrlf`.

## 4. Gated models and fallbacks

| Model | Before | Now |
|---|---|---|
| `pyannote/segmentation-3.0`, `speaker-diarization-3.1` | required, token hard-coded in `my_secrets.py` | used when the user's token has access (pyannote.audio 4, which also needs `speaker-diarization-community-1`); otherwise Silero VAD + SpeechBrain ECAPA clustering (`SPEECH_BACKEND=local`) |
| `ai4bharat/indicwav2vec-hindi` | required | used when accessible; otherwise `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` |
| Token | `HF_TOKEN` in `my_secrets.py` | asked once (hidden input), validated with `whoami`, saved by `huggingface_hub`; or `--token`, `HF_TOKEN`, `--no-token` |

`run_qc.py` lists the gated pages the token cannot use yet, lets the user accept them and re-check, or continue
with the fallbacks.

## 5. More languages

- ASR was fixed to Hindi (`transcribe_folder_to_csv(..., "Hindi")`). It now follows the run's language, chosen
  from the folder name, `--language`, or a language name/code in the prompt (`llm.parse_prompt`, `language_of`).
- `config.ASR_MODELS` lists candidates per language, tried in order (`utility_functions.asr_model_for`):
  Hindi, Bengali, Gujarati, Odia (ai4bharat first, Vakyansh fallback); Tamil, Telugu, Marathi, Kannada, Malayalam,
  English, Punjabi, Urdu, Assamese, Nepali (Vakyansh); Sanskrit (unchanged).
  `ai4bharat/indicwav2vec-marathi` and `-telugu` (in the original list) do not exist on Hugging Face and the
  `indicwav2vec_v1_marathi/_telugu/_tamil` repos contain only a README, so Vakyansh is used for those.
- CTC alignment uses Vakyansh Hindi for Hindi (as before) and the language's own ASR model otherwise
  (`align_model_for`); the Hindi model cannot align other scripts. This applies to both the CTC score (task 10)
  and forced alignment (task 14).
- The script check (task 6) checks the language's script (`config.SCRIPTS`: Devanagari, Bengali, Gujarati, Odia,
  Tamil, Telugu, Kannada, Malayalam, Gurmukhi, Arabic, Latin) instead of always Devanagari.
- Domain and LLM-score prompts name the run's language instead of always Hindi; the domain list gained
  Healthcare, Insurance, Finance and Sales.

## 6. Correctness fixes

| Check | Problem | Fix |
|---|---|---|
| Script check (6) | LLM-written code reported a fully Tamil transcript as not Tamil | direct code; also outputs the share of letters in the script |
| Vocabulary (5) | LLM-written code returned empty lists for Indic text | direct code |
| English word count (9) | the LLM counted 145 English words where the reference had 335 | count of Latin-script words (tags excluded) |
| Text handling (5, 6, 9, 24) | Python's `\w` / `isalpha` treat Indic vowel signs as non-letters, splitting words | shared `words()` helper keeps combining marks |
| WER (24) | paired reference and ASR rows by position; failed when only some files had a reference | pairs by filename; strips punctuation and `[tags]` on both sides |
| New/old speaker (12) | pyannote labels speakers per file (SPEAKER_00 in every file), so "same label" ≠ "same person"; on the full dataset the LLM-written script for it also crashed (Hindi, Kannada) | speaker IDs made global across the folder by matching speaker embeddings (cosine ≥ 0.6); the check itself is now direct code |
| Code-writing agent | no check that the generated CSV is complete (a run produced a speaker table with only one of three files) | output row count validated against the input files/rows; mismatches go back to the LLM to fix (up to 3 attempts) |
| Task planning | small LLMs dropped requested checks, added unrequested ones and produced invalid task orders | checks named in the prompt are always added (`keyword_tasks`); dependencies enforced (`TASK_DEPS`); invalid orders replaced by a computed layering (`layer_tasks`); reference-only checks dropped when no reference exists |
| IndicLID | loaded from hard-coded server paths; BERT checkpoint failed to load under current PyTorch | models in `models/indiclid` (downloaded automatically); checkpoint loaded with its full pickle; works with `fasttext-predict` |
| State keys | several checks wrote to keys the graph did not declare, so their results were lost | `utterance_duplicate_output`, `wer_output`, `transcript_quality_output` declared |
| Ground-truth CTC (10) | scored the ASR hypothesis even when a reference existed | prefers the reference CSV |
| Domain (13) | output contained a "Domain:" prefix; on a long reference the LLM wrote an essay instead of a label | stripped; reply capped at 30 tokens and must name a listed domain, else `Unknown` |
| Upsampling (11) | audio was resampled to 16 kHz before the test, so every native 8 kHz file was reported upsampled | measured at the file's own rate; only files stored above 8 kHz can be upsampled |
| Transcript quality (3) | "unique words < 70 %" failed every long conversation (vocabulary repeats naturally) and missed ASR that dropped most of the speech | fails on ≤ 3 words, looping (> 20 % repeated 4-word phrases) or < 5 letters per second of speech |
| CTC score (10) | with a reference CSV, files without a reference got no score | per file: reference if present, else ASR (`Transcript_Source`) |
| Transliteration (18) | never received the reference CSV from `run_qc.py`; asking the LLM to rewrite whole transcripts did not finish (> 14 min); English and Odia rejected | reads `results/indicconf_hypothesis-gt.csv`; LLM maps only Roman words (JSON, validated script); all Indic languages; English skipped |
| Word splitting (5, 6, 9, 24) | contractions split in two ("don't" → "don", "t"), overcounting English words by ~4 % | apostrophes/hyphens inside a word keep it one word |
| Code-writing agent | qwen3:8b mis-copied a long path (two characters dropped) in all 3 attempts | `fix_paths` replaces near-miss path literals with the real path |
| Prompt checker | added unrequested tasks (29 % of planned actions on an 18-query benchmark) | its additions were ignored (6 %). **Reverted 2026-10-22** on request: its verified list is the plan again, as in the paper; additions are recorded in `results/plan.json` |
| LLM score (17) | the judge gave ASR transcripts and the same words shuffled the same score | a fluency gate was added. **Reverted 2026-10-22** on request (every transcript scored, as in the paper); per-chunk scores were added instead (section 9) |
| Transliteration (18) | one LLM answer per word could be wrong but plausible | converted only when two differently worded LLM runs agree and, when available, IndicXlit lists the same spelling in its top 5 (`docker/indicxlit` for Mac/Windows); per-word decisions in `transliteration_words.csv` |

## 7. Unchanged on purpose

- The 24 checks, their numbering and the LangGraph orchestration (planner → prompt checker → DAG → parallel execution).
  Two checks were added as 25 and 26 (section 9). The default run (`run_qc.py` without `--prompt`) requests all 26
  (18 needs reference transcripts and an Indic language; 9, 16, 24, 25, 26 need reference transcripts).
- SpeechQM's file rules: only `.wav` passes the extension check and `.mp3` is reported as corrupt. The sample rate
  is reported, not judged (2026-10-22).
- The LLM writes the code of 7 checks at run time (corruption, extension, sample rate, audio length, character list,
  normalization, alignment); its outputs are validated, not replaced by fixed code.
- Output CSV names and columns of each check (some gained columns, none lost).
- `llm.py` still works as the original prompt-driven entry point (`PROMPT` in `my_secrets.py`).

## 8. Other edits

- `README.md` rewritten as a user guide; this file and `implementation.md` added.
- `.gitignore`: `.venv/`, `models/`, `pipeline.log`, `__pycache__/`, `log.txt`, `*_qc/`; `.gitattributes` for line endings.
- `env/denv.yml`: line endings only (CRLF → LF); content unchanged.
- `my_secrets.py`: example prompt updated; no keys stored.

## 9. 2026-10-22: Azure VM, new checks and the gaps closed

Measured before keeping: every change to a prompt or model below was compared on a benchmark in `../qc_audit/`
(outside the repository). The numbers are from those benchmarks.

| Area | Change | Evidence |
|---|---|---|
| Settings | `.env` (git-ignored) read by `config.load_dotenv()`: `HF_TOKEN`, models. `JUDGE_MODEL`, `MONITOR_MODEL`, `CODER_MAX_TOKENS`, `ASR_ENGINE`, `SPEAKER_DB`, `PUBLIC_INDEX`, `REPLAN` | |
| Code-writing agent | every generated CSV checked per task (exact file list, no blanks, values that match the files); the output is deleted when all tries fail; reply length capped | all 7 generated checks pass within 3 tries |
| Tool library | scripts that passed their checks are kept in `tool_library/` and run only when every new attempt fails (the LLM still writes code on every run) | |
| Monitoring LLM (paper §2.4) | reviews every step's status and output with computed facts; LLM-written steps re-run once on RETRY. Missing files, repeated rows and empty columns in LLM-written outputs are decided without the LLM | broken outputs caught 49/64 → 64/64, false alarms 3/246 both (`monitor_bench.py`) |
| Re-planning | steps that failed or that the monitor wanted re-run (RETRY) go back to the planner, which picks those to run once more (`results/replan_log.csv`) | |
| Plan record | `results/plan.json`: what the planner selected, what the checker, keywords and dependencies added, what was skipped | |
| ASR | IndicConformer-600M (RNNT) for 14 Indic languages, Whisper-large-v3 for English; wav2vec2 kept as second opinion (`ASR_Disagreement`), for CTC and alignment. Chunks RNNT leaves empty get the CTC decoder's text | pooled WER 0.857 → 0.659 → 0.635 (chunk fallback; Gujarati 0.88 → 0.83) |
| Audio language ID (15) | Whisper-large-v3 and MMS-LID-126 on 30 s of speech, the more confident wins, next to IndicLID on the text | 172/180 labelled public clips (Whisper alone 154, MMS 165); 19/22 files of our data |
| Upsampling (11) | effective bandwidth → estimated original rate (any standard rate, not only 8 kHz) | synthetic 8→16, 11.025→16, 16→22.05/44.1/48 kHz all correct |
| Speakers (2, 12, 22) | speech time and overlap per file; speakers remembered across batches (`speaker_db/`); re-running a file replaces its entries | |
| Domain (13) | label definitions for the confusable labels, reference text when present, text capped at 3000 characters with the instruction repeated after it (a long Kannada text pushed the instructions out of the context) | 45/66 → 60/66 (`domain_bench.py`, 22 files × 3) |
| LLM judge (17) | per-file score as in the paper, plus 0-10 scores for up to 30 ASR chunks per file | Spearman with file WER: per-file score +0.10, chunk score −0.58 (`judge_bench.py`) |
| Duplicates (23) | near-duplicate utterances (character 3-gram TF-IDF), shared audio (landmark fingerprints: re-encoded, noisy, trimmed and embedded copies), public corpora (`build_corpus_index.py`); runs without references too | no false audio matches among the 22 files; all 6 altered copies found |
| WER (24) | CER, second ASR's WER, and WER per reference utterance (with and without cross-talk) | |
| Timestamp segmentation (25, new) | each reference utterance cut out and aligned; time-stamp issues (inverted, beyond the audio, same-speaker overlap, > 30 s) | |
| Code-mixing (26, new) | script per word, Code-Mixing Index per utterance and file, other scripts in the reference | found Cyrillic look-alike letters in two references |
| Dashboard | `dashboard.html` next to `qc_summary.csv` | |
| Environment | torch pinned to 2.11.0, the version of the last torchaudio (2.14 ran with the CPU build of torchaudio); `onnxruntime-gpu`, `regex` | |

## 10. 2026-10-23: MoE score (several ASR models per file)

| Area | Change | Evidence |
|---|---|---|
| ASR experts (1) | every file is also transcribed by `bodhan-ai/indic-transcribe-flex` (all 14 Indic languages and English), `nvidia/nemotron-speech-streaming-en-0.6b` (English; in `.venv-nemotron` via `nemotron_worker.py`, it needs transformers ≥ 5.13) and the wav2vec2 model; `asr_experts.csv`, `asr_moe.csv` replace `asr_second_opinion.csv` (`ASR_Disagreement` in §9 is superseded). `MOE_EXPERTS`, `MOE_MAX_CER` | added without an ASR benchmark (user decision); WER on the 11 reference files: Bodhan 0.624, main 0.671, Nemotron 0.620 (en001), wav2vec2 0.867 |
| Transcript quality (3) | also fails when even the closest expert differs from the main transcript on > 50 % of the characters | 5 new failures in `audios_qc_moe/`, each a real problem (English audio, dropped speech, suspected wrong language); `Main_Agreement_CER` vs WER Spearman 0.66 |
| WER (24) | WER and CER of every expert (`WER_bodhan`, `WER_nemotron`, `WER_wav2vec2`) | |
| MoE score | pairwise differences capped at 1 (a model that drops most speech gave 3.5) | |
| Setup | `requirements-nemotron.txt`; `setup.sh` / `setup.ps1` build `.venv-nemotron`; `download_models.py` fetches the experts | |
| Main ASR (2026-10-23, user decision) | Bodhan Indic-Transcribe-Flex is the main ASR for the 14 Indic languages and English (`ASR_ENGINE=bodhan`; falls back to IndicConformer / Whisper when it cannot load; `ASR_ENGINE=best` for the old main ASR). IndicConformer and Whisper become MoE experts | full run `audios_qc_bodhan/`: WER 0.671 → 0.625 on 11 reference files (9 better), no empty chunks, task 3 passes 17/22 |
| LLM judge chunks (17) | `asr_chunks.csv` keeps 8 s pieces (`JUDGE_CHUNK_S`): Bodhan transcribes 25 s pieces for the transcript and the 8 s pieces for the judge | `judge_bench.py` on Bodhan: chunk score vs file WER Spearman −0.56 with 8 s pieces, 0.00 with 25 s |
| MoE verdict (3) | `MoE_Verdict`: `no agreement` (closest pair of models > 50 % CER apart) or `main differs` (two experts agree, the main transcript is far from both) fail; one weak expert disagreeing with everyone no longer fails the main transcript | same 5 failures on `audios_qc_bodhan/`, now with the right reason (e.g. gj001, kn001: no agreement, not a bad main transcript) |

