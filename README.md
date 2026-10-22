# SpeechQM-Agent (local edition)

Automatic quality checks for speech datasets (audio + transcripts) in Indian languages and English, running
entirely on your own computer: Mac, Windows or Linux. No OpenAI/Groq keys; the only account you need is a free
Hugging Face token (optional, but it gives better speaker and Hindi results).

Point it at a folder of recordings and you get one table with, per file: duration, corruption, sample rate,
upsampling (and from which rate), silence, number of speakers and their talk time, whether a speaker also appears in
another file or an earlier batch, the spoken language (from the audio and from the transcript), script check, ASR
transcript, word and character error rate against your reference transcript (per file and per utterance), alignment
(CTC) score, LLM-as-a-judge scores, the conversation's domain, duplicates (also near-duplicates, shared audio and
matches in public corpora), reference time-stamp problems, code-mixing and English words. A `dashboard.html` lists
the files that need attention.

All 26 checks run by default: the paper's 24 (Table 10) plus timestamp-based segmentation and code-mixing detection
(Table 1). A local LLM plans which checks to run from a plain-English request, writes the code for 7 of them at run
time, judges transcripts, labels domains and reviews every step's output (the monitoring LLM).

Based on *Rolling Out Data Quality Overnight, without losing the plot: A Multi-Agent System for Speech Data Quality
Management* (ACL 2026, `ACL2026_Rolling_Out_Data_Quality.pdf`). See `changes.md` for what differs from the original
code and `implementation.md` for how it works inside.

---

## 1. Requirements

| | Minimum |
|---|---|
| OS | macOS 13+ (Apple Silicon or Intel), Windows 10/11 (64-bit), Linux (x86_64 or arm64) |
| RAM | 32 GB (the 8B language model needs ~6-16 GB, the speech models ~10 GB) |
| Disk | ~40 GB free (first run downloads ~20 GB of models) |
| GPU | optional: NVIDIA (CUDA) or Apple Silicon is used automatically; CPU works, just slower |
| Internet | for setup and the first run only; later runs work offline |

You do not need to install Python yourself: setup downloads its own Python 3.11 into the project folder.

## 2. Setup (once)

Open a terminal in the `SpeechQM-Agent` folder.

**Mac / Linux**
```bash
./setup.sh
```

**Windows** (PowerShell)
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Setup installs [uv](https://docs.astral.sh/uv/) (Python package manager), creates `.venv` with Python 3.11 and all
packages (the CUDA build of PyTorch if an NVIDIA GPU is present), and installs [Ollama](https://ollama.com) (the
local LLM server) if it is missing. On a Mac without Homebrew, install the Ollama app from
https://ollama.com/download yourself.

## 3. Hugging Face token (recommended)

1. Sign up at https://huggingface.co and create a token at https://huggingface.co/settings/tokens (type **Read**).
2. While logged in, open each page below and click **Agree / Access repository** (access is granted instantly):
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual (the main ASR for the Indian languages)
   - https://huggingface.co/ai4bharat/indicwav2vec-hindi
   - only if you have such data: `ai4bharat/indicwav2vec_v1_bengali`, `ai4bharat/indicwav2vec_v1_gujarati`,
     `ai4bharat/indicwav2vec-odia`
3. The first run asks for the token (typing is hidden) and saves it; you are not asked again. You can also put it in
   a `.env` file in this folder (`HF_TOKEN=hf_...`; copy `.env.example`), which is never committed.

Without a token, or for models whose terms you did not accept, the run still works using ungated replacements
(see section 7) and tells you which ones it used.

## 4. Run

**Mac / Linux**
```bash
./run.sh /path/to/audio
```

**Windows**
```bat
run.bat C:\path\to\audio
```

The first run downloads the models it needs (you see progress), then checks every file. Expect roughly 2-3
minutes per 10 minutes of audio on a recent laptop GPU; CPU-only machines are slower.

### Options

| Option | Meaning |
|---|---|
| `--language Hindi` | language of audio that is not inside a language-named folder (see 5) |
| `--prompt "check sample rate and count speakers"` | run only the checks you describe in plain English (default: all) |
| `--out D:\qc_results` | where results go (default: `<audio folder>_qc` next to the audio folder) |
| `--token hf_xxx` | use this Hugging Face token (default: saved login, `HF_TOKEN` env var, or ask) |
| `--no-token` | do not use a token; ungated replacement models only |
| `--yes` | never ask anything (for scheduled/scripted runs) |

`./run.sh --help` (or `run.bat --help`) lists them.

## 5. Preparing your audio folder

Supported audio: `.wav`, `.mp3`, `.flac` (any sample rate, mono or stereo).

**One language:** put the files in one folder and pass the language:
```
./run.sh /data/calls --language Telugu
```

**Several languages:** put each language's files under a folder named after the language, at any depth:
```
audio/
├── healthcare/
│   ├── hindi/   hi001.wav, hi001_human_transcribed.json
│   └── telugu/  tel001.wav
└── sales/
    └── marathi/ sales_1.wav, sales_2.wav
```
Recognised folder names: Hindi, Bengali (Bangla), Gujarati (Gujrati), Marathi, Odia (Oriya), Tamil, Telugu, Kannada,
Malayalam (Malyalam), Punjabi, Urdu, Assamese, Nepali, Sanskrit, English. Upper/lower case does not matter.

**Reference transcripts (optional):** a `.json` file next to an audio file, whose name contains the audio's name
(`hi001_human_transcribed.json` or `Copy of hi001_human_transcribed.json` for `hi001.wav`), in this format:
```json
{"language_code": "hin",
 "segments": [{"text": "नमस्ते", "start_time": 0.1, "end_time": 1.2, "speaker": {"id": "speaker_0"}}]}
```
With references you also get WER and CER (per file and per time-stamped utterance), CTC against the reference, tag
normalization, English-word count, transliteration (Roman → native script), timestamp segmentation (each utterance
cut out, aligned and its time-stamps checked) and code-mixing detection. Without them these checks are skipped (CTC
then scores the ASR transcript, and the duplicate check compares the ASR chunks).

Your audio folder is never modified: files are copied into the results folder before processing.

## 6. Results

Everything is written to `<audio folder>_qc/` (or `--out`):

```
audio_qc/
├── dashboard.html            <- open in a browser: files that need attention, how each run went, all results
├── qc_summary.csv            <- one row per audio file, all checks (open in Excel / Google Sheets)
├── run_info.json             <- models, Hugging Face account, settings and run time of this run
├── hindi/
│   ├── audios/               copies of the Hindi audio
│   └── results/              one CSV per check (details: transcripts, aligned segments, ...), plus
│                             plan.json (which checks ran and why), monitor_log.csv (the monitoring LLM's verdict
│                             per step) and replan_log.csv (steps the planner chose to run again)
└── telugu/ ...
```

`qc_summary.csv` columns (a column is missing when its check did not run):

| Column | Meaning | Look out for |
|---|---|---|
| Language, Filename | run language and file | |
| Duration_s | length in seconds | |
| Corruption | `valid` / `corrupt` (unreadable, empty, or `.mp3`: the SpeechQM spec accepts WAV only) | `corrupt` |
| Valid_Extension | `.wav` | `False` |
| Sample_Rate, Channels | as stored in the file (reported, not judged) | |
| Upsampled, Bandwidth_Hz, Estimated_Original_Rate | the highest frequency with real content, the lowest standard rate that holds it, and `True` when that is below the stored rate (e.g. 8 kHz phone audio stored at 48 kHz) | `True` |
| Silence_s | seconds without speech | high share of Duration_s |
| Speakers, Speaker_Durations_h, Speech_s, Overlapping_Speech_s | number of speakers, talk time per speaker in hours (a speaker's time includes talking over someone, so the sum can exceed the file), seconds with speech, seconds with two or more speakers | unexpected count |
| Speaker_New_Old, Shares_Speaker_With, Speaker_In_Earlier_Batches | `Old` = a voice in this file also speaks in the listed file(s) of this batch or of an earlier run (`speaker_db/`) | `Old` where every speaker should be unique |
| Detected_Language, LID_Confidence | IndicLID language of the ASR transcript | not the folder's language |
| Audio_Language, Audio_LID_Confidence, Audio_Language_OK | language heard in the audio itself (Whisper-large-v3 and MMS-LID-126); Hindi and Urdu count as the same spoken language | `False`: audio in another language |
| Script_Share, Script_OK | share of letters in the language's script | `False` |
| ASR_Transcript_Quality | `passed`, or `failed:` with the reason (almost empty, looping, or far too little text for the speech: the ASR model dropped speech) | `failed` |
| ASR_Disagreement | word difference between the main ASR and a second ASR model; needs no reference, and tracks WER (Spearman 0.61 here) | > 1 |
| CTC_Score, CTC_Status, CTC_Scored_Text | how well the transcript aligns with the audio, 0-1; Good > 0.7, Medium > 0.5; the file's reference if it has one, else the ASR transcript | `Poor` |
| ASR_Alignment_Score | forced-alignment score of the ASR transcript (task 14) | |
| LLM_Score, LLM_Comment | 0-10 coherence/fluency of the whole ASR transcript, judged by an LLM (as in the paper) | low |
| LLM_Chunk_Score, Low_Score_Chunk_Share, Empty_ASR_Chunk_Share | the judge's mean 0-10 score over up to 30 short ASR chunks, the share it scores 4 or less, and the share of speech chunks the ASR left empty. Separates good from bad transcripts much better than LLM_Score | low score, high shares |
| Domain, Domain_From | topic (Healthcare, Insurance, Finance, Sales, Daily Life, ...), judged from the reference where there is one, else the ASR transcript | unexpected topic |
| WER, CER, WER_Second_ASR | error rates of the ASR vs the reference, and of the second ASR model | |
| Segment_WER, Segment_WER_No_Cross_Talk, Bad_Segment_Share | WER with every reference utterance transcribed on its own (from the time-stamps), without utterances that overlap another speaker, and the share of utterances with WER > 0.8 | |
| Near_Duplicate_Utterances, Repeated_Utterances_In_File, Shares_Audio_With, Public_Corpus_Matches | utterances (nearly) the same as one in another file / the same file, files sharing audio with this one (also re-encoded or cut copies), matches in the public-corpus index | any |
| English_Words_in_Reference | Latin-script words in the reference transcript | |
| Roman_Words_Left_After_Transliteration | Roman-script words not converted, because the two sources did not agree (see 7) | |
| Ref_Segments, Segments_With_Timing_Issues, Segment_Alignment_Median | reference utterances, how many have wrong time-stamps (end before start, beyond the audio, a speaker overlapping themselves, over 30 s), median alignment of utterance audio and text | issues > 0 |
| Code_Mixed_Utterance_Share, Mean_CMI, Embedded_Word_Share, Other_Scripts_In_Reference | code-mixing in the reference (Code-Mixing Index, Das & Gambäck 2014) and scripts that do not belong (e.g. Cyrillic look-alike letters) | other scripts |

**Reading WER:** the ASR writes everything in the language's own script, while many references write English words
in Latin script ("Hospital", "insurance"). Every such word counts as an error, so WER is high on code-mixed speech
even when recognition is good. Compare WER between files and vendors rather than against an absolute target.

**Reading the LLM scores:** both are judged from the ASR text alone. LLM_Score (one score for the whole transcript)
came out 8 for almost every file in our tests; LLM_Chunk_Score (the mean of scores for short pieces) follows the
real WER (Spearman -0.58 on 11 reference files) and is the one to use for ranking files.

## 7. Models used

| Purpose | Model (with token + accepted terms) | Without access |
|---|---|---|
| Planner, prompt checker, domain, monitor, code writing, transliteration | `qwen3:8b` via Ollama (`qwen3:8b-fp16` with the settings in `.env.example`) | same |
| LLM-as-a-judge | `llama3.3:70b-instruct-q8_0` via Ollama when `JUDGE_MODEL` is set (needs ~75 GB GPU memory), else the planner | |
| Speech/silence detection | `pyannote/segmentation-3.0` | Silero VAD |
| Speakers | `pyannote/speaker-diarization-3.1` | Silero VAD + SpeechBrain ECAPA clustering |
| ASR, 14 Indian languages | `ai4bharat/indic-conformer-600m-multilingual` (RNNT; a chunk RNNT leaves empty gets the CTC decoder's text) | the wav2vec2 model of the language |
| ASR, English | `openai/whisper-large-v3` | |
| Second ASR opinion, CTC score, forced alignment | wav2vec2: `ai4bharat/indicwav2vec-hindi` (Hindi), ai4bharat Bengali/Gujarati/Odia, else Vakyansh (`Harveenchadha/vakyansh-wav2vec2-<language>-...`); CTC on Hindi uses Vakyansh Hindi | Vakyansh model of the language |
| Language identification, text | AI4Bharat IndicLID (FastText + IndicBERT) | same |
| Language identification, audio | `openai/whisper-large-v3` and `facebook/mms-lid-126`, the more confident wins | same (not gated) |
| Transliteration | two LLM runs that must agree, checked against AI4Bharat IndicXlit when available (see below) | words stay in Roman script |
| Duplicates | landmark audio fingerprints and character 3-gram text similarity (no model) | |

`ASR_ENGINE=wav2vec` makes the wav2vec2 models the main ASR again (as before 2026-10-22).

**Transliteration and IndicXlit (optional).** A Roman-script word in a reference is converted to the native script
only when two independent answers agree: two differently worded LLM runs, and AI4Bharat's IndicXlit model listing
the same spelling among its top 5. IndicXlit needs `fairseq`, which only installs on Linux; on Linux run
`.venv/bin/pip install ai4bharat-transliteration`, on Mac/Windows build its Docker image once (needs Docker Desktop,
~3 GB): `docker build -t speechqm-indicxlit docker/indicxlit`. Without IndicXlit, two agreeing LLM runs are
enough. Every word's answers and the decision are in `results/transliteration_words.csv`.

**Public corpora for the duplicate check (optional).** Index any corpora you want new data checked against, once:
`.venv/bin/python build_corpus_index.py /data/fleurs_hi /data/kathbath/manifest.csv` (folders of audio, `.txt`
transcripts next to the audio, or a manifest CSV with an audio path column and a text column). The index goes to
`models/public_corpus_index/`; every later run checks audio (re-encoded, trimmed and noisy copies included) and
utterance text against it.

Weak spots: Gujarati ASR is poor (WER 0.83 on our sample: IndicConformer leaves many chunks empty); 8 kHz phone
recordings transcribe worse than 16 kHz recordings; the audio language ID is unreliable for Odia (no model heard
our Odia file as Odia).

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama is not running` | open the Ollama app (Mac/Windows) or run `ollama serve` (Linux), then run again |
| `That Hugging Face token was rejected` | create a new **Read** token; run with `--token hf_...` |
| It keeps listing gated models | open each listed link while logged in to the same account and accept; then press Enter |
| `... is not inside a folder named after a language` | add `--language <name>` or rename the folders |
| `./setup.sh: Permission denied` | `chmod +x setup.sh run.sh` |
| PowerShell refuses to run `setup.ps1` | use exactly `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| Very slow | no GPU found: check the `Device:` line at step 4 (`cuda`, `mps` or `cpu`). On Windows/Linux, stop the computer from sleeping during a run (`run.sh` does this on a Mac with `caffeinate`) |
| Out of memory | close other apps; or use a smaller LLM: `PLANNER_MODEL=qwen3:4b CODER_MODEL=qwen3:4b ./run.sh ...` |
| Something failed for one language | the other languages still run; details are in `pipeline.log` in the project folder |

## 9. Advanced

- **Settings** are in `config.py`; any model can be overridden with an environment variable of the same name, in
  the shell or in `.env` (`PLANNER_MODEL`, `CODER_MODEL`, `JUDGE_MODEL`, `MONITOR_MODEL` (`none` turns the monitor
  off), `OLLAMA_BASE_URL` for an Ollama server on another machine, `ASR_ENGINE`, `SPEECH_BACKEND=local`,
  `SPEAKER_MATCH_THRESHOLD`, `DEFAULT_LANGUAGE`, `CODER_MAX_TOKENS`).
- **Speakers across batches:** every run adds its speakers' voice embeddings to `speaker_db/` (`SPEAKER_DB=<folder>`
  to use another, `SPEAKER_DB=none` to turn it off). Re-running the same files replaces their entries, so a re-run
  does not mark its own speakers as `Old`.
- **Re-planning:** after the checks, steps that failed or that the monitor wanted re-run go back to the planner, which
  picks those worth one more run (`REPLAN=0` turns this off).
- **Tool library:** every script the code-writing LLM wrote that passed its checks is kept in `tool_library/`. The
  LLM still writes a new script on every run; a kept one only runs when all of its attempts fail.
- **Original prompt-driven mode:** write a prompt containing an absolute folder path into `my_secrets.py`
  (`PROMPT = "Check sample rate and count speakers in '/data/hindi/audios'"`) and run `.venv/bin/python llm.py`.
- **Building blocks:** `prepare_runs.py` (sort a dataset into language folders), `download_models.py` (pre-download
  models, e.g. `download_models.py all`), `summarize.py <results folder>` (rebuild `qc_summary.csv` and
  `dashboard.html`), `build_corpus_index.py` (public corpora for the duplicate check).

## 10. SpeechQM-Dataset and paper

SpeechQM-Dataset is a synthetic benchmark of 15.5 hours of Hindi dialogue with injected errors, released with the
paper. The dataset and the paper's results are at
[this link](https://drive.google.com/drive/folders/1VOGTY486zEhL_H4hM5s1H2ONgRgwWYmg?usp=sharing).

<p align="center">
<img src="components/IntroSpeechQCAgent_page-0001.jpg" alt="Intro to SpeechQM Agent" width="80%">
</p>

If you use SpeechQM-Agent in your research, please cite the paper (ACL 2026).
