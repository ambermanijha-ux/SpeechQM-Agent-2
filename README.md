# SpeechQM-Agent (local edition)

Automatic quality checks for speech datasets (audio + transcripts) in Indian languages and English, running
entirely on your own computer: Mac, Windows or Linux. No OpenAI/Groq keys; the only account you need is a free
Hugging Face token (optional, but it gives better speaker and Hindi results).

Point it at a folder of recordings and you get one table with, per file: duration, corruption, sample rate,
upsampling, silence, number of speakers and their talk time, whether a speaker also appears in another file,
detected language, script check, ASR transcript, word error rate against your reference transcript, alignment
(CTC) score, an LLM fluency score, the conversation's domain and the number of English words.

Based on *Rolling Out Data Quality Overnight, without losing the plot: A Multi-Agent System for Speech Data Quality
Management* (ACL 2026, `ACL2026_Rolling_Out_Data_Quality.pdf`). See `changes.md` for what differs from the original
code and `implementation.md` for how it works inside.

---

## 1. Requirements

| | Minimum |
|---|---|
| OS | macOS 13+ (Apple Silicon or Intel), Windows 10/11 (64-bit), Linux (x86_64 or arm64) |
| RAM | 16 GB (the 8B language model needs ~6 GB) |
| Disk | ~20 GB free (first run downloads ~8-10 GB of models) |
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
   - https://huggingface.co/ai4bharat/indicwav2vec-hindi
   - only if you have such data: `ai4bharat/indicwav2vec_v1_bengali`, `ai4bharat/indicwav2vec_v1_gujarati`,
     `ai4bharat/indicwav2vec-odia`
3. The first run asks for the token (typing is hidden) and saves it; you are not asked again.

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
With references you also get WER, CTC against the reference, tag normalization, English-word count and duplicate
checks. Without them these checks are skipped (CTC then scores the ASR transcript).

Your audio folder is never modified: files are copied into the results folder before processing.

## 6. Results

Everything is written to `<audio folder>_qc/` (or `--out`):

```
audio_qc/
├── qc_summary.csv            <- one row per audio file, all checks (open in Excel / Google Sheets)
├── hindi/
│   ├── audios/               copies of the Hindi audio
│   └── results/              one CSV per check (details: transcripts, aligned segments, ...)
└── telugu/ ...
```

`qc_summary.csv` columns:

| Column | Meaning | Look out for |
|---|---|---|
| Language, Filename | run language and file | |
| Duration_s | length in seconds | |
| Corruption | `valid` / `corrupt` (unreadable, empty, or `.mp3`: the SpeechQM spec accepts WAV only) | `corrupt` |
| Valid_Extension | `.wav` | `False` |
| Sample_Rate, Sample_Rate_Status | Hz; Pass = 16 kHz | `Fail` |
| Upsampled | `True` = almost no energy above 4 kHz: narrowband (8 kHz / phone) audio, possibly stored at a higher rate | `True` |
| Silence_s | seconds without speech | high share of Duration_s |
| Speakers, Speaker_Durations_h | number of speakers and talk time per speaker in hours; speaker IDs are shared across files | unexpected count |
| Speaker_New_Old, Shares_Speaker_With | `Old` = a voice in this file also speaks in the listed file(s) | `Old` where every speaker should be unique |
| Detected_Language, LID_Confidence | IndicLID language of the ASR transcript | not the folder's language |
| Script_Share, Script_OK | share of letters in the language's script | `False` |
| CTC_Score, CTC_Status | how well the reference (or ASR) transcript aligns with the audio, 0-1; Good > 0.7, Medium > 0.5 | `Poor` |
| LLM_Score | 0-10 fluency/coherence of the ASR transcript judged by the local LLM | low |
| Domain | topic (Healthcare, Insurance, Finance, Sales, Daily Life, ...) | unexpected topic |
| WER | word error rate of ASR vs reference (needs reference) | |
| English_Words_in_Reference | Latin-script words in the reference transcript | |

**Reading WER:** the ASR writes everything in the language's own script, while many references write English words
in Latin script ("Hospital", "insurance"). Every such word counts as an error, so WER is high on code-mixed speech
even when recognition is good. Compare WER between files and vendors rather than against an absolute target.

## 7. Models used

| Purpose | Model (with token + accepted terms) | Without access |
|---|---|---|
| Planner, domain, LLM score, code writing | `qwen3:8b` via Ollama | same |
| Speech/silence detection | `pyannote/segmentation-3.0` | Silero VAD |
| Speakers | `pyannote/speaker-diarization-3.1` | Silero VAD + SpeechBrain ECAPA clustering |
| ASR Hindi | `ai4bharat/indicwav2vec-hindi` | `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` |
| ASR Bengali / Gujarati / Odia | `ai4bharat/indicwav2vec_v1_bengali` / `_v1_gujarati` / `indicwav2vec-odia` | Vakyansh model of the language |
| ASR other languages | Vakyansh (`Harveenchadha/vakyansh-wav2vec2-<language>-...`) | same (not gated) |
| CTC alignment | Vakyansh Hindi for Hindi, otherwise the language's ASR model | same |
| Language identification | AI4Bharat IndicLID (FastText + IndicBERT) | same |

Weak spots: the Malayalam (8 h of training data), Assamese (8 h) and Punjabi (10 h) Vakyansh models give poor
transcripts; 8 kHz phone recordings transcribe worse than 16 kHz recordings.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Ollama is not running` | open the Ollama app (Mac/Windows) or run `ollama serve` (Linux), then run again |
| `That Hugging Face token was rejected` | create a new **Read** token; run with `--token hf_...` |
| It keeps listing gated models | open each listed link while logged in to the same account and accept; then press Enter |
| `... is not inside a folder named after a language` | add `--language <name>` or rename the folders |
| `./setup.sh: Permission denied` | `chmod +x setup.sh run.sh` |
| PowerShell refuses to run `setup.ps1` | use exactly `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| Very slow | no GPU found: check the `Device:` line at step 4 (`cuda`, `mps` or `cpu`) |
| Out of memory | close other apps; or use a smaller LLM: `PLANNER_MODEL=qwen3:4b CODER_MODEL=qwen3:4b ./run.sh ...` |
| Something failed for one language | the other languages still run; details are in `pipeline.log` in the project folder |

## 9. Advanced

- **Settings** are in `config.py`; any model can be overridden with an environment variable of the same name
  (`PLANNER_MODEL`, `CODER_MODEL`, `OLLAMA_BASE_URL` for an Ollama server on another machine,
  `SPEECH_BACKEND=local`, `SPEAKER_MATCH_THRESHOLD`, `DEFAULT_LANGUAGE`).
- **Original prompt-driven mode:** write a prompt containing an absolute folder path into `my_secrets.py`
  (`PROMPT = "Check sample rate and count speakers in '/data/hindi/audios'"`) and run `.venv/bin/python llm.py`.
- **Building blocks:** `prepare_runs.py` (sort a dataset into language folders), `download_models.py` (pre-download
  models, e.g. `download_models.py all`), `summarize.py <results folder>` (rebuild `qc_summary.csv`).

## 10. SpeechQM-Dataset and paper

SpeechQM-Dataset is a synthetic benchmark of 15.5 hours of Hindi dialogue with injected errors, released with the
paper. The dataset and the paper's results are at
[this link](https://drive.google.com/drive/folders/1VOGTY486zEhL_H4hM5s1H2ONgRgwWYmg?usp=sharing).

<p align="center">
<img src="components/IntroSpeechQCAgent_page-0001.jpg" alt="Intro to SpeechQM Agent" width="80%">
</p>

If you use SpeechQM-Agent in your research, please cite the paper (ACL 2026).
