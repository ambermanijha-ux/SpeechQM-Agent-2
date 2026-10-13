# SpeechQM-Agent: Every Script and Every Command, Explained

This is the operator's manual for this edition of SpeechQM-Agent. It goes through every file in the repository,
explains each command inside the setup and run scripts line by line, follows one run from the moment you press
Enter to the final CSV, and lists the commands you will type yourself.

Companion documents:
- `README.md`: the short "how do I use it" guide.
- `implementation.md`: the architecture in compact reference form (tables, data contracts, how to extend).
- `changes.md`: what differs from the original research code.
- `simplified_methodology.md` / `input-output.md`: the research method and the original prompt-driven usage.

Functions are referred to by name (`llm.main`, `utility_functions.diarize`) rather than by line number, so this
guide stays correct when lines move.

---

## Contents

1. [The repository at a glance](#1-the-repository-at-a-glance)
2. [The big picture: what happens when you run it](#2-the-big-picture-what-happens-when-you-run-it)
3. [`setup.sh` (Mac / Linux), line by line](#3-setupsh-mac--linux-line-by-line)
4. [`setup.ps1` (Windows), line by line](#4-setupps1-windows-line-by-line)
5. [`run.sh` and `run.bat`, line by line](#5-runsh-and-runbat-line-by-line)
6. [`requirements.txt`: every package and why it is there](#6-requirementstxt-every-package-and-why-it-is-there)
7. [`run_qc.py`: the conductor](#7-run_qcpy-the-conductor)
8. [`prepare_runs.py`: sorting a dataset by language](#8-prepare_runspy-sorting-a-dataset-by-language)
9. [`download_models.py`: access checks and downloads](#9-download_modelspy-access-checks-and-downloads)
10. [`config.py`: every setting](#10-configpy-every-setting)
11. [`llm.py`: the agent system](#11-llmpy-the-agent-system)
12. [`utility_functions.py`: the speech and language tools](#12-utility_functionspy-the-speech-and-language-tools)
13. [`summarize.py`: the one-row-per-file table](#13-summarizepy-the-one-row-per-file-table)
14. [Small files: `my_secrets.py`, `.gitignore`, `.gitattributes`, `env/denv.yml`](#14-small-files)
15. [Commands you will type, explained](#15-commands-you-will-type-explained)
16. [Where things are stored on disk](#16-where-things-are-stored-on-disk)
17. [Reading the console output and `pipeline.log`](#17-reading-the-console-output-and-pipelinelog)
18. [Troubleshooting by symptom](#18-troubleshooting-by-symptom)

---

## 1. The repository at a glance

| File | What it is | Who runs it |
|---|---|---|
| `setup.sh` | One-time installer for Mac and Linux | you, once |
| `setup.ps1` | One-time installer for Windows | you, once |
| `run.sh` | Starts a quality-check run on Mac and Linux | you, every run |
| `run.bat` | Starts a quality-check run on Windows | you, every run |
| `run_qc.py` | The conductor: token, model access, downloads, sorting, running, summary | started by `run.sh` / `run.bat` |
| `prepare_runs.py` | Copies your audio into one folder per language and converts transcript JSONs | called by `run_qc.py` (can also be run alone) |
| `download_models.py` | Checks which models your token can use and downloads them | called by `run_qc.py` (can also be run alone) |
| `config.py` | All model names, languages, thresholds | imported by everything |
| `llm.py` | The agent system: plans the checks with an LLM, builds a graph, runs the 24 checks | called by `run_qc.py` (or `python llm.py` in the original prompt mode) |
| `utility_functions.py` | Speech and text tools: loading audio, speech detection, ASR, speakers, alignment, language ID | imported by `llm.py` |
| `summarize.py` | Builds `qc_summary.csv` | called by `run_qc.py` (can also be run alone) |
| `requirements.txt` | The Python packages and exact versions | read by `setup.sh` / `setup.ps1` |
| `my_secrets.py` | The prompt for the original `python llm.py` mode; no keys | optional |
| `.gitignore`, `.gitattributes` | Git rules: what not to commit, which line endings to use | git |
| `env/denv.yml` | The original authors' Linux conda environment (not used by this edition) | nobody |
| `README.md`, `explanation.md`, `implementation.md`, `changes.md`, `simplified_methodology.md`, `input-output.md` | Documentation | you, reading |
| `ACL2026_Rolling_Out_Data_Quality.pdf`, `components/` | The paper and its figures | |

Created on your machine when you use it (never committed):

| Path | What |
|---|---|
| `.venv/` | The project's private Python 3.11 with all packages (~3-5 GB) |
| `models/indiclid/` | IndicLID language-identification models (~1.5 GB) |
| `models/spkrec-ecapa-voxceleb/` | The speaker-embedding model, only if the ungated speaker backend is used |
| `pipeline.log` | Detailed log of every run |
| `<your audio folder>_qc/` | The results of a run |

---

## 2. The big picture: what happens when you run it

```
./run.sh /data/calls --language Hindi            (Windows: run.bat C:\data\calls --language Hindi)
   │
   └─► .venv/bin/python run_qc.py /data/calls --language Hindi
         │
         ├─ 1/5  prepare_runs.main()       copies audio to /data/calls_qc/hindi/audios
         │                                  converts transcript JSONs to /data/calls_qc/hindi/results/indicconf_hypothesis-gt.csv
         │
         ├─ 2/5  online()?                 is huggingface.co reachable?
         │        ├─ yes: setup_token()     use saved token, or ask for one; validate it with Hugging Face
         │        │       check_gated()     which gated models can this token use? → pick pyannote or local backend
         │        └─ no:  offline mode      use only models already downloaded
         │
         ├─ 3/5  check_ollama()            is the local LLM server running? start it; pull qwen3:8b if missing
         │
         ├─ 4/5  ensure_hf(), ensure_indiclid()   download speech models and IndicLID (first run only)
         │
         ├─ 5/5  for each language:
         │          llm.main(prompt, audio_dir, lang_code)
         │             ├─ LLM picks which of the 24 checks the prompt asks for (+ deterministic corrections)
         │             ├─ checks are ordered into layers by their dependencies
         │             ├─ LangGraph runs each layer's checks in parallel
         │             └─ each check writes one CSV into results/
         │
         └─ summarize.write_summary()      merges all CSVs into /data/calls_qc/qc_summary.csv
```

Two programs run side by side during a run:
1. **Python** (`run_qc.py` and everything it imports) does the work.
2. **Ollama** is a separate background program that holds the language model `qwen3:8b` in memory and answers
   questions over a local web address, `http://localhost:11434`. Python sends it text and gets text back. Nothing
   leaves your computer.

---

## 3. `setup.sh` (Mac / Linux), line by line

```bash
#!/usr/bin/env bash
```
The **shebang**. When you type `./setup.sh`, the system reads this first line to know which program should run
the file. `/usr/bin/env bash` means "find `bash` wherever it is installed and use it", which works on every Mac
and Linux system even if bash lives in a different folder.

```bash
set -euo pipefail
```
Makes the script stop at the first problem instead of carrying on with a broken install:
- `-e`: exit as soon as any command fails.
- `-u`: treat use of an undefined variable as an error (catches typos).
- `-o pipefail`: in a pipeline like `curl ... | sh`, fail if *any* part fails, not only the last one. Without it,
  a failed download piped into `sh` would look like success.

```bash
cd "$(dirname "$0")"
```
`$0` is the path of the script itself, `dirname` strips the file name, so this moves into the folder that
contains `setup.sh`. That makes the script work no matter where you run it from (`./setup.sh`,
`~/SpeechQM-Agent-2/setup.sh`, ...). The quotes keep paths with spaces intact.

```bash
if ! command -v uv >/dev/null 2>&1; then
```
`command -v uv` prints where the `uv` program is, or fails if it is not installed. `>/dev/null 2>&1` throws the
output away (we only care about success or failure). `!` inverts it: "if uv is NOT installed, then...".

**What is uv?** A very fast Python package manager (from Astral). It can download a Python interpreter by itself,
create an isolated environment, and install packages, so you never need to install Python manually.

```bash
  echo "Installing uv (Python package manager) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
```
Downloads uv's official installer script and runs it.
- `curl` downloads a URL. `-L` follows redirects, `-s` is silent (no progress bar), `-S` still shows errors
  despite `-s`, `-f` fails on HTTP errors (e.g. 404) instead of piping an error page into `sh`.
- `| sh` feeds the downloaded installer to the shell to run it. It installs `uv` into `~/.local/bin`.

```bash
  export PATH="$HOME/.local/bin:$PATH"
fi
```
`PATH` is the list of folders the shell searches for programs. The new `uv` is in `~/.local/bin`, which this
already-running script does not know about yet; this line adds it so the next lines can find `uv`. (New terminal
windows pick it up automatically because the installer edits your shell profile.)

```bash
echo "Creating .venv (Python 3.11) and installing packages ..."
uv venv -p 3.11 --allow-existing .venv
```
Creates a **virtual environment** in the folder `.venv`: a private Python installation used only by this
project, so its packages never clash with anything else on the computer.
- `-p 3.11`: use Python 3.11. If the computer does not have 3.11, uv downloads a standalone copy.
- `--allow-existing`: if `.venv` already exists (you run setup a second time), keep it instead of failing.

```bash
uv pip install --python .venv/bin/python --torch-backend=auto -r requirements.txt
```
Installs every package listed in `requirements.txt` into `.venv`.
- `--python .venv/bin/python`: install into this project's environment, not into a system Python.
- `-r requirements.txt`: read the package list from the file.
- `--torch-backend=auto`: PyTorch (the deep-learning library) comes in different builds: for NVIDIA GPUs (CUDA,
  several versions) and for CPUs. uv looks at the machine (is there an NVIDIA driver? which version?) and picks
  the matching build from PyTorch's own download server. On a Mac it uses the normal build, which already
  supports the Apple GPU ("MPS").

```bash
if [ -z "${OLLAMA_BASE_URL:-}" ] && ! command -v ollama >/dev/null 2>&1 && [ ! -d /Applications/Ollama.app ]; then
```
Decides whether Ollama needs installing. All three must be true:
- `[ -z "${OLLAMA_BASE_URL:-}" ]`: the variable `OLLAMA_BASE_URL` is empty. If you set it, you are pointing the
  tool at an Ollama server on another machine, so nothing is installed locally. (`${...:-}` gives an empty string
  instead of an error when the variable is not set, which matters because of `set -u`.)
- `! command -v ollama`: the `ollama` command is not installed.
- `[ ! -d /Applications/Ollama.app ]`: the Mac Ollama app is not installed either.

```bash
  if [ "$(uname)" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then brew install ollama
    else echo "Install Ollama from https://ollama.com/download, open it once, then run ./run.sh"; exit 1; fi
```
`uname` prints the operating system name; `Darwin` means macOS. On a Mac with [Homebrew](https://brew.sh)
(`brew`), Ollama is installed with `brew install ollama`. Without Homebrew the script asks you to install the
Ollama app from the website and stops (`exit 1` = stop with an error code).

```bash
  else
    echo "Installing Ollama ..."
    curl -fsSL https://ollama.com/install.sh | sh
  fi
fi
```
On Linux: download and run Ollama's official installer. It needs administrator rights and may ask for your
password (`sudo`). On most distributions it also registers Ollama as a background service that starts with the
computer.

```bash
echo
echo "Setup done. Run the checks with:  ./run.sh /path/to/audio/folder"
```
Prints a blank line and the next step.

**What setup does not do:** it does not download any AI models. Those are fetched on the first run, after you
have given your Hugging Face token, so that only the models for your languages are downloaded.

---

## 4. `setup.ps1` (Windows), line by line

PowerShell is Windows' modern command shell. You start the script with:
```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```
- `powershell`: start Windows PowerShell.
- `-ExecutionPolicy Bypass`: by default Windows refuses to run downloaded scripts; this allows *this one run* of
  this script without changing your system settings.
- `-File setup.ps1`: the script to run.

```powershell
$ErrorActionPreference = "Stop"
```
Like `set -e` in bash: stop on the first PowerShell error instead of continuing.

```powershell
Set-Location $PSScriptRoot
```
`$PSScriptRoot` is the folder containing the script; `Set-Location` is `cd`. Same purpose as the `cd` line in
`setup.sh`.

```powershell
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
```
`Get-Command uv` looks for the `uv` program; `-ErrorAction SilentlyContinue` makes it return nothing instead of
an error if missing; `-not` inverts. So: "if uv is not installed".

```powershell
    Write-Host "Installing uv (Python package manager) ..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
uv's official Windows installer. `irm` (`Invoke-RestMethod`) downloads the script, `iex`
(`Invoke-Expression`) runs it. It installs `uv.exe` into `%USERPROFILE%\.local\bin`.

```powershell
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
```
Adds that folder to this window's search path (`$env:Path`, Windows' `PATH`) so the next lines find `uv`.

```powershell
uv venv -p 3.11 --allow-existing .venv
if ($LASTEXITCODE -ne 0) { exit 1 }
```
Creates the private Python 3.11 environment, exactly as on Mac/Linux. `$LASTEXITCODE` holds the exit code of the
last external program; anything other than 0 (`-ne 0` = not equal to 0) means it failed, so the script stops.
This check is needed because `$ErrorActionPreference` only covers PowerShell's own commands, not external
programs like `uv`.

```powershell
uv pip install --python .venv\Scripts\python.exe --torch-backend=auto -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit 1 }
```
Installs the packages. On Windows the environment's Python is `.venv\Scripts\python.exe` (on Mac/Linux it is
`.venv/bin/python`). `--torch-backend=auto` picks the CUDA build of PyTorch if an NVIDIA GPU and driver are
present, else the CPU build.

```powershell
if (-not (Get-Command ollama -ErrorAction SilentlyContinue) -and
    -not (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe")) {
```
Is Ollama missing? Checked two ways: the `ollama` command on the path, and the file where the Ollama installer
puts it (`%LOCALAPPDATA%\Programs\Ollama\ollama.exe`). `Test-Path` checks whether a file exists.

```powershell
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Installing Ollama ..."
        winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
```
`winget` is Windows' built-in package manager (Windows 10 1809+ and 11).
- `install -e --id Ollama.Ollama`: install the package whose ID is exactly (`-e`) `Ollama.Ollama`.
- `--accept-source-agreements --accept-package-agreements`: answer winget's licence prompts automatically.

The Ollama app starts itself after installing and sits in the system tray.

```powershell
    } else {
        Write-Host "Install Ollama from https://ollama.com/download, start it, then run run.bat"
        exit 1
    }
}
```
Without winget: asks you to install Ollama by hand and stops.

```powershell
Write-Host ""
Write-Host "Setup done. Run the checks with:  run.bat C:\path\to\audio\folder"
```

---

## 5. `run.sh` and `run.bat`, line by line

Both are tiny "launchers". They exist so you don't have to remember where the project's Python lives or type the
UTF-8 settings yourself.

### `run.sh` (Mac / Linux)

```bash
#!/usr/bin/env bash
D="$(cd "$(dirname "$0")" && pwd)"
```
Works out the absolute path of the project folder: go to the script's folder (`cd "$(dirname "$0")"`) and print
it (`pwd`), and store that in `D`. This runs in a sub-shell (`$(...)`), so *your* current directory does not
change. That matters: if you type `./run.sh ./data`, the relative path `./data` must still mean "data next to
where I am", not "data next to the script".

```bash
[ -x "$D/.venv/bin/python" ] || { echo "Run $D/setup.sh first"; exit 1; }
```
`[ -x file ]` tests that the file exists and is executable. If the project's Python is not there, setup has not
been run: print a hint and stop. `A || B` means "if A fails, do B".

```bash
PYTHONUTF8=1 exec "$D/.venv/bin/python" "$D/run_qc.py" "$@"
```
- `PYTHONUTF8=1`: an environment variable set only for this command. It switches Python into **UTF-8 mode**, so
  every file is read and written as UTF-8 (needed for Devanagari, Tamil, Telugu... text).
- `exec`: replace the shell with Python instead of starting Python as a child. Ctrl-C then goes straight to
  Python, and the exit code is Python's.
- `"$D/.venv/bin/python" "$D/run_qc.py"`: run the conductor with the project's own Python.
- `"$@"`: pass on every argument you typed (`/data/calls --language Hindi`), each kept intact even with spaces.

### `run.bat` (Windows)

```bat
@echo off
```
Don't print each command before running it (the `@` hides this line itself too).

```bat
rem Usage: run.bat C:\path\to\audio\folder [--language Hindi] [--prompt "..."]   (see: run.bat --help)
```
`rem` = a comment.

```bat
if not exist "%~dp0.venv\Scripts\python.exe" (echo Run setup.ps1 first & exit /b 1)
```
- `%~dp0` is the **d**rive and **p**ath of argument **0** (the batch file itself), ending in `\`, i.e. the
  project folder.
- If the project's Python is missing, print a hint and stop. `&` runs two commands on one line;
  `exit /b 1` leaves the batch file (not the whole window) with error code 1.

```bat
set PYTHONUTF8=1
```
UTF-8 mode, as in `run.sh`. On Windows this matters even more: Python's default there is the old cp1252
character set, which cannot represent Indic scripts at all.

```bat
set PATH=%LOCALAPPDATA%\Programs\Ollama;%PATH%
```
Puts Ollama's install folder on the search path for this window, so `run_qc.py` can start `ollama` even right
after installation, before Windows has refreshed the path.

```bat
"%~dp0.venv\Scripts\python.exe" "%~dp0run_qc.py" %*
```
Run the conductor with the project's Python; `%*` passes on all your arguments.

---

## 6. `requirements.txt`: every package and why it is there

Exact versions (`==`) are pinned so every machine gets the same, tested combination.

| Package | Version | Used for |
|---|---|---|
| `torch` | 2.14.1 | PyTorch: runs all neural networks (ASR, speakers, alignment) on GPU or CPU |
| `torchaudio` | 2.11.0 | Audio utilities used by pyannote (resampling) |
| `transformers` | 4.57.6 | Hugging Face library that loads the wav2vec2 ASR models and the IndicBERT tokenizer |
| `huggingface_hub` | 0.36.2 | Login, access checks and model downloads from huggingface.co |
| `librosa` | 0.11.0 | Reads `.wav`/`.mp3`/`.flac` and converts everything to 16 kHz mono |
| `soundfile` | 0.14.0 | Reads audio headers (sample rate, corruption check in generated scripts) |
| `pydub` | 0.25.1 | Kept for generated scripts that may use it; not used by the fixed code |
| `numpy` | 2.2.6 | Arrays and maths (FFT for upsampling, embeddings) |
| `pandas` | 3.0.6 | Reading and writing every CSV |
| `scikit-learn` | 1.9.1 | Agglomerative clustering in the ungated speaker backend |
| `speechbrain` | 1.1.1 | ECAPA speaker-embedding model (ungated speaker backend) |
| `silero-vad` | 6.2.3 | Ungated speech/silence detector |
| `fasttext-predict` | 0.9.2.4 | Runs IndicLID's FastText models (prediction-only build with wheels for every OS) |
| `jiwer` | 4.0.0 | Word error rate |
| `sentencepiece` | 0.2.2 | Needed by the IndicBERT tokenizer |
| `langchain-ollama` | 0.3.10 | Talks to the Ollama server (`ChatOllama`) |
| `langgraph` | 0.6.11 | Builds and runs the graph of checks |
| `grandalf` | 0.8 | Draws the ASCII picture of the graph printed at the start of each language |
| `pyannote.audio` | 4.0.7 | Speech detection and speaker diarization (gated models) |

Everything else (about 130 packages) is pulled in automatically as dependencies of these.

---

## 7. `run_qc.py`: the conductor

### 7.1 Module-level code (runs as soon as the file starts)

```python
HERE = Path(__file__).resolve().parent
```
The project folder, found from the script's own location.

```python
if os.name == "nt" and not sys.flags.utf8_mode and __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv], env={**os.environ, "PYTHONUTF8": "1"}))
```
Safety net for Windows (`os.name == "nt"`) when someone starts `python run_qc.py` directly instead of through
`run.bat`: if UTF-8 mode is off, the script starts itself again with `python -X utf8` (the command-line switch
for UTF-8 mode), waits for it, and exits with its exit code. UTF-8 mode can only be chosen when Python starts,
which is why a restart is needed.

```python
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, str(HERE))
```
Make any Python started from here also use UTF-8, and make the project's own modules (`config`, `llm`, ...)
importable regardless of the current directory.

`DEFAULT_CHECKS` and `REFERENCE_CHECKS` are the plain-English prompts used when you don't give `--prompt`.
They are written in the same words the planner LLM understands. `REFERENCE_CHECKS` (WER, normalization, English
word count, duplicates) is appended only for languages that have reference transcripts.

### 7.2 `main()`: the five steps

**Arguments** (`argparse`): `audio` (required folder), `--language`, `--prompt`, `--out`, `--token`,
`--no-token`, `--yes`. `--help` prints them with the module's docstring.

```python
ask = sys.stdin.isatty() and not args.yes
```
Only ask questions when a person is at a keyboard (`isatty()` is false when input comes from a pipe or a
scheduler) and `--yes` was not given.

```python
src = Path(args.audio).expanduser().resolve()
out = Path(args.out)... if args.out else src.parent / f"{src.name}_qc"
os.chdir(HERE)
```
Turns your folder argument into an absolute path *before* changing directory (so `./data` keeps meaning what you
meant), chooses the output folder (default: next to your folder, with `_qc` appended), then moves into the
project folder so `pipeline.log` is written there.

**Step 1/5, sorting:** `prepare_runs.main(src, out, args.language)` (section 8). Returns
`{"hindi": <out>/hindi, "telugu": <out>/telugu, ...}`. If a file is outside a language folder and no
`--language` was given, it stops with an example command.

**Step 2/5, Hugging Face access:**
- `online()` tries to fetch `https://huggingface.co/api/models?limit=1` within 8 seconds. It returns False if
  that fails or if you set `HF_HUB_OFFLINE=1` yourself.
- **Online:**
  - `setup_token()` takes the token from `--token`, else the `HF_TOKEN` environment variable, else the one saved
    by an earlier login (`huggingface_hub.get_token()`), else asks you with `getpass` (your typing is not shown).
    Pressing Enter without a token skips it. The token is validated with `whoami()` (a Hugging Face API call that
    returns your user name). A rejected token (HTTP 401) stops the run with a clear message. A new valid token is
    saved with `login()` so you are not asked again.
  - `check_gated()` calls `has_access()` (section 9) for the three pyannote repositories and for the gated ASR
    candidates of the languages found (e.g. `ai4bharat/indicwav2vec-hindi`). If some are not accessible, it
    prints their links and waits: accept the terms in your browser and press Enter to re-check, or type `skip`.
    Result: `"pyannote"` if all pyannote repos are accessible, else `"local"`. Missing ai4bharat access is fine:
    the Vakyansh fallback is used automatically later.
- **Offline:** sets `HF_HUB_OFFLINE=1` (tells the Hugging Face libraries to use only the local cache) and uses
  pyannote only if it is already downloaded.
- Without a token (`--no-token`, or Enter at the prompt) the backend is `"local"`.
- The choice is stored in both `os.environ["SPEECH_BACKEND"]` and `config.SPEECH_BACKEND`, because `config` was
  already imported in step 1 and would otherwise keep its default.

**Step 3/5, Ollama:** `check_ollama()` asks the Ollama server for its model list (`GET /api/tags`). If nothing
answers, it starts Ollama: on a Mac with the app, `open -a Ollama`; otherwise `ollama serve` in the background.
It waits up to 20 seconds. If Ollama still does not answer, it stops with install instructions for your OS. Then
`ensure_ollama()` downloads `qwen3:8b` if it is missing.

**Step 4/5, models:** `ensure_hf(languages, backend)` and `ensure_indiclid()` (section 9). On the first run this
is where most of the downloading happens; later it takes seconds. Then it prints the compute device (`cuda`,
`mps` or `cpu`) and the speech backend.

**Step 5/5, checks:** for each language:
1. `clear_results(run)` deletes old outputs in `results/` except the reference CSV. This is needed because the
   transcription step skips its work when `indicconf_hypothesis.csv` already exists.
2. Builds the prompt: your `--prompt`, or `DEFAULT_CHECKS` plus `REFERENCE_CHECKS` if references exist, plus
   `(language: X)`.
3. Calls `llm.main(prompt, audio_dir=..., lang_code=...)` (section 11). The language is passed directly as an
   ISO code (`hi`, `te`, ...), so the LLM never has to guess it.
4. Any exception is caught and reported; the remaining languages still run.

Finally `summarize.write_summary(out, runs)` writes `qc_summary.csv`, and the exit code is non-zero if any
language failed.

---

## 8. `prepare_runs.py`: sorting a dataset by language

The pipeline itself (`llm.py`) only understands **one flat folder of one language**, with results in a
`results/` folder next to it. `prepare_runs.py` turns any layout into that.

| Function | What it does |
|---|---|
| `language(path, root, default)` | Walks up the file's folders from the closest one; the first folder whose name (lower-case, after `ALIASES` such as `gujrati → gujarati`) is a configured language wins. Otherwise the `--language` default, otherwise an error. |
| `transcript_text(json)` | Reads `{"segments": [...]}`, sorts segments by `start_time`, joins their `text` with single spaces |
| `transcript_for(audio, jsons)` | Finds the audio's transcript in the same folder: same name (`hi001.json`) first; otherwise a JSON whose name contains the audio name as a whole token (`Copy of hi001_human_transcribed.json`). `sample_1` does not match `sample_10_x.json` (the regex refuses a letter or digit right before or after). If several match, none is used and it says so. |
| `main(src, out, default)` | Finds all audio recursively (skipping macOS `._` files and anything already inside `out`), groups by language, then per language: copies files to `<out>/<lang>/audios` (again if size or modification time changed), deletes copies whose source is gone, rebuilds `results/indicconf_hypothesis-gt.csv` (`filename,ground_truth`) |

`shutil.copy2` copies the file and keeps its modification time, which is how a later run notices a changed
source file. Your original folder is only read, never written.

Run it alone: `.venv/bin/python prepare_runs.py <dataset> <output> [default language]`.

---

## 9. `download_models.py`: access checks and downloads

| Function | What it does |
|---|---|
| `has_access(repo)` | Online: `auth_check(repo)` asks Hugging Face whether your token may download the repository. Gated repos whose terms you have not accepted, missing repos, or no token for a gated repo all give False. Offline: True only if the repo is already in the local cache (`snapshot_download(..., local_files_only=True)`). |
| `ollama_models()` | `GET http://localhost:11434/api/tags`: the names of the models Ollama has |
| `ensure_ollama()` | For each missing model: `POST /api/pull` with `{"model": "qwen3:8b"}`. Ollama streams progress as JSON lines, shown as one updating line (`pulling ... 42%`). |
| `asr_candidate(language)` | The first model in `config.ASR_MODELS[language]` that `has_access` allows |
| `ensure_hf(languages, backend)` | Downloads the chosen ASR model of each language, the Hindi alignment model, the IndicBERT tokenizer, then loads the speech-detection and diarization models once. Loading is what pulls in their own sub-models (pyannote's speaker-embedding model and PLDA weights, or SpeechBrain ECAPA). |
| `ensure_indiclid()` | Downloads the three IndicLID zips from AI4Bharat's GitHub release in 1 MB pieces with a progress line, into a temporary folder; extracts there; moves the finished model folder into `models/indiclid/`. An interrupted download therefore never leaves a half-written model that a later run would trust. |

`snapshot_download(repo, allow_patterns=[...])` downloads a whole model repository into the Hugging Face cache,
limited to the file types needed (`*.json`, `*.bin`, `*.safetensors`, `*.txt`, `*.model`). Files already in the
cache are not downloaded again.

Run it alone: `.venv/bin/python download_models.py Hindi Tamil` (or `all`).

---

## 10. `config.py`: every setting

Every value can be overridden by an environment variable of the same name, e.g. `PLANNER_MODEL=qwen3:4b ./run.sh ...`.

| Setting | Default | Meaning |
|---|---|---|
| `MODELS_DIR`, `INDICLID_DIR` | `models/`, `models/indiclid/` | where non-Hugging-Face models are stored |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | the Ollama server's address (point it elsewhere to use another machine's GPU) |
| `PLANNER_MODEL` | `qwen3:8b` | LLM for task selection, checking, ordering, domain and fluency scoring |
| `CODER_MODEL` | `qwen3:8b` | LLM that writes the Python scripts for some checks |
| `LLM_NUM_CTX` | 8192 | how many tokens of context the LLM gets |
| `ASR_MODELS` | 15 languages | per language, a list of ASR models tried in order |
| `DEFAULT_LANGUAGE` | Hindi | when nothing names a language |
| `LANG_CODES` | `hi → Hindi`, ... | ISO 639-1 code to language name |
| `SCRIPTS` | Hindi → Devanagari U+0900–097F, ... | the Unicode block each language's script uses (script check) |
| `ALIGN_MODEL` | Vakyansh Hindi | CTC alignment model for Hindi |
| `SPEECH_BACKEND` | `pyannote` | `pyannote` or `local` (Silero + ECAPA) |
| `HF_TOKEN` | env var, else `True` | `True` tells the libraries "use the token saved by login" |
| `PYANNOTE_VAD_MODEL`, `PYANNOTE_DIARIZATION`, `PYANNOTE_REPOS` | segmentation-3.0, speaker-diarization-3.1, + community-1 | the gated pyannote repositories |
| `SPEAKER_MODEL` | `speechbrain/spkrec-ecapa-voxceleb` | speaker embeddings for the local backend |
| `INDICBERT_TOKENIZER` | `ai4bharat/IndicBERTv2-MLM-only` | tokenizer for IndicLID's BERT model |
| `DIAR_WINDOW_S`, `DIAR_HOP_S` | 3.0 s, 1.5 s | local backend: embedding window length and step |
| `DIAR_CLUSTER_THRESHOLD` | 0.7 | local backend: cosine distance at which windows stop being merged into one speaker |
| `SPEAKER_MATCH_THRESHOLD` | 0.6 | both backends: cosine similarity at which voices in two files count as the same person |
| `torch_device()` | | returns `cuda` if an NVIDIA GPU works, else `mps` (Apple GPU), else `cpu` |

---

## 11. `llm.py`: the agent system

### 11.1 What runs when the module is imported

- Imports the tools from `utility_functions` (which imports PyTorch and transformers: this is the slow part of
  starting up).
- `PROMPT` from `my_secrets.py` if that file exists (only used by `python llm.py`).
- `logging.basicConfig(filename="pipeline.log", ...)`: every `logging.info/warning/error` goes to
  `pipeline.log` with a timestamp.
- `llm = local_llm(PLANNER_MODEL)`, `coder_llm = local_llm(CODER_MODEL)`: two `ChatOllama` clients with
  `temperature=0` (always pick the most likely answer, so runs are as repeatable as possible) and, for qwen3,
  `reasoning=False` (qwen3 otherwise "thinks out loud" first, and those thoughts are full of numbers that would
  confuse the code that reads task numbers out of the reply). Creating the clients does not contact Ollama yet.

### 11.2 `main(prompt, audio_dir, lang_code)`: from prompt to results

1. **Inputs.** With `audio_dir` given (from `run_qc.py`), it is used directly. Without it (`python llm.py`
   mode), `parse_prompt` extracts a quoted absolute path, a `.csv` path and a language (name such as "Telugu",
   or code such as `te`) from the text, asking the LLM if the regexes find nothing.
2. **`select_tasks(prompt)`.** The planner LLM gets the menu of 24 checks plus keyword rules and replies with
   task numbers.
3. **`prompt_checker_agent(prompt, selection)`.** A second LLM call compares the selection with the prompt and
   answers `Status: Correct`, `Missing` (with which ones, then the selection is redone with that feedback) or
   `Extra` (the extras are removed). Up to 3 rounds. Only its removals are used: tasks it adds are ignored,
   because with qwen3:8b they were mostly never requested (see `implementation.md` §3).
4. **Deterministic corrections** (because an 8B model gets steps 2-3 wrong regularly):
   - `keyword_tasks(prompt)`: regexes for each check's name ("sample rate" → 21, "WER" → 24, ...); any check the
     prompt names but the LLM dropped is added back.
   - `add_dependencies(tasks)`: adds prerequisites using `TASK_DEPS` (e.g. WER needs normalization, which needs
     ASR).
   - `REFERENCE_TASKS` {9, 16, 18, 23, 24} are removed when no reference CSV exists, with a message, since they
     could only fail.
5. **Ordering.** `topological_sort_tasks` asks the LLM for layers such as `[[1, 2, 7], [5, 6, 12], [24]]`.
   `structure_is_valid` checks that every task appears exactly once and comes after its prerequisites. If not,
   `layer_tasks` computes the layers itself (Kahn's algorithm: repeatedly take every task whose prerequisites are
   done).
6. **`build_graph_from_structure`.** Builds a LangGraph `StateGraph`: a `start` node, edges from `start` to
   every check in layer 1, from every check in layer *n* to every check in layer *n+1*, and from the last layer
   to `END`. LangGraph runs all checks of a layer at the same time (threads) and waits for all of them before
   the next layer. `print_ascii()` draws this graph in the console (the large box diagram you see).
7. **`invoke(initial_state)`.** Runs the graph. The state is a dictionary every check reads and adds its own
   output key to (`A` = path of the ASR CSV, `D` = silence, `E` = speakers, `wer_output`, ...).
8. **Summary.** Prints each output key's value (usually `CSV saved at: ...` or an `Error: ...`).

### 11.3 The checks (nodes)

Each check is a function `xxx_agent(state)` or `xxx_func(state)` that returns `{"<its output key>": "<status>"}`
and writes a CSV into `results/`. The mapping from number to function is `node_map`. All 24, with how each is
implemented, are tabulated in `implementation.md` §4; the groups are:

- **Fixed code around a model** (1 ASR, 2 speakers, 8 silence, 10 CTC, 15 IndicLID): call functions in
  `utility_functions.py`.
- **Fixed Python** (3 transcript quality, 5 vocabulary, 6 script check, 9 English words, 11 upsampling,
  12 new/old speaker,
  22 speaker hours, 23 duplicates, 24 WER): deterministic, same result every run.
- **LLM judge** (13 domain, 17 fluency score): the LLM reads each transcript and answers; the reply is parsed
  with a regex (`Score: 7` → 7). The fluency score is gated: `pseudo_perplexity` (IndicBERTv2 masks a sample of up
  to 400 tokens one at a time and averages how surprised it is), the ASR alignment score and transcript quality
  are measured first, and only transcripts passing all three are sent to the LLM.
- **Code-writing agent** (4 characters, 7 length, 14 forced alignment, 16 normalization,
  19 corruption, 20 extension, 21 sample rate): the LLM writes a Python script, see below.

### 11.4 The code-writing agent (`run_agent`)

This is the "agentic" part from the paper: instead of a fixed function, the LLM is told what to compute and
writes the program itself.

1. The task prompt (e.g. "for every file in this folder write Filename, Sample_Rate, Status...") is extended by
   `describe_inputs()` with the real column names and first row of every CSV the prompt mentions, so the model
   does not guess column names.
2. On Windows, backslashes in paths become `/` (a path like `C:\Users\...` pasted into Python code is a syntax
   error, because `\U` starts an escape sequence).
3. `coder_llm.invoke(...)` returns text; `extract_code()` takes the longest ```` ```python ```` block (or a JSON
   tool call, which some models produce instead).
4. `run_code()` executes it with `exec()` in a fresh namespace, capturing what it prints. Errors are caught and
   reduced to the last few traceback lines; a script calling `exit()` cannot stop the pipeline.
5. The output CSV is validated: it must exist, be readable, and have the expected number of rows (one per audio
   file or input row). If anything is wrong, the error or the row-count mismatch goes back to the LLM as a new
   message ("Running your script printed ... Fix it"), up to 3 attempts.
6. `AGENT_LOCK` makes only one agent run at a time (the code temporarily redirects `print` output, and Ollama
   answers one request at a time anyway).

Every generated script and its output is written to `pipeline.log`, so you can always see exactly what code ran.

**Security note:** generated code runs with your user's permissions. It is only ever generated from the fixed
task prompts in `llm.py`, but run the tool on data you are allowed to process, as with any program.

### 11.5 Helpers worth knowing

| Helper | Purpose |
|---|---|
| `results_dir(audio_dir)` | `<parent of audio_dir>/results`, created if missing |
| `words(text)` | Splits text into words after removing `[tags]` and `<tags>`, splitting on Unicode punctuation and symbols but keeping vowel signs. Python's own `\w` treats Indic vowel signs as non-letters and would cut words apart. |
| `language_of(state)` | `lang_code` → language name (`te` → Telugu), default Hindi |
| `list_files`, `csv_rows` | counting files and rows for the agent's validation |

---

## 12. `utility_functions.py`: the speech and language tools

### 12.1 Basics

- `SR = 16000`: everything is processed at 16 kHz mono, the rate the models were trained on.
- `DEVICE = torch_device()`: `cuda`, `mps` or `cpu`.
- `MODEL_LOCK`: one lock around every model load and every model call. LangGraph runs a layer's checks in
  parallel threads; two threads inside the same GPU model at once corrupted memory and crashed Python on the
  Mac. With the lock, models take turns, while file I/O and LLM calls still overlap.
- `load_audio(path)`: `librosa.load(path, sr=16000, mono=True)`: decodes wav/mp3/flac, mixes stereo to mono,
  resamples to 16 kHz.
- `@lru_cache` on `get_asr`, `get_vad_model`, `get_diarization_pipeline`, `get_speaker_encoder`,
  `get_indiclid`: each model is loaded once per run and reused.

### 12.2 Speech detection (VAD) and silence

- `get_vad_model()`: pyannote `VoiceActivityDetection` on `segmentation-3.0` (backend `pyannote`), or Silero VAD
  (backend `local`; Silero changes PyTorch's thread count, so it is restored afterwards).
- `speech_regions(audio)`: list of `(start, end)` seconds containing speech. The audio is passed as an in-memory
  waveform, so pyannote does not need FFmpeg to read files.
- `get_total_silence_time(file)`: duration minus the union of speech regions. `process_folder_vad` writes
  `vad_silence_stats.csv`.

### 12.3 ASR (speech to text)

- `asr_model_for(language)`: tries each model in `ASR_MODELS[language]` by loading it; a gated model without
  access raises an error, so the next candidate (Vakyansh) is used. Cached per language.
- `asr_chunks(audio)`: speech regions merged into pieces of at most 8 s. wav2vec2 models handle short pieces
  best (on Hindi and Telugu tests, 8 s gave lower WER than 30 s or 5 s), and long pieces would not fit in memory.
- `transcribe_audio(path, language)`: for each piece, the processor normalizes the waveform, the model outputs a
  score for every character at every 20 ms frame (*logits*), `argmax` picks the best character per frame, and
  `batch_decode` collapses repeats and blanks into words (CTC greedy decoding).
- `transcribe_folder_to_csv`: all audio files → `indicconf_hypothesis.csv` (`Filename`,
  `Indiconformer_Hypothesis`). Skips the work if that file already exists.

### 12.4 Speakers (diarization) and new/old speakers

- `diarize(audio)` returns `{speaker label: (seconds spoken, voice fingerprint)}` for one file. The fingerprint
  ("embedding") is a vector that is similar for the same voice and different for different voices.
  - `diarize_pyannote`: pyannote's `speaker-diarization-3.1` pipeline with `return_embeddings=True`. Speakers too
    short to get a fingerprint are counted but never matched across files.
  - `diarize_local`: speech regions → overlapping 3 s windows every 1.5 s → ECAPA fingerprints → agglomerative
    clustering (windows closer than cosine distance 0.7 join the same speaker) → tiny clusters (< 5 % of the
    speech or < 1 s) are merged into their nearest big cluster.
- `save_num_speakers(folder)`: per file, matches each local speaker against a **folder-wide registry** of known
  voices: cosine similarity ≥ 0.6 → same person, same global ID (`SPEAKER_03` in two files is the same voice);
  otherwise a new ID. Writes `num_speakers.csv` (`File Name`, `Number of Speakers`, `Speaker Durations` as JSON
  in hours). pyannote alone would call the first speaker of *every* file `SPEAKER_00`, which is why this
  registry exists. Check 12 (new/old) and 22 (hours per speaker) read this file.

### 12.5 CTC score (forced alignment)

`force_alignment_and_ctc_score(audio, transcript, model)` measures how well a transcript fits the audio:
1. The model's per-frame character probabilities (*emissions*) are computed in 30 s pieces and joined.
2. The transcript is turned into the model's character IDs (`|` marks word boundaries).
3. A *trellis* (a table of the best score for "first *j* characters spoken by frame *t*") is filled in, then
   *backtracked* from the end to find the single best timing for every character.
4. Characters are grouped into words with start and end times; the score is the average probability along that
   path (0-1). Good > 0.7, Medium > 0.5, else Poor.

`align_model_for(language)`: Vakyansh Hindi for Hindi, the language's own ASR model otherwise.
`process_audio_directory` runs this for every row of the reference (or ASR) CSV → `ctc_scores.csv`.

### 12.6 Upsampling

`is_upsampled_from_8k_v2`: takes the frequency spectrum (FFT) of the audio at the file's own sample rate; if the
file is stored above 8 kHz and less than 2 % of the energy is between 4 and 8 kHz, the recording has no real
high-frequency content: it was narrowband (8 kHz, typical of phone calls) and later upsampled, even though the file
says 16 or 48 kHz. A file stored at 8 kHz is not upsampled (the sample rate check reports it). An earlier version
resampled every file to 16 kHz first, which made every 8 kHz file look upsampled.

### 12.7 Language identification (IndicLID)

`IndicLID` (AI4Bharat) predicts the language and script of a text, e.g. `hin_Deva`, `tel_Telu`, `hin_Latn`.
- Text mostly in native script → **FTN** (FastText, native-script model).
- Text mostly in Latin letters → **FTR** (FastText, romanized model); if its confidence is below 0.6, the
  **IndicBERT** classifier decides.
- `get_indiclid()` loads all three once; the BERT checkpoint is a complete pickled model, loaded with
  `weights_only=False` (PyTorch's newer default refuses pickles).

### 12.8 Transliteration

`transliterate_file(csv, lang_code, llm)`: converts romanized words to native script with AI4Bharat IndicXlit if
it is installed (it needs `fairseq`, which has no Apple Silicon/Windows build), asks the local LLM twice. Only the unique Roman-script words are sent, 40 at a time, with two differently worded
prompts, each answered with a JSON word map (reply length capped). A word is replaced only if both answers are the
same, in the language's script, and, when IndicXlit is available (installed on Linux, or the Docker image built
from `docker/indicxlit` via `xlit_candidates`), among IndicXlit's top 5 spellings. Only those words are replaced,
so nothing else in the transcript can change; `[tags]` stay as they are. (Asking the LLM to rewrite a whole
transcript made it generate for over 14 minutes without finishing.) Writes `<input>_transliterated.csv` with
`roman_words_left` and `transliteration_words.csv` (both LLM answers, IndicXlit's candidates and the decision per
word). Needs reference transcripts; skipped for English.

---

## 13. `summarize.py`: the one-row-per-file table

- Starts from the list of audio files in each `<language>/audios` folder, so every file gets a row even if a
  check failed for it.
- `COLUMNS` says which columns to take from which result file and what to call them in the summary
  (`sample_rate_check.csv: Status → Sample_Rate_Status`, ...). Matching is case-insensitive and supports
  wildcards (`Is_*` matches `Is_Tamil`).
- Each result file is joined on the file name (`Filename`, `filename` and `File Name` are all recognized).
  Missing files or columns leave empty cells.
- Saved as UTF-8 with a byte-order mark (`utf-8-sig`), so Excel displays Indic text correctly.

Run it alone: `.venv/bin/python summarize.py <results folder>`.

---

## 14. Small files

- **`my_secrets.py`**: only `PROMPT` is used, and only by `python llm.py`. The `HF_TOKEN`, `GPT_API_KEY` and
  `GROQ_KEY` variables are left over from the original code and unused; leave them empty.
- **`.gitignore`**: tells git never to commit `.venv/`, `models/`, `pipeline.log`, `__pycache__/`, `log.txt`
  or results folders (`*_qc/`). These are large, machine-specific or private.
- **`.gitattributes`**: fixes line endings per file type, whatever the cloning computer's git settings:
  `*.sh` and `*.py` always LF (bash refuses CRLF: `bad interpreter: /usr/bin/env bash^M`), `*.bat` and `*.ps1`
  always CRLF (Windows tools expect it).
- **`env/denv.yml`**: the original authors' conda environment for Linux + CUDA. Not used by this edition (use
  `requirements.txt` through the setup scripts); kept for reproducing the paper.

---

## 15. Commands you will type, explained

### Getting the code

| Command | What it does |
|---|---|
| `git clone https://github.com/ambermanijha-ux/SpeechQM-Agent-2.git` | Downloads the repository into a new folder `SpeechQM-Agent-2` |
| `cd SpeechQM-Agent-2` | Enter that folder |

### Setup and running

| Command | What it does |
|---|---|
| `./setup.sh` | One-time install on Mac/Linux (section 3). `./` means "the file in this folder". |
| `powershell -ExecutionPolicy Bypass -File setup.ps1` | One-time install on Windows (section 4) |
| `./run.sh /path/to/audio` / `run.bat C:\path\to\audio` | Run all checks (sections 5, 7) |
| `./run.sh /path --language Telugu` | Audio not in language-named folders is Telugu |
| `./run.sh /path --prompt "check sample rate and count speakers"` | Only the checks you describe |
| `./run.sh /path --out /elsewhere` | Put results somewhere else |
| `./run.sh /path --token hf_xxx` / `--no-token` / `--yes` | Use this token / no token / never ask |
| `./run.sh --help` | List the options |
| `chmod +x setup.sh run.sh` | Only if you see "Permission denied": marks the scripts as executable |

### Hugging Face

| Command | What it does |
|---|---|
| `.venv/bin/hf auth login` | Log in by pasting a token (alternative to being asked during a run) |
| `.venv/bin/hf auth whoami` | Shows which account the saved token belongs to |
| `.venv/bin/hf auth logout` | Removes the saved token |
| `HF_HUB_OFFLINE=1 ./run.sh /path` | Force offline mode: use only downloaded models |

On Windows replace `.venv/bin/` with `.venv\Scripts\`.

### Ollama

| Command | What it does |
|---|---|
| `ollama list` | Models Ollama has downloaded (should include `qwen3:8b`) |
| `ollama ps` | Models currently loaded in memory |
| `ollama pull qwen3:8b` | Download a model by hand |
| `ollama rm <model>` | Delete a model to free disk space |
| `ollama serve` | Start the Ollama server in this terminal (Linux, or if the app is not running) |
| `open -a Ollama` | Mac: start the Ollama app |
| `curl http://localhost:11434/api/version` | Is the server answering? |
| `brew services start ollama` | Mac with Homebrew-installed Ollama only: run it as a background service. (Not needed when Ollama is the app: then `brew` reports "Formula ollama is not installed", which is harmless.) |

### Environment variables you may set before a run

Prefix them on Mac/Linux (`PLANNER_MODEL=qwen3:4b ./run.sh ...`); on Windows run
`set PLANNER_MODEL=qwen3:4b` first.

| Variable | Effect |
|---|---|
| `PLANNER_MODEL`, `CODER_MODEL` | Use a different Ollama model (e.g. smaller for low-memory machines) |
| `OLLAMA_BASE_URL` | Use an Ollama server on another computer |
| `SPEECH_BACKEND=local` | Never use pyannote |
| `SPEAKER_MATCH_THRESHOLD=0.7` | Stricter "same speaker across files" |
| `DEFAULT_LANGUAGE` | Language when nothing else says |
| `HF_TOKEN` | Token for this run only |
| `HF_HUB_OFFLINE=1` | Offline mode |

### Updating your copy and publishing changes (git)

| Command | What it does |
|---|---|
| `git pull` | Fetch and apply the latest version from GitHub |
| `git status` | What changed locally |
| `git add -A` | Stage every change for the next commit |
| `git commit -m "message"` | Save the staged changes as a commit |
| `git push` | Upload commits to GitHub (`origin` = your repository) |
| `gh auth login --web --git-protocol https` | Log the GitHub CLI in through the browser; `gh auth setup-git` then lets `git push` use that login |

### The original prompt mode

| Command | What it does |
|---|---|
| `.venv/bin/python llm.py` | Runs `PROMPT` from `my_secrets.py` (must contain a quoted absolute folder path) |
| `.venv/bin/python llm.py 2>&1 \| tee log.txt` | Same, also saving everything printed to `log.txt` (`2>&1` includes error output; `tee` shows and saves) |

---

## 16. Where things are stored on disk

| What | Mac / Linux | Windows |
|---|---|---|
| Python environment | `<project>/.venv` | `<project>\.venv` |
| Hugging Face models + saved token | `~/.cache/huggingface/` | `%USERPROFILE%\.cache\huggingface\` |
| IndicLID, ECAPA | `<project>/models/` | `<project>\models\` |
| Ollama models | `~/.ollama/models` | `%USERPROFILE%\.ollama\models` |
| uv's Python downloads and package cache | `~/.local/share/uv`, `~/.cache/uv` | `%APPDATA%\uv`, `%LOCALAPPDATA%\uv` |
| Run log | `<project>/pipeline.log` | `<project>\pipeline.log` |
| Results | `<audio folder>_qc/` (or `--out`) | same |

To uninstall everything: delete the project folder, the Hugging Face cache, and the Ollama models (`ollama rm`)
or Ollama itself.

---

## 17. Reading the console output and `pipeline.log`

Console, in order:
```
=== 1/5 Sorting audio by language
hindi        6 audio,   3 with transcript -> /data/calls_qc/hindi
=== 2/5 Hugging Face access
Hugging Face: logged in as <user>
=== 3/5 Local LLM (Ollama)
=== 4/5 Models (downloaded once, then cached)
Device: mps  speech backend: pyannote
=== 5/5 Quality checks
--- Hindi: /data/calls_qc/hindi/audios
Using structure: [[1, 2, 7, 8, 11, 19, 20, 21], [5, 6, 10, 12, 13, 15, 16, 17, 22], [9, 23, 24]]
<ASCII graph>                                 the checks as boxes, top to bottom = execution order
Transcribing ... using ai4bharat/indicwav2vec-hindi on mps
TRANSCRIPT: [...]                              the recognised words (first ones)
Processed x.wav: 2 unique speakers, ...
Final Pipeline Results:                        one status line per check
--- Hindi done in 167 s
Summary (one row per file): /data/calls_qc/qc_summary.csv
```

`pipeline.log` (in the project folder) adds, with timestamps:
- `Selected Tasks: ...` and every `Prompt Checker` round: what the LLM chose and why.
- `Added tasks named in the prompt ...` / `Added dependency tasks ...` / `No reference transcripts: skipping ...`:
  the deterministic corrections.
- `agent attempt N for <file>.csv:` followed by **the full generated script** and its output.
- Model loading, warnings and errors of every check.

Search it for `ERROR` first when something looks wrong.

---

## 18. Troubleshooting by symptom

| Symptom | Cause and fix |
|---|---|
| `Permission denied` on `./setup.sh` | Executable bit missing (e.g. copied via a zip): `chmod +x setup.sh run.sh` |
| `bad interpreter: /usr/bin/env bash^M` | Windows line endings in a `.sh` file (copied, not cloned): `sed -i '' 's/\r$//' *.sh` on Mac, `sed -i 's/\r$//' *.sh` on Linux |
| PowerShell: "running scripts is disabled" | Start it exactly as `powershell -ExecutionPolicy Bypass -File setup.ps1` |
| `Run setup.sh first` / `Run setup.ps1 first` | `.venv` missing: run setup |
| `Ollama is not running at http://localhost:11434` | Start the Ollama app / `ollama serve` |
| `That Hugging Face token was rejected` | Token deleted or mistyped: make a new **Read** token |
| Gated model links keep being listed | Accept the terms while logged in to the *same* account as the token, then press Enter |
| `... is not inside a folder named after a language` | Add `--language <name>` or rename folders |
| `Device: cpu` although you have a GPU | NVIDIA: driver missing when setup ran → install the driver, run setup again. Mac Intel: no MPS, CPU is expected |
| Very slow first run | Model downloads (~8-10 GB); later runs are much faster |
| Out of memory | Close other programs; `PLANNER_MODEL=qwen3:4b CODER_MODEL=qwen3:4b` |
| A check shows `Error: Failed to generate ...csv` | The code-writing agent failed 3 times: see its scripts in `pipeline.log`; re-run, results vary slightly between runs |
| WER close to 1 on good audio | Code-mixed references (English in Latin script) or weak language model; see README §6 |
| `.mp3` files marked corrupt / invalid extension | By design of the SpeechQM spec (WAV only) |
