# SpeechQM-Agent: How the Code Works and How to Run It

This guide covers every script in the repo: what each one is responsible for, the order in which things execute, how to run the pipeline, and **what you must change before your first run**. (`simplified_methodology.md` covers the research and model side. This file is the operator's manual.)

---

## 1. What is in the repo

| File | Role | Do you run it? |
|---|---|---|
| `llm.py` | **The entry point and orchestrator.** It parses your prompt, asks the LLM which checks to run, verifies that choice, orders the checks, builds a LangGraph DAG, runs it and prints a summary. It also defines all 24 task "agents" (nodes). | **Yes.** It is the only script you run. |
| `utility_functions.py` | **Library of predefined tools** used by the nodes in `llm.py`: ASR transcription, VAD silence, speaker diarization, upsampling detection, forced alignment and CTC score, IndicLID language ID, and transliteration. | No. `llm.py` imports it. |
| `my_secrets.py` | **Configuration:** API keys (`HF_TOKEN`, `GPT_API_KEY`, `GROQ_KEY`) and the natural-language `PROMPT` that says what to check and where. | No. You edit it. |
| `env/denv.yml` | Conda environment export (Python 3.10.15, torch 2.5.1, transformers 4.44.2, langchain 0.3.23, langgraph 0.4.1, pyannote-audio 3.3.2, and others). **It is a Linux build.** | Used once to create the environment. |
| `components/*.jpg/png` | Figures for the README. | No |
| `ACL2026_Rolling_Out_Data_Quality.pdf` | The paper. | No |
| `pipeline.log` *(created at runtime)* | Log file written by `llm.py` (`logging.basicConfig`, `llm.py:31`). | Output |

There is **one run command**. No other scripts need to run before or after it.

---

## 2. The big picture

```
my_secrets.py (PROMPT + keys)
        │
        ▼
python llm.py
        │
        ├── import time: loads pyannote VAD model, builds LLM + ReAct agent
        │
        └── main(PROMPT)
              1. parse_prompt          → audio_dir, ground_truth_csv, lang_code
              2. select_tasks          → LLM picks task numbers from 1–24
              3. prompt_checker_agent  → LLM verifies (Correct / Missing / Extra), ≤3 rounds
              4. topological_sort_tasks→ LLM returns groups, e.g. [[1,2],[13,22],[24]]
              5. build_graph_from_structure → LangGraph StateGraph
              6. main_graph.invoke     → runs groups in order, tasks inside a group in parallel
              7. print summary         → each task's output string (mostly "CSV saved at …")
```

---

## 3. Execution order in detail

### 3.0 Import time (before `main` runs)

Python executes module-level code as soon as `llm.py` starts. This matters because these steps run **every time, even if your prompt doesn't need them**:

1. `llm.py:10` imports `my_secrets`. If a name is missing there, the run fails immediately.
2. `llm.py:18` imports `utility_functions`, which runs the following at import:
   - `utility_functions.py:180` calls `Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=HF_TOKEN)` to download the **gated** pyannote VAD model. This needs a valid `HF_TOKEN` and requires that you accepted the model's terms on Hugging Face. Otherwise the whole script fails here.
   - `utility_functions.py:181-186` builds the `VoiceActivityDetection` pipeline.
3. `llm.py:31` sets up logging to `pipeline.log` in the **current working directory**.
4. `llm.py:32-35` creates the LLM client: `ChatOpenAI(model_name="gpt-4.1-mini", openai_api_key=GPT_API_KEY)`.
5. `llm.py:125-138` creates a **LangChain ReAct agent** with a single tool, `python_repl`. Several tasks use this agent: the LLM *writes Python code and executes it on your machine*.

### 3.1 `parse_prompt` (`llm.py:43`)

- It uses regex to pull three things out of `PROMPT`:
  - `audio_dir`: the first quoted string that **starts with `/`**, so it must be an absolute Unix-style path.
  - `ground_truth_csv`: a quoted `/…/*.csv` path, if you give one.
  - `lang_code`: one of `bn gu hi kn ml mr pa sd si ta te ur` as a standalone word.
- If regex finds neither path, it asks the LLM to extract them as JSON.
- Each path is checked with `os.path.isdir` / `os.path.isfile`. An invalid path becomes `None`.
- If both are `None`, `main` prints `Error: Please provide audio directory…` and stops.

### 3.2 `select_tasks` (`llm.py:140`)

The LLM gets the menu of 24 tasks plus keyword rules (for example, "duplicate" means 23, "IndicLID" means 1+15, "durations" means 2+22). It returns a comma-separated list of numbers. Any digits in the reply are taken as task IDs.

### 3.3 `prompt_checker_agent` (`llm.py:187`)

A second LLM call grades the selection and returns:
- **Correct**: the selection is accepted.
- **Missing**: the missing tasks are added and `select_tasks` is re-run with feedback.
- **Extra**: the extra tasks are removed.

It runs at most 3 rounds. One rule it enforces through the prompt: **if task 9, 23 or 24 is selected, task 16 must be too.**

### 3.4 `topological_sort_tasks` (`llm.py:304`)

The LLM gets the dependency rules and returns a list of lists, for example `[[1, 2], [13, 22], [24]]`. Each inner list is a parallel stage. The reply is parsed with `ast.literal_eval`, with a regex fallback. If parsing fails, all tasks go into **one group** (no ordering), which can break dependent tasks.

Dependency rules given to the LLM:

```
1  → 3, 4, 5, 6, 10, 13, 14, 15, 16     (need ASR output)
2  → 12, 22                             (need num_speakers.csv)
16 → 9, 23, 24                          (need normalized_list.csv)
```

### 3.5 `build_graph_from_structure` (`llm.py:1176`)

It builds a LangGraph `StateGraph(CombinedStateDict)`:
`start → every node in group 0 → every node in group 1 → … → END`.
Nodes in the same group run in the same LangGraph super-step, which means in parallel. The ASCII drawing of the graph printed to the console needs the `grandalf` package, which is in the environment.

### 3.6 Graph execution (`llm.py:1234`)

Every node takes the shared state (`audio_dir`, `ground_truth_csv`, `lang_code`, `user_prompt` plus earlier outputs) and returns a dict containing one output string. Results pass between tasks in two ways:
- **Through state:** task 1 stores the ASR CSV path in key `A`, and later tasks read `state['A']`.
- **Through fixed file paths** in `results/`, for example task 22 reads `results/num_speakers.csv`.

### 3.7 Summary

All non-input state keys are printed, truncated to 100 characters. The CSVs are the real output.

---

## 4. The 24 tasks: what each one does, how, and where it writes

Legend for **How**: **Py** = fixed Python code in the repo. **LLM** = calls the LLM directly once per row. **Agent** = the ReAct agent writes and executes Python code at run time, so results can vary between runs.

`R/` = `<parent of audio_dir>/results/`  ·  `A/` = `audio_dir` itself

| # | Task | Node function (`llm.py`) | How | Needs | Writes |
|---|---|---|---|---|---|
| 1 | ASR transcription | `transcription_func` → `transcribe_folder_to_csv` | Py: wav2vec2 (`ai4bharat/indicwav2vec-hindi`) | audio | `R/indicconf_hypothesis.csv` (`Filename, Indiconformer_Hypothesis`) |
| 2 | Number of speakers + per-speaker duration | `num_speaker_func` → `save_num_speakers` | Py: `pyannote/speaker-diarization-3.1` | audio, HF token | `R/num_speakers.csv` |
| 3 | Transcript quality | `transcript_quality_agent` | Agent (calls `utility_functions.transcript_quality`) | task 1 | `transcript_quality.csv` |
| 4 | Character list | `character_agent` | Agent | task 1 | `R/character_list.csv` |
| 5 | Vocabulary list | `vocab_agent` | Agent | task 1 | `R/vocab_list.csv` |
| 6 | Language verification (Devanagari script) | `language_verification_agent` | Agent | task 1 | `R/language_verification.csv` |
| 7 | Audio length | `audio_length_agent` | Agent | audio | `A/audio_length.csv` |
| 8 | Silence (VAD) | `silence_vad_func` → `process_folder_vad` | Py: pyannote segmentation-3.0 | audio | `R/vad_silence_stats.csv` |
| 9 | English word count (code-mix) | `english_word_count_agent` | LLM per row | task 16 | `R/english_word_count.csv` |
| 10 | CTC score + alignment | `ctc_score_agent` → `process_audio_directory` | Py: wav2vec2 (`Harveenchadha/vakyansh-wav2vec2-hindi-him-4200`) | audio + transcript CSV | `R/ctc_scores.csv` |
| 11 | Upsampling (8 kHz→16 kHz) detection | `upsampling_agent` → `check_upsampling_folder` | Py: FFT energy in the 4–8 kHz band | audio | `R/upsampling_check.csv` |
| 12 | New vs. old speaker | `valid_speaker_agent` | Agent | task 2 | `R/valid_speaker.csv` |
| 13 | Domain classification | `domain_checker_agent` | LLM per row | task 1 | `R/domain_check.csv` |
| 14 | Audio↔transcript forced alignment | `audio_transcript_matching_agent` | Agent (calls `force_alignment_and_ctc_score`) | task 1 | `R/audio_transcript_matching.csv` |
| 15 | Language ID (IndicLID) | `language_identification_indiclid_agent` | Py: IndicLID (FastText + IndicBERT) | task 1, local IndicLID models | `R/indiclid_language_identification.csv` |
| 16 | Normalization (strip HTML, `[TAGS]`, symbols) | `normalization_remove_tags_agent` | Agent | **`R/indicconf_hypothesis-gt.csv` (you provide it)** | `R/normalized_list.csv` |
| 17 | LLM-as-judge fluency score (0–10) | `llm_score_agent` | Agent | task 1 | `R/llm_scores.csv` |
| 18 | Transliteration Roman→native | `transliteration_agent` → `transliterate_file` | Py: AI4Bharat `XlitEngine` | `ground_truth_csv` + `lang_code` in prompt | `<gt>_transliterated.csv` |
| 19 | Corruption (0-byte / mp3) | `corruption_agent` | Agent | audio | `A/corruption_check.csv` |
| 20 | Extension/format (`.wav` only) | `extension_agent` | Agent | audio | `A/audio_format_check.csv` |
| 21 | Sample rate (=16 kHz) | `sample_rate_agent` | Agent | audio | `A/sample_rate_check.csv` |
| 22 | Total duration per speaker | `speaker_duration_agent` | Py | task 2 | `R/speaker_durations.csv` |
| 23 | Duplicate utterances | `utterance_duplicate_checker_agent` | Py (pandas `duplicated`) | task 16 | `R/duplicate_utterances.csv` |
| 24 | WER | `wer_computation_agent` | Py (`jiwer.wer`) | task 1 + task 16 | `R/wer.csv` |

The mapping from task number to function is `node_map` at `llm.py:1148`.

---

## 5. Expected folder layout

```
/abs/path/my_dataset/
├── audios/                         ← put this path in PROMPT
│   ├── utt_001.wav                 (16 kHz mono .wav works best; .mp3 is converted on the fly by task 1)
│   └── ...
└── results/                        ← created automatically; ALL main outputs go here
    ├── indicconf_hypothesis-gt.csv ← YOU provide this if you want tasks 16 / 9 / 23 / 24
    └── (generated CSVs)
```

**Ground-truth file for normalization (`results/indicconf_hypothesis-gt.csv`):** two columns, `filename,ground_truth`. Task 9 also reads `ground_truth` from the normalized output.

**Important for WER (task 24):** it pairs references and hypotheses **by row position, not by filename** (`llm.py:1120-1125`). Task 1 lists files with `os.listdir`, whose order is not guaranteed. Your `-gt.csv` rows must be in **the same order and count** as `indicconf_hypothesis.csv`. If they are not, WER is computed on mismatched pairs, or the task errors on a row-count mismatch. The safe workflow is to run task 1 first, then sort or merge your ground truth to match the `Filename` order of that file.

---

## 6. How to run

### 6.1 Set up the environment (Linux or WSL)

`env/denv.yml` pins Linux-only packages (`ld_impl_linux-64`, `libgcc`, CUDA libs), and the code assumes `/`-style paths. **On Windows, run inside WSL2 (Ubuntu)** rather than native Windows.

```bash
conda env create -n speechqm -f env/denv.yml      # or: conda create -n speechqm python=3.10 && conda env update -n speechqm -f env/denv.yml
conda activate speechqm
```

If the full export fails to solve, install the core packages by hand:
`torch==2.5.1 torchaudio==2.5.1 transformers==4.44.2 langchain==0.3.23 langchain-community langchain-experimental==0.3.4 langgraph==0.4.1 pyannote-audio==3.3.2 fasttext jiwer librosa pydub soundfile pandas openai grandalf ai4bharat-transliteration`, plus system `ffmpeg` (needed for pydub's mp3 conversion).

A GPU is optional. Diarization, alignment and IndicLID use CUDA if it is available. ASR (task 1) always runs on CPU.

### 6.2 Fill in `my_secrets.py`

```python
HF_TOKEN    = "hf_..."     # needed for every run (pyannote loads at import)
GPT_API_KEY = "sk-..."     # OpenAI key for gpt-4.1-mini
GROQ_KEY    = ""           # only if you switch to Groq (the README calls it GROQ_API_KEY; the code imports GROQ_KEY)
PROMPT = "Check sample rate, corruption and the domain of the transcripts of the audio files located at '/abs/path/my_dataset/audios/'."
```

### 6.3 Run

```bash
cd /path/to/SpeechQM-Agent
python llm.py 2>&1 | tee log.txt
```

- The console shows the chosen structure (`Using structure: [[...]]`), the ASCII DAG, the agent's verbose reasoning and the final summary.
- `pipeline.log` (in the directory you ran from) has the detailed log, including the selected tasks and each prompt-checker round.
- Results are in `<parent of audios>/results/`, plus the few audio-folder CSVs listed in section 4.

### 6.4 Writing good prompts

- Always quote the **absolute** audio path, with a trailing `/` being fine: `'/data/x/audios/'`.
- Name the checks with the keywords the selector understands: *ASR / transcription, number of speakers, durations, new or old speakers, silence / VAD, upsampling, sample rate, corruption, extension, domain, IndicLID / language identification, verify language, CTC score, forced alignment, normalization / remove HTML tags, English word count, duplicates, WER, LLM score, transliteration (+ `hi` and a quoted `.csv` path)*.
- Check `pipeline.log` → `Selected Tasks` / `Topological Structure` to confirm the LLM picked what you meant.

Examples (taken from `my_secrets.py` / `llm.py`):
```
"Perform QC Checks, Sample rate check, VAD silence detection, upsampling, IndicLID language identification, calculate speaker duration, verify if speakers are new or old in :'/abs/.../QC1/audios/'"
"verify language as Hindi (' hi '), CTC SCORE, WER computation, normalization, calculate LLM score, check domain, english word counter, and check for utterance duplicate in :'/abs/.../QC2/audios/'"
```

---

## 7. What to change before you run on your own data ⚠️

### 7.1 Must change (the run fails or gives wrong results otherwise)

| # | What | Where | Why / what to do |
|---|---|---|---|
| 1 | API keys + prompt | `my_secrets.py` | Add `HF_TOKEN` and `GPT_API_KEY`, and point `PROMPT` at **your** absolute audio path. The current path (`/home/priya/...`) is the authors' machine. |
| 2 | Accept gated HF models | huggingface.co | While logged in with the same account as `HF_TOKEN`, accept the terms for **`pyannote/segmentation-3.0`** and **`pyannote/speaker-diarization-3.1`**. Without this, the script fails at import. |
| 3 | IndicLID model path | `utility_functions.py:494` | `absolute_path = "/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/trash"` is hard-coded. Download the IndicLID models (`indiclid-ftn/model_baseline_roman.bin`, `indiclid-ftr/model_baseline_roman.bin`, `indiclid-bert/basline_nn_simple.pt` from AI4Bharat's IndicLID release) and set this to your folder. Only needed for task 15. |
| 4 | Transliteration import | `utility_functions.py:25` | `from ai4bharat.transliteration import XlitEngine` is **commented out**, so task 18 always returns `Error: name 'XlitEngine' is not defined`. Uncomment it. The package is already in the environment. |
| 5 | ASR language | `llm.py:482` | `transcribe_folder_to_csv(audio_dir, source_language="Hindi")` is hard-coded, and the `lang_code` from your prompt is ignored. For Tamil, Marathi, Telugu or Sanskrit, change it to one of the keys in `utility_functions.py:88-94`, or wire it to `state['lang_code']`. |
| 6 | Stale-result caching | `results/` folder | Tasks 1, 2, 8, 15 and 16 **skip work if their output CSV already exists** (for example `indicconf_hypothesis.csv`, `num_speakers.csv`, `vad_silence_stats.csv`, `normalized_list.csv`, `indiclid_language_identification.csv`). When you change datasets or re-run after a fix, **delete or rename the old `results/` folder** first. |
| 7 | Ground-truth CSV (tasks 16/9/23/24) | `results/indicconf_hypothesis-gt.csv` | Provide it with `filename,ground_truth` columns, in the same row order as the ASR output (see section 5). |

### 7.2 Should change to fit your dataset (assumptions are hard-coded for Hindi)

| What | Where | Current assumption |
|---|---|---|
| Domain labels | `llm.py:747-748` | 10 fixed IndicVoices-style domains (`Conversation`, `GK Questions`, `KYP - Cooking`, …). Replace them with your own taxonomy. |
| Domain prompt language | `llm.py:751` | "You are a **Hindi** language expert". |
| Script check | `llm.py:587-606` (task 6) | Only checks **Devanagari** (U+0900–U+097F). Change the Unicode range for other scripts. |
| LLM judge language | `llm.py:939` | "We are evaluating the accuracy of **hindi**". |
| Alignment/CTC model | `utility_functions.py:290` | Hindi-only `vakyansh-wav2vec2-hindi-him-4200`. |
| Expected sample rate | `llm.py:461` | Pass only if exactly **16000 Hz**. |
| Allowed extension | `llm.py:431-435` | Only `.wav` is valid. |
| Corruption rule | `llm.py:408-409` | Every `.mp3` is flagged "corrupt", as well as 0-byte files. |
| CTC status bands | `llm.py:652` | Good > 0.7, Medium > 0.5, else Poor. |
| Upsampling threshold | `utility_functions.py:433` | `threshold_ratio=0.02` (share of energy in 4–8 kHz). |
| Transcript-quality rule | `utility_functions.py:281-288` | Passes if > 3 words and < 30% repeated words. |
| VAD hyper-params | `utility_functions.py:182-185` | `min_duration_on/off = 0.0`. |

### 7.3 Optional: change the LLM

- **OpenAI model:** edit `model_name` at `llm.py:33`. Consider `temperature=0` for more repeatable task selection.
- **Groq:** comment out `llm.py:32-35` and uncomment `llm.py:36-42` (`llama-3.1-8b-instant`). Put the key in `GROQ_KEY`.
- **Local / Ollama:** `ChatOllama` is imported (`llm.py:12`) but not wired up. Replace the `llm = …` line with `llm = ChatOllama(model="llama3.1")` with a local Ollama server running. Smaller models are noticeably worse at the planner, checker and ordering steps and at writing agent code.

---

## 8. Known issues and gotchas in the current code

Read these so that surprising output doesn't confuse you. None of them stop the pipeline as a whole: each node catches its own errors and returns an `Error: …` string.

1. **LLM-written code runs on your machine.** Tasks 3, 4, 5, 6, 7, 12, 14, 16, 17, 19, 20 and 21 have the ReAct agent write and execute Python through `PythonREPL` with no sandbox. Run the pipeline on a machine or account where that is acceptable, and keep backups of your data.
2. **Misleading "Invalid" status for tasks 19/20/21.** `corruption_agent` reads `response.get("Success")` and `extension_agent`/`sample_rate_agent` check `"Success" in response` (a dict), so they nearly always report `Invalid` even when they succeed. **Look at the CSVs** they write in the audio folder, not the summary line.
3. **Agent tasks are strict about the reply.** Several nodes count success only if the agent's final answer is exactly `"Success"` *and* the file exists at the expected path. A correct CSV saved somewhere else, or a wordier reply, is reported as an error.
4. **Prompt and path mismatches in some agent tasks:**
   - Task 12 tells the agent that `num_speakers.csv` is in `audio_dir`, but it is actually in `results/`. The success check looks in `results/`.
   - Task 16 tells the agent the file is `indicconf_hypothesis.csv` in `audio_dir`, while the node checks for `results/indicconf_hypothesis-gt.csv`.
   - Task 17 says the CSV is in `audio_dir` (it is in `results/`), and the generated script has **no LLM access**, so the "LLM-as-judge" scores come from whatever heuristic the agent invents. The prompt also has a typo (`[pythom_repl]`).
   - Task 3 defaults to `audio_dir/indicconf_hypothesis.csv` instead of `results/`.
   In practice the agent often finds the right file anyway, but if these tasks fail, these mismatches are the first place to look.
5. **Task 10 (CTC) column names.** `process_audio_directory` (`utility_functions.py:386-387`) reads columns `filename` and `ground_truth`, but the task-1 ASR CSV has `Filename` and `Indiconformer_Hypothesis`. As written, task 10 on ASR output fails with `Error: No results generated`. Either give it a CSV with `filename,ground_truth` or change those two lines to match.
6. **Output keys missing from the state schema.** Tasks 3, 23 and 24 return keys (`transcript_quality_output`, `utterance_duplicate_output`, `wer_output`) that are not the names in `CombinedStateDict` (`llm.py:96-123`). Their CSVs are still written, but their status may not appear in the final summary.
7. **Task 1 reloads the ASR model for every file** (`utility_functions.py:98-99`), and IndicLID is re-loaded for every row (`utility_functions.py:34`). This is slow on large datasets. Hoisting the model load outside the loop is an easy speed-up.
8. **Task 1 always runs on CPU** (the model is never moved to GPU).
9. **The final ordering comes from the LLM.** If `topological_sort_tasks` returns something unparseable, all tasks run in one parallel group, and dependent tasks fail because their input CSVs don't exist yet. Check `Using structure:` in the console.
10. **The audio folder gets extra files.** Tasks 7, 19, 20 and 21 write CSVs *into* `audio_dir`. Task 1 converts `.mp3` to a temporary `.wav` in the audio folder and deletes it afterwards.
11. `get_required_inputs` (`utility_functions.py:601`) is imported but never used, so there is no up-front check that a task has the inputs it needs.

---

## 9. Quick pre-flight checklist

- [ ] Running on Linux or WSL with the conda env active
- [ ] `HF_TOKEN` and `GPT_API_KEY` set in `my_secrets.py`
- [ ] pyannote `segmentation-3.0` and `speaker-diarization-3.1` terms accepted on HF
- [ ] `PROMPT` names the checks and the **absolute, quoted** audio path
- [ ] Old `results/` folder removed if you are re-running
- [ ] If using WER, normalization, English word count or duplicates: `results/indicconf_hypothesis-gt.csv` in place, rows in the same order as the ASR output
- [ ] If using IndicLID: models downloaded, path fixed at `utility_functions.py:494`
- [ ] If using transliteration: `XlitEngine` import uncommented, `lang_code` and a quoted `.csv` path in the prompt
- [ ] Non-Hindi data: ASR language, alignment model, script check, domain list and judge prompt updated
- [ ] `python llm.py 2>&1 | tee log.txt`, then check `pipeline.log` and `results/`
