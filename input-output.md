# SpeechQM-Agent: Input, Output, and Task Selection

Short answers: the input is one natural-language prompt that contains an audio folder path. The output is a set of CSV files, one per check, plus short status lines printed to the console. Each prompt runs only the checks it asks for, not all 24.

## 1. Input

**What you give it:** a single text prompt, set as `PROMPT` in `my_secrets.py`. `parse_prompt` (`llm.py:59`) pulls three things out of it:

- **The audio folder.** The first absolute path that starts with `/`. If no audio folder and no CSV path are found, it asks the LLM to extract them instead.
- **A ground-truth CSV path** (optional). In practice only task 18 (transliteration) uses this.
- **A language code** (optional): `hi`, `ta`, `bn` and so on. Also only used by task 18.

**Folder layout the code expects:**

```
<parent>/audios/*.wav | *.mp3      ← the path in your prompt
<parent>/results/                  ← every task reads and writes its CSVs here
```

Transcription is hardcoded to Hindi using IndicConformer (`llm.py:498`). If a result CSV already exists, most tasks skip the work and reuse the old file, so stale results carry over between runs.

### When only audio is present

The key point: **the system's "transcript" is the ASR output, not a human label.** Task 1 transcribes the audio into `results/indicconf_hypothesis.csv`. Nearly every transcript task then reads that file through `state.get('A', ...)`. With audio only, the tasks fall into three groups:

| Status | Tasks |
|---|---|
| **Work on the audio itself** | 1 ASR, 2 speaker count, 7 audio length, 8 VAD silence, 11 upsampling, 12 new/old speaker, 19 corruption, 20 extension, 21 sample rate, 22 speaker durations |
| **Run, but on the ASR's own output** (they check the model, not the data's labels) | 4 characters, 5 vocab, 6 Devanagari check, 13 domain, 15 IndicLID, 17 LLM score, 10/14 CTC and forced alignment. The CTC tasks align the ASR output to its own audio, so the scores come out inflated and mean little. |
| **Fail without a human transcript** | **16 normalization** needs a hand-made `results/indicconf_hypothesis-gt.csv` with `filename, ground_truth` columns. **9, 23 and 24** (English word count, duplicates, WER) read the normalized output of 16, so they fail too. **18** needs a ground-truth CSV and a language code. |

The README says to put "transcript.csv" in `results/`, but the code actually looks for `indicconf_hypothesis-gt.csv`.

## 2. Output

- **Console:** the DAG printed as ASCII art, then a "Final Pipeline Results" section with one status line per task (`CSV saved at: ...`, `Success`/`Invalid`, or `Error: ...`). Everything is also logged to `pipeline.log`.
- **Files:** one CSV per task. The README mentions a structured report, but no combined report is produced.

| Task | File | Key columns |
|---|---|---|
| 1 | `indicconf_hypothesis.csv` | Filename, Indiconformer_Hypothesis |
| 2 | `num_speakers.csv` | File Name, Number of Speakers, Speaker Durations (JSON, in hours) |
| 6 | `language_verification.csv` | Filename, Transcription, Is_Devanagari |
| 8 | `vad_silence_stats.csv` | Filename, Total Silence (s) |
| 10 | `ctc_scores.csv` | …, CTC_Score, CTC_Status (Good >0.7 / Medium >0.5 / Poor) |
| 11 | `upsampling_check.csv` | Filename, Is_Upsampled |
| 12 | `valid_speaker.csv` | Filename, Speaker_Status (New/Old), Common_File |
| 13 | `domain_check.csv` | + domain |
| 15 | `indiclid_language_identification.csv` | Language_Code, Confidence |
| 17 | `llm_scores.csv` | LLM_Score (0–10), Evaluation_Comment |
| 21 | `sample_rate_check.csv` | Sample_Rate, Status (Pass if 16 kHz) |
| 22 | `speaker_durations.csv` | Speaker, Total_Duration_Hours |
| 23 | `duplicate_utterances.csv` | column_name, utterance, count |
| 24 | `wer.csv` | Reference, Hypothesis, WER |

Some bugs showed up while reading the code (nothing was run):

- **Wrong folder:** the tasks that write their own scripts (7, 17, 19, 20, 21) save their CSVs in the audio folder, not `results/`.
- **Task 12 can't find its input:** it looks for `num_speakers.csv` in the audio folder, but task 2 writes it to `results/`.
- **Tasks 19, 20 and 21 always report `Invalid`:** they test the agent's response dictionary the wrong way, so the console status says `Invalid` even when every file passed.

## 3. Does every prompt run all 24?

**No. Each prompt runs only a subset.** The selection works in three steps:

1. **`select_tasks`** (`llm.py:156`): the LLM reads the prompt and picks task numbers using keyword rules.
2. **`prompt_checker_agent`** (`llm.py:203`): a second LLM pass, up to 3 rounds, adds missing tasks or removes extra ones.
3. **`topological_sort_tasks`** (`llm.py:320`): the chosen tasks are sorted by dependency into parallel groups, and only those nodes are built into the graph.

Some tasks get pulled in automatically:

- Task 1 whenever an audio folder is given.
- 2 and 22 for "speakers" or "durations".
- 2 and 12 for "new or old".
- 1 and 15 for "IndicLID".
- 16 whenever 9, 23 or 24 is chosen.

There is no "run everything" mode; you'd have to name all the checks in the prompt. For example, the default prompt ("Classify domain…") should end up as about `[[1], [13]]`: ASR first, then domain classification.

## Note for Windows

The path regex only matches paths that start with `/`. A path like `C:\...` will miss the regex and fall back to the LLM to extract it. Running under WSL with a POSIX path is safer.
