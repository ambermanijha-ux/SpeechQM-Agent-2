# Design gaps

Behaviour that is working as designed but leaves a gap a user should know about. Found by the full all-tasks run
`audios_qc_alltasks/` (2026-10-23, `qc_audit/gap_check.py`). Each gap is left open on purpose until someone decides
how to close it.

## 1. Many English words in the references stay in Roman script (task 18, transliteration)

**What happens.** A Roman-script word in a reference transcript is converted to the native script only when two
independent answers agree (`transliterate_file` in `utility_functions.py`): two LLM runs with differently worded
prompts must give the same spelling in the language's script, and that spelling must be one of IndicXlit's top-5
candidates. Every other word is left as it was and counted in `Roman_Words_Left_After_Transliteration`.
`transliteration_words.csv` records the decision for each word.

**How big it is** (distinct words, `audios_qc_alltasks/`):

| Language | Roman words | Converted | Left | Why left: IndicXlit does not list it / LLM runs disagree / not in script |
|---|---|---|---|---|
| Hindi | 509 | 263 | 246 | 99 / 127 / 20 |
| Tamil | 163 | 41 | 122 | 45 / 77 / 0 |
| Gujarati | 177 | 57 | 120 | 86 / 32 / 2 |
| Telugu | 164 | 113 | 51 | 30 / 19 / 2 |
| Marathi | 94 | 65 | 29 | 6 / 21 / 2 |
| Kannada | 52 | 30 | 22 | 15 / 7 / 0 |
| Bengali | 60 | 54 | 6 | 6 / 0 / 0 |

In word occurrences: Hindi sample 2 837 left (the audio is English, so most of the reference is English), Tamil
222, Gujarati 185, Telugu 99.

**Why it is this way.** A wrong spelling would silently change the reference that WER is measured against; a word
left in Roman script is visible and counted. Precision was chosen over coverage.

**Effect.** The WER of files with many English words is pessimistic: the ASR writes those words in the native
script and the reference keeps them in Roman, so they count as errors.

**Options to close it.**
- Accept IndicXlit's top candidate when the two LLM runs disagree (most of Tamil's and Hindi's leftovers).
- Accept the LLM spelling when both runs agree but IndicXlit does not list it (most of Gujarati's).
- Score WER in a script-independent way (e.g. romanize both sides before comparing) instead of converting the
  reference.

Any of these trades precision for coverage; measure it on the reference files before keeping it.

## 2. The monitor's REVIEW flags vary between runs (all tasks, monitor LLM)

**What happens.** After every step the monitor (`monitor_verdict` in `llm.py`, `MONITOR_MODEL`, qwen3:8b-fp16 at
temperature 0) answers OK, RETRY or REVIEW. Problems that the computed facts prove by themselves (missing files,
repeated rows, empty LLM-written columns) decide without the LLM. Everything else, including whether a result looks
like something a person should check, is the LLM's judgement.

**What varied.** The audio language ID (task 15) gave the same result in both runs: ml001 → Tamil, od001 → Bengali,
the Tamil insurance call → Telugu, Hindi sample 2 → English, all with `Audio_Language_OK = False`. The monitor flagged:

| Run | Task 15 REVIEW for |
|---|---|
| `audios_qc_bodhan_v2/` | Hindi sample 2, ml001, od001 |
| `audios_qc_alltasks/` | Hindi sample 2 only |

The Tamil file was flagged in neither run. The monitor also raised REVIEWs that were false alarms: task 18 on Marathi
("transliteration did nothing", though the output differs) and task 25 on Gujarati and Marathi (overlapping speech in
two-party calls).

**Why it is this way.** The monitor's prompt says to judge whether a step *ran correctly*, not the data ("a file
reported as failing ... is a valid result"). A language mismatch is data, so not flagging it is arguably correct; the
inconsistency is that it sometimes does. Small changes in the shown rows and long prompts make an 8B model's answer
drift even at temperature 0.

**Effect.** `monitor_log.csv` is not a reliable list of the files to look at. The data problems are in the result
columns and the summary (`Audio_Language_OK`, `Script_OK`, `MoE_Verdict`, `ASR_Transcript_Quality`, ...), which are
deterministic.

**Options to close it.**
- Keep the monitor to "did the step run" and build the list of files to review from the result columns with fixed
  rules (e.g. `Audio_Language_OK == False`, `MoE_Verdict != agrees`), in the summary or the dashboard.
- Or add fixed rules to the monitor for the known data flags, as was done for the output-shape problems.

Measure any monitor change with `qc_audit/monitor_bench.py` first (see the memory note on prompt changes).
