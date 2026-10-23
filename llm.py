
import json
import logging
import os
import re
import threading
import time
from typing import Dict, Optional, TypedDict
from concurrent.futures import ThreadPoolExecutor  
import ast
import contextlib
import io
import traceback
try:
    from my_secrets import PROMPT
except ImportError:   # run_qc.py passes the prompt and folder directly
    PROMPT = None
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from utility_functions import (
    transcribe_folder_to_csv,
    process_folder_vad,
    save_num_speakers,
    process_audio_directory,
    transcript_quality,
    force_alignment_and_ctc_score,
    check_upsampling_folder,
    language_identification_indiclid,
    transliterate_file,
    load_audio,
    SR,
    align_model_for,
    get_required_inputs,
    audio_language,
    transcribe_pieces,
    main_asr_for,
    words,
)
import librosa
import pandas as pd
from config import (OLLAMA_BASE_URL, PLANNER_MODEL, CODER_MODEL, JUDGE_MODEL, MONITOR_MODEL, CODER_MAX_TOKENS, LLM_NUM_CTX, LANG_CODES,
                    DEFAULT_LANGUAGE, SCRIPTS, TOOL_LIBRARY, REPLAN, PUBLIC_INDEX, MOE_MAX_CER)
import soundfile as sf
import unicodedata
from collections import Counter
import numpy as np
logging.basicConfig(filename="pipeline.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Local LLMs through Ollama (originally gpt-4.1-mini via the OpenAI API)
def local_llm(model):
    # thinking models (qwen3, deepseek-r1) would put reasoning text, full of numbers, into the answers we parse
    thinking = {"reasoning": False} if any(k in model for k in ("qwen3", "deepseek-r1")) else {}
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=LLM_NUM_CTX, **thinking)


llm = local_llm(PLANNER_MODEL)
judge_llm = llm if JUDGE_MODEL == PLANNER_MODEL else local_llm(JUDGE_MODEL)
monitor_llm = None if MONITOR_MODEL.lower() in ("", "none", "off") else \
    llm if MONITOR_MODEL == PLANNER_MODEL else local_llm(MONITOR_MODEL)
coder_llm = local_llm(CODER_MODEL).model_copy(update={"num_predict": CODER_MAX_TOKENS})


def results_dir(audio_dir: str) -> str:
    d = os.path.join(os.path.dirname(audio_dir.rstrip("/\\")), "results")
    os.makedirs(d, exist_ok=True)
    return d


CODER_SYSTEM = ("You write complete, self-contained Python 3 scripts. Reply with exactly one ```python code block "
                "containing the whole script; it will be executed for you and you will see its output or error. "
                "Available libraries: os, csv, json, re, regex, unicodedata, pandas, numpy, soundfile, librosa.")


def extract_code(text: str) -> str:
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return max(m, key=len)
    try:   # some models answer with a JSON tool call instead of a code block
        obj = json.loads(text.strip().strip("`").removeprefix("json"))
        return obj.get("arguments", obj.get("args", {})).get("code", "")
    except Exception:
        return clean_code(text)


def run_code(code: str) -> str:
    """Execute a generated script in a fresh namespace; returns its stdout plus a short traceback.
    Catches SystemExit so a script calling exit() cannot end the pipeline."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<generated>", "exec"), {"__name__": "__main__"})
    except SystemExit:
        pass
    except Exception:
        buf.write("\nError:\n" + traceback.format_exc(limit=-3))
    return buf.getvalue()


PATH_RE = r"(?:[A-Za-z]:)?/[^\s'\"]+"


def fix_paths(code: str, prompt: str) -> str:
    """Small models miscopy long paths (seen: '...fd5b0e03f1' written for '...fd5b0e0303f1', so the script looked
    for a file that does not exist). Replace any path literal in the script that is nearly, but not exactly, one of
    the paths given in the prompt with that path."""
    import difflib
    known = {m.rstrip(".,;:)") for m in re.findall(PATH_RE, prompt)}

    def repair(m):
        q, lit = m.group(1), m.group(2)
        if lit in known or os.path.exists(lit):
            return m.group(0)
        best = max(known, key=lambda k: difflib.SequenceMatcher(None, lit, k).ratio(), default=None)
        if best and difflib.SequenceMatcher(None, lit, best).ratio() >= 0.9:
            logging.info(f"fixed a miscopied path in generated code: {lit} -> {best}")
            return q + best + q
        return m.group(0)
    return re.sub(r"(['\"])(" + PATH_RE + r")\1", repair, code) if known else code


def describe_inputs(task_prompt: str, output_path: str) -> str:
    """Show the model the real columns of the CSVs it is asked to read, so it does not guess them."""
    notes = []
    for path in dict.fromkeys(re.findall(r"(/[^\s'\"]+\.csv)", task_prompt)):
        if path != output_path and os.path.isfile(path):
            try:
                df = pd.read_csv(path, nrows=1)
                row = {k: str(v)[:80] for k, v in df.iloc[0].to_dict().items()} if len(df) else {}
                notes.append(f"- {path} has columns {list(df.columns)}; first row: {row}")
            except Exception:
                pass
    return ("\n    Input files (exact column names):\n    " + "\n    ".join(notes)) if notes else ""


AUDIO_EXT = (".wav", ".mp3", ".flac")


def list_files(folder: str, ext=None) -> list:
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))
            and not f.startswith(".") and (ext is None or f.lower().endswith(ext))]


def csv_filenames(path: str) -> list:
    """The filename column of an input CSV (case-insensitive 'filename'): the rows a generated output must keep."""
    df = pd.read_csv(path)
    col = next(c for c in df.columns if c.lower() == "filename")
    return df[col].astype(str).tolist()


# ---- checks of generated outputs. Each takes the output DataFrame and returns problems (sentences for the LLM).
# The row count alone let through padded 'dummy_file' rows, files missing from the CSV, blank durations and
# rule-breaking values (mp3 marked valid), so every output is checked against what its task asked for.

def check_files(df, expected: list, col: str = "Filename", columns=()) -> list:
    """Required columns present, one row per expected file (none missing, invented or repeated), no blank values."""
    missing_cols = [c for c in (col, *columns) if c not in df.columns]
    if missing_cols:
        return [f"The CSV must have the columns {[col, *columns]} (exact names); it has {list(df.columns)}."]
    names = df[col].astype(str).map(os.path.basename)
    problems = []
    if absent := sorted(set(expected) - set(names)):
        problems.append(f"These files are missing from the CSV: {absent}.")
    if extra := sorted(set(names) - set(expected)):
        problems.append(f"These rows are not input files and must not be in the CSV: {extra}.")
    if dup := sorted(names[names.duplicated()].unique()):
        problems.append(f"These files appear more than once: {dup}.")
    for c in columns:
        if (blank := names[df[c].isna() | (df[c].astype(str).str.strip() == "")].tolist()):
            problems.append(f"Column '{c}' is empty for: {blank}.")
    return problems


def by_file(df, col="Filename"):
    return {os.path.basename(str(r[col])): r for _, r in df.iterrows()}


def audio_facts(path):
    """(sample rate, duration in s) read independently of the generated script; None if unreadable."""
    try:
        info = sf.info(path)
        return info.samplerate, info.frames / info.samplerate
    except Exception:
        try:
            return librosa.get_samplerate(path), librosa.get_duration(path=path)
        except Exception:
            return None


def wrong_values(rows) -> list:
    """rows: (file, value found, value expected). Lists up to 5 mismatches."""
    bad = [f"{f}: {got!r} (expected {want!r})" for f, got, want in rows if got != want]
    return [f"Wrong values in {len(bad)} rows, e.g. " + "; ".join(bad[:5]) + "."] if bad else []


def check_corruption(audio_dir):
    def expected(f):
        p = os.path.join(audio_dir, f)
        if os.path.getsize(p) == 0 or f.lower().endswith(".mp3"):
            return "corrupt"
        try:
            sf.info(p)
            return "valid"
        except Exception:
            return "corrupt"
    files = list_files(audio_dir)

    def check(df):
        problems = check_files(df, files, columns=["Status"])
        if not problems:
            rows = by_file(df)
            problems = wrong_values((f, str(rows[f]["Status"]).strip().lower(), expected(f)) for f in files)
        return problems
    return check


def check_extension(audio_dir):
    files = list_files(audio_dir)

    def check(df):
        problems = check_files(df, files, columns=["Valid_Extension"])
        if not problems:
            rows = by_file(df)
            problems = wrong_values((f, str(rows[f]["Valid_Extension"]).strip().lower(), str(f.lower().endswith(".wav")).lower())
                                    for f in files)
        return problems
    return check


def check_sample_rate(audio_dir):
    files = list_files(audio_dir, AUDIO_EXT)

    def check(df):
        problems = check_files(df, files, columns=["Sample_Rate", "Channels"])
        if problems:
            return problems
        rows, out = by_file(df), []
        for f in files:
            facts = audio_facts(os.path.join(audio_dir, f))
            if facts:
                out.append((f, str(rows[f]["Sample_Rate"]).split(".")[0], str(facts[0])))
        return wrong_values(out)
    return check


def check_audio_length(audio_dir):
    files = list_files(audio_dir, AUDIO_EXT)

    def check(df):
        problems = check_files(df, files, columns=["Audio_length"])
        if problems:
            return problems
        rows, bad = by_file(df), []
        for f in files:
            facts = audio_facts(os.path.join(audio_dir, f))
            try:
                got = float(rows[f]["Audio_length"])
            except (TypeError, ValueError):
                got = None
            if facts and (got is None or abs(got - facts[1]) > 0.05):
                bad.append(f"{f}: {rows[f]['Audio_length']!r} (expected {facts[1]:.2f} s)")
        return [f"Wrong durations in {len(bad)} rows, e.g. " + "; ".join(bad[:5]) + "."] if bad else []
    return check


def text_column(df):
    return next(c for c in df.columns if c.lower() in ("transcription", "ground_truth", "indiconformer_hypothesis"))


def check_characters(csv_path):
    src = pd.read_csv(csv_path)
    files, col = csv_filenames(csv_path), text_column(src)
    texts = dict(zip(files, src[col].fillna("").astype(str)))

    def check(df):
        problems = check_files(df, files, columns=["character_list"])
        if problems:
            return problems
        bad = []
        for f, r in by_file(df).items():
            try:
                chars = json.loads(r["character_list"]) if str(r["character_list"]).startswith("[\"") else ast.literal_eval(str(r["character_list"]))
                ok = isinstance(chars, list) and {c for c in chars if not str(c).isspace()} == {c for c in texts[f] if not c.isspace()}
            except Exception:
                ok = False
            if not ok:
                bad.append(f)
        return [f"'character_list' must be a list (Python or JSON) of every distinct character of the transcription; "
                f"wrong for: {bad[:5]}."] if bad else []
    return check


def check_normalization(csv_path):
    files = csv_filenames(csv_path)
    src = pd.read_csv(csv_path)
    gt = next(c for c in src.columns if c.lower() == "ground_truth")
    # compared in NFKC form: a script that also normalizes Unicode (precomposed ड़ -> ड + nukta) changes no letter
    letters = lambda t: "".join(ch for ch in unicodedata.normalize("NFKC", t) if unicodedata.category(ch)[0] in "LMN")
    want = {f: letters(re.sub(r"\[[^\]]*\]|<[^>]*>", " ", str(t))) for f, t in zip(files, src[gt].fillna(""))}

    def check(df):
        col = next((c for c in df.columns if c.lower() == "filename"), "filename")
        problems = check_files(df, files, col=col)   # an empty transcript is judged by the letter check below
        if not problems and "normalized_transcripts" not in df.columns:
            problems = [f"The CSV must have a 'normalized_transcripts' column; it has {list(df.columns)}."]
        if problems:
            return problems
        for f, r in by_file(df, col).items():
            t = "" if pd.isna(r["normalized_transcripts"]) else str(r["normalized_transcripts"])
            if re.search(r"<[^>]*>|\[[^\]]*\]", t):
                problems.append(f"{f}: tags or [bracketed text] are still present.")
            elif "  " in t:
                problems.append(f"{f}: repeated spaces are still present.")
            elif letters(t) != want[f]:
                lost = "".join(sorted(set(want[f]) - set(letters(t)))[:15])
                problems.append(f"{f}: letters were removed or changed" + (f" (e.g. these are gone: {lost!r})" if lost else "")
                                + ". Only tags, [bracketed text] and symbols may be removed; keep every letter of every "
                                f"script (English words in code-mixed text stay), digits and combining marks (Indic "
                                f"vowel signs are Unicode category M, which Python's \\w does not match). Keep a character when "
                                f"unicodedata.category(ch)[0] is 'L', 'M' or 'N', or ch.isspace(); drop the rest.")
        return problems[:5]
    return check


def check_alignment(csv_path, audio_dir, model_name):
    files = csv_filenames(csv_path)
    src = pd.read_csv(csv_path)
    texts = dict(zip(files, src[text_column(src)].fillna("").astype(str)))

    def check(df):
        problems = check_files(df, files, columns=["Aligned_Segments", "CTC_Score"])
        if problems:
            return problems
        rows = by_file(df)
        segs_text = pd.Series({f: str(r["Aligned_Segments"]) for f, r in rows.items()})
        if len(files) > 1 and (same := sorted(segs_text[segs_text.duplicated(keep=False) & (segs_text != "[]")].index)):
            return [f"Files {same[:5]} have identical Aligned_Segments; each must come from aligning that file's own "
                    f"audio and transcript with force_alignment_and_ctc_score."]
        # recompute the shortest file to make sure the scores come from the real alignment
        f = min(files, key=lambda f: (audio_facts(os.path.join(audio_dir, f)) or (0, 1e9))[1])
        _, score = force_alignment_and_ctc_score(os.path.join(audio_dir, f), texts[f], model_name)
        try:
            ok = abs(float(rows[f]["CTC_Score"]) - score) <= 0.01
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return [f"{f}: CTC_Score is {rows[f]['CTC_Score']} but force_alignment_and_ctc_score returns {score:.3f} "
                    f"for it; use the function's results, do not invent or round them."]
        empty, bad = [], []
        for f, r in rows.items():
            try:
                segs = json.loads(r["Aligned_Segments"])
                mean = sum(float(s[3]) for s in segs) / len(segs) if segs else 0.0
                if not segs:
                    empty.append(f)
                if abs(float(r["CTC_Score"]) - mean) > 0.01:
                    bad.append(f"{f}: CTC_Score {r['CTC_Score']} but the mean of its segment scores is {mean:.3f}")
            except Exception:
                bad.append(f"{f}: Aligned_Segments is not a JSON list of [token, start, end, score]")
        if bad:
            problems.append("CTC_Score must be the avg_score returned by force_alignment_and_ctc_score for that file: "
                            + "; ".join(bad[:5]) + ".")
        if empty and len(empty) == len(files):
            problems.append("Aligned_Segments is empty for every file: call force_alignment_and_ctc_score(audio_path, "
                            "transcript, model) for each row with the full audio path and that row's transcript.")
        return problems
    return check


# ---------------------------------------------------------------- tool library
# Every script the code writer produced that passed its task's checks is kept (per output file, paths replaced by
# placeholders). The LLM still writes a new script on every run; a kept one runs only when all its attempts fail.
def prompt_paths(task_prompt: str) -> list:
    """The absolute paths a task prompt names, in order (the same paths fix_paths knows)."""
    return list(dict.fromkeys(m.rstrip(".,;:)") for m in re.findall(PATH_RE, task_prompt)))


def library_file(output_path: str) -> str:
    return os.path.join(TOOL_LIBRARY, os.path.basename(output_path) + ".json")


def save_tool(task_prompt: str, output_path: str, code: str):
    paths = prompt_paths(task_prompt)
    for i, p in sorted(enumerate(paths), key=lambda ip: -len(ip[1])):   # longest first: no partial replacements
        code = code.replace(p, f"{{{{PATH{i}}}}}")
    os.makedirs(TOOL_LIBRARY, exist_ok=True)
    with open(library_file(output_path), "w", encoding="utf-8") as f:
        json.dump({"paths": len(paths), "code": code, "model": CODER_MODEL, "saved": time.strftime("%Y-%m-%d %H:%M:%S")},
                  f, ensure_ascii=False, indent=1)


def run_saved_tool(task_prompt: str, output_path: str, check) -> bool:
    try:
        with open(library_file(output_path), encoding="utf-8") as f:
            tool = json.load(f)
    except (OSError, ValueError):
        return False
    paths = prompt_paths(task_prompt)
    if tool["paths"] != len(paths):
        return False
    code = tool["code"]
    for i, p in enumerate(paths):
        code = code.replace(f"{{{{PATH{i}}}}}", p)
    result = run_code(code)
    try:
        df = pd.read_csv(output_path)
        ok = not df.empty and not (check(df) if check else [])
    except Exception:
        ok = False
    logging.info(f"tool library: saved script for {os.path.basename(output_path)} (written by {tool['model']} on "
                 f"{tool['saved']}) {'passed' if ok else 'failed'}:\n{result[:1000]}")
    log = os.path.join(os.path.dirname(output_path), "tool_library_log.csv")
    pd.DataFrame([{"Output": os.path.basename(output_path), "Saved_By": tool["model"], "Saved": tool["saved"],
                   "Passed": ok}]).to_csv(log, mode="a", header=not os.path.exists(log), index=False)
    if not ok and os.path.exists(output_path):
        os.remove(output_path)
    return ok


def run_agent(task_prompt: str, output_path: str, check=None, attempts: int = 3) -> bool:
    """Code-generating agent: the LLM writes a script for the task, we execute it, and on an error, a
    missing/empty output CSV or an output that fails `check` the problem goes back to the LLM for a corrected
    script. When every attempt fails the output is deleted, so no partial CSV reaches the summary."""
    if os.path.exists(output_path):
        os.remove(output_path)
    prompt = task_prompt + describe_inputs(task_prompt, output_path)
    note = MONITOR_FEEDBACK.pop(output_path, None)
    if note:   # this is a re-run requested by the monitoring LLM
        prompt += f"\n    A reviewer rejected the previous output of this step: {note}"
    if os.sep == "\\":   # Windows: C:\Users\... pasted into a Python string literal is a syntax error ("\U")
        prompt = prompt.replace("\\", "/")
    messages = [("system", CODER_SYSTEM), ("user", prompt)]
    with AGENT_LOCK:
        for attempt in range(1, attempts + 1):
            if os.path.exists(output_path):
                os.remove(output_path)   # a failed try must not leave its CSV for the next one to pass with
            response = coder_llm.invoke(messages)
            reply = response.content
            if response.response_metadata.get("done_reason") == "length":
                reply = reply[:1500] + "\n... [cut off]"
                result, problems = "", [f"Your reply was cut off after {CODER_MAX_TOKENS} tokens. Write a much shorter "
                                        f"script: use regular-expression character classes or unicodedata categories "
                                        f"instead of listing characters one by one."]
            else:
                code = fix_paths(extract_code(reply), prompt)
                result = run_code(code) if code.strip() else "No code found in the reply."
                logging.info(f"agent attempt {attempt} for {os.path.basename(output_path)}:\n{code}\n--- output:\n{result[:2000]}")
                try:
                    df = pd.read_csv(output_path)
                    problems = check(df) if check else []
                except Exception:   # missing, empty or unreadable
                    df, problems = None, [f"The output file {output_path} was not created or cannot be read as a CSV."]
                if df is not None and df.empty:
                    problems = [f"The output file {output_path} has no rows."]
                if not problems:
                    save_tool(task_prompt, output_path, code)
                    return True
                if "\nError:\n" in result:
                    problems.insert(0, "The script failed.")
                # qwen3 writes \p{M} for vowel signs but calls re instead of regex: an error ("bad escape"), or with
                # an escaped backslash ('\\p{L}') a silent character class that deletes most of the text
                if "bad escape \\p" in result or re.search(r"\bre\.\w+\(\s*r?['\"][^'\"]*p\{", code):
                    problems.insert(1, "Python's re module does not support \\p{...}: `import regex` and call "
                                       "regex.sub / regex.findall for those patterns.")
                if re.search(r"""r['"][^'"]*\\\\[psdwPSDW]""", code):   # r'[^\\p{L}\\s]': a backslash and the letters p, {, L, ...
                    problems.insert(1, "In a raw string write ONE backslash: r'[^\\p{L}\\p{M}\\p{N}\\s]'. With two "
                                       "(r'\\\\p{L}') the class means a backslash and the letters p, {, L, }, so it "
                                       "deletes almost every letter.")
            logging.info(f"agent attempt {attempt} for {os.path.basename(output_path)} rejected: {problems}")
            messages += [("assistant", reply),
                         ("user", f"Running your script printed:\n{result[:3000]}\nProblems: {' '.join(problems)} "
                                  f"Fix it and reply with the full corrected script.")]
        if run_saved_tool(task_prompt, output_path, check):   # every new script failed: an earlier one that passed
            return True
    if os.path.exists(output_path):
        os.remove(output_path)
    return False


def parse_prompt(prompt: str) -> Dict[str, Optional[str]]:
    result = {
        "audio_dir": None,
        "ground_truth_csv": None,
        "lang_code": None
    }
    path_pattern = r"['\"]?(/[^'\"]+(?:/[^'\"]+)*/?)['\"]?"
    csv_pattern = r"['\"]?(/[^'\"]+(?:/[^'\"]+)*\.csv)['\"]?"
    lang_pattern = r"\b(bn|gu|hi|kn|ml|mr|pa|sd|si|ta|te|ur)\b"
    audio_dir_match = re.search(path_pattern, prompt)
    if audio_dir_match:
        result["audio_dir"] = audio_dir_match.group(1)
    csv_match = re.search(csv_pattern, prompt)
    if csv_match:
        result["ground_truth_csv"] = csv_match.group(1)
    names = {v.lower(): k for k, v in LANG_CODES.items() if v != "English"}
    name_match = re.search(r"\b(" + "|".join(names) + r")\b", prompt, re.IGNORECASE)
    lang_match = re.search(lang_pattern, prompt, re.IGNORECASE)
    if name_match:
        result["lang_code"] = names[name_match.group(1).lower()]
    elif re.search(r"\benglish\b(?!\s+words?)", prompt, re.IGNORECASE):
        result["lang_code"] = "en"
    elif lang_match:
        result["lang_code"] = lang_match.group(1).lower()
    if not result["audio_dir"] and not result["ground_truth_csv"]:
        llm_prompt = f"""Extract the following from the prompt:
        1. Audio directory path (e.g., /path/to/audio/)
        2. Ground truth CSV path (e.g., /path/to/file.csv)
        3. Language code (e.g., hi, te)
        If any are unclear, return None for that field.
        Prompt: {prompt}
        Return a JSON object with keys 'audio_dir', 'ground_truth_csv', 'lang_code'.
        """
        try:
            response = llm.invoke(llm_prompt)
            content = response.content.strip()
            if content.startswith('```json'):                                                                       #markdown format correction 
                content = content.replace('```json', '').replace('```', '').strip()
            llm_result = json.loads(content)
            for key in result:
                if not result[key]:
                    result[key] = llm_result.get(key)
        except Exception as e:
            logging.error(f"LLM prompt parsing failed: {e}")                                                         # LLM failsafe
            if audio_dir_match:
                result["audio_dir"] = audio_dir_match.group(1)
            elif "Audio_dir" in prompt:
                audio_dir_match = re.search(r"Audio_dir\s*=\s*['\"]?(/[^'\"]+/)['\"]?", prompt)
                if audio_dir_match:
                    result["audio_dir"] = audio_dir_match.group(1)
    
    if result["audio_dir"] and not os.path.isdir(result["audio_dir"]):
        logging.error(f"Invalid audio directory: {result['audio_dir']}")
        result["audio_dir"] = None
    if result["ground_truth_csv"] and not os.path.isfile(result["ground_truth_csv"]):
        logging.error(f"Invalid ground truth CSV: {result['ground_truth_csv']}")
        result["ground_truth_csv"] = None
    return result

def language_of(state) -> str:
    """Language named in the prompt (via its code), else DEFAULT_LANGUAGE."""
    return LANG_CODES.get((state.get('lang_code') or "").lower(), DEFAULT_LANGUAGE)

class CombinedStateDict(TypedDict, total=False):
    audio_dir: str
    ground_truth_csv: str
    lang_code: str
    user_prompt: str
    A: str
    D: str
    E: str
    character_output: str
    vocab_output: str
    audio_length_output: str
    ctc_score_output: str
    language_verification_output: str
    segmentation_output: str
    code_mixing_output: str
    upsampling_output: str
    valid_speaker_output: str
    domain_checker_output: str
    audio_transcript_matching_output: str
    language_identification_indiclid_output: str
    normalization_remove_tags_output: str
    llm_score_output: str
    transliteration_output: str
    corruption_output: str
    extension_output: str
    sample_rate_output: str
    english_word_count_output: str
    speaker_duration_output: str
    utterance_duplicate_output: str
    wer_output: str
    transcript_quality_output: str

AGENT_LOCK = threading.Lock()   # the REPL swaps sys.stdout and Ollama serves one request at a time


def clean_code(command: str) -> str:
    """Small local models often wrap code in fences or prefix every line with 'Action Input:'."""
    command = re.sub(r"^\s*```(?:python|py)?\s*$", "", command, flags=re.MULTILINE)
    return re.sub(r"^\s*Action Input:\s?", "", command, flags=re.MULTILINE).strip()


def select_tasks(user_prompt: str) -> str:
    prompt_1 = f"""You are given the following functions:
    1. ASR Transcription
    2. Number of Speakers calculation and duration per speaker
    3. Quality of Transcript
    4. Graphene or character calculation
    5. Vocab calculation
    6. Language verification (verify if transcriptions match an expected language)
    7. Audio length calculation
    8. Silence calculation (using VAD)
    9. English Word Counter
    10. CTC score calculation
    11. Upsampling Check
    12. Check if speakers are new or old
    13. Check the domain of the speech dataset
    14. Map transcriptions to audio files using forced alignment
    15. Language identification using ASR transcriptions and IndicLID
    16. Normalization by removing HTML and other tags from transcriptions in JSON or XML files
    17. Evaluate transcript coherence and fluency using LLM-as-a-Judge and score out of 10
    18. Transliteration - Convert Roman script words to Native script using Transliteration
    19. Audio corruption check
    20. Audio extension and format check
    21. Audio sample rate check
    22. Speaker Durations
    23. utterance duplicate checker
    24. WER computation 
    25. Timestamp-based segmentation of the audio into utterances using the reference time-stamps
    26. Code-mixing detection in the reference transcripts
    Based on the prompt, identify the task numbers that must be executed to fulfill the request.
    - Inlcude task 1 whenever an audio_dir is provided.
    - If the prompt mentions 'Vocab calculation', include task 5.
    - If the prompt mentions 'Character calculation', include task 4.
    - If the prompt mentions 'new or old' or 'valid speaker', include tasks 2 and 12.
    - if the prompt mentions 'duplicate' or 'duplicates', include task 23.
    - Whenever prompt mentions "Durations" or "durations" , include both task 2 and task 22.
    - If the prompt mentions 'verify' and 'language' or 'expected language', include task 6.
    - If the prompt mentions 'language identification' or 'IndicLID', include tasks 1 and 15 (task 15 requires task 1 to generate transcriptions).
    - Include QC tasks (19, 20, 21) only if explicitly mentioned (e.g., 'corruption', 'extension', 'sample rate')
    - If the prompt mentions 'silence' or 'VAD', include task 8.
    - If the prompt mentions 'upsampling' or 'upsampled', include task 11.
    - If the prompt mentions 'domain', include task 13.
    - If the prompt mentions 'English words', include task 9. If it mentions 'WER', include task 24.
    - If the prompt mentions 'segmentation' or 'time-stamps', include task 25. If it mentions 'code-mixing', include task 26.
    - Only include tasks explicitly relevant to the prompt.
    Return ONLY the task numbers as a comma-separated string (e.g., '4,5,6'), with no other text.
    Prompt: {user_prompt}
    """
    resp_1 = llm.invoke(prompt_1).content
    task_numbers = list(dict.fromkeys(n for n in re.findall(r'\b\d+\b', resp_1) if 1 <= int(n) <= 26))
    expected_tasks = set()
    resp_1 = ','.join(task_numbers) if task_numbers else ''
    logging.info(f"Selected Tasks: {resp_1}")
    return resp_1

def prompt_checker_agent(user_prompt: str, selected_tasks: str) -> str:
    max_iterations = 3
    current_tasks = set(selected_tasks.split(',') if selected_tasks else [])
    iteration = 0
    task_definitions = """
    1. ASR Transcription
    2. Number of Speakers calculation and duration per speaker
    3. Quality of Transcript
    4. Graphene or character calculation
    5. Vocab calculation
    6. Language verification (verify if transcriptions match an expected language)
    7. Audio length calculation
    8. Silence calculation (using VAD)
    9. English Word Counter
    10. CTC score calculation
    11. Upsampling Check
    12. Check if speakers are new or old
    13. Check the domain of the speech dataset
    14. Map transcriptions to audio files using forced alignment
    15. Language identification using ASR transcriptions and IndicLID
    16. Normalization by removing HTML and other tags from transcriptions in JSON or XML files
    17. Evaluate transcript coherence and fluency using LLM-as-a-Judge and score out of 10
    18. Transliteration - Convert Roman script words to Native script using Transliteration
    19. Audio corruption check
    20. Audio extension and format check
    21. Audio sample rate check
    22. Speaker Durations
    23. utterance duplicate checker
    24. WER computation 
    25. Timestamp-based segmentation of the audio into utterances using the reference time-stamps
    26. Code-mixing detection in the reference transcripts
    """
    rules = """
    - If the prompt mentions 'Vocab calculation', include task 5.
    - If the prompt mentions 'Character calculation', include task 4.
    - If the prompt mentions 'verify' and 'language' or 'expected language', include task 6.
    - If the prompt mentions 'language identification' or 'IndicLID', include tasks 1 and 15 (task 15 requires task 1 to generate transcriptions).
    - Include QC tasks (19, 20, 21) only if explicitly mentioned (e.g., 'corruption', 'extension', 'sample rate').
    - If the prompt mentions 'silence' or 'VAD', include task 8.
    - If the prompt mentions 'upsampling' or 'upsampled', include task 11.
    - If the prompt mentions 'domain', include task 13.
    - If the prompt mentions 'new or old' or 'valid speaker', include tasks 2 and 12.
    - If the prompt mentions 'English words', include task 9. If it mentions 'WER', include task 24. If it mentions 'duplicate', include task 23.
    - If the prompt mentions 'number of speakers' or 'speakers' or 'duration' or 'durations',  include both task 2 and task 22.
    - If the prompt mentions 'ASR' or 'transcription', include task 1.
    - if Task 9 or task 23 or task 24 are included , then definitely include task 16. 
    - If the prompt mentions 'segmentation' or 'time-stamps', include task 25. If it mentions 'code-mixing', include task 26.
    - Only include tasks explicitly relevant to the prompt.
    """
    while iteration < max_iterations:
        iteration += 1
        logging.info(f"Prompt Checker Iteration {iteration}: Current tasks: {','.join(current_tasks)}")
        checker_prompt = f"""
        You are a prompt analysis expert. Your task is to verify if the tasks selected for a given user prompt are correct and complete.
        **User Prompt**: {user_prompt}
        **Task Definitions**:
        {task_definitions}
        **Selection Rules**:
        {rules}
        **Currently Selected Tasks**: {','.join(current_tasks) if current_tasks else 'None'}
        **Your Task**:
        1. Analyze the user prompt in detail using natural language and in depth context understanding to identify all tasks that should be executed based on the task definitions and selection rules.
        2. Compare the expected tasks ( based on your analysis ) with the currently selected tasks
        3. Determine if the selected tasks are correct, if any are missing, or if any are extra:
        - 'Correct': All selected tasks match the expected tasks.
        - 'Missing': Some expected tasks are not in the selected tasks.
        - 'Extra': Some selected tasks are not relevant to the prompt.
        4. Strictly adhere to the the ouput format specified below.Do not deviate from this format even if your explaination is detailed
        **Output Format**:
        - Status: Correct or Missing or Extra (any one of these three)
        - Tasks: <comma-separated task numbers which are either missing or extra>
        - Explanation: <brief explanation if Missing or Extra, otherwise empty>
        **Important Notes**:
        - If there are both missing and extra tasks, prioritize 'Extra' as the Status and list only the extra tasks under 'Tasks'. Mention missing tasks (if any) in the explanation.
        - Ensure the 'Status' and 'Tasks' fields align with your final conclusion, not intermediate reasoning.
        - Do not include tasks in 'Tasks' field unless they are missing or extra.
        **Example Output**:
        Example1:
        Status: Missing
        Tasks: 1,2
        Explanation: The prompt mentions 'ASR transcription' (task 1) and 'number of speakers' (task 2), but these were not selected.

        Example2:
        Status: Extra
        Tasks: 3,4
        Explanation: The prompt does not mention 'Quality of Transcript' (task 3) or 'Graphene calculation' (task 4), but these were included in the selected tasks.
        """
        try:
            response = llm.invoke(checker_prompt).content.strip()
            logging.info(f"Prompt Checker Response: {response}")
            status_match = re.search(r'Status:\s*(Correct|Missing|Extra)', response)
            tasks_match = re.search(r'Tasks:\s*([\d,]+)', response)
            explanation_match = re.search(r'Explanation:\s*(.*)', response, re.DOTALL)
            if not status_match or not tasks_match:
                logging.error(f"Invalid prompt checker response format: {response}")
                break
            status = status_match.group(1)
            tasks = tasks_match.group(1).replace(' ', '')
            explanation = explanation_match.group(1).strip() if explanation_match else ''
            if status == 'Correct':
                logging.info(f"Prompt Checker: All tasks correct: {tasks}")
                return tasks
            if status == 'Missing': ####################3 WITH FEEDBACK
                logging.info(f"Prompt Checker with feedback: Missing/Extra tasks detected: {tasks}. Explanation: {explanation}")
                current_tasks.update(tasks.split(','))
                new_selected_tasks = select_tasks(user_prompt + f"\n\n here is the feedback for missing/extra tasks detected from your previous response: {explanation}\n\n")
                if new_selected_tasks:
                    current_tasks.update(new_selected_tasks.split(','))
                current_tasks = {t for t in current_tasks if t and t.isdigit() and 1 <= int(t) <= 26}
                logging.info(f"Updated tasks after iteration {iteration}: {','.join(sorted(current_tasks))}")

            if status == 'Extra': ###########################3 Additional code for extra tasks
                logging.info(f"Prompt Checker with feedback: Extra tasks detected: {tasks}. Explanation: {explanation}")
                current_tasks.difference_update(tasks.split(','))
                if not current_tasks:
                    logging.info("No tasks left after removing extra tasks.")
                    return ''    
        except Exception as e:
            logging.error(f"Prompt Checker failed: {e}")
            break
    final_tasks = ','.join(sorted(current_tasks)) if current_tasks else selected_tasks
    logging.info(f"Prompt Checker: Max iterations reached. Final tasks: {final_tasks}")
    return final_tasks

def topological_sort_tasks(resp_1: str) -> list[list[int]]:
    prompt_2 = f"""You are given a list of 26 tasks related to ASR, audio, and transcription processing:
        Tasks:
        1. ASR Transcription using audio files
        2. Number of Speakers calculation and duration per speaker using audio files
        3. Quality of Transcript using audio files and ground truth file
        4. Graphene or character calculation using ground truth file
        5. Vocab calculation using ground truth file
        6. Language verification (verify if transcriptions match an expected language) using ground truth file
        7. Audio length calculation using audio files
        8. Silence calculation (using VAD) audio files
        9. English Word Counter
        10. CTC score calculation using audio files and ground truth file
        11. Upsampling Check using audio files
        12. Check if speakers are new or old using the results from number of speakers calculation
        13. Check the domain of the speech dataset using transcriptions from ASR
        14. Map transcriptions to audio files using forced alignment, using ground truth transcriptions
        15. Language identification using ASR transcriptions and IndicLID, using transcriptions from ASR
        16. Normalization by removing HTML and other tags from transcriptions in JSON or XML files
        17. Evaluate transcript coherence and fluency using LLM-as-a-Judge and score out of 10
        18. Transliteration - Convert Roman script words to Native script using Transliteration
        19. Audio corruption check using audio files
        20. Audio extension and format check using audio files
        21. Audio sample rate check using audio files
        22. Calculate durations of each speaker
        23. Utterance duplicate checker
        24. WER computation
        25. Timestamp-based segmentation of the audio using the reference time-stamps
        26. Code-mixing detection in the reference transcripts
        ## INPUT
        - You will receive a subset of the above task numbers (e.g., {resp_1}). Only consider these tasks for execution.
        ## TASK
        You must perform dependency-aware topological sorting of the requested tasks so that all dependencies are respected and tasks can be grouped for maximum concurrency where possible.
        ### General Instructions
        - Work only with the task numbers included in {resp_1}. Exclude any other tasks.
        - Group independent tasks together so they can execute concurrently (in parallel).
        - Arrange dependent tasks such that dependencies always run first and their dependents run strictly after.
        - Very strict dependency: Tasks 9, 23, and 24 depend on task 16. Task 16 must be completed before any of these are started.
        - Strict dependency: Tasks (3,4,5,6,10,13,14,15,16) all depend on task 1. Task 12 depends on task 2. Task 22 depends on task 2. Task 14 also depends on ground truth being available. Tasks 25 and 26 have no dependencies.
        - Place all tasks that depend on a particular task in subsequent groups, never in the same group as their dependency.
        - Use the provided audio directory for audio-related tasks.
        ### OUTPUT FORMAT
        - You MUST return only **one Python list of lists**.
        - Do NOT repeat the same group multiple times.
        - Do NOT add extra explanations or variations.
        - Return the topologically sorted tasks as a list of lists, where each sub-list contains tasks that can be run in parallel (e.g. [[1], [3,6], [12]]).
        - If the task subset has no dependencies or a valid structure can't be formed, group all tasks into a single list of list (e.g. [[6]]).
        ### EXAMPLES
        - If you receive [1,3,4], output should reflect that 3 and 4 depend on 1, e.g., [[1], [3,4]].
        - If you receive [2,12,22], output should reflect that 12 and 22 depend on 2, e.g., [[2],[12,22]].
        """


    import ast, re, logging

    try:
        resp_2 = llm.invoke(prompt_2).content
        resp_2 = resp_2.strip()

        # Remove code fences if any
        resp_2 = resp_2.replace('```python', '').replace('```', '').strip()

        # Remove any trailing commas in lists
        resp_2 = re.sub(r',(\s*[\]\}])', r'\1', resp_2)

        # Attempt to parse using ast.literal_eval
        try:
            structure = ast.literal_eval(resp_2)
        except:
            # fallback: extract all groups of numbers
            groups = re.findall(r'\[([0-9,\s]*)\]', resp_2)
            structure = [[int(t.strip()) for t in group.split(',') if t.strip()] for group in groups]

        # Validate structure
        if not isinstance(structure, list) or not all(isinstance(group, list) for group in structure):
            raise ValueError("Invalid structure: must be a list of lists")

        # Keep only requested tasks
        tasks_set = set(int(t) for t in resp_1.split(',') if t.strip().isdigit())
        structure = [[task for task in group if task in tasks_set] for group in structure]
        structure = [group for group in structure if group]

        # If empty after filtering, return all tasks as a single group
        if not structure:
            structure = [list(tasks_set)] if tasks_set else []

        logging.info(f"Topological Structure: {structure}")
        return structure

    except Exception as e:
        logging.error(f"Failed to parse topological structure: {e}")
        tasks = [int(t) for t in resp_1.split(',') if t.strip().isdigit()]
        return [tasks] if tasks else []

def corruption_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for corruption check: {audio_dir}")
        return {"corruption_output": "Invalid: No audio directory"}
    output_path = os.path.join(results_dir(audio_dir), "corruption_check.csv")
    task_prompt = f"""You are given a folder with audios at this path: {audio_dir}.
    Write a Python script to do the following:
    - Import the 'os' and 'soundfile' modules.
    - Create a CSV at exactly this path: {output_path}
    - The CSV should have columns: Filename, Status.
    - For each file in the folder:
        - Use os.path.getsize(file_path) to get the file size.
        - If the file size is 0 bytes, mark its 'Status' column as 'corrupt'.
        - If the file is in .mp3 format, mark its 'Status' column as 'corrupt'.
        - If soundfile.info(file_path) raises an exception, mark its 'Status' column as 'corrupt'.
        - Otherwise, mark its 'Status' column as 'valid'.
    - Save the CSV and print how many files are corrupt.
    """
    try:
        if not run_agent(task_prompt, output_path, check_corruption(audio_dir)):
            return {"corruption_output": f"Error: Failed to generate {output_path}"}
        df = pd.read_csv(output_path)
        ok = (df["Status"].astype(str).str.lower() == "valid").all()
        return {"corruption_output": ("Success" if ok else "Invalid") + f" - CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Corruption check failed: {e}")
        return {"corruption_output": f"Error: {e}"}

def extension_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for extension check: {audio_dir}")
        return {"extension_output": "Invalid: No audio directory"}
    output_path = os.path.join(results_dir(audio_dir), "audio_format_check.csv")
    task_prompt = f"""You are given a folder with audios at this path: {audio_dir}.
    Write a Python script to do the following:
    - Import the 'os' module and use it to iterate through files in the directory.

    Steps:
    1. Check that each file has a valid audio extension (only .wav ).
    2. Create a CSV at exactly this path: {output_path}
    3. The CSV should have columns: Filename, Valid_Extension.
    4. For each file:
    - 'Valid_Extension' should be True if the filename ends with .wav (case insensitive), else False.
    5. Save the CSV.
    """
    try:
        if not run_agent(task_prompt, output_path, check_extension(audio_dir)):
            return {"extension_output": f"Error: Failed to generate {output_path}"}
        df = pd.read_csv(output_path)
        ok = df["Valid_Extension"].astype(str).str.lower().eq("true").all()
        return {"extension_output": ("Success" if ok else "Invalid") + f" - CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Extension check failed: {e}")
        return {"extension_output": f"Error: {e}"}

def sample_rate_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for sample rate check: {audio_dir}")
        return {"sample_rate_output": "Invalid: No audio directory"}
    output_path = os.path.join(results_dir(audio_dir), "sample_rate_check.csv")
    task_prompt = f"""You are given a folder with audio files at this path: {audio_dir}.
    Write a Python script to do the following:
    1. For each audio file (.wav, .mp3, .flac), read its sample rate and number of channels with
       soundfile.info(path).samplerate and .channels (use librosa.get_samplerate(path) if soundfile cannot read it).
    2. Create a CSV with columns: Filename, Sample_Rate, Channels
    3. Save the CSV at exactly this path: {output_path}
    """
    try:
        if not run_agent(task_prompt, output_path, check_sample_rate(audio_dir)):
            return {"sample_rate_output": f"Error: Failed to generate {output_path}"}
        rates = pd.read_csv(output_path)["Sample_Rate"].astype(int).value_counts().sort_index()
        found = ", ".join(f"{r} Hz x{n}" for r, n in rates.items())
        return {"sample_rate_output": f"Sample rates: {found} - CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Sample rate check failed: {e}")
        return {"sample_rate_output": f"Error: {e}"}

def transcription_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for transcription: {audio_dir}")
        return {"A": "Error: Invalid audio directory"}
    logging.info(f"Running Transcription ({language_of(state)})")
    try:
        result = transcribe_folder_to_csv(audio_dir, source_language=language_of(state))
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        result = f"Error: {e}"
    return {"A": result} #, "audio_dir": audio_dir

def silence_vad_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for silence detection: {audio_dir}")
        return {"D": "Error: Invalid audio directory"}
    logging.info("Running Silence Detection")
    result = process_folder_vad(audio_dir)
    return {"D": result}

def num_speaker_func(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for speaker diarization: {audio_dir}")
        return {"E": "Error: Invalid audio directory"}
    logging.info("Running Speaker Diarization and Duration Calculation")
    result = save_num_speakers(audio_dir)
    return {"E": result}

def vocab_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    csv_path = state.get('A', os.path.join(results_dir(audio_dir), "indicconf_hypothesis.csv"))

    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for vocab calculation: {csv_path}")
        return {"vocab_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running vocab_agent")
    output_path = os.path.join(os.path.dirname(csv_path), "vocab_list.csv")
    # deterministic: model-written scripts split Indic words at vowel signs (not \w / isalpha in Python)
    try:
        df = pd.read_csv(csv_path)
        col = next(c for c in df.columns if c.lower() in ("transcription", "ground_truth", "indiconformer_hypothesis"))
        df['vocab_list'] = df[col].fillna("").astype(str).apply(lambda t: json.dumps(list(dict.fromkeys(words(t))), ensure_ascii=False))
        df.to_csv(output_path, index=False)
        return {"vocab_output": f"CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Vocab extraction failed: {e}")
        return {"vocab_output": f"Error: {e}"}

def character_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    csv_path = state.get('A', os.path.join(results_dir(audio_dir), "indicconf_hypothesis.csv"))
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for character calculation: {csv_path}")
        return {"character_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running character_agent")
    output_path = os.path.join(os.path.dirname(csv_path), "character_list.csv")
    task_prompt = f"""You are given a CSV file at this path: {csv_path}.
    It has a transcription column called 'Transcription', 'Ground_Truth' or 'Indiconformer_Hypothesis' (search case insensitively).
    Write a Python script to do the following:
    1. For each row, extract a list of unique characters from the transcription column and store it in a new column called 'character_list'.
    2. Save the updated CSV with the new column at exactly this path: {output_path}
    """
    try:
        if run_agent(task_prompt, output_path, check_characters(csv_path)):
            return {"character_output": f"CSV saved at: {output_path}"}
        error_msg = f"Failed to generate {output_path}"
        logging.error(error_msg)
        return {"character_output": error_msg}
    except Exception as e:
        logging.error(f"Character calculation failed: {e}")
        return {"character_output": f"Error: {e}"}

def audio_length_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for audio length calculation: {audio_dir}")
        return {"audio_length_output": "Error: Invalid audio directory"}
    logging.info("Running audio_length_agent")
    output_path = os.path.join(results_dir(audio_dir), "audio_length.csv")
    task_prompt = f"""You are given audio files in the folder: {audio_dir}.
    Write a Python script to do the following:
    1. Create a CSV with columns Filename and Audio_length.
    2. For each audio file (.wav, .mp3, .flac), calculate its duration in seconds (use soundfile.info(path).duration) and store it in 'Audio_length' column.
    3. Save the CSV at exactly this path: {output_path}
    """
    try:
        if run_agent(task_prompt, output_path, check_audio_length(audio_dir)):
            return {"audio_length_output": f"CSV saved at: {output_path}"}
        return {"audio_length_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Audio length calculation failed: {e}")
        return {"audio_length_output": f"Error: {e}"}

def language_verification_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    csv_path = state.get('A', os.path.join(results_dir(audio_dir), "indicconf_hypothesis.csv"))
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for language verification: {csv_path}")
        return {"language_verification_output": f"Error: CSV file {csv_path} not found"}
    script, lo, hi = SCRIPTS.get(language_of(state), SCRIPTS["Hindi"])
    col = f"Is_{script}"
    logging.info(f"Running {script} script verification with CSV: {csv_path}")
    output_path = os.path.join(os.path.dirname(csv_path), "language_verification.csv")
    try:
        df = pd.read_csv(csv_path)
        fcol = next(c for c in df.columns if c.lower() == "filename")
        tcol = next(c for c in df.columns if c.lower() in ("ground_truth", "transcription", "indiconformer_hypothesis"))
        texts = df[tcol].fillna("").astype(str)
        def share(t):   # share of letters (tags removed) inside the script's Unicode block
            letters = [ch for w in words(t) for ch in w if not ch.isdigit()]
            return sum(lo <= ord(ch) <= hi for ch in letters) / len(letters) if letters else 0.0
        out = pd.DataFrame({"Filename": df[fcol], "Transcription": texts, f"{script}_Share": texts.apply(share).round(3)})
        out[col] = texts.apply(lambda t: bool(words(t)) and share(t) == 1.0)
        out.to_csv(output_path, index=False)
        logging.info(f"{script} verification CSV saved to: {output_path}")
        return {"language_verification_output": f"CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Script verification failed: {e}")
        return {"language_verification_output": f"Error: {e}"}

def ctc_score_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/\\"))
    # per file: score the reference against the audio when the file has one, else the ASR hypothesis
    gt_csv = state.get('ground_truth_csv') or os.path.join(parent, 'results', "indicconf_hypothesis-gt.csv")
    hyp_csv = state.get('A', os.path.join(parent, 'results', "indicconf_hypothesis.csv"))
    csv_path, sources = hyp_csv, {}
    gt_cols = {c.lower(): c for c in pd.read_csv(gt_csv, nrows=0).columns} if os.path.isfile(gt_csv) else {}
    if {"filename", "ground_truth"} <= gt_cols.keys() and os.path.isfile(hyp_csv):
        gt, hyp = pd.read_csv(gt_csv), pd.read_csv(hyp_csv)
        refs = dict(zip(gt[gt_cols["filename"]].astype(str), gt[gt_cols["ground_truth"]]))
        rows = []
        for f, h in zip(hyp["Filename"].astype(str), hyp["Indiconformer_Hypothesis"]):
            has_ref = f in refs and str(refs[f]).strip() not in ("", "nan")
            rows.append({"Filename": f, "Transcription": refs[f] if has_ref else h})
            sources[f] = "reference" if has_ref else "ASR"
        csv_path = os.path.join(parent, 'results', "ctc_input.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    elif os.path.isfile(gt_csv):
        csv_path = gt_csv
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for CTC score: {audio_dir}")
        return {"ctc_score_output": "Error: Invalid audio directory"}
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for CTC score: {csv_path}")
        return {"ctc_score_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running CTC score calculation")
    try:
        output_path = os.path.join(os.path.dirname(csv_path), "ctc_scores.csv")
        results = process_audio_directory(audio_dir, csv_path, output_path, language=language_of(state))
        if results:
            df = pd.DataFrame(results)
            if df.empty:
                logging.error("No valid results generated from process_audio_directory")
                return {"ctc_score_output": "Error: No valid results generated"}
            grouped = df.groupby('filename').agg({
                'label': lambda x: ' '.join(x), 
                'average_ctc_score': 'first',  
            }).reset_index()
            grouped['Aligned_Segments'] = grouped['filename'].apply(
                lambda x: json.dumps([
                    {'label': row['label'], 'start': row['start_time'], 'end': row['end_time'], 'score': row['score']}
                    for _, row in df[df['filename'] == x].iterrows()
                ])
            )
            grouped.columns = ['Filename', 'Aligned_Transcript', 'CTC_Score', 'Aligned_Segments']
            grouped['CTC_Status'] = grouped['CTC_Score'].apply(
                lambda x: "Good" if float(x) > 0.7 else "Medium" if float(x) > 0.5 else "Poor"
            )
            grouped['Transcript_Source'] = grouped['Filename'].astype(str).map(sources).fillna(
                "reference" if csv_path == gt_csv else "ASR")
            grouped = grouped[['Filename', 'Aligned_Segments', 'Aligned_Transcript', 'CTC_Score', 'CTC_Status',
                               'Transcript_Source']]
            grouped.to_csv(output_path, index=False)
            logging.info(f"CTC scores saved to: {output_path}")
            return {"ctc_score_output": f"CSV saved at: {output_path}"}
        else:
            logging.error(f"No results generated for CTC score calculation")
            return {"ctc_score_output": f"Error: No results generated"}
    except Exception as e:
        logging.error(f"CTC score calculation failed: {e}")
        return {"ctc_score_output": f"Error: {e}"}

def transcript_quality_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    csv_path = state.get('A', os.path.join(results_dir(audio_dir), "indicconf_hypothesis.csv"))
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transcript quality: {csv_path}")
        return {"transcript_quality_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running transcript quality check")
    try:
        df = pd.read_csv(csv_path)
        col = next(c for c in df.columns if c.lower() in ("ground_truth", "transcription", "indiconformer_hypothesis"))
        fcol = next(c for c in df.columns if c.lower() == "filename")
        # speech time = duration - silence (task 8), so text-per-second is measured over speech only
        audio = {f: librosa.get_duration(path=os.path.join(audio_dir, f)) for f in df[fcol]
                 if os.path.isfile(os.path.join(audio_dir, f))}
        sil_csv = os.path.join(os.path.dirname(csv_path), "vad_silence_stats.csv")
        silence = dict(pd.read_csv(sil_csv).values[:, :2]) if os.path.isfile(sil_csv) else {}
        speech = {f: d - float(silence.get(f, 0)) for f, d in audio.items()}
        df['Speech_s'] = df[fcol].map(speech).round(1)
        df['Quality_Check'] = [transcript_quality(t, speech.get(f)) for f, t in zip(df[fcol], df[col].fillna("").astype(str))]
        # MoE score (paper: WER/CER across several ASR models): the ASR experts' transcripts against the main one
        moe_path = os.path.join(os.path.dirname(csv_path), "asr_moe.csv")
        if col.lower() == "indiconformer_hypothesis" and os.path.isfile(moe_path):
            moe = pd.read_csv(moe_path).drop(columns=["Main_ASR_Model"], errors="ignore")
            df = df.merge(moe, left_on=fcol, right_on="Filename", how="left", suffixes=("", "_moe"))
            df = df.drop(columns=[c for c in df.columns if c.endswith("_moe")])
            disagree = (df["Quality_Check"] == "passed") & (pd.to_numeric(df["Main_Agreement_CER"], errors="coerce") > MOE_MAX_CER)
            df.loc[disagree, "Quality_Check"] = df.loc[disagree, "Main_Agreement_CER"].map(
                lambda c: f"failed: ASR experts disagree (closest expert differs on {c:.0%} of the characters)")
        output_path = os.path.join(os.path.dirname(csv_path), "transcript_quality.csv")
        df.to_csv(output_path, index=False)
        return {"transcript_quality_output": f"CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Transcript quality check failed: {e}")
        return {"transcript_quality_output": f"Error: {e}"}

def upsampling_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for upsampling check: {audio_dir}")
        return {"upsampling_output": "Error: Invalid audio directory"}
    logging.info("Running upsampling check")
    result = check_upsampling_folder(audio_dir)
    return {"upsampling_output": result}

def valid_speaker_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for valid speaker check: {audio_dir}")
        return {"valid_speaker_output": "Error: Invalid audio directory"}
    logging.info("Running valid speaker check")
    res = results_dir(audio_dir)
    input_path = os.path.join(res, "num_speakers.csv")
    output_path = os.path.join(res, "valid_speaker.csv")
    # deterministic: the model-written scripts for this simple join failed on real data (column handling)
    try:
        df = pd.read_csv(input_path)
        df = df[df["Number of Speakers"].astype(str) != "Error"]
        speakers = {str(f): set(json.loads(d)) for f, d in zip(df["File Name"], df["Speaker Durations"])}
        # speakers heard in earlier batches (config.SPEAKER_DB), as written by the diarization step
        earlier = {str(f): json.loads(d) if isinstance(d, str) and d.strip() else {}
                   for f, d in zip(df["File Name"], df.get("Earlier Batches", pd.Series([None] * len(df))))}
        rows = []
        for f, ids in speakers.items():
            others = [g for g, other in speakers.items() if g != f and ids & other]
            before = sorted({x for files in earlier.get(f, {}).values() for x in files})
            rows.append({"Filename": f, "Speaker_Status": "Old" if others or before else "New",
                         "Common_File": ";".join(others), "Earlier_Batch_Files": ";".join(before)})
        pd.DataFrame(rows, columns=["Filename", "Speaker_Status", "Common_File", "Earlier_Batch_Files"]).to_csv(output_path, index=False)
        return {"valid_speaker_output": f"CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Valid speaker check failed: {e}")
        return {"valid_speaker_output": f"Error: {e}"}

DOMAINS = ['Conversation', 'Keywords Spotting', 'Product Review', 'GK Questions', 'District Specific', 'KYP - Games',
           'Task of Fives', 'DOI - Religion', 'KYP - Cooking', 'Daily Life', 'Healthcare', 'Insurance', 'Finance', 'Sales']
# what the labels most often confused mean (without these the model called every sales call Finance;
# qc_audit/domain_bench)
DOMAIN_DEFINITIONS = """- Healthcare: a doctor, counsellor or hospital talking with a patient about their health: symptoms, feelings, tests, diagnosis, treatment or medicine.
- Insurance: insurance policies, premiums, coverage, claims or cashless treatment, including health insurance.
- Finance: loans, EMIs, payment reminders or collection of dues, bank accounts, credit or payments.
- Sales: a seller offering a new product, course or service and trying to get the customer to buy it or sign up (features, price, offer, discount, demo)."""


def classify_domain(text: str, language: str, model=None) -> str:
    """One label of DOMAINS for a transcript, or Unknown. The first 3000 characters only: Indic scripts take several
    tokens per character, and a longer text pushed the instructions out of the context (Ollama cuts the prompt's start),
    so the model continued the Kannada transcript instead of labelling it."""
    prompt = f"""You are a {language} language expert. Read this {language} transcript of a recorded conversation and decide what it is about.
    Labels: {DOMAINS}
    What the most easily confused labels mean:
{DOMAIN_DEFINITIONS}
    Reply with exactly one label from the list and nothing else.
    Transcript: {text[:3000]}
    (End of transcript. Reply with one label from {DOMAINS}.)
    Label:"""
    # short bounded reply, accepted only if it names a listed domain: on long texts the model sometimes writes an
    # essay instead of a label (seen on a Bengali reference transcript)
    reply = (model or llm).model_copy(update={"num_predict": 30}).invoke(prompt).content
    reply = re.sub(r"^\W*(domain|label)\W*", "", reply.strip(), flags=re.IGNORECASE).lower()
    found = sorted((d for d in DOMAINS if d.lower() in reply), key=lambda d: reply.index(d.lower()))
    return found[0] if found else "Unknown"


def domain_checker_agent(state: CombinedStateDict) -> CombinedStateDict:
    """Domain per file, from the reference transcript where there is one (the ASR text loses the words that tell the
    domain apart), else from the ASR transcript."""
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for domain checker: {audio_dir}")
        return {"domain_checker_output": "Error: Invalid audio directory"}
    res = results_dir(audio_dir)
    csv_path = os.path.join(res, "indicconf_hypothesis.csv")
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return {"domain_checker_output": f"Error: CSV file not found at {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        if 'Indiconformer_Hypothesis' not in df.columns:
            logging.error("Column 'Indiconformer_Hypothesis' not found in CSV.")
            return {"domain_checker_output": "Error: 'Indiconformer_Hypothesis' column missing."}
        refs = {}
        gt = os.path.join(res, "indicconf_hypothesis-gt.csv")
        if os.path.isfile(gt):
            g = pd.read_csv(gt).rename(columns=str.lower)
            refs = {str(f): str(t) for f, t in zip(g["filename"], g["ground_truth"]) if words(str(t))}
        domains, sources = [], []
        for i, (fname, transcript) in enumerate(zip(df['Filename'].astype(str), df['Indiconformer_Hypothesis'].fillna("").astype(str))):
            text, source = (" ".join(words(refs[fname])), "reference") if fname in refs else (transcript, "ASR")
            try:
                domains.append(classify_domain(text, language_of(state)))
            except Exception as e:
                logging.warning(f"LLM failed on row {i}: {e}")
                domains.append("Unknown")
            sources.append(source)
        df['domain'] = domains
        df['Domain_From'] = sources
        output_path = os.path.join(res, "domain_check.csv")
        df.to_csv(output_path, index=False)
        logging.info(f"Domain check completed. Output saved to {output_path}")
        return {"domain_checker_output": f"CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Domain checker failed: {e}")
        return {"domain_checker_output": f"Error: {e}"}


def audio_transcript_matching_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    csv_path = state.get('A', os.path.join(results_dir(audio_dir), "indicconf_hypothesis.csv"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for transcript matching: {audio_dir}")
        return {"audio_transcript_matching_output": "Error: Invalid audio directory"}
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transcript matching: {csv_path}")
        return {"audio_transcript_matching_output": f"Error: CSV file {csv_path} not found"}
    logging.info("Running audio and transcript matching")
    output_path = os.path.join(results_dir(audio_dir), "audio_transcript_matching.csv")
    task_prompt = f"""You are given a folder at this path: {audio_dir} containing audio files (.wav or .mp3) and a CSV file at {csv_path}.
    The CSV has columns 'Filename' and 'Indiconformer_Hypothesis', where 'Indiconformer_Hypothesis' contains the transcriptions.
    Write a Python script to:
    1. Load the CSV file.
    2. For each row in the CSV:
    - Get the audio file path by joining the folder path with the 'Filename'.
    - Call `from utility_functions import force_alignment_and_ctc_score` and run
      `aligned_segments, avg_score = force_alignment_and_ctc_score(audio_path, transcript, "{align_model_for(language_of(state))}")`.
      Each aligned segment is a list [token, start_time, end_time, score].
    - Create an aligned transcript by joining the tokens from aligned_segments.
    3. Create a CSV with columns 'Filename', 'Aligned_Segments' (JSON string), 'Aligned_Transcript' and 'CTC_Score' (avg_score).
    4. Save the CSV at exactly this path: {output_path}
    5. Handle errors gracefully.
    """
    try:
        if run_agent(task_prompt, output_path, check_alignment(csv_path, audio_dir, align_model_for(language_of(state)))):
            return {"audio_transcript_matching_output": f"CSV saved at: {output_path}"}
        logging.error(f"Transcript matching failed to generate {output_path}")
        return {"audio_transcript_matching_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Transcript matching failed: {e}")
        return {"audio_transcript_matching_output": f"Error: {e}"}


SPOKEN_SAME = [{"Hindi", "Urdu"}]   # one spoken language (Hindustani), two scripts: audio cannot tell them apart


def same_spoken_language(heard: str, expected: str) -> bool:
    return heard == expected or any({heard, expected} <= group for group in SPOKEN_SAME)


def language_identification_indiclid_agent(state: CombinedStateDict) -> CombinedStateDict:
    """IndicLID on the ASR transcript (as in the paper) and, from the audio itself, Whisper / MMS-LID: the text can
    only be in the language the ASR was told to write, so English audio in a Hindi folder came out as Hindi-script
    text that IndicLID called Kashmiri; the audio models hear English."""
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error("Invalid or missing audio directory")
        return {"language_identification_indiclid_output": "Error: Invalid audio directory"}
    parent = os.path.dirname(audio_dir.rstrip("/\\"))
    output_path = os.path.join(parent, 'results', "indiclid_language_identification.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        logging.info(f"Skipping language identification: File already exists at {output_path}")
        return {"language_identification_indiclid_output": f"File already exists: {output_path}"}
    csv_path = state.get('A', os.path.join(parent, 'results', "indicconf_hypothesis.csv"))
    if not os.path.exists(csv_path):
        logging.error(f"Transcription CSV not found at {csv_path}")
        return {"language_identification_indiclid_output": f"Error: Transcription CSV not found at {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        if 'Indiconformer_Hypothesis' not in df.columns:
            logging.error("Missing 'Indiconformer_Hypothesis' column in CSV")
            return {"language_identification_indiclid_output": "Error: Missing 'Indiconformer_Hypothesis' column"}
        expected = language_of(state)
        results = []
        for idx, row in df.iterrows():
            transcription = "" if pd.isna(row['Indiconformer_Hypothesis']) else str(row['Indiconformer_Hypothesis'])
            filename = row.get('Filename', f"unknown_{idx}")
            r = {"Filename": filename, "Transcription": transcription, "Language_Code": "Unknown", "Confidence": 0.0,
                 "Model_Used": "IndicLID"}
            if transcription.strip():
                try:
                    _, r["Language_Code"], r["Confidence"], r["Model_Used"] = language_identification_indiclid(transcription)[0]
                except Exception as e:
                    logging.error(f"Language identification failed for {filename}: {e}")
                    r["Language_Code"] = "Error"
            else:
                logging.warning(f"Empty transcription for {filename}")
            try:
                r.update(audio_language(os.path.join(audio_dir, str(filename))))
                r["Audio_Matches_Expected"] = same_spoken_language(r["Audio_Language"], expected)
            except Exception as e:
                logging.error(f"Audio language identification failed for {filename}: {e}")
                r["Audio_Language"] = "Error"
            results.append(r)
        if not results:
            logging.error("No language identification results generated")
            return {"language_identification_indiclid_output": "Error: No language identification results"}
        out = pd.DataFrame(results)
        out.to_csv(output_path, index=False)
        logging.info(f"Language identification results saved to {output_path}")
        other = [f"{f} ({lang})" for f, lang, ok in zip(out["Filename"], out.get("Audio_Language", []), out.get("Audio_Matches_Expected", []))
                 if ok is False]
        found = (f"files that sound like another language than {expected} (a finding about the data): {', '.join(other)}"
                 if other else f"every file sounds like {expected}")
        return {"language_identification_indiclid_output": f"{found} - CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Error processing language identification: {e}")
        return {"language_identification_indiclid_output": f"Error: {e}"}


def normalization_remove_tags_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    res = results_dir(audio_dir)
    csv_path = os.path.join(res, "indicconf_hypothesis-gt.csv")
    output_path = os.path.join(res, "normalized_list.csv")
    if os.path.exists(output_path):
        return {"normalization_remove_tags_output": f"CSV already exists at: {output_path}"}
    if not os.path.isfile(csv_path):
        logging.error(f"Ground truth CSV for normalization not found: {csv_path}")
        return {"normalization_remove_tags_output": f"Error: CSV file {csv_path} not found (needs columns filename, ground_truth)"}
    logging.info("Running normalizing_agent")
    task_prompt = f"""You are given a CSV file at this path: {csv_path}
    It contains two columns: filename and ground_truth (search case insensitively).
    Write a Python script that cleans the ground_truth text, saves it in a new column called normalized_transcripts, then writes the updated data to a new CSV.
    The cleaning steps are:
    -Remove HTML tags like <b> or </b>.
    -Remove any text enclosed in square brackets (e.g., [START]).
    -Remove all symbols like ['#','$','%']
    -Keep every letter of any script (English words in code-mixed text stay), digits and combining marks (Indic vowel signs).
    -Collapse repeated whitespace.
    -The output CSV should preserve the original columns and include the cleaned transcript in the normalized_transcripts column.
    -Save the updated CSV at exactly this path: {output_path}
    """
    try:
        if run_agent(task_prompt, output_path, check_normalization(csv_path)):
            return {"normalization_remove_tags_output": f"CSV saved at: {output_path}"}
        error_msg = f"Failed to generate {output_path}"
        logging.error(error_msg)
        return {"normalization_remove_tags_output": error_msg}
    except Exception as e:
        logging.error(f"Normalization failed: {e}")
        return {"normalization_remove_tags_output": f"Error: {e}"}

JUDGE_CHUNKS = 30   # ASR chunks (<= 8 s of speech each) per file the judge rates one by one


def judge_chunks(texts: list, language: str) -> list:
    """A 0-10 score per ASR chunk text for how likely it is an accurate transcript of real {language} speech (None
    when the reply had none for it). Scoring short pieces separates good from bad transcripts far better than one
    0-10 score per file, which came out 8 for almost every file: Spearman with the file WER -0.57 vs +0.10, and
    better than OK/garbled verdicts per chunk (0.43) or counting wrong-looking words (0.58 but noisier per chunk;
    qc_audit/judge_bench)."""
    lines = "\n".join(f"{i + 1}: {t[:300]}" for i, t in enumerate(texts))
    prompt = f"""You check pieces of an automatic {language} speech transcript, each a few seconds of a conversation.
    For every piece give a score from 0 to 10 for how likely it is an accurate transcript of real {language} speech:
    10 = natural, sensible {language}; 5 = partly sensible with some broken or wrong-looking words; 0 = garbled word salad.
    A short or unfinished fragment can still score high. Use the whole scale.
    Reply with one line per piece, in order, exactly like "3: 7", and nothing else.
{lines}"""
    out = judge_llm.model_copy(update={"num_predict": 12 * len(texts) + 50}).invoke(prompt).content
    scores = {int(n): min(10.0, float(v)) for n, v in re.findall(r"(\d+)\s*[:.)-]\s*(\d+(?:\.\d+)?)", out)}
    return [scores.get(i + 1) for i in range(len(texts))]


def llm_score_agent(state: CombinedStateDict) -> CombinedStateDict:
    """LLM-as-a-judge (as in the paper): every ASR transcript is scored 0-10 for coherence and fluency, with a comment.
    Also, from results/asr_chunks.csv, up to JUDGE_CHUNKS chunks per file are scored 0-10 one by one:
    LLM_Chunk_Score is their mean, Low_Score_Chunk_Share the share scored 4 or less (asr_chunk_judgements.csv has
    every chunk's score)."""
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for LLM score: {audio_dir}")
        return {"llm_score_output": "Error: Invalid audio directory"}
    logging.info("Running LLM score evaluation")
    res = results_dir(audio_dir)
    csv_path = state.get('A', os.path.join(res, "indicconf_hypothesis.csv"))
    if not os.path.isfile(csv_path):
        return {"llm_score_output": f"Error: CSV file {csv_path} not found"}
    try:
        df = pd.read_csv(csv_path)
        fcol = next(c for c in df.columns if c.lower() == "filename")
        tcol = next(c for c in df.columns if c.lower() in ("ground_truth", "transcriptions", "transcription", "indiconformer_hypothesis"))
        chunks_path = os.path.join(res, "asr_chunks.csv")
        chunks = pd.read_csv(chunks_path) if os.path.isfile(chunks_path) else None
        rows, judged = [], []
        for fname, text in zip(df[fcol].astype(str), df[tcol].fillna("").astype(str)):
            prompt = f"""You are a strict judge of {language_of(state)} speech transcriptions.
    Evaluate the coherence and fluency of this transcription. Understand the meaning before scoring; do not give every transcription the same score.
    Score from 0 to 10: 10 for fluent, meaningful sentences, lower the more meaningless it gets, and 0 if it is in a language other than {language_of(state)}.
    Reply in exactly this format:
    Score: <0-10>
    Comment: <one sentence>
    Transcription: {text[:4000]}"""
            row = {"Filename": fname, "Transcription": text, "LLM_Score": None, "Evaluation_Comment": None}
            try:
                out = judge_llm.model_copy(update={"num_predict": 200}).invoke(prompt).content
                m = re.search(r'Score:\s*(\d+(?:\.\d+)?)', out)
                c = re.search(r'Comment:\s*(.*)', out, re.DOTALL)
                row["LLM_Score"] = min(10.0, float(m.group(1))) if m else None
                row["Evaluation_Comment"] = c.group(1).strip() if c else "Error: reply had no score"
            except Exception as e:
                logging.warning(f"LLM score failed for {fname}: {e}")
                row["Evaluation_Comment"] = f"Error: {e}"
            if chunks is not None:
                mine = chunks[(chunks["Filename"].astype(str) == fname)]
                spoken = mine[mine["Text"].fillna("").astype(str).str.split().str.len() > 0]
                row["Empty_Chunk_Share"] = round(1 - len(spoken) / len(mine), 3) if len(mine) else None
                if len(spoken):
                    pick = spoken.iloc[np.unique(np.linspace(0, len(spoken) - 1, min(JUDGE_CHUNKS, len(spoken))).astype(int))]
                    try:
                        scores = judge_chunks(pick["Text"].astype(str).tolist(), language_of(state))
                    except Exception as e:
                        logging.warning(f"chunk judging failed for {fname}: {e}")
                        scores = [None] * len(pick)
                    rated = [x for x in scores if x is not None]
                    row["Chunks_Judged"] = len(rated)
                    row["LLM_Chunk_Score"] = round(float(np.mean(rated)), 2) if rated else None
                    row["Low_Score_Chunk_Share"] = round(sum(x <= 4 for x in rated) / len(rated), 3) if rated else None
                    judged += [{"Filename": fname, "Chunk": c, "Start_s": a, "End_s": e, "Text": t, "Judge_Score": x}
                               for c, a, e, t, x in zip(pick["Chunk"], pick["Start_s"], pick["End_s"], pick["Text"], scores)]
            rows.append(row)
        if judged:
            pd.DataFrame(judged).to_csv(os.path.join(res, "asr_chunk_judgements.csv"), index=False)
        output_path = os.path.join(res, "llm_scores.csv")
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return {"llm_score_output": f"CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"LLM score evaluation failed: {e}")
        return {"llm_score_output": f"Error: {e}"}


def transliteration_agent(state: CombinedStateDict) -> CombinedStateDict:
    # the reference transcripts: given in the prompt, else the ones prepare_runs.py put into results/
    csv_path = state.get('ground_truth_csv') or (state.get('audio_dir') and os.path.join(
        results_dir(state['audio_dir']), "indicconf_hypothesis-gt.csv"))
    lang_code = state.get('lang_code')
    if not csv_path or not os.path.isfile(csv_path):
        logging.error(f"Invalid ground truth CSV for transliteration: {csv_path}")
        return {"transliteration_output": f"Error: CSV file {csv_path} not found"}
    if not lang_code:
        logging.error("No language code provided for transliteration")
        return {"transliteration_output": "Error: No language code provided"}
    if LANG_CODES.get(lang_code) == "English":
        return {"transliteration_output": "Skipped: English is written in Roman script; nothing to transliterate"}
    logging.info("Running transliteration")
    try:
        result = transliterate_file(csv_path, lang_code, llm=llm)
        return {"transliteration_output": result}
    except Exception as e:
        logging.error(f"Transliteration failed: {e}")
        return {"transliteration_output": f"Error: {e}"}



def speaker_duration_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/\\"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error("Invalid or missing audio directory")
        return {"speaker_duration_output": "Error: Invalid audio directory"}
    
    csv_path = os.path.join(parent, 'results', "num_speakers.csv")
    output_path = os.path.join(parent,'results', "speaker_durations.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(csv_path):
        logging.error(f"Speaker CSV not found: {csv_path}")
        return {"speaker_duration_output": f"Error: Speaker CSV not found: {csv_path}"}
    
    try:
        df = pd.read_csv(csv_path)
        if not all(col in df.columns for col in ['File Name', 'Number of Speakers', 'Speaker Durations']):
            logging.error("Missing required columns in num_speakers.csv")
            return {"speaker_duration_output": "Error: Missing required columns in num_speakers.csv"}
        
        speaker_durations = {}
        for _, row in df.iterrows():
            try:
                durations = json.loads(row['Speaker Durations'].replace('""', '"'))
                for speaker, duration in durations.items():
                    speaker_durations[speaker] = speaker_durations.get(speaker, 0.0) + duration
            except Exception:
                continue
        
        if not speaker_durations:
            logging.error("No speaker durations found")
            return {"speaker_duration_output": "Error: No speaker durations found"}
        
        results = [
            {'Speaker': speaker, 'Total_Duration_Hours': round(duration, 6)}
            for speaker, duration in sorted(speaker_durations.items())
        ]
        
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_path, index=False)
        logging.info(f"Speaker durations saved to {output_path}")
        return {"speaker_duration_output": output_path}
    
    except Exception as e:
        logging.error(f"Error processing speaker durations: {e}")
        return {"speaker_duration_output": f"Error: {e}"}


def english_word_count_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/\\"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for word counter: {audio_dir}")
        return {"english_word_count_output": "Error: Invalid audio directory"}

    csv_path = os.path.join(parent, 'results', "normalized_list.csv")
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return {"english_word_count_output": f"Error: CSV file not found at {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        if 'ground_truth' not in df.columns:
            logging.error("Column 'ground_truth' not found in CSV.")
            return {"english_word_count_output": "Error: 'ground_truth' column missing."}
        # deterministic: English in these transcripts is written in Latin script, so count Latin-script words
        # ([tags] excluded); the LLM's counts were off by 2x and more on long transcripts
        word_counts = [sum(1 for w in words(t) if re.fullmatch(r"[A-Za-z][A-Za-z'.-]*", w))
                       for t in df['ground_truth'].fillna("").astype(str)]
        df['english_word_count'] = word_counts
        output_path = os.path.join(parent, 'results', "english_word_count.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        logging.info(f"English word count completed. Output saved to {output_path}")
        return {"english_word_count_output": f"CSV saved at: {output_path}"}

    except Exception as e:
        logging.error(f"English word count agent failed: {e}")
        return {"english_word_count_output": f"Error: {e}"}

def utterance_duplicate_checker_agent(state: CombinedStateDict) -> CombinedStateDict:
    """Duplicates (task 23), written to two files:
    - duplicate_utterances.csv: exact repeats in the transcript columns (as in the paper), and
    - near_duplicates.csv: utterances whose text is nearly the same (TF-IDF of character 3-grams, cosine >= 0.9,
      utterances of 4+ words), recordings that share audio (landmark fingerprints: also re-encoded, trimmed or
      embedded copies), and both against the public corpora indexed by build_corpus_index.py.
    Utterances are the reference's time-stamped segments where there are any, else whole references, else the ASR
    chunks, so the check also runs on a batch without references."""
    import duplicates
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for duplicate checker: {audio_dir}")
        return {"utterance_duplicate_output": "Error: Invalid audio directory"}
    res = results_dir(audio_dir)
    try:
        rows = []
        csv_path = os.path.join(res, "normalized_list.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for column in df.columns:
                if df[column].dtype == 'object':
                    dup = df[column][df[column].duplicated(keep=False)]
                    for utterance, count in dup.value_counts().items():
                        rows.append({"column_name": column, "utterance": utterance, "count": count})
        output_path = os.path.join(res, "duplicate_utterances.csv")
        # written even when empty, so the summary and the monitor can see the check ran
        pd.DataFrame(rows, columns=["column_name", "utterance", "count"]).to_csv(output_path, index=False)

        # utterances: (file, utterance label, text)
        seg_path, gt_path, ch_path = (os.path.join(res, n) for n in ("reference_segments.csv", "indicconf_hypothesis-gt.csv", "asr_chunks.csv"))
        if os.path.isfile(seg_path):
            u = pd.read_csv(seg_path)
            utts, source = [(str(f), f"{a:.1f}-{b:.1f} s", t) for f, a, b, t in zip(u.filename, u.start, u.end, u.text.fillna(""))], "reference segments"
        elif os.path.isfile(gt_path):
            u = pd.read_csv(gt_path).rename(columns=str.lower)
            utts, source = [(str(f), "whole file", t) for f, t in zip(u.filename, u.ground_truth.fillna(""))], "references"
        elif os.path.isfile(ch_path):
            u = pd.read_csv(ch_path)
            utts, source = [(str(f), f"{a:.1f}-{b:.1f} s", t) for f, a, b, t in zip(u.Filename, u.Start_s, u.End_s, u.Text.fillna(""))], "ASR chunks"
        else:
            utts, source = [], "none"
        texts = [" ".join(words(str(t))).lower() for _, _, t in utts]
        near = []
        for i, j, sim in duplicates.text_matches(texts, texts, same_set=True):
            near.append({"Kind": "text", "Scope": "batch", "File_A": utts[i][0], "Where_A": utts[i][1],
                         "File_B": utts[j][0], "Where_B": utts[j][1], "Similarity": sim, "Same_File": utts[i][0] == utts[j][0],
                         "Text_A": texts[i], "Text_B": texts[j]})
        paths = [os.path.join(audio_dir, f) for f in list_files(audio_dir, AUDIO_EXT)]
        pairs, prints = duplicates.audio_near_duplicates(paths, load_audio)
        for r in pairs:
            near.append({"Kind": "audio", "Scope": "batch", "File_A": r["File_A"], "Where_A": f"from {max(0.0, -r['Offset_s']):.1f} s",
                         "File_B": r["File_B"], "Where_B": f"from {max(0.0, r['Offset_s']):.1f} s", "Similarity": r["Match_Strength"],
                         "Same_File": False, "Shared_Audio_s": r["Shared_Audio_s"], "Detail": r["Kind"]})
        public = duplicates.load_public_index(PUBLIC_INDEX)
        if public:
            for r in duplicates.public_audio_matches(prints, public):
                near.append({"Kind": "audio", "Scope": f"public: {r['Corpus']}", "File_A": r["Filename"],
                             "Where_A": f"from {max(0.0, -r['Offset_s']):.1f} s", "File_B": r["Corpus_Item"], "Where_B": None,
                             "Similarity": r["Similarity"], "Same_File": False, "Shared_Audio_s": r["Shared_Audio_s"]})
            corpus = [(m["corpus"], m["item"], " ".join(words(m["text"])).lower()) for m in public["meta"] if m.get("text")]
            for i, j, sim in duplicates.text_matches(texts, [c[2] for c in corpus]):
                near.append({"Kind": "text", "Scope": f"public: {corpus[j][0]}", "File_A": utts[i][0], "Where_A": utts[i][1],
                             "File_B": corpus[j][1], "Where_B": None, "Similarity": sim, "Same_File": False,
                             "Text_A": texts[i], "Text_B": corpus[j][2]})
        cols = ["Kind", "Scope", "File_A", "Where_A", "File_B", "Where_B", "Similarity", "Same_File", "Shared_Audio_s",
                "Detail", "Text_A", "Text_B"]
        near_df = pd.DataFrame(near, columns=cols)
        near_df.to_csv(os.path.join(res, "near_duplicates.csv"), index=False)
        # per file, for the summary
        per_file = []
        for f in list_files(audio_dir, AUDIO_EXT):
            batch = near_df[(near_df.Scope == "batch") & ((near_df.File_A == f) | (near_df.File_B == f))]
            text, audio = batch[batch.Kind == "text"], batch[batch.Kind == "audio"]
            per_file.append({"Filename": f, "Near_Duplicate_Utterances": int((~text.Same_File.astype(bool)).sum()),
                             "Repeated_Utterances_In_File": int(text.Same_File.astype(bool).sum()),
                             "Shares_Audio_With": ";".join(sorted({b if a == f else a for a, b in zip(audio.File_A, audio.File_B)})),
                             "Public_Corpus_Matches": int(((near_df.Scope != "batch") & (near_df.File_A == f)).sum())})
        pd.DataFrame(per_file).to_csv(os.path.join(res, "duplicates_per_file.csv"), index=False)
        logging.info(f"Duplicate check completed. Output saved to {output_path}")
        found = (f"{len(rows)} exact duplicate utterances, {int((near_df.Kind == 'text').sum())} near-duplicate text pairs "
                 f"({source}), {int((near_df.Kind == 'audio').sum())} audio matches"
                 + ("" if public else " (no public corpus index: run build_corpus_index.py)"))
        # the near-duplicate table first: the monitor reads the first CSV named (the exact list is often empty)
        return {"utterance_duplicate_output": f"{found} - CSV saved at: {os.path.join(res, 'near_duplicates.csv')} "
                                              f"and {output_path}"}
    except Exception as e:
        logging.error(f"Duplicate utterance checker failed: {e}")
        return {"utterance_duplicate_output": f"Error: {e}"}


import jiwer


def segment_wer(audio_dir, seg_path, language, out_path, max_piece_s=20.0):
    """WER per reference utterance: each time-stamped segment is cut from the audio and transcribed on its own, so an
    error is found where it happens, and cross-talk (segments overlapping another speaker's) can be left out. Writes
    every segment to out_path; returns per file: pooled WER over all segments, over segments without cross-talk,
    the median segment WER and the share of segments with WER > 0.8."""
    segs = pd.read_csv(seg_path)
    engine = main_asr_for(language)[0]
    rows = []
    for fname, g in segs.groupby("filename", sort=False):
        path = os.path.join(audio_dir, str(fname))
        if not os.path.isfile(path):
            continue
        audio = load_audio(path)
        dur = len(audio) / SR
        g = g.sort_values("start").reset_index(drop=True)
        spans = list(zip(g.start.astype(float), g.end.astype(float)))
        pieces, owner = [], []
        for k, ((a, b), text) in enumerate(zip(spans, g.text.fillna("").astype(str))):
            a, b = max(0.0, a), min(b, dur)
            if not words(text) or b - a < 0.2:
                continue
            n = int(np.ceil((b - a) / max_piece_s))
            for m in range(n):   # long segments in equal pieces (the ASR models take up to ~30 s)
                s0, s1 = a + (b - a) * m / n, a + (b - a) * (m + 1) / n
                pieces.append(audio[int(s0 * SR):int(s1 * SR)])
                owner.append(k)
        texts = transcribe_pieces(pieces, language, engine) if pieces else []
        hyps = {}
        for k, t in zip(owner, texts):
            hyps[k] = (hyps.get(k, "") + " " + t).strip()
        for k in sorted(hyps):
            a, b = spans[k]
            ref, hyp = " ".join(words(str(g.text[k]))), " ".join(words(hyps[k]))
            if hyp:
                o = jiwer.process_words(ref, hyp)
                errors = o.substitutions + o.deletions + o.insertions
            else:
                errors = len(ref.split())   # nothing recognised: every word deleted
            cross = any(j != k and g.speaker[j] != g.speaker[k] and min(b, e) - max(a, s0) > 0.2 for j, (s0, e) in enumerate(spans))
            rows.append({"Filename": fname, "Segment": k, "Start_s": a, "End_s": b, "Speaker": g.speaker[k],
                         "Cross_Talk": cross, "Reference": ref, "Hypothesis": hyp, "Ref_Words": len(ref.split()),
                         "Errors": errors, "WER": round(errors / len(ref.split()), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    out = []
    for fname, g in df.groupby("Filename", sort=False):
        clean = g[~g.Cross_Talk]
        out.append({"Filename": fname, "Segments_Scored": len(g),
                    "Segment_WER": round(g.Errors.sum() / g.Ref_Words.sum(), 4),
                    "Segment_WER_No_Cross_Talk": round(clean.Errors.sum() / clean.Ref_Words.sum(), 4) if len(clean) else None,
                    "Median_Segment_WER": round(float(g.WER.median()), 4), "Bad_Segment_Share": round(float((g.WER > 0.8).mean()), 3)})
    return pd.DataFrame(out, columns=["Filename", "Segments_Scored", "Segment_WER", "Segment_WER_No_Cross_Talk",
                                      "Median_Segment_WER", "Bad_Segment_Share"])


def wer_computation_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get("audio_dir")
    parent = os.path.dirname(audio_dir.rstrip("/\\"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for WER computation: {audio_dir}")
        return {"wer_output": "Error: Invalid audio directory"}
    try:
        ref_csv_path = os.path.join(parent, 'results', "normalized_list.csv")
        hyp_csv_path = os.path.join(parent, 'results', "indicconf_hypothesis.csv")
        if not os.path.exists(ref_csv_path) or not os.path.exists(hyp_csv_path):
            return {"wer_output": "Error: One or both CSV files not found"}
        ref_df = pd.read_csv(ref_csv_path)
        hyp_df = pd.read_csv(hyp_csv_path)
        # pair by filename (only some files may have a reference); compare words without punctuation/tags
        refs = dict(zip(ref_df[next(c for c in ref_df.columns if c.lower() == "filename")].astype(str),
                        ref_df["normalized_transcripts"].fillna("").astype(str)))
        experts_path = os.path.join(parent, 'results', "asr_experts.csv")   # the MoE ASR experts (task 1)
        experts = {}
        if os.path.exists(experts_path):
            d = pd.read_csv(experts_path)
            for f, e, t in zip(d["Filename"].astype(str), d["Expert"], d["Hypothesis"].fillna("").astype(str)):
                experts.setdefault(f, {})[e] = t
        wer_rows = []
        for fname, hyp in zip(hyp_df["Filename"].astype(str), hyp_df["Indiconformer_Hypothesis"].fillna("").astype(str)):
            if fname not in refs:
                continue
            ref, hyp = " ".join(words(refs[fname])), " ".join(words(hyp))
            row = {"Filename": fname, "Reference": ref, "Hypothesis": hyp}
            try:
                row["WER"] = round(jiwer.wer(ref, hyp), 4) if ref else "No reference"
                row["CER"] = round(jiwer.cer(ref.replace(" ", ""), hyp.replace(" ", "")), 4) if ref else None
                for expert, text in experts.get(fname, {}).items():   # paper's "WER/CER from multiple ASR models"
                    if ref:
                        other = " ".join(words(text))
                        row[f"WER_{expert}"] = round(jiwer.wer(ref, other), 4)
                        row[f"CER_{expert}"] = round(jiwer.cer(ref.replace(" ", ""), other.replace(" ", "")), 4)
            except Exception as e:
                row["WER"] = "Error"
                logging.warning(f"WER computation failed for {fname}: {e}")
            wer_rows.append(row)
        if not wer_rows:
            return {"wer_output": "Error: no transcribed file has a reference transcript"}
        wer_df = pd.DataFrame(wer_rows)
        seg_path = os.path.join(parent, 'results', "reference_segments.csv")
        if os.path.isfile(seg_path):
            per_file = segment_wer(audio_dir, seg_path, language_of(state), os.path.join(parent, 'results', "segment_wer.csv"))
            wer_df = wer_df.merge(per_file, on="Filename", how="left")
        output_path = os.path.join(parent, 'results', "wer.csv")
        wer_df.to_csv(output_path, index=False)
        logging.info(f"WER computation completed. CSV saved to {output_path}")
        return {"wer_output": f"WER CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"WER computation failed: {e}")
        return {"wer_output": f"Error: {e}"}
def timestamp_segmentation_agent(state: CombinedStateDict) -> CombinedStateDict:
    """Task 25 (paper Table 1): cut every audio file into utterances at the reference time-stamps, check the timing
    (inverted, beyond the audio, overlapping, over 30 s, empty) and align each utterance with its text."""
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        return {"segmentation_output": "Error: Invalid audio directory"}
    res = results_dir(audio_dir)
    seg_path = os.path.join(res, "reference_segments.csv")
    if not os.path.isfile(seg_path):
        return {"segmentation_output": f"Skipped: no reference time-stamps ({seg_path} not found)"}
    logging.info("Running timestamp segmentation")
    try:
        segs = pd.read_csv(seg_path)
        out_dir = os.path.join(res, "segments")
        os.makedirs(out_dir, exist_ok=True)
        model = align_model_for(language_of(state))
        rows, summary = [], []
        for fname, g in segs.groupby("filename", sort=False):
            path = os.path.join(audio_dir, fname)
            if not os.path.isfile(path):
                summary.append({"Filename": fname, "Segments": len(g), "Issue": "audio file not found"})
                continue
            audio = load_audio(path)
            duration, stem = len(audio) / SR, os.path.splitext(fname)[0]
            prev_end, speaker_end = 0.0, {}
            for i, r in enumerate(g.sort_values("start").itertuples(index=False)):
                start, end = float(r.start), float(r.end)
                text = "" if pd.isna(r.text) else str(r.text)
                issues, notes = [], []   # issues: the time-stamps are wrong; notes: normal but worth knowing
                if end <= start:
                    issues.append("ends before it starts")
                if start >= duration or end > duration + 0.05:
                    issues.append(f"beyond the audio end ({duration:.1f} s)")
                if start < speaker_end.get(r.speaker, 0.0) - 0.05:
                    issues.append("overlaps the same speaker's previous segment")
                if end - start > 30:
                    issues.append("longer than 30 s")
                if start < prev_end - 0.05 and start >= speaker_end.get(r.speaker, 0.0) - 0.05:
                    notes.append("overlapping speech (another speaker)")
                if not words(text):
                    notes.append("no words (tags only)")
                prev_end, speaker_end[r.speaker] = max(prev_end, end), max(speaker_end.get(r.speaker, 0.0), end)
                seg_file, score = None, None
                s0, s1 = int(max(0.0, start) * SR), int(min(end, duration) * SR)
                if s1 - s0 >= int(0.1 * SR):
                    seg_file = os.path.join(out_dir, f"{stem}_{i:04d}.wav")
                    sf.write(seg_file, audio[s0:s1], SR)
                    if words(text):
                        score = round(force_alignment_and_ctc_score(seg_file, " ".join(words(text)), model)[1], 3)
                rows.append({"Filename": fname, "Segment": i, "Start_s": start, "End_s": end,
                             "Duration_s": round(end - start, 3), "Speaker": r.speaker, "Text": text,
                             "Segment_File": seg_file, "Alignment_Score": score, "Issues": "; ".join(issues),
                             "Notes": "; ".join(notes)})
            mine = [x for x in rows if x["Filename"] == fname]
            scores = [x["Alignment_Score"] for x in mine if x["Alignment_Score"] is not None]
            summary.append({"Filename": fname, "Segments": len(mine),
                            "Segments_With_Issues": sum(bool(x["Issues"]) for x in mine),
                            "Overlapping_Speech_Segments": sum("overlapping" in x["Notes"] for x in mine),
                            "Segment_Speech_s": round(sum(max(0.0, x["Duration_s"]) for x in mine), 1),
                            "Median_Segment_Alignment": round(float(np.median(scores)), 3) if scores else None})
        pd.DataFrame(rows).to_csv(os.path.join(res, "segments.csv"), index=False)
        output_path = os.path.join(res, "segmentation_summary.csv")
        pd.DataFrame(summary).to_csv(output_path, index=False)
        bad = sum(bool(x["Issues"]) for x in rows)
        return {"segmentation_output": f"{len(rows)} segments, {bad} with timing issues - CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Timestamp segmentation failed: {e}")
        return {"segmentation_output": f"Error: {e}"}


def code_mixing_agent(state: CombinedStateDict) -> CombinedStateDict:
    """Task 26 (paper Table 1): per utterance of the reference (time-stamped segments, else the whole transcript),
    the script of every word, whether the utterance mixes languages and its Code-Mixing Index (Das & Gambäck 2014:
    100 * (1 - words of the dominant language / words that belong to a language))."""
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        return {"code_mixing_output": "Error: Invalid audio directory"}
    res = results_dir(audio_dir)
    seg_path, gt_path = os.path.join(res, "reference_segments.csv"), os.path.join(res, "indicconf_hypothesis-gt.csv")
    logging.info("Running code-mixing detection")
    try:
        if os.path.isfile(seg_path):
            utts = pd.read_csv(seg_path)[["filename", "text"]]
        elif os.path.isfile(gt_path):
            utts = pd.read_csv(gt_path).rename(columns=str.lower)[["filename", "ground_truth"]].rename(columns={"ground_truth": "text"})
        else:
            return {"code_mixing_output": "Skipped: no reference transcripts (the ASR writes one script only)"}
        language = language_of(state)
        _, lo, hi = SCRIPTS.get(language, ("Devanagari", 0x0900, 0x097F))
        native = SCRIPTS.get(language, ("Devanagari",))[0].upper()

        def script_of(word):
            names = Counter()
            for ch in word:
                if unicodedata.category(ch)[0] == "L":
                    names[native if lo <= ord(ch) <= hi else unicodedata.name(ch, "UNKNOWN").split()[0]] += 1
            return names.most_common(1)[0][0] if names else None   # None: digits only, language-independent

        rows = []
        for i, (fname, text) in enumerate(zip(utts["filename"].astype(str), utts["text"].fillna("").astype(str))):
            scripts = [script_of(w) for w in words(text)]
            langs = Counter(s for s in scripts if s)
            counted = sum(langs.values())
            cmi = round(100 * (1 - max(langs.values()) / counted), 1) if counted else 0.0
            rows.append({"Filename": fname, "Utterance": i, "Words": len(scripts), "Native_Words": langs.get(native, 0),
                         "Latin_Words": 0 if native == "LATIN" else langs.get("LATIN", 0),
                         "Other_Script_Words": sum(n for s, n in langs.items() if s not in (native, "LATIN")),
                         "Scripts": ";".join(sorted(langs)), "CMI": cmi, "Code_Mixed": len(langs) > 1, "Text": text})
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(res, "code_mixing.csv"), index=False)
        summary = []
        for fname, g in df.groupby("Filename", sort=False):
            other = sorted({s for x in g["Scripts"] for s in str(x).split(";") if s and s not in (native, "LATIN")})
            total = g["Words"].sum()
            summary.append({"Filename": fname, "Utterances": len(g), "Code_Mixed_Utterances": int(g["Code_Mixed"].sum()),
                            "Code_Mixed_Share": round(g["Code_Mixed"].mean(), 3), "Mean_CMI": round(g["CMI"].mean(), 1),
                            "Embedded_Word_Share": round((total - g["Native_Words"].sum()) / total, 3) if total else 0.0,
                            "Other_Scripts": ";".join(other)})
        output_path = os.path.join(res, "code_mixing_summary.csv")
        pd.DataFrame(summary).to_csv(output_path, index=False)
        return {"code_mixing_output": f"Code-mixed utterances: {int(df['Code_Mixed'].sum())} of {len(df)} - CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Code-mixing detection failed: {e}")
        return {"code_mixing_output": f"Error: {e}"}


# ---------------------------------------------------------------- monitoring LLM (paper §2.4)
# After each node a separate LLM reviews what the node returned and the file it wrote. Nodes whose output an LLM
# produced are run once more when it asks for a retry (with its reason); a deterministic node would only repeat its
# output, so a rejection there is logged for manual review. Every verdict goes to results/monitor_log.csv.
TASK_NAMES = {1: "ASR transcription of every audio file", 2: "number of speakers and duration per speaker (diarization)",
              3: "quality of the ASR transcripts", 4: "grapheme (character) list per transcript",
              5: "vocabulary list per transcript", 6: "language verification: share of the expected script",
              7: "audio length of every file", 8: "silence duration per file (VAD)", 9: "English word count in the references",
              10: "CTC score of the transcripts", 11: "upsampling check (bandwidth and estimated original sample rate)",
              12: "new or old speaker across files", 13: "domain label per transcript",
              14: "forced alignment of the transcripts to the audio",
              15: "language identification: IndicLID on the transcript, Whisper/MMS-LID on the audio",
              16: "normalization of the reference transcripts (tags and symbols removed)",
              17: "LLM-as-a-judge: coherence score (0-10) per transcript and per ASR chunk",
              18: "transliteration of Roman-script words in the references to the native script",
              19: "corruption check of the audio files", 20: "file extension / format check", 21: "sample rate of every file",
              22: "total speaking time per speaker",
              23: "duplicates: exact and near-duplicate utterances, shared audio, public-corpus matches",
              24: "WER and CER of the ASR against the references, per file and per reference segment", 25: "timestamp-based segmentation of the audio into utterances",
              26: "code-mixing detection in the reference transcripts"}
LLM_TASKS = {4, 7, 13, 14, 16, 17, 18, 19, 20, 21}   # outputs written by an LLM: a retry can change them
MONITOR_RULES = True   # problems found by the computed facts decide without the LLM (qc_audit/monitor_bench.py)
MONITOR_FEEDBACK = {}   # output CSV -> the monitor's reason, added to the code writer's prompt on the retry
MONITOR_LOCK = threading.Lock()


def output_facts(status: str, audio_dir: str, task_id: int = 0):
    """(text describing the CSV a node reported, its path, problems) for the monitor: facts computed here (which
    audio files the rows cover, repeated rows, empty cells) so the LLM does not have to count. problems: the facts
    that make an output wrong by themselves."""
    for path in re.findall(r"(/[^\s'\"]+\.csv)", status):
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path)
            except Exception as e:
                return f"{path} cannot be read ({e})", path, [f"the output cannot be read ({e})"]
            facts, problems = [f"{len(df)} rows", f"columns {list(df.columns)}"], []
            fcol = next((c for c in df.columns if c.lower().replace("_", "") == "filename"), None)
            if fcol is not None:
                audio = set(list_files(audio_dir, AUDIO_EXT))
                expected, what = audio, "audio files"
                gt = os.path.join(results_dir(audio_dir), "indicconf_hypothesis-gt.csv")
                if task_id in REFERENCE_TASKS and os.path.isfile(gt):   # these steps only cover files with a reference
                    expected, what = set(csv_filenames(gt)), "audio files that have reference transcripts (the only files this step covers)"
                named = df[fcol].astype(str).map(os.path.basename)
                covered = expected & set(named)
                facts.append(f"rows name {len(covered)} of the {len(expected)} {what}")
                if expected - covered:
                    facts.append(f"expected files with no row: {sorted(expected - covered)[:10]}")
                    problems.append(f"{len(expected - covered)} of the {len(expected)} expected files have no row")
                if set(named) - audio:
                    facts.append(f"rows naming files that are not in the folder: {sorted(set(named) - audio)[:10]}")
            if (dup := int(df.duplicated().sum())):
                facts.append(f"{dup} rows are exact repeats of other rows")
                problems.append(f"{dup} rows are exact repeats of other rows")
            empty = {c: int(n) for c, n in df.isna().sum().items() if n}
            facts.append(f"empty cells per column: {empty or 'none'}")
            # code the LLM wrote must fill every column (run_agent checks the same); fixed-code steps may leave an
            # optional column empty (no shared speaker, no other script)
            if task_id in LLM_TASKS and len(df) and (blank := [c for c in df.columns if df[c].isna().all()]):
                problems.append(f"column(s) {blank} have no values")
            head = df.head(3).astype(str).apply(lambda col: col.str[:100]).to_dict("records")
            facts.append(f"first rows (cells shortened to 100 characters for this display): {head}")
            return f"{path}: " + "; ".join(facts), path, problems
    return "no output file", None, []


def monitor_verdict(task_id: int, status: str, audio_dir: str):
    preview, path, problems = output_facts(status, audio_dir, task_id)
    if problems and MONITOR_RULES:   # wrong by the computed facts alone: no need to ask the LLM (it missed 1 in 4 of these)
        return "RETRY", "; ".join(problems), path
    prompt = f"""You monitor the steps of a speech-dataset quality-control pipeline and decide whether a step ran correctly.
    Step {task_id}: {TASK_NAMES.get(task_id, 'unknown')}
    The audio folder has {len(list_files(audio_dir, AUDIO_EXT))} audio files.
    Status returned by the step: {status[:600]}
    Output: {preview[:2500]}
    Judge only whether the step ran and its output is complete and plausible for the step. Do not judge the data:
    a file reported as corrupt, failing, low-scoring, upsampled or skipped for a stated reason is a valid result.
    Reply in exactly this format:
    Verdict: OK or RETRY or REVIEW
    Reason: <one sentence>
    (RETRY: the output is missing, incomplete or clearly wrong and running the step again could fix it.
     REVIEW: something is wrong that a person should look at.)"""
    out = monitor_llm.model_copy(update={"num_predict": 120}).invoke(prompt).content
    v = re.search(r"Verdict:\s*(OK|RETRY|REVIEW)", out, re.IGNORECASE)
    r = re.search(r"Reason:\s*(.*)", out)
    return (v.group(1).upper() if v else "UNPARSED"), (r.group(1).strip() if r else out.strip()[:300]), path


def log_monitor(audio_dir, task_id, attempt, verdict, reason, status):
    path = os.path.join(results_dir(audio_dir), "monitor_log.csv")
    row = pd.DataFrame([{"Task": task_id, "Step": TASK_NAMES.get(task_id), "Attempt": attempt, "Verdict": verdict,
                         "Reason": reason, "Status": status[:300]}])
    with MONITOR_LOCK:
        row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def monitored(task_id: int, func):
    if monitor_llm is None:
        return func

    def node(state):
        audio_dir = state.get("audio_dir")
        for attempt in (1, 2):
            out = func(state)
            status = next((str(v) for v in out.values()), "") if isinstance(out, dict) else str(out)
            try:
                verdict, reason, path = monitor_verdict(task_id, status, audio_dir)
            except Exception as e:
                verdict, reason, path = "UNCHECKED", f"monitor failed: {e}", None
            retry = verdict == "RETRY" and task_id in LLM_TASKS and attempt == 1
            if verdict == "RETRY" and not retry:
                verdict = "RETRY->REVIEW"   # deterministic step, or already retried
            log_monitor(audio_dir, task_id, attempt, verdict, reason, status)
            if not retry:
                return out
            logging.info(f"monitor: re-running task {task_id}: {reason}")
            if path:
                MONITOR_FEEDBACK[path] = reason
                os.remove(path)
        return out
    return node


node_map = {
    1: ("node_transcription", transcription_func, "A"),
    2: ("node_num_speaker", num_speaker_func, "E"),
    3: ("node_transcript_quality", transcript_quality_agent, "transcript_quality_output"),
    4: ("node_character", character_agent, "character_output"),
    5: ("node_vocab", vocab_agent, "vocab_output"),
    6: ("node_language_verification", language_verification_agent, "language_verification_output"),
    7: ("node_audio_length", audio_length_agent, "audio_length_output"),
    8: ("node_silence", silence_vad_func, "D"),
    9: ("node_code-mix-counter", english_word_count_agent, "english_word_count_output"),
    10: ("node_ctc_score", ctc_score_agent, "ctc_score_output"),
    11: ("node_upsampling", upsampling_agent, "upsampling_output"),
    12: ("node_valid_speaker", valid_speaker_agent, "valid_speaker_output"),
    13: ("node_domain_checker", domain_checker_agent, "domain_checker_output"),
    14: ("node_audio_transcript_matching", audio_transcript_matching_agent, "audio_transcript_matching_output"),
    15: ("node_language_identification_indiclid", language_identification_indiclid_agent, "language_identification_indiclid_output"),
    16: ("node_normalization_remove_tags", normalization_remove_tags_agent, "normalization_remove_tags_output"),
    17: ("node_llm_score", llm_score_agent, "llm_score_output"),
    18: ("node_transliteration", transliteration_agent, "transliteration_output"),
    19: ("node_corruption", corruption_agent, "corruption_output"),
    20: ("node_extension", extension_agent, "extension_output"),
    21: ("node_sample_rate", sample_rate_agent, "sample_rate_output"),
    22: ("speaker_duration",speaker_duration_agent, "speaker_duration_output"),
    23: ("node_utterance_duplicate_checker",utterance_duplicate_checker_agent,"utterance_duplicate_output"),
    24: ("wer_computation",wer_computation_agent,"wer_output"),
    25: ("node_timestamp_segmentation", timestamp_segmentation_agent, "segmentation_output"),
    26: ("node_code_mixing", code_mixing_agent, "code_mixing_output"),
    
}

# Explicit mentions always select their task (the same keyword rules the selection prompt states), so a
# small local model cannot silently drop a check the user asked for by name.
KEYWORD_TASKS = [(r"\bASR\b|transcri(be|ption)", 1), (r"number of speakers|diariz", 2), (r"transcript quality", 3),
                 (r"character|grapheme", 4), (r"\bvocab", 5), (r"verify (the )?language|language verification|expected language", 6),
                 (r"audio length", 7), (r"silence|\bVAD\b", 8), (r"english word", 9), (r"\bCTC\b", 10),
                 (r"upsampl", 11), (r"new or old|valid speaker", 12), (r"\bdomain", 13), (r"forced alignment", 14),
                 (r"IndicLID|language identification", 15), (r"normali[sz]|HTML", 16), (r"LLM score|LLM-as-a-judge|coherence", 17),
                 (r"transliterat", 18), (r"corrupt", 19), (r"extension|file format", 20), (r"sample rate", 21),
                 (r"speaker durations?|duration (for|of) each speaker|durations", 22), (r"duplicate", 23), (r"\bWER\b", 24),
                 (r"segment|time-?stamp", 25), (r"code-?mix", 26)]


def keyword_tasks(prompt: str) -> set:
    return {task for pattern, task in KEYWORD_TASKS if re.search(pattern, prompt, re.IGNORECASE)}


# Hard dependencies between tasks (the same rules the prompts give the LLM), enforced in code so a
# small local model cannot schedule a task before its inputs exist.
REFERENCE_TASKS = {9, 16, 18, 24, 25, 26}   # need results/indicconf_hypothesis-gt.csv (25: its time-stamps)
TASK_DEPS = {3: [1, 8], 17: [1], 4: [1], 5: [1], 6: [1], 10: [1], 13: [1], 14: [1], 15: [1], 16: [1],
             12: [2], 22: [2], 9: [16], 23: [1, 16], 24: [16]}   # 23: the ASR chunks when there is no reference


def add_dependencies(tasks: set) -> set:
    tasks, todo = set(tasks), list(tasks)
    while todo:
        for d in TASK_DEPS.get(todo.pop(), []):
            if d not in tasks:
                tasks.add(d)
                todo.append(d)
    return tasks


def layer_tasks(tasks: set) -> list[list[int]]:
    """Deterministic topological layering: each group holds tasks whose dependencies ran earlier."""
    done, groups, left = set(), [], set(tasks)
    while left:
        ready = sorted(t for t in left if all(d in done or d not in tasks for d in TASK_DEPS.get(t, [])))
        groups.append(ready)
        done.update(ready)
        left -= set(ready)
    return groups


def structure_is_valid(structure: list[list[int]], tasks: set) -> bool:
    flat = [t for g in structure for t in g]
    if sorted(flat) != sorted(tasks):
        return False
    seen = set()
    for g in structure:
        if any(d in tasks and d not in seen for t in g for d in TASK_DEPS.get(t, [])):
            return False
        seen.update(g)
    return True


def build_graph_from_structure(structure: list[list[int]], valid_tasks: set) -> StateGraph:
    graph_builder = StateGraph(CombinedStateDict)
    added_nodes = set()
    valid_structure = [[task for task in group if str(task) in valid_tasks] for group in structure]
    valid_structure = [group for group in valid_structure if group]
    if not valid_structure and valid_tasks:
        valid_structure = [[int(task) for task in valid_tasks if task.isdigit()]]
    for group in valid_structure:
        for task_id in group:
            if task_id in node_map and task_id not in added_nodes:
                node_name, func, _ = node_map[task_id]
                graph_builder.add_node(node_name, monitored(task_id, func))
                added_nodes.add(task_id)
    graph_builder.add_node("start", lambda state: state)
    for i in range(len(valid_structure)):
        current_group = valid_structure[i]
        if i == 0:
            for task_id in current_group:
                node_name, _, _ = node_map[task_id]
                graph_builder.add_edge("start", node_name)
        if i < len(valid_structure) - 1:
            next_group = valid_structure[i + 1]
            for curr_task in current_group:
                curr_node_name, _, _ = node_map[curr_task]
                for next_task in next_group:
                    next_node_name, _, _ = node_map[next_task]
                    graph_builder.add_edge(curr_node_name, next_node_name)
        if i == len(valid_structure) - 1:
            for task_id in current_group:
                node_name, _, _ = node_map[task_id]
                graph_builder.add_edge(node_name, END)
    if valid_structure:
        graph_builder.set_entry_point("start")
    else:
        raise ValueError("No valid tasks provided in structure")
    return graph_builder.compile()
def main(user_prompt: str, audio_dir: Optional[str] = None, lang_code: Optional[str] = None):
    """audio_dir/lang_code given (run_qc.py) are used as is; otherwise both are parsed from the prompt
    (a POSIX absolute path in quotes)."""
    if audio_dir:
        parsed_inputs = {"audio_dir": os.path.abspath(audio_dir), "ground_truth_csv": None, "lang_code": lang_code}
    else:
        parsed_inputs = parse_prompt(user_prompt)
    if not parsed_inputs["audio_dir"] and not parsed_inputs["ground_truth_csv"]:
        logging.error("No valid audio directory or ground truth CSV provided in prompt")
        print("Error: Please provide audio directory and/or ground truth CSV in the prompt")
        return
    log = parsed_inputs["audio_dir"] and os.path.join(results_dir(parsed_inputs["audio_dir"]), "monitor_log.csv")
    if log and os.path.exists(log):
        os.remove(log)   # the monitoring LLM's verdicts are per run
    resp_1 = select_tasks(user_prompt)
    
    verified_tasks = prompt_checker_agent(user_prompt, resp_1)
    selected = {int(x) for x in resp_1.split(',') if x.strip().isdigit() and 1 <= int(x) <= 26}
    checked = {int(x) for x in verified_tasks.split(',') if x.strip().isdigit() and 1 <= int(x) <= 26}
    plan = {"prompt": user_prompt, "selected_by_planner": sorted(selected), "verified_by_checker": sorted(checked)}
    # as in the paper: the prompt checker's verified list is the plan, tasks it adds included
    if checked - selected:
        logging.info(f"Tasks added by the prompt checker: {sorted(checked - selected)}")
    plan["added_by_checker"] = sorted(checked - selected)
    plan["removed_by_checker"] = sorted(selected - checked) if verified_tasks.strip() else []
    tasks = checked if verified_tasks.strip() else selected
    named = keyword_tasks(user_prompt) - tasks
    if named:
        logging.info(f"Added tasks named in the prompt but not selected by the LLM: {sorted(named)}")
        tasks |= named
    plan["added_by_keywords"] = sorted(named)
    with_deps = add_dependencies(tasks)
    if with_deps != tasks:
        logging.info(f"Added dependency tasks: {sorted(with_deps - tasks)}")
    plan["added_as_dependencies"] = sorted(with_deps - tasks)
    tasks = with_deps
    # English-word count, normalization, transliteration, WER, segmentation and code-mixing read the reference
    # transcripts: without them they can only fail, so drop them even when the planner picked them
    reference = parsed_inputs["ground_truth_csv"] or (parsed_inputs["audio_dir"] and os.path.join(
        results_dir(parsed_inputs["audio_dir"]), "indicconf_hypothesis-gt.csv"))
    plan["skipped_no_reference"] = []
    if not (reference and os.path.isfile(reference)) and tasks & REFERENCE_TASKS:
        logging.info(f"No reference transcripts: skipping tasks {sorted(tasks & REFERENCE_TASKS)}")
        print(f"No reference transcripts: skipping tasks {sorted(tasks & REFERENCE_TASKS)} (need indicconf_hypothesis-gt.csv)")
        plan["skipped_no_reference"] = sorted(tasks & REFERENCE_TASKS)
        tasks -= REFERENCE_TASKS
    verified_tasks = ','.join(map(str, sorted(tasks)))
    valid_tasks = {str(x) for x in tasks}
    structure = topological_sort_tasks(verified_tasks)
    plan["structure_from"] = "LLM"
    if not structure_is_valid(structure, tasks):
        logging.info(f"LLM structure {structure} breaks a dependency or drops a task; using deterministic layering")
        structure = layer_tasks(tasks)
        plan["structure_from"] = "deterministic layering (the LLM's broke a dependency or dropped a task)"
    plan.update(tasks=sorted(tasks), structure=structure)
    print("Using structure:", structure)
    if parsed_inputs["audio_dir"]:
        with open(os.path.join(results_dir(parsed_inputs["audio_dir"]), "plan.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=1, ensure_ascii=False)

    main_graph = build_graph_from_structure(structure, valid_tasks)
    main_graph.get_graph().print_ascii()

    initial_state = {
        "audio_dir": parsed_inputs["audio_dir"],
        "ground_truth_csv": parsed_inputs["ground_truth_csv"],
        "lang_code": parsed_inputs["lang_code"],
        "user_prompt": user_prompt
    }
    final_result = main_graph.invoke(initial_state)
    if REPLAN and parsed_inputs["audio_dir"]:
        final_result.update(replan(initial_state, final_result, tasks))
    print("\nFinal Pipeline Results:")
    for key, value in final_result.items():
        if key not in ["audio_dir", "ground_truth_csv", "lang_code", "user_prompt"]:
            print(f"{key}: {value[:100]}..." if isinstance(value, str) and len(value) > 100 else f"{key}: {value}")


# steps that skip their work when their output file exists
REUSED_OUTPUTS = {1: ["indicconf_hypothesis.csv", "asr_experts.csv", "asr_moe.csv", "asr_chunks.csv"], 2: ["num_speakers.csv"],
                  8: ["vad_silence_stats.csv"], 15: ["indiclid_language_identification.csv"], 16: ["normalized_list.csv"]}


def failed_steps(audio_dir: str, result: dict, tasks: set) -> dict:
    """task -> why it failed: an error status, or the monitor's last verdict asked for a re-run (RETRY, also when it
    was already retried). REVIEW (something for a person, often a finding about the data) is left to the person."""
    failed = {}
    for t in tasks:
        status = str(result.get(node_map[t][2], ""))
        if status.startswith("Error") or "Failed to generate" in status:
            failed[t] = f"status: {status[:200]}"
    log = os.path.join(results_dir(audio_dir), "monitor_log.csv")
    if os.path.exists(log):
        last = pd.read_csv(log).groupby("Task").tail(1)
        for r in last.itertuples():
            if int(r.Task) in tasks and str(r.Verdict).startswith("RETRY") and int(r.Task) not in failed:
                failed[int(r.Task)] = f"monitor {r.Verdict}: {r.Reason}"
    return failed


def replan(state: dict, result: dict, tasks: set) -> dict:
    """One re-planning round (paper §2.4: the planner revises the plan from the execution feedback): the planner sees
    the steps that failed or that the monitor wanted re-run, with the reasons, and picks those worth running again (a
    temporary failure, or a missing input that another re-run step will now provide). Their outputs are removed and
    they run once more as a new graph. Decisions go to results/replan_log.csv."""
    audio_dir = state["audio_dir"]
    failed = failed_steps(audio_dir, result, tasks)
    if not failed:
        return {}
    listing = "\n".join(f"Step {t} ({TASK_NAMES.get(t)}): {why}" for t, why in sorted(failed.items()))
    prompt = f"""You planned a speech-dataset quality-control run. These steps failed or were flagged by the reviewer:
{listing}
    Each step can be run once more. Choose the steps worth re-running: a failure that looks temporary (a timeout, a
    model or memory error, an incomplete or malformed output) or a step whose input was missing because another step
    you re-run failed. Do not re-run a step whose output is a valid finding about the data (files that really are
    corrupt, upsampled or in another language, a step skipped for lack of references).
    Reply in exactly this format:
    Rerun: <comma-separated step numbers, or none>
    Reason: <one sentence>"""
    try:
        reply = llm.model_copy(update={"num_predict": 300}).invoke(prompt).content
    except Exception as e:
        reply = f"Rerun: none\nReason: planner failed ({e})"
    m = re.search(r"Rerun:\s*(.*)", reply)
    chosen = {int(x) for x in re.findall(r"\d+", m.group(1) if m else "")} & set(failed)
    reason = (re.search(r"Reason:\s*(.*)", reply) or re.search(r"(.*)", reply)).group(1).strip()
    rows, out = [], {}
    if chosen:
        logging.info(f"re-planning: re-running {sorted(chosen)}: {reason}")
        for t in chosen:   # these steps reuse their output when it exists (a failed run may have left one): remove it
            paths = re.findall(r"(/[^\s'\"]+\.csv)", str(result.get(node_map[t][2], "")))
            paths += [os.path.join(results_dir(audio_dir), f) for f in REUSED_OUTPUTS.get(t, [])]
            for path in paths:
                if os.path.isfile(path) and os.path.dirname(path) == results_dir(audio_dir):
                    os.remove(path)
        out = build_graph_from_structure(layer_tasks(chosen), {str(t) for t in chosen}).invoke(dict(state))
        out = {k: v for k, v in out.items() if k not in state}
    after = failed_steps(audio_dir, {**result, **out}, chosen) if chosen else {}
    for t, why in sorted(failed.items()):
        rows.append({"Task": t, "Step": TASK_NAMES.get(t), "Problem": why, "Rerun": t in chosen, "Planner_Reason": reason,
                     "After_Rerun": ("still failing: " + after[t]) if t in after else ("OK" if t in chosen else None)})
    path = os.path.join(results_dir(audio_dir), "replan_log.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return out


if __name__ == "__main__":
    
    user_prompt = PROMPT
    # "Perform QC Checks, Sample rate check, VAD silence detection, upsampling, IndicLID language identification,calculate speaker duration in the audio files located at '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/'."
    # user_prompt = "I want to check the number of speakers, calculate the duration for each speaker, verify if speakers are new or old, compute WER, count English words, check for utterance duplicates, and clean the transcriptions by removing HTML tags in the audio files located at '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/'."
    # user_prompt= "verify language as Hindi (' hi '),CTC SCORE,   WER computation , normalization , calculate LLM score , check domain ,english word counter, and check for utterance duplicate  in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/' "
    # user_prompt= "Perform QC Checks, Sample rate check, VAD silence detection, upsampling, IndicLID language identification, calculate speaker duration, verify if speakers are new or old in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/' "
    # user_prompt = "I need to analyze my Hindi-English conversational dataset to assess the quality of dialogue flow, contextual consistency across speaker turns, and conversational coherence for training our dialogue state tracking model. The data contains multi-turn conversations where speakers ask questions and provide answers in alternating fashion. audios present in directory: '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/' "
    
    main(user_prompt)
    
