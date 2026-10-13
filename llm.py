
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
    align_model_for,
    pseudo_perplexity,
    get_required_inputs
)
import librosa
import pandas as pd
from config import (OLLAMA_BASE_URL, PLANNER_MODEL, CODER_MODEL, LLM_NUM_CTX, LANG_CODES, DEFAULT_LANGUAGE, SCRIPTS,
                    FLUENCY_MAX_PPL, FLUENCY_MIN_ALIGN)
logging.basicConfig(filename="pipeline.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Local LLMs through Ollama (originally gpt-4.1-mini via the OpenAI API)
def local_llm(model):
    # thinking models (qwen3, deepseek-r1) would put reasoning text, full of numbers, into the answers we parse
    thinking = {"reasoning": False} if any(k in model for k in ("qwen3", "deepseek-r1")) else {}
    return ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=0, num_ctx=LLM_NUM_CTX, **thinking)


llm = local_llm(PLANNER_MODEL)
coder_llm = local_llm(CODER_MODEL)


def results_dir(audio_dir: str) -> str:
    d = os.path.join(os.path.dirname(audio_dir.rstrip("/\\")), "results")
    os.makedirs(d, exist_ok=True)
    return d


CODER_SYSTEM = ("You write complete, self-contained Python 3 scripts. Reply with exactly one ```python code block "
                "containing the whole script; it will be executed for you and you will see its output or error. "
                "Available libraries: os, csv, json, re, pandas, numpy, soundfile, librosa.")


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


def csv_rows(path: str, skip_error: bool = False) -> Optional[int]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if skip_error:   # num_speakers.csv rows whose diarization failed
        df = df[df.iloc[:, 1].astype(str) != "Error"]
    return len(df)


def run_agent(task_prompt: str, output_path: str, expected_rows: Optional[int] = None, attempts: int = 3) -> bool:
    """Code-generating agent: the LLM writes a script for the task, we execute it, and on an error or a
    missing/empty output CSV the error goes back to the LLM for a corrected script."""
    if os.path.exists(output_path):
        os.remove(output_path)
    prompt = task_prompt + describe_inputs(task_prompt, output_path)
    if os.sep == "\\":   # Windows: C:\Users\... pasted into a Python string literal is a syntax error ("\U")
        prompt = prompt.replace("\\", "/")
    messages = [("system", CODER_SYSTEM), ("user", prompt)]
    with AGENT_LOCK:
        for attempt in range(1, attempts + 1):
            reply = coder_llm.invoke(messages).content
            code = fix_paths(extract_code(reply), prompt)
            result = run_code(code) if code.strip() else "No code found in the reply."
            logging.info(f"agent attempt {attempt} for {os.path.basename(output_path)}:\n{code}\n--- output:\n{result[:2000]}")
            rows = csv_rows(output_path)   # None: missing, empty or unreadable
            if rows and (expected_rows is None or rows == expected_rows):
                return True
            if "\nError:\n" in result:
                problem = "The script failed."
            elif rows and expected_rows is not None:
                problem = f"The output CSV has {rows} rows but must have exactly {expected_rows} (one per input file/row)."
            else:
                problem = f"The output file {output_path} was not created or is empty."
            messages += [("assistant", reply),
                         ("user", f"Running your script printed:\n{result[:3000]}\n{problem} "
                                  f"Fix it and reply with the full corrected script.")]
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

def words(text: str) -> list:
    """Words of a transcript without [tags]/<tags> and punctuation; keeps Indic vowel signs (unicode category M)
    and apostrophes/hyphens inside a word (don't, check-up)."""
    import unicodedata
    t = re.sub(r"\[[^\]]*\]|<[^>]*>", " ", str(text))
    inword = lambda ch: unicodedata.category(ch)[0] in "LMN"
    return "".join(ch if inword(ch) or (ch in "'’-" and 0 < i < len(t) - 1 and inword(t[i - 1]) and inword(t[i + 1]))
                   else " " if unicodedata.category(ch)[0] in "PSZ" or unicodedata.category(ch) == "Cc" else ch
                   for i, ch in enumerate(t)).split()

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
    - Only include tasks explicitly relevant to the prompt.
    Return ONLY the task numbers as a comma-separated string (e.g., '4,5,6'), with no other text.
    Prompt: {user_prompt}
    """
    resp_1 = llm.invoke(prompt_1).content
    task_numbers = list(dict.fromkeys(n for n in re.findall(r'\b\d+\b', resp_1) if 1 <= int(n) <= 24))
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
                current_tasks = {t for t in current_tasks if t and t.isdigit() and 1 <= int(t) <= 24}
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
    prompt_2 = f"""You are given a list of 24 tasks related to ASR, audio, and transcription processing:
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
        ## INPUT
        - You will receive a subset of the above task numbers (e.g., {resp_1}). Only consider these tasks for execution.
        ## TASK
        You must perform dependency-aware topological sorting of the requested tasks so that all dependencies are respected and tasks can be grouped for maximum concurrency where possible.
        ### General Instructions
        - Work only with the task numbers included in {resp_1}. Exclude any other tasks.
        - Group independent tasks together so they can execute concurrently (in parallel).
        - Arrange dependent tasks such that dependencies always run first and their dependents run strictly after.
        - Very strict dependency: Tasks 9, 23, and 24 depend on task 16. Task 16 must be completed before any of these are started.
        - Strict dependency: Tasks (3,4,5,6,10,13,14,15,16) all depend on task 1. Task 12 depends on task 2. Task 22 depends on task 2. Task 14 also depends on ground truth being available.
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
        if not run_agent(task_prompt, output_path, len(list_files(audio_dir))):
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
        if not run_agent(task_prompt, output_path, len(list_files(audio_dir))):
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
    1. Check each audio file's sample rate with soundfile.info(path).samplerate
    2. Create a CSV with columns: Filename, Sample_Rate, Status
    3. Store "Pass" in Status if sample rate is 16000 Hz, otherwise "Fail"
    4. Save the CSV at exactly this path: {output_path}
    """
    try:
        if not run_agent(task_prompt, output_path, len(list_files(audio_dir, AUDIO_EXT))):
            return {"sample_rate_output": f"Error: Failed to generate {output_path}"}
        df = pd.read_csv(output_path)
        ok = (df["Status"].astype(str) == "Pass").all()
        return {"sample_rate_output": ("Success" if ok else "Invalid") + f" - CSV saved at: {output_path}"}
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
        if run_agent(task_prompt, output_path, csv_rows(csv_path)):
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
        if run_agent(task_prompt, output_path, len(list_files(audio_dir, AUDIO_EXT))):
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
        rows = []
        for f, ids in speakers.items():
            others = [g for g, other in speakers.items() if g != f and ids & other]
            rows.append({"Filename": f, "Speaker_Status": "Old" if others else "New", "Common_File": ";".join(others)})
        pd.DataFrame(rows, columns=["Filename", "Speaker_Status", "Common_File"]).to_csv(output_path, index=False)
        return {"valid_speaker_output": f"CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"Valid speaker check failed: {e}")
        return {"valid_speaker_output": f"Error: {e}"}

def domain_checker_agent(state: CombinedStateDict) -> CombinedStateDict:
    audio_dir = state.get('audio_dir')
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for domain checker: {audio_dir}")
        return {"domain_checker_output": "Error: Invalid audio directory"}
    parent = os.path.dirname(audio_dir.rstrip("/\\"))
    csv_path = os.path.join(parent, 'results' ,"indicconf_hypothesis.csv")
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return {"domain_checker_output": f"Error: CSV file not found at {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        if 'Indiconformer_Hypothesis' not in df.columns:
            logging.error("Column 'Indiconformer_Hypothesis' not found in CSV.")
            return {"domain_checker_output": "Error: 'Indiconformer_Hypothesis' column missing."}
        original_domains = ['Conversation', 'Keywords Spotting', 'Product Review','GK Questions', 'District Specific', 'KYP - Games',
       'Task of Fives', 'DOI - Religion', 'KYP - Cooking', 'Daily Life', 'Healthcare', 'Insurance', 'Finance', 'Sales']
        domains = []
        for i, transcript in enumerate(df['Indiconformer_Hypothesis']):
            prompt = f"""You are a {language_of(state)} language expert. Analyze the following normalized {language_of(state)} transcript and determine the general domain of the speech dataset. some original domains to choose from are {original_domains}.
            ## OUTPUT FORMAT: Return the domain from original domains as a one word phrase suitable from original domains. no additional terms
            Now classify this transcript:
            Transcript: {transcript}
            Domain: 
    """
            try:
                # short bounded reply, accepted only if it names a listed domain: on long texts the model
                # sometimes writes an essay instead of a label (seen on a Bengali reference transcript)
                reply = llm.model_copy(update={"num_predict": 30}).invoke(prompt).content
                reply = re.sub(r"^\W*domain\W*", "", reply.strip(), flags=re.IGNORECASE).lower()
                found = sorted((d for d in original_domains if d.lower() in reply), key=lambda d: reply.index(d.lower()))
                domains.append(found[0] if found else "Unknown")

            except Exception as e:
                logging.warning(f"LLM failed on row {i}: {e}")
                domains.append("Unknown")

        df['domain'] = domains
        output_path = os.path.join(parent, 'results', "domain_check.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
        if run_agent(task_prompt, output_path, csv_rows(csv_path)):
            return {"audio_transcript_matching_output": f"CSV saved at: {output_path}"}
        logging.error(f"Transcript matching failed to generate {output_path}")
        return {"audio_transcript_matching_output": f"Error: Failed to generate {output_path}"}
    except Exception as e:
        logging.error(f"Transcript matching failed: {e}")
        return {"audio_transcript_matching_output": f"Error: {e}"}


def language_identification_indiclid_agent(state: CombinedStateDict) -> CombinedStateDict:
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
        
        results = []
        for idx, row in df.iterrows():
            transcription = str(row['Indiconformer_Hypothesis'])
            filename = row.get('Filename', f"unknown_{idx}")
            
            if pd.isna(transcription) or transcription.strip() == "":
                logging.warning(f"Empty transcription for {filename}")
                results.append({
                    "Filename": filename,
                    "Transcription": transcription,
                    "Language_Code": "Unknown",
                    "Confidence": 0.0,
                    "Model_Used": "IndicLID"
                })
                continue
            
            try:
                lid_results = language_identification_indiclid(transcription)
                for _, lang_code, confidence, model_used in lid_results:
                    results.append({
                        "Filename": filename,
                        "Transcription": transcription,
                        "Language_Code": lang_code,
                        "Confidence": confidence,
                        "Model_Used": model_used
                    })
            except Exception as e:
                logging.error(f"Language identification failed for {filename}: {e}")
                results.append({
                    "Filename": filename,
                    "Transcription": transcription,
                    "Language_Code": "Error",
                    "Confidence": 0.0,
                    "Model_Used": "IndicLID"
                })
        
        if not results:
            logging.error("No language identification results generated")
            return {"language_identification_indiclid_output": "Error: No language identification results"}
        
        output_df = pd.DataFrame(results)
        output_df.to_csv(output_path, index=False)
        logging.info(f"Language identification results saved to {output_path}")
        return {"language_identification_indiclid_output": output_path}
        
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
    -Collapse repeated whitespace.
    -The output CSV should preserve the original columns and include the cleaned transcript in the normalized_transcripts column.
    -Save the updated CSV at exactly this path: {output_path}
    """
    try:
        if run_agent(task_prompt, output_path, csv_rows(csv_path)):
            return {"normalization_remove_tags_output": f"CSV saved at: {output_path}"}
        error_msg = f"Failed to generate {output_path}"
        logging.error(error_msg)
        return {"normalization_remove_tags_output": error_msg}
    except Exception as e:
        logging.error(f"Normalization failed: {e}")
        return {"normalization_remove_tags_output": f"Error: {e}"}

def llm_score_agent(state: CombinedStateDict) -> CombinedStateDict:
    """LLM-as-a-judge, gated by measured signals. Every file gets three numbers that cannot be hallucinated:
    transcript quality (task 3), CTC alignment of the ASR transcript (task 14, else computed here) and the language
    model's pseudo-perplexity. Only transcripts that pass all three are scored 0-10 by the LLM; on the others the
    judge cannot tell the transcript from shuffled words (tested), so LLM_Score stays empty with the reason."""
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
        def column(name, key, value):
            p = os.path.join(res, name)
            if not os.path.isfile(p):
                return {}
            d = pd.read_csv(p)
            return dict(zip(d[key].astype(str), d[value])) if {key, value} <= set(d.columns) else {}
        quality = column("transcript_quality.csv", fcol, "Quality_Check")
        align = column("audio_transcript_matching.csv", "Filename", "CTC_Score")
        rows = []
        for fname, text in zip(df[fcol].astype(str), df[tcol].fillna("").astype(str)):
            if fname not in align or pd.isna(align[fname]):   # task 14 missing or failed: align here
                try:
                    align[fname] = force_alignment_and_ctc_score(os.path.join(audio_dir, fname), text,
                                                                 align_model_for(language_of(state)))[1]
                except Exception as e:
                    logging.warning(f"alignment for the fluency gate failed on {fname}: {e}")
                    align[fname] = None
            ppl = pseudo_perplexity(text)
            reasons = []
            if not str(quality.get(fname, "passed")).startswith("passed"):
                reasons.append(f"transcript quality {quality[fname]}")
            if align[fname] is None or float(align[fname]) < FLUENCY_MIN_ALIGN:
                reasons.append(f"alignment {align[fname] if align[fname] is None else round(float(align[fname]), 2)} < {FLUENCY_MIN_ALIGN}")
            if ppl is None or ppl > FLUENCY_MAX_PPL:
                reasons.append(f"perplexity {ppl if ppl is None else round(ppl)} > {FLUENCY_MAX_PPL:g} (closer to shuffled words than to fluent text)")
            row = {"Filename": fname, "Transcription": text, "LM_Pseudo_Perplexity": None if ppl is None else round(ppl, 1),
                   "Alignment_Score": None if align[fname] is None else round(float(align[fname]), 3),
                   "Transcript_Quality": quality.get(fname), "Fluency_Gate": "passed" if not reasons else "failed: " + "; ".join(reasons),
                   "LLM_Score": None, "Evaluation_Comment": "not scored by the LLM: " + "; ".join(reasons) if reasons else None}
            if not reasons:
                prompt = f"""You are a strict judge of {language_of(state)} speech transcriptions.
    Evaluate the coherence and fluency of this transcription. Understand the meaning before scoring; do not give every transcription the same score.
    Score from 0 to 10: 10 for fluent, meaningful sentences, lower the more meaningless it gets, and 0 if it is in a language other than {language_of(state)}.
    Reply in exactly this format:
    Score: <0-10>
    Comment: <one sentence>
    Transcription: {text[:4000]}"""
                try:
                    out = llm.model_copy(update={"num_predict": 200}).invoke(prompt).content
                    m = re.search(r'Score:\s*(\d+(?:\.\d+)?)', out)
                    c = re.search(r'Comment:\s*(.*)', out, re.DOTALL)
                    row["LLM_Score"] = min(10.0, float(m.group(1))) if m else None
                    row["Evaluation_Comment"] = c.group(1).strip() if c else "Error: reply had no score"
                except Exception as e:
                    logging.warning(f"LLM score failed for {fname}: {e}")
                    row["Evaluation_Comment"] = f"Error: {e}"
            rows.append(row)
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
    audio_dir = state.get('audio_dir')
    parent = os.path.dirname(audio_dir.rstrip("/\\"))
    if not audio_dir or not os.path.isdir(audio_dir):
        logging.error(f"Invalid audio directory for duplicate checker: {audio_dir}")
        return {"utterance_duplicate_output": "Error: Invalid audio directory"}
    csv_path = os.path.join(parent, 'results',"normalized_list.csv")
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found: {csv_path}")
        return {"utterance_duplicate_output": f"Error: CSV file not found at {csv_path}"}
    try:
        df = pd.read_csv(csv_path)
        rows = []
        for column in df.columns:
            if df[column].dtype == 'object':
                duplicates = df[column][df[column].duplicated(keep=False)]
                if not duplicates.empty:
                    counts = duplicates.value_counts()
                    for utterance, count in counts.items():
                        rows.append({
                            "column_name": column,
                            "utterance": utterance,
                            "count": count
                        })
        if rows:
            dup_df = pd.DataFrame(rows)
            output_path = os.path.join(parent, 'results', "duplicate_utterances.csv")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            dup_df.to_csv(output_path, index=False)
            logging.info(f"Duplicate check completed. Output saved to {output_path}")
            return {"utterance_duplicate_output": f"CSV saved at: {output_path}"}
        else:
            return {"utterance_duplicate_output": "No duplicate utterances found."}
    except Exception as e:
        logging.error(f"Duplicate utterance checker failed: {e}")
        return {"utterance_duplicate_output": f"Error: {e}"}

import jiwer
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
        wer_rows = []
        for fname, hyp in zip(hyp_df["Filename"].astype(str), hyp_df["Indiconformer_Hypothesis"].fillna("").astype(str)):
            if fname not in refs:
                continue
            ref, hyp = " ".join(words(refs[fname])), " ".join(words(hyp))
            try:
                error = round(jiwer.wer(ref, hyp), 4) if ref else "No reference"
            except Exception as e:
                error = "Error"
                logging.warning(f"WER computation failed for {fname}: {e}")
            wer_rows.append({"Filename": fname, "Reference": ref, "Hypothesis": hyp, "WER": error})
        if not wer_rows:
            return {"wer_output": "Error: no transcribed file has a reference transcript"}
        wer_df = pd.DataFrame(wer_rows)
        output_path = os.path.join(parent, 'results', "wer.csv")
        wer_df.to_csv(output_path, index=False)
        logging.info(f"WER computation completed. CSV saved to {output_path}")
        return {"wer_output": f"WER CSV saved at: {output_path}"}
    except Exception as e:
        logging.error(f"WER computation failed: {e}")
        return {"wer_output": f"Error: {e}"}
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
    
}

# Explicit mentions always select their task (the same keyword rules the selection prompt states), so a
# small local model cannot silently drop a check the user asked for by name.
KEYWORD_TASKS = [(r"\bASR\b|transcri(be|ption)", 1), (r"number of speakers|diariz", 2), (r"transcript quality", 3),
                 (r"character|grapheme", 4), (r"\bvocab", 5), (r"verify (the )?language|language verification|expected language", 6),
                 (r"audio length", 7), (r"silence|\bVAD\b", 8), (r"english word", 9), (r"\bCTC\b", 10),
                 (r"upsampl", 11), (r"new or old|valid speaker", 12), (r"\bdomain", 13), (r"forced alignment", 14),
                 (r"IndicLID|language identification", 15), (r"normali[sz]|HTML", 16), (r"LLM score|LLM-as-a-judge|coherence", 17),
                 (r"transliterat", 18), (r"corrupt", 19), (r"extension|file format", 20), (r"sample rate", 21),
                 (r"speaker durations?|duration (for|of) each speaker|durations", 22), (r"duplicate", 23), (r"\bWER\b", 24)]


def keyword_tasks(prompt: str) -> set:
    return {task for pattern, task in KEYWORD_TASKS if re.search(pattern, prompt, re.IGNORECASE)}


# Hard dependencies between tasks (the same rules the prompts give the LLM), enforced in code so a
# small local model cannot schedule a task before its inputs exist.
REFERENCE_TASKS = {9, 16, 18, 23, 24}   # need results/indicconf_hypothesis-gt.csv
TASK_DEPS = {3: [1, 8], 17: [1, 3, 14], 4: [1], 5: [1], 6: [1], 10: [1], 13: [1], 14: [1], 15: [1], 16: [1],
             12: [2], 22: [2], 9: [16], 23: [16], 24: [16]}


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
                graph_builder.add_node(node_name, func)
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
    resp_1 = select_tasks(user_prompt)
    
    verified_tasks = prompt_checker_agent(user_prompt, resp_1)
    selected = {int(x) for x in resp_1.split(',') if x.strip().isdigit() and 1 <= int(x) <= 24}
    checked = {int(x) for x in verified_tasks.split(',') if x.strip().isdigit() and 1 <= int(x) <= 24}
    # the checker may only remove tasks: with qwen3:8b most tasks it added were never requested (planner benchmark:
    # 29% of planned actions unrequested vs 6% this way); missing tasks come back via keywords and dependencies
    if checked - selected:
        logging.info(f"Ignoring tasks added by the prompt checker: {sorted(checked - selected)}")
    tasks = selected & checked
    named = keyword_tasks(user_prompt) - tasks
    if named:
        logging.info(f"Added tasks named in the prompt but not selected by the LLM: {sorted(named)}")
        tasks |= named
    with_deps = add_dependencies(tasks)
    if with_deps != tasks:
        logging.info(f"Added dependency tasks: {sorted(with_deps - tasks)}")
    tasks = with_deps
    # English-word count, normalization, duplicates and WER read the reference transcripts: without them they can
    # only fail, so drop them even when the planner picked them
    reference = parsed_inputs["ground_truth_csv"] or (parsed_inputs["audio_dir"] and os.path.join(
        results_dir(parsed_inputs["audio_dir"]), "indicconf_hypothesis-gt.csv"))
    if not (reference and os.path.isfile(reference)) and tasks & REFERENCE_TASKS:
        logging.info(f"No reference transcripts: skipping tasks {sorted(tasks & REFERENCE_TASKS)}")
        print(f"No reference transcripts: skipping tasks {sorted(tasks & REFERENCE_TASKS)} (need indicconf_hypothesis-gt.csv)")
        tasks -= REFERENCE_TASKS
    verified_tasks = ','.join(map(str, sorted(tasks)))
    valid_tasks = {str(x) for x in tasks}
    structure = topological_sort_tasks(verified_tasks)
    if not structure_is_valid(structure, tasks):
        logging.info(f"LLM structure {structure} breaks a dependency or drops a task; using deterministic layering")
        structure = layer_tasks(tasks)
    print("Using structure:", structure)

    main_graph = build_graph_from_structure(structure, valid_tasks)
    main_graph.get_graph().print_ascii()

    initial_state = {
        "audio_dir": parsed_inputs["audio_dir"],
        "ground_truth_csv": parsed_inputs["ground_truth_csv"],
        "lang_code": parsed_inputs["lang_code"],
        "user_prompt": user_prompt
    }
    final_result = main_graph.invoke(initial_state)
    print("\nFinal Pipeline Results:")
    for key, value in final_result.items():
        if key not in ["audio_dir", "ground_truth_csv", "lang_code", "user_prompt"]:
            print(f"{key}: {value[:100]}..." if isinstance(value, str) and len(value) > 100 else f"{key}: {value}")

if __name__ == "__main__":
    
    user_prompt = PROMPT
    # "Perform QC Checks, Sample rate check, VAD silence detection, upsampling, IndicLID language identification,calculate speaker duration in the audio files located at '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/'."
    # user_prompt = "I want to check the number of speakers, calculate the duration for each speaker, verify if speakers are new or old, compute WER, count English words, check for utterance duplicates, and clean the transcriptions by removing HTML tags in the audio files located at '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/'."
    # user_prompt= "verify language as Hindi (' hi '),CTC SCORE,   WER computation , normalization , calculate LLM score , check domain ,english word counter, and check for utterance duplicate  in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/' "
    # user_prompt= "Perform QC Checks, Sample rate check, VAD silence detection, upsampling, IndicLID language identification, calculate speaker duration, verify if speakers are new or old in :'/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC1/audios/' "
    # user_prompt = "I need to analyze my Hindi-English conversational dataset to assess the quality of dialogue flow, contextual consistency across speaker turns, and conversational coherence for training our dialogue state tracking model. The data contains multi-turn conversations where speakers ask questions and provide answers in alternating fashion. audios present in directory: '/raid/ganesh/pdadiga/chriss/agent/AI_Agent_Final/time-task/QC2/audios/' "
    
    main(user_prompt)
    
