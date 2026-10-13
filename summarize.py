"""Combine the per-check CSVs of every language run into one table: one row per audio file.
    python summarize.py <runs folder>
"""
import sys
from pathlib import Path

import pandas as pd

# result file -> {column in that file (case-insensitive): column in the summary}
COLUMNS = {
    "audio_length.csv": {"Audio_length": "Duration_s"},
    "corruption_check.csv": {"Status": "Corruption"},
    "audio_format_check.csv": {"Valid_Extension": "Valid_Extension"},
    "sample_rate_check.csv": {"Sample_Rate": "Sample_Rate", "Status": "Sample_Rate_Status"},
    "upsampling_check.csv": {"Is_Upsampled": "Upsampled"},
    "vad_silence_stats.csv": {"Total Silence (s)": "Silence_s"},
    "num_speakers.csv": {"Number of Speakers": "Speakers", "Speaker Durations": "Speaker_Durations_h"},
    "valid_speaker.csv": {"Speaker_Status": "Speaker_New_Old", "Common_File": "Shares_Speaker_With"},
    "indiclid_language_identification.csv": {"Language_Code": "Detected_Language", "Confidence": "LID_Confidence"},
    "language_verification.csv": {"*_Share": "Script_Share", "Is_*": "Script_OK"},
    "wer.csv": {"WER": "WER"},
    "transcript_quality.csv": {"Quality_Check": "ASR_Transcript_Quality"},
    "ctc_scores.csv": {"CTC_Score": "CTC_Score", "CTC_Status": "CTC_Status", "Transcript_Source": "CTC_Scored_Text"},
    "audio_transcript_matching.csv": {"CTC_Score": "ASR_Alignment_Score"},
    "llm_scores.csv": {"LM_Pseudo_Perplexity": "LM_Perplexity", "Fluency_Gate": "Fluency_Gate", "LLM_Score": "LLM_Score"},
    "domain_check.csv": {"domain": "Domain"},
    "english_word_count.csv": {"english_word_count": "English_Words_in_Reference"},
    "indicconf_hypothesis-gt_transliterated.csv": {"roman_words_left": "Roman_Words_Left_After_Transliteration"},
}


def _match(columns, pattern):
    for c in columns:
        lc, lp = c.lower(), pattern.lower()
        if lc == lp or (lp.startswith("*") and lc.endswith(lp[1:])) or (lp.endswith("*") and lc.startswith(lp[:-1])):
            return c
    return None


def _filename_col(columns):
    return next((c for c in columns if c.lower().replace(" ", "").replace("_", "") == "filename"), None)


def summarize_language(results: Path, language: str) -> pd.DataFrame:
    audio = sorted(p.name for p in (results.parent / "audios").iterdir() if p.is_file())
    table = pd.DataFrame({"Language": language, "Filename": audio})
    for name, wanted in COLUMNS.items():
        path = results / name
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        fcol = _filename_col(df.columns)
        if fcol is None:
            continue
        picked = {}
        for pattern, out in wanted.items():
            col = _match([c for c in df.columns if c != fcol], pattern)
            if col is not None:
                picked[col] = out
        if picked:
            part = df[[fcol, *picked]].rename(columns={fcol: "Filename", **picked})
            part["Filename"] = part["Filename"].astype(str).map(lambda f: Path(f).name)
            table = table.merge(part.drop_duplicates("Filename"), on="Filename", how="left")
    return table


def write_summary(out: Path, runs: dict) -> Path:
    tables = [summarize_language(run / "results", lang.title()) for lang, run in runs.items()]
    summary = pd.concat(tables, ignore_index=True)
    path = Path(out) / "qc_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")   # utf-8-sig: Excel shows Indic text correctly
    return path


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    runs = {p.name: p for p in sorted(root.iterdir()) if (p / "audios").is_dir()}
    print(write_summary(root, runs))
