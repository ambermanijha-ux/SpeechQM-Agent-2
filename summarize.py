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
    "sample_rate_check.csv": {"Sample_Rate": "Sample_Rate", "Channels": "Channels"},
    "upsampling_check.csv": {"Is_Upsampled": "Upsampled", "Bandwidth_Hz": "Bandwidth_Hz",
                             "Estimated_Original_Rate": "Estimated_Original_Rate"},
    "vad_silence_stats.csv": {"Total Silence (s)": "Silence_s"},
    "num_speakers.csv": {"Number of Speakers": "Speakers", "Speaker Durations": "Speaker_Durations_h",
                         "Speech (s)": "Speech_s", "Overlap (s)": "Overlapping_Speech_s"},
    "valid_speaker.csv": {"Speaker_Status": "Speaker_New_Old", "Common_File": "Shares_Speaker_With",
                          "Earlier_Batch_Files": "Speaker_In_Earlier_Batches"},
    "indiclid_language_identification.csv": {"Language_Code": "Detected_Language", "Confidence": "LID_Confidence",
                                             "Audio_Language": "Audio_Language", "Audio_Confidence": "Audio_LID_Confidence",
                                             "Audio_Matches_Expected": "Audio_Language_OK"},
    "language_verification.csv": {"*_Share": "Script_Share", "Is_*": "Script_OK"},
    "wer.csv": {"WER": "WER", "CER": "CER", "WER_bodhan": "WER_Bodhan", "WER_indicconformer": "WER_IndicConformer",
                "WER_whisper": "WER_Whisper", "WER_nemotron": "WER_Nemotron", "WER_wav2vec2": "WER_wav2vec2",
                "Segment_WER": "Segment_WER",
                "Segment_WER_No_Cross_Talk": "Segment_WER_No_Cross_Talk", "Bad_Segment_Share": "Bad_Segment_Share"},
    "asr_moe.csv": {"Experts": "ASR_Experts", "MoE_WER": "MoE_WER", "MoE_CER": "MoE_CER",
                    "Main_Agreement_CER": "Main_Agreement_CER", "Agreeing_Pair": "Agreeing_ASR_Pair",
                    "Agreeing_Pair_CER": "Agreeing_Pair_CER", "Least_Agreeing": "Least_Agreeing_ASR",
                    "MoE_Verdict": "MoE_Verdict"},
    "transcript_quality.csv": {"Quality_Check": "ASR_Transcript_Quality"},
    "ctc_scores.csv": {"CTC_Score": "CTC_Score", "CTC_Status": "CTC_Status", "Transcript_Source": "CTC_Scored_Text"},
    "audio_transcript_matching.csv": {"CTC_Score": "ASR_Alignment_Score"},
    "llm_scores.csv": {"LLM_Score": "LLM_Score", "LLM_Chunk_Score": "LLM_Chunk_Score",
                       "Low_Score_Chunk_Share": "Low_Score_Chunk_Share",
                       "Empty_Chunk_Share": "Empty_ASR_Chunk_Share", "Evaluation_Comment": "LLM_Comment"},
    "domain_check.csv": {"domain": "Domain", "Domain_From": "Domain_From"},
    "duplicates_per_file.csv": {"Near_Duplicate_Utterances": "Near_Duplicate_Utterances",
                                "Repeated_Utterances_In_File": "Repeated_Utterances_In_File",
                                "Shares_Audio_With": "Shares_Audio_With", "Public_Corpus_Matches": "Public_Corpus_Matches"},
    "english_word_count.csv": {"english_word_count": "English_Words_in_Reference"},
    "indicconf_hypothesis-gt_transliterated.csv": {"roman_words_left": "Roman_Words_Left_After_Transliteration"},
    "segmentation_summary.csv": {"Segments": "Ref_Segments", "Segments_With_Issues": "Segments_With_Timing_Issues",
                                 "Median_Segment_Alignment": "Segment_Alignment_Median"},
    "code_mixing_summary.csv": {"Code_Mixed_Share": "Code_Mixed_Utterance_Share", "Mean_CMI": "Mean_CMI",
                                "Embedded_Word_Share": "Embedded_Word_Share", "Other_Scripts": "Other_Scripts_In_Reference"},
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
    import dashboard
    dashboard.write(Path(out), runs, summary)
    return path


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    runs = {p.name: p for p in sorted(root.iterdir()) if (p / "audios").is_dir()}
    print(write_summary(root, runs))
