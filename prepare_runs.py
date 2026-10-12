"""Turn a nested dataset (<domain>/<language>/<audio + transcript json>) into one run folder per language,
the flat layout the pipeline reads:

    <out>/<language>/audios/<audio files>                         (copies; the originals are never touched)
    <out>/<language>/results/indicconf_hypothesis-gt.csv          (filename,ground_truth from the transcript JSONs)

A transcript JSON ({"segments": [{"text", "start_time", ...}]}) belongs to the audio whose name it contains,
e.g. "Copy of hi001_human_transcribed.json" -> hi001.wav.
Audio not under a folder named after a language goes to the default language (if one is given).
Usage: python prepare_runs.py <dataset folder> <output folder> [default language]
"""
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from config import ASR_MODELS

AUDIO_EXT = {".wav", ".mp3", ".flac"}
# folder spellings in the dataset -> language name used by config.ASR_MODELS
ALIASES = {"gujrati": "gujarati", "malyalam": "malayalam", "oriya": "odia", "bangla": "bengali",
           "panjabi": "punjabi"}
LANGUAGES = {name.lower() for name in ASR_MODELS}


def language(path: Path, root: Path, default=None) -> str:
    for part in reversed(path.relative_to(root).parts[:-1]):
        name = ALIASES.get(part.lower(), part.lower())
        if name in LANGUAGES:
            return name
    if default:
        return default.lower()
    raise ValueError(f"{path} is not inside a folder named after a language ({', '.join(sorted(LANGUAGES))}); "
                     "pass the language of such files explicitly")


def transcript_text(json_path: Path) -> str:
    segments = json.loads(json_path.read_text(encoding="utf-8"))["segments"]
    return " ".join(" ".join(s["text"].split()) for s in sorted(segments, key=lambda s: s["start_time"])
                    if s["text"].strip())


def main(src: Path, out: Path, default_language=None) -> dict:
    """Returns {language: run folder}."""
    if default_language and default_language.lower() not in LANGUAGES:
        raise ValueError(f"unknown language {default_language}; supported: {', '.join(sorted(LANGUAGES))}")
    audio = defaultdict(list)
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT and out not in p.parents:
            audio[language(p, src, default_language)].append(p)
    if not audio:
        raise ValueError(f"no .wav/.mp3/.flac files under {src}")
    jsons = [p for p in src.rglob("*.json") if p.is_file()]
    for lang, files in sorted(audio.items()):
        dest = out / lang / "audios"
        dest.mkdir(parents=True, exist_ok=True)
        rows, names = [], set()
        for f in files:
            if f.name in names:
                raise ValueError(f"two {lang} files are named {f.name}; rename one first")
            names.add(f.name)
            if not (dest / f.name).exists():
                shutil.copy2(f, dest / f.name)
            match = [j for j in jsons if j.parent == f.parent and f.stem in j.stem]
            if len(match) == 1:
                try:
                    rows.append((f.name, transcript_text(match[0])))
                except (ValueError, KeyError, TypeError) as e:
                    print(f"  skipped transcript {match[0].name}: not the expected segments JSON ({e})")
        res = out / lang / "results"
        res.mkdir(exist_ok=True)
        if rows:
            with open(res / "indicconf_hypothesis-gt.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["filename", "ground_truth"])
                w.writerows(rows)
        print(f"{lang:10} {len(files):3} audio, {len(rows):3} with transcript -> {out / lang}")
    return {lang: out / lang for lang in sorted(audio)}


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3] if len(sys.argv) > 3 else None)
