"""Index public speech corpora once, for the duplicate check (task 23): audio fingerprints and transcripts.

    python build_corpus_index.py <source> [<source> ...] [--name NAME] [--out models/public_corpus_index]

A source is a folder of audio files (.wav/.mp3/.flac; a .txt next to an audio file with the same name is its
transcript) or a manifest CSV with an audio path column (audio_path / path / audio / file) and optionally a text column
(text / text_raw / transcript / sentence). Relative audio paths in a manifest are relative to the CSV's folder.
Sources given again replace their earlier entries; the others in the index are kept."""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PUBLIC_INDEX  # noqa: E402
import duplicates  # noqa: E402

AUDIO_EXT = (".wav", ".mp3", ".flac")


def items_of(source):
    """[(audio path, item name, text or None)]"""
    src = Path(source).expanduser().resolve()
    if src.is_dir():
        out = []
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() in AUDIO_EXT:
                txt = p.with_suffix(".txt")
                out.append((str(p), str(p.relative_to(src)), txt.read_text(encoding="utf-8").strip() if txt.is_file() else None))
        return out
    df = pd.read_csv(src)
    cols = {c.lower(): c for c in df.columns}
    pcol = next((cols[c] for c in ("audio_path", "path", "audio", "file", "filename") if c in cols), None)
    tcol = next((cols[c] for c in ("text", "text_raw", "transcript", "sentence", "ground_truth") if c in cols), None)
    if pcol is None:
        sys.exit(f"{src}: no audio path column (audio_path / path / audio / file)")
    return [(str((src.parent / p).resolve()) if not os.path.isabs(p) else p, str(p),
             None if tcol is None or pd.isna(t) else str(t)) for p, t in zip(df[pcol].astype(str), df[tcol] if tcol else [None] * len(df))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--name", help="corpus name shown in the results (default: the source's name)")
    ap.add_argument("--out", default=str(PUBLIC_INDEX))
    args = ap.parse_args()
    from utility_functions import load_audio

    old = duplicates.load_public_index(args.out)
    index, meta = duplicates.FingerprintIndex(), []
    names = {args.name or Path(s).stem for s in args.sources}
    if old:   # keep the other corpora (re-fingerprinting them is not needed: copy their hashes)
        o = old["audio"].freeze()
        for rid, m in enumerate(old["meta"]):
            if m["corpus"] not in names:
                sel = o.rids == rid
                index.add(o.names[rid], o.keys[sel], o.frames[sel], o.durations[rid])
                meta.append(m)
    t = time.time()
    for source in args.sources:
        corpus = args.name or Path(source).stem
        items = items_of(source)
        print(f"{corpus}: {len(items)} recordings", flush=True)
        for i, (path, item, text) in enumerate(items):
            try:
                audio = load_audio(path)
            except Exception as e:
                print(f"  skipped {path}: {e}")
                continue
            h, fr = duplicates.landmarks(audio, 16000)
            index.add(path, h, fr, len(audio) / 16000)
            meta.append({"corpus": corpus, "item": item, "text": text})
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(items)} ({time.time() - t:.0f} s)", flush=True)
    duplicates.save_public_index(index.freeze(), meta, args.out)
    print(f"Index: {len(meta)} recordings, {len(index.keys)} landmarks, {sum(m['text'] is not None for m in meta)} "
          f"transcripts -> {args.out}")


if __name__ == "__main__":
    main()
