"""Near-duplicate detection for task 23, for audio and text, within a batch and against public corpora.

Audio: landmark fingerprints (pairs of spectral peaks, as in Wang 2003 / Shazam), computed at 8 kHz so that telephone
copies, re-encodings (mp3), gain changes and added noise of the same recording still match, as does a clip that was
cut out of a longer file. Two recordings match when many landmark hashes agree at one consistent time offset.
Text: TF-IDF of character 3-grams per utterance, cosine similarity (catches the same sentence with small edits).
Public corpora: build_corpus_index.py indexes folders of audio or a manifest CSV (audio path + text) once; the index
is read from config.PUBLIC_INDEX."""
import hashlib
import logging
import os
import pickle

import librosa
import numpy as np
from scipy.ndimage import maximum_filter

from config import PUBLIC_INDEX

FP_SR = 8000
N_FFT, HOP = 512, 128            # 64 ms windows, 16 ms hop
PEAKS_PER_S = 30
FAN_OUT, MAX_DT = 5, 63          # each peak pairs with the next 5 peaks up to 63 frames (1 s) later
MAX_REPEATS = 3                  # a hash seen more often in one recording comes from a stationary sound
MIN_MATCHES = 40                 # landmark hashes agreeing at one offset for two recordings to match
MIN_MATCH_SHARE = 0.05           # ... and that share of the shorter recording's landmarks over the matched stretch
TEXT_SIMILARITY = 0.9            # cosine similarity of character 3-gram TF-IDF for a text near-duplicate
MIN_TEXT_WORDS = 4               # shorter utterances ("yes", "ok sir") repeat naturally


def file_md5(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def landmarks(audio, sr):
    """(hash, frame) pairs of one recording; frames are HOP / FP_SR = 16 ms apart."""
    y = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=FP_SR) if sr != FP_SR else audio
    if len(y) < N_FFT:
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    spec = np.log(np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) + 1e-6)
    peaks = (spec == maximum_filter(spec, size=(15, 9))) & (spec > np.median(spec) + 2.0)
    f, t = np.nonzero(peaks)
    if not len(t):
        return np.zeros(0, np.int64), np.zeros(0, np.int32)
    # keep the strongest peaks, about PEAKS_PER_S per second
    keep = np.argsort(-spec[f, t])[: int(PEAKS_PER_S * len(y) / FP_SR) + 1]
    order = np.lexsort((f[keep], t[keep]))
    f, t = f[keep][order], t[keep][order]
    hashes, frames = [], []
    for i in range(len(t)):
        n = 0
        for j in range(i + 1, len(t)):
            dt = t[j] - t[i]
            if dt > MAX_DT:
                break
            if dt == 0:
                continue
            hashes.append((int(f[i]) << 15) | (int(f[j]) << 6) | int(dt))
            frames.append(t[i])
            n += 1
            if n == FAN_OUT:
                break
    hashes, frames = np.asarray(hashes, np.int64), np.asarray(frames, np.int32)
    # a stationary sound (tone, hum, beep, music bed) repeats the same hash every frame and would match any recording
    # that has it: keep only hashes that occur at most MAX_REPEATS times in the recording
    uniq, inv, cnt = np.unique(hashes, return_inverse=True, return_counts=True)
    keep = cnt[inv] <= MAX_REPEATS
    return hashes[keep], frames[keep]


class FingerprintIndex:
    """Landmark hashes of many recordings, as sorted arrays: hash, recording id, frame."""

    def __init__(self):
        self.names, self.durations = [], []
        self._parts = []
        self.keys = self.rids = self.frames = None

    def add(self, name, hashes, frames, duration):
        rid = len(self.names)
        self.names.append(name)
        self.durations.append(duration)
        self._parts.append((hashes, np.full(len(hashes), rid, np.int32), frames))
        self.keys = None
        return rid

    def freeze(self):
        if self.keys is None:
            parts = self._parts or [(np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int32))]
            k, r, f = (np.concatenate(x) for x in zip(*parts))
            order = np.argsort(k, kind="stable")
            self.keys, self.rids, self.frames = k[order], r[order], f[order]
        return self

    def __getstate__(self):   # pickled frozen, without the parts
        self.freeze()
        return {**self.__dict__, "_parts": []}

    def query(self, hashes, frames, exclude=()):
        """[(recording id, matched hashes, offset in s, matched stretch in s, share of the query's landmarks there)]"""
        self.freeze()
        lo, hi = np.searchsorted(self.keys, hashes, "left"), np.searchsorted(self.keys, hashes, "right")
        n = hi - lo
        if not n.sum():
            return []
        q = np.repeat(np.arange(len(hashes)), n)                     # query landmark of every posting
        pos = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
        rid, fq = self.rids[pos].astype(np.int64), frames[q].astype(np.int64)
        off = self.frames[pos].astype(np.int64) - fq
        if exclude:
            keep = ~np.isin(rid, list(exclude))
            rid, fq, off, q = rid[keep], fq[keep], off[keep], q[keep]
        key = rid * (1 << 32) + (off + (1 << 31))
        uk, cnt = np.unique(np.unique(np.stack([key, q]), axis=1)[0], return_counts=True)   # distinct query landmarks
        # neighbouring offsets (+-1 frame) count together: frame jitter after re-encoding
        near = cnt.copy()
        for d in (-1, 1):
            i = np.searchsorted(uk, uk + d)
            ok = (i < len(uk)) & (uk[np.minimum(i, len(uk) - 1)] == uk + d)
            near[ok] += cnt[i[ok]]
        out = []
        r_of = uk >> 32
        for r in np.unique(r_of[near >= MIN_MATCHES]):
            sel = np.nonzero(r_of == r)[0]
            b = sel[np.argmax(near[sel])]
            best = uk[b]
            m = np.abs(key - best) <= 1
            a, z = int(fq[m].min()), int(fq[m].max())
            in_span = int(((frames >= a) & (frames <= z)).sum()) or 1
            share = near[b] / in_span
            if share >= MIN_MATCH_SHARE:
                o = int(best & ((1 << 32) - 1)) - (1 << 31)
                out.append((int(r), int(near[b]), round(o * HOP / FP_SR, 2), round((z - a) * HOP / FP_SR, 1),
                            round(float(min(1.0, share)), 3)))
        return sorted(out, key=lambda x: -x[1])


def audio_near_duplicates(paths, load):
    """Pairs of recordings in `paths` that share audio (same recording, re-encoded, trimmed or embedded).
    load(path) -> 16 kHz mono float array. Returns rows for a CSV and the fingerprints (for the public check)."""
    index, prints, rows = FingerprintIndex(), {}, []
    md5 = {p: file_md5(p) for p in paths}
    for p in paths:
        audio = load(p)
        h, fr = landmarks(audio, 16000)
        prints[p] = (h, fr, len(audio) / 16000)
        index.add(p, h, fr, len(audio) / 16000)
    for i, p in enumerate(paths):
        h, fr, dur = prints[p]
        for rid, n, off, stretch, share in index.query(h, fr, exclude={i}):
            q = index.names[rid]
            if paths.index(q) < i:   # each pair once
                continue
            rows.append({"File_A": os.path.basename(p), "File_B": os.path.basename(q),
                         "Kind": "identical file" if md5[p] == md5[q] else "same audio",
                         "Matched_Landmarks": n, "Offset_s": off, "Shared_Audio_s": stretch,
                         "Share_Of_A": round(min(1.0, stretch / dur), 3) if dur else None, "Match_Strength": share})
    return rows, prints


def load_public_index(path=PUBLIC_INDEX):
    """The index written by build_corpus_index.py, or None when there is none."""
    f = os.path.join(path, "index.pkl")
    if not os.path.isfile(f):
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def save_public_index(index, meta, path=PUBLIC_INDEX):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "index.pkl"), "wb") as fh:
        pickle.dump({"audio": index, "meta": meta}, fh, protocol=pickle.HIGHEST_PROTOCOL)


def text_matches(queries, corpus, threshold=TEXT_SIMILARITY, same_set=False):
    """[(query i, corpus j, similarity)] for texts with at least MIN_TEXT_WORDS words. same_set: corpus is queries
    (pairs i < j only)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    qi = [i for i, t in enumerate(queries) if len(t.split()) >= MIN_TEXT_WORDS]
    cj = qi if same_set else [j for j, t in enumerate(corpus) if len(t.split()) >= MIN_TEXT_WORDS]
    if not qi or not cj:
        return []
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True)
    vec.fit([queries[i] for i in qi] + ([] if same_set else [corpus[j] for j in cj]))
    Q, C = vec.transform([queries[i] for i in qi]), vec.transform([corpus[j] for j in cj])
    out = []
    for start in range(0, len(qi), 2000):   # blocks keep the similarity matrix small
        sims = (Q[start:start + 2000] @ C.T).tocoo()
        for a, b, s in zip(sims.row, sims.col, sims.data):
            i, j = qi[start + a], cj[b]
            if s >= threshold and (not same_set or i < j):
                out.append((i, j, round(float(s), 3)))
    return out


def public_audio_matches(prints, public):
    rows = []
    if not public:
        return rows
    index, meta = public["audio"], public["meta"]
    for p, (h, fr, dur) in prints.items():
        for rid, n, off, stretch, share in index.query(h, fr)[:5]:
            rows.append({"Filename": os.path.basename(p), "Match": "audio", "Corpus": meta[rid]["corpus"],
                         "Corpus_Item": meta[rid]["item"], "Matched_Landmarks": n, "Offset_s": off,
                         "Shared_Audio_s": stretch, "Similarity": share, "Text": None})
    logging.info(f"public corpus audio check: {len(rows)} matches")
    return rows
