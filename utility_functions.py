
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from dataclasses import dataclass
import pandas as pd
import logging
import os
import fasttext
import json
import csv
import re
import threading
from functools import lru_cache
from typing import List, Tuple
from torch.utils.data import Dataset
from config import (ASR_MODELS, ALIGN_MODEL, DEFAULT_LANGUAGE, SPEAKER_MODEL, INDICBERT_TOKENIZER, INDICLID_DIR, MODELS_DIR,
                    SPEECH_BACKEND, HF_TOKEN, PYANNOTE_VAD_MODEL, PYANNOTE_DIARIZATION,
                    DIAR_WINDOW_S, DIAR_HOP_S, DIAR_CLUSTER_THRESHOLD, SPEAKER_MATCH_THRESHOLD, torch_device)

SR = 16000
DEVICE = torch_device()
ASR_CHUNK_S = 8.0   # wav2vec2 runs on <=8 s chunks built from VAD speech regions (lowest WER of 30/8/5 s on Hindi and Telugu)
# LangGraph runs independent tasks in parallel threads; the TorchScript VAD and the MPS models are not
# thread-safe (concurrent calls corrupt the heap), so all model inference goes through this lock.
MODEL_LOCK = threading.RLock()


def load_audio(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)
    return audio.astype(np.float32)


# ---------------------------------------------------------------- language identification (IndicLID)
def language_identification_indiclid(transcription: str) -> List[Tuple[str, str, float, str]]:
    logging.info(f"Processing transcription for IndicLID: {transcription[:50]}...")
    try:
        if not transcription or transcription.strip() == "":
            return [(transcription, "Unknown", 0.0, "IndicLID")]

        indiclid_model = get_indiclid()
        with MODEL_LOCK:
            results = indiclid_model.batch_predict([transcription], batch_size=1)

        if not results or not isinstance(results, list):
            raise ValueError(f"Invalid result format from IndicLID: {results}")

        output = []
        for text, lang_code, confidence, model_used in results:
            if not isinstance(confidence, (int, float)):
                confidence = 0.0
            output.append((text, lang_code, float(confidence), model_used))

        return output if output else [(transcription, "Unknown", 0.0, "IndicLID")]

    except Exception as e:
        logging.warning(f"IndicLID failed for transcription '{transcription[:50]}...': {e}")
        return [(transcription, "Error", 0.0, "IndicLID")]


# ---------------------------------------------------------------- transliteration
def transliterate_file(file_path: str, lang_code: str, llm=None) -> str:
    """Roman -> native script. Uses AI4Bharat IndicXlit when installed (needs fairseq, which does not
    build on Apple Silicon); otherwise asks the local LLM."""
    try:
        supported_langs = {'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'pa', 'sd', 'si', 'ta', 'te', 'ur'}
        if lang_code not in supported_langs:
            raise ValueError(f"Unsupported language code: {lang_code}. Supported codes: {', '.join(supported_langs)}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        df = pd.read_csv(file_path)
        if "ground_truth" not in df.columns:
            raise ValueError("Column 'ground_truth' not found in the CSV file")
        try:
            from ai4bharat.transliteration import XlitEngine
            if not hasattr(np, 'float'):
                np.float = float
            engine = XlitEngine(beam_width=10, src_script_type="roman")

            def translit(text):
                result = engine.translit_sentence(str(text), lang_code)
                return result[lang_code] if isinstance(result, dict) and lang_code in result else result
        except ImportError:
            if llm is None:
                raise RuntimeError("IndicXlit is not installed and no LLM was given for transliteration")
            logging.info("IndicXlit not installed; transliterating with the local LLM")

            def translit(text):
                prompt = (f"Transliterate the following romanized text into the native script of the language with ISO code "
                          f"'{lang_code}'. Keep words that are already in native script unchanged. Do not translate. "
                          f"Return only the transliterated text.\nText: {text}")
                return llm.invoke(prompt).content.strip()

        def transliterate(text):
            if pd.isna(text) or str(text).strip() == "":
                return ""
            try:
                return translit(text)
            except Exception as e:
                print(f"Error transliterating '{text}': {e}")
                return text
        df["ground_truth_transliterated"] = df["ground_truth"].apply(transliterate)
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_transliterated.csv"
        df.to_csv(output_path, index=False, encoding='utf-8')
        return f"Output saved at: {output_path}"
    except Exception as e:
        print(f"Error in transliterate_file: {e}")
        return f"Error: {e}"


# ---------------------------------------------------------------- VAD (pyannote/segmentation-3.0, or Silero)
def audio_input(audio: np.ndarray) -> dict:
    """In-memory audio for pyannote (avoids its file decoders)."""
    return {"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SR}


@lru_cache(maxsize=1)
def get_vad_model():
    with MODEL_LOCK:
        if SPEECH_BACKEND == "pyannote":
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection
            model = Model.from_pretrained(PYANNOTE_VAD_MODEL, token=HF_TOKEN)
            pipeline = VoiceActivityDetection(segmentation=model)
            pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
            return pipeline
        from silero_vad import load_silero_vad
        threads = torch.get_num_threads()
        model = load_silero_vad()
        torch.set_num_threads(threads)   # silero drops torch to one thread, which slows every other CPU model
        return model


def speech_regions(audio: np.ndarray) -> List[Tuple[float, float]]:
    vad = get_vad_model()
    with MODEL_LOCK:
        if SPEECH_BACKEND == "pyannote":
            return [(float(s.start), float(s.end)) for s in vad(audio_input(audio)).get_timeline().support()]
        from silero_vad import get_speech_timestamps
        ts = get_speech_timestamps(torch.from_numpy(audio), vad, sampling_rate=SR, return_seconds=True)
    return [(float(t["start"]), float(t["end"])) for t in ts]


# ---------------------------------------------------------------- ASR
@lru_cache(maxsize=2)
def get_asr(model_id):
    with MODEL_LOCK:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id).to(DEVICE).eval()
        return processor, model


def asr_chunks(audio):
    """Merge VAD speech regions into chunks of at most ASR_CHUNK_S seconds."""
    chunks = []
    for s, e in speech_regions(audio):
        while e - s > ASR_CHUNK_S:
            chunks.append((s, s + ASR_CHUNK_S))
            s += ASR_CHUNK_S
        if chunks and e - chunks[-1][0] <= ASR_CHUNK_S and s - chunks[-1][1] < 1.0:
            chunks[-1] = (chunks[-1][0], e)
        else:
            chunks.append((s, e))
    return chunks


@lru_cache(maxsize=None)
def asr_model_for(language):
    """First ASR candidate for the language that loads (ai4bharat ones fail until their terms are accepted)."""
    if language not in ASR_MODELS:
        raise ValueError(f"No ASR model configured for {language}; supported: {', '.join(ASR_MODELS)}")
    errors = []
    for model_id in ASR_MODELS[language]:
        try:
            get_asr(model_id)
            return model_id
        except Exception as e:
            errors.append(f"{model_id}: {type(e).__name__}")
            logging.warning(f"ASR model {model_id} unavailable ({type(e).__name__}), trying the next one")
    raise RuntimeError(f"No usable ASR model for {language}: {errors}")


def align_model_for(language):
    return ALIGN_MODEL if language == "Hindi" else asr_model_for(language)


def transcribe_audio(audio_path, source_lang):
    model_id = asr_model_for(source_lang)
    processor, model = get_asr(model_id)
    print(f"Transcribing {audio_path} using {model_id} on {DEVICE}")
    audio_array = load_audio(audio_path)
    words = []
    for s, e in asr_chunks(audio_array):
        piece = audio_array[int(s * SR):int(e * SR)]
        if len(piece) < SR // 10:
            continue
        inputs = processor(piece, sampling_rate=SR, return_tensors="pt")
        with MODEL_LOCK, torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits
            predicted_ids = torch.argmax(logits, dim=-1).cpu()
        words += processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip().split()
    print("TRANSCRIPT:", words)
    return " ".join(words)


def transcribe_folder_to_csv(folder_path: str, source_language: str):
    parent = os.path.dirname(folder_path.rstrip("/\\"))
    output_path = os.path.join(parent, 'results', "indicconf_hypothesis.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        logging.info(f"Skipping transcription: File already exists at {output_path}")
        return output_path

    results = []
    for filename in sorted(os.listdir(folder_path)):
        audio_path = os.path.join(folder_path, filename)
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logging.warning(f"Skipping {filename}: File size is 0 bytes")
            continue

        if not filename.lower().endswith((".wav", ".mp3", ".flac")):
            logging.warning(f"Skipping {filename}: not an audio file")
            continue
        transcribe_filename = filename

        try:
            hypothesis = transcribe_audio(audio_path, source_language)
            results.append({
                "Filename": transcribe_filename,
                "Indiconformer_Hypothesis": hypothesis
            })
            logging.info(f"Transcribed {transcribe_filename}: {hypothesis[:50]}...")
        except Exception as e:
            logging.error(f"Failed to transcribe {transcribe_filename}: {e}")

    if not results:
        logging.error("No valid transcriptions generated")
        return "Error: No valid transcriptions"

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logging.info(f"Transcriptions saved to {output_path}")
    return output_path


# ---------------------------------------------------------------- silence
def get_total_silence_time(audio_file_path):
    audio = load_audio(audio_file_path)
    total_audio_duration = len(audio) / SR
    total_silence = 0.0
    last_end = 0.0
    for start, end in speech_regions(audio):
        if start > last_end:
            total_silence += start - last_end
        last_end = max(last_end, end)
    if last_end < total_audio_duration:
        total_silence += total_audio_duration - last_end
    return total_silence


def process_folder_vad(audio_folder: str):
    parent  = os.path.dirname(audio_folder.rstrip("/\\"))
    output_path = os.path.join(parent, 'results', "vad_silence_stats.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        logging.info(f"Skipping VAD processing: File already exists at {output_path}")
        return f"File already exists: {output_path}"

    results = []
    for filename in sorted(os.listdir(audio_folder)):
        if filename.lower().endswith((".wav", ".flac", ".mp3")):
            audio_path = os.path.join(audio_folder, filename)
            try:
                silence = get_total_silence_time(audio_path)
                results.append({
                    "Filename": filename,
                    "Total Silence (s)": round(silence, 2)
                })
                print(f"Processed: {filename}, Silence: {round(silence, 2)}s")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logging.info(f"VAD results saved to {output_path}")
    return f"CSV saved at: {output_path}"


# ---------------------------------------------------------------- speakers (pyannote/speaker-diarization-3.1, or local ECAPA clustering)
@lru_cache(maxsize=1)
def get_speaker_encoder():
    with MODEL_LOCK:
        from speechbrain.inference.speaker import EncoderClassifier
        return EncoderClassifier.from_hparams(source=SPEAKER_MODEL, savedir=str(MODELS_DIR / "spkrec-ecapa-voxceleb"),
                                              run_opts={"device": DEVICE})


def embed_windows(audio, windows, batch=32):
    enc = get_speaker_encoder()
    out = []
    for i in range(0, len(windows), batch):
        pieces = [audio[int(s * SR):int(e * SR)] for s, e in windows[i:i + batch]]
        n = max(len(p) for p in pieces)
        wavs = torch.zeros(len(pieces), n)
        for j, p in enumerate(pieces):
            wavs[j, :len(p)] = torch.from_numpy(p)
        lens = torch.tensor([len(p) / n for p in pieces])
        with MODEL_LOCK, torch.no_grad():
            out.append(enc.encode_batch(wavs, lens).squeeze(1).cpu().numpy())
    emb = np.concatenate(out)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


@lru_cache(maxsize=1)
def get_diarization_pipeline():
    with MODEL_LOCK:
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(PYANNOTE_DIARIZATION, token=HF_TOKEN)
        return pipeline.to(torch.device(DEVICE))


def diarize(audio):
    """Returns {local_label: (seconds, unit-norm speaker embedding)} for one file."""
    if SPEECH_BACKEND == "pyannote":
        return diarize_pyannote(audio)
    return diarize_local(audio)


def diarize_pyannote(audio):
    pipeline = get_diarization_pipeline()
    with MODEL_LOCK:
        out = pipeline(audio_input(audio), return_embeddings=True)
    ann, emb = (out.speaker_diarization, out.speaker_embeddings) if hasattr(out, "speaker_diarization") else out
    result = {}
    for i, label in enumerate(ann.labels()):   # embedding rows follow ann.labels() order
        secs = float(ann.label_duration(label))
        e = np.asarray(emb[i], dtype=np.float32) if emb is not None and i < len(emb) else None
        # very short speakers get no embedding (NaN): still counted, never matched across files
        ok = e is not None and np.isfinite(e).all() and np.linalg.norm(e) > 0
        result[label] = (secs, e / np.linalg.norm(e) if ok else None)
    return result


def diarize_local(audio):
    """Silero VAD -> ECAPA windows -> agglomerative clustering."""
    from sklearn.cluster import AgglomerativeClustering
    windows, owned = [], []   # owned: the stretch of speech each window accounts for
    for s, e in speech_regions(audio):
        if e - s < 0.4:
            continue
        starts = list(np.arange(s, max(s, e - DIAR_WINDOW_S) + 1e-6, DIAR_HOP_S))
        seg = [(t, min(t + DIAR_WINDOW_S, e)) for t in starts]
        if seg[-1][1] < e - DIAR_HOP_S / 2:
            seg.append((e - DIAR_WINDOW_S, e))
        mids = [(a + b) / 2 for a, b in seg]
        bounds = [s] + [(mids[k] + mids[k + 1]) / 2 for k in range(len(mids) - 1)] + [e]
        windows += seg
        owned += [bounds[k + 1] - bounds[k] for k in range(len(seg))]
    if not windows:
        return {}
    emb = embed_windows(audio, windows)
    owned = np.array(owned)
    if len(windows) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=DIAR_CLUSTER_THRESHOLD,
                                         metric="cosine", linkage="average").fit_predict(emb)
    # fold clusters with under 5% of the speech (or under 1 s) into the nearest larger cluster
    total = owned.sum()
    big = [c for c in set(labels) if owned[labels == c].sum() >= max(1.0, 0.05 * total)] or [np.bincount(labels).argmax()]
    cents = {c: emb[labels == c].mean(0) for c in big}
    labels = np.array([l if l in cents else max(cents, key=lambda c: emb[i] @ cents[c]) for i, l in enumerate(labels)])
    result = {}
    for c in sorted(set(labels)):
        cent = emb[labels == c].mean(0)
        result[c] = (float(owned[labels == c].sum()), cent / np.linalg.norm(cent))
    return result


def save_num_speakers(folder_path: str) -> str:
    """Per-file speaker count and durations. Speaker IDs are global across the folder: a voice that
    matches one heard in an earlier file gets the same SPEAKER_xx, so the new/old check is meaningful."""
    parent = os.path.dirname(folder_path.rstrip("/\\"))
    output_csv = os.path.join(parent, 'results', "num_speakers.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if os.path.exists(output_csv):
        logging.info(f"Skipping speaker number calculation: File already exists at {output_csv}")
        return output_csv
    registry = []   # global speaker centroids
    try:
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["File Name", "Number of Speakers", "Speaker Durations"])
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith((".wav", ".mp3", ".flac")):
                    audio_path = os.path.join(folder_path, filename)
                    try:
                        local = diarize(load_audio(audio_path))
                        speaker_durations, used = {}, set()
                        for _, (secs, cent) in sorted(local.items(), key=lambda kv: -kv[1][0]):
                            sims = [] if cent is None else [(float(cent @ g), k) for k, g in enumerate(registry)
                                                            if k not in used and g is not None]
                            best = max(sims, default=(-1.0, None))
                            if best[0] >= SPEAKER_MATCH_THRESHOLD:
                                k = best[1]
                            else:
                                registry.append(cent)
                                k = len(registry) - 1
                            used.add(k)
                            speaker_durations[f"SPEAKER_{k:02d}"] = round(secs / 3600, 4)
                        speaker_durations_json = json.dumps(speaker_durations)
                        writer.writerow([filename, len(speaker_durations), speaker_durations_json])
                        print(f"Processed {filename}: {len(speaker_durations)} unique speakers, Durations: {speaker_durations_json}")
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        writer.writerow([filename, "Error", f"Error: {e}"])
        return f"CSV saved at: {output_csv}"
    except Exception as e:
        return f"Error writing to CSV: {e}"


def transcript_quality(transcript):
    words = transcript.strip().split()
    repeated = len(set(words)) < len(words) * 0.7
    if len(words) > 3 and not repeated:
        return "passed"
    else:
        return "failed"


# ---------------------------------------------------------------- forced alignment / CTC score
def force_alignment_and_ctc_score(speech_file, given_transcript, model_name=ALIGN_MODEL):
    try:
        logging.info(f"Using device in force_alignment_and_ctc_score: {DEVICE}")
        processor, model = get_asr(model_name)
        wav = load_audio(speech_file)
        # emissions in contiguous 30 s chunks: one pass over a long file does not fit in memory
        step = int(ASR_CHUNK_S * SR)
        parts = []
        for i in range(0, len(wav), step):
            piece = wav[i:i + step]
            if len(piece) < 400:
                continue
            input_values = processor(piece, return_tensors="pt", sampling_rate=SR).input_values.to(DEVICE)
            with MODEL_LOCK, torch.no_grad():
                parts.append(torch.log_softmax(model(input_values).logits, dim=-1).cpu())
        emissions = torch.cat(parts, dim=1)
        # the vocabulary marks word boundaries with "|"
        token_ids = processor.tokenizer.encode(" ".join(str(given_transcript).split()), add_special_tokens=False)
        if not token_ids:
            return [], 0
        def get_trellis(emission, token_ids, blank_id=0):
            num_frame = emission.size(0)
            num_tokens = len(token_ids)
            trellis = torch.zeros((num_frame, num_tokens))
            trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
            trellis[0, 1:] = -float("inf")
            trellis[-num_tokens + 1:, 0] = float("inf")
            for t in range(num_frame - 1):
                trellis[t + 1, 1:] = torch.maximum(
                    trellis[t, 1:] + emission[t, blank_id],
                    trellis[t, :-1] + emission[t, token_ids[1:]],
                )
            return trellis
        trellis = get_trellis(emissions[0], token_ids)
        @dataclass
        class Point:
            token_index: int
            time_index: int
            score: float
        def backtrack(trellis, emission, token_ids, blank_id=0):
            t, j = trellis.size(0) - 1, trellis.size(1) - 1
            path = [Point(j, t, emission[t, blank_id].exp().item())]
            while j > 0 and t > 0:
                p_stay = emission[t - 1, blank_id]
                p_change = emission[t - 1, token_ids[j]]
                stayed = trellis[t - 1, j] + p_stay
                changed = trellis[t - 1, j - 1] + p_change
                t -= 1
                if changed > stayed:
                    j -= 1
                prob = (p_change if changed > stayed else p_stay).exp().item()
                path.append(Point(j, t, prob))
            while t > 0:
                prob = emission[t - 1, blank_id].exp().item()
                path.append(Point(j, t - 1, prob))
                t -= 1
            return path[::-1]
        path = backtrack(trellis, emissions[0], token_ids)
        @dataclass
        class Segment:
            label: str
            start: int
            end: int
            score: float
        def merge_repeats(path):
            i1, i2 = 0, 0
            segments = []
            while i1 < len(path):
                while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                    i2 += 1
                avg_score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
                segments.append(Segment(
                    processor.tokenizer.convert_ids_to_tokens(token_ids[path[i1].token_index]),
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    avg_score
                ))
                i1 = i2
            return segments
        segments = merge_repeats(path)
        ratio = len(wav) / trellis.size(0)
        fa = []
        for i, seg in enumerate(segments):
            if seg.label not in ["|", "<unk>", "<pad>"]:
                x0 = int(ratio * seg.start)
                x1 = int(ratio * seg.end)
                fa.append([seg.label, f"{x0 / SR:.2f}", f"{x1 / SR:.2f}", f"{seg.score:.2f}"])
        filtered_segments = [seg for seg in segments if seg.label not in ["|", "<unk>", "<pad>"]]
        total_score = sum(seg.score for seg in filtered_segments)
        average_ctc_score = total_score / len(filtered_segments) if filtered_segments else 0
        return fa, average_ctc_score
    except Exception as e:
        logging.error(f"Error in forced alignment for {speech_file}: {e}")
        return [], 0


def _col(df, *names):
    """Case-insensitive column lookup."""
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    raise KeyError(f"none of {names} in columns {list(df.columns)}")


def process_audio_directory(audio_dir, transcript_csv, output_csv_path=None, language=DEFAULT_LANGUAGE):
    try:
        df = pd.read_csv(transcript_csv)
        fcol = _col(df, "filename")
        tcol = _col(df, "ground_truth", "transcription", "Indiconformer_Hypothesis")
        model_name = align_model_for(language)
        results = []
        for idx, row in df.iterrows():
            filename = row[fcol]
            transcript = str(row[tcol])
            audio_path = os.path.join(audio_dir, filename)
            if not os.path.exists(audio_path):
                logging.warning(f"Audio file not found: {audio_path}")
                results.append({
                    "filename": filename,
                    "label": "Error",
                    "start_time": "0.00",
                    "end_time": "0.00",
                    "score": "0.00",
                    "average_ctc_score": "0.00"
                })
                continue
            logging.info(f"Processing: {filename}")
            aligned_segments, avg_score = force_alignment_and_ctc_score(audio_path, transcript, model_name)
            if not aligned_segments:
                results.append({
                    "filename": filename,
                    "label": "Error",
                    "start_time": "0.00",
                    "end_time": "0.00",
                    "score": "0.00",
                    "average_ctc_score": f"{avg_score:.2f}"
                })
                continue
            for seg in aligned_segments:
                results.append({
                    "filename": filename,
                    "label": seg[0],
                    "start_time": seg[1],
                    "end_time": seg[2],
                    "score": seg[3],
                    "average_ctc_score": f"{avg_score:.2f}"
                })
        if output_csv_path:
            keys = results[0].keys() if results else ["filename", "label", "start_time", "end_time", "score", "average_ctc_score"]
            with open(output_csv_path, "w", newline='', encoding="utf-8") as out_file:
                writer = csv.DictWriter(out_file, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            logging.info(f"Results saved to {output_csv_path}")
        return results
    except Exception as e:
        logging.error(f"Error in process_audio_directory: {e}")
        return []


# ---------------------------------------------------------------- upsampling
def is_upsampled_from_8k_v2(audio_path, threshold_ratio=0.02):
    try:
        waveform = load_audio(audio_path)
    except Exception as e:
        print(f"Encountered an error loading file {audio_path}")
        print(f"ERROR: {e}")
        return False
    spectrum = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), 1 / SR)
    total_energy = np.sum(spectrum)
    high_freq_band = (freqs >= 4000) & (freqs <= 8000)
    high_freq_energy = np.sum(spectrum[high_freq_band])
    ratio = high_freq_energy / total_energy
    return bool(ratio < threshold_ratio)


def check_upsampling_folder(folder_path: str):
    results = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".wav", ".mp3")):
            audio_path = os.path.join(folder_path, filename)
            try:
                is_upsampled = is_upsampled_from_8k_v2(audio_path)
                results.append({
                    "Filename": filename,
                    "Is_Upsampled": is_upsampled
                })
                print(f"Processed: {filename}, Upsampled: {is_upsampled}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                results.append({
                    "Filename": filename,
                    "Is_Upsampled": "Error"
                })
    df = pd.DataFrame(results)
    parent = os.path.dirname(folder_path.rstrip("/\\"))
    os.makedirs(os.path.join(parent, 'results'), exist_ok=True)
    output_path = os.path.join(parent, 'results', "upsampling_check.csv")
    df.to_csv(output_path, index=False)
    return f"CSV saved at: {output_path}"


# ---------------------------------------------------------------- IndicLID
class IndicBERT_Data(Dataset):
    def __init__(self, indices, X):
        self.size = len(X)
        self.x = X
        self.i = indices
    def __len__(self):
        return (self.size)
    def __getitem__(self, idx):
        text = self.x[idx]
        index = self.i[idx]
        return tuple([index, text])


@lru_cache(maxsize=1)
def get_indiclid():
    with MODEL_LOCK:
        return IndicLID(input_threshold=0.5, roman_lid_threshold=0.6)


class IndicLID():
    def __init__(self, input_threshold=0.5, roman_lid_threshold=0.6):
        self.device = torch.device(DEVICE)
        self.IndicLID_FTN_path = str(INDICLID_DIR / 'indiclid-ftn' / 'model_baseline_roman.bin')
        self.IndicLID_FTR_path = str(INDICLID_DIR / 'indiclid-ftr' / 'model_baseline_roman.bin')
        self.IndicLID_BERT_path = str(INDICLID_DIR / 'indiclid-bert' / 'basline_nn_simple.pt')
        self.IndicLID_FTN = fasttext.load_model(self.IndicLID_FTN_path)
        self.IndicLID_FTR = fasttext.load_model(self.IndicLID_FTR_path)
        # the released BERT checkpoint is a pickled model from an older transformers version:
        # rebuild it with the installed transformers and copy the weights across
        old = torch.load(self.IndicLID_BERT_path, map_location="cpu", weights_only=False)
        cfg = type(old.config).from_dict(old.config.to_dict())
        self.IndicLID_BERT = AutoModelForSequenceClassification.from_config(cfg)
        sd = {k: v for k, v in old.state_dict().items() if not k.endswith("position_ids")}
        missing, unexpected = self.IndicLID_BERT.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"IndicLID-BERT weights do not match: missing={missing} unexpected={unexpected}")
        del old
        self.IndicLID_BERT.to(self.device).eval()
        self.IndicLID_BERT_tokenizer = AutoTokenizer.from_pretrained(INDICBERT_TOKENIZER)
        self.input_threshold = input_threshold
        self.model_threshold = roman_lid_threshold
        self.classes = 47
        self.IndicLID_lang_code_dict = {
            'asm_Latn': 0, 'ben_Latn': 1, 'brx_Latn': 2, 'guj_Latn': 3, 'hin_Latn': 4, 'kan_Latn': 5,
            'kas_Latn': 6, 'kok_Latn': 7, 'mai_Latn': 8, 'mal_Latn': 9, 'mni_Latn': 10, 'mar_Latn': 11,
            'nep_Latn': 12, 'ori_Latn': 13, 'pan_Latn': 14, 'san_Latn': 15, 'snd_Latn': 16, 'tam_Latn': 17,
            'tel_Latn': 18, 'urd_Latn': 19, 'eng_Latn': 20, 'other': 21, 'asm_Beng': 22, 'ben_Beng': 23,
            'brx_Deva': 24, 'doi_Deva': 25, 'guj_Gujr': 26, 'hin_Deva': 27, 'kan_Knda': 28, 'kas_Arab': 29,
            'kas_Deva': 30, 'kok_Deva': 31, 'mai_Deva': 32, 'mal_Mlym': 33, 'mni_Beng': 34, 'mni_Meti': 35,
            'mar_Deva': 36, 'nep_Deva': 37, 'ori_Orya': 38, 'pan_Guru': 39, 'san_Deva': 40, 'sat_Olch': 41,
            'snd_Arab': 42, 'tam_Taml': 43, 'tel_Telu': 44, 'urd_Arab': 45
        }
        self.IndicLID_lang_code_dict_reverse = {v: k for k, v in self.IndicLID_lang_code_dict.items()}
    def pre_process(self, input):
        return input
    def char_percent_check(self, input):
        input_len = len(list(input))
        special_char_pattern = re.compile(r'[@_!#$%^&*()<>?/\|}{~:]')
        special_char_matches = special_char_pattern.findall(input)
        special_chars = len(special_char_matches)
        spaces = len(re.findall(r'\s', input))
        newlines = len(re.findall('\n', input))
        total_chars = input_len - (special_chars + spaces + newlines)
        en_pattern = re.compile('[a-zA-Z0-9]')
        en_matches = en_pattern.findall(input)
        en_chars = len(en_matches)
        if total_chars == 0:
            return 0
        return (en_chars/total_chars)
    @staticmethod
    def ft_predict(model, texts):
        # one text at a time: fasttext-predict's list form returns a different shape than fasttext's
        preds = [model.predict(t) for t in texts]
        return [p[0] for p in preds], [p[1] for p in preds]
    def native_inference(self, input_list, output_dict):
        if not input_list:
            return output_dict
        input_texts = [line[1].replace("\n", " ") for line in input_list]
        IndicLID_FTN_predictions = self.ft_predict(self.IndicLID_FTN, input_texts)
        for input, pred_label, pred_score in zip(input_list, IndicLID_FTN_predictions[0], IndicLID_FTN_predictions[1]):
            output_dict[input[0]] = (input[1], pred_label[0][9:], pred_score[0], 'IndicLID-FTN')
        return output_dict
    def roman_inference(self, input_list, output_dict, batch_size):
        if not input_list:
            return output_dict
        input_texts = [line[1].replace("\n", " ") for line in input_list]
        IndicLID_FTR_predictions = self.ft_predict(self.IndicLID_FTR, input_texts)
        IndicLID_BERT_inputs = []
        for input, pred_label, pred_score in zip(input_list, IndicLID_FTR_predictions[0], IndicLID_FTR_predictions[1]):
            if pred_label[0][9:] == 'eng_Latn' and pred_score[0] > self.model_threshold:
                output_dict[input[0]] = (input[1], pred_label[0][9:], pred_score[0], 'IndicLID-FTR')
            else:
                IndicLID_BERT_inputs.append(input)
        output_dict = self.IndicBERT_roman_inference(IndicLID_BERT_inputs, output_dict, batch_size)
        return output_dict
    def IndicBERT_roman_inference(self, IndicLID_BERT_inputs, output_dict, batch_size):
        if not IndicLID_BERT_inputs:
            return output_dict
        df = pd.DataFrame(IndicLID_BERT_inputs)
        dataloader = self.get_dataloaders(df.iloc[:,0], df.iloc[:,1], batch_size)
        with torch.no_grad():
            for data in dataloader:
                batch_indices = data[0]
                batch_inputs = data[1]
                word_embeddings = self.IndicLID_BERT_tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
                word_embeddings = word_embeddings.to(self.device)
                batch_outputs = self.IndicLID_BERT(word_embeddings['input_ids'],
                            token_type_ids=word_embeddings['token_type_ids'],
                            attention_mask=word_embeddings['attention_mask'])
                _, batch_predicted = torch.max(batch_outputs.logits, 1)
                for index, input, pred_label, logit in zip(batch_indices, batch_inputs, batch_predicted, batch_outputs.logits):
                    output_dict[int(index)] = (input, self.IndicLID_lang_code_dict_reverse[pred_label.item()],
                                        logit[pred_label.item()].item(), 'IndicLID-BERT')
        return output_dict
    def post_process(self, output_dict):
        results = []
        keys = list(output_dict.keys())
        keys.sort()
        for index in keys:
            results.append(output_dict[index])
        return results
    def get_dataloaders(self, indices, input_texts, batch_size):
        data_obj = IndicBERT_Data(indices, input_texts)
        dl = torch.utils.data.DataLoader(data_obj, batch_size=batch_size, shuffle=False)
        return dl
    def predict(self, input):
        input_list = [input,]
        return self.batch_predict(input_list, 1)
    def batch_predict(self, input_list, batch_size):
        output_dict = {}
        roman_inputs = []
        native_inputs = []
        for index, input in enumerate(input_list):
            if self.char_percent_check(input) > self.input_threshold:
                roman_inputs.append((index, input))
            else:
                native_inputs.append((index, input))
        output_dict = self.native_inference(native_inputs, output_dict)
        output_dict = self.roman_inference(roman_inputs, output_dict, batch_size)
        results = self.post_process(output_dict)
        return results

def get_required_inputs(task_id: int) -> list[str]:
    input_requirements = {
        1: ["audio_dir"],
        2: ["audio_dir"],
        3: ["ground_truth_csv"],
        4: ["ground_truth_csv"],
        5: ["ground_truth_csv"],
        6: ["ground_truth_csv"],
        7: ["audio_dir"],
        8: ["audio_dir"],
        9: ["audio_dir"],
        10: ["audio_dir", "ground_truth_csv"],
        11: ["audio_dir"],
        12: ["audio_dir"],
        13: ["audio_dir"],
        14: ["audio_dir", "ground_truth_csv"],
        15: ["audio_dir"],
        16: ["audio_dir"],
        17: ["audio_dir"],
        18: ["ground_truth_csv", "lang_code"],
        19: ["audio_dir"],
        20: ["audio_dir"],
        21: ["audio_dir"],
        22: ["audio_dir"],
        23: ["audio_dir"],
        24: ["audio_dir"],
    }
    return input_requirements.get(task_id, [])
