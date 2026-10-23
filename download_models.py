"""Download the models SpeechQM-Agent uses, so runs work offline afterwards. Used by run_qc.py; also runnable:
    python download_models.py [language ...]        (default: Hindi; "all" = every configured language)
For each language the first ASR candidate in config.ASR_MODELS the Hugging Face token can download is used
(a gated ai4bharat model whose terms were not accepted falls back to the Vakyansh model).
"""
import json
import shutil
import tempfile
import sys
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import auth_check, constants, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from config import (ASR_MODELS, ALIGN_MODEL, INDICBERT_TOKENIZER, INDICLID_DIR, OLLAMA_BASE_URL, PLANNER_MODEL, JUDGE_MODEL, MONITOR_MODEL,
                    ASR_ENGINE, INDICCONFORMER, INDICCONFORMER_LANGS, ENGLISH_ASR, AUDIO_LID_MODELS,
                    CODER_MODEL, MOE_EXPERTS, BODHAN_ASR, NEMOTRON_ASR, NEMOTRON_PYTHON)

INDICLID_URL = "https://github.com/AI4Bharat/IndicLID/releases/download/v1.0/"
INDICLID = {"indiclid-ftn": "model_baseline_roman.bin", "indiclid-ftr": "model_baseline_roman.bin",
            "indiclid-bert": "basline_nn_simple.pt"}
HF_WEIGHTS = ["*.json", "*.bin", "*.safetensors", "*.txt", "*.model"]


def has_access(repo: str) -> bool:
    """True if the current Hugging Face token may download the repo (gated repos need their terms accepted).
    Offline (HF_HUB_OFFLINE=1): True if the repo is already in the local cache."""
    if constants.HF_HUB_OFFLINE:
        try:
            snapshot_download(repo, local_files_only=True)
            return True
        except Exception:
            return False
    try:
        auth_check(repo)
        return True
    except HfHubHTTPError:   # GatedRepoError, RepositoryNotFoundError, 401 without a token
        return False


def ollama_models() -> list:
    """Models present in the Ollama server; raises OSError if the server is not reachable."""
    with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
        return [m["name"] for m in json.load(r)["models"]]


def ensure_ollama(models=tuple(m for m in (PLANNER_MODEL, CODER_MODEL, JUDGE_MODEL, MONITOR_MODEL) if m.lower() not in ("none", "off"))):
    have = ollama_models()
    for m in dict.fromkeys(models):
        if m in have or f"{m}:latest" in have:
            continue
        print(f"ollama: pulling {m} (one-time download, ~5 GB)")
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/pull", data=json.dumps({"model": m}).encode(),
                                     headers={"Content-Type": "application/json"})
        last = ""
        with urllib.request.urlopen(req) as r:
            for line in r:
                msg = json.loads(line)
                if "error" in msg:
                    raise RuntimeError(f"ollama pull {m}: {msg['error']}")
                status = msg.get("status", "")
                if msg.get("total"):
                    status += f" {100 * msg.get('completed', 0) // msg['total']}%"
                if status != last:
                    print(f"\r  {status:60}", end="", flush=True)
                    last = status
        print()


def asr_candidate(language: str):
    """First ASR model of the language the token can download, or None."""
    return next((repo for repo in ASR_MODELS[language] if has_access(repo)), None)


def ensure_hf(languages, speech_backend="pyannote"):
    """ASR (first accessible candidate per language), CTC alignment, IndicBERT tokenizer, VAD/diarization."""
    repos = [ALIGN_MODEL]
    for lang in languages:
        repo = asr_candidate(lang)
        if repo is None:
            raise RuntimeError(f"no downloadable ASR model for {lang}: {ASR_MODELS[lang]}")
        repos.append(repo)
    for repo in dict.fromkeys(repos):
        print(f"hf: {repo}")
        snapshot_download(repo, allow_patterns=HF_WEIGHTS)   # offline: resolves from the cache
    import utility_functions as U
    # IndicConformer and Whisper: the main ASR with ASR_ENGINE=best (and the fallback if Bodhan is gated), else experts
    if ASR_ENGINE in ("best", "bodhan") and any(lang in INDICCONFORMER_LANGS for lang in languages):
        print(f"hf: {INDICCONFORMER} (~2.4 GB)")
        U.get_indicconformer()
    if ASR_ENGINE in ("best", "bodhan") and "English" in languages:
        print(f"hf: {ENGLISH_ASR} (~3 GB)")
        snapshot_download(ENGLISH_ASR, allow_patterns=HF_WEIGHTS)
    # ASR experts of the MoE score (task 3); one that cannot be downloaded only leaves the score with fewer models
    if ("bodhan" in MOE_EXPERTS or ASR_ENGINE == "bodhan") and any(lang in INDICCONFORMER_LANGS or lang == "English"
                                                                   for lang in languages):
        if has_access(BODHAN_ASR):
            print(f"hf: {BODHAN_ASR} (main ASR, ~4.6 GB)")
            U.get_bodhan()
        else:
            print(f"  skipped {BODHAN_ASR}: accept its terms on Hugging Face (until then IndicConformer / Whisper is the main ASR)")
    if "nemotron" in MOE_EXPERTS and "English" in languages:
        print(f"hf: {NEMOTRON_ASR} (MoE expert, ~2.5 GB)")
        snapshot_download(NEMOTRON_ASR, allow_patterns=HF_WEIGHTS)
        if not Path(NEMOTRON_PYTHON).is_file():
            print(f"  {NEMOTRON_PYTHON} not found: set up the Nemotron venv (README, 'MoE experts')")
    for repo in AUDIO_LID_MODELS:   # audio language ID (task 15): Whisper-large-v3 ~3 GB, MMS-LID-126 ~3.9 GB
        print(f"hf: {repo} (audio language ID)")
        snapshot_download(repo, allow_patterns=HF_WEIGHTS)
    print(f"hf: {INDICBERT_TOKENIZER} (IndicLID tokenizer, ~1.1 GB)")
    snapshot_download(INDICBERT_TOKENIZER, allow_patterns=HF_WEIGHTS)
    # loading once downloads everything the VAD/diarization models pull in (embedding model, PLDA, ECAPA)
    print(f"speech: {speech_backend} VAD + diarization")
    U.get_vad_model()
    U.get_diarization_pipeline() if speech_backend == "pyannote" else U.get_speaker_encoder()


def ensure_indiclid():
    """AI4Bharat IndicLID (GitHub release, not on Hugging Face), ~1.5 GB. Downloaded and extracted into a temporary
    folder first, so an interrupted download never leaves a half-written model behind."""
    INDICLID_DIR.mkdir(parents=True, exist_ok=True)
    for name, f in INDICLID.items():
        if (INDICLID_DIR / name / f).exists():
            continue
        with tempfile.TemporaryDirectory(dir=INDICLID_DIR) as tmp:
            zip_path = Path(tmp) / f"{name}.zip"
            with urllib.request.urlopen(INDICLID_URL + name + ".zip") as r, open(zip_path, "wb") as out:
                total, done = int(r.headers.get("Content-Length") or 0), 0
                while chunk := r.read(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    pct = f" {100 * done // total}%" if total else ""
                    print(f"\r  indiclid: downloading {name}{pct} ({done >> 20} MB)", end="", flush=True)
            print()
            zipfile.ZipFile(zip_path).extractall(tmp)
            if (INDICLID_DIR / name).exists():
                shutil.rmtree(INDICLID_DIR / name)
            shutil.move(str(Path(tmp) / name), str(INDICLID_DIR / name))


if __name__ == "__main__":
    args = sys.argv[1:] or ["Hindi"]
    langs = list(ASR_MODELS) if args == ["all"] else [a.title() for a in args]
    from config import SPEECH_BACKEND
    ensure_ollama()
    ensure_hf(langs, SPEECH_BACKEND)
    ensure_indiclid()
    print("done")
