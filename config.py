# config.py - local (Mac) model choices and paths. Override any model with the env var of the same name.
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
INDICLID_DIR = MODELS_DIR / "indiclid"

# LLMs served by Ollama (replace the original gpt-4.1-mini / Groq llama-3.1-8b)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "qwen3:8b")   # task selection, DAG, domain, judge (best of qwen2.5 7b/14b, qwen3 8b on task selection)
CODER_MODEL = os.environ.get("CODER_MODEL", "qwen3:8b")   # writes + repairs the per-task Python scripts (9/9 vs 2/9 for qwen2.5-coder:7b)
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "8192"))
# LLM-as-a-judge of task 17 (fluency score), kept apart from the planner so a model is not grading its own pipeline.
# It can be served by another Ollama, e.g. a GPU machine through `ssh -N -L 11435:localhost:11434 <host>` and
# JUDGE_BASE_URL=http://localhost:11435 (Llama 3.3 70B needs ~43 GB at 4-bit, ~75 GB at 8-bit).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", PLANNER_MODEL)
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", OLLAMA_BASE_URL)

# Speech models. The ai4bharat ones are gated: accept their terms on Hugging Face and log in
# (`.venv/bin/hf auth login`) before running download_models.py.
# Candidates per language, tried in order: the ai4bharat model when its weights are published and your
# account has accepted its terms, else the ungated Vakyansh model of the same language.
# (ai4bharat/indicwav2vec-marathi and -telugu do not exist; the indicwav2vec_v1_ ones hold only a README.)
ASR_MODELS = {
    "Hindi": ["ai4bharat/indicwav2vec-hindi", "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"],
    "Tamil": ["Harveenchadha/vakyansh-wav2vec2-tamil-tam-250"],
    "Sanskrit": ["addy88/wav2vec2-sanskrit-stt"],
    "Marathi": ["Harveenchadha/vakyansh-wav2vec2-marathi-mrm-100"],
    "Telugu": ["Harveenchadha/vakyansh-wav2vec2-telugu-tem-100"],
    "Bengali": ["ai4bharat/indicwav2vec_v1_bengali", "Harveenchadha/vakyansh-wav2vec2-bengali-bnm-200"],
    "Gujarati": ["ai4bharat/indicwav2vec_v1_gujarati", "Harveenchadha/vakyansh-wav2vec2-gujarati-gnm-100"],
    "Odia": ["ai4bharat/indicwav2vec-odia", "Harveenchadha/vakyansh-wav2vec2-odia-orm-100"],
    "Kannada": ["Harveenchadha/vakyansh-wav2vec2-kannada-knm-560"],
    "Malayalam": ["Harveenchadha/vakyansh-wav2vec2-malayalam-mlm-8"],   # trained on only 8 h: expect a high WER
    "English": ["Harveenchadha/vakyansh-wav2vec2-indian-english-enm-700"],
    "Punjabi": ["Harveenchadha/vakyansh-wav2vec2-punjabi-pam-10"],      # 10 h of training data
    "Urdu": ["Harveenchadha/vakyansh-wav2vec2-urdu-urm-60"],
    "Assamese": ["Harveenchadha/vakyansh-wav2vec2-assamese-asm-8"],    # 8 h of training data
    "Nepali": ["Harveenchadha/vakyansh-wav2vec2-nepali-nem-130"],
}
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_LANGUAGE", "Hindi")   # when the prompt names no language
# ISO 639-1 code -> language, and the Unicode block its native script uses (for the script check)
LANG_CODES = {"hi": "Hindi", "ta": "Tamil", "sa": "Sanskrit", "mr": "Marathi", "te": "Telugu", "bn": "Bengali",
              "gu": "Gujarati", "or": "Odia", "kn": "Kannada", "ml": "Malayalam", "en": "English",
              "pa": "Punjabi", "ur": "Urdu", "as": "Assamese", "ne": "Nepali"}
SCRIPTS = {"Hindi": ("Devanagari", 0x0900, 0x097F), "Sanskrit": ("Devanagari", 0x0900, 0x097F),
           "Marathi": ("Devanagari", 0x0900, 0x097F), "Bengali": ("Bengali", 0x0980, 0x09FF),
           "Gujarati": ("Gujarati", 0x0A80, 0x0AFF), "Odia": ("Odia", 0x0B00, 0x0B7F),
           "Tamil": ("Tamil", 0x0B80, 0x0BFF), "Telugu": ("Telugu", 0x0C00, 0x0C7F),
           "Kannada": ("Kannada", 0x0C80, 0x0CFF), "Malayalam": ("Malayalam", 0x0D00, 0x0D7F),
           "English": ("Latin", 0x0041, 0x007A), "Punjabi": ("Gurmukhi", 0x0A00, 0x0A7F),
           "Urdu": ("Arabic", 0x0600, 0x06FF), "Assamese": ("Bengali", 0x0980, 0x09FF),
           "Nepali": ("Devanagari", 0x0900, 0x097F)}
# CTC alignment model: Vakyansh Hindi for Hindi (as in the original), else the language's own ASR model
ALIGN_MODEL = os.environ.get("ALIGN_MODEL", "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200")
# VAD + diarization: "pyannote" (original models; gated, needs `hf auth login`) or "local" (Silero VAD +
# SpeechBrain ECAPA clustering; no token needed)
SPEECH_BACKEND = os.environ.get("SPEECH_BACKEND", "pyannote")
HF_TOKEN = os.environ.get("HF_TOKEN") or True     # True = the token saved by `hf auth login`
PYANNOTE_VAD_MODEL = "pyannote/segmentation-3.0"
PYANNOTE_DIARIZATION = "pyannote/speaker-diarization-3.1"
# pyannote 4 also pulls the PLDA weights of speaker-diarization-community-1
PYANNOTE_REPOS = [PYANNOTE_VAD_MODEL, PYANNOTE_DIARIZATION, "pyannote/speaker-diarization-community-1"]
SPEAKER_MODEL = os.environ.get("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")   # "local" backend only
INDICBERT_TOKENIZER = "ai4bharat/IndicBERTv2-MLM-only"

# "local" diarization (Silero VAD + ECAPA embeddings + agglomerative clustering)
DIAR_WINDOW_S = 3.0             # embedding window (tuned on 3 labelled Hindi clips: 1, 2 and 3 speakers all correct)
DIAR_HOP_S = 1.5
DIAR_CLUSTER_THRESHOLD = 0.7    # cosine distance at which two windows stop being merged within a file
# fluency gate for the LLM score (task 17): IndicBERTv2 pseudo-perplexity. Calibrated on 11 human reference
# transcripts (9.7-52.5) vs the same words shuffled (397-1512); 150 lies between the two
FLUENCY_MAX_PPL = float(os.environ.get("FLUENCY_MAX_PPL", "150"))
FLUENCY_MIN_ALIGN = float(os.environ.get("FLUENCY_MIN_ALIGN", "0.5"))   # CTC alignment score of the ASR transcript
SPEAKER_MATCH_THRESHOLD = float(os.environ.get("SPEAKER_MATCH_THRESHOLD", "0.6"))   # cosine similarity: same speaker across files (both backends)


def torch_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
