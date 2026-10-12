"""One-command SpeechQM quality check of an audio folder, on Mac, Windows or Linux.

    python run_qc.py <audio folder> [--language Hindi] [--prompt "..."] [--out <folder>]

1. asks for your Hugging Face token (once; it is saved by huggingface_hub) and checks which gated models
   it can use: without pyannote access speaker checks use the local Silero + ECAPA backend, without ai4bharat
   access ASR uses the Vakyansh model of the language
2. checks Ollama and pulls the LLM if missing, downloads the speech models for the languages found
3. sorts the folder by language (subfolders named after a language, e.g. data/hindi/..., else --language),
   picks up transcript JSONs next to the audio as references
4. runs every QC check (or the checks your --prompt asks for) per language and writes
   <out>/<language>/results/*.csv plus <out>/qc_summary.csv with one row per file
"""
import argparse
import getpass
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Indic text in file contents and console output: make Python use UTF-8 everywhere (Windows defaults to cp1252)
if os.name == "nt" and not sys.flags.utf8_mode and __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv], env={**os.environ, "PYTHONUTF8": "1"}))
os.environ.setdefault("PYTHONUTF8", "1")   # inherited by anything run from here
sys.path.insert(0, str(HERE))

DEFAULT_CHECKS = ("Perform QC checks for corruption and file extension, sample rate check, audio length, "
                  "VAD silence detection, upsampling, number of speakers, speaker duration, verify if speakers are "
                  "new or old, IndicLID language identification, verify language as {language}, CTC score, "
                  "LLM score and check domain")
# these compare against reference transcripts, so they run only where transcript JSONs were found
REFERENCE_CHECKS = ", WER computation, normalization, English word count and utterance duplicate check"
OLLAMA_HELP = {
    "darwin": "install the Ollama app from https://ollama.com/download (or `brew install ollama`), then open it",
    "win32": "install Ollama from https://ollama.com/download (or `winget install Ollama.Ollama`), then start it",
    "linux": "curl -fsSL https://ollama.com/install.sh | sh   (then `ollama serve` if it is not running as a service)",
}


def step(msg):
    print(f"\n=== {msg}", flush=True)


def setup_token(token_arg, ask):
    """Use --token / HF_TOKEN / the saved login, else ask. Returns the user name or None (no token)."""
    from huggingface_hub import get_token, login, whoami
    token = token_arg or os.environ.get("HF_TOKEN") or get_token()
    if not token and ask:
        print("A Hugging Face token gives access to the gated pyannote and ai4bharat models (better speaker and\n"
              "Hindi results). Create one at https://huggingface.co/settings/tokens (type: Read).")
        token = getpass.getpass("Paste your Hugging Face token (input hidden; Enter to skip): ").strip()
    if not token:
        print("No Hugging Face token: using the ungated fallback models.")
        return None
    try:
        user = whoami(token=token)["name"]
    except Exception as e:
        sys.exit(f"That Hugging Face token was rejected ({type(e).__name__}). Check it and run again.")
    if token != get_token():
        login(token=token, add_to_git_credential=False)   # saved for the next runs
    os.environ["HF_TOKEN"] = token
    print(f"Hugging Face: logged in as {user}")
    return user


def check_gated(languages, ask):
    """Pick the speech backend from pyannote access; report ai4bharat models that will use their fallback."""
    from config import ASR_MODELS, PYANNOTE_REPOS
    from download_models import has_access
    while True:
        blocked = [r for r in PYANNOTE_REPOS if not has_access(r)]
        gated_asr = [r for lang in languages for r in ASR_MODELS[lang][:-1] if not has_access(r)]
        if not blocked and not gated_asr:
            return "pyannote"
        print("Your token cannot use these gated models yet:")
        for r in blocked + gated_asr:
            print(f"  https://huggingface.co/{r}")
        print("Open each link while logged in and accept the terms (access is granted instantly).")
        if not ask or input("Press Enter after accepting to re-check, or type 'skip' to use the fallbacks: "
                            ).strip().lower() == "skip":
            break
    if gated_asr:
        print(f"ASR fallback (Vakyansh) instead of: {', '.join(gated_asr)}")
    if blocked:
        print("Speaker checks fallback: Silero VAD + SpeechBrain ECAPA instead of pyannote")
        return "local"
    return "pyannote"


def check_ollama():
    from config import OLLAMA_BASE_URL
    from download_models import ensure_ollama, ollama_models
    try:
        ollama_models()
    except OSError:
        # installed but not running: start it in the background
        if sys.platform == "darwin" and Path("/Applications/Ollama.app").exists():
            subprocess.call(["open", "-a", "Ollama"])
        elif shutil.which("ollama"):
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(20):
            time.sleep(1)
            try:
                ollama_models()
                break
            except OSError:
                pass
        try:
            ollama_models()
        except OSError:
            sys.exit(f"Ollama is not running at {OLLAMA_BASE_URL}. "
                     f"{OLLAMA_HELP.get(sys.platform, OLLAMA_HELP['linux'])}")
    ensure_ollama()


def clear_results(run: Path):
    """Old outputs would be reused (the transcription step skips existing files): keep only the references."""
    for f in (run / "results").glob("*"):
        if f.is_file() and f.name != "indicconf_hypothesis-gt.csv":
            f.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="folder with .wav/.mp3/.flac files (subfolders allowed)")
    ap.add_argument("--language", help="language of audio that is not inside a folder named after a language")
    ap.add_argument("--prompt", help="checks to run, in plain English (default: every check)")
    ap.add_argument("--out", help="where results go (default: <audio folder>_qc next to it)")
    ap.add_argument("--token", help="Hugging Face token (default: saved login, HF_TOKEN, or asked)")
    ap.add_argument("--no-token", action="store_true", help="skip the token and use the ungated fallback models")
    ap.add_argument("--yes", action="store_true", help="never ask; use fallbacks for models you cannot access")
    args = ap.parse_args()
    ask = sys.stdin.isatty() and not args.yes

    src = Path(args.audio).expanduser().resolve()
    if not src.is_dir():
        sys.exit(f"Not a folder: {src}")
    out = Path(args.out).expanduser().resolve() if args.out else src.parent / f"{src.name}_qc"
    os.chdir(HERE)   # pipeline.log lives next to the code; paths above are already absolute

    step("1/5 Sorting audio by language")
    import prepare_runs
    try:
        runs = prepare_runs.main(src, out, args.language)
    except ValueError as e:
        sys.exit(f"{e}\nExample: python run_qc.py \"{src}\" --language Hindi")
    languages = [lang.title() for lang in runs]

    step("2/5 Hugging Face access")
    user = None if args.no_token else setup_token(args.token, ask)
    backend = check_gated(languages, ask) if user else "local"
    import config
    os.environ["SPEECH_BACKEND"] = config.SPEECH_BACKEND = backend   # config is already imported: set both

    step("3/5 Local LLM (Ollama)")
    check_ollama()

    step("4/5 Models (downloaded once, then cached)")
    from config import SPEECH_BACKEND, torch_device
    from download_models import ensure_hf, ensure_indiclid
    ensure_hf(languages, SPEECH_BACKEND)
    ensure_indiclid()
    print(f"Device: {torch_device()}  speech backend: {SPEECH_BACKEND}")

    step("5/5 Quality checks")
    import llm
    from config import LANG_CODES
    codes = {v: k for k, v in LANG_CODES.items()}
    failed = []
    for lang, run in runs.items():
        language = lang.title()
        clear_results(run)
        has_reference = (run / "results" / "indicconf_hypothesis-gt.csv").is_file()
        checks = DEFAULT_CHECKS.format(language=language) + (REFERENCE_CHECKS if has_reference else "")
        prompt = (args.prompt or checks) + f" (language: {language})"
        if not has_reference and not args.prompt:
            print(f"(no reference transcripts for {language}: WER, normalization, English-word and duplicate "
                  "checks skipped)")
        print(f"\n--- {language}: {run / 'audios'}", flush=True)
        t = time.time()
        try:
            llm.main(prompt, audio_dir=str(run / "audios"), lang_code=codes[language])
            print(f"--- {language} done in {time.time() - t:.0f} s", flush=True)
        except Exception as e:
            failed.append(language)
            print(f"--- {language} FAILED: {type(e).__name__}: {e} (details in {HERE / 'pipeline.log'})")

    import summarize
    path = summarize.write_summary(out, runs)
    print(f"\nPer-check CSVs: {out}/<language>/results/\nSummary (one row per file): {path}")
    if failed:
        sys.exit(f"Failed languages: {', '.join(failed)}")


if __name__ == "__main__":
    main()
