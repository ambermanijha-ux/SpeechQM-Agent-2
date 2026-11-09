#!/usr/bin/env bash
# (Re)create the SpeechQM-Agent Python environments with uv from the Linux/CUDA lock files.
# (setup.sh / setup.ps1 stay the portable installers: they resolve requirements*.txt fresh and install Ollama.)
#
# Usage: ./setup_env.sh [ENV ...] [--force] [--prefix DIR]
#   ENV       main      .venv            Python 3.11  requirements.lock.txt
#             nemotron  .venv-nemotron   Python 3.11  requirements-nemotron.lock.txt
#             (none given = both)
#   --force   rebuild an env that already exists (old env renamed <venv>.bak-<time>, not deleted)
#   --prefix  build under DIR instead (test builds)
# Ollama models, HF token, IndicXlit docker image and local model copies: see ../MODELS.md.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SYSTEM_TOOLS=(ffmpeg ollama)
ENVS=(
  "main|$HERE/.venv|3.11|requirements.lock.txt"
  "nemotron|$HERE/.venv-nemotron|3.11|requirements-nemotron.lock.txt"
)
check_main() { "$1/bin/python" -c "
import torch, torchaudio, pyannote.audio, onnxruntime as o, onnx_asr, transformers, langgraph, langchain_ollama, silero_vad, librosa
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| onnxruntime CUDA EP:', 'CUDAExecutionProvider' in o.get_available_providers())"; }
check_nemotron() { "$1/bin/python" -c "
import torch, transformers
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"; }

# ---------------------------------------------------------------- engine (same in every project's setup_env.sh)
PREFIX=""; FORCE=0; SEL=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX=$(realpath -m "$2"); shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) echo "unknown option $1 (see --help)"; exit 2 ;;
    *) SEL+=("$1"); shift ;;
  esac
done
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv ..."; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"
fi
for tool in "${SYSTEM_TOOLS[@]}"; do command -v "$tool" >/dev/null || echo "WARNING: '$tool' not found (sudo apt-get install -y $tool)"; done
known=" "; for row in "${ENVS[@]}"; do known+="${row%%|*} "; done
for s in "${SEL[@]}"; do [[ "$known" == *" $s "* ]] || { echo "unknown env '$s'; choose from:$known"; exit 2; }; done

for row in "${ENVS[@]}"; do
  IFS='|' read -r name venv py req <<<"$row"
  [ ${#SEL[@]} -eq 0 ] || [[ " ${SEL[*]} " == *" $name "* ]] || continue
  [ -n "$PREFIX" ] && venv="$PREFIX/$(basename "$venv")"
  echo; echo "=== [$name] $venv  (Python $py, $req)"
  if declare -F "pre_$name" >/dev/null; then "pre_$name" "$venv"; fi
  if [ -e "$venv" ]; then
    if [ "$FORCE" = 1 ]; then
      bak="$venv.bak-$(date +%Y%m%d-%H%M%S)"; mv "$venv" "$bak"
      echo "existing env moved to $bak — delete it once the new env works"
    else
      echo "$venv already exists. Re-run with --force to rebuild it (the old env is kept as <venv>.bak-<time>)."; exit 1
    fi
  fi
  uv venv -q -p "$py" "$venv"
  uv pip install --python "$venv/bin/python" --no-deps --index-strategy unsafe-best-match -r "$HERE/$req"
  if declare -F "post_$name" >/dev/null; then "post_$name" "$venv"; fi
  echo "--- [$name] check"; "check_$name" "$venv" && echo "[$name] OK: $venv"
done
