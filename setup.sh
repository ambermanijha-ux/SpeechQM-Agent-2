#!/usr/bin/env bash
# One-time setup on Mac or Linux: Python 3.11 environment (.venv) with all packages, and Ollama.
# Usage: ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating .venv (Python 3.11) and installing packages ..."
uv venv -p 3.11 --allow-existing .venv
# auto: the CUDA build of torch on machines with an NVIDIA GPU, CPU build otherwise (Apple GPU works with the default)
uv pip install --python .venv/bin/python --torch-backend=auto -r requirements.txt

# OLLAMA_BASE_URL set = an Ollama server elsewhere (another machine / the host of a container): nothing to install
if [ -z "${OLLAMA_BASE_URL:-}" ] && ! command -v ollama >/dev/null 2>&1 && [ ! -d /Applications/Ollama.app ]; then
  if [ "$(uname)" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then brew install ollama
    else echo "Install Ollama from https://ollama.com/download, open it once, then run ./run.sh"; exit 1; fi
  else
    echo "Installing Ollama ..."
    curl -fsSL https://ollama.com/install.sh | sh
  fi
fi
echo
echo "Setup done. Run the checks with:  ./run.sh /path/to/audio/folder"
