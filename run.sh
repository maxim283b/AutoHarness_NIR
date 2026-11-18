#!/usr/bin/env bash
# Entry point of the AutoSaddler shell.
#   ./run.sh setup               clone AutoSaddler (pinned commit), install it with uv, check Ollama
#   ./run.sh autosaddler [args]  run AutoSaddler on the code agent (args go to harness_lab.run_autosaddler)
#   ./run.sh test                unit tests of harness_lab
set -euo pipefail
cd "$(dirname "$0")"
AS=third_party/AutoSaddler
PYBIN="$AS/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="$AS/.venv/Scripts/python.exe"

case "${1:-}" in
  setup)
    [ -d "$AS/.git" ] || { mkdir -p third_party; git clone https://github.com/microsoft/AutoSaddler.git "$AS"; git -C "$AS" checkout --detach e841543ce6cf6bdf690e65134b12a829ec9f66d1; }
    (cd "$AS" && uv sync --extra dev)
    ollama list | grep -q 'qwen2.5:7b' || ollama pull qwen2.5:7b
    echo "setup done" ;;
  autosaddler)
    shift
    PYTHONPATH="$PWD" "$PYBIN" -m harness_lab.run_autosaddler "$@" ;;
  test)
    PYTHONPATH="$PWD" "$PYBIN" -m pytest harness_lab/tests -q ;;
  *) sed -n 2,5p "$0"; exit 1 ;;
esac
