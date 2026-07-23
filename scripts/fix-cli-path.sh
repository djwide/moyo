#!/usr/bin/env bash
# Fix stale moyo console scripts in ~/.local/bin that point at system Python.
# Run from the repo root after activating your pyenv env (e.g. pyenv activate sente).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v python &>/dev/null; then
  echo "error: no 'python' on PATH — activate pyenv first, e.g.: pyenv activate sente"
  exit 1
fi

PYTHON="$(command -v python)"
echo "Using Python: $PYTHON"
"$PYTHON" -c "import sys; print('  executable:', sys.executable)"

if ! "$PYTHON" -c "import moyo" 2>/dev/null; then
  echo "Installing moyo (editable) into this environment..."
  "$PYTHON" -m pip install -U pip setuptools wheel
  "$PYTHON" -m pip install -e .
else
  echo "Reinstalling console scripts..."
  "$PYTHON" -m pip install -e . --force-reinstall --no-deps
  "$PYTHON" -m pip install -e .
fi

STALE_DIR="${HOME}/.local/bin"
STALE=(moyo moyo-datainput moyo-corpus moyo-gather moyo-probe moyo-redteam moyo-gui)
FOUND=()
for name in "${STALE[@]}"; do
  if [[ -x "${STALE_DIR}/${name}" ]]; then
    FOUND+=("${STALE_DIR}/${name}")
  fi
done

if ((${#FOUND[@]} > 0)); then
  echo ""
  echo "Stale scripts found in ${STALE_DIR} (often use /usr/bin/python3):"
  for f in "${FOUND[@]}"; do
    echo "  $f -> $(head -1 "$f")"
  done
  echo ""
  if rm -f "${FOUND[@]}" 2>/dev/null; then
    echo "Removed stale scripts from ${STALE_DIR}."
  else
    echo "Could not remove (permission denied?). Run:"
    echo "  sudo rm -f ${FOUND[*]}"
  fi
fi

if command -v pyenv &>/dev/null; then
  pyenv rehash 2>/dev/null || true
fi

echo ""
echo "Verify:"
echo "  which moyo-datainput    # should NOT be ${STALE_DIR}/moyo-datainput"
"$PYTHON" -m pip show moyo | sed -n '1,5p'
echo ""
echo "Try:"
echo "  moyo-datainput process \"secret text\""
echo "  # or if PATH is still wrong:"
echo "  $PYTHON -m moyo.privateside.datainput.cli process \"secret text\""
