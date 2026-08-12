#!/usr/bin/env bash
# Launch the moyo desktop GUI from WSL (used by the Windows Desktop shortcut).
# Safe to call from:  wsl.exe -d Ubuntu -- bash /home/david/moyo/scripts/launch-moyo-gui.sh
set -u

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

LOG="${MOYO_GUI_LOG:-/tmp/moyo-gui-launch.log}"
exec > >(tee -a "$LOG") 2>&1
echo "---- $(date -Iseconds) launching moyo GUI ----"
echo "cwd=$PWD user=$(whoami) DISPLAY=${DISPLAY-} WAYLAND_DISPLAY=${WAYLAND_DISPLAY-}"

# WSLg usually injects DISPLAY=:0; fall back if a bare non-login shell missed it.
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  if [[ -S /tmp/.X11-unix/X0 || -d /mnt/wslg ]]; then
    export DISPLAY=:0
    export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
    echo "Set DISPLAY=:0 for WSLg"
  fi
fi

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export QT_X11_NO_MITSHM="${QT_X11_NO_MITSHM:-1}"

PYTHON="${MOYO_GUI_PYTHON:-$HOME/.pyenv/versions/sente/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "ERROR: No Python found. Expected $HOME/.pyenv/versions/sente/bin/python" >&2
  exit 1
fi

echo "Using $PYTHON"
exec "$PYTHON" -m moyo.gui.app "$@"
