#!/usr/bin/env bash
# AirPen launcher. Gesture control runs in the background, independent of any window.
#   ./run.sh            start (if needed) and open the preview window
#   ./run.sh start      start in the background only (for autostart)
#   ./run.sh show       open / hide the camera preview window
#   ./run.sh stop       stop gesture control
#   ./run.sh status     is it running? last log lines
#   ./run.sh fg         run in this terminal with the preview open (Ctrl+C to quit)
#   ./run.sh calibrate  fit gesture thresholds to your hand (~1 minute)
set -euo pipefail
cd "$(dirname "$0")"
PIDFILE="${XDG_RUNTIME_DIR:-/tmp}/airpen.pid"
LOG="${XDG_STATE_HOME:-$HOME/.local/state}/airpen.log"

setup() {
  if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
  fi
  if [ ! -f models/hand_landmarker.task ]; then
    mkdir -p models
    curl -fL -o models/hand_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
  fi
}

running_pid() {
  [ -s "$PIDFILE" ] || return 1
  local pid; pid=$(cat "$PIDFILE")
  kill -0 "$pid" 2>/dev/null && echo "$pid"
}

cmd="${1:-launch}"; [ $# -gt 0 ] && shift
case "$cmd" in
  launch)
    # start if needed, then show the preview so you can see it's working
    if ! running_pid >/dev/null; then "$0" start "$@" || exit 1; fi
    kill -USR2 "$(running_pid)"
    echo "Preview window opened. Closing it only hides it; ./run.sh stop to quit."
    ;;
  start)
    if pid=$(running_pid); then echo "AirPen already running (pid $pid)"; exit 0; fi
    setup
    mkdir -p "$(dirname "$LOG")"
    nohup setsid .venv/bin/python -u -m airpen "$@" >"$LOG" 2>&1 </dev/null &
    sleep 3
    if pid=$(running_pid); then echo "AirPen running in the background (pid $pid). Log: $LOG"
    else echo "AirPen failed to start. Log:"; tail -n 20 "$LOG"; exit 1; fi
    ;;
  stop)
    if pid=$(running_pid); then kill -TERM "$pid"; echo "AirPen stopped"; else echo "AirPen is not running"; fi
    ;;
  show)
    if pid=$(running_pid); then kill -USR1 "$pid"; else echo "AirPen is not running (./run.sh start)"; exit 1; fi
    ;;
  status)
    if pid=$(running_pid); then echo "running (pid $pid)"; else echo "not running"; fi
    [ -f "$LOG" ] && tail -n 5 "$LOG" | grep -v '^[WI]0000' || true
    ;;
  calibrate)
    setup
    was_running=0
    if pid=$(running_pid); then was_running=1; kill -TERM "$pid"; sleep 1.5; fi
    .venv/bin/python -m airpen.calibrate || true
    if [ "$was_running" = 1 ]; then exec "$0" start; fi
    ;;
  fg)
    setup
    exec .venv/bin/python -m airpen --show "$@"
    ;;
  *)
    echo "usage: $0 [start|stop|show|status|fg|calibrate] [airpen options]"; exit 2
    ;;
esac
