#!/usr/bin/env bash
# AirPen launcher. Gesture control runs in the background, independent of any window.
#   ./run.sh            start (if needed) and open the preview window
#   ./run.sh start      start in the background only (for autostart)
#   ./run.sh show       open / hide the camera preview window
#   ./run.sh stop       stop gesture control
#   ./run.sh status     is it running? last log lines
#   ./run.sh fg         run in this terminal with the preview open (Ctrl+C to quit)
#   ./run.sh calibrate  fit gesture thresholds to your hand (~1 minute)
#   ./run.sh hand       show the dominant hand; ./run.sh hand left|right sets it (restart to apply)
#   ./run.sh record     record raw landmarks of your hand, step by step (captures/)
#   ./run.sh replay [dir]  check the action recognizer against a recording (newest by default)
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")"
PIDFILE="${XDG_RUNTIME_DIR:-/tmp}/airpen.pid"
LOG="${XDG_STATE_HOME:-$HOME/.local/state}/airpen.log"

setup() {
  if [ ! -x .venv/bin/python ]; then
    uv venv --python 3.12 .venv
    uv pip install --python .venv/bin/python -r requirements.txt
  fi
  if [ ! -f models/hand_landmarker.task ]; then
    mkdir -p models
    local model_tmp
    model_tmp=$(mktemp models/hand_landmarker.task.XXXXXX)
    if curl -fL --connect-timeout 15 --max-time 120 -o "$model_tmp" \
      https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task; then
      mv "$model_tmp" models/hand_landmarker.task
    else
      rm -f "$model_tmp"
      return 1
    fi
  fi
}

# pid of the running AirPen, or fail. The pidfile is not trusted blindly: after a crash the pid may
# belong to an unrelated process, so it must be a positive integer whose command line says airpen.
running_pid() {
  [ -s "$PIDFILE" ] || return 1
  local pid; pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
  case "$pid" in ''|*[!0-9]*|0*) return 1 ;; esac
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q airpen || return 1
  echo "$pid"
}

# run an airpen module with the service stopped, then restart the service; keep the module's exit code
# RESTART=0 leaves the service stopped afterwards (the camera turns off).
run_exclusive() {
  local was_running=0 pid rc=0 i
  if pid=$(running_pid); then
    was_running=1
    kill -TERM "$pid"
    for i in $(seq 1 32); do kill -0 "$pid" 2>/dev/null || break; sleep 0.25; done   # up to ~8 s
  fi
  .venv/bin/python -m "$@" || rc=$?
  if [ "$was_running" = 1 ]; then
    if [ "${RESTART:-1}" = 1 ]; then "$SELF" start || true
    else echo "AirPen was stopped for this and stays off (camera off). ./run.sh start to turn it back on."; fi
  fi
  exit "$rc"
}

cmd="${1:-launch}"; [ $# -gt 0 ] && shift
case "$cmd" in
  launch)
    # start if needed, then show the preview so you can see it's working
    if ! running_pid >/dev/null; then "$SELF" start "$@" || exit 1; fi
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
    run_exclusive airpen.calibrate
    ;;
  fg)
    setup
    exec .venv/bin/python -m airpen --show "$@"
    ;;
  hand)
    # no arg: print the dominant hand. With arg: save it (AirPen must restart to pick it up).
    py=python3; [ -x .venv/bin/python ] && py=.venv/bin/python
    if [ $# -eq 0 ]; then
      $py -c 'from airpen import settings; print(settings.load_hand())'
    else
      if ! hand=$($py -c 'import sys; from airpen import settings; print(settings.save_hand(sys.argv[1]))' "$1"); then
        echo "usage: $0 hand [left|right]"; exit 2
      fi
      echo "dominant hand set to $hand"
      if running_pid >/dev/null; then echo "AirPen is running: ./run.sh stop && ./run.sh start to apply it"; fi
    fi
    ;;
  record)
    # guided capture of raw landmarks. Stops AirPen while recording and leaves it off afterwards.
    setup
    RESTART=0 run_exclusive airpen.record "$@"
    ;;
  replay)
    # offline check of a recording: expected vs got counts per step. Exit 1 on mismatch.
    setup
    exec .venv/bin/python -m airpen.replay "$@"
    ;;
  *)
    echo "usage: $0 [start|stop|show|status|fg|calibrate|hand|record|replay] [options]"
    echo "  hand [left|right]   show or set the dominant hand"
    echo "  record [--hand ..]  record your hand doing each action (for checking accuracy)"
    echo "  replay [dir]        replay a recording through the recognizer"
    exit 2
    ;;
esac
