#!/bin/bash
# Stops only r3035 if the hard-stop epoch passes. Quiet unless it acts.
set -u
source /workspace/tools/setup.sh
ROOT=/workspace/work/tig/035r3-self-consistency
LOG=$ROOT/results/watchdog.log
mkdir -p "$ROOT/results"
while true; do
  if [ -f "$ROOT/results/EXPERIMENT_DONE" ]; then
    exit 0
  fi
  if [ -f "$ROOT/results/hard_stop_epoch.txt" ]; then
    hs=$(cat "$ROOT/results/hard_stop_epoch.txt")
    now=$(date +%s)
    if [ "$hs" -gt 0 ] && [ "$now" -ge "$hs" ]; then
      echo "WATCHDOG_STOP r3035 now=$now hard=$hs" | tee -a "$LOG"
      colab stop -s r3035 >>"$LOG" 2>&1 || true
      exit 0
    fi
  fi
  sleep 30
done
