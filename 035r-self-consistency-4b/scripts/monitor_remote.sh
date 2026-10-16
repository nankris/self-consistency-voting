#!/usr/bin/env bash
source /workspace/tools/setup.sh
cd /workspace/work/tig/035r-self-consistency-4b
for attempt in 1 2 3; do
  if colab exec -s tig035r-oct15 -f scripts/monitor.py --timeout 60 > results/monitor_latest.txt 2>&1; then
    cat results/monitor_latest.txt
    exit 0
  fi
  echo "Monitoring reconnect ${attempt}/3"
done
tail -4 results/monitor_latest.txt
exit 1
