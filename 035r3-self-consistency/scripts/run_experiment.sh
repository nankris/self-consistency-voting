#!/bin/bash
# Source Colab credentials and run the resumable driver.
set -u
source /workspace/tools/setup.sh
cd /workspace/work/tig/035r3-self-consistency
exec python3 scripts/drive.py
