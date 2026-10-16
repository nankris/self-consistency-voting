"""Stop only the inference process group started by remote_launch.py."""
import os
import signal

pid = int(open("/content/035r3/pid").read().strip())
try:
    os.killpg(pid, signal.SIGTERM)
    print("KILLED", pid)
except ProcessLookupError:
    print("NOT_RUNNING", pid)
