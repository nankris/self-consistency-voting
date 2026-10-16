"""Start infer.py detached from the notebook kernel."""
import os
import subprocess
import sys

os.makedirs("/content/035r3/batches", exist_ok=True)
log = open("/content/infer.log", "a")
env = os.environ.copy()
env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
env["PYTHONUNBUFFERED"] = "1"
proc = subprocess.Popen(
    [sys.executable, "-u", "/content/infer.py"],
    stdout=log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env=env,
)
open("/content/035r3/pid", "w").write(str(proc.pid))
print("LAUNCHED", proc.pid)
