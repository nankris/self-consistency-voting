"""Machine-readable status. Does not print model answers."""
import os

print("BEGIN_STATUS")
pid_path = "/content/035r3/pid"
pid = open(pid_path).read().strip() if os.path.exists(pid_path) else ""
alive = bool(pid) and os.path.exists(f"/proc/{pid}")
print("PID", pid)
print("ALIVE", alive)
prog = "/content/035r3/progress.json"
print("PROGRESS", open(prog).read().replace("\n", " ") if os.path.exists(prog) else "NONE")
err = "/content/035r3/error.txt"
print("ERROR", "YES" if os.path.exists(err) else "NO")
log = "/content/infer.log"
if os.path.exists(log):
    lines = open(log, errors="replace").read().splitlines()
    # Drop tqdm carriage-return noise; keep the last short lines.
    tail = [ln for ln in lines if ln.strip()][-8:]
    print("LOGTAIL")
    for ln in tail:
        print(ln[:300])
else:
    print("LOGTAIL")
print("END_STATUS")
