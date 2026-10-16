"""Local independent guard: stop only this experiment session at its deadline."""
import datetime,time,subprocess,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
deadline=datetime.datetime(2026,10,15,9,49,tzinfo=datetime.timezone.utc).timestamp()
while time.time()<deadline:
 if (root/'results/session_stopped').exists():raise SystemExit(0)
 time.sleep(min(30,max(0,deadline-time.time())))
subprocess.run(['/workspace/tools/bin/colab','stop','-s','tig035r-oct15'])
(root/'results/budget_guard_triggered').write_text('Stopped at conservative 3-hour deadline.\n')
