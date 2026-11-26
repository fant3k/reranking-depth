import sys
from pathlib import Path
from pilot_queue import queue
py=sys.executable
tasks=[("prepare",[py,"study.py","prepare","amazon_extra"],[])]
for i in range(12):
    tasks.append((f"score-{i}",[py,"study.py","score","amazon_extra",str(i)],["prepare"]))
tasks.append(("curves",[py,"learning_curve.py"],[f"score-{i}" for i in range(12)]))
status=queue(tasks,Path("learning-logs"))
sys.exit(int(any(v["status"]!="ok" for v in status.values())))
