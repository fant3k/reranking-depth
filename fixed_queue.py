from pathlib import Path
import sys
from pilot_queue import queue
p=sys.executable
jobs=[("quality",[p,"fixed_depth.py","quality"],[])]
for rep in range(4):
    for rate in (10,26):
        jobs.append((f"load-{rep}-{rate}",[p,"fixed_depth.py","bench",str(rep),str(rate)],[]))
jobs.append(("audit",[p,"fixed_depth.py","audit"],[k for k,_,_ in jobs if k.startswith("load-")]))
s=queue(jobs,Path("fixed-logs"))
sys.exit(int(any(v["status"]!="ok" for v in s.values())))
