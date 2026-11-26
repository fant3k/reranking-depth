import sys
from pathlib import Path
from pilot_queue import queue
p=sys.executable
tasks=[("prepare",[p,"study.py","prepare","esci_test"],[]),("fit",[p,"confirm.py","fit"],[])]
scores=[]
for i in range(12):
    k=f"score{i}"; tasks.append((k,[p,"study.py","score","esci_test",str(i)],["prepare"]));scores.append(k)
tasks.append(("quality",[p,"confirm.py","quality"],scores+["fit"]))
for rep in range(4):
    for rate in [10,26]:
        tasks.append((f"bench-{rep}-{rate}",[p,"confirm.py","bench",str(rep),str(rate)],["quality"]))
state=queue(tasks,Path("confirm-logs"))
sys.exit(int(any(v["status"]!="ok" for v in state.values())))
