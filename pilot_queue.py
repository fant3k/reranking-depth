"""Durable serial queue. Retry after peers, checkpoint success, expose terminal errors."""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

def queue(tasks, folder):
    folder.mkdir(exist_ok=True)
    with (folder/"lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=folder/"state.json"
        data=json.loads(state.read_text()) if state.exists() else {}
        def update():
            tmp=state.with_suffix(".tmp")
            tmp.write_text(json.dumps(data,indent=2))
            os.replace(tmp,state)
        for sweep in range(3):
            for key,argv,deps in tasks:
                if data.get(key,{}).get("status")=="ok": continue
                if any(data.get(dep,{}).get("status")!="ok" for dep in deps): continue
                item=data.setdefault(key,{"attempts":0})
                if item["attempts"]>=3 and item.get("status")=="failed": continue
                item["status"]="running"; update()
                try:
                    with (folder/(key+".log")).open("a") as log:
                        rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=7200).returncode
                except (OSError,subprocess.TimeoutExpired) as e:
                    rc=1; item["error"]=str(e)
                item.update(attempts=item["attempts"]+1,status="ok" if rc==0 else "failed",exit=rc)
                update()
        for key,_,deps in tasks:
            if key not in data:
                data[key]={"status":"blocked","attempts":0,"dependencies":deps}
        update()
        return data

def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); flag=root/"seen"
        py=lambda s:[sys.executable,"-c",s]
        tasks=[("transient",py(f"from pathlib import Path;p=Path({str(flag)!r}); seen=p.exists();p.touch();raise SystemExit(0 if seen else 1)"),[]),
               ("independent",py("pass"),[]),("broken",py("raise SystemExit(1)"),[]),
               ("after",py("pass"),["transient"]),("blocked",py("pass"),["broken"])]
        d=queue(tasks,root)
        assert d["transient"]["attempts"]==2 and d["independent"]["attempts"]==1
        assert d["broken"]["attempts"]==3 and d["blocked"]["status"]=="blocked" and d["after"]["status"]=="ok"
        assert queue(tasks,root)==d
        print("PASS transient retry, persistent failure, isolated dependency, continuation and restart.")

if __name__=="__main__":
    if sys.argv[1:] == ["--selftest"]: selftest(); sys.exit()
    py=[sys.executable,"-u","study.py"]
    tasks=[]; scores=[]
    for source in ["amazon","wands"]:
        key="prepare-"+source
        tasks.append((key,py+["prepare",source],[]))
    for i in range(12):
        for source in ["amazon","wands"]:
            key=f"{source}-{i:02d}"
            tasks.append((key,py+["score",source,str(i)],["prepare-"+source]))
            scores.append(key)
    tasks.extend([("evaluate",py+["evaluate"],scores),("latency",py+["latency"],["evaluate"])])
    result=queue(tasks,Path("logs"))
    sys.exit(int(any(item["status"]!="ok" for item in result.values())))
