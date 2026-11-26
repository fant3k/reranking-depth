"""Rebuild the operating-point table and validate all paired measurements using stdlib."""
import csv
import json
import math
import statistics
from pathlib import Path
from collections import defaultdict

def percentile(v,p):
    v=sorted(v); t=(len(v)-1)*p; i=int(t)
    return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(t-i)
def audit(folder):
    read=lambda name:json.loads((folder/name).read_text())
    quality=read("quality.json"); routes=read("decisions.json")
    n=quality["n"]; policies=["fixed10","fixed20","split20","pre","post"]
    assert n>0
    for key,mask in routes.items():
        assert len(mask)==n and set(mask)<= {0,1}
        assert abs(statistics.mean(mask)-quality["metrics"][key]["deep"])<1e-10
    for key in ("pre","post"):
        for control in ("_oracle","_shuffle"):
            assert sum(routes[key])==sum(routes[key+control])
        assert quality["metrics"][key+"_oracle"]["quality"]+1e-12>=quality["metrics"][key]["quality"]
    rows=[]
    for rep in range(4):
        for rate in (10,26):
            reference=None
            for policy in policies:
                records=read(f"lat-{rate}-{rep}-{policy}.json")
                trace=[(i,off) for i,off,*rest in records]
                if reference is None: reference=trace
                assert trace==reference and len(set(i for i,_ in trace))==len(trace)==240
                for i,offset,ms,total,count in records:
                    assert math.isfinite(total) and 0<=ms<=total
                    deep=True if policy=="split20" else routes[policy][i]
                    assert count==10+10*deep
                svc=[r[2] for r in records]; total=[r[3] for r in records]
                rows.append(dict(repetition=rep,rate=rate,policy=policy,
                                 service_p95=percentile(svc,.95),response_p95=percentile(total,.95),
                                 response_mean=statistics.mean(total)))
    with (folder/"latency.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    for rate in (10,26):
        print("Offered queries per second:",rate)
        for p in policies:
            v=[row["response_p95"] for row in rows if row["rate"]==rate and row["policy"]==p]
            print(p,round(statistics.median(v),2),"ms; trace range",round(min(v),2),round(max(v),2))
    print("Validated",len(rows)*240,"real CPU calls to the policy (each may use one or two model batches).")
if __name__=="__main__": audit(Path("results/confirm"))
