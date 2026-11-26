"""Compare a clean reconstruction with the released primary run."""
import argparse
import csv
import json
from pathlib import Path
import statistics

def read(root,key):
    return json.loads((root/key).read_text())

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("repeat",type=Path,help="working directory of the separate reconstruction")
    args=cli.parse_args()
    first=Path("results/confirm")
    repeat=args.repeat/"results/confirm"
    groups={}
    for filename in ("quality.json","decisions.json","pool_audit.json","batch_parity.json","allocation.json"):
        groups[filename]=read(first,filename)==read(repeat,filename)
    p1=read(Path("results"),"routing.json");p2=read(args.repeat/"results","routing.json")
    a=p1["metrics"];b=p2["metrics"]
    maxdiff=max(abs(a[k]["ndcg"]-b[k]["ndcg"]) for k in a)
    for name in ("pre","post","fixed20","fixed10","split20"):
        for rep in range(4):
            for rate in (10,26):
                key=f"lat-{rate}-{rep}-{name}.json"
                x=read(first,key);y=read(repeat,key)
                if [(v[0],v[1],v[4]) for v in x]!=[(v[0],v[1],v[4]) for v in y]:
                    raise ValueError("Input trace or routing mismatch in "+key)
    def table(folder):
        with (folder/"latency.csv").open() as f: rows=list(csv.DictReader(f))
        return {str(rate):{policy:statistics.median(float(r["response_p95"]) for r in rows
                  if int(r["rate"])==rate and r["policy"]==policy)
                  for policy in ("fixed10","fixed20","split20","pre","post")} for rate in (10,26)}
    result=dict(equality=groups,pilot_max_abs_ndcg_difference=maxdiff,identical_trace_inputs=True,
                primary_median_p95_ms=table(first),repeat_median_p95_ms=table(repeat))
    (first/"independent_reproduction.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
    if not all(groups.values()) or maxdiff>1e-10:
        raise SystemExit("Investigate the archived differences.")
if __name__=="__main__":
    main()
