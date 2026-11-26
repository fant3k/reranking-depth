"""Prespecified constant-depth comparator and same-trace load measurements."""
import argparse
import json
from pathlib import Path
import statistics
import time
import joblib
import numpy as np
import study

OUT=Path("results/fixed-depth");CONF=Path("results/confirm")
WEIGHTS=1/np.log2(np.arange(2,12))

def read(p):
    return json.loads(p.read_text())

def value(row,scores):
    rank=np.argsort(-np.asarray(scores),kind="stable")[:10]
    return float(WEIGHTS@np.asarray(row["gain"])[rank]/max(WEIGHTS@row["ideal"],1e-12))

def measurements(rows,source):
    scores=[read(study.OUT/(source+"-scores")/(study.order(r["q"])+".json")) for r in rows]
    return np.array([[value(r,s[:n]) for n in range(10,21)] for r,s in zip(rows,scores)])

def summary(values,ref):
    d=values-ref;rng=np.random.default_rng(548)
    bs=d[rng.integers(len(d),size=(2000,len(d)))].mean(1)
    return dict(ndcg=float(values.mean()),difference=float(d.mean()),ci=np.quantile(bs,[.025,.975]).tolist())

def quality():
    rows=read(study.OUT/"esci_test.json")
    y=measurements(rows,"esci_test")
    masks=read(CONF/"decisions.json")
    result={str(n):summary(y[:,n-10],y[:,-1]) for n in range(10,21)}
    routes={}
    for name in ("pre","post"):
        routed=np.where(masks[name],y[:,-1],y[:,0])
        routes[name]=summary(routed,y[:,5])
    infer=study.model()
    actual=np.array([value(r,infer(r["q"],r["titles"][:15])) for r in rows])
    curve_file=study.OUT/"learning/curve.json"
    probe_ids=set(read(curve_file)["eval_hashes"])
    extra=[r for r in read(study.OUT/"amazon_extra.json") if study.order(r["q"]) in probe_ids]
    assert len(extra)==len(probe_ids)
    z=measurements(extra,"amazon_extra")
    study.save(OUT/"quality.json",dict(n=len(rows),official=result,router_minus_fixed15=routes,
        fixed15_actual=summary(actual,y[:,-1]),queries_with_batch_metric_difference=int(np.count_nonzero(actual!=y[:,5])),
        new_probe_n=len(z),new_probe={str(n):summary(z[:,n-10],z[:,-1]) for n in range(10,21)}))

def bench(rep,rate):
    rows=read(study.OUT/"esci_test.json")
    base=read(CONF/f"lat-{rate}-{rep}-fixed20.json")
    model,threshold=joblib.load(CONF/"post.joblib");infer=study.model()
    for row in rows[:6]:infer(row["q"],row["titles"])
    for policy in np.random.default_rng(1920+rep+rate).permutation(["fixed15","fixed20","post"]):
        dest=OUT/f"lat-{rate}-{rep}-{policy}.json"
        if dest.exists():continue
        records=[];start=time.monotonic()
        for i,offset,*_ in base:
            due=start+offset;t=time.monotonic();wait=due-t
            if wait>0:time.sleep(wait)
            t=time.monotonic();r=rows[i];q=r["q"];titles=r["titles"]
            if policy=="post":
                s=infer(q,titles[:10])
                if model.predict([study.features(r,s)])[0]>threshold:s=np.r_[s,infer(q,titles[10:])]
            else:
                s=infer(q,titles[:15] if policy=="fixed15" else titles)
            np.argsort(-s,kind="stable")[:10]
            end=time.monotonic();records.append([i,offset,(end-t)*1000,(end-due)*1000,len(s)])
        study.save(dest,records)

def audit():
    table=[]
    for rep in range(4):
        for rate in (10,26):
            ref=read(CONF/f"lat-{rate}-{rep}-fixed20.json")
            for k in ("fixed15","fixed20","post"):
                v=read(OUT/f"lat-{rate}-{rep}-{k}.json")
                assert len(v)==len(ref)==240 and [r[:2] for r in v]==[r[:2] for r in ref]
                depths=([r[4] for r in read(CONF/f"lat-{rate}-{rep}-post.json")] if k=="post"
                        else [15 if k=="fixed15" else 20]*len(v))
                assert [r[4] for r in v]==depths and all(0<=r[2]<=r[3] for r in v)
                table.append(dict(rep=rep,rate=rate,policy=k,p95_ms=float(np.quantile([r[3] for r in v],.95)),
                    service_p95_ms=float(np.quantile([r[2] for r in v],.95))))
    study.save(OUT/"latency.json",dict(traces=table,n_calls=len(table)*240))
    print("Validated",len(table)*240,"new fixed-depth control measurements")

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("task",choices=["quality","bench","audit"])
    p.add_argument("repetition",type=int,nargs="?");p.add_argument("rate",type=int,nargs="?")
    a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if a.task=="quality":quality()
    elif a.task=="bench":bench(a.repetition,a.rate)
    else:audit()
