"""Confirmatory query holdout and open-loop CPU replay; run from the study root."""
import json
import sys
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import study

OUT=Path("results/confirm")
OUT.mkdir(exist_ok=True,parents=True)
POLICIES=("fixed10","fixed20","split20","pre","post")
def cheap(row):
    b=row["bm"]
    return [len(study.tokens(row["q"])), b[0],b[9],b[19]]
def fit():
    rows,x,y,_=study.load("amazon")
    for key,v in [("pre",np.array([cheap(r) for r in rows])),("post",x)]:
        m=RandomForestRegressor(n_estimators=80,min_samples_leaf=8,random_state=17,n_jobs=4)
        m.fit(v[200:],(y[:,1]-y[:,0])[200:])
        cutoff=np.median(m.predict(v[:200]))
        m.n_jobs=1
        joblib.dump((m,cutoff),OUT/(key+".joblib"))
def quality():
    rows,x,y,_=study.load("esci_test")
    rng=np.random.default_rng(2723)
    boot=rng.integers(len(rows),size=(2000,len(rows)))
    d=y[:,1]-y[:,0]
    res={}; decisions={}
    def add(key,a):
        v=np.where(a,y[:,1],y[:,0])
        loss=v-y[:,1]
        decisions[key]=a.astype(int).tolist()
        res[key]={"quality":v.mean(),"delta_full":loss.mean(),"ci":np.quantile(loss[boot].mean(1),[.025,.975]).tolist(),"deep":a.mean()}
    add("fixed20",np.ones(len(y),bool)); add("fixed10",np.zeros(len(y),bool))
    for key,v in [("pre",np.array([cheap(r) for r in rows])),("post",x)]:
        m,c=joblib.load(OUT/(key+".joblib"))
        a=m.predict(v)>c
        add(key,a); add(key+"_shuffle",rng.permutation(a))
        oracle=np.zeros(len(a),bool)
        oracle[np.argsort(-d,kind="stable")[:a.sum()]]=True
        add(key+"_oracle",oracle)
    study.save(OUT/"quality.json",{"n":len(rows),"metrics":res})
    study.save(OUT/"decisions.json",decisions)
def bench(rep,rate):
    rows=json.loads((study.OUT/"esci_test.json").read_text())
    rng=np.random.default_rng(5300+rep)
    ids=rng.choice(len(rows),size=240,replace=False)
    arrivals=np.cumsum(rng.exponential(1/rate,len(ids)))
    arrivals-=arrivals[0]
    models={k:joblib.load(OUT/(k+".joblib")) for k in ["pre","post"]}
    scorer=study.model()
    for r in rows[:6]: scorer(r["q"],r["titles"])
    for k in rng.permutation(POLICIES):
        dest=OUT/f"lat-{rate}-{rep}-{k}.json"
        if dest.exists(): continue
        records=[]
        start=time.monotonic()
        for i,offset in zip(ids,arrivals):
            due=start+offset
            wait=due-time.monotonic()
            if wait>0: time.sleep(wait)
            t=time.monotonic()
            r=rows[i]; titles=r["titles"]; q=r["q"]
            if k=="pre":
                m,c=models[k]; deep=m.predict([cheap(r)])[0]>c
                s=scorer(q,titles if deep else titles[:10])
            elif k in ("fixed10","fixed20"):
                s=scorer(q,titles if k=="fixed20" else titles[:10])
            else:
                s=scorer(q,titles[:10]); deep=True
                if k=="post":
                    m,c=models[k]; deep=m.predict([study.features(r,s)])[0]>c
                if deep: s=np.r_[s,scorer(q,titles[10:])]
            np.argsort(-s,kind="stable")[:10]
            end=time.monotonic()
            records.append([int(i),float(offset),(end-t)*1000,(end-due)*1000,len(s)])
        study.save(dest,records)
if __name__=="__main__":
    job=sys.argv[1]
    if job=="fit": fit()
    elif job=="quality": quality()
    elif job=="bench": bench(*map(int,sys.argv[2:]))
