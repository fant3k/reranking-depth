"""Nested-sample routing study on official and previously unevaluated probes."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import study
from confirm import cheap

OUT=Path("results/learning")
SEEDS=(7,19,41,83,137)

def format_pool(source):
    rows,x,y,_=study.load(source)
    return rows,dict(pre=np.array([cheap(r) for r in rows]),post=x),y

def main():
    OUT.mkdir(exist_ok=True)
    base,bx,by=format_pool("amazon");extra,ex,ey=format_pool("amazon_extra")
    official,fx,fy=format_pool("esci_test")
    key=lambda r:hashlib.sha256(("new-probe-v1:"+r["q"]).encode()).digest()
    order=sorted(range(len(extra)),key=lambda i:key(extra[i]))
    n_test=min(800,len(order)//2);test_i=order[:n_test];train_i=order[n_test:]
    if n_test<100:raise ValueError("Insufficient new evaluation queries")
    normalized=lambda rows:{" ".join(study.tokens(r["q"])) for r in rows}
    assert not (normalized(extra)&(normalized(base)|normalized(official)))
    assert len(normalized(extra))==len(extra)
    assert not set(test_i)&set(train_i)
    X={k:np.vstack((bx[k][200:],ex[k][train_i])) for k in bx}
    y=np.r_[by[200:,1]-by[200:,0],(ey[:,1]-ey[:,0])[train_i]]
    probes={"new_probe":({k:v[test_i] for k,v in ex.items()},ey[test_i]),"official":(fx,fy)}
    sizes=sorted({n for n in (100,400,1600,3200,len(y)) if n<=len(y)})
    records=[];differences={}
    for seed in SEEDS:
        shuffled=np.random.default_rng(seed).permutation(len(y))
        for n in sizes:
            idx=shuffled[:n]
            for name in X:
                reg=RandomForestRegressor(n_estimators=80,min_samples_leaf=8,random_state=seed,n_jobs=4)
                reg.fit(X[name][idx],y[idx])
                c=np.median(reg.predict(bx[name][:200]))
                for probe,(features,scores) in probes.items():
                    mask=reg.predict(features[name])>c
                    quality=np.where(mask,scores[:,1],scores[:,0])
                    change=quality-scores[:,1]
                    key_=(probe,n,name)
                    differences.setdefault(key_,[]).append(change)
                    expected=scores[:,0].mean()+np.mean(scores[:,1]-scores[:,0])*mask.mean()
                    records.append(dict(seed=seed,n_train=n,router=name,probe=probe,n_eval=len(mask),
                        quality=float(quality.mean()),delta_full=float(change.mean()),
                        delta_random=float(quality.mean()-expected),deep=float(mask.mean())))
    summary=[]
    for (probe,n,name),d in differences.items():
        avg=np.mean(d,axis=0);rng=np.random.default_rng(1907)
        boot=avg[rng.integers(len(avg),size=(2000,len(avg)))].mean(1)
        summary.append(dict(probe=probe,n_train=n,router=name,mean_delta_full=float(avg.mean()),
                            ci=np.quantile(boot,[.025,.975]).tolist()))
    study.save(OUT/"curve.json",dict(seeds=SEEDS,additional=len(extra),new_probe=n_test,available_train=len(y),
              eval_hashes=[study.order(extra[i]["q"]) for i in test_i],runs=records,summary=summary))
    print(len(records),"evaluations on",len(y),"available training queries and",n_test,"new held-out queries")
if __name__=="__main__":
    main()
