"""English judged-pool pilot. Run from the study root."""
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

DATA, OUT = Path("data"), Path("results")
SHARDS = 12

def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, allow_nan=False))
    os.replace(tmp, path)

def order(q):
    return hashlib.sha256(q.strip().casefold().encode()).hexdigest()

def tokens(s):
    return re.findall(r"\w+", s.casefold())

def prepare(source):
    if source in {"amazon","esci_test","amazon_extra"}:
        split = "test" if source=="esci_test" else "train"
        ex = (pl.scan_parquet(DATA/"examples.parquet")
              .filter((pl.col("product_locale")=="us") & (pl.col("split")==split) & (pl.col("small_version")==1))
              .select("query","product_id",pl.col("esci_label").alias("label")).collect())
        products = (pl.scan_parquet(DATA/"products.parquet").filter(pl.col("product_locale")=="us")
                    .select("product_id",pl.col("product_title").alias("title")).collect())
    else:
        def read(name):
            return pl.read_csv(DATA/(name+".csv"),separator="\t",infer_schema=False)
        ex = read("label").join(read("query"),on="query_id").select("query","product_id","label")
        products = read("product").select("product_id",pl.col("product_name").alias("title"))
    ex = ex.join(products.unique("product_id"),on="product_id").drop_nulls()
    gains={"E":1., "S":.1, "C":.01, "I":0., "Exact":1., "Partial":.5, "Irrelevant":0.}
    # Multiple recorded labels for the same product are equally weighted once
    # per distinct gain. Group before selecting candidates so ties cannot vary
    # with the parallel join order.
    ex = (ex.with_columns(pl.col("query").str.strip_chars().str.to_lowercase(),
                          pl.col("label").replace_strict(gains,return_dtype=pl.Float64).alias("gain"))
          .group_by(["query","product_id"]).agg(pl.col("title").min(),pl.col("gain").unique().mean()))
    pool = {}
    for query, pid, title, gain in ex.iter_rows():
        pool.setdefault(query.casefold(),[]).append((str(pid), title, gain))
    rows=[]
    exclude=set()
    if source=="esci_test":
        for old in ["amazon","wands"]:
            exclude.update(" ".join(tokens(row["q"])) for row in json.loads((OUT/(old+".json")).read_text()))
    if source=="amazon_extra":
        for old in ["amazon","wands","esci_test"]:
            exclude.update(" ".join(tokens(row["q"])) for row in json.loads((OUT/(old+".json")).read_text()))
    for q in sorted(pool,key=order):
        norm=" ".join(tokens(q))
        if norm in exclude: continue
        docs=sorted(pool[q])
        if len(docs)<20:
            continue
        qw=set(tokens(q))
        words=[tokens(d[1]) for d in docs]
        avg=max(1.,np.mean([len(w) for w in words]))
        freq=Counter(t for w in words for t in set(w) & qw)
        scores=[]
        for w in words:
            tf=Counter(w)
            scores.append(sum(math.log1p((len(words)-freq[t]+.5)/(freq[t]+.5))*tf[t]*2.2/
                              (tf[t]+1.2*(.25+.75*len(w)/avg)) for t in sorted(qw) if tf[t]))
        rank=sorted(range(len(docs)),key=lambda i:(-scores[i],docs[i][0]))[:20]
        ideal=sorted((d[2] for d in docs),reverse=True)[:10]
        rows.append(dict(q=q,titles=[docs[i][1] for i in rank],gain=[docs[i][2] for i in rank],
                         bm=[scores[i] for i in rank],ideal=ideal))
        if source=="amazon_extra":
            exclude.add(norm)
            if len(rows)==4800: break
        if source=="amazon" and len(rows)==1800: break
    if len(rows)<250: raise ValueError(f"Only {len(rows)} eligible {source} queries")
    save(OUT/(source+".json"),rows)
    print(source,len(rows),"eligible sample",flush=True)

def model():
    import onnxruntime as ort
    from tokenizers import Tokenizer
    tok=Tokenizer.from_file(str(DATA/"tokenizer.json"))
    tok.enable_truncation(max_length=96)
    tok.enable_padding(pad_id=0,pad_token="[PAD]")
    opt=ort.SessionOptions()
    opt.intra_op_num_threads=6
    opt.inter_op_num_threads=1
    engine=ort.InferenceSession(str(DATA/"model.onnx"),opt,providers=["CPUExecutionProvider"])
    def score(q,titles):
        seq=tok.encode_batch([(q,t) for t in titles])
        features={"input_ids":[x.ids for x in seq],"attention_mask":[x.attention_mask for x in seq],
                  "token_type_ids":[x.type_ids for x in seq]}
        return engine.run(None,{i.name:np.asarray(features[i.name],dtype=np.int64) for i in engine.get_inputs()})[0].ravel()
    return score

def score_part(source,part):
    records=json.loads((OUT/(source+".json")).read_text())
    dest=OUT/(source+"-scores")
    dest.mkdir(exist_ok=True)
    score=model()
    for r in records[part::SHARDS]:
        path=dest/(order(r["q"])+".json")
        if path.exists():
            s=json.loads(path.read_text())
            if len(s)==20 and np.isfinite(s).all(): continue
        s=score(r["q"],r["titles"])
        if len(s)!=20 or not np.isfinite(s).all(): raise ValueError("invalid logits")
        save(path,s.tolist())
    print(source,part,"complete")

def features(r,s):
    return [len(tokens(r["q"])),r["bm"][0],r["bm"][9],r["bm"][19],max(s[:10]),min(s[:10]),np.std(s[:10])]

def load(source):
    rows=json.loads((OUT/(source+".json")).read_text())
    x=[]; y=[]; changed=[]
    w=1/np.log2(np.arange(2,12))
    for r in rows:
        s=np.array(json.loads((OUT/(source+"-scores")/(order(r["q"])+".json")).read_text()))
        a=np.argsort(-s[:10],kind="stable"); b=np.argsort(-s,kind="stable")[:10]
        denom=max(np.dot(w,r["ideal"]),1e-12)
        gains=np.array(r["gain"])
        x.append(features(r,s)); y.append([w@gains[a]/denom,w@gains[b]/denom])
        changed.append(int(any(b>=10)))
    return rows,np.array(x),np.array(y),np.array(changed)

def evaluate():
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    _,x,y,t=load("amazon")
    rows,v,z,u=load("wands")
    cutoff=200
    test=z[cutoff:]
    predictions={}
    report={}
    def measure(key,mask):
        result=test[np.arange(len(test)),mask.astype(int)]
        delta=result-test[:,1]
        rng=np.random.default_rng(1907)
        boot=delta[rng.integers(len(test),size=(2000,len(test)))].mean(axis=1)
        report[key]=dict(ndcg=float(result.mean()),fraction_deep=float(mask.mean()),
                         delta_vs_full=float(delta.mean()),ci=np.quantile(boot,[.025,.975]).tolist())
        predictions[key]=mask.astype(int).tolist()
    measure("always10",np.zeros(len(test),dtype=bool)); measure("always20",np.ones(len(test),dtype=bool))
    for n in [0,50,200]:
        for label in ["teacher","human"]:
            target=t if label=="teacher" else y[:,1]-y[:,0]
            local=u[:n] if label=="teacher" else z[:n,1]-z[:n,0]
            reg=RandomForestRegressor(n_estimators=80,min_samples_leaf=8,random_state=17,n_jobs=4)
            # Reserve Amazon calibration queries to set the routing budget without test feedback.
            reg.fit(np.vstack([x[200:],v[:n]]),np.r_[target[200:],local])
            threshold=np.median(reg.predict(x[:200]))
            mask=reg.predict(v[cutoff:])>threshold
            key=f"{label}-{n}"
            reg.n_jobs=1
            joblib.dump((reg,threshold),OUT/(key+".joblib"))
            measure(key,mask)
            measure(key+"-shuffled",np.random.default_rng(91).permutation(mask))
    rng=np.random.default_rng(219)
    indices=rng.integers(len(test),size=(2000,len(test)))
    comparisons={}
    for n in [0,50,200]:
        name=f"human-{n}"
        for other in [f"teacher-{n}",name+"-shuffled","always10","always20"]:
            a=test[np.arange(len(test)),predictions[name]]
            b=test[np.arange(len(test)),predictions[other]]
            d=a-b
            comparisons[name+" vs "+other]=dict(delta=float(d.mean()),ci=np.quantile(d[indices].mean(axis=1),[.025,.975]).tolist())
    oracle=test.max(axis=1)
    save(OUT/"routing.json",dict(n_test=len(test),n_amazon=len(x),metrics=report,comparisons=comparisons,
         test_query_ids=[order(r["q"]) for r in rows[cutoff:]],oracle_ndcg=float(oracle.mean()),
         share_deeper_better=float(np.mean(test[:,1]>test[:,0])),share_deeper_worse=float(np.mean(test[:,1]<test[:,0])),
         teacher_changes_top10=float(u[cutoff:].mean())))
    save(OUT/"decisions.json",predictions)

def latency():
    rows=json.loads((OUT/"wands.json").read_text())[200:]
    policies=json.loads((OUT/"decisions.json").read_text())
    score=model()
    for r in rows[:5]: score(r["q"],r["titles"][:10])
    import joblib
    routers={k:joblib.load(OUT/(k+".joblib")) for k in policies if (OUT/(k+".joblib")).exists()}
    times={k:[] for k in policies}
    rng=np.random.default_rng(581)
    for ix in rng.permutation(len(rows))[:100]:
        r=rows[ix]
        for key in rng.permutation(list(policies)):
            start=time.perf_counter()
            first=score(r["q"],r["titles"] if key=="always20" else r["titles"][:10])
            decision=policies[key][ix]
            if key in routers:
                reg,cut=routers[key]
                decision=reg.predict([features(r,first)])[0]>cut
                if decision!=policies[key][ix]: raise ValueError("Online routing mismatch")
            if key!="always20" and decision:
                first=np.r_[first,score(r["q"],r["titles"][10:])]
            np.argsort(-first)
            times[key].append((time.perf_counter()-start)*1000)
    save(OUT/"cpu_replay.json",{k:dict(ms=v,p95=float(np.percentile(v,95)),mean=float(np.mean(v))) for k,v in times.items()})

if __name__=="__main__":
    OUT.mkdir(exist_ok=True)
    job,*args=sys.argv[1:]
    if job=="prepare": prepare(args[0])
    elif job=="score": score_part(args[0],int(args[1]))
    elif job=="evaluate": evaluate()
    elif job=="latency": latency()
    else: raise ValueError(job)
