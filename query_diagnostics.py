"""Post-hoc diagnostics: paired bootstrap and conditional random assignment at fixed depth budget."""
import csv
import json
from pathlib import Path
import numpy as np
from study import load,order,save

def main():
    out=Path("results/confirm")
    rows,_,scores,_=load("esci_test")
    decisions=json.loads((out/"decisions.json").read_text())
    n=len(rows)
    d=scores[:,1]-scores[:,0]
    report={}
    with (out/"query_effects.csv").open("w",newline="") as f:
        writer=csv.writer(f);writer.writerow(["query_sha256","ndcg10","ndcg20","pre_deep","post_deep"])
        writer.writerows((order(r["q"]),*s,a,b) for r,s,a,b in zip(rows,scores,decisions["pre"],decisions["post"],strict=True))
    for name in ("pre","post"):
        mask=np.asarray(decisions[name],dtype=bool)
        rng=np.random.default_rng(772)
        # Compare the observed allocation with the expected value over random allocations
        # of the same size. Recompute the budget share inside each query bootstrap sample.
        observed=(d*mask).mean()-d.mean()*mask.mean()
        paired=[]
        for _ in range(3000):
            i=rng.integers(n,size=n)
            paired.append((d[i]*mask[i]).mean()-d[i].mean()*mask[i].mean())
        null=[(d*rng.permutation(mask)).mean()-d.mean()*mask.mean() for _ in range(5000)]
        report[name]=dict(deep_queries=int(mask.sum()),delta_vs_random=float(observed),
                          ci=np.quantile(paired,[.025,.975]).tolist(),
                          permutation_p_one_sided=float((1+np.count_nonzero(np.array(null)>=observed))/5001))
    save(out/"allocation.json",report)
if __name__=="__main__": main()
