"""Plot three paired comparisons of the same two fixed-depth policies."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
root=Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((root/name).read_text())
a=read("results/fixed-depth/quality.json")["fixed15_actual"]
b=read("external-retail/results/summary.json")["paired"]["fixed15-fixed20"]["ndcg"]
c=read("trec-replication/summary.json")["fixed15_minus"]["fixed20"]
fig,ax=plt.subplots(figsize=(7.8,3.0),layout="constrained")
for y,(v,ci) in enumerate([(a["difference"],a["ci"]),(b["mean"],b["ci95"]),(c["delta"],c["ci"])]):
    ax.errorbar(v,y,xerr=[[v-ci[0]],[ci[1]-v]],fmt="o",capsize=4,color="#234d6c")
ax.axvline(-.01,color="#b04123",ls="--",label="Maximum tolerated loss")
ax.axvline(0,color="#333333",lw=.8)
ax.set(yticks=[0,1,2],yticklabels=["ESCI · 1,803","Quissly · 1,187","TREC · 186"],ylim=(2.6,-.6),
       xlabel="Mean nDCG@10 difference, depth 15 minus depth 20 (95% paired interval)")
ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
ax.legend(loc="lower left",frameon=False);ax.grid(axis="x",alpha=.18)
fig.savefig(root/"trec-replication/transfer.png",dpi=160)
