"""Build the English report and plot from audited measurements."""
import csv
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("results/confirm")
quality = json.loads((root/"quality.json").read_text())
parity = json.loads((root/"batch_parity.json").read_text())
pools = json.loads((root/"pool_audit.json").read_text())
alloc = json.loads((root/"allocation.json").read_text())
with (root/"latency.csv").open() as f:
    timings = list(csv.DictReader(f))
policies = ["fixed10", "fixed20", "split20", "pre", "post"]
def ms(policy, rate, field="response_p95"):
    return [float(r[field]) for r in timings if r["policy"] == policy and int(r["rate"]) == rate]
def typical(policy, rate):
    return statistics.median(ms(policy, rate))
lines = [
    "# Can a search engine learn when to score ten more products?",
    "",
    f"On {quality['n']:,} held-out Amazon queries, scoring twenty candidates in one batch achieved the highest measured quality among deployable policies. Both learned routers sent about half the queries to depth twenty. Neither met the predeclared quality requirement.",
    "",
    "## Quality on the official ESCI test split",
    "",
    "| Policy | Mean nDCG@10 | Difference from fixed 20 | Paired 95% interval | Share sent to depth 20 |",
    "|---|---:|---:|---:|---:|",
]
for k, info in quality["metrics"].items():
    lo, hi = info["ci"]
    lines.append(f"| {k} | {info['quality']:.4f} | {info['delta_full']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {info['deep']:.1%} |")
lines += ["", "The target required the lower interval bound to exceed −0.01. Each shuffle is one random allocation at the corresponding router’s pair budget. Each oracle selects queries retrospectively using known relevance labels. A deeper rerank can displace a relevant item, so the oracle can outperform fixed 20.",
          "", "## Does the router pick the right queries?", ""]
for k,v in alloc.items():
    lo,hi = v["ci"]
    lines.append(f"* {k} gained {v['delta_vs_random']:+.4f} nDCG over the expected value of a uniformly random allocation at the same pair budget (paired bootstrap 95% interval [{lo:+.4f}, {hi:+.4f}], one-sided permutation p = {v['permutation_p_one_sided']:.3f}).")
lines += ["", "These exploratory diagnostics measure the covariance between the decision to spend more and the resulting change in quality. They do not equate the CPU time of two different batching strategies.",
          "", "## Measured CPU response time", "",
          "Median p95 across four traces, in milliseconds. Brackets show the range across traces. Each trace has 240 queries, identical arrival offsets and query order for competing policies. Response time includes queueing. Separate service times are in [the trace summary](results/confirm/latency.csv).", "",
          "| Policy | 10 queries/s | 26 queries/s |", "|---|---:|---:|"]
for k in policies:
    fields = []
    for rate in (10, 26):
        v = ms(k,rate)
        fields.append(f"{statistics.median(v):.1f} [{min(v):.1f}, {max(v):.1f}]")
    lines.append("| "+ " | ".join([k]+fields)+" |")
lines += ["", "![Measured search quality and CPU latency](results/confirm/frontier.png)", "",
          "The pre router chooses a single batch before inference. The post router can issue a second batch after seeing the first ten scores. Split 20 always issues both batches. The fixed baselines use one model call. Measurements used an Intel Core i5-12400 on Ubuntu 26.04.1, six ONNX Runtime intra-op threads and one inter-op thread. Tail times depend on the arrival process, batch shape and CPU.",
          "", "## Interpretation and scope", "",
          "For a deployment requiring the declared quality margin, the original adaptive policies did not qualify. The same-budget fixed-depth follow-up below evaluates another deployment option. The label-informed upper bound suggests scope for a better allocation rule, but supplies no deployable way to identify those queries. The Wayfair pilot is exploratory. The ESCI test protocol was frozen separately. Both datasets use judged pools and title-only candidates. Training and official test queries are disjoint under the stated normalization; shared products and semantically similar queries can still occur. Latency excludes retrieval, networking and production contention. Four synthetic traces on one machine characterize this implementation only.",
          "", "[Frozen protocol](PROTOCOL.md) · [Per-query audit](results/confirm/query_effects.csv) · [Diagnostic estimates](results/confirm/allocation.json)"]
lines += ["", "## Implementation checks", "",
          f"The independent batch-shape replay scored all {parity['n']:,} queries with separate batches of ten. It changed {parity['post_decision_flips']} post-router decisions. Actual mean nDCG@10 was " + ", ".join(f"{k} {v['quality']:.5f}" for k,v in parity["metrics"].items()) + ".",
          "", f"Checking the original data found {pools['test_normalized_overlap']} normalized test queries shared with the training or pilot samples. The full [pool audit](results/confirm/pool_audit.json) also counts zero-gain queries, conflicting labels and spelling variants.",
          "", "## Earlier transfer pilot", ""]
pilot = json.loads(Path("results/routing.json").read_text())
pilot_cpu = json.loads(Path("results/cpu_replay.json").read_text())
for k in ("always20","human-200"):
    lines.append(f"* {k} on {pilot['n_test']} held-out WANDS queries: nDCG@10 {pilot['metrics'][k]['ndcg']:.4f}, CPU service p95 {pilot_cpu[k]['p95']:.1f} ms.")
lines += ["", "The transfer pilot used an unloaded replay and adapted the human-label router on 200 separate WANDS queries. Its service p95 and the later queue-inclusive response p95 describe different experiments."]
lines += ["", f"The original WANDS labels contain {pools['wands']['conflicting_labels']} conflicting query–product pairs. The reproducible rule averages distinct gains per pair."]
history=root/"label_resolution.json"
if history.exists():
    h=json.loads(history.read_text())
    lines += ["",f"The change affected {h['changed_queries']} eligible queries. Fixed 20 moved from {h['original']['always20']:.5f} to {h['resolved']['always20']:.5f}. The adapted router moved from {h['original']['human-200']:.5f} to {h['resolved']['human-200']:.5f}."]
lines += ["", f"At 10 queries/s the post router cut median trace p95 by {1-typical('post',10)/typical('fixed20',10):.1%}; at 26 queries/s the reduction was {1-typical('post',26)/typical('fixed20',26):.1%}. Queueing magnifies the difference at high load. These speedups alone do not satisfy the joint quality and latency criterion."]
repeat=root/"independent_reproduction.json"
if repeat.exists():
    v=json.loads(repeat.read_text())
    n=sum(v["equality"].values())
    lines+=["","## Independent reconstruction on Ubuntu","",f"A separate working directory rebuilt the datasets, model outputs, routing decisions and all 9,600 timing calls from verified source files. {n} of {len(v['equality'])} quality and diagnostic artifacts were numerically identical. The maximum absolute pilot nDCG difference was {v['pilot_max_abs_ndcg_difference']:.2g}. The trace inputs and route depths matched. Clock measurements can vary.","",
            "| Queries/s | Primary fixed 20 / post p95 (ms) | Repeat fixed 20 / post p95 (ms) |","|---|---:|---:|"]
    for rate in ("10","26"):
        a=v["primary_median_p95_ms"][rate];b=v["repeat_median_p95_ms"][rate]
        lines.append(f"| {rate} | {a['fixed20']:.1f} / {a['post']:.1f} | {b['fixed20']:.1f} / {b['post']:.1f} |")
curve=Path("results/learning/curve.json")
if curve.exists():
    data=json.loads(curve.read_text());rr=data["runs"];ss=data["summary"]
    lines+=["","## Does the router need more training labels?","",
        f"The follow-up scored {data['additional']:,} additional query pools and reserved {data['new_probe']:,} previously unevaluated queries from the official training partition. Each of five seeds sampled nested training sets. All models used the same 200 calibration queries. The table reports the five-seed mean difference from fixed 20 for every training size. This follow-up was designed after opening the official test. [Protocol](LABEL_EFFICIENCY.md) · [All runs and query-bootstrap intervals](results/learning/curve.json)", "",
        "| Labeled fitting queries | New probe, pre | New probe, post | Official, pre | Official, post |","|---:|---:|---:|---:|---:|"]
    sizes=sorted({r["n_train"] for r in rr})
    for size in sizes:
        cells=[]
        for probe in ("new_probe","official"):
            for name in ("pre","post"):
                row=next(q for q in ss if q["probe"]==probe and q["n_train"]==size and q["router"]==name)
                cells.append(f"{row['mean_delta_full']:+.4f}")
        lines.append("| "+str(size)+" | "+" | ".join(cells)+" |")
    fig3,ax3=plt.subplots(figsize=(6,3.5),layout="constrained")
    for probe in ("new_probe","official"):
        for name in ("pre","post"):
            pts=sorted([r for r in ss if r["probe"]==probe and r["router"]==name],key=lambda r:r["n_train"])
            ax3.plot(sizes,[q["mean_delta_full"] for q in pts],marker="o",label=probe+" "+name)
    ax3.axhline(-.01,linestyle="--",color="gray");ax3.set_xscale("log")
    ax3.set(xlabel="Labeled fitting queries (log scale)",ylabel="Mean nDCG difference from fixed 20")
    ax3.grid(alpha=.18);ax3.legend(fontsize=8)
    fig3.savefig(curve.parent/"learning_curve.png",dpi=160);plt.close(fig3)
    lines+=["","![Router quality across training label budgets](results/learning/learning_curve.png)"]
fixed=Path("results/fixed-depth")
if (fixed/"latency.json").exists() and (fixed/"quality.json").exists():
    q=json.loads((fixed/"quality.json").read_text())
    raw=json.loads((fixed/"latency.json").read_text())
    m=q["fixed15_actual"]
    lines+=["","## A fixed-depth comparator at approximately the same pair budget","",
            f"Fixed 15 scored {m['ndcg']:.4f} nDCG@10. Its difference from fixed 20 was {m['difference']:+.4f} (95% paired interval [{m['ci'][0]:+.4f}, {m['ci'][1]:+.4f}]). Independent depth-15 inference found {q['queries_with_batch_metric_difference']} queries with different nDCG from the prefix scores. On the additional {q['new_probe_n']}-query probe the fixed-15 difference from fixed 20 was {q['new_probe']['15']['difference']:+.4f}.",
            "", "| Policy | Median response p95 at 10 queries/s (ms) | At 26 queries/s (ms) |","|---|---:|---:|"]
    med={}
    for k in ("fixed15","fixed20","post"):
        med[k]=[statistics.median(r["p95_ms"] for r in raw["traces"] if r["policy"]==k and r["rate"]==rate) for rate in (10,26)]
        lines.append(f"| {k} | {med[k][0]:.1f} | {med[k][1]:.1f} |")
    lines[4:4]=["",f"In the subsequent same-budget comparison, a constant depth of 15 achieved {m['ndcg']:.4f} nDCG@10 and had lower p95 than the learned post router at both loads ({med['fixed15'][0]:.1f} versus {med['post'][0]:.1f} ms at 10 queries/s; {med['fixed15'][1]:.1f} versus {med['post'][1]:.1f} ms at 26 queries/s). Within this setup, constant depth 15 is the supported operating choice. This control was specified after seeing the original results.",""]
    dd=q["router_minus_fixed15"]["post"]
    speed=[1-a/b for a,b in zip(med["fixed15"],med["fixed20"])]
    quality_pass=m["ci"][0]>-.01
    lines+=["",f"The post router minus fixed 15 was {dd['difference']:+.4f} nDCG (95% paired interval [{dd['ci'][0]:+.4f}, {dd['ci'][1]:+.4f}]). Constant depth 15 {'met' if quality_pass else 'missed'} the quality margin. Its latency reductions versus fixed 20 were {speed[0]:.1%} and {speed[1]:.1%}, respectively. This follow-up used {raw['n_calls']:,} further timed requests and was designed after the original test. [Prespecified comparator](FIXED_DEPTH.md) · [Measured traces](results/fixed-depth/latency.json)"]
    fig4,ax4=plt.subplots(figsize=(5.8,3.2),layout="constrained")
    depths=list(range(10,21))
    for label,key in [("Official ESCI","official"),("Additional probe","new_probe")]:
        ax4.plot(depths,[q[key][str(n)]["difference"] for n in depths],marker="o",label=label)
    ax4.axhline(-.01,color="gray",linestyle="--");ax4.set(xlabel="Constant candidate depth",ylabel="Mean nDCG difference from fixed 20",xticks=depths);ax4.legend();ax4.grid(alpha=.2)
    fig4.savefig(fixed/"depth_curve.png",dpi=160);plt.close(fig4)
    lines+=["","![Constant reranking depth versus quality](results/fixed-depth/depth_curve.png)"]
if Path("EXTERNAL_VALIDATION.md").exists():
    lines[1:1]=["","**External validation.** Depth 15 exceeded the allowed quality loss on Quissly; TREC left the margin unresolved. [Results on 1,373 additional queries](EXTERNAL_VALIDATION.md)"]
Path("REPORT.md").write_text("\n".join(lines)+"\n")
fig, axes = plt.subplots(1,2,figsize=(8.4,3.7),layout="constrained",sharey=True)
for ax,rate in zip(axes,(10,26)):
    for k,marker in zip(policies,["o","s","^","D","P"]):
        q=parity["metrics"][k]["quality"]
        v=ms(k,rate); m=statistics.median(v)
        ax.errorbar(m,q,xerr=[[m-min(v)],[max(v)-m]],fmt=marker,capsize=3,label=k)
    ax.set(title=f"{rate} queries per second",xlabel="CPU response p95 (ms)")
    ax.grid(alpha=.18)
axes[0].set_ylabel("Mean nDCG@10")
axes[1].legend(fontsize=8)
fig.savefig(root/"frontier.png",dpi=160)
