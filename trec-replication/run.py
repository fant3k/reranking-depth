"""Run from search-pilot with python trec-replication/run.py."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import study

FOLDER = Path(__file__).resolve().parent
DATA = ROOT / "data/trec"
SOURCES = [
    ("product-search-corpus", "2d39a0a369995216d464c32c631633af64770ce0", "corpus.jsonl.gz"),
    ("product-search-2023-qrels", "089d56861cdfc4d4327c40775f23a12af663d4e7", "2023test.qrel"),
    ("product-search-2023-queries", "7335185ccf29cd09fe5148473078a2dfbc74fd36", "2023_test_queries.tsv.gz"),
]
WEIGHTS = 1 / np.log2(np.arange(2, 12))

def fetch():
    DATA.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for repo, rev, name in SOURCES:
        dest = DATA / name
        if not dest.exists():
            url = f"https://huggingface.co/datasets/trec-product-search/{repo}/resolve/{rev}/{name}?download=true"
            tmp = dest.with_suffix(".download")
            subprocess.run(["curl", "-fLsS", "--retry", "3", url, "-o", str(tmp)], check=True)
            tmp.replace(dest)
        with dest.open("rb") as f:
            hashes[name] = hashlib.file_digest(f, "sha256").hexdigest()
    expected = {
        "corpus.jsonl.gz": "cc226c6b4620695da43d902ef1a4e46ca9d93911e8e9e5b530b0b4d5c3aada27",
        "2023test.qrel": "be97b96c0bbd942dcc133ac01f5edd67840dc15fa720c0b62c39f3fe7a3ad30b",
        "2023_test_queries.tsv.gz": "f06fefe7f775a1bdbf8dfa45cd3cf7816555e8677b3178d8783f76c022e266b0",
    }
    if hashes != expected:
        raise ValueError("TREC input checksum mismatch")
    return hashes

def pool(q, items):
    tokens = [study.tokens(title) for _,title,_ in items]
    words = sorted(set(study.tokens(q)))
    lengths = max(1., np.mean([len(w) for w in tokens]))
    df = Counter(t for doc in tokens for t in set(doc).intersection(words))
    scores = []
    for doc in tokens:
        counts = Counter(doc)
        scores.append(sum(math.log1p((len(items)-df[t]+.5)/(df[t]+.5))*counts[t]*2.2/
                          (counts[t]+1.2*(.25+.75*len(doc)/lengths)) for t in words if counts[t]))
    order = sorted(range(len(items)),key=lambda i:(-scores[i],items[i][0]))[:20]
    return dict(q=q,titles=[items[i][1] for i in order],gain=[items[i][2] for i in order],
                bm=[scores[i] for i in order],ideal=sorted((g for _,_,g in items),reverse=True)[:10])

def ndcg(row, logits):
    top = np.argsort(-logits,kind="stable")[:10]
    return float(WEIGHTS @ np.asarray(row["gain"])[top]/max(WEIGHTS @ row["ideal"],1e-12))

def main():
    for entry in (FOLDER/"assets.sha256").read_text().splitlines():
        expected, name = entry.split(maxsplit=1)
        with (ROOT/name).open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest() != expected:
                raise ValueError(f"Frozen asset changed: {name}")
    hashes = fetch()
    labels = defaultdict(dict)
    for line in (DATA/"2023test.qrel").read_text().splitlines():
        qid,_,pid,grade = line.split()
        if int(grade) >= 0:
            if pid in labels[qid] and labels[qid][pid] != int(grade):
                raise ValueError("Conflicting TREC qrels")
            labels[qid][pid] = int(grade)
    needed = set().union(*(v.keys() for v in labels.values()))
    titles = {}
    with gzip.open(DATA/"corpus.jsonl.gz","rt") as f:
        for line in f:
            r = json.loads(line)
            pid = str(r["docid"])
            if pid in needed and r.get("title","").strip():
                titles[pid] = r["title"]
    excluded = set((FOLDER/"excluded-query-hashes.txt").read_text().splitlines())
    seen = set()
    infer = study.model()
    reg, cutoff = joblib.load(ROOT/"results/confirm/post.joblib")
    counts = Counter()
    observations = []
    with gzip.open(DATA/"2023_test_queries.tsv.gz","rt") as f:
        for line in f:
            qid,q = line.strip().split("\t",1)
            counts["topics"] += 1
            if qid not in labels:
                counts["no_judgments"] += 1
                continue
            norm = " ".join(study.tokens(q))
            digest = hashlib.sha256(norm.encode()).hexdigest()
            if digest in excluded or digest in seen:
                counts["overlap_or_duplicate"] += 1
                continue
            items = [(pid,titles[pid],v) for pid,v in labels[qid].items() if pid in titles]
            if len(items)<20:
                counts["short_pool"] += 1
                continue
            seen.add(digest)
            row = pool(q.strip().casefold(),items)
            counts["zero_ideal"] += int(not max(row["ideal"]))
            fixed15 = infer(row["q"],row["titles"][:15])
            fixed20 = infer(row["q"],row["titles"])
            first = infer(row["q"],row["titles"][:10])
            deep = reg.predict([study.features(row,first)])[0]>cutoff
            post = np.r_[first,infer(row["q"],row["titles"][10:])] if deep else first
            observations.append([qid,digest,len(items),int(deep),*[ndcg(row,v) for v in (fixed15,fixed20,post)]])
    if not observations:raise ValueError("No eligible TREC topics")
    with (FOLDER/"per-query.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["qid","normalized_hash","pool","deep","fixed15","fixed20","post"]);w.writerows(observations)
    y = np.array([r[4:] for r in observations])
    boot = np.random.default_rng(51129).integers(len(y),size=(5000,len(y)))
    comparisons = {}
    for idx,base in ((1,"fixed20"),(2,"post")):
        d=y[:,0]-y[:,idx]
        comparisons[base] = dict(delta=float(d.mean()),ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
    result = dict(source_sha256=hashes,counts=counts,n=len(y),judged_ids_missing_title=len(needed-titles.keys()),
                  deep=float(np.mean([r[3] for r in observations])),mean=dict(zip(["fixed15","fixed20","post"],y.mean(0))),
                  fixed15_minus=comparisons,passes_margin=bool(comparisons["fixed20"]["ci"][0]>-.01))
    (FOLDER/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result))
if __name__=="__main__":
    main()
