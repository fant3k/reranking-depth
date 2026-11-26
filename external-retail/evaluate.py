"""Run from the search-pilot root. All new outputs live in external-retail."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import study

POLICIES = ("fixed15", "fixed20", "post")
GAINS = {"exact": 1., "substitute": .1, "complementary": .01, "complement": .01, "irrelevant": 0.}
W = 1 / np.log2(np.arange(2, 12))


def pool(entry, audit):
    docs = {}
    conflicts = set()
    for provider in entry["providers"].values():
        for hit in provider["hits"]:
            audit["hits"] += 1
            pid = str(hit.get("id") or "").lower().replace("-", "")
            title = (hit.get("title") or "").strip()
            gain = GAINS.get(str(hit.get("label")).lower())
            if hit.get("failed") or hit.get("reasoning") == "judge failed after retries" or gain is None:
                audit["unjudged_hits"] += 1
                continue
            if not pid or not title:
                audit["missing_metadata_hits"] += 1
                continue
            if hit["score"] != gain:
                raise ValueError("score disagrees with label")
            if pid in docs:
                old_title, old_gain = docs[pid]
                if old_gain != gain:
                    conflicts.add(pid)
                title = min(title, old_title)
            docs[pid] = title, gain
    for pid in conflicts:
        del docs[pid]
    audit["conflicting_pairs"] += len(conflicts)
    audit["valid_pairs"] += len(docs)
    return docs


def prepare(entry, docs):
    query = entry["text_query"].strip().casefold()
    terms = sorted(set(study.tokens(query)))
    words = {pid: Counter(study.tokens(t)) for pid, (t, _) in docs.items()}
    avg = max(1., sum(sum(tf.values()) for tf in words.values()) / len(words))
    df = Counter(t for tf in words.values() for t in tf)
    scores = {}
    for pid, tf in words.items():
        norm = 1.2 * (.25 + .75 * sum(tf.values()) / avg)
        scores[pid] = sum(math.log1p((len(words) - df[t] + .5) / (df[t] + .5)) *
                          tf[t] * 2.2 / (tf[t] + norm) for t in terms if tf[t])
    ids = sorted(docs, key=lambda p: (-scores[p], p))[:20]
    ideal = sorted((g for _, g in docs.values()), reverse=True)[:10]
    return dict(q=query, ids=ids, titles=[docs[p][0] for p in ids], bm=[scores[p] for p in ids],
                gain=[docs[p][1] for p in ids], idcg=float(W @ ideal))


def collect():
    previous = set(json.loads((HERE / "previous_hashes.json").read_text()))
    seen = set()
    audit = Counter()
    rows = []
    for path in sorted((HERE / "source/comparison_final_judged/judged").glob("[a-z]*_judged.json.gz")):
        sector = path.name.split("_judged")[0]
        with gzip.open(path, "rt") as stream:
            for entry in json.load(stream):
                audit["queries"] += 1
                docs = pool(entry, audit)
                key = study.order(" ".join(study.tokens(entry["text_query"])))
                if key in previous:
                    audit["prior_overlap"] += 1
                elif len(docs) < 20:
                    audit["short_pool"] += 1
                elif key in seen:
                    audit["repeat"] += 1
                else:
                    seen.add(key)
                    rows.append(dict(sector=sector, query_id=entry["query_id"], **prepare(entry, docs)))
    assert audit["queries"] == 1259 and len({r["sector"] for r in rows}) == 7
    return rows, audit


def rank(policy, row, infer, router):
    docs = row["titles"]
    if policy == "post":
        reg, threshold = router
        scores = infer(row["q"], docs[:10])
        if reg.predict([study.features(row, scores)])[0] > threshold:
            scores = np.r_[scores, infer(row["q"], docs[10:])]
    else:
        scores = infer(row["q"], docs[:int(policy[5:])])
    if not np.isfinite(scores).all():
        raise ValueError("invalid model output")
    return np.argsort(-scores, kind="stable")[:10], len(scores)


def paired(records):
    rng = np.random.default_rng(548)
    n = len(records)
    indices = rng.integers(n, size=(5000, n))
    result = {}
    for a, b in [("fixed15", "fixed20"), ("fixed15", "post"), ("post", "fixed20")]:
        contrasts = {}
        for metric in ["ndcg", "cpu_ms", "wall_ms"]:
            delta = np.array([r[a][metric] - r[b][metric] for r in records])
            contrasts[metric] = dict(mean=delta.mean(), ci95=np.quantile(
                delta[indices].mean(axis=1), [.025, .975]).tolist())
        result[a + "-" + b] = contrasts
    return result


def main(out):
    assets = {"results/confirm/post.joblib": "2113dd7a7b5d2f0c42b3f47b6fe950ae2854087ca9c16e8fbb23e66de0e56ddf",
              "data/model.onnx": "5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a",
              "data/tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"}
    for path, sha in assets.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError(path)
    rows, audit = collect()
    infer = study.model()
    router = joblib.load("results/confirm/post.joblib")
    for r in rows[:6]:
        for p in POLICIES:
            rank(p, r, infer, router)
        infer(r["q"], r["titles"][10:])
    rng = np.random.default_rng(1920)
    records = []
    for idx in rng.permutation(len(rows)):
        r = rows[idx]
        rec = {k: r[k] for k in ("sector", "query_id", "ids", "idcg", "gain")}
        for p in rng.permutation(POLICIES):
            wall = time.perf_counter_ns()
            cpu = time.process_time_ns()
            order, depth = rank(p, r, infer, router)
            cpu_ms = (time.process_time_ns() - cpu) / 1e6
            wall_ms = (time.perf_counter_ns() - wall) / 1e6
            ndcg = W @ np.array(r["gain"])[order] / r["idcg"] if r["idcg"] else 0.
            rec[p] = dict(depth=depth, order=order.tolist(), ndcg=ndcg, cpu_ms=cpu_ms, wall_ms=wall_ms)
        records.append(rec)
    out.mkdir(parents=True, exist_ok=True)
    study.save(out / "queries.json", records)
    summary = {}
    for p in POLICIES:
        summary[p] = {k: np.mean([r[p][k] for r in records]) for k in ("ndcg", "cpu_ms", "wall_ms", "depth")}
        for k in ("cpu_ms", "wall_ms"):
            summary[p][k + "_p95"] = np.quantile([r[p][k] for r in records], .95)
    sectors = defaultdict(list)
    for r in records:
        sectors[r["sector"]].append(r)
    by_sector = {s: dict(n=len(v), **{p: np.mean([r[p]["ndcg"] for r in v]) for p in POLICIES})
                 for s, v in sorted(sectors.items())}
    study.save(out / "summary.json", dict(n=len(records), audit=audit, assets=assets, policies=summary,
               zero_idcg=sum(r["idcg"] == 0 for r in records), paired=paired(records), sectors=by_sector,
               deep=sum(r["post"]["depth"] == 20 for r in records), threshold=router[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    main(parser.parse_args().out)
