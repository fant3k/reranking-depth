"""Recompute metrics from the saved positions and check sample accounting."""
import json
import math
from pathlib import Path
import sys
from statistics import mean

def check(directory):
    rows = json.loads((directory / "queries.json").read_text())
    s = json.loads((directory / "summary.json").read_text())
    audit = s["audit"]
    assert len({(r["sector"], r["query_id"]) for r in rows}) == len(rows) == s["n"]
    assert len(rows) + sum(audit.get(k, 0) for k in ("prior_overlap", "short_pool", "repeat")) == 1259
    assert sum(r["post"]["depth"] == 20 for r in rows) == s["deep"]
    for p in s["policies"]:
        values = []
        for r in rows:
            m = r[p]
            assert len(set(r["ids"])) == len(r["gain"]) == 20
            assert len(set(m["order"])) == len(m["order"]) == 10
            assert m["depth"] in ({"fixed15": [15], "fixed20": [20], "post": [10,20]}[p])
            assert all(0 <= i < m["depth"] for i in m["order"])
            assert min(m["wall_ms"], m["cpu_ms"]) > 0
            dcg = sum(r["gain"][i] / math.log2(k + 2) for k,i in enumerate(m["order"]))
            value = dcg / r["idcg"] if r["idcg"] else 0.
            assert 0 <= value <= 1 + 1e-12 and math.isclose(value, m["ndcg"], abs_tol=1e-12)
            values.append(value)
        assert math.isclose(mean(values), s["policies"][p]["ndcg"])
    for pair, metrics in s["paired"].items():
        a,b = pair.split("-")
        for k, result in metrics.items():
            assert math.isclose(mean(r[a][k]-r[b][k] for r in rows), result["mean"], abs_tol=1e-12)
    print(f"Verified {len(rows)} queries and {3*len(rows)} measured rankings.")

if __name__ == "__main__":
    check(Path(sys.argv[1]))
