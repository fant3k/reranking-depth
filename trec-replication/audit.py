"""Offline check of the TREC snapshot. Run with the standard Python library."""
from pathlib import Path
import csv
import json
import math
from statistics import mean

root = Path(__file__).resolve().parent
summary = json.loads((root / "summary.json").read_text())
with (root / "per-query.csv").open() as f:
    rows = list(csv.DictReader(f))
assert len(rows) == len({r["normalized_hash"] for r in rows}) == summary["n"]
assert not ({r["normalized_hash"] for r in rows} & set((root/"excluded-query-hashes.txt").read_text().splitlines()))
assert all(int(r["pool"]) >= 20 for r in rows)
for name in ("fixed15","fixed20","post"):
    vals = [float(r[name]) for r in rows]
    assert all(0 <= v <= 1.00000001 for v in vals)
    assert math.isclose(mean(vals),summary["mean"][name],abs_tol=1e-12)
for base in ("post","fixed20"):
    delta = mean(float(r["fixed15"])-float(r[base]) for r in rows)
    assert math.isclose(delta,summary["fixed15_minus"][base]["delta"],abs_tol=1e-12)
assert summary["passes_margin"] == (summary["fixed15_minus"]["fixed20"]["ci"][0] > -.01)
print("TREC held-out query identities, pool sizes, means and margin decision verified.")
