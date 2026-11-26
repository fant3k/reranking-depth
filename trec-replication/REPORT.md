# Does the depth-15 quality margin transfer to TREC product search?

The frozen quality margin did not replicate on TREC 2023. Fixed depth 15 scored 0.7285 nDCG@10 versus 0.7368 for depth 20. The mean loss of 0.0084 was within the 0.01 tolerance, but the paired 95% interval for the difference extended to −0.0160. These data leave open a loss larger than the tolerance.

| Comparison | Mean difference in nDCG@10 | Paired 95% query-bootstrap interval |
|---|---:|---:|
| Fixed 15 minus fixed 20 | −0.0084 | [−0.0160, −0.0011] |
| Fixed 15 minus original post router | +0.0054 | [−0.0038, +0.0143] |

All 186 officially judged queries qualified. None matched the 6,273 normalized historical query hashes. The frozen post router sent 39.2% of queries to depth 20; its mean candidate budget was 13.92. This is a comparison of the exact previously fitted policies, not a guarantee of equal budgets after transfer.

The organizers describe NIST judging for the TREC Product Search track. We used official grades and explicitly judged candidates, dropped 158 distinct judged IDs whose titles were unavailable, selected the BM25 top 20 within each judged pool, then executed the original MiniLM model for each policy. All topics retained a positive ideal gain. This is a reranking evaluation, not a full-catalog TREC submission. The catalog shares the broader Amazon ecosystem with ESCI; no product-disjointness is claimed. Query-bootstrap intervals cover variation across these judged topics. No latency claim is made for TREC. [Track overview](https://trec.nist.gov/pubs/trec32/papers/trackorg.P.pdf) · [Official dataset inventory](https://pages.nist.gov/trec-browser/trec32/product/data/)

## Reproduce

From the search project directory after the original model and frozen router have been reconstructed, run `python trec-replication/run.py`. The script downloads pinned copies of the public corpus and evaluation files (about 1.1 GB compressed). Run `python3 trec-replication/audit.py` to verify the archived query-level evidence without downloads. [Protocol](PROTOCOL.md) · [Per-query results](per-query.csv) · [Raw summary](summary.json)
