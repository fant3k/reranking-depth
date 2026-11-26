import unittest
from collections import Counter
import numpy as np
from evaluate import pool, prepare, rank, paired


class ExternalTest(unittest.TestCase):
    def test_judgments_and_ranking(self):
        hits = [dict(id=str(i), title="widget", label="Exact", score=1.) for i in range(21)]
        hits += [dict(hits[0], id="-0"), dict(hits[1], label="Irrelevant", score=0.),
                 dict(hits[2], id="failed", label="Irrelevant", score=0., reasoning="judge failed after retries"),
                 dict(hits[3], id="real-zero", label="Irrelevant", score=0.)]
        entry = dict(text_query="widget", providers={"a": {"hits": hits}})
        audit = Counter()
        docs = pool(entry, audit)
        self.assertEqual(set(docs) - {str(i) for i in range(21)}, {"realzero"})
        self.assertNotIn("1", docs)
        self.assertEqual(audit["unjudged_hits"], 1)
        row = prepare(entry, docs)
        self.assertEqual(row["ids"], sorted(docs)[:20])
        for policy, pred, expected in [("post", 0, [10]), ("post", 1, [10,10]),
                                       ("fixed15", 0, [15]), ("fixed20", 0, [20])]:
            calls = []
            def infer(q, titles):
                calls.append(len(titles))
                return np.zeros(len(titles))
            class Router:
                def predict(self, x):
                    return [pred]
            order, depth = rank(policy, row, infer, (Router(), 0))
            self.assertEqual(calls, expected)
            self.assertEqual(depth, sum(expected))
            self.assertEqual(order.tolist(), list(range(10)))

    def test_paired(self):
        records = [{p: dict.fromkeys(("ndcg", "cpu_ms", "wall_ms"), i + offset)
                    for p, offset in [("fixed15", 1), ("fixed20", 0), ("post", 2)]} for i in range(8)]
        self.assertEqual(paired(records)["fixed15-post"]["ndcg"]["ci95"], [-1., -1.])
