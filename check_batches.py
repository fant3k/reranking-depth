"""Measure any quality drift caused by changing the online batch shape."""
import json
from pathlib import Path
import joblib
import numpy as np
import study

def main():
    root = Path("results/confirm")
    rows, _, _, _ = study.load("esci_test")
    decisions = json.loads((root/"decisions.json").read_text())
    post_model, threshold = joblib.load(root/"post.joblib")
    infer = study.model()
    weights = 1 / np.log2(np.arange(2, 12))
    values = {k: [] for k in ("fixed10", "fixed20", "split20", "pre", "post")}
    flips = 0
    for i, row in enumerate(rows):
        query, titles = row["q"], row["titles"]
        first = infer(query, titles[:10])
        halves = np.concatenate([first, infer(query, titles[10:])])
        full = np.asarray(json.loads((study.OUT/"esci_test-scores"/(study.order(query)+".json")).read_text()))
        deep = post_model.predict([study.features(row, first)])[0] > threshold
        flips += int(deep != decisions["post"][i])
        layouts = (first, full, halves, full if decisions["pre"][i] else first, halves if deep else first)
        for k, scores in zip(values, layouts, strict=True):
            ranks = np.argsort(-scores, kind="stable")[:10]
            denominator = max(np.dot(weights, row["ideal"]), 1e-12)
            values[k].append(float(weights @ np.asarray(row["gain"])[ranks] / denominator))
    rng = np.random.default_rng(2723)
    bootstrap = rng.integers(len(rows), size=(2000, len(rows)))
    quality = {}
    for k, v in values.items():
        d = np.subtract(v, values["fixed20"])
        quality[k] = dict(quality=float(np.mean(v)), delta=float(d.mean()),
                          ci=np.quantile(d[bootstrap].mean(1), [.025,.975]).tolist())
    study.save(root/"batch_parity.json", dict(n=len(rows), post_decision_flips=flips, metrics=quality))
if __name__ == "__main__":
    main()
