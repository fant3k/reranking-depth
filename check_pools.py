"""Check candidate identity after text normalization against the original judged pools."""
from collections import Counter
import json
import polars as pl
from study import DATA, OUT, tokens, save

def main():
    samples = {name:json.loads((OUT/(name+".json")).read_text()) for name in ("amazon","wands","esci_test")}
    keys = {name:{" ".join(tokens(row["q"])) for row in rows} for name,rows in samples.items()}
    overlap = len(keys["esci_test"] & (keys["amazon"] | keys["wands"]))
    if overlap:
        raise ValueError("test overlaps with training or pilot")
    counts = {"test_normalized_overlap": overlap}
    for name,rows in samples.items():
        if name == "wands":
            frame = (pl.read_csv(DATA/"label.csv",separator="\t",infer_schema=False)
                     .join(pl.read_csv(DATA/"query.csv",separator="\t",infer_schema=False),on="query_id"))
        else:
            split = "train" if name == "amazon" else "test"
            frame = (pl.scan_parquet(DATA/"examples.parquet")
                     .filter((pl.col("product_locale")=="us") & (pl.col("split")==split) & (pl.col("small_version")==1))
                     .select("query","product_id",pl.col("esci_label").alias("label")).collect())
        selected={r["q"] for r in rows}
        variants={}
        for q,pid,label in frame.select("query","product_id","label").iter_rows():
            key=q.strip().casefold()
            if key in selected: variants.setdefault((key,pid),set()).add((q,label))
        conflicts=sum(len({l for _,l in pairs})>1 for pairs in variants.values())
        joined_variants=sum(len({q for q,_ in pairs})>1 for pairs in variants.values())
        counts[name]=dict(rows=len(rows),within_normalized_collisions=len(rows)-len(keys[name]),
                          multi_spelling_query_product=joined_variants,conflicting_labels=conflicts,
                          zero_idcg=sum(max(r["ideal"])==0 for r in rows))
    save(OUT/"confirm"/"pool_audit.json",counts)
    print(counts)
if __name__ == "__main__":
    main()
