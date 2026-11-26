"""Inspect the two possible rankings for one real query in the previously viewed WANDS pilot."""
import argparse
import json
from pathlib import Path
import numpy as np
from study import order

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("text",nargs="?",default="",help="substring from an English furniture query")
    cli.add_argument("--list",action="store_true",help="list up to 30 matching queries")
    args=cli.parse_args()
    root=Path("results")
    pool=json.loads((root/"wands.json").read_text())[200:]
    matches=[r for r in pool if args.text.casefold() in r["q"]]
    if args.list or not matches:
        print("\n".join(r["q"] for r in matches[:30]) or "No matching query in this evaluation subset.")
        return
    row=next((r for r in matches if r["q"]==args.text.casefold()),matches[0])
    scores=np.asarray(json.loads((root/"wands-scores"/(order(row["q"])+".json")).read_text()))
    print(row["q"])
    for depth in (10,20):
        print(f"\nReranked {depth} candidates")
        for index in np.argsort(-scores[:depth],kind="stable")[:5]:
            print(f"  [{row['gain'][index]:.1f}] {row['titles'][index]}")
    print("\nBrackets show WANDS relevance gain for each product.")
if __name__=="__main__": main()
