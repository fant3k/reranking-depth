#!/usr/bin/env bash
set -euo pipefail
src=$(cd "$(dirname "$0")" && pwd)
work=${1:-"$src/reproduction"}
mkdir -p "$work"
cd "$work"
py=${PYTHON:-python}
exec 9>.reproduction.lock
flock -n 9 || { echo "This reproduction directory already has a running experiment." >&2; exit 1; }
"$py" - "$src" <<'STAMP'
import hashlib, json, sys
from pathlib import Path
src=Path(sys.argv[1])
if src.resolve()==Path.cwd():
    raise SystemExit("Use a separate reproduction directory.")
inputs=sorted(src.glob("*.py"))+[src/"requirements.lock",src/"reproduce.sh"]
stamp=Path("source-hashes.json")
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
if stamp.exists() and json.loads(stamp.read_text())!=hashes:
    raise SystemExit("Source changed since this run. Use a fresh reproduction directory.")
if not stamp.exists() and Path("logs/state.json").exists():
    raise SystemExit("Legacy queue outputs predate source hashes. Use a fresh directory to preserve the original run.")
temp=stamp.with_suffix(".tmp");temp.write_text(json.dumps(hashes,indent=2));temp.replace(stamp)
STAMP
for script in "$src"/*.py; do
  cp "$script" .
done
export OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=4 TOKENIZERS_PARALLELISM=false
"$py" download.py
"$py" pilot_queue.py --selftest
"$py" pilot_queue.py
"$py" confirm_queue.py
"$py" audit_confirm.py
"$py" query_diagnostics.py
"$py" check_pools.py
"$py" check_batches.py
"$py" learning_queue.py
"$py" fixed_queue.py
"$py" report.py
