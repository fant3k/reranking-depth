#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python_bin=${PYTHON:-.venv/bin/python}
export TOKENIZERS_PARALLELISM=false
if [[ ! -d external-retail/source/comparison_final_judged/judged ]]; then
  git clone --no-checkout https://github.com/quissly/search-benchmark external-retail/source
  git -C external-retail/source checkout --detach a4b56b2732a00e7baac67444b38b0fdf4a65d308
fi
(cd external-retail && shasum -a 256 -c inputs.sha256)
"$python_bin" -m unittest discover -s external-retail -p test_external.py
"$python_bin" external-retail/evaluate.py --out "${1:-external-retail/replay}"
"$python_bin" external-retail/verify.py "${1:-external-retail/replay}"
