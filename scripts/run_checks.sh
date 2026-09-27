#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python scripts/index_document.py --check
.venv/bin/python -m pytest -q --junitxml=reports/pytest.xml
.venv/bin/python -m aimedicine.cli demo --mode offline --output reports/offline
.venv/bin/python -m aimedicine.cli evaluate --mode offline --output reports/evaluation
.venv/bin/python scripts/build_notebook.py
.venv/bin/python scripts/execute_notebook.py
