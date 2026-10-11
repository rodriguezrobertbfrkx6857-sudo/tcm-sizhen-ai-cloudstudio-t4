#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Cloud platform owns the NVIDIA host driver. Never install/replace it here.
python3 -c 'import sys; assert (3,10)<=sys.version_info[:2]<=(3,12), "Use Python 3.10-3.12"'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-cu126.txt
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
mkdir -p outputs
.venv/bin/python -m pip freeze > outputs/ACTUAL_INSTALLED_REQUIREMENTS.txt
echo "Dependencies installed. No model download or training was started."
