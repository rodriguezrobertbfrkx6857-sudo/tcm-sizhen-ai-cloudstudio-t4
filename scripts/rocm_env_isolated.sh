#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-inspect}"
case "$MODE" in
 inspect) python3 "$ROOT/scripts/amd_rocm_probe.py" --workspace /mnt/workspace ;;
 --create-venv)
  python3 -c 'import torch; assert torch.version.hip, "Existing ROCm Torch required; no install performed"'
  ENV_DIR="$ROOT/.venv-rocm"
  if [[ -e "$ENV_DIR" ]]; then echo "Existing environment: refusing overwrite" >&2; exit 2; fi
  python3 -m venv --system-site-packages "$ENV_DIR"
  "$ENV_DIR/bin/python" "$ROOT/scripts/amd_rocm_probe.py" --workspace /mnt/workspace
  echo "Isolated environment created; no wheel installed. Never install CUDA requirements here." ;;
 *) echo "Usage: $0 [inspect|--create-venv]" >&2; exit 2 ;;
esac
