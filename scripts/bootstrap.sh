#!/usr/bin/env bash
set -euo pipefail
echo "AMD branch: automatic CUDA installation is disabled. Use scripts/rocm_env_isolated.sh inspect." >&2
echo "For T4, use the separate codex/cloudstudio-t4-20261011 branch." >&2
exit 2
