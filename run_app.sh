#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/streamlit" ]]; then
  echo "Missing .venv/bin/streamlit. Set up the virtualenv first." >&2
  exit 1
fi

export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

exec .venv/bin/streamlit run app.py --server.fileWatcherType none "$@"
