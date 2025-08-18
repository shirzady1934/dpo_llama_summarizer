#!/usr/bin/env bash
set -euo pipefail
CONFIG=${1:-src/configs/config.yaml}
python -m src.cli.train --config "$CONFIG"
