#!/usr/bin/env bash
# Full pipeline refresh, in dependency order. Safe to re-run from scratch --
# scripts/fetch_data.py skips re-downloading a file that's already on disk,
# but fetch_depth_charts.py and fetch_injuries.py always refetch since those
# sources update same-day.
set -euo pipefail
cd "$(dirname "$0")/.."

# Install with the same interpreter the pipeline runs under. On the cloud
# sandbox the bare `pip` command can belong to a different Python than
# `python3`, so a plain `pip install` lands packages where the scripts
# can't see them.
python3 -m pip install -q -r requirements.txt || echo "pip install failed; continuing with the existing environment"

python3 scripts/fetch_data.py
python3 scripts/fetch_depth_charts.py
python3 scripts/fetch_injuries.py
python3 scripts/build_player_game.py
python3 scripts/build_features.py
python3 scripts/build_scores.py
python3 scripts/build_props.py
python3 scripts/build_boxcounts.py
python3 scripts/build_site_data.py

echo "Pipeline complete. site/data.json is ready to publish."
