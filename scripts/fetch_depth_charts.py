"""Fetch the depth_charts release (updated daily, unlike the week-1-only
roster snapshot) so we can correct stale team assignments (trades) and
identify the CURRENT starter at each position -- this is what catches
cases like a player being traded mid-week or a starter returning from
injury while a backup still shows up in the season-long stat trail.
"""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    url = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2026.parquet"
    dc = pd.read_parquet(url)
    path = os.path.join(DATA_DIR, "depth_charts_2026.parquet")
    dc.to_parquet(path)
    latest = dc["dt"].max()
    print(f"saved {path}, {dc.shape}, latest snapshot: {latest}")


if __name__ == "__main__":
    main()
