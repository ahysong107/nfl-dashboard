"""Fetch and cache raw nflverse data locally as parquet files."""
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

SEASONS = [2023, 2024, 2025, 2026]

BASE = "https://github.com/nflverse/nflverse-data/releases/download"


def cache(name, url):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        print(f"[skip] {name} already cached")
        return
    print(f"[fetch] {url}")
    try:
        df = pd.read_parquet(url)
        df.to_parquet(path)
        print(f"[ok] {name} shape={df.shape}")
    except Exception as e:
        print(f"[fail] {name}: {e}")


def main():
    for season in SEASONS:
        cache(f"pbp_{season}.parquet", f"{BASE}/pbp/play_by_play_{season}.parquet")
        cache(f"weekly_{season}.parquet", f"{BASE}/stats_player/stats_player_week_{season}.parquet")

    for season in [2024, 2025]:
        cache(f"ftn_{season}.parquet", f"{BASE}/ftn_charting/ftn_charting_{season}.parquet")

    for season in [2023, 2024, 2025, 2026]:
        cache(f"snaps_{season}.parquet", f"{BASE}/snap_counts/snap_counts_{season}.parquet")

    sched_path = os.path.join(DATA_DIR, "schedules.parquet")
    if not os.path.exists(sched_path):
        # nfl_data_py.import_schedules() hard-codes a fetch from
        # habitatring.com/games.csv, a personal mirror that some sandboxed
        # network environments block by egress policy. It's just a mirror of
        # this same nflverse/nfldata GitHub file, so pull that directly.
        sched = pd.read_csv("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv")
        sched = sched[sched["season"].isin([2025, 2026])]
        sched.to_parquet(sched_path)
        print(f"[ok] schedules.parquet shape={sched.shape}")
    else:
        print("[skip] schedules.parquet already cached")

    cache("rosters_2026.parquet", f"{BASE}/rosters/roster_2026.parquet")
    cache("players.parquet", f"{BASE}/players/players.parquet")


if __name__ == "__main__":
    main()
