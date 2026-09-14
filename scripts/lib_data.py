"""Shared data loading + feature engineering for the NFL prop dashboard."""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SEASONS = [2023, 2024, 2025, 2026]


def load_pbp():
    dfs = [pd.read_parquet(os.path.join(DATA_DIR, f"pbp_{s}.parquet")) for s in SEASONS]
    pbp = pd.concat(dfs, ignore_index=True)
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    return pbp


def load_weekly():
    dfs = [pd.read_parquet(os.path.join(DATA_DIR, f"weekly_{s}.parquet")) for s in SEASONS]
    wk = pd.concat(dfs, ignore_index=True)
    wk = wk[wk["season_type"] == "REG"].copy()
    return wk


def load_ftn():
    dfs = []
    for s in [2024, 2025]:
        p = os.path.join(DATA_DIR, f"ftn_{s}.parquet")
        if os.path.exists(p):
            dfs.append(pd.read_parquet(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_snaps():
    dfs = [pd.read_parquet(os.path.join(DATA_DIR, f"snaps_{s}.parquet")) for s in SEASONS]
    return pd.concat(dfs, ignore_index=True)


def load_schedules():
    return pd.read_parquet(os.path.join(DATA_DIR, "schedules.parquet"))


def load_rosters():
    return pd.read_parquet(os.path.join(DATA_DIR, "rosters_2026.parquet"))


def zone_bucket(yardline_100):
    return np.select(
        [yardline_100 <= 5, yardline_100 <= 20, yardline_100 <= 40],
        ["goal_line", "red_zone", "fringe"],
        default="open_field",
    )


ZONES = ["goal_line", "red_zone", "fringe", "open_field"]
