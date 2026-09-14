"""Box Count Clone: each rusher's YPC / success rate / TD rate split by
defenders in the box (light <=6, neutral 7, stacked 8+), using FTN charting
data (the only free public source for box counts). Uses full-season totals
from the most recent charted season for adequate sample size.

Outputs data/player_boxcount.parquet
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from lib_data import DATA_DIR


def box_bucket(n):
    return np.select([n <= 6, n == 7], ["light", "neutral"], default="stacked")


def season_boxsplits(season):
    pbp = pd.read_parquet(os.path.join(DATA_DIR, f"pbp_{season}.parquet"))
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    ftn = pd.read_parquet(os.path.join(DATA_DIR, f"ftn_{season}.parquet"))

    rush = pbp[(pbp["play_type"] == "run") & pbp["rusher_player_id"].notna()].copy()
    rush["play_id"] = rush["play_id"].astype(int)
    ftn_small = ftn[["nflverse_game_id", "nflverse_play_id", "n_defense_box"]].rename(
        columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"}
    )
    merged = rush.merge(ftn_small, on=["game_id", "play_id"], how="inner")
    merged = merged[merged["n_defense_box"].notna() & (merged["n_defense_box"] > 0)]
    merged["bucket"] = box_bucket(merged["n_defense_box"])
    merged["success"] = (merged["epa"] > 0).astype(int)

    grp = merged.groupby(["rusher_player_id", "bucket"]).agg(
        attempts=("rush_attempt", "sum"),
        yards=("rushing_yards", "sum"),
        tds=("rush_touchdown", "sum"),
        success=("success", "sum"),
    ).reset_index()
    grp["season"] = season
    return grp


def main():
    frames = []
    for season in [2025, 2024]:
        try:
            frames.append(season_boxsplits(season))
        except Exception as e:
            print(f"skip {season}: {e}")
    all_grp = pd.concat(frames, ignore_index=True)

    # prefer 2025 data per player; only fall back to 2024 if a player has <8 total attempts in 2025
    totals_2025 = all_grp[all_grp.season == 2025].groupby("rusher_player_id")["attempts"].sum()
    use_2025 = set(totals_2025[totals_2025 >= 8].index)

    final = pd.concat([
        all_grp[(all_grp.season == 2025) & (all_grp.rusher_player_id.isin(use_2025))],
        all_grp[(all_grp.season == 2024) & (~all_grp.rusher_player_id.isin(use_2025))],
    ], ignore_index=True)

    final["ypc"] = (final["yards"] / final["attempts"]).round(2)
    final["success_rate"] = (final["success"] / final["attempts"]).round(3)
    final["td_rate"] = (final["tds"] / final["attempts"]).round(3)

    out_path = os.path.join(DATA_DIR, "player_boxcount.parquet")
    final.to_parquet(out_path)
    print("saved", out_path, final.shape)


if __name__ == "__main__":
    main()
