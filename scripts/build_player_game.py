"""Build a per-player-per-game table with zone-level touch breakdowns,
league zone conversion rates, and team-level red-zone touch denominators.
Saves data/player_game.parquet and data/zone_rates.json
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from lib_data import load_pbp, load_weekly, zone_bucket, ZONES, DATA_DIR


def main():
    pbp = load_pbp()
    print("pbp rows:", len(pbp))

    rush = pbp[(pbp["play_type"] == "run") & pbp["rusher_player_id"].notna()].copy()
    rush["zone"] = zone_bucket(rush["yardline_100"])
    tgt = pbp[(pbp["play_type"] == "pass") & pbp["receiver_player_id"].notna()].copy()
    tgt["zone"] = zone_bucket(tgt["yardline_100"])

    # ---- league zone conversion rates (used for xTD / TD debt) ----
    rush_rates = rush.groupby("zone").apply(lambda d: d["rush_touchdown"].sum() / len(d)).to_dict()
    tgt_rates = tgt.groupby("zone").apply(lambda d: d["pass_touchdown"].sum() / len(d)).to_dict()
    zone_rates = {"rush": rush_rates, "target": tgt_rates}
    with open(os.path.join(DATA_DIR, "zone_rates.json"), "w") as f:
        json.dump(zone_rates, f, indent=2)
    print("zone rates:", zone_rates)

    # ---- per player-game zone carry/target pivot ----
    rush_piv = rush.pivot_table(
        index=["rusher_player_id", "game_id"], columns="zone", values="rush_attempt",
        aggfunc="sum", fill_value=0,
    )
    rush_piv.columns = [f"rush_{c}" for c in rush_piv.columns]
    rush_piv = rush_piv.reset_index().rename(columns={"rusher_player_id": "player_id"})

    tgt_piv = tgt.pivot_table(
        index=["receiver_player_id", "game_id"], columns="zone", values="pass_attempt",
        aggfunc="sum", fill_value=0,
    )
    tgt_piv.columns = [f"tgt_{c}" for c in tgt_piv.columns]
    tgt_piv = tgt_piv.reset_index().rename(columns={"receiver_player_id": "player_id"})

    zone_cols = [f"rush_{z}" for z in ZONES] + [f"tgt_{z}" for z in ZONES]
    zone_pg = pd.merge(rush_piv, tgt_piv, on=["player_id", "game_id"], how="outer")
    for c in zone_cols:
        if c not in zone_pg.columns:
            zone_pg[c] = 0
    zone_pg[zone_cols] = zone_pg[zone_cols].fillna(0)

    # ---- team red-zone (<=20) touch totals per team-game (denominator for red zone role) ----
    rush["team_rz_touch"] = (rush["zone"].isin(["goal_line", "red_zone"])).astype(int)
    tgt["team_rz_touch"] = (tgt["zone"].isin(["goal_line", "red_zone"])).astype(int)
    team_rz_rush = rush.groupby(["posteam", "game_id"])["team_rz_touch"].sum()
    team_rz_tgt = tgt.groupby(["posteam", "game_id"])["team_rz_touch"].sum()
    team_rz = (team_rz_rush.add(team_rz_tgt, fill_value=0)).reset_index()
    team_rz.columns = ["team", "game_id", "team_rz_touches"]

    # player own rz touches (goal_line + red_zone combined, carries+targets)
    zone_pg["player_rz_touches"] = (
        zone_pg["rush_goal_line"] + zone_pg["rush_red_zone"] +
        zone_pg["tgt_goal_line"] + zone_pg["tgt_red_zone"]
    )

    # ---- merge onto weekly stats (yards, TDs, target_share, team, opp, week/season) ----
    wk = load_weekly()
    keep_cols = [
        "player_id", "player_display_name", "position", "position_group", "team",
        "opponent_team", "season", "week", "game_id", "headshot_url",
        "carries", "rushing_yards", "rushing_tds", "targets", "receptions",
        "receiving_yards", "receiving_tds", "target_share", "attempts",
        "completions", "passing_yards", "passing_tds", "passing_interceptions",
        "fantasy_points_ppr",
    ]
    wk = wk[keep_cols].copy()
    wk["touches"] = wk["carries"].fillna(0) + wk["targets"].fillna(0)
    wk["total_td"] = wk["rushing_tds"].fillna(0) + wk["receiving_tds"].fillna(0)

    pg = pd.merge(wk, zone_pg, on=["player_id", "game_id"], how="left")
    for c in zone_cols + ["player_rz_touches"]:
        pg[c] = pg[c].fillna(0)

    pg = pd.merge(pg, team_rz, on=["team", "game_id"], how="left")
    pg["team_rz_touches"] = pg["team_rz_touches"].fillna(0)

    pg = pg.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    out_path = os.path.join(DATA_DIR, "player_game.parquet")
    pg.to_parquet(out_path)
    print("saved", out_path, pg.shape)


if __name__ == "__main__":
    main()
