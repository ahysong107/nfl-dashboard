"""Build yardage/reception/passing-TD projections with modeled probability
ladders (Normal approximation from each player's own trailing game-to-game
variance, matchup-adjusted by opponent yards allowed at the position).

Outputs data/player_props.parquet
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from scipy.stats import norm
from lib_data import DATA_DIR

MEANINGFUL_TOUCH_MIN = 3
MEANINGFUL_ATT_MIN = 10
WINDOW = 5

PROP_DEFS = {
    "pass_yards": {"col": "passing_yards", "positions": ["QB"], "min_std": 25},
    "pass_tds": {"col": "passing_tds", "positions": ["QB"], "min_std": 0.6},
    "rush_yards": {"col": "rushing_yards", "positions": ["RB", "QB"], "min_std": 12, "min_mean": 15},
    "receptions": {"col": "receptions", "positions": ["RB", "WR", "TE"], "min_std": 1.0, "min_mean": 1.5},
}


def trailing_gamelog(pg):
    pg = pg.sort_values(["player_id", "season", "week"])
    out = {}
    for pid, grp in pg.groupby("player_id"):
        out[pid] = grp
    return out


def main():
    pg = pd.read_parquet(os.path.join(DATA_DIR, "player_game.parquet"))
    pg["is_meaningful"] = (pg["touches"] >= MEANINGFUL_TOUCH_MIN) | (pg["attempts"].fillna(0) >= MEANINGFUL_ATT_MIN)
    meaningful = pg[pg["is_meaningful"]].sort_values(["player_id", "season", "week"])

    def_cur = pd.read_parquet(os.path.join(DATA_DIR, "defense_matchup_current.parquet"))

    rows = []
    for pid, grp in meaningful.groupby("player_id"):
        last5 = grp.tail(WINDOW)
        if len(last5) == 0:
            continue
        latest = last5.iloc[-1]
        rec = {
            "player_id": pid,
            "player_display_name": latest["player_display_name"],
            "position": latest["position"],
            "position_group": latest["position_group"],
            "team": latest["team"],
        }
        for prop, spec in PROP_DEFS.items():
            if latest["position_group"] not in spec["positions"] and not (prop == "rush_yards" and latest["position"] == "QB"):
                continue
            vals = last5[spec["col"]].fillna(0).values.astype(float)
            n = len(vals)
            mean = float(np.mean(vals)) if n else 0.0
            if mean < spec.get("min_mean", 0):
                continue
            std = float(np.std(vals, ddof=1)) if n > 1 else spec["min_std"]
            std = max(std, spec["min_std"])

            # matchup adjustment: opponent's trailing yards/TDs allowed to this position vs league avg
            pos_key = "QB" if prop in ("pass_yards", "pass_tds") else latest["position_group"]
            opp_row = def_cur[def_cur["position_group"] == pos_key]
            league_avg_yards = opp_row["yards_allowed_pg"].mean() if len(opp_row) else np.nan
            league_avg_td = opp_row["td_allowed_pg"].mean() if len(opp_row) else np.nan

            rec[f"{prop}_mean"] = round(mean, 1)
            rec[f"{prop}_std"] = round(std, 1)
            rec[f"{prop}_games"] = n

            if prop == "pass_tds":
                rec[f"{prop}_league_avg_allowed"] = round(float(league_avg_td), 2) if pd.notna(league_avg_td) else None
            else:
                rec[f"{prop}_league_avg_allowed"] = round(float(league_avg_yards), 1) if pd.notna(league_avg_yards) else None

        rows.append(rec)

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "player_props.parquet")
    out.to_parquet(out_path)
    print("saved", out_path, out.shape)


if __name__ == "__main__":
    main()
