"""Build trailing (last-5-meaningful-game) player features, both as
'current state' (for this week's predictions) and 'as-of-history' (shifted,
used to train the anytime-TD probability calibration).

Also builds a defense-vs-position matchup table (TDs allowed per game),
trailing over each defense's last 8 games.

Outputs:
  data/player_current.parquet   -- one row per player, latest trailing state
  data/player_history.parquet   -- one row per player-game, trailing-as-of-before state + label
  data/defense_matchup_current.parquet
  data/defense_matchup_history.parquet
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from lib_data import DATA_DIR, load_schedules

MEANINGFUL_TOUCH_MIN = 3
MEANINGFUL_ATT_MIN = 10
WINDOW = 5
DEF_WINDOW = 8

ROLL_SUM_COLS = [
    "touches", "total_td", "player_rz_touches", "team_rz_touches",
    "rush_goal_line", "rush_red_zone", "rush_fringe", "rush_open_field",
    "tgt_goal_line", "tgt_red_zone", "tgt_fringe", "tgt_open_field",
    "attempts", "passing_yards", "passing_tds", "receptions",
    "receiving_yards", "rushing_yards", "receiving_tds", "rushing_tds",
]


def add_rolling(df, group_col, cols, window, shift):
    df = df.sort_values([group_col, "season", "week"]).copy()
    g = df.groupby(group_col, group_keys=False)
    for c in cols:
        if shift:
            roll = g[c].apply(lambda s: s.rolling(window, min_periods=1).sum().shift(1))
        else:
            roll = g[c].apply(lambda s: s.rolling(window, min_periods=1).sum())
        df[f"tr_{c}"] = roll
    # rolling count of games (for per-game rates) + mean target share
    if shift:
        cnt = g["touches"].apply(lambda s: s.rolling(window, min_periods=1).count().shift(1))
        ts = g["target_share"].apply(lambda s: s.rolling(window, min_periods=1).mean().shift(1))
    else:
        cnt = g["touches"].apply(lambda s: s.rolling(window, min_periods=1).count())
        ts = g["target_share"].apply(lambda s: s.rolling(window, min_periods=1).mean())
    df["tr_games"] = cnt
    df["tr_target_share"] = ts
    return df


def derive_ratios(df):
    df["volume_pg"] = df["tr_touches"] / df["tr_games"].replace(0, np.nan)
    df["goal_line_pg"] = df["tr_rush_goal_line"] / df["tr_games"].replace(0, np.nan)
    df["red_zone_role"] = df["tr_player_rz_touches"] / df["tr_team_rz_touches"].replace(0, np.nan)
    df["td_rate"] = df["tr_total_td"] / df["tr_touches"].replace(0, np.nan)
    for c in ["volume_pg", "goal_line_pg", "red_zone_role", "td_rate", "tr_target_share"]:
        df[c] = df[c].fillna(0)
    return df


def build_history():
    pg = pd.read_parquet(os.path.join(DATA_DIR, "player_game.parquet"))
    pg["is_meaningful"] = (pg["touches"] >= MEANINGFUL_TOUCH_MIN) | (pg["attempts"].fillna(0) >= MEANINGFUL_ATT_MIN)
    meaningful = pg[pg["is_meaningful"]].copy()

    hist = add_rolling(meaningful, "player_id", ROLL_SUM_COLS, WINDOW, shift=True)
    hist = derive_ratios(hist)
    hist["label_td"] = (hist["total_td"] > 0).astype(int)
    hist = hist[hist["tr_games"] >= 1]  # need at least 1 prior meaningful game

    hist_path = os.path.join(DATA_DIR, "player_history.parquet")
    hist.to_parquet(hist_path)
    print("saved", hist_path, hist.shape)
    return hist


def build_current():
    pg = pd.read_parquet(os.path.join(DATA_DIR, "player_game.parquet"))
    pg["is_meaningful"] = (pg["touches"] >= MEANINGFUL_TOUCH_MIN) | (pg["attempts"].fillna(0) >= MEANINGFUL_ATT_MIN)
    meaningful = pg[pg["is_meaningful"]].copy()

    cur = add_rolling(meaningful, "player_id", ROLL_SUM_COLS, WINDOW, shift=False)
    cur = derive_ratios(cur)

    # keep only the latest row per player (their current trailing state)
    cur = cur.sort_values(["player_id", "season", "week"])
    latest = cur.groupby("player_id").tail(1).copy()

    # also compute TD drought (consecutive games without a TD) using ALL games (not just meaningful)
    pg_sorted = pg.sort_values(["player_id", "season", "week"]).copy()
    pg_sorted["scored"] = pg_sorted["total_td"] > 0
    droughts = {}
    last_meta = {}
    for pid, grp in pg_sorted.groupby("player_id"):
        streak = 0
        for _, row in grp.iterrows():
            if row["touches"] < 1:
                continue  # game with no offensive touches doesn't break/extend drought
            if row["scored"]:
                streak = 0
            else:
                streak += 1
        droughts[pid] = streak
    latest["games_since_td"] = latest["player_id"].map(droughts).fillna(0).astype(int)

    cur_path = os.path.join(DATA_DIR, "player_current.parquet")
    latest.to_parquet(cur_path)
    print("saved", cur_path, latest.shape)
    return latest


def build_defense_grid():
    pg = pd.read_parquet(os.path.join(DATA_DIR, "player_game.parquet"))
    sched = load_schedules()

    # full defteam-game grid so 0-TD games count toward the average
    games_long = pd.concat([
        sched[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "defteam"}),
        sched[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "defteam"}),
    ], ignore_index=True)
    games_long = games_long[games_long["game_id"].isin(pg["game_id"].unique())]

    position_groups = ["RB", "WR", "TE", "QB"]
    grid = games_long.merge(pd.DataFrame({"position_group": position_groups}), how="cross")

    pos_td = pg.groupby(["opponent_team", "game_id", "position_group"])["total_td"].sum().reset_index()
    pos_td = pos_td.rename(columns={"opponent_team": "defteam"})
    pos_yards = pg.copy()
    pos_yards["yards_allowed"] = pos_yards["rushing_yards"].fillna(0) + pos_yards["receiving_yards"].fillna(0) + pos_yards["passing_yards"].fillna(0)
    pos_yards_g = pos_yards.groupby(["opponent_team", "game_id", "position_group"])["yards_allowed"].sum().reset_index()
    pos_yards_g = pos_yards_g.rename(columns={"opponent_team": "defteam"})

    grid = grid.merge(pos_td, on=["defteam", "game_id", "position_group"], how="left")
    grid = grid.merge(pos_yards_g, on=["defteam", "game_id", "position_group"], how="left")
    grid["total_td"] = grid["total_td"].fillna(0)
    grid["yards_allowed"] = grid["yards_allowed"].fillna(0)

    grid = grid.sort_values(["defteam", "position_group", "season", "week"]).reset_index(drop=True)

    def roll(df, shift):
        g = df.groupby(["defteam", "position_group"], group_keys=False)
        td = g["total_td"].apply(lambda s: s.rolling(DEF_WINDOW, min_periods=1).mean().shift(1) if shift else s.rolling(DEF_WINDOW, min_periods=1).mean())
        yd = g["yards_allowed"].apply(lambda s: s.rolling(DEF_WINDOW, min_periods=1).mean().shift(1) if shift else s.rolling(DEF_WINDOW, min_periods=1).mean())
        out = df.copy()
        out["td_allowed_pg"] = td
        out["yards_allowed_pg"] = yd
        return out

    hist = roll(grid, shift=True)
    hist_path = os.path.join(DATA_DIR, "defense_matchup_history.parquet")
    hist.to_parquet(hist_path)
    print("saved", hist_path, hist.shape)

    cur = roll(grid, shift=False)
    cur = cur.sort_values(["defteam", "position_group", "season", "week"])
    cur_latest = cur.groupby(["defteam", "position_group"]).tail(1)
    cur_path = os.path.join(DATA_DIR, "defense_matchup_current.parquet")
    cur_latest.to_parquet(cur_path)
    print("saved", cur_path, cur_latest.shape)


if __name__ == "__main__":
    build_history()
    build_current()
    build_defense_grid()
