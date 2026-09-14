"""Compute Baseline / Opportunity / Venom scores, calibrated anytime-TD
probability (via logistic regression trained on real historical outcomes),
due bonus, and TD-debt zone grids for every player with recent volume.

Outputs data/player_scores.parquet
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from lib_data import DATA_DIR, ZONES

BASELINE_WEIGHTS = {
    "red_zone_role": 0.35,
    "volume_pg": 0.25,
    "goal_line_pg": 0.15,
    "tr_target_share": 0.15,
    "td_rate": 0.10,
}


def pct_rank(series):
    return series.rank(pct=True, method="average") * 100


def add_matchup(df, defense_tbl, on_season_week=True):
    d = defense_tbl.rename(columns={"defteam": "opponent_team"})
    d = d[["opponent_team", "position_group", "season", "week", "td_allowed_pg", "yards_allowed_pg"]]
    merged = df.merge(d, on=["opponent_team", "position_group", "season", "week"], how="left")
    merged["td_allowed_pg"] = merged["td_allowed_pg"].fillna(merged["td_allowed_pg"].median())
    merged["yards_allowed_pg"] = merged["yards_allowed_pg"].fillna(merged["yards_allowed_pg"].median())
    return merged


def train_calibration(hist):
    hist = hist[hist["position_group"].isin(["RB", "WR", "TE"])].copy()
    feat_cols = ["red_zone_role", "volume_pg", "goal_line_pg", "tr_target_share", "td_rate", "td_allowed_pg"]
    for c in feat_cols:
        hist[f"{c}_pct"] = pct_rank(hist[c])
    X = hist[[f"{c}_pct" for c in feat_cols]].values
    y = hist["label_td"].values
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    train_acc = model.score(X, y)
    base_rate = y.mean()
    print(f"Calibration trained on {len(y)} player-games. Base TD rate={base_rate:.3f}, train acc={train_acc:.3f}")
    return model, feat_cols


def main():
    hist = pd.read_parquet(os.path.join(DATA_DIR, "player_history.parquet"))
    def_hist = pd.read_parquet(os.path.join(DATA_DIR, "defense_matchup_history.parquet"))
    hist = add_matchup(hist, def_hist)

    model, feat_cols = train_calibration(hist)

    cur = pd.read_parquet(os.path.join(DATA_DIR, "player_current.parquet"))
    cur = cur[cur["position_group"].isin(["RB", "WR", "TE", "QB"])].copy()

    # ---- baseline percentiles computed within current RB/WR/TE pool ----
    skill = cur[cur["position_group"].isin(["RB", "WR", "TE"])].copy()
    for c in ["red_zone_role", "volume_pg", "goal_line_pg", "tr_target_share", "td_rate"]:
        skill[f"{c}_pct"] = pct_rank(skill[c])

    skill["baseline_score"] = sum(skill[f"{c}_pct"] * w for c, w in BASELINE_WEIGHTS.items())

    # ---- opportunity: needs the player's UPCOMING opponent, filled in later
    # by build_site_data.py once the week-2 schedule is joined. For now we
    # keep the raw components and compute the percentile at assembly time.

    # due bonus
    def due_bonus(row):
        if row["red_zone_role_pct"] >= 80 and row["games_since_td"] >= 3:
            return 8
        if row["volume_pg"] >= 12 and row["games_since_td"] >= 4:
            return 5
        return 0

    skill["due_bonus"] = skill.apply(due_bonus, axis=1)

    # ---- TD debt / zone grid (uses trailing zone touch sums already in tr_ cols) ----
    zone_rates = json.load(open(os.path.join(DATA_DIR, "zone_rates.json")))
    for z in ZONES:
        rr = zone_rates["rush"].get(z, 0)
        tr = zone_rates["target"].get(z, 0)
        skill[f"xtd_{z}"] = skill[f"tr_rush_{z}"] * rr + skill[f"tr_tgt_{z}"] * tr
    skill["expected_td"] = sum(skill[f"xtd_{z}"] for z in ZONES)
    skill["scored_td"] = skill["tr_total_td"]
    skill["td_debt"] = skill["expected_td"] - skill["scored_td"]

    out_path = os.path.join(DATA_DIR, "player_scores.parquet")
    skill.to_parquet(out_path)
    print("saved", out_path, skill.shape)

    # persist the trained calibration model coefficients for reuse downstream
    coef = dict(zip([f"{c}_pct" for c in feat_cols], model.coef_[0].tolist()))
    coef["intercept"] = float(model.intercept_[0])
    with open(os.path.join(DATA_DIR, "td_calibration.json"), "w") as f:
        json.dump({"coef": coef, "feat_cols": feat_cols}, f, indent=2)
    print("saved calibration", coef)


if __name__ == "__main__":
    main()
