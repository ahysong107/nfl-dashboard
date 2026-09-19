"""Final assembly: join upcoming-week schedule, compute per-opponent
Opportunity Score + calibrated Anytime-TD probability, attach TD-debt zone
grids, box count clones, prop-yardage ladders, and reason tags. Writes
site/data.json for the dashboard.
"""
import sys, os, json
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from scipy.stats import norm
from lib_data import DATA_DIR, ZONES, load_schedules

SITE_DIR = os.path.join(os.path.dirname(__file__), "..", "site")
os.makedirs(SITE_DIR, exist_ok=True)


def detect_target_week(sched):
    """The next fully-upcoming week (zero completed games) in the current
    season -- so this script needs no manual edits from one week to the
    next. Assumes a Tue/Fri-or-earlier refresh cadence, i.e. it always runs
    after the prior week's Monday night game has finished."""
    now = datetime.now(timezone.utc)
    season_guess = now.year if now.month >= 3 else now.year - 1
    s = sched[sched["season"] == season_guess]
    if s.empty:
        season_guess = int(sched["season"].max())
        s = sched[sched["season"] == season_guess]

    completed = s.groupby("week")["home_score"].apply(lambda x: x.notna().sum())
    total = s.groupby("week")["home_score"].size()
    fully_upcoming = completed[completed == 0]
    if len(fully_upcoming) == 0:
        target_week = int(s["week"].max())  # season is over / no upcoming week found
    else:
        target_week = int(fully_upcoming.index.min())
    return season_guess, target_week


def pct_rank(series):
    return series.rank(pct=True, method="average") * 100


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def main():
    sched = load_schedules()
    TARGET_SEASON, TARGET_WEEK = detect_target_week(sched)
    print(f"Auto-detected target: season {TARGET_SEASON}, week {TARGET_WEEK}")
    week_games = sched[(sched["season"] == TARGET_SEASON) & (sched["week"] == TARGET_WEEK)].copy()
    print(f"Week {TARGET_WEEK} games:", len(week_games))

    team_opponent = {}
    for _, g in week_games.iterrows():
        team_opponent[g["home_team"]] = {"opp": g["away_team"], "home": True, "game_id": g["game_id"], "gameday": str(g["gameday"])}
        team_opponent[g["away_team"]] = {"opp": g["home_team"], "home": False, "game_id": g["game_id"], "gameday": str(g["gameday"])}

    scores = pd.read_parquet(os.path.join(DATA_DIR, "player_scores.parquet"))
    props = pd.read_parquet(os.path.join(DATA_DIR, "player_props.parquet"))
    boxc = pd.read_parquet(os.path.join(DATA_DIR, "player_boxcount.parquet"))
    def_cur = pd.read_parquet(os.path.join(DATA_DIR, "defense_matchup_current.parquet"))
    calib = json.load(open(os.path.join(DATA_DIR, "td_calibration.json")))

    # depth_charts is refreshed daily (unlike the week-1-only roster snapshot),
    # so it's the authoritative source for CURRENT team (catches trades) and
    # CURRENT starter (pos_rank==1) -- this is what catches a traded/benched
    # player still showing up because his season-long stat trail is on his
    # old team, or a returning-from-injury starter losing out to a backup
    # who has more recent (but no-longer-relevant) games logged.
    dc = pd.read_parquet(os.path.join(DATA_DIR, "depth_charts_2026.parquet"))
    dc_latest_dt = dc["dt"].max()
    dc_cur = dc[dc["dt"] == dc_latest_dt].dropna(subset=["gsis_id"])
    dc_cur = dc_cur.sort_values("pos_rank").drop_duplicates("gsis_id", keep="first")
    current_team = dict(zip(dc_cur["gsis_id"], dc_cur["team"]))
    current_rank = dict(zip(dc_cur["gsis_id"], dc_cur["pos_rank"]))
    print(f"depth chart snapshot from {dc_latest_dt}: {len(current_team)} players with a current team/rank")

    def apply_current_team(df):
        df = df[df["player_id"].isin(current_team)].copy()
        df["team"] = df["player_id"].map(current_team)
        df["current_pos_rank"] = df["player_id"].map(current_rank)
        return df

    # a player who's fallen down the CURRENT depth chart (benched, injured,
    # beaten out) still shows up on trailing box-score stats until his role
    # change shows up in several games of data -- the depth chart already
    # reflects it today, even before an injury report catches up. Cut
    # anyone clearly buried at their position, regardless of why.
    DEPTH_RELEVANCE = {"RB": 3, "WR": 3, "TE": 2, "QB": 2}

    def apply_depth_relevance(df):
        threshold = df["position_group"].map(DEPTH_RELEVANCE).fillna(3)
        return df[df["current_pos_rank"] <= threshold].copy()

    # injury report for the target week -- filed progressively (practice
    # reports Wed-Fri, then a final game-status designation), so this may be
    # empty until the week's reports actually start coming in. "Out" players
    # are dropped outright since that's a near-certain scratch; Doubtful/
    # Questionable are kept but flagged rather than removed, since most
    # Questionable players (and plenty of Doubtful ones) do end up playing.
    inj_path = os.path.join(DATA_DIR, "injuries_2026.parquet")
    injury_status = {}
    if os.path.exists(inj_path):
        inj = pd.read_parquet(inj_path)
        inj_wk = inj[(inj["season"] == TARGET_SEASON) & (inj["week"] == TARGET_WEEK) & inj["report_status"].notna()]
        injury_status = dict(zip(inj_wk["gsis_id"], inj_wk["report_status"]))
        print(f"injury report for week {TARGET_WEEK}: {len(injury_status)} players listed "
              f"({sum(1 for v in injury_status.values() if v == 'Out')} Out, will be excluded)")
    else:
        print("no injuries_2026.parquet found -- run scripts/fetch_injuries.py; skipping injury filter")

    def apply_injury_status(df):
        df = df.copy()
        df["injury_status"] = df["player_id"].map(injury_status)
        return df[df["injury_status"] != "Out"].copy()

    # only currently-rostered players (per today's depth chart) whose team plays this week
    scores = apply_current_team(scores)
    scores = apply_depth_relevance(scores)
    scores = apply_injury_status(scores)
    scores = scores[scores["team"].isin(team_opponent.keys())].copy()
    scores["opponent_next"] = scores["team"].map(lambda t: team_opponent[t]["opp"])
    scores["game_id_next"] = scores["team"].map(lambda t: team_opponent[t]["game_id"])
    scores["is_home_next"] = scores["team"].map(lambda t: team_opponent[t]["home"])

    dm = def_cur.rename(columns={"defteam": "opponent_next"})[["opponent_next", "position_group", "td_allowed_pg", "yards_allowed_pg"]]
    scores = scores.merge(dm, on=["opponent_next", "position_group"], how="left")
    scores["td_allowed_pg"] = scores["td_allowed_pg"].fillna(scores["td_allowed_pg"].median())

    # Opportunity Score = percentile of matchup softness among this week's matchups (higher TDs allowed = juicier)
    scores["opportunity_score"] = pct_rank(scores["td_allowed_pg"])
    scores["venom_score"] = (scores["baseline_score"] * 0.60 + scores["opportunity_score"] * 0.40 + scores["due_bonus"]).clip(1, 100)

    # calibrated probability using the SAME feature percentiles as training (recomputed within this week's live pool)
    feat_cols = calib["feat_cols"]
    coef = calib["coef"]
    for c in ["red_zone_role", "volume_pg", "goal_line_pg", "tr_target_share", "td_rate"]:
        scores[f"{c}_pct"] = pct_rank(scores[c])
    scores["td_allowed_pg_pct"] = pct_rank(scores["td_allowed_pg"])
    z = coef["intercept"]
    for c in feat_cols:
        z = z + coef[f"{c}_pct"] * scores[f"{c}_pct"]
    scores["td_probability"] = sigmoid(z)

    # ---- prop boards (pass yards/TDs, rush yards, receptions) are independent
    # of the anytime-TD skill-position scoring table, since QBs never appear
    # in `scores` (baseline formula is RB/WR/TE-shaped) but must appear here.
    props = apply_current_team(props)
    props = apply_depth_relevance(props)
    props = apply_injury_status(props)
    props = props[props["team"].isin(team_opponent.keys())].copy()
    props["opponent_next"] = props["team"].map(lambda t: team_opponent[t]["opp"])
    props["game_id_next"] = props["team"].map(lambda t: team_opponent[t]["game_id"])
    props["is_home_next"] = props["team"].map(lambda t: team_opponent[t]["home"])

    # QB is a hard single-starter position: never show a backup's stale
    # trailing numbers instead of the current QB1 (e.g. a returning starter
    # vs. a backup who made a spot start more recently).
    is_qb_prop_row = props["position_group"] == "QB"
    props = props[~is_qb_prop_row | (props["current_pos_rank"] == 1)].copy()

    # attach the player's ACTUAL upcoming opponent's matchup rate (not a league avg)
    dm_qb = def_cur[def_cur["position_group"] == "QB"].rename(columns={"defteam": "opponent_next"})
    dm_qb = dm_qb[["opponent_next", "td_allowed_pg", "yards_allowed_pg"]].rename(
        columns={"td_allowed_pg": "opp_pass_td_allowed_pg", "yards_allowed_pg": "opp_pass_yards_allowed_pg"})
    props = props.merge(dm_qb, on="opponent_next", how="left")

    dm_pos = def_cur.rename(columns={"defteam": "opponent_next"})[["opponent_next", "position_group", "yards_allowed_pg"]]
    dm_pos = dm_pos.rename(columns={"yards_allowed_pg": "opp_yards_allowed_pg"})
    props = props.merge(dm_pos, on=["opponent_next", "position_group"], how="left")

    # ---- reason tags ----
    def reason_tags(r):
        tags = []
        if pd.notna(r.get("injury_status")):
            tags.append(r["injury_status"].upper())
        if r["red_zone_role_pct"] >= 80:
            tags.append("Elite Red Zone Role")
        if r["goal_line_pg"] >= 1.2:
            tags.append("Goal Line Back")
        if r["tr_target_share_pct"] >= 80:
            tags.append("Target Monster")
        if r["volume_pg_pct"] >= 80:
            tags.append("High Volume")
        if r["td_allowed_pg_pct"] >= 75:
            tags.append("Soft TD Defense")
        if r["due_bonus"] > 0:
            tags.append("Due For TD")
        elif r["td_debt"] < -1.0:
            tags.append("Regression Risk")
        return tags

    scores["reason_tags"] = scores.apply(reason_tags, axis=1)

    # ---- box count clone lookup ----
    boxc_by_player = {}
    for pid, grp in boxc.groupby("rusher_player_id"):
        boxc_by_player[pid] = {
            row["bucket"]: {
                "attempts": int(row["attempts"]), "ypc": row["ypc"],
                "success_rate": row["success_rate"], "td_rate": row["td_rate"],
                "season": int(row["season"]),
            } for _, row in grp.iterrows()
        }

    games_out = {}
    for _, g in week_games.iterrows():
        games_out[g["game_id"]] = {
            "game_id": g["game_id"], "home_team": g["home_team"], "away_team": g["away_team"],
            "gameday": str(g["gameday"]), "gametime": str(g.get("gametime", "")),
            "players": {"anytime_td": [], "pass_yards": [], "pass_tds": [], "rush_yards": [], "receptions": []},
        }

    def make_zone_grid(r):
        grid = []
        labels = {"goal_line": "Goal Line (in 5)", "red_zone": "Red Zone (6-20)", "fringe": "Fringe (21-40)", "open_field": "Open Field"}
        for z in ZONES:
            carries = int(r[f"tr_rush_{z}"])
            targets = int(r[f"tr_tgt_{z}"])
            grid.append({
                "zone": labels[z], "carries": carries, "targets": targets,
                "xtd": round(float(r[f"xtd_{z}"]), 2),
            })
        return grid

    def player_payload(r, prop=None):
        pid = r["player_id"]
        payload = {
            "player_id": pid,
            "name": r["player_display_name"],
            "position": r["position"],
            "team": r["team"],
            "opponent": r["opponent_next"],
            "is_home": bool(r["is_home_next"]),
            "reason_tags": r["reason_tags"],
            "injury_status": r.get("injury_status") if pd.notna(r.get("injury_status")) else None,
            "baseline": {
                "red_zone_role": round(float(r["red_zone_role"]) * 100, 1),
                "volume_pg": round(float(r["volume_pg"]), 1),
                "goal_line_pg": round(float(r["goal_line_pg"]), 1),
                "target_share": round(float(r["tr_target_share"]) * 100, 1),
                "td_rate": round(float(r["td_rate"]) * 100, 1),
            },
            "opportunity": {
                "matchup_td_pg": round(float(r["td_allowed_pg"]), 2),
                "opportunity_score": round(float(r["opportunity_score"]), 1),
                "games_since_td": int(r["games_since_td"]),
            },
            "venom_score": round(float(r["venom_score"]), 1),
            "td_probability": round(float(r["td_probability"]) * 100, 1),
            "due_bonus": int(r["due_bonus"]),
            "td_debt": round(float(r["td_debt"]), 2),
            "expected_td": round(float(r["expected_td"]), 2),
            "scored_td": round(float(r["scored_td"]), 1),
            "zone_grid": make_zone_grid(r),
            "box_count": boxc_by_player.get(pid, None),
        }
        return payload

    def prop_payload(r, prop_key, reason_tags):
        mean = r.get(f"{prop_key}_mean")
        std = r.get(f"{prop_key}_std")
        if pd.isna(mean):
            return None
        thresholds = sorted(set([
            round((mean - 0.75 * std) * 2) / 2,
            round(mean * 2) / 2,
            round((mean + 0.75 * std) * 2) / 2,
        ]))
        thresholds = [t for t in thresholds if t > 0]
        lines = []
        for t in thresholds:
            prob_over = float(1 - norm.cdf(t, loc=mean, scale=std))
            lines.append({"line": t, "prob_over": round(prob_over * 100, 1)})
        opp_allowed = r.get("opp_pass_yards_allowed_pg") if prop_key == "pass_yards" else (
            r.get("opp_pass_td_allowed_pg") if prop_key == "pass_tds" else r.get("opp_yards_allowed_pg"))
        league_avg = r.get(f"{prop_key}_league_avg_allowed")
        return {
            "player_id": r["player_id"], "name": r["player_display_name"], "position": r["position"],
            "team": r["team"], "opponent": r["opponent_next"], "is_home": bool(r["is_home_next"]),
            "projected": round(float(mean), 1), "std": round(float(std), 1),
            "games_sample": int(r.get(f"{prop_key}_games", 0)),
            "opponent_allowed_pg": round(float(opp_allowed), 2) if pd.notna(opp_allowed) else None,
            "league_avg_allowed": round(float(league_avg), 2) if pd.notna(league_avg) else None,
            "lines": lines,
            "reason_tags": reason_tags,
            "injury_status": r.get("injury_status") if pd.notna(r.get("injury_status")) else None,
        }

    def prop_reason_tags(r, prop_key, pct_within_pool):
        tags = []
        if pd.notna(r.get("injury_status")):
            tags.append(r["injury_status"].upper())
        if pct_within_pool >= 80:
            tags.append("High Volume")
        opp_allowed = r.get("opp_pass_yards_allowed_pg") if prop_key == "pass_yards" else (
            r.get("opp_pass_td_allowed_pg") if prop_key == "pass_tds" else r.get("opp_yards_allowed_pg"))
        league_avg = r.get(f"{prop_key}_league_avg_allowed")
        if pd.notna(opp_allowed) and pd.notna(league_avg) and league_avg:
            ratio = opp_allowed / league_avg
            if ratio >= 1.12:
                tags.append("Soft Matchup")
            elif ratio <= 0.88:
                tags.append("Tough Matchup")
        return tags

    for _, r in scores.iterrows():
        gid = r["game_id_next"]
        if gid not in games_out:
            continue
        games_out[gid]["players"]["anytime_td"].append(player_payload(r))

    for prop_key in ["pass_yards", "pass_tds", "rush_yards", "receptions"]:
        pool = props[props[f"{prop_key}_mean"].notna()].copy()
        if len(pool) == 0:
            continue
        pool["_pct"] = pct_rank(pool[f"{prop_key}_mean"])
        for _, r in pool.iterrows():
            gid = r["game_id_next"]
            if gid not in games_out:
                continue
            tags = prop_reason_tags(r, prop_key, r["_pct"])
            payload = prop_payload(r, prop_key, tags)
            if payload:
                games_out[gid]["players"][prop_key].append(payload)

    # sort full (untrimmed) lists -- used for rankings + parlay leg pool below,
    # before we cap each game card down to its top 12 for display.
    for gid, g in games_out.items():
        g["players"]["anytime_td"].sort(key=lambda p: -p["td_probability"])
        for k in ["pass_yards", "pass_tds", "rush_yards", "receptions"]:
            g["players"][k].sort(key=lambda p: -p["projected"])

    # ---- league-wide rankings: same players, flattened across all 16 games ----
    rankings = {}
    for cat in ["anytime_td", "pass_yards", "pass_tds", "rush_yards", "receptions"]:
        flat = []
        for g in games_out.values():
            for p in g["players"][cat]:
                entry = dict(p)
                entry["game_id"] = g["game_id"]
                entry["game_label"] = f"{g['away_team']} @ {g['home_team']}"
                flat.append(entry)
        if cat == "anytime_td":
            flat.sort(key=lambda p: -p["td_probability"])
        else:
            flat.sort(key=lambda p: -p["projected"])
        rankings[cat] = flat[:25]

    # ---- system parlays: built from the same probability-scored legs, no odds feed involved ----
    def build_leg_pool():
        legs = []
        for g in games_out.values():
            label = f"{g['away_team']} @ {g['home_team']}"
            for p in g["players"]["anytime_td"]:
                legs.append({
                    "player_id": p["player_id"], "name": p["name"], "team": p["team"],
                    "opponent": p["opponent"], "game_id": g["game_id"], "game_label": label,
                    "category": "Anytime TD", "description": f"{p['name']} Anytime TD",
                    "prob": p["td_probability"] / 100,
                })
            cat_labels = {"pass_yards": "Pass Yards", "pass_tds": "Pass TDs", "rush_yards": "Rush Yards", "receptions": "Receptions"}
            min_projected = {"pass_yards": 150, "pass_tds": 0.8, "rush_yards": 35, "receptions": 2.5}
            for key, label_name in cat_labels.items():
                for p in g["players"][key]:
                    if p["games_sample"] < 3:
                        continue  # too little trailing data to trust a probability estimate
                    if p["projected"] < min_projected[key]:
                        continue  # keep parlay legs to players with a real weekly role, not committee scraps clearing a trivial personal bar
                    for line in p["lines"]:
                        legs.append({
                            "player_id": p["player_id"], "name": p["name"], "team": p["team"],
                            "opponent": p["opponent"], "game_id": g["game_id"], "game_label": label,
                            "category": label_name, "description": f"{p['name']} Over {line['line']} {label_name.lower()}",
                            "prob": line["prob_over"] / 100,
                        })
        return legs

    def fair_odds(p):
        p = min(max(p, 0.01), 0.99)
        if p <= 0.5:
            american = round(100 * (1 - p) / p)
            return f"+{american}"
        else:
            american = round(-100 * p / (1 - p))
            return str(american)

    def pick_parlay(pool, n_legs, offset, exclude_desc):
        chosen, used_players = [], set()
        i = offset
        attempts = 0
        while len(chosen) < n_legs and attempts < len(pool) * 2:
            leg = pool[i % len(pool)]
            i += 1
            attempts += 1
            if leg["player_id"] in used_players or leg["description"] in exclude_desc:
                continue
            chosen.append(leg)
            used_players.add(leg["player_id"])
        return chosen

    def build_tier(pool, leg_counts, tier_name):
        parlays = []
        exclude_desc = set()
        for idx, n_legs in enumerate(leg_counts):
            legs = pick_parlay(pool, n_legs, offset=idx * max(1, len(pool) // (len(leg_counts) + 1)), exclude_desc=exclude_desc)
            if len(legs) < n_legs:
                continue
            for l in legs:
                exclude_desc.add(l["description"])
            combined = 1.0
            for l in legs:
                combined *= l["prob"]
            games_used = {l["game_id"] for l in legs}
            parlays.append({
                "tier": tier_name,
                "legs": [{k: v for k, v in l.items() if k != "player_id"} for l in legs],
                "combined_prob": round(combined * 100, 1),
                "fair_odds": fair_odds(combined),
                "same_game": len(games_used) < len(legs),
            })
        return parlays

    all_legs = build_leg_pool()
    all_legs.sort(key=lambda l: -l["prob"])
    safe_pool = [l for l in all_legs if l["prob"] >= 0.68]
    standard_pool = [l for l in all_legs if 0.42 <= l["prob"] < 0.68]
    bold_pool = [l for l in all_legs if 0.15 <= l["prob"] < 0.42]

    parlays = {
        "safe": build_tier(safe_pool, [2, 3, 2], "Safe"),
        "standard": build_tier(standard_pool, [3, 3, 4], "Standard"),
        "bold": build_tier(bold_pool, [3, 4, 4], "Bold"),
    }

    # trim each game card's display lists down to a manageable top 12
    for gid, g in games_out.items():
        for k in g["players"]:
            g["players"][k] = g["players"][k][:12]

    out = {
        "season": TARGET_SEASON, "week": TARGET_WEEK,
        "generated_note": "Stats from nflverse (play-by-play, weekly, FTN charting). Opportunity Score uses real defensive matchup data; Vegas implied team totals are not yet connected.",
        "games": list(games_out.values()),
        "rankings": rankings,
        "parlays": parlays,
    }

    out_path = os.path.join(SITE_DIR, "data.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("saved", out_path)
    print("games with players:", sum(1 for g in out["games"] if g["players"]["anytime_td"]))


if __name__ == "__main__":
    main()
