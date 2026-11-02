#!/usr/bin/env python3
"""
Refresh the NFL Simulator page with the latest nflverse schedule and results.

What it does
  1. Downloads games.csv, teams.csv and teamcolors.csv from nflverse/nfldata
     (or reads them from a local folder with --data-dir).
  2. Picks the most recent season that has a regular-season schedule.
  3. Fits each team's offense and defense rating from that season's scores
     plus betting spreads/totals (why: after a few weeks, the market knows
     about injuries and QB changes that the scores don't yet show).
  4. Runs sanity checks and STOPS without touching the page if anything looks off
     (wrong game count, a team missing games, a score on a future date, etc.).
  5. Swaps the new data into simulator.html between the DATA_START/DATA_END markers.
     Nothing else in the page changes.

Usage
  python update_simulator.py                       # updates ./simulator.html
  python update_simulator.py --html path/to/simulator.html
  python update_simulator.py --data-dir nfldata/data   # use a local clone instead of downloading
  python update_simulator.py --season 2026         # force a season

Needs: Python 3.9+, pandas, numpy  (pip install pandas numpy)
"""
import argparse, json, re, sys, urllib.request, io
from datetime import date
import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/"

DIVISIONS = {
    "AFC East": "BUF MIA NE NYJ", "AFC North": "BAL CIN CLE PIT",
    "AFC South": "HOU IND JAX TEN", "AFC West": "DEN KC LAC LV",
    "NFC East": "DAL NYG PHI WAS", "NFC North": "CHI DET GB MIN",
    "NFC South": "ATL CAR NO TB", "NFC West": "ARI LA SEA SF",
}
DIV_OF = {t: d for d, ts in DIVISIONS.items() for t in ts.split()}

# Model settings (documented so changes are deliberate)
MARKET_WEIGHT = 4.0   # one betting line counts as much as 4 actual scores
RIDGE = 8.0           # pulls ratings toward average; larger = more cautious early in the season
# Backtest (2012-2025, ~3,700 games, ratings fit only on games before each week): the fitted ratings are
# too compressed -- actual margins ran ~1.25-1.3x the model's predicted margins (slope >1 in 13 of 14
# seasons). Scaling ratings by 1.25 improves win-probability accuracy (log loss .6305 -> ~.627) and
# stays out-of-sample (leave-one-season-out check). See backtest.py.
RATING_SCALE = 1.25
# Backtest (Nov 2): games count less as they get older (half-life 10 weeks), and last season's games are
# blended in at half weight at the start of a season, fading to zero by Week 5. Cut the average miss
# on game margins from ~13.40 to ~13.29 and week-1 error from ~13.8 to ~13.1 (13.35 -> 13.21 with the
# x1.25 scale on top). See BACKTEST_RESULTS.md.
HALF_LIFE_WEEKS = 10.0
PRIOR_SEASON_WEIGHT = 0.5
PRIOR_FADE_WEEKS = 4.0


def load(name, data_dir):
    if data_dir:
        return pd.read_csv(f"{data_dir.rstrip('/')}/{name}")
    with urllib.request.urlopen(RAW + name, timeout=60) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def fail(msg):
    print(f"STOPPED, page not changed: {msg}", file=sys.stderr)
    sys.exit(1)


def fit_ratings(s, teams, now_week, prev=None):
    ix = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    rows, y, w = [], [], []

    def add(team, opp, pts, home, weight):
        r = np.zeros(2 * n + 2)
        r[ix[team]] = 1          # offense of scoring team
        r[n + ix[opp]] = -1      # defense of opponent
        r[2 * n] = 1             # league-average points
        r[2 * n + 1] = 0.5 * home  # half of home-field edge
        rows.append(r); y.append(pts); w.append(weight)

    # time index in weeks: this season = its week number; last season sits 22 weeks earlier
    cur = s.assign(_t=s.week.astype(float), _sw=1.0)
    parts = [cur]
    carry = PRIOR_SEASON_WEIGHT * max(0.0, 1 - (now_week - 1) / PRIOR_FADE_WEEKS)
    if prev is not None and carry > 0:
        parts.append(prev.assign(_t=prev.week.astype(float) - 22, _sw=carry))
    for _, g in pd.concat(parts, ignore_index=True).iterrows():
        h = 1 if g.location == "Home" else 0
        sw = g._sw * 0.5 ** (max(now_week - g._t, 0) / HALF_LIFE_WEEKS)
        if pd.notna(g.home_score):
            add(g.home_team, g.away_team, g.home_score, h, 1.0 * sw)
            add(g.away_team, g.home_team, g.away_score, -h, 1.0 * sw)
        if pd.notna(g.spread_line) and pd.notna(g.total_line):
            home_implied = (g.total_line + g.spread_line) / 2
            away_implied = (g.total_line - g.spread_line) / 2
            add(g.home_team, g.away_team, home_implied, h, MARKET_WEIGHT * sw)
            add(g.away_team, g.home_team, away_implied, -h, MARKET_WEIGHT * sw)

    if not rows:
        fail("no scores or betting lines yet for this season")
    X, Y, W = np.array(rows), np.array(y), np.array(w)
    lam = np.zeros(2 * n + 2); lam[:2 * n] = RIDGE
    p = np.linalg.solve(X.T @ (X * W[:, None]) + np.diag(lam), X.T @ (W * Y))
    off, de = (p[:n] - p[:n].mean()) * RATING_SCALE, (p[n:2 * n] - p[n:2 * n].mean()) * RATING_SCALE
    return off, de, float(p[2 * n]), float(p[2 * n + 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", default="simulator.html")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--season", type=int, default=None)
    a = ap.parse_args()

    games, tm, tc = (load(f, a.data_dir) for f in ("games.csv", "teams.csv", "teamcolors.csv"))
    reg = games[games.game_type == "REG"]
    season = a.season or int(reg.season.max())
    s = reg[reg.season == season].copy().reset_index(drop=True)
    teams = sorted(s.home_team.unique())

    # ---- sanity checks (prove it before publishing)
    if set(teams) != set(DIV_OF):
        fail(f"team codes don't match the 32 known teams: {sorted(set(teams) ^ set(DIV_OF))}")
    if len(s) != 272:
        fail(f"expected 272 regular-season games for {season}, found {len(s)}")
    per_team = pd.concat([s.home_team, s.away_team]).value_counts()
    if (per_team != 17).any():
        fail(f"some teams don't have 17 games: {per_team[per_team != 17].to_dict()}")
    played = s[s.home_score.notna()]
    if (played.home_score.isna() != played.away_score.isna()).any() or s.away_score.notna().sum() != len(played):
        fail("a game has only one team's score")
    today = date.today().isoformat()
    if (played.gameday > today).any():
        fail("a game dated in the future already has a score")
    if ((played.home_score < 0) | (played.away_score < 0) | (played.home_score > 80) | (played.away_score > 80)).any():
        fail("a score is outside 0-80")

    unplayed = s[s.home_score.isna()]
    now_week = int(unplayed.week.min()) if len(unplayed) else int(s.week.max()) + 1
    prev = reg[reg.season == season - 1].copy()
    prev = prev[prev.home_team.isin(teams) & prev.away_team.isin(teams)]
    off, de, mu, hfa = fit_ratings(s, teams, now_week, prev if len(prev) else None)
    if not (15 < mu < 30 and -1 < hfa < 5 and np.abs(off + de).max() < 15):
        fail(f"ratings look implausible (avg pts {mu:.1f}, home edge {hfa:.1f}, max rating {np.abs(off + de).max():.1f})")

    tm = tm[tm.season == tm.season.max()].set_index("team")
    tc = tc.set_index("team")
    ix = {t: i for i, t in enumerate(teams)}
    data = {
        "season": season,
        "asOf": str(played.gameday.max()) if len(played) else f"{season}-09-01",
        "mu": round(mu, 2), "hfa": round(hfa, 2),
        "teams": [{"id": t, "name": tm.loc[t, "full"], "nick": tm.loc[t, "nickname"], "div": DIV_OF[t],
                   "conf": DIV_OF[t][:3], "c1": tc.loc[t, "color"], "c2": tc.loc[t, "color2"],
                   "off": round(float(off[ix[t]]), 2), "def": round(float(de[ix[t]]), 2)} for t in teams],
        "games": [{"wk": int(g.week), "d": g.gameday, "a": g.away_team, "h": g.home_team,
                   "n": int(g.location != "Home"),
                   "as": None if pd.isna(g.away_score) else int(g.away_score),
                   "hs": None if pd.isna(g.home_score) else int(g.home_score),
                   "sp": None if pd.isna(g.spread_line) else float(g.spread_line)} for _, g in s.iterrows()],
    }

    with open(a.html, encoding="utf-8") as f:
        page = f.read()
    pat = re.compile(r"/\*DATA_START\*/.*?/\*DATA_END\*/", re.S)
    if len(pat.findall(page)) != 1:
        fail(f"couldn't find exactly one DATA_START/DATA_END block in {a.html}")
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = pat.sub(lambda _: f"/*DATA_START*/{blob}/*DATA_END*/", page)
    with open(a.html, "w", encoding="utf-8") as f:
        f.write(page)

    # ---- summary
    rec = {t: [0, 0, 0] for t in teams}
    for _, g in played.iterrows():
        hw, aw = g.home_score > g.away_score, g.away_score > g.home_score
        rec[g.home_team][0 if hw else 1 if aw else 2] += 1
        rec[g.away_team][0 if aw else 1 if hw else 2] += 1
    top = sorted(teams, key=lambda t: -(off[ix[t]] + de[ix[t]]))[:5]
    print(f"Updated {a.html}: {season} season, results through {data['asOf']}, "
          f"{len(played)} played / {len(s) - len(played)} left")
    print(f"League avg {mu:.1f} pts/team, home edge {hfa:.1f}")
    print("Top 5 ratings: " + ", ".join(f"{t} {off[ix[t]] + de[ix[t]]:+.1f} ({'-'.join(map(str, rec[t][:2]))}"
                                       f"{'-' + str(rec[t][2]) if rec[t][2] else ''})" for t in top))


if __name__ == "__main__":
    main()
