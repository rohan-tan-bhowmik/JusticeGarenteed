#!/usr/bin/env python3
"""
Fetches a match timeline via Riot’s API, records each participant’s XP per frame,
converts those XP values into corresponding levels (1–18), and saves a JSON file
that maps each champion to their level at each whole‐minute mark.
"""

import json
from riotwatcher import LolWatcher, ApiError

# ─────────────────────────────────────────────────────────────────────────────
# 1) XP thresholds: cumulative XP required to reach each level (1 through 18)
# ─────────────────────────────────────────────────────────────────────────────
xp_thresholds = [
    0,      # Level 1
    280,    # Level 2
    660,    # Level 3
    1140,   # Level 4
    1720,   # Level 5
    2400,   # Level 6
    3180,   # Level 7
    4060,   # Level 8
    5040,   # Level 9
    6120,   # Level 10
    7300,   # Level 11
    8580,   # Level 12
    9960,   # Level 13
    11440,  # Level 14
    13020,  # Level 15
    14700,  # Level 16
    16480,  # Level 17
    18360   # Level 18
]

def xp_to_level(xp: int) -> int:
    """
    Given a raw XP value, return the corresponding level (1–18) by finding
    the highest index i where xp_thresholds[i] <= xp. If xp < 0, returns 1.
    """
    for i in range(len(xp_thresholds) - 1, -1, -1):
        if xp >= xp_thresholds[i]:
            return i + 1
    return 1  # Default to level 1 if xp is negative or below threshold

# ─────────────────────────────────────────────────────────────────────────────
# 2) Riot API configuration
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = "RGAPI-53f2229c-5fc5-4133-917d-66ac1d6ae3c2"
watcher = LolWatcher(API_KEY)

platform = "NA1"             # e.g. NA1, EUW1, KR, etc.
routing_region = "americas"  # “americas” endpoint handles NA/BR/LA1/LA2

match_id = "NA1_5297785121"  # ← your specific game ID

# ─────────────────────────────────────────────────────────────────────────────
# 3) Fetch match details to map participantId → championName
# ─────────────────────────────────────────────────────────────────────────────
try:
    match_info = watcher.match.by_id(platform, match_id)
except ApiError as e:
    print(f"Failed to fetch match info: {e}")
    raise

participant_to_champion = {
    p["participantId"]: p["championName"]
    for p in match_info["info"]["participants"]
}

# ─────────────────────────────────────────────────────────────────────────────
# 4) Fetch timeline for that match
# ─────────────────────────────────────────────────────────────────────────────
try:
    timeline = watcher.match.timeline_by_match(routing_region, match_id)
except ApiError as e:
    print(f"Failed to fetch timeline: {e}")
    raise

# ─────────────────────────────────────────────────────────────────────────────
# 5) Build raw XP‐over‐time arrays:
#    For each frame, timeline["info"]["frames"] is a dict with:
#       "timestamp" (milliseconds)
#       "participantFrames": { "1": { xp: …, … }, … "10": { xp: … } }
# ─────────────────────────────────────────────────────────────────────────────
xp_over_time = { pid: [] for pid in range(1, 11) }
timestamps_ms = []

for frame in timeline["info"]["frames"]:
    timestamps_ms.append(frame["timestamp"])
    for pid_str, pframe in frame["participantFrames"].items():
        pid = int(pid_str)
        xp_over_time[pid].append(pframe["xp"])

# Convert timestamps to seconds
timestamps_s = [t // 1000 for t in timestamps_ms]

# ─────────────────────────────────────────────────────────────────────────────
# 6) Convert raw XP lists into level lists per participant
# ─────────────────────────────────────────────────────────────────────────────
levels_over_time = {}
for pid, xp_list in xp_over_time.items():
    level_list = [xp_to_level(xp) for xp in xp_list]
    levels_over_time[pid] = level_list

# ─────────────────────────────────────────────────────────────────────────────
# 7) Remap keys from participantId → championName
# ─────────────────────────────────────────────────────────────────────────────
level_by_champion = {
    participant_to_champion[pid]: levels_over_time[pid]
    for pid in levels_over_time
}

# ─────────────────────────────────────────────────────────────────────────────
# 8) For each champion, sample the level at each whole‐minute mark
# ─────────────────────────────────────────────────────────────────────────────
# Determine the maximum minute index in the match
if timestamps_s:
    max_minute = timestamps_s[-1] // 60
else:
    max_minute = 0

levels_by_minute = {}
for champ, level_list in level_by_champion.items():
    minute_map = {}
    for m in range(0, max_minute + 1):
        # Find the first frame index where timestamp >= m * 60
        idx = next((i for i, t in enumerate(timestamps_s) if t >= m * 60), len(timestamps_s) - 1)
        minute_map[str(m)] = level_list[idx]
    levels_by_minute[champ] = minute_map

# ─────────────────────────────────────────────────────────────────────────────
# 9) Save to JSON file
# ─────────────────────────────────────────────────────────────────────────────
output_path = "levels_by_minute.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(levels_by_minute, f, indent=2)

print(f"Saved levels‐by‐minute data to {output_path}")
