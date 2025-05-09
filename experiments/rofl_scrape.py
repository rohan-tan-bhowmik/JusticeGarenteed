#!/usr/bin/env python3
"""
Replay → JSON scraper
• launches Riot Client → LeagueClientUx
• opens a .rofl in the official client
• runs a console scraper while the replay plays
• prints all child-process output and waits for 'q' to quit
"""

import os, time, subprocess, shutil, socket, requests, traceback, sys
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────
RIOT_CLIENT  = r"C:\Riot Games\Riot Client\RiotClientServices.exe"
LOCKFILE     = Path(r"C:\Riot Games\League of Legends\lockfile")
REPLAY_DIR   = Path(rf"{os.environ['USERPROFILE']}\Documents\League of Legends\Replays")

CONSOLE_EXE  = r"C:\path\to\ConsoleApplication.exe"          # ← your LView fork
CONSOLE_CWD  = r"C:\root\dir\of\ConsoleApplication.exe"

JSON_OUT_DIR = Path("jsons")
REPLAY_SPEED = 16           # 64 obs/s ÷ 4  → 16× speed
SCRAPE_SECS  = 300          # scrape first five minutes
PAUSE_SECS   = 3            # delay before scraper starts
# ──────────────────────────────────────────────────────────────────────────

import urllib3; urllib3.disable_warnings()

def wait_tcp(port: int, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if not s.connect_ex(("127.0.0.1", port)):
                return
        time.sleep(.5)
    raise RuntimeError(f"port {port} never opened")

def open_with_client(rofl_file: Path):
    """Launch Riot Client and open the replay inside LeagueClientUx."""
    # 0️⃣ ensure file lives in the official replay folder
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    local_path = REPLAY_DIR / rofl_file.name
    if rofl_file.resolve() != local_path.resolve():
        shutil.copy2(rofl_file, local_path)

    game_id = rofl_file.stem.split('-')[1]  # "NA1-123.rofl" → "123"

    # 1️⃣ spawn Riot Client (this boots Vanguard + UX)
    subprocess.Popen(
        [RIOT_CLIENT, "--launch-product=league_of_legends", "--launch-patchline=live"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2️⃣ wait for lockfile, read port/token
    while not LOCKFILE.exists(): time.sleep(.5)
    _, _, port, token, _ = LOCKFILE.read_text().split(':')
    port = int(port); wait_tcp(port)

    sess = requests.Session()
    sess.verify = False
    sess.auth   = ("riot", token)

    # 3️⃣ force a folder scan so the client registers fresh files
    sess.post(f"https://127.0.0.1:{port}/lol-replays/v1/rofls/scan")

    # 4️⃣ launch replay
    r = sess.post(
        f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{game_id}/watch",
        json={"gameId": int(game_id)})
    r.raise_for_status()
    if r.content:
        print("▶ replay accepted:", r.json())
    else:
        print(f"▶ replay accepted (HTTP {r.status_code}, empty body)")

def run_scraper(out_json: Path, seconds: int, speed: int):
    cmd = [CONSOLE_EXE, str(out_json), str(seconds), str(speed)]
    proc = subprocess.run(cmd, cwd=CONSOLE_CWD,
                          capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr: print(proc.stderr, file=sys.stderr)
    if proc.returncode:
        raise RuntimeError(f"scraper exited {proc.returncode}")

def main():
    JSON_OUT_DIR.mkdir(exist_ok=True)
    replays = [Path("rofls/NA1-5275875289.rofl")]          # put your files here

    for idx, rf in enumerate(replays, 1):
        if not rf.exists(): raise FileNotFoundError(rf)
        print(f"\n── {idx}/{len(replays)} {rf.name} ───")
        open_with_client(rf)
        time.sleep(PAUSE_SECS)
        out_json = JSON_OUT_DIR / f"{rf.stem}.json"
        run_scraper(out_json, SCRAPE_SECS, REPLAY_SPEED)
        print(f"✓ saved {out_json}")

    input("\nDone. Press <Enter> to quit…")

if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); input("Press <Enter> to exit…")
