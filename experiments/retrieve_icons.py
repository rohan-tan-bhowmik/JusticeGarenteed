#!/usr/bin/env python3
"""
Download every League-of-Legends item icon on Fandom.

Creates ./item_icons and saves one <name>.png per file.

Requires: requests, beautifulsoup4, tqdm (optional progress bar)
"""

import os, re, time, urllib.parse as ul
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

START_URL = "https://leagueoflegends.fandom.com/wiki/Category:Item_icons?from=R"
OUT_DIR   = "item_icons"
DELAY_S   = 0.2                       # be nice to the server
HEADERS   = {"User-Agent": "IconScraper/1.0 (+https://github.com/you)"}

# strip trailing "scale-to-width-down/<n>" part but KEEP .png
SCALE_RE  = re.compile(r"/scale-to-width-down/\d+.*$")

os.makedirs(OUT_DIR, exist_ok=True)
session = requests.Session()
session.headers.update(HEADERS)

def get_soup(url: str) -> BeautifulSoup:
    for attempt in range(3):
        resp = session.get(url, timeout=15)
        if resp.ok:
            return BeautifulSoup(resp.text, "html.parser")
        time.sleep(2 * attempt + 1)
    resp.raise_for_status()

def clean_img_url(raw: str) -> str:
    """Remove scaling part, keep direct .png URL."""
    return SCALE_RE.sub("", raw)

def scrape_page(url: str) -> tuple[list[tuple[str,str]], str|None]:
    soup = get_soup(url)

    # 1. collect thumbnails
    icons: list[tuple[str,str]] = []
    for link in soup.select("img.category-page__member-thumbnail"):
        src = link.get("data-src") or link.get("src")
        if not src or ".png" not in src:
            continue
        src = src.split("/revision")[0]
        full = clean_img_url(src)
        fname = ul.unquote(os.path.basename(full).split("?")[0])
        icons.append((full, fname))

    return icons

def download(url: str, path: str):
    tmp = path + ".part"
    with session.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(8192):
                fh.write(chunk)
    os.replace(tmp, path)

def main():
    url = START_URL
    seen = set()
    icons = scrape_page(url)
    for src, name in tqdm(icons, desc="page", leave=False):
        # print(src, name)
        if name in seen:
            continue
        seen.add(name)
        out_path = os.path.join(OUT_DIR, name)
        if not os.path.exists(out_path):
            try:
                download(src, out_path)
                time.sleep(DELAY_S)
            except Exception as e:
                print(f"[warn] {name}: {e}")

if __name__ == "__main__":
    main()
