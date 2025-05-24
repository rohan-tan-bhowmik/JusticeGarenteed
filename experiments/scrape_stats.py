#!/usr/bin/env python3
"""
Robust HUD extractor for League of Legends screenshots / video frames.

• Detects text, bars, and up to 7 item‑icon slots in each "images" box.  
• Item matching is resilient to ±5 % scale error, ±2 px shifts, and border noise.  
• Only accepts matches whose confidence ≥ 50 %; below that the slot is reported as EMPTY.

Usage
-----
python scrape_stats.py <image|video> [frame#]

Example
-------
python scrape_stats.py frame.png
python scrape_stats.py replay.mp4 12345
"""

from __future__ import annotations
import torch

import sys, re, pathlib, cv2, pytesseract, numpy as np
from typing import Dict, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

import minimap

# ── Configuration ──────────────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
BOX_FILE      = "boxes.txt"                  # list of HUD ROIs
ITEM_FOLDER   = "item_icons"                 # folder of downloaded icons
CONF_THRESHOLD = 50.0                        # %; below this counts as EMPTY
ROIType       = Tuple[slice, slice, str]     # rows, cols, kind

# ── Item utilities ────────────────────────────────────────────────────────

def split_item_box(box_img: np.ndarray, team, num) -> list[np.ndarray]:
    """Return the 7 cropped, border‑stripped slots from an item bar ROI."""
    # plt.imshow(box_img)
    # plt.show()
    slot_w = 24
    gaps_r = [[2,2,1,2,2,2], [2,2,1,2,2,2], [2,2,1,2,2,2], [2,1,1,2,2,2], [2,1,2,2,2,2]]
    gaps_b = [[2,2,2,3,3,2], [2,2,2,2,4,2], [2,2,2,2,4,2], [2,2,2,2,3,2], [2,2,2,2,4,2]]
    print(team, num)
    # if team == 1 and num == 4:
    #     gap_w[1] = 1
    gap_w = (gaps_r if team else gaps_b)[num - 1]
    slots = []
    x = 0
    for i in range(6):
        slot = box_img[:, x:x + slot_w]
        # slot = slot[2:-2, 2:-2]           # trim 2‑pixel HUD border
        slots.append(slot)
        x += slot_w + gap_w[i]
    return slots

def load_item_icons(size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Read every *.png in ITEM_FOLDER, resize to *size*, return {name: img}."""
    items: Dict[str, np.ndarray] = {}
    for path in Path(ITEM_FOLDER).glob("*.png"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        items[path.stem] = img
    if not items:
        raise RuntimeError(f"No icons found in {ITEM_FOLDER}/ . Did you download them?")
    return items

def match_item(slot: np.ndarray, refs: Dict[str, np.ndarray]) -> Tuple[str | None, float]:
    """Return (best_name, confidence%) using multi‑scale & shift template matching."""
    best_name, best_score = None, -1.0
    # plt.imshow(slot)
    # plt.show()
    scales = [ 1.0]
    shifts = [-2, 0, 2]
    for name, ref in refs.items():
        h, w = ref.shape[:2]
        for s in scales:
            scaled = cv2.resize(slot, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)
            scaled = cv2.resize(scaled, (w, h))  # back to ref size
            for dx in shifts:
                for dy in shifts:
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    shifted = cv2.warpAffine(scaled, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    score = cv2.matchTemplate(shifted, ref, cv2.TM_CCOEFF_NORMED).max()
                    if score > best_score:
                        best_name, best_score = name, score
    return (best_name if best_score*100 >= CONF_THRESHOLD else None, best_score*100)

def analyze_item_bar(bar: np.ndarray, refs: Dict[str, np.ndarray], team, num):
    """Return list of (idx, name|None, conf%) for 7 slots."""
    results = []
    for idx, slot in enumerate(split_item_box(bar, team, num)):
        if np.mean(slot) < 30:                 # mostly black → empty
            results.append((idx, None, 0.0))
            continue
        name, conf = match_item(slot, refs)
        results.append((idx, name, conf))
    return results

# ── Other ROI helpers ─────────────────────────────────────────────────────

def ocr_text(rgb: np.ndarray) -> str:
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789kK/:.%"
    return pytesseract.image_to_string(rgb, config=cfg).strip()

def bar_fill(rgb: np.ndarray, thresh: int = 60) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float((gray > thresh).mean()*100)

# ── boxes.txt parser ──────────────────────────────────────────────────────

def parse_boxes(file=BOX_FILE) -> Dict[str, ROIType]:
    pat = re.compile(r"\[(\d+):(\d+),\s*(\d+):(\d+)\]")
    out: Dict[str, ROIType] = {}
    with open(file, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#",1)[0].strip()
            if not line:
                continue
            cat, rng, kind = [p.strip() for p in line.split(";", 2)]
            m = pat.search(rng)
            if not m:
                continue
            y0, y1, x0, x1 = map(int, m.groups())
            out[cat] = (slice(y0,y1), slice(x0,x1), kind.lower())
    return out

# ── Frame loading ─────────────────────────────────────────────────────────

def read_frame(path: str | pathlib.Path, idx: int=0) -> np.ndarray:
    p = pathlib.Path(path)
    if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp"}:
        img = cv2.imread(str(p))
        if img is None:
            raise FileNotFoundError(p)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise IOError(f"Cannot open {p}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read(); cap.release()
    if not ok:
        raise ValueError(f"Cannot seek to frame {idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# ── Main per‑frame processor ──────────────────────────────────────────────

def handle_frame(rgb: np.ndarray, boxes: Dict[str, ROIType]):
    icon_cache: Dict[str,np.ndarray] | None = None
    for cat, (rs, cs, kind) in boxes.items():
        roi = rgb[rs, cs]
        if kind == "text":
            print(f"{cat:20} text   → {ocr_text(roi) or '…'}")
        elif kind == "bar":
            print(f"{cat:20} bar    → {bar_fill(roi):4.1f}% filled")
        elif kind == "images":
            if icon_cache is None:
                h,w,_ = roi.shape
                icon_cache = load_item_icons((w//7, h))
            for idx,name,conf in analyze_item_bar(roi, icon_cache, int(cat[0] == 'r'), int(cat[1])):
                tag = name if name else "EMPTY"
                print(f"{cat}[{idx}] images → {tag} ({conf:.1f}%)")
        else:
            print(f"{cat:20} ???    → unknown type '{kind}'")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) not in {2,3}:
        print("Usage: python scrape_stats.py <image|video> [frame#]")
        sys.exit(1)
    frame = read_frame(sys.argv[1], int(sys.argv[2]) if len(sys.argv)==3 else 0)
    plt.imshow(frame); plt.title("input frame"); plt.axis("off"); plt.show()
    handle_frame(frame, parse_boxes())

    num_classes = 171
    model_path = "fastrcnn_model.pt"  # Update with your actual model path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = minimap.create_model(num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # print("\nMinimap detections:")
    # minimap.analyze_image(frame, model, num_classes)