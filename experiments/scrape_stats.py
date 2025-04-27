#!/usr/bin/env python3
"""
Extract HUD information from a League-of-Legends screenshot or video frame.

boxes.txt format (three columns, separated by ';')
--------------------------------------------------
category ; [y0:y1, x0:x1] ; type     # type ∈ {text, bar, images}

Usage
-----
python scrape_stats.py path/to/frame.png            # single image
python scrape_stats.py path/to/replay.mp4 12345     # video at frame 12 345
"""

from __future__ import annotations
import pathlib, re, sys, cv2, pytesseract, numpy as np
from typing import Dict, Tuple
import os
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import matplotlib.pyplot as plt

# Point to your Tesseract binary if needed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BOX_FILE = "boxes.txt"           # master list
ROIType  = Tuple[slice, slice, str]   # rows, cols, kind
ITEM_FOLDER = "item_icons"      # your downloaded icons

def split_item_box(box_img: np.ndarray) -> list[np.ndarray]:
    """Split an item box into 7 individual item slots based on expected size."""
    slot_width = 24
    gap_width = 3
    slots = []

    # Starting x coordinate
    x = 0
    for _ in range(7):
        slot = box_img[:, x:x + slot_width]
        slots.append(slot)
        x += slot_width + gap_width  # move past item + gap
    return slots

def load_item_icons(size: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """Load and resize all item icons to the given slot size."""
    items = {}
    for path in Path(ITEM_FOLDER).glob("*.png"):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # <<< ADD THIS
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        items[path.stem] = img
    return items

def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean squared error between two images."""
    err = np.mean((img1.astype("float32") - img2.astype("float32")) ** 2)
    return err

def match_item(slot_img: np.ndarray, items: Dict[str, np.ndarray]) -> Tuple[str, float]:
    """Return (best item name, similarity%) for a given slot using template matching."""
    best_name, best_score = None, -1.0
    for name, ref in items.items():
        # Resize slot to match reference if needed
        resized = cv2.resize(slot_img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
        # Match using Normalized Cross-Correlation
        result = cv2.matchTemplate(resized, ref, cv2.TM_CCOEFF_NORMED)
        score = result.max()  # highest correlation score
        if score > best_score:
            best_name, best_score = name, score
    similarity = best_score * 100.0   # Convert [0, 1] → [0, 100]%
    return best_name, similarity


def analyze_item_box(box_img: np.ndarray, item_icons: Dict[str, np.ndarray]):
    """Analyze a single 'images' box."""
    slots = split_item_box(box_img)
    found = []
    for i, slot in enumerate(slots):
        if np.mean(slot) < 30:
            found.append((i, None, 0.0))
            continue

        # Resize to icon size
        slot_resized = cv2.resize(slot, next(iter(item_icons.values())).shape[:2][::-1])
        best_item, similarity = match_item(slot_resized, item_icons)
        found.append((i, best_item, similarity))
    return found


# ── utilities ──────────────────────────────────────────────────────────────
def parse_boxes(path: str | pathlib.Path = BOX_FILE) -> Dict[str, ROIType]:
    """
    Return {category: (row_slice, col_slice, kind)}.
    Ignores blank lines and comments (# ...).
    """
    boxes: Dict[str, ROIType] = {}
    pat = re.compile(r"\[(\d+):(\d+),\s*(\d+):(\d+)\]")
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.split("#", 1)[0].strip()
            if not ln:
                continue
            parts = [p.strip() for p in ln.split(";", 2)]
            if len(parts) != 3:
                continue
            cat, rng, kind = parts
            m = pat.search(rng)
            if not m:
                continue
            y0, y1, x0, x1 = map(int, m.groups())
            boxes[cat] = (slice(y0, y1), slice(x0, x1), kind.lower())
    return boxes

def ocr_roi(rgb: np.ndarray, psm: int = 7) -> str:
    """Run Tesseract on one small ROI already in RGB."""
    cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789kK/:.%"
    return pytesseract.image_to_string(rgb, config=cfg).strip()

def bar_fill(bgr: np.ndarray, thresh: int = 60) -> float:
    """
    Estimate bar fill as percentage of pixels whose brightness > thresh.
    Very dark pixels (~black) are counted as empty.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float((gray > thresh).mean() * 100.0)

# ── entry points ───────────────────────────────────────────────────────────
def process_frame(frame: np.ndarray, boxes: Dict[str, ROIType]) -> None:
    """Process every ROI in *frame* according to its declared type."""
    # Load once
    item_icons = None

    for cat, (rs, cs, kind) in boxes.items():
        roi = frame[rs, cs]

        if kind == "text":
            txt = ocr_roi(roi)
            print(f"{cat:20} text   → {txt or '…'}")

        elif kind == "bar":
            pct = bar_fill(roi)
            print(f"{cat:20} bar    → {pct:5.1f}% filled")

        elif kind == "images":
            # First time: load item icons resized to slot size
            if item_icons is None:
                H, W, _ = roi.shape
                item_icons = load_item_icons((W//7, H))
            results = analyze_item_box(roi, item_icons)
            for idx, item, sim in results:
                if item:
                    print(f"{cat}[{idx}] images → {item} ({sim:.1f}%)")
                else:
                    print(f"{cat}[{idx}] images → EMPTY")

        else:
            print(f"{cat:20} ???    → unknown type '{kind}'")

def grab_frame(source: str | pathlib.Path, idx: int = 0) -> np.ndarray:
    """Return *idx*-th frame from an image or video, in RGB."""
    path = pathlib.Path(source)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # <<< ADD THIS
        return img

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("Could not open " + str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not seek to frame {idx}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # <<< ADD THIS
    return frame

# ── main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        print("Usage: python scrape_stats.py <image|video> [frame#]")
        sys.exit(1)

    source = sys.argv[1]
    frame_id = int(sys.argv[2]) if len(sys.argv) == 3 else 0

    box_map = parse_boxes()
    frame_bgr = grab_frame(source, frame_id)
    plt.imshow(frame_bgr)
    plt.show()
    process_frame(frame_bgr, box_map)
