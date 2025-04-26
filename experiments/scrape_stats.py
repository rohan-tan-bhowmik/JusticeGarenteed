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

# Point to your Tesseract binary if needed
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

BOX_FILE = "boxes.txt"           # master list
ROIType  = Tuple[slice, slice, str]   # rows, cols, kind


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


def ocr_roi(bgr: np.ndarray, psm: int = 7) -> str:
    """Run Tesseract on one small ROI and return cleaned text."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
    for cat, (rs, cs, kind) in boxes.items():
        roi = frame[rs, cs]

        if kind == "text":
            txt = ocr_roi(roi)
            print(f"{cat:20} text   → {txt or '…'}")

        elif kind == "bar":
            pct = bar_fill(roi)
            print(f"{cat:20} bar    → {pct:5.1f}% filled")

        elif kind == "images":
            print(f"{cat:20} images → TODO")

        else:
            print(f"{cat:20} ???    → unknown type '{kind}'")


def grab_frame(source: str | pathlib.Path, idx: int = 0) -> np.ndarray:
    """Return *idx*-th frame from an image or video."""
    path = pathlib.Path(source)
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        return img

    # Else treat as video
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError("Could not open " + str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not seek to frame {idx}")
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
    import matplotlib.pyplot as plt
    plt.imshow(frame_bgr)
    plt.show()
    process_frame(frame_bgr, box_map)
