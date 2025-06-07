import cv2
import numpy as np
import argparse
import os
import csv
from tqdm import tqdm
from pathlib import Path

import supervision as sv
from PIL import Image
from rfdetr import RFDETRBase
from classes import CLASSES

from scipy.optimize import linear_sum_assignment
import math
from collections import deque, defaultdict

import time
import json

# ─────────────────────────────────────────────────────────────────────────────
# 1) Classes and constants for detection/health‐bar matching
# ─────────────────────────────────────────────────────────────────────────────
HEALTHBAR_CLASSES = [
    "BlueChampionHealthbar",
    "RedChampionHealthbar",
    "BlueMinionHealthbar",
    "RedMinionHealthbar"
]

MINION_CLASSES = [
    "Blue Cannon",
    "Blue Caster",
    "Blue Melee",
    "Blue Siege",
    "Red Cannon",
    "Red Caster",
    "Red Melee",
    "Red Siege"
]

TOWER_LABELS = [
    'BlueInhibitor',
    'BlueNexus',
    'BlueTower',
    'RedInhibitor',
    'RedNexus',
    'RedTower',
]

CHAMPION_BODY_CLASSES = [
    name for name in CLASSES.values()
    if name not in HEALTHBAR_CLASSES and name not in MINION_CLASSES
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility: convert [xmin,ymin,xmax,ymax] → [x, y, w, h]
# ─────────────────────────────────────────────────────────────────────────────
def xyxy_to_xywh(box_xyxy):
    xmin, ymin, xmax, ymax = box_xyxy
    w = xmax - xmin
    h = ymax - ymin
    return [int(xmin), int(ymin), int(w), int(h)]


def get_matches(
    hb_boxes: np.ndarray,   # shape (N,4) in [x,y,w,h]
    sp_boxes: np.ndarray,   # shape (M,4) in [x,y,w,h]
    distance_threshold: float = 200.0,
    x_weight: float = 3.0
):
    """
    Matches healthbar boxes (hb_boxes) to sprite boxes (sp_boxes).
    Returns:
      - matches: list of (hb_idx, sp_idx) pairs whose weighted distance ≤ threshold
      - unmatched_hb: set of healthbar indices not matched
      - unmatched_sp: set of sprite indices not matched
    """
    N = hb_boxes.shape[0]
    M = sp_boxes.shape[0]
    if N == 0 or M == 0:
        return [], set(range(N)), set(range(M))

    # (a) Compute centers of healthbar and top‐centers of sprite
    hb_centers = hb_boxes[:, :2] + hb_boxes[:, 2:] / 2.0
    sprite_top_centers = sp_boxes[:, :2] + np.column_stack((sp_boxes[:, 2] / 2.0, np.zeros(M)))

    # (b) Weighted difference
    diff = hb_centers[:, None, :] - sprite_top_centers[None, :, :]
    diff[..., 0] *= x_weight
    distances = np.linalg.norm(diff, axis=2)  # shape (N, M)

    # (c) Disallow any pair > threshold
    large_cost = distance_threshold * 1000.0
    cost_matrix = distances.copy()
    cost_matrix[cost_matrix > distance_threshold] = large_cost

    # (d) Hungarian
    row_inds, col_inds = linear_sum_assignment(cost_matrix)
    matches = []
    for hb_idx, sp_idx in zip(row_inds, col_inds):
        if cost_matrix[hb_idx, sp_idx] < large_cost:
            matches.append((int(hb_idx), int(sp_idx)))

    matched_hb = {hb for hb, _ in matches}
    matched_sp = {sp for _, sp in matches}
    unmatched_hb = set(range(N)) - matched_hb
    unmatched_sp = set(range(M)) - matched_sp
    return matches, unmatched_hb, unmatched_sp


# ─────────────────────────────────────────────────────────────────────────────
# Health‐percent computation (for minions or champions). Uses HSV thresholds.
# ─────────────────────────────────────────────────────────────────────────────
def minion_health_percent(hb_img_bgr: np.ndarray, bar_color: str = 'blue') -> float:
    """
    Given a cropped healthbar image (hb_img_bgr) and bar_color ('blue'/'red'),
    returns percentage of bar filled (0.0–100.0).
    """
    hsv = cv2.cvtColor(hb_img_bgr, cv2.COLOR_BGR2HSV)
    if bar_color.lower() == 'blue':
        lower = np.array([100, 150,  50])
        upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif bar_color.lower() == 'red':
        lower1 = np.array([0,   100,  80])
        upper1 = np.array([10,  255, 255])
        lower2 = np.array([160, 100,  80])
        upper2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask1, mask2)
    else:
        raise ValueError("bar_color must be 'blue' or 'red'")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    col_sum = np.sum(mask, axis=0)  # shape (W,)
    W = hb_img_bgr.shape[1]
    nonzero_cols = np.where(col_sum > 0)[0]
    if nonzero_cols.size == 0:
        return 0.0
    x_right = int(nonzero_cols[-1])
    return ((x_right + 1) / float(W)) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Camera‐motion & minimap functions
# ─────────────────────────────────────────────────────────────────────────────
def estimate_camera_motion(prev: np.ndarray, curr: np.ndarray):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    feats = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=5
    )
    if feats is None:
        return np.eye(2, 3, np.float32), np.zeros(2, dtype=np.float32)
    pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, feats, None)
    if pts is None or st is None:
        return np.eye(2, 3, np.float32), np.zeros(2, dtype=np.float32)
    old = feats[st.flatten() == 1]
    new = pts[st.flatten() == 1]
    if len(old) < 10:
        return np.eye(2, 3, np.float32), np.zeros(2, dtype=np.float32)
    M, _ = cv2.estimateAffine2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return np.eye(2, 3, np.float32), np.zeros(2, dtype=np.float32)
    return M, M[:, 2]


def find_camera_box_in_minimap(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    full_w: int,
    full_h: int
) -> tuple[int, int] | None:
    x1, x2, y1, y2 = roi
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    roi_h, roi_w = gray.shape

    # 1) Threshold for white border
    _, mask_white = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2) Find contours on white mask
    cnts, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    target_ar = full_w / float(full_h)
    min_area = full_w * full_h * 0.02
    tol = 0.3

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            break

        bx, by, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh if bh > 0 else 1)

        # 2a) Exact‐size match within tolerance
        if (abs(ar - target_ar) < tol
            and abs(bw - full_w) < full_w * tol
            and abs(bh - full_h) < full_h * tol):
            cx_px = x1 + bx + bw // 2
            cy_px = y1 + by + bh // 2
            return cx_px, cy_px

        # 2b) Polygon‐based inference
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        pts = approx.reshape(-1, 2)

        verticals = []
        horizontals = []
        for i in range(len(pts)):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % len(pts)])
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            if dx < full_w * 0.2 and dy > full_h * 0.4:
                verticals.append((p1, p2))
            elif dy < full_h * 0.2 and dx > full_w * 0.4:
                horizontals.append((p1, p2))

        if verticals or horizontals:
            inferred_cx_px = None
            inferred_cy_px = None

            if verticals:
                (vx1, vy1), (vx2, vy2) = verticals[0]
                v_x_mean = (vx1 + vx2) / 2
                if v_x_mean < roi_w / 2:
                    inferred_cx_px = x1 + int(v_x_mean + full_w / 2)
                else:
                    inferred_cx_px = x1 + int(v_x_mean - full_w / 2)

            if horizontals:
                (hx1, hy1), (hx2, hy2) = horizontals[0]
                h_y_mean = (hy1 + hy2) / 2
                if h_y_mean < roi_h / 2:
                    inferred_cy_px = y1 + int(h_y_mean + full_h / 2)
                else:
                    inferred_cy_px = y1 + int(h_y_mean - full_h / 2)

            if verticals and not horizontals:
                inferred_cy_px = y1 + by + bh // 2
            if horizontals and not verticals:
                inferred_cx_px = x1 + bx + bw // 2

            if inferred_cx_px is not None and inferred_cy_px is not None:
                inferred_cx_px = np.clip(inferred_cx_px, x1, x2)
                inferred_cy_px = np.clip(inferred_cy_px, y1, y2)
                return inferred_cx_px, inferred_cy_px

    # 3) Template‐matching fallback
    best_score = 0.0
    best_center = None
    template_base = np.zeros((full_h, full_w), dtype=np.uint8)
    # cv2.rectangle(template_base, (0, 0), (full_w - 1, full_h - 1), 255, 1)

    for scale_t in np.linspace(0.9, 1.1, 9):
        tw = max(3, int(round(full_w * scale_t)))
        th = max(3, int(round(full_h * scale_t)))
        templ = cv2.resize(template_base, (tw, th), interpolation=cv2.INTER_LINEAR)
        if tw > roi_w or th > roi_h:
            continue
        res = cv2.matchTemplate(mask_white, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score and max_val > 0.4:
            best_score = max_val
            top_left = max_loc
            cx_px = x1 + top_left[0] + tw // 2
            cy_px = y1 + top_left[1] + th // 2
            best_center = (cx_px, cy_px)

    if best_center:
        return best_center

    # 4) HoughLinesP fallback
    edges = cv2.Canny(mask_white, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=int(full_h * 0.5),
        maxLineGap=5
    )
    if lines is not None:
        verticals = []
        horizontals = []
        for x1_l, y1_l, x2_l, y2_l in lines.reshape(-1, 4):
            dx = abs(x2_l - x1_l)
            dy = abs(y2_l - y1_l)
            if dx < full_w * 0.2 and dy > full_h * 0.4:
                verticals.append(((x1_l, y1_l), (x2_l, y2_l)))
            elif dy < full_h * 0.2 and dx > full_w * 0.4:
                horizontals.append(((x1_l, y1_l), (x2_l, y2_l)))

        cx_px = None
        cy_px = None
        if verticals:
            (vx1, vy1), (vx2, vy2) = verticals[0]
            v_x_mean = (vx1 + vx2) / 2
            if v_x_mean < roi_w / 2:
                cx_px = x1 + int(v_x_mean + full_w / 2)
            else:
                cx_px = x1 + int(v_x_mean - full_w / 2)
        if horizontals:
            (hx1, hy1), (hx2, hy2) = horizontals[0]
            h_y_mean = (hy1 + hy2) / 2
            if h_y_mean < roi_h / 2:
                cy_px = y1 + int(h_y_mean + full_h / 2)
            else:
                cy_px = y1 + int(h_y_mean - full_h / 2)
        if cx_px is not None and cy_px is not None:
            cx_px = np.clip(cx_px, x1, x2)
            cy_px = np.clip(cy_px, y1, y2)
            return cx_px, cy_px

    return None


def draw_arrow(frame: np.ndarray, mv: np.ndarray):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    # Scale the motion vector for visibility
    end = (int(center[0] - mv[0] * 10), int(center[1] - mv[1] * 10))
    cv2.arrowedLine(frame, center, end, (0, 0, 255), 2, tipLength=0.3)
    return frame


def resize_frame(frame: np.ndarray, scale: float):
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def draw_detections(image: Image.Image, detections: sv.Detections, labels: list[str]) -> np.ndarray:
    """
    Draw bounding boxes and labels onto the image. Returns BGR numpy array.
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    boxed = sv.BoxAnnotator().annotate(img_bgr, detections)
    annotated = sv.LabelAnnotator().annotate(boxed, detections, labels)
    return annotated


def predict(model: RFDETRBase, image: Image.Image, threshold: float = 0.5) -> (sv.Detections, list[str]):
    """
    Predict bounding boxes and labels for the given PIL image using RF-DETR.
    """
    img_array = np.array(image)  # H×W×3, RGB
    detections = model.predict(img_array, threshold=threshold)
    labels = [
        f"{CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    return detections, labels


def direction_from_vector(mv: np.ndarray) -> str:
    """
    Convert a 2D motion vector into the nearest quantized angle (in multiples of 15°),
    returning that angle as a string (e.g., "0°", "15°", "30°", …, "345°"). If the magnitude
    of mv is below 1.0, returns "STOPPED".
    """
    dx, dy = mv[0], mv[1]
    # If vector is very small, treat as stopped
    if dx**2 + dy**2 < 0.25:
        return "STOPPED"

    # Compute standard atan2 angle (0° = right, measured clockwise)
    raw_deg = (math.degrees(math.atan2(-dy, -dx)) + 360) % 360

    # Quantize to nearest multiple of 15°
    quant = int(round(raw_deg / 15.0) * 15) % 360
    return quant


# ─────────────────────────────────────────────────────────────────────────────
# 2) Main: uses a 10‐frame “lookahead” buffer to compute HP loss in the future
# ─────────────────────────────────────────────────────────────────────────────
def main(replay_folder: str, game_name: str, scale: float, skip_n: int, wild_thresh: float,
         weights: str, csv_path: str | None, start_frame: int, draw_annotations: bool = None):
    # Coordinates of the minimap ROI in the original frame:
    MM_Y1, MM_Y2 = 778, 1064
    MM_X1, MM_X2 = 1620, 1906
    CB_H, CB_W = 39, 70  # camera‐box size in pixels

    # Scale ROI and box dimensions
    roi = (
        int(MM_X1 * scale),
        int(MM_X2 * scale),
        int(MM_Y1 * scale),
        int(MM_Y2 * scale)
    )
    minimap_width  = roi[1] - roi[0]
    minimap_height = roi[3] - roi[2]
    full_w = int(CB_W * scale)
    full_h = int(CB_H * scale)

    meta_path = Path(replay_folder) / "metadata.json"
    game_data = None
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    for replay in meta["replays"]:
        if replay["name"] == game_name:
            game_data = replay
    
    if game_data is None:
        raise ValueError(f"Game {game_name} not found")
    
    video_path = Path(replay_folder) / game_data["full_video"]
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_iters = (max(0, total_frames - start_frame) + skip_n - 1) // skip_n

    # Jump to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame at start_frame.")
    prev = resize_frame(prev, scale)

    idx = start_frame

    # Tracking variables
    last_agent_position = None
    last_mv = None   # last nonzero motion vector from Garen

    # Buffer to hold 10‐frame lookahead entries
    pending_frames = deque()  # each entry is a dict for one frame

    last_champ_hp: dict[str, float] = {}   # maps key → last seen hp_pct
    last_tower_hp: dict[str, float] = {}   # same for towers

    champ_tracks: dict[str, dict] = {}  

    # Prepare CSV writer if requested
    csv_file = None
    writer = None
    if csv_path:
        csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        fieldnames = [
            "frame",
            "minimap_x_ratio", "minimap_y_ratio",
            "move_dir",
            "target",
            "champions",  # pipe-separated list: label,cx,cy,hp_pct,team
            "minions",    # same format
            "towers"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    # Instantiate RF-DETR model
    model = RFDETRBase(pretrain_weights=weights)
    model.device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    # Threshold for large motion jump (used in minimap logic)
    jump_thresh = min(full_w // 2, full_h // 2)
    persistence_frames = 12

    last_center = None
    candidate_center = None
    candidate_count = 0

    with tqdm(total=total_iters, desc="Processing frames") as pbar:
        while True:
            # Skip (skip_n - 1) frames between processing
            for _ in range(skip_n - 1):
                cap.read()
                idx += 1

            bitchgabe = time.time()

            ok, frame = cap.read()
            if not ok:
                break
            idx += 1

            frame = resize_frame(frame, scale)

            # 1) Estimate camera motion
            M, mv = estimate_camera_motion(prev, frame)
            move_dir = direction_from_vector(mv)

            # 2) Detect RF-DETR objects on the resized frame
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections, labels = predict(model, pil_img, threshold=0.5)
            annotated = draw_detections(pil_img, detections, labels)

            # 3) Find camera box in minimap if motion is “wild”
            minimap_center_ratio = ("", "")
            if np.linalg.norm(mv) > wild_thresh:
                found = find_camera_box_in_minimap(frame, roi, full_w, full_h)
                if found:
                    cx_px, cy_px = found

                    # — ONLY update last_center (with persistence logic) —
                    if last_center is None:
                        last_center = (cx_px, cy_px)
                        candidate_center = None
                        candidate_count = 0
                    else:
                        dist = np.hypot(cx_px - last_center[0], cy_px - last_center[1])
                        if dist < jump_thresh:
                            last_center = (cx_px, cy_px)
                            candidate_center = None
                            candidate_count = 0
                        else:
                            if candidate_center is None:
                                candidate_center = (cx_px, cy_px)
                                candidate_count = 1
                            else:
                                dc = np.hypot(cx_px - candidate_center[0], cy_px - candidate_center[1])
                                if dc < jump_thresh * 0.5:
                                    candidate_count += 1
                                else:
                                    candidate_center = (cx_px, cy_px)
                                    candidate_count = 1

                            if candidate_count >= persistence_frames:
                                last_center = candidate_center
                                candidate_center = None
                                candidate_count = 0

            # 3b) Derive minimap_center_ratio from last_center
            if last_center is not None:
                minimap_center_ratio = (
                    (last_center[0] - roi[0]) / float(minimap_width),
                    (last_center[1] - roi[2]) / float(minimap_height)
                )
            else:
                minimap_center_ratio = ("", "")

            # 4) Draw the last detected minimap‐box center (if any) onto annotated
            if last_center is not None:
                cv2.circle(annotated, last_center, radius=5, color=(0, 0, 255), thickness=-1)
                if minimap_center_ratio != ("", ""):
                    ratio_text = f"Minimap: ({minimap_center_ratio[0]:.2f}, {minimap_center_ratio[1]:.2f})"
                    cv2.putText(
                        annotated,
                        ratio_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA
                    )
                dir_text = f"Move Dir: {move_dir}"
                cv2.putText(
                    annotated,
                    dir_text,
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            # 5) Draw the motion arrow onto annotated
            annotated = draw_arrow(annotated, mv)

            # ─────────────────────────────────────────────────────────────
            # 6) HEALTH‐BAR MATCHING & PERCENT OVERLAY (with visualization)
            # ─────────────────────────────────────────────────────────────
            boxes_xyxy = detections.xyxy    # shape (K, 4)
            cls_ids    = detections.class_id

            all_xywh   = []
            all_labels = []
            for i, cid in enumerate(cls_ids):
                xmin, ymin, xmax, ymax = boxes_xyxy[i]
                x = int(xmin)
                y = int(ymin)
                w = int(xmax - xmin)
                h = int(ymax - ymin)
                all_xywh.append([x, y, w, h])
                all_labels.append(CLASSES[int(cid)])

            champion_data = []  # tuples: (label, cx, cy, hp_pct, team)
            minion_data   = []  # tuples: (label, mx, my, hp_pct, team)
            tower_data = []

            if len(all_xywh) > 0:
                all_xywh = np.array(all_xywh, dtype=np.int32)

                hb_boxes_list  = []
                hb_labels_list = []
                sp_boxes_list  = []
                sp_labels_list = []
                for i, lbl in enumerate(all_labels):
                    if lbl in HEALTHBAR_CLASSES:
                        hb_boxes_list.append(all_xywh[i])
                        hb_labels_list.append(lbl)
                    else:
                        sp_boxes_list.append(all_xywh[i])
                        sp_labels_list.append(lbl)

                hb_boxes  = (np.array(hb_boxes_list, dtype=np.int32)
                             if hb_boxes_list else np.empty((0,4), dtype=np.int32))
                hb_labels = hb_labels_list
                sp_boxes  = (np.array(sp_boxes_list, dtype=np.int32)
                             if sp_boxes_list else np.empty((0,4), dtype=np.int32))
                sp_labels = sp_labels_list

                hb_minion_inds = [
                    i for i, lbl in enumerate(hb_labels)
                    if "MinionHealthbar" in lbl
                ]
                hb_champ_inds  = [
                    i for i, lbl in enumerate(hb_labels)
                    if "ChampionHealthbar" in lbl
                ]
                sp_minion_inds = [
                    i for i, lbl in enumerate(sp_labels)
                    if lbl in MINION_CLASSES
                ]
                sp_champ_inds  = [
                    i for i, lbl in enumerate(sp_labels)
                    if lbl in CHAMPION_BODY_CLASSES and lbl not in TOWER_LABELS and (lbl in game_data["blue_team"] or lbl in game_data["red_team"])
                ]
                sp_tower_inds = [
                    i for i, lbl in enumerate(sp_labels)
                    if lbl in TOWER_LABELS
                ]
                hb_tower_inds = [
                    i for i, lbl in enumerate(hb_labels)
                    if lbl in ("BlueChampionHealthbar", "RedChampionHealthbar",
                            "BlueMinionHealthbar", "RedMinionHealthbar")
                ]

                # (e) Match minion healthbars → minion bodies
                if hb_minion_inds and sp_minion_inds:
                    hb_boxes_minion = hb_boxes[hb_minion_inds]
                    sp_boxes_minion = sp_boxes[sp_minion_inds]
                    matches_minion, _, _ = get_matches(
                        hb_boxes_minion,
                        sp_boxes_minion,
                        distance_threshold=200
                    )
                    for hb_idx0, sp_idx0 in matches_minion:
                        actual_hb_idx = hb_minion_inds[hb_idx0]
                        actual_sp_idx = sp_minion_inds[sp_idx0]
                        hb_box = hb_boxes[actual_hb_idx]    # [x, y, w, h]
                        sp_box = sp_boxes[actual_sp_idx]
                        hb_label = hb_labels[actual_hb_idx]  # "BlueMinionHealthbar" or "RedMinionHealthbar"
                        minion_label = sp_labels[actual_sp_idx]

                        # Crop healthbar, compute %:
                        x_hb, y_hb, w_hb, h_hb = hb_box
                        crop_hb = frame[y_hb:y_hb+h_hb, x_hb:x_hb+w_hb]
                        bar_color = "blue" if hb_label.startswith("Blue") else "red"
                        if crop_hb.size == 0:
                            # No valid pixels in this crop—treat health as 0% or skip this box entirely
                            pct = 0.0
                        else:
                            pct = minion_health_percent(crop_hb, bar_color=bar_color)


                        # Determine minion center (body center)
                        mx, my, mw, mh = sp_box
                        minion_center = (int(mx + mw/2), int(my + mh/2))
                        team_color = "Blue" if hb_label.startswith("Blue") else "Red"
                        minion_data.append((
                            minion_label,
                            minion_center[0], minion_center[1],
                            pct,
                            team_color
                        ))

                        # Draw rectangles & line in GREEN for minions:
                        if draw_annotations:
                            cv2.rectangle(
                                annotated,
                                (x_hb, y_hb),
                                (x_hb + w_hb, y_hb + h_hb),
                                (0, 255, 0),  # green
                                1
                            )
                            cv2.rectangle(
                                annotated,
                                (mx, my),
                                (mx + mw, my + mh),
                                (0, 255, 0),
                                1
                            )
                            hb_center = (int(x_hb + w_hb/2), int(y_hb + h_hb/2))
                            sp_center = (int(mx + mw/2), int(my + mh/2))
                            cv2.line(
                                annotated,
                                hb_center,
                                sp_center,
                                (0, 255, 0),
                                1
                            )
                            sx, sy, sw, sh = sp_box
                            text = f"{int(pct)}%"
                            color_text = (255, 255, 255) if bar_color == "blue" else (0, 0, 0)
                            cv2.putText(
                                annotated,
                                text,
                                (int(sx), int(sy) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color_text,
                                1,
                                cv2.LINE_AA
                            )

                # (f) Match champion healthbars → champion bodies
                if hb_champ_inds and sp_champ_inds:
                    hb_boxes_champ = hb_boxes[hb_champ_inds]
                    sp_boxes_champ = sp_boxes[sp_champ_inds]
                    matches_champ, unmatched_hb_champs, unmatched_sp_champs = get_matches(
                        hb_boxes_champ,
                        sp_boxes_champ,
                        distance_threshold=200
                    )
                    #  (1) Matched champions
                    for hb_idx0, sp_idx0 in matches_champ:
                        actual_hb_idx = hb_champ_inds[hb_idx0]
                        actual_sp_idx = sp_champ_inds[sp_idx0]
                        hb_box = hb_boxes[actual_hb_idx]
                        sp_box = sp_boxes[actual_sp_idx]
                        hb_label = hb_labels[actual_hb_idx]  # e.g. "RedChampionHealthbar"
                        champ_label = sp_labels[actual_sp_idx]


                        # Crop healthbar, compute %:
                        x_hb, y_hb, w_hb, h_hb = hb_box
                        crop_hb = frame[y_hb:y_hb+h_hb, x_hb:x_hb+w_hb]
                        bar_color = "blue" if hb_label.startswith("Blue") else "red"
                        pct = minion_health_percent(crop_hb, bar_color=bar_color)

                        # Determine champion center (body center)
                        cx, cy, cw, ch = sp_box
                        champ_center = (int(cx + cw/2), int(cy + ch/2))
                        team_color = "Blue" if hb_label.startswith("Blue") else "Red"

                        key = champ_label
                        champ_tracks[key] = {
                            "last_center": champ_center,   # exact pixel (cx, cy)
                            "last_hp": pct
                        }

                        # Save last seen HP under a quantized key:
                        key = f"Champion:{champ_label}:{champ_center[0]//10}:{champ_center[1]//10}"
                        last_champ_hp[key] = pct

                        champion_data.append((
                            champ_label,
                            champ_center[0], champ_center[1],
                            pct,
                            team_color
                        ))

                        # (Optional) draw matched boxes:
                        if draw_annotations:
                            cv2.rectangle(
                                annotated,
                                (x_hb, y_hb),
                                (x_hb + w_hb, y_hb + h_hb),
                                (255, 0, 0),
                                1
                            )
                            cv2.rectangle(
                                annotated,
                                (cx, cy),
                                (cx + cw, cy + ch),
                                (255, 0, 0),
                                1
                            )
                            hb_center = (int(x_hb + w_hb/2), int(y_hb + h_hb/2))
                            sp_center = (int(cx + cw/2), int(cy + ch/2))
                            cv2.line(
                                annotated,
                                hb_center,
                                sp_center,
                                (255, 0, 0),
                                1
                            )
                            sx, sy, sw, sh = sp_box
                            text = f"{int(pct)}%"
                            color_text = (255, 255, 255) if bar_color == "blue" else (0, 0, 0)
                            cv2.putText(
                                annotated,
                                text,
                                (int(sx), int(sy) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color_text,
                                1,
                                cv2.LINE_AA
                            )

                    # (2) Unmatched champion‐bars: place synthetic box 200 px below HB center,
                    #     HP = last seen (or 100 if never seen), name = “UNKNOWN” if we cannot match.
                    for hb_idx0 in unmatched_hb_champs:
                        actual_hb_idx = hb_champ_inds[hb_idx0]
                        hb_box = hb_boxes[actual_hb_idx]
                        hb_label = hb_labels[actual_hb_idx]

                        # HB center:
                        x_hb, y_hb, w_hb, h_hb = hb_box
                        hb_cx = x_hb + w_hb // 2
                        hb_cy = y_hb + h_hb // 2

                        # Try to “snap” this HB to any existing champion track
                        best_match = None
                        best_dist = float("inf")
                        for champ_name, track in champ_tracks.items():
                            cx_prev, cy_prev = track["last_center"]
                            dist = math.hypot(hb_cx - cx_prev, hb_cy - cy_prev)
                            # use a larger threshold so a fast‐moving champion still snaps
                            if dist < 100.0 and dist < best_dist:
                                best_match = champ_name
                                best_dist = dist

                        if best_match is not None:
                            # Re‐use the exact last_center for this champion
                            pct = champ_tracks[best_match]["last_hp"]
                            team_color = "Blue" if hb_label.startswith("Blue") else "Red"
                            cx_prev, cy_prev = champ_tracks[best_match]["last_center"]
                            champion_data.append((
                                best_match,
                                int(cx_prev), int(cy_prev),
                                pct,
                                team_color
                            ))
                            cv2.circle(annotated, (int(cx_prev), int(cy_prev)), 5, (128, 0, 128), thickness=1)

                        else:
                            # No existing track is “close enough,” remain UNKNOWN
                            synth_cx = hb_cx
                            synth_cy = hb_cy + 200
                            key = f"Champion:UNKNOWN:{synth_cx//10}:{synth_cy//10}"
                            pct = last_champ_hp.get(key, 100.0)
                            team_color = "Blue" if hb_label.startswith("Blue") else "Red"
                            champion_data.append((
                                "UNKNOWN",
                                synth_cx, synth_cy,
                                pct,
                                team_color
                            ))
                            # draw synthetic box if desired

                        
                # (h) Match tower healthbars → tower bodies
                if hb_tower_inds and sp_tower_inds:
                    hb_boxes_tower = hb_boxes[hb_tower_inds]
                    sp_boxes_tower = sp_boxes[sp_tower_inds]
                    matches_tower, unmatched_hb_towers, unmatched_sp_towers = get_matches(
                        hb_boxes_tower,
                        sp_boxes_tower,
                        distance_threshold=200
                    )
                    # (1) Matched towers
                    for hb_idx0, sp_idx0 in matches_tower:
                        actual_hb_idx = hb_tower_inds[hb_idx0]
                        actual_sp_idx = sp_tower_inds[sp_idx0]
                        hb_box = hb_boxes[actual_hb_idx]
                        sp_box = sp_boxes[actual_sp_idx]
                        hb_label = hb_labels[actual_hb_idx]    # e.g. “RedChampionHealthbar”
                        tower_label = sp_labels[actual_sp_idx] # “RedTower” or “BlueTower”

                        # Crop healthbar, compute %:
                        x_hb, y_hb, w_hb, h_hb = hb_box
                        crop_hb = frame[y_hb:y_hb+h_hb, x_hb:x_hb+w_hb]
                        bar_color = "blue" if hb_label.startswith("Blue") else "red"
                        pct = minion_health_percent(crop_hb, bar_color=bar_color)

                        # Store tower’s “center” (bottom‐center of the sprite box)
                        tx, ty, tw, th = sp_box
                        tower_center = (int(tx + tw/2), int(ty + th/2))
                        team_color = "Blue" if hb_label.startswith("Blue") else "Red"

                        # Save last seen HP under a quantized key:
                        key = f"Tower:{tower_label}:{tower_center[0]//10}:{tower_center[1]//10}"
                        last_tower_hp[key] = pct

                        tower_data.append((
                            tower_label,
                            tower_center[0],
                            tower_center[1],
                            pct,
                            team_color
                        ))
                        # (Optional) draw boxes:
                        if draw_annotations:
                            cv2.rectangle(
                                annotated,
                                (x_hb, y_hb),
                                (x_hb + w_hb, y_hb + h_hb),
                                (0, 255, 255), 1
                            )
                            cv2.rectangle(
                                annotated,
                                (tx, ty),
                                (tx + tw, ty + th),
                                (0, 255, 255), 1
                            )
                            hb_center = (int(x_hb + w_hb/2), int(y_hb + h_hb/2))
                            sp_center = (int(tx + tw/2), int(ty + th/2))
                            cv2.line(
                                annotated,
                                hb_center,
                                sp_center,
                                (0, 255, 255),
                                1
                            )
                            color_text = (255, 255, 255) if bar_color == "blue" else (0, 0, 0)
                            cv2.putText(
                                annotated,
                                f"{int(pct)}%",
                                (int(tx), int(ty) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color_text,
                                1,
                                cv2.LINE_AA
                            )

                    # # (2) Unmatched tower‐bars: synthetic box 200 px below, HP from last seen or 100
                    # for hb_idx0 in unmatched_hb_towers:
                    #     actual_hb_idx = hb_tower_inds[hb_idx0]
                    #     hb_box = hb_boxes[actual_hb_idx]
                    #     hb_label = hb_labels[actual_hb_idx]
                    #     # HB center:
                    #     x_hb, y_hb, w_hb, h_hb = hb_box
                    #     hb_cx = x_hb + w_hb//2
                    #     hb_cy = y_hb + h_hb//2

                    #     # Synthetic “tower” center 200 px below:
                    #     synth_cx = hb_cx
                    #     synth_cy = hb_cy + 200
                    #     # Name “UNKNOWN_TOWER”
                    #     tower_label = "UNKNOWN_TOWER"
                    #     key = f"Tower:{tower_label}:{synth_cx//10}:{synth_cy//10}"
                    #     pct = last_tower_hp.get(key, 100.0)
                    #     team_color = "Blue" if hb_label.startswith("Blue") else "Red"

                    #     tower_data.append((
                    #         tower_label,
                    #         synth_cx, synth_cy,
                    #         pct,
                    #         team_color
                    #     ))
                    #     # (Optional) draw synthetic box:
                    #     box_half = 20
                    #     cv2.rectangle(
                    #         annotated,
                    #         (synth_cx - box_half, synth_cy - box_half),
                    #         (synth_cx + box_half, synth_cy + box_half),
                    #         (0, 128, 128),  # teal
                    #         1
                    #     )
                    #     color_text = (255, 255, 255) if hb_label.startswith("Blue") else (0, 0, 0)
                    #     cv2.putText(
                    #         annotated,
                    #         f"{int(pct)}%",
                    #         (synth_cx - 20, synth_cy - box_half - 5),
                    #         cv2.FONT_HERSHEY_SIMPLEX,
                    #         0.5,
                    #         color_text,
                    #         1,
                    #         cv2.LINE_AA
                    #     )

            # ────────────────────────────────────────────────────────────────────
            # 2) Score each enemy’s HP right now (this frame) to buffer for lookahead
            # ────────────────────────────────────────────────────────────────────
            # Build future‐buffer entry: collect all “enemy” candidates at this frame
            # along with hp0 and a key.
            AGENT_NAME = "Garen"
            if AGENT_NAME in game_data["blue_team"]:
                AGENT_TEAM = "Blue"
            else:
                AGENT_TEAM = "Red"

            # First, locate Garen’s feet (bottom-center) at this frame
            agent_entry = None  # will be (agent_x, agent_y)
            # (a) Try matched champion_data
            for (label, cx, cy, hp_pct, team) in champion_data:
                if label == AGENT_NAME and team == AGENT_TEAM:
                    # find raw DETR bounding box to get bottom-center
                    for i, class_id in enumerate(detections.class_id):
                        if CLASSES[int(class_id)] == AGENT_NAME:
                            xmin, ymin, xmax, ymax = detections.xyxy[i]
                            agent_entry = (int((xmin + xmax) / 2), int(ymax))
                            break
                    if agent_entry is not None:
                        break
            # (b) If not found, fall back to raw DETR
            if agent_entry is None:
                for i, class_id in enumerate(detections.class_id):
                    lbl = CLASSES[int(class_id)]
                    if lbl == AGENT_NAME:
                        xmin, ymin, xmax, ymax = detections.xyxy[i]
                        agent_entry = (int((xmin + xmax) / 2), int(ymax))
                        break
            # (c) If found, update last_agent_position
            if agent_entry is not None:
                last_agent_position = agent_entry
            # (d) If still not found, fallback to last_agent_position or bottom-center of screen
            if agent_entry is None:
                if last_agent_position is not None:
                    agent_entry = last_agent_position
                else:
                    h_img, w_img = frame.shape[:2]
                    agent_entry = (w_img // 2, h_img)

            # Save a per‐frame “framerec” dict to pending_frames
            #  • Collect champion_str and minion_str for CSV
            champ_str = "|".join(
                f"{label},{cx},{cy},{pct:.1f},{team}"
                for (label, cx, cy, pct, team) in champion_data
            )
            minion_str = "|".join(
                f"{label},{mx},{my},{pct:.1f},{team}"
                for (label, mx, my, pct, team) in minion_data
            )
            tower_str = "|".join(
                f"{label},{tx},{ty},{pct:.1f},{team}" 
                for (label, tx, ty, pct, team) in tower_data
            )

            #  • Build list of “enemy” candidates at this frame, with hp0 & key & bottom‐center
            enemy_candidates_at_frame = []
            #   – Champions:
            for (label, cx, cy, hp_pct, team) in champion_data:
                if team != AGENT_TEAM:
                    # find raw DETR bounding box to get bottom-center and position‐based key
                    for i, class_id in enumerate(detections.class_id):
                        if CLASSES[int(class_id)] == label:
                            xmin, ymin, xmax, ymax = detections.xyxy[i]
                            ex = int((xmin + xmax) / 2)
                            ey = int(ymax)
                            # key quantizes position so that over next 10 frames a small movement still preserves match
                            key = f"Champion:{label}:{ex//10}:{ey//10}"
                            enemy_candidates_at_frame.append((label, ex, ey, hp_pct, key))
                            break
            #   – Minions:
            for (label, mx, my, hp_pct, team) in minion_data:
                if team != AGENT_TEAM:
                    for i, class_id in enumerate(detections.class_id):
                        if CLASSES[int(class_id)] == label:
                            xmin, ymin, xmax, ymax = detections.xyxy[i]
                            ex = int((xmin + xmax) / 2)
                            ey = int(ymax)
                            key = f"Minion:{label}:{ex//10}:{ey//10}"
                            enemy_candidates_at_frame.append((label, ex, ey, hp_pct, key))
                            break
            #   – Towers:
            for (label, tx, ty, hp_pct, team) in tower_data:
                if team != AGENT_TEAM:
                    # find raw DETR bbox to get exact bottom‐center:
                    for i, class_id in enumerate(detections.class_id):
                        if CLASSES[int(class_id)] == label:
                            xmin, ymin, xmax, ymax = detections.xyxy[i]
                            ex = int((xmin + xmax) / 2)
                            ey = int(ymax)
                            key = f"Tower:{label}:{ex//10}:{ey//10}"
                            enemy_candidates_at_frame.append((label, ex, ey, hp_pct, key))
                            break


            #  • Record the last_mv (motion vector) at this frame
            mv_snapshot = mv.copy()

            #  • Build the record and append to pending_frames
            pending_frames.append({
                "frame": idx - start_frame,
                "minimap_ratio": minimap_center_ratio,
                "move_dir": move_dir,
                "champ_str": champ_str,
                "minion_str": minion_str,
                "tower_str": tower_str,
                "agent_entry": agent_entry,
                "last_mv": mv_snapshot,
                "enemy_candidates": enemy_candidates_at_frame
            })

            # ─────────────────────────────────────────────────────────────
            # 3) If we have > 10 entries, pop the oldest and compute its lookahead
            # ─────────────────────────────────────────────────────────────
            if len(pending_frames) > 10:
                old = pending_frames.popleft()
                frame_old = old["frame"]
                minimap_old = old["minimap_ratio"]
                move_dir_old = old["move_dir"]
                champ_str_old = old["champ_str"]
                minion_str_old = old["minion_str"]
                tower_str_old = old["tower_str"]
                agent_x_old, agent_y_old = old["agent_entry"]
                last_mv_old = old["last_mv"]
                enemy_list_old = old["enemy_candidates"]

                # Build hp_future mapping from current frame
                hp_future_map = {}
                for (label, cx, cy, hp_pct, team) in champion_data:
                    if team != AGENT_TEAM:
                        key = f"Champion:{label}:{int(cx)//10}:{int(cy + (hp_pct*0)/100)//10}"  # cy approx bottom center? Instead, we search by any champion that matches key in enemy_list_old
                        # Actually use same key definition as above:
                        # We must recompute bottom‐center of this champion
                        for i, class_id in enumerate(detections.class_id):
                            if CLASSES[int(class_id)] == label:
                                xmin, ymin, xmax, ymax = detections.xyxy[i]
                                ex = int((xmin + xmax) / 2)
                                ey = int(ymax)
                                key_c = f"Champion:{label}:{ex//10}:{ey//10}"
                                hp_future_map[key_c] = hp_pct
                                break

                for (label, mx, my, hp_pct, team) in minion_data:
                    if team != AGENT_TEAM:
                        for i, class_id in enumerate(detections.class_id):
                            if CLASSES[int(class_id)] == label:
                                xmin, ymin, xmax, ymax = detections.xyxy[i]
                                ex = int((xmin + xmax) / 2)
                                ey = int(ymax)
                                key_m = f"Minion:{label}:{ex//10}:{ey//10}"
                                hp_future_map[key_m] = hp_pct
                                break

                for (label, tx, ty, hp_pct, team) in tower_data:
                    if team != AGENT_TEAM:
                        # recompute bottom‐center exactly as above
                        for i, class_id in enumerate(detections.class_id):
                            if CLASSES[int(class_id)] == label:
                                xmin, ymin, xmax, ymax = detections.xyxy[i]
                                ex = int((xmin + xmax) / 2)
                                ey = int(ymax)
                                key_t = f"Tower:{label}:{ex//10}:{ey//10}"
                                hp_future_map[key_t] = hp_pct
                                break

                # Now compute cost for each old enemy candidate
                if not enemy_list_old or move_dir_old != "STOPPED":
                    target_str = "NONE"
                else:
                    # Constants
                    WEIGHT_ANGLE = 1.0
                    WEIGHT_HP    = -2.0
                    WEIGHT_DIST  =  0.02 
                    # Normalize old last_mv
                    if last_mv_old is None or (last_mv_old[0] == 0 and last_mv_old[1] == 0):
                        mv_unit_old = None
                    else:
                        tmp = np.array(last_mv_old, dtype=np.float32)
                        norm_val = math.hypot(tmp[0], tmp[1])
                        mv_unit_old = (tmp / norm_val) if norm_val > 1e-6 else tmp

                    best_score = float("inf")
                    best_entry = None

                    for (label_o, ex_o, ey_o, hp0_o, key_o) in enemy_list_old:
                        # Angle component
                        if mv_unit_old is None:
                            angle_diff = 0.0
                        else:
                            vx = ex_o - agent_x_old
                            vy = ey_o - agent_y_old
                            dist_dir = math.hypot(vx, vy)
                            if dist_dir < 1e-6:
                                angle_diff = 0.0
                            else:
                                cand_unit = np.array([vx/dist_dir, vy/dist_dir], dtype=np.float32)
                                cosang = float(np.clip(mv_unit_old.dot(cand_unit), -1.0, 1.0))
                                angle_diff = math.acos(cosang)

                        # HP drop: hp0_o - hp_future
                        hp_future = hp_future_map.get(key_o, 0.0)
                        hp_drop = max(0.0, hp0_o - hp_future)

                        cost = WEIGHT_ANGLE * angle_diff + WEIGHT_HP * (-hp_drop) + WEIGHT_DIST  * dist_dir
                        if cost < best_score:
                            best_score = cost
                            best_entry = (label_o, ex_o, ey_o)

                    if best_entry is None:
                        target_str = "NONE"
                    else:
                        target_str = f"{best_entry[0]},{best_entry[1]},{best_entry[2]}"

                # Write CSV row for old frame
                if writer:
                    row = {
                        "frame": frame_old,
                        "minimap_x_ratio": f"{minimap_old[0]:.3f}" if minimap_old != ("", "") else "",
                        "minimap_y_ratio": f"{minimap_old[1]:.3f}" if minimap_old != ("", "") else "",
                        "move_dir": move_dir_old,
                        "target": target_str,
                        "champions": champ_str_old,
                        "minions": minion_str_old,
                        "towers": tower_str_old
                    }
                    writer.writerow(row)

            # ─────────────────────────────────────────────────────────────
            # 4) Display the fully‐annotated frame (optional)
            # ─────────────────────────────────────────────────────────────
            # We choose not to overlay “Attack:” on the live frame,
            # since targets correspond to old frames. If you want, you
            # could draw on the frame when computing the old target above.

            # print(time.time() - bitchgabe)

            if draw_annotations:
                cv2.imshow("Detections + Motion + Minimap + Health + Stats", annotated)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

            prev = frame.copy()
            pbar.update(1)

        # After loop ends, flush remaining pending_frames (no lookahead available)
        while pending_frames:
            old = pending_frames.popleft()
            frame_old = old["frame"]
            minimap_old = old["minimap_ratio"]
            move_dir_old = old["move_dir"]
            champ_str_old = old["champ_str"]
            minion_str_old = old["minion_str"]
            tower_str_old = old["tower_str"]
            # No lookahead → default to NONE
            target_str = "NONE"
            if writer:
                row = {
                    "frame": frame_old,
                    "minimap_x_ratio": f"{minimap_old[0]:.3f}" if minimap_old != ("", "") else "",
                    "minimap_y_ratio": f"{minimap_old[1]:.3f}" if minimap_old != ("", "") else "",
                    "move_dir": move_dir_old,
                    "target": target_str,
                    "champions": champ_str_old,
                    "minions": minion_str_old,
                    "towers": tower_str_old
                }
                writer.writerow(row)

    cap.release()
    cv2.destroyAllWindows()
    if csv_file:
        csv_file.close()
        print(f"Saved stats to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("video", help="Path to League replay clip (e.g., .mp4)")
    parser.add_argument("--replay_folder", type=str)
    parser.add_argument("--game", type=str)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Downscale factor for resolution (e.g., 0.5 = 50%)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Only process every Nth frame")
    parser.add_argument("--wild", type=float, default=0.0,
                        help="Optical‐flow magnitude threshold to trigger minimap search")
    parser.add_argument("--weights", type=str, default="checkpoint_best_ema.pth",
                        help="RF-DETR .pth weights file")
    parser.add_argument("--csv", type=str, default=None,
                        help="Output CSV file path for saving stats")
    parser.add_argument("--start", type=int, default=0,
                        help="Zero‐based frame index at which to begin processing")
    parser.add_argument("--draw", type=bool, default=False)
    args = parser.parse_args()

    if not os.path.exists(args.replay_folder):
        raise FileNotFoundError(f"Video file not found: {args.replay_folder}")
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    

    main(args.replay_folder, args.game, args.scale, args.skip, args.wild, args.weights, args.csv, args.start, args.draw)
