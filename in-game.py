### Play the policy in-game ###

import os
import sys
import json
import re
import time
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import easyocr
import mss
import keyboard
from PIL import Image
import utils.minimap
from utils.minimap import ImageDrawer

import supervision as sv
from rfdetr import RFDETRBase
from classes import CLASSES
from scipy.optimize import linear_sum_assignment
from collections import deque, defaultdict
import math

import pyautogui
import keyboard

pyautogui.FAILSAFE = True

# Emergency stop when pressing '='
keyboard.add_hotkey('=', lambda: (_ for _ in ()).throw(SystemExit("Failsafe pressed")))

reward = 0.0


def execute_action(action):
    """
    action: dict from policy.predict with keys:
      - move_dir: tensor([0..24])
      - attack:  tensor([[0/1]])
      - xy_target: tensor([[x_norm, y_norm]])
      - abilities: tensor([[q,w,e,r,d,f]])
    """
    md = int(action['move_dir'].item())
    atk = float(action['attack'].item())
    xt, yt = action['xy_target'][0].tolist()
    ab = action['abilities'][0].tolist()

    cx, cy = 960, 540  # screen center
    # 1) Attack vs Move
    if atk < 0.5:
        # Movement
        if md == 24:
            # STOPPED
            keyboard.press_and_release('s')
        else:
            angle_deg = md * 15
            rad = math.radians(angle_deg)
            ex = int(cx + math.cos(rad) * 150)
            ey = int(cy + math.sin(rad) * 150)
            pyautogui.moveTo(ex, ey)
            pyautogui.click(button='right')
    else:
        # Attack target
        px = int(xt * 1920)
        py = int(yt * 1080)
        pyautogui.moveTo(px, py)
        pyautogui.click()  # left click

    # 2) Abilities Q, W, E, R, D, F
    for key, flag in zip(['q','w','e','r','d','f'], ab):
        if flag > 0.5:
            keyboard.press_and_release(key)


from policy import GarenPolicy
from create_expert_trajectories import (
    MAX_NUM_CHAMPION_DETECTIONS,
    MAX_NUM_MINION_DETECTIONS,
    MAX_NUM_TOWER_DETECTIONS,
    NUM_CONTINUOUS_F,
    MAX_NUM_DETECTIONS,
)
from classes import CONDENSED_CHAMPIONS_TO_I
# List your continuous‐feature keys here in the exact order you want them:
# CONTINUOUS_KEYS = [
#     # e.g. 'game-time', 'xp-bar', 'health-bar', 'attack-dmg', 'armor', …
# ]  
# assert len(CONTINUOUS_KEYS) == NUM_CONTINUOUS_F

NUM_CHAMPION_CLASSES = len(CONDENSED_CHAMPIONS_TO_I)
NUM_MINION_CLASSES = 8
NUM_TOWER_CLASSES = 6

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
# Globals for item matching
# ─────────────────────────────────────────────────────────────────────────────
ITEM_FOLDER = "item_icons"
ITEM_SIZE   = (32, 33)
icon_rgb_dict  = {}
icon_gray_dict = {}

USE_TORCH_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_TORCH_CUDA else "cpu")

policy = GarenPolicy().to(device)
policy.eval()

checkpoint_data = torch.load("policy_best.pth")
policy.load_state_dict(checkpoint_data["model_state"])

# 1b) Load model weights exactly as in analyze_video.py:
minimap_model = utils.minimap.create_model(171, device=device)
minimap_model.load_state_dict(torch.load(
    "utils/minimap_model.pt", map_location=device
))
minimap_model.to(device).eval()


# 1c) Champion-ID mapping (reuse your ImageDrawer):
icons_path = 'league_icons/'
image_drawer = ImageDrawer(
    os.path.join(icons_path, 'champions'),
    os.path.join(icons_path, 'minimap'),
    os.path.join(icons_path, 'fog'),
    os.path.join(icons_path, 'misc'),
    resize=(256,256)
)

whitelist_ids = { image_drawer.champion_to_id["garen"] }

id_to_champ = image_drawer.id_to_champion

rfdetr_model = RFDETRBase(pretrain_weights="utils/checkpoint_best_ema.pth")
rfdetr_model.device = "cuda" if USE_TORCH_CUDA else "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Parse ROIs from in-game-boxes.txt, splitting on "### TAB ###"
# ─────────────────────────────────────────────────────────────────────────────
def parse_boxes(file="utils/in-game-boxes.txt"):
    pat = re.compile(r"\[(\d+):(\d+),\s*(\d+):(\d+)\]")
    perm_rois = {}
    tab_rois  = {}
    mode = "perm"
    with open(file, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            if line == "=== TAB ===":
                mode = "tab"
                continue
            cat, rng, kind = [p.strip() for p in line.split(";", 2)]
            m = pat.search(rng)
            if not m:
                continue
            y0,y1,x0,x1 = map(int, m.groups())
            target = perm_rois if mode=="perm" else tab_rois
            target[cat] = (slice(y0,y1), slice(x0,x1), kind.lower())
    return perm_rois, tab_rois

def preprocess_for_ocr(cat_name: str, rgb_roi: np.ndarray) -> np.ndarray:
    """
    Enhance any high-contrast (bright) text, regardless of hue.
    If 'cd' in cat_name, only keep nearly-pure-white pixels (value > 244).
    """
    # 1) Luma channel
    ycrcb = cv2.cvtColor(rgb_roi, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[..., 0]

    factor = 3
    y = cv2.resize(y, (y.shape[1]*factor, y.shape[0]*factor))

    # 2) CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    y_eq = clahe.apply(y)

    # If this is a cooldown field, do a hard white-threshold and return
    if "cd" in cat_name.lower():
        # threshold to keep only nearly-white pixels
        _, binary_cd = cv2.threshold(y_eq, 235, 255, cv2.THRESH_BINARY)

        # structuring element for dilation/closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

        # 1) Dilate once to fatten strokes
        binary_cd = cv2.dilate(binary_cd, kernel, iterations=1)

        # 2) Close small holes (dilate→erode)
        binary_cd = cv2.morphologyEx(binary_cd, cv2.MORPH_CLOSE, kernel, iterations=1)

        return binary_cd



    # 3) Adaptive brightness threshold for everything else
    mean, std = float(y_eq.mean()), float(y_eq.std())
    thresh_val = mean + 1.0 * std
    _, binary = cv2.threshold(y_eq, thresh_val, 255, cv2.THRESH_BINARY)

    return binary



def load_and_cache_item_icons(size):
    for fn in os.listdir(ITEM_FOLDER):
        if not fn.endswith(".png"):
            continue
        stem = os.path.splitext(fn)[0]
        img = cv2.imread(os.path.join(ITEM_FOLDER, fn))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        icon_rgb_dict[stem]  = resized
        icon_gray_dict[stem] = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

def match_item(slot_rgb: np.ndarray):
    """Return (name,score) or (None,0) if slot empty or no match."""
    gray = cv2.cvtColor(slot_rgb, cv2.COLOR_RGB2GRAY)
    if gray.std() < 10:
        return None, 0.0
    best_name, best_score = None, -1.0
    for name, ref in icon_gray_dict.items():
        score = cv2.matchTemplate(gray, ref, cv2.TM_CCOEFF_NORMED).max()
        if score > best_score:
            best_name, best_score = name, float(score)
    return (best_name, best_score) if best_score>=0 else (None,0.0)

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
    if hb_img_bgr is None or hb_img_bgr.size == 0:
        return 0.0
    
    hsv = cv2.cvtColor(hb_img_bgr, cv2.COLOR_BGR2HSV)
    if bar_color.lower() == 'blue':
        lower = np.array([0, 150,  50])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
    elif bar_color.lower() == 'red':
        lower1 = np.array([0,   100,  80])
        upper1 = np.array([255,  255, 255])
        lower2 = np.array([0, 100,  80])
        upper2 = np.array([255, 255, 255])
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
    if prev is None or prev.size == 0:
        return 0.0
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
# OCR worker
# ─────────────────────────────────────────────────────────────────────────────
ocr_queue   = Queue()
ocr_results = {}
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
def ocr_worker():
    while True:
        key, img = ocr_queue.get()
        if key is None:
            break
        texts = reader.readtext(img, detail=0, allowlist="0123456789.:k/")
        ocr_results[key] = ' '.join(texts)
        ocr_queue.task_done()

Thread(target=ocr_worker, daemon=True).start()

def detect_whitelist_only(minimap_crop: np.ndarray,
                          model: torch.nn.Module,
                          device: torch.device,
                          whitelist_ids: set[int],
                          score_thresh: float = 0.5):
    """
    - minimap_crop: uint8 array of shape (H_raw, W_raw, 3).
    - model: a Faster R‐CNN whose classes match champion IDs (with background index 0).
    - whitelist_ids: set of integer class‐IDs to allow.
    - score_thresh: threshold on “best‐within‐whitelist” probability.
    """

    model.eval()

    # 1) Convert the raw crop into a batch‐of‐one float tensor in [0,1]
    x = (
        torch.from_numpy(minimap_crop)
             .permute(2, 0, 1)   # → (3, H_raw, W_raw)
             .unsqueeze(0)       # → (1, 3, H_raw, W_raw)
             .to(device)
             .float()
             .div(255.0)
    )

    with torch.no_grad():
        # 2) Run transform → backbone → RPN → box head
        images, _ = model.transform(x, None)
        features = model.backbone(images.tensors)
        proposals, _ = model.rpn(images, features, None)

        box_features = model.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes
        )
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)

        # Now `boxes_all` lives in the *resized* image space
        boxes_all  = proposals[0]          # Tensor of shape (num_props, 4)
        logits_all = class_logits          # Tensor of shape (num_props, num_classes)

        # Also remember the network’s resized dimensions:
        H_resized, W_resized = images.image_sizes[0]

    # 3) Turn logits → per‐proposal probabilities (softmax over classes)
    probs_all = logits_all.softmax(dim=-1)  # shape = (num_props, num_classes)

    # 4) Zero‐out any classes not in the whitelist
    num_classes = probs_all.shape[-1]  # e.g. if you trained on 10 champs + background, num_classes = 11
    allow_mask = torch.zeros(num_classes, device=device)
    for cid in whitelist_ids:
        allow_mask[cid] = 1.0

    masked_probs = probs_all * allow_mask.unsqueeze(0)  # still shape = (num_props, num_classes)

    # 5) For each proposal, find best “allowed” class & its prob
    per_prop_best_prob, per_prop_best_label = masked_probs.max(dim=-1)
    #   …where per_prop_best_label = 0 if no whitelist class passed ≥ threshold

    keep_mask = (per_prop_best_prob >= score_thresh) & (per_prop_best_label != 0)
    keep_props = keep_mask.nonzero(as_tuple=False).squeeze(1)  # indices of proposals to keep

    final_labels = per_prop_best_label[keep_props]   # champion IDs
    final_scores = per_prop_best_prob[keep_props]    # “within‐whitelist” confidence
    kept_boxes   = boxes_all[keep_props]             # (K,4) Tensor

    # 6) Plain NMS over all kept proposals
    kept_boxes_np  = kept_boxes.cpu().numpy()        # shape = (K,4)
    kept_labels_np = final_labels.cpu().numpy()      # shape = (K,)
    kept_scores_np = final_scores.cpu().numpy()      # shape = (K,)

    import torchvision.ops as ops
    keep_after_nms = ops.nms(
        torch.from_numpy(kept_boxes_np),
        torch.from_numpy(kept_scores_np),
        iou_threshold=0.5
    )

    nms_boxes   = kept_boxes_np[keep_after_nms.cpu().numpy()]
    nms_labels  = kept_labels_np[keep_after_nms.cpu().numpy()]
    nms_scores  = kept_scores_np[keep_after_nms.cpu().numpy()]

    # 7) Keep only one box per champion (highest‐score)
    best_per_champ = {}
    for box, lbl, score in zip(nms_boxes, nms_labels, nms_scores):
        if (lbl not in best_per_champ) or (score > best_per_champ[lbl][1]):
            best_per_champ[lbl] = (box, float(score))

    # 8) Build final “(champion_name, norm_x, norm_y, score)” list
    results = []
    for lbl, (box, score) in best_per_champ.items():
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) * 0.5) / float(W_resized)
        cy = ((y1 + y2) * 0.5) / float(H_resized)
        champ_name = id_to_champ[int(lbl)]
        results.append((champ_name, cx, cy, score))

    return results

def to_float(s):
    try:
        return float(s.strip('%'))
    except:
        return 0.0

def make_det_buffer(raw, max_slots):
    """
    raw: list of (class_id, x, y, score)
    returns np.float32 array shape (max_slots*5,)
    padded with zeros
    """
    buf = np.zeros((max_slots, 5), dtype=np.float32)
    for i, (cid, x, y, score) in enumerate(raw[:max_slots]):
        buf[i, 0] = cid
        buf[i, 1] = x
        buf[i, 2] = y
        buf[i, 3] = score
        # buf[i,4] stays 0
    return buf.ravel()

# ─────────────────────────────────────────────────────────────────────────────
# Main live analysis loop
# ─────────────────────────────────────────────────────────────────────────────
def analyze_live():
    CB_H, CB_W = 39, 70
    minimap_center_ratio = ("", "")
    perm_rois, tab_rois = parse_boxes()
    load_and_cache_item_icons(ITEM_SIZE)

    last_tab_time = time.time() - 5
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        try:
            while True:
                now = time.time()
                last_minimap_time = time.time()
                # --- every 5s, press-and-hold Tab and extract tab stats once ---
                if now - last_tab_time >= 5:
                    keyboard.press('tab')
                    time.sleep(0.25)
                    img2 = np.array(sct.grab(monitor))[:,:,:3]
                    keyboard.release('tab')

                    frame2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    stats_tab = []
                    # OCR dispatch
                    for cat,(rs,cs,kind) in tab_rois.items():
                        if kind=='text':
                            roi = frame2[rs,cs]
                            key = f"tab:{now}:{cat}"
                            proc = preprocess_for_ocr(cat, roi)
                            ocr_queue.put((key, proc))
                            
                    ocr_queue.join()
                    # OCR retrieval
                    for cat,(rs,cs,kind) in tab_rois.items():
                        if kind=='text':
                            key = next(k for k in ocr_results if k.startswith("tab:") and k.endswith(f":{cat}"))
                            stats_tab.append((cat, ocr_results.pop(key,'')))
                        elif kind=='images':
                            # exactly the same logic as in perm_rois
                            roi = frame2[rs,cs]
                            name, _ = match_item(roi)
                            stats_tab.append((cat, name or ''))
                        elif kind=='bar':
                            roi = frame2[rs,cs]
                            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                            stats_tab.append((cat, f"{(gray>60).mean()*100:.1f}%"))
                        else:
                            stats_tab.append((cat, ''))
                    # print("=== TAB STATS ===")
                    # print(json.dumps(stats_tab, ensure_ascii=False))
                    last_tab_time = now

                # --- permanent extraction each frame ---
                img = np.array(sct.grab(monitor))[:,:,:3]
                frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                stats_perm = []
                # OCR dispatch
                for cat,(rs,cs,kind) in perm_rois.items():
                    if kind=='text':
                        roi = frame[rs,cs]
                        key = f"perm:{now}:{cat}"
                        proc = preprocess_for_ocr(cat, roi)
                        ocr_queue.put((key, proc))
                ocr_queue.join()
                # OCR retrieval + others
                for cat,(rs,cs,kind) in perm_rois.items():
                    roi = frame[rs,cs]
                    if kind=='text':
                        key = next(k for k in ocr_results if k.startswith("perm:") and k.endswith(f":{cat}"))
                        stats_perm.append((cat, ocr_results.pop(key,'')))
                    elif kind=='bar':
                        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                        stats_perm.append((cat, f"{(gray>60).mean()*100:.1f}%"))
                    elif kind=='images':
                        name,_ = match_item(roi)
                        stats_perm.append((cat, name or ''))
                    elif kind == "minimap":
                        # only re-run every 5s:
                        # import matplotlib.pyplot as plt
                        # plt.imshow(roi)
                        # plt.show()
                        if time.time() - last_minimap_time >= 0.0:
                            results = detect_whitelist_only(
                                roi,
                                minimap_model,
                                device,
                                whitelist_ids,
                                score_thresh=0.5
                            )
                            if not results:
                                val = ""
                            else:
                                val = "|".join(
                                f"{nm}:{x:.3f},{y:.3f}:{s:.2f}"
                                for nm,x,y,s in results
                                )
                            last_minimap_val  = val
                            last_minimap_time = time.time()
                        else:
                            val = last_minimap_val

                        stats_perm.append((cat, val))
                    else:
                        stats_perm.append((cat, ''))
                # print("=== PERM STATS ===")
                # print (json.dumps(stats_perm, ensure_ascii=False))

                MM_Y1, MM_Y2 = 778, 1064
                MM_X1, MM_X2 = 1620, 1906
                roi_box = (MM_X1,MM_X2,MM_Y1,MM_Y2)
                found = find_camera_box_in_minimap(frame, roi_box, CB_W, CB_H)
                if found:
                    cx_px, cy_px = found

                    # — ONLY update last_center (with persistence logic) —
                    last_center = (cx_px, cy_px)
                    candidate_center = None
                    candidate_count = 0

                # 3b) Derive minimap_center_ratio from last_center
                if last_center is not None:
                    minimap_center_ratio = (
                        (last_center[0] - roi_box[0]) / float(MM_X2 - MM_X1),
                        (last_center[1] - roi_box[2]) / float(MM_Y2 - MM_Y1)
                    )
                
                stats_perm.append(("minimap_x_ratio", minimap_center_ratio[0]))
                stats_perm.append(("minimap_y_ratio", minimap_center_ratio[1]))

                # ─── UNIT DETECTION + HEALTHBAR MATCHING ──────────────────────────────
                pil   = Image.fromarray(frame)
                dets, lbls = predict(rfdetr_model, pil, threshold=0.5)
                # lbls e.g. ["BlueChampionHealthbar 0.85", "Blue Melee 0.72", "Garen 0.94", …]
                annotated = draw_detections(pil, dets, lbls)

                # clean up class names & get boxes
                all_xyxy    = dets.xyxy                     # (K×4) numpy array
                all_xywh    = [xyxy_to_xywh(b) for b in all_xyxy]
                class_names = [l.rsplit(" ",1)[0] for l in lbls]  # drop the “0.85” part

                # which detections are healthbars?
                hb_inds = [i for i,c in enumerate(class_names) if "Healthbar" in c]
                sp_inds = [i for i in range(len(class_names)) if i not in hb_inds]

                hb_boxes  = np.array([all_xywh[i] for i in hb_inds],  dtype=int)
                hb_labels = [class_names[i]                   for i in hb_inds]
                sp_boxes  = np.array([all_xywh[i] for i in sp_inds],  dtype=int)
                sp_labels = [class_names[i]                   for i in sp_inds]

                # bucket the sprites
                minion_sp_inds = [i for i,n in enumerate(sp_labels) if n in MINION_CLASSES]
                champ_sp_inds  = [i for i,n in enumerate(sp_labels) if n in CHAMPION_BODY_CLASSES]
                tower_sp_inds  = [i for i,n in enumerate(sp_labels) if n in TOWER_LABELS]

                # ─── BUILD RAW DETECTION LISTS FOR POLICY ─────────────────────────
                # champ_raw: [(class_id, norm_x, norm_y, score), ...]
                champ_raw  = []
                for i in champ_sp_inds:
                    name = sp_labels[i]
                    if name == "Pet":
                        continue
                    cid  = CONDENSED_CHAMPIONS_TO_I[name]
                    x,y,w,h = sp_boxes[i]
                    cx = (x + w/2) / float(1920)
                    cy = (y + h/2) / float(1080)
                    # if you have a per-detection confidence you can use it here:
                    score = 1.0
                    champ_raw.append((cid, cx, cy, score))

                # minion_raw
                minion_raw = []
                for i in minion_sp_inds:
                    name = sp_labels[i]
                    cid  = NUM_CHAMPION_CLASSES + MINION_CLASSES.index(name)
                    x,y,w,h = sp_boxes[i]
                    cx = (x + w/2) / float(1920)
                    cy = (y + h/2) / float(1080)
                    minion_raw.append((cid, cx, cy, 1.0))

                # tower_raw
                tower_raw  = []
                for i in tower_sp_inds:
                    name = sp_labels[i]
                    cid  = NUM_CHAMPION_CLASSES + NUM_MINION_CLASSES + TOWER_LABELS.index(name)
                    x,y,w,h = sp_boxes[i]
                    cx = (x + w/2) / float(1920)
                    cy = (y + h/2) / float(1080)
                    tower_raw.append((cid, cx, cy, 1.0))

                # helper to subset a box‐array by indices (or get an empty (0,4) array)
                def subset(boxes, inds):
                    return boxes[inds] if inds else np.empty((0,4), int)

                # match + record routine
                def record_matches(category, hb_substr, sprite_inds):
                    # pick only the hb slots for this category
                    hb_sub_inds = [i for i,hl in enumerate(hb_labels) if hb_substr in hl]
                    hb_sub      = subset(hb_boxes,  hb_sub_inds)
                    sp_sub      = subset(sp_boxes,  sprite_inds)

                    matches, _, unmatched_sp = get_matches(hb_sub, sp_sub)
                    # matched → crop & compute true %
                    for h_i, s_i in matches:
                        hb_i = hb_sub_inds[h_i]
                        sp_i = sprite_inds[s_i]
                        x,y,w_,h_ = hb_boxes[hb_i]
                        color = 'blue' if hb_labels[hb_i].startswith("Blue") else 'red'
                        pct   = minion_health_percent(frame[y:y+h_, x:x+w_], bar_color=color)
                        key   = f"{category}_{sp_labels[sp_i].replace(' ','_')}"
                        stats_perm.append((key, f"{pct:.1f}%"))
                    # any unmatched sprite → default
                    for s_i in unmatched_sp:
                        key = f"{category}_{sp_labels[sprite_inds[s_i]].replace(' ','_')}"
                        stats_perm.append((key, "75.0%"))

                # run for each group
                record_matches("minion",    "MinionHealthbar",   minion_sp_inds)
                record_matches("champ",     "ChampionHealthbar", champ_sp_inds)
                record_matches("tower",     "Healthbar",         tower_sp_inds)

                # now stats_perm has keys like:
                #   "minion_Blue_Melee": "82.3%",
                #   "champ_Garen":        "47.0%",
                #   "tower_BlueTower":    "99.0%",
                # and any lonely sprite will be "75.0%".

                # print(stats_perm)

                text_rois = []
                for cat, (rs, cs, kind) in perm_rois.items():
                    if kind == 'text':
                        text_rois.append((cat, preprocess_for_ocr(cat, frame[rs, cs])))

                if text_rois:
                    # 2) find max dims
                    max_h = max(roi.shape[0] for _, roi in text_rois)
                    max_w = max(roi.shape[1] for _, roi in text_rois)

                    # 3) build canvas
                    n = len(text_rois)
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(n / cols))
                    canvas = np.zeros((rows*max_h, cols*max_w, 3), dtype=np.uint8)

                    # 4) pad & blit each ROI
                    for idx, (cat, roi) in enumerate(text_rois):
                        h, w = roi.shape[:2]

                        # pad bottom/right to (max_h, max_w)
                        pad_h = max_h - h
                        pad_w = max_w - w
                        padded = cv2.copyMakeBorder(
                            roi,
                            top=0, bottom=pad_h,
                            left=0, right=pad_w,
                            borderType=cv2.BORDER_CONSTANT,
                            value=[0,0,0]
                        )
                        # annotate label in green BGR
                        patch = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
                        cv2.putText(
                            patch, cat,
                            (2, max_h-4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0,255,0), 1, cv2.LINE_AA
                        )

                        r, c = divmod(idx, cols)
                        canvas[r*max_h:(r+1)*max_h, c*max_w:(c+1)*max_w] = patch

                    # 5) show and refresh
                    # cv2.imshow('Text ROIs', annotated)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     break

                # small delay
                # Merge perm + tab stats
                stats = dict(stats_perm + stats_tab)

                # 1) to_float helper (handles %, times, empty)
                def to_float(v):
                    if isinstance(v, (int, float)): return float(v)
                    if isinstance(v, str) and v.endswith('%'):
                        try: return float(v.strip('%'))/100.0
                        except: return 0.0
                    if isinstance(v, str) and ':' in v:
                        m,s = v.split(':',1)
                        try: return (int(m)*60 + int(s)) / (30*3600)  # normalized by max time
                        except: return 0.0
                    try:
                        return float(v)
                    except:
                        return 0.0

                # 2) Continuous keys in order with their normalizations baked in
                cont_f = [
                    # game-time normalized to fraction of 30m
                    to_float(stats.get('game-time', '')),  

                    # replace garen_pos with center if needed:
                    960/1920, 540/1080,                     

                    # minimap ratios already 0–1
                    to_float(stats.get('minimap_x_ratio', 0)),
                    to_float(stats.get('minimap_y_ratio', 0)),

                    # move_dir & xp_bar (we don’t normalize these here)
                    0.0,  # move_dir placeholder
                    0.0,  # xp-bar placeholder

                    # combat stats: hp/100, dmg/300, armor/300, mr/300
                    to_float(stats.get('health-bar', ''))/100.0,  
                    to_float(stats.get('attack-dmg', 0))/300.0,
                    to_float(stats.get('armor', 0))/300.0,
                    to_float(stats.get('magic-resist', 0))/300.0,

                    # move-speed normalized by 600
                    to_float(stats.get('move-speed', 0))/600.0,

                    # cooldowns normalized by their maxima
                    to_float(stats.get('q-cd', 0))/8.0,
                    to_float(stats.get('w-cd', 0))/10.0,
                    to_float(stats.get('e-cd', 0))/12.0,
                    to_float(stats.get('r-cd', 0))/120.0,
                    to_float(stats.get('d-cd', 0))/300.0,
                    to_float(stats.get('f-cd', 0))/300.0,
                ]

                # 3) Parse KDA for b1…b5, r1…r5 => 30 floats
                def parse_kda(raw):
                    parts = raw.split('/')
                    if len(parts)==2: parts.append('0')
                    parts = (parts + ['0','0','0'])[:3]
                    try:
                        ret = [float(parts[0]), float(parts[1]), float(parts[2])]
                    except:
                        return ['0','0','0']
                    return ret

                kdas = []
                for side in ['b','r']:
                    for i in range(1,6):
                        kdas += parse_kda(stats.get(f"{side}{i}-kda", "0/0/0"))

                # 4) CS for b1…b5, r1…r5
                cs = []
                for side in ['b','r']:
                    for i in range(1,6):
                        cs.append(to_float(stats.get(f"{side}{i}-cs", 0)))

                # 5) Health levels & champion levels (already 0–1)
                health_levels = [ to_float(stats.get(f"{side}{i}-health", 100))/100.0
                                for side in ['b','r'] for i in range(1,6) ]

                levels = [ to_float(stats.get(f"{side}{i}-level", 1))/18.0
                        for side in ['b','r'] for i in range(1,6) ]

                # 6) Objectives normalized (towers/11, grubs/3, heralds-barons/2, dragons/2, kills/50)
                objectives = []
                for obj, div in zip(
                    ["towers","grubs","heralds-barons","dragons","kills"],
                    [11.0,     3.0,     2.0,              2.0,       50.0]
                ):
                    objectives.append(to_float(stats.get(f"b-{obj}",0))/div)
                    objectives.append(to_float(stats.get(f"r-{obj}",0))/div)

                # 7) Build state
                state = cont_f + kdas + cs + health_levels + levels + objectives

                # ────────────────────────────────────────────────────────────────
                # 8) Build detection block exactly as before
                # ────────────────────────────────────────────────────────────────
                champ_block  = make_det_buffer(champ_raw,  MAX_NUM_CHAMPION_DETECTIONS)
                minion_block = make_det_buffer(minion_raw, MAX_NUM_MINION_DETECTIONS)
                tower_block  = make_det_buffer(tower_raw,  MAX_NUM_TOWER_DETECTIONS)

                det_block = np.concatenate([champ_block, minion_block, tower_block], axis=0)
                # ensure float32
                det_block = det_block.astype(np.float32)

                # 9) Items (zero‐filled for now)
                items_flat = np.zeros((policy.num_items_in_game,), dtype=np.float32)

                # 10) Final obs vector
                obs_np = np.concatenate([
                    np.array(state, dtype=np.float32),
                    det_block,
                    items_flat
                ], axis=0)

                # 11) Wrap for policy: (1,1,obs_dim)
                obs_tensor = torch.from_numpy(obs_np).view(1,1,-1).to(device)

                # 12) Predict & act
                with torch.no_grad():
                    action = policy.predict(obs_tensor)

                print("Action:", action, f"Reward={reward:.1f}")
                execute_action(action)

                # 13) Tiny delay
                time.sleep(0.1)


        except KeyboardInterrupt:
            pass
        finally:
            ocr_queue.put((None,None))

if __name__=='__main__':
    analyze_live()
