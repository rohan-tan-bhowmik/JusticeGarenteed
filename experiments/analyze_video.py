import os
import sys
import pathlib
from pathlib import Path
import cv2
import easyocr
import numpy as np
import torch
from typing import Dict, Tuple
from pathlib import Path
import minimap
import re
from PIL import Image
import csv
import time
from queue import Queue
from threading import Thread
from tqdm import tqdm  # NEW
import matplotlib.pyplot as plt  # NEW: for testing frame ROIs
import json
from minimap import ImageDrawer

# Configuration
BOX_FILE = "boxes.txt"
ITEM_FOLDER = "item_icons"
CONF_THRESHOLD = 0.0
ITEM_SIZE = (24, 24)
ROIType = Tuple[slice, slice, str]

USE_CV2_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0
USE_TORCH_CUDA = torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

icons_path = 'league_icons/'
image_drawer = ImageDrawer(
    os.path.join(icons_path, 'champions'),
    os.path.join(icons_path, 'minimap'),
    os.path.join(icons_path, 'fog'),
    os.path.join(icons_path, 'misc'),
    resize=(256, 256)
)
id_to_champ = image_drawer.id_to_champion

# Initialize EasyOCR (run in one thread only)
reader = easyocr.Reader(['en'], gpu=USE_TORCH_CUDA)

# OCR worker system
ocr_queue = Queue()
ocr_results = {}
def ocr_worker():
    while True:
        key, image = ocr_queue.get()
        if key is None:
            break
        text = reader.readtext(
            image,
            detail=0,
            paragraph=False,
            allowlist="0123456789.:k/"
        )
        ocr_results[key] = ' '.join(text).strip()
        ocr_queue.task_done()

ocr_thread = Thread(target=ocr_worker, daemon=True)
ocr_thread.start()

def load_metadata(replay_dir: str, replay_name: str) -> dict:
    meta_path = Path(replay_dir) / "metadata.json"
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    for replay in meta["replays"]:
        if replay["name"] == replay_name:
            return replay
    raise ValueError(f"Replay '{replay_name}' not found in {meta_path}")

icon_rgb_dict  = {}  # name → RGB image (resized to 24×24)
icon_gray_dict = {}  # name → grayscale 24×24

def load_and_cache_item_icons(size: Tuple[int,int]) -> None:
    """
    Populate icon_rgb_dict and icon_gray_dict so match_item can be much faster.
    """
    for path in Path(ITEM_FOLDER).glob("*.png"):
        img = cv2.imread(str(path))
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        icon_rgb_dict[path.stem] = resized_rgb
        # compute grayscale once:
        icon_gray_dict[path.stem] = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2GRAY)


def match_item(slot_rgb: np.ndarray) -> Tuple[str,float]:
    """
    slot_rgb: 24×24 RGB numpy array
    Returns (best_name, best_score) or (None, 0.0).
    """
    slot_gray = cv2.cvtColor(slot_rgb, cv2.COLOR_RGB2GRAY)
    best_name, best_score = None, -1.0

    # All icon_gray_dict values are already 24×24 grayscale.
    for name, ref_gray in icon_gray_dict.items():
        # Both slot_gray and ref_gray are same size (24×24).
        # TM_CCOEFF_NORMED on 24×24 is very fast.
        score = cv2.matchTemplate(slot_gray, ref_gray, cv2.TM_CCOEFF_NORMED).max()
        if score > best_score:
            best_name, best_score = name, score

    # If none found (score stays < 0), we return (None,0).
    if best_score < 0:
        return None, 0.0
    return best_name, best_score


def ocr_text(rgb: np.ndarray, key: str):
    gray     = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized  = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    blurred  = cv2.GaussianBlur(resized, (3, 3), 0)
    # kernel   = np.ones((2, 2), np.uint8)
    # dilated  = cv2.dilate(blurred, kernel, iterations=1)


    # # — re-sharpen with a simple kernel —
    # sh_kernel = np.array([[ 0, -1,  0],
    #                     [-1,  5, -1],
    #                     [ 0, -1,  0]], dtype=np.float32)
    # cleaned   = cv2.filter2D(dilated, ddepth=-1, kernel=sh_kernel)

    ocr_queue.put((key, blurred))


def bar_fill(rgb: np.ndarray, thresh: int = 60) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float((gray > thresh).mean() * 100)

def parse_boxes(file: str = BOX_FILE) -> Dict[str, ROIType]:
    pat = re.compile(r"\[(\d+):(\d+),\s*(\d+):(\d+)\]")
    rois: dict[str, ROIType] = {}
    cooldown_map: dict[str, float] = {}
    with open(file, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if not line:
                continue
            cat, rng, kind_str = [p.strip() for p in line.split(";", 2)]
            m = pat.search(rng)
            if not m:
                continue

            y0, y1, x0, x1 = map(int, m.groups())
            # handle “cooldown – N” specially
            kind_lower = kind_str.lower()
            if kind_lower.startswith("cooldown"):
                # expect format “cooldown - 8”
                _, cd_sec = kind_lower.split("-", 1)
                cooldown_map[cat] = float(cd_sec.strip())
                kind = "cooldown"
            else:
                kind = kind_lower

            rois[cat] = (slice(y0, y1), slice(x0, x1), kind)
    return rois, cooldown_map

def show_test_frame(video_path: pathlib.Path, frame_idx: int):
    """Load a single frame and display all 'text' ROIs via matplotlib."""
    boxes, _ = parse_boxes()
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame_bgr = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Cannot read frame {frame_idx} from {video_path}")

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # collect all text ROIs
    text_rois = []
    for cat, (rs, cs, kind) in boxes.items():
        if kind in ("images"):
            roi = frame_rgb[rs, cs]
            text_rois.append((cat, roi))

    n = len(text_rois) + 1
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    axes[0].imshow(frame_rgb)
    axes[0].set_title(f"Frame {frame_idx}")
    axes[0].axis('off')

    for i, (cat, roi) in enumerate(text_rois, start=1):
        # — preprocess as before —
        gray     = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        resized  = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cleaned  = cv2.GaussianBlur(resized, (3, 3), 0)

        axes[i].imshow(cleaned, cmap='gray')
        axes[i].set_title(cat)
        axes[i].axis('off')



    plt.tight_layout()
    plt.show()
    sys.exit(0)

def is_colored(roi_rgb: np.ndarray, sat_thresh: float = 15.0) -> bool:
    """Return True if ROI has enough color saturation."""
    hsv = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2HSV)
    # mean saturation channel
    return float(hsv[...,1].mean()) > sat_thresh

def analyze_video(path: pathlib.Path,
                  frame_interval: int = 30,
                  csv_path: str | None = None,
                  start_frame: int = 0):
    boxes, cooldown_map = parse_boxes()
    cooldown_start: dict[str, int | None] = {cat: None for cat in cooldown_map}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    device = torch.device("cuda" if USE_TORCH_CUDA else "cpu")
    model = minimap.create_model(171, device=device)
    model.load_state_dict(torch.load("fastrcnn_model.pt", map_location=device))
    model.to(device).eval()

    last_rois, last_results, last_values = {}, {}, {}
    frame_idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    csv_file = open(csv_path, 'w', newline='', encoding='utf-8') if csv_path else None

    all_fields = ["frame"] + [cat for cat in boxes]
    writer = csv.DictWriter(csv_file, fieldnames=all_fields) if csv_file else None
    if writer:
        writer.writeheader()

    load_and_cache_item_icons(ITEM_SIZE)

    MINIMAP_INTERVAL_SECONDS = 5.0
    MINIMAP_INTERVAL_FRAMES = int(MINIMAP_INTERVAL_SECONDS * fps)
    last_minimap_frame = start_frame - MINIMAP_INTERVAL_FRAMES

    with tqdm(total=total_frames, desc="Processing frames") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                if USE_CV2_CUDA:
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(frame_bgr)
                    frame = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2RGB).download()
                else:
                    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                row_data = {"frame": frame_idx}

                # first pass: queue OCR tasks for changed text ROIs
                for cat, (rs, cs, kind) in boxes.items():
                    roi = frame[rs, cs]
                    changed = cat not in last_rois or not np.array_equal(last_rois[cat], roi)
                    if kind == "text" and changed:
                        key = f"{frame_idx}:{cat}"
                        ocr_text(roi, key)

                ocr_queue.join()

                # second pass: compute values or reuse last
                for cat, (rs, cs, kind) in boxes.items():
                    roi = frame[rs, cs]
                    changed = cat not in last_rois or not np.array_equal(last_rois[cat], roi)

                    if not changed:
                        val = last_values[cat]
                    else:
                        if kind == "text":
                            key = f"{frame_idx}:{cat}"
                            val = ocr_results.pop(key, last_values.get(cat, ""))
                        elif kind == "bar":
                            val = bar_fill(roi)
                        elif kind == "images":
                            # Check uniform color first:
                            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                            if gray_roi.std() < 5:     # essentially constant
                                val = ""
                            else:
                                # match_item uses icon_gray_dict directly
                                best_name, _ = match_item(roi)
                                val = "" if best_name is None else best_name

                        elif kind == "minimap":
                            # only do a new detection if at least MINIMAP_INTERVAL_FRAMES have elapsed
                            if (frame_idx - last_minimap_frame) >= MINIMAP_INTERVAL_FRAMES:
                                minimap_crop = roi
                                results = detect_whitelist_only(
                                    minimap_crop,
                                    model,
                                    device,
                                    whitelist_ids=whitelist_ids,
                                    score_thresh=0.5
                                )
                                if not results:
                                    val = ""
                                else:
                                    val = "|".join(f"{nm}:{x:.3f},{y:.3f}:{s:.2f}"
                                                for nm, x, y, s in results)

                                # update our “last seen” for the minimap‐ROI so future frames within the interval reuse it
                                last_minimap_frame = frame_idx
                                last_rois[cat]   = minimap_crop.copy()
                                last_values[cat] = val
                            else:
                                # we are still within the 5‐second window: reuse whatever we had last time
                                val = last_values.get(cat, "")


                        elif kind == "cooldown":
                            SAT_THRESH = 80
                            VAL_THRESH = 40
                            max_cd = cooldown_map[cat]            # seconds
                            # ability is available → cancel any running countdown
                            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                            # mean_sat = hsv[..., 1].mean()
                            any_saturated = np.any(hsv[...,1] > SAT_THRESH)
                            pixel_y, pixel_x = 1, 14  # Relative to the cropped cooldown ROI
                            pixel_val = hsv[pixel_y, pixel_x][2] if pixel_y < hsv.shape[0] and pixel_x < hsv.shape[1] else 0
                            # plt.imshow(roi)
                            # plt.show()

                            if any_saturated:
                                # Overall still colored → available
                                cooldown_start[cat] = None
                                val = ""
                            elif pixel_val > VAL_THRESH and cooldown_start[cat] is None and cat == 'q-cd':
                                # Region desaturated, but this pixel still colored → still active
                                cooldown_start[cat] = None
                                val = "enabled"
                                # val = pixel_val
                            else:
                                # Fully unsaturated and pixel also desaturated → start/continue cooldown
                                if cooldown_start[cat] is None:
                                    cooldown_start[cat] = frame_idx
                                elapsed = (frame_idx - cooldown_start[cat]) / fps
                                remaining = cooldown_map[cat] - elapsed
                                val = f"{remaining:.1f}" if remaining > 0 else "0.0"

                        else:
                            # fallback for text, bar, images, etc.
                            val = '…'

                        last_rois[cat] = roi.copy()
                        last_results[cat] = f"{cat}→{val}"
                        last_values[cat] = val

                    row_data[cat] = val

                # helper to round values for CSV
                def round_to_sigfigs(value, sigfigs=3):
                    try:
                        num = float(value)
                        if num == 0:
                            return "0"
                        digits = sigfigs - int(np.floor(np.log10(abs(num)))) - 1
                        return str(round(num, digits))
                    except (ValueError, TypeError):
                        return value

                def process_row(row):
                    def process_val(v):
                        if isinstance(v, str) and ':' in v:
                            segments = v.split('|')
                            new_segments = []
                            for segment in segments:
                                parts = segment.split(':')
                                if len(parts) >= 3 and parts[2].replace('.', '', 1).isdigit():
                                    parts[2] = round_to_sigfigs(parts[2])
                                new_segments.append(':'.join(parts))
                            return '|'.join(new_segments)
                        return round_to_sigfigs(v)
                    return {k: process_val(v) for k, v in row.items()}

                if writer:
                    writer.writerow(process_row(row_data))

            frame_idx += 1
            pbar.update(1)

    cap.release()
    ocr_queue.put((None, None))
    ocr_thread.join()
    if csv_file:
        csv_file.close()
        print(f"Saved stats to {csv_path}")
    print("Done.")

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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_video.py <metadata.json> <replay_name> [frame_interval] [output_csv] [--test-frame N]")
        sys.exit(1)

    replay_dir = sys.argv[1]
    replay_name = sys.argv[2]
    replay = load_metadata(replay_dir, replay_name)

    blue_names = set(replay["blue_team"])
    red_names  = set(replay["red_team"])
    whitelist_names = blue_names | red_names
    whitelist_ids = { image_drawer.champion_to_id[name.lower()] for name in whitelist_names }


    video_path = Path(replay_dir) / replay["stats_video"]
    start_frame = int(replay["start_frame_stats"])
    frame_interval = int(sys.argv[3]) if len(sys.argv) >= 4 and not sys.argv[3].startswith('--') else 30
    output_csv = None
    test_frame = None

    # parse optional args
    for i, arg in enumerate(sys.argv[4:], start=4):
        if arg == "--test-frame" and i + 1 < len(sys.argv):
            test_frame = int(sys.argv[i + 1])
        elif not arg.startswith('--') and output_csv is None:
            output_csv = arg

    if test_frame is not None:
        show_test_frame(video_path, test_frame)
    else:
        analyze_video(video_path, frame_interval, output_csv, start_frame=start_frame)
