import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm

def estimate_camera_motion(prev: np.ndarray, curr: np.ndarray):
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    feats = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=5)
    if feats is None:
        return np.eye(2, 3, np.float32), np.zeros(2)
    pts, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, feats, None)
    old = feats[st.flatten() == 1]
    new = pts[st.flatten() == 1]
    if len(old) < 10:
        return np.eye(2, 3, np.float32), np.zeros(2)
    M, _ = cv2.estimateAffine2D(old, new, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return np.eye(2, 3, np.float32), np.zeros(2)
    return M, M[:, 2]

def find_camera_box_in_minimap(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    full_w: int,
    full_h: int
) -> tuple[int, int] | None:
    """
    Find the center of a ~70×40 white‐bordered “camera box” in the minimap ROI.
    Returns (cx_px, cy_px) in frame coordinates, or None if not found.
    """
    x1, x2, y1, y2 = roi
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    roi_h, roi_w = gray.shape

    # 1) Threshold to isolate white border pixels
    _, mask_white = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 2) Find contours on that white mask
    cnts, _ = cv2.findContours(mask_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    target_ar = full_w / float(full_h)
    min_area = full_w * full_h * 0.02   # ignore tiny specks
    tol = 0.3                            # ±30% tolerance in size/aspect

    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            break

        bx, by, bw, bh = cv2.boundingRect(c)
        ar = bw / float(bh if bh > 0 else 1)

        # 2a) If bounding‐box roughly matches expected size/aspect → accept.
        if (abs(ar - target_ar) < tol
            and abs(bw - full_w) < full_w * tol
            and abs(bh - full_h) < full_h * tol):
            cx_px = x1 + bx + bw // 2
            cy_px = y1 + by + bh // 2
            return cx_px, cy_px

        # 2b) Otherwise, try polygon approximation to detect partial edges:
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

            # “Long enough” vertical segment?
            if dx < full_w * 0.2 and dy > full_h * 0.4:
                verticals.append((p1, p2))

            # “Long enough” horizontal segment?
            elif dy < full_h * 0.2 and dx > full_w * 0.4:
                horizontals.append((p1, p2))

        # If we see at least one vertical and/or one horizontal, infer missing corner:
        if verticals or horizontals:
            inferred_cx_px = None
            inferred_cy_px = None

            # If we have a vertical edge, infer exact x using that edge
            if verticals:
                (vx1, vy1), (vx2, vy2) = verticals[0]
                v_x_mean = (vx1 + vx2) / 2
                # If this segment is left‐side of crop, p_x = vx1
                if v_x_mean < roi_w / 2:
                    inferred_cx_px = x1 + int(v_x_mean + full_w / 2)
                else:
                    # right‐side segment
                    inferred_cx_px = x1 + int(v_x_mean - full_w / 2)

            # If we have a horizontal edge, infer exact y using that edge
            if horizontals:
                (hx1, hy1), (hx2, hy2) = horizontals[0]
                h_y_mean = (hy1 + hy2) / 2
                if h_y_mean < roi_h / 2:
                    inferred_cy_px = y1 + int(h_y_mean + full_h / 2)
                else:
                    inferred_cy_px = y1 + int(h_y_mean - full_h / 2)

            # If only vertical edge is seen (no horizontals), use contour bbox y‐center
            if verticals and not horizontals:
                inferred_cy_px = y1 + by + bh // 2

            # If only horizontal edge is seen (no verticals), use contour bbox x‐center
            if horizontals and not verticals:
                inferred_cx_px = x1 + bx + bw // 2

            if inferred_cx_px is not None and inferred_cy_px is not None:
                # Clamp to ROI to avoid out‐of‐bounds jitter
                inferred_cx_px = np.clip(inferred_cx_px, x1, x2)
                inferred_cy_px = np.clip(inferred_cy_px, y1, y2)
                return inferred_cx_px, inferred_cy_px

    # 3) Template‐matching fallback against mask_white:
    best_score = 0.0
    best_center = None
    template_base = np.zeros((full_h, full_w), dtype=np.uint8)
    cv2.rectangle(template_base, (0, 0), (full_w - 1, full_h - 1), 255, 1)

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

    # 4) HoughLinesP fallback on mask_white
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
    end = (int(center[0] - mv[0] * 10), int(center[1] - mv[1] * 10))
    cv2.arrowedLine(frame, center, end, (0, 0, 255), 2, tipLength=0.3)
    return frame

def resize_frame(frame: np.ndarray, scale: float):
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

def main(video_path: str, scale: float, skip_n: int, wild_thresh: float):
    MM_Y1, MM_Y2 = 778, 1064
    MM_X1, MM_X2 = 1620, 1906
    CB_H, CB_W = 39, 70

    roi = (
        int(MM_X1 * scale),
        int(MM_X2 * scale),
        int(MM_Y1 * scale),
        int(MM_Y2 * scale)
    )
    full_w = int(CB_W * scale)
    full_h = int(CB_H * scale)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_iters = (total_frames + skip_n - 1) // skip_n

    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame.")
    prev = resize_frame(prev, scale)

    # Threshold (in px) beyond which a new “found” center is considered a jump
    jump_thresh = min(full_w//2, full_h//2)
    # Number of consecutive frames needed to confirm a large jump
    persistence_frames = 12

    last_center = None
    candidate_center = None
    candidate_count = 0
    idx = 0

    with tqdm(total=total_iters, desc="Processing frames") as pbar:
        while True:
            for _ in range(skip_n - 1):
                cap.read()
                idx += 1

            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            frame = resize_frame(frame, scale)

            M, mv = estimate_camera_motion(prev, frame)

            if np.linalg.norm(mv) > wild_thresh:
                found = find_camera_box_in_minimap(frame, roi, full_w, full_h)
                if found:
                    cx_px, cy_px = found
                    if last_center is None:
                        # First detection: accept immediately
                        last_center = (cx_px, cy_px)
                        candidate_center = None
                        candidate_count = 0
                    else:
                        dist = np.hypot(cx_px - last_center[0], cy_px - last_center[1])
                        if dist < jump_thresh:
                            # Small move: accept immediately
                            last_center = (cx_px, cy_px)
                            candidate_center = None
                            candidate_count = 0
                        else:
                            # Big jump: check persistence
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
                # If found is None, keep last_center unchanged

            if last_center is not None:
                cv2.circle(frame, last_center, radius=5, color=(0, 0, 255), thickness=-1)

            prev = frame

            frame = draw_arrow(frame, mv)

            cv2.imshow("Tracked Frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            pbar.update(1)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to League replay clip (e.g., .mp4)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Downscale factor for resolution (e.g., 0.5 = 50%)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Only process every Nth frame")
    parser.add_argument("--wild", type=float, default=0.0,
                        help="Threshold above which optical‐flow is considered 'wild'")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    main(args.video, args.scale, args.skip, args.wild)
