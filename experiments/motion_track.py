import cv2
import numpy as np
import argparse
import os


def build_roi_mask(h: int, w: int) -> np.ndarray:
    """
    Mask out UI/minimap at edges, leaving central 80%×80% for tracking.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    top, bottom = int(0.1 * h), int(0.9 * h)
    left, right = int(0.1 * w), int(0.9 * w)
    mask[top:bottom, left:right] = 255
    return mask


def robust_estimate_motion(prev_gray: np.ndarray,
                           curr_gray: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
    """
    1) LK + RANSAC to reject moving outliers
    2) iterative reweight (>1σ drop)
    3) median‐flow fallback
    """
    pts0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300,
                                   qualityLevel=0.01, minDistance=4,
                                   mask=mask)
    if pts0 is None:
        return np.array([0.0, 0.0])
    pts0 = pts0.reshape(-1, 2)

    pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts0, None)
    if pts1 is None or st is None:
        return np.array([0.0, 0.0])
    good0 = pts0[st.flatten() == 1]
    good1 = pts1[st.flatten() == 1]
    if len(good0) < 10:
        return np.array([0.0, 0.0])

    # RANSAC‐based translation
    M, inliers = cv2.estimateAffinePartial2D(
        good0, good1,
        method=cv2.RANSAC,
        ransacReprojThreshold=3,
        maxIters=1000
    )
    if M is not None and inliers is not None:
        dx, dy = M[0, 2], M[1, 2]
        sel = inliers.flatten() == 1
        flows = good1[sel] - good0[sel]
        if flows.shape[0] >= 5:
            dists = np.linalg.norm(flows - np.array([dx, dy]), axis=1)
            sigma = dists.std()
            if sigma > 0:
                keep = dists < sigma
                if keep.sum() >= 5:
                    dx, dy = flows[keep].mean(axis=0)
        return np.array([dx, dy])

    # fallback: median flow
    all_flows = good1 - good0
    return np.median(all_flows, axis=0)


def phase_correlation_motion(prev_gray: np.ndarray,
                             curr_gray: np.ndarray,
                             mask: np.ndarray) -> np.ndarray:
    """
    Phase‐correlation on masked floats to get (dx, dy).
    """
    prev_f = prev_gray.astype(np.float32) * (mask / 255.0)
    curr_f = curr_gray.astype(np.float32) * (mask / 255.0)
    shift, _ = cv2.phaseCorrelate(prev_f, curr_f)
    # shift is (dx, dy)
    return np.array(shift)


def draw_arrow(frame: np.ndarray, motion_vec: np.ndarray) -> np.ndarray:
    """Draw a red arrow at the center representing motion_vec."""
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    end = (
        int(center[0] - motion_vec[0] * 10),
        int(center[1] - motion_vec[1] * 10)
    )
    cv2.arrowedLine(frame, center, end, color=(0, 0, 255),
                    thickness=2, tipLength=0.3)
    return frame


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Resize frame by scale."""
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame,
                      (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def main(video_path: str, scale: float, skip_n: int):
    cap = cv2.VideoCapture(video_path)
    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Cannot read first frame.")
    first = resize_frame(first, scale)
    prev_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    # Build static ROI mask
    h, w = prev_gray.shape
    roi_mask = build_roi_mask(h, w)

    # Running average background for static‐mask (optional)
    bg_gray = prev_gray.astype(np.float32)
    alpha_bg = 0.03
    static_thresh = 10

    # Temporal smoothing state
    smooth_dx = 0.0
    smooth_dy = 0.0
    alpha_smooth = 0.9

    cam_x = cam_y = 0.0
    frame_idx = 0

    while True:
        # skip frames
        for _ in range(skip_n - 1):
            cap.read()
            frame_idx += 1

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = resize_frame(frame, scale)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # static‐mask (optional): compare to background
        diff = cv2.absdiff(curr_gray, cv2.convertScaleAbs(bg_gray))
        static_mask = (diff < static_thresh).astype(np.uint8) * 255
        bg_gray = (1 - alpha_bg) * bg_gray + alpha_bg * curr_gray

        combined_mask = cv2.bitwise_and(roi_mask, static_mask)

        # two motion estimates
        dx1, dy1 = robust_estimate_motion(prev_gray, curr_gray, combined_mask)
        dx2, dy2 = phase_correlation_motion(prev_gray, curr_gray, combined_mask)

        # combine and smooth
        dx = 0.5 * (dx1 + dx2)
        dy = 0.5 * (dy1 + dy2)
        smooth_dx = alpha_smooth * smooth_dx + (1 - alpha_smooth) * dx
        smooth_dy = alpha_smooth * smooth_dy + (1 - alpha_smooth) * dy

        cam_x += smooth_dx
        cam_y += smooth_dy

        print(f"[Frame {frame_idx}] Cam disp: x={cam_x:.2f}, y={cam_y:.2f}")

        vis = draw_arrow(frame.copy(), np.array([smooth_dx, smooth_dy]))
        cv2.imshow("Camera Motion", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            cam_x = cam_y = 0.0
            smooth_dx = smooth_dy = 0.0
            print(f"[Frame {frame_idx}] Reset camera position")

        prev_gray = curr_gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to League replay clip")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Resize scale (e.g. 0.5)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    main(args.video, args.scale, args.skip)
