import cv2
import numpy as np
import argparse
import os


def estimate_camera_motion(prev: np.ndarray, curr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate affine camera motion from prev → curr frame using background features."""
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    features = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=5)
    if features is None:
        return np.eye(2, 3, dtype=np.float32), np.array([0.0, 0.0])

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None)
    good_old = features[status.flatten() == 1]
    good_new = new_pts[status.flatten() == 1]

    if len(good_old) < 10:
        return np.eye(2, 3, dtype=np.float32), np.array([0.0, 0.0])

    M, _ = cv2.estimateAffine2D(good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3)
    if M is None:
        return np.eye(2, 3, dtype=np.float32), np.array([0.0, 0.0])

    motion_vector = M[:, 2]
    return M, motion_vector


def draw_arrow(frame: np.ndarray, motion_vec: np.ndarray):
    """Draw a motion vector arrow on the frame (OpenCV-style)."""
    h, w = frame.shape[:2]
    center = (int(w // 2), int(h // 2))
    end = (
        int(center[0] - motion_vec[0] * 10),
        int(center[1] - motion_vec[1] * 10)
    )
    cv2.arrowedLine(frame, center, end, color=(0, 0, 255), thickness=2, tipLength=0.3)
    return frame


def resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    """Resize frame by a given scale factor."""
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def main(video_path: str, scale: float, skip_n: int):
    cap = cv2.VideoCapture(video_path)
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Could not read first frame.")
    prev = resize_frame(prev, scale)

    transform_accum = np.eye(3)
    frame_idx = 0

    while True:
        # Skip frames
        for _ in range(skip_n - 1):
            cap.read()
            frame_idx += 1

        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame = resize_frame(frame, scale)

        M, motion_vec = estimate_camera_motion(prev, frame)
        M_hom = np.vstack([M, [0, 0, 1]])
        transform_accum = transform_accum @ M_hom
        cam_x, cam_y = transform_accum[0, 2], transform_accum[1, 2]

        print(f"[Frame {frame_idx}] Camera Position Estimate: x={cam_x:.2f}, y={cam_y:.2f}")
        vis = draw_arrow(frame.copy(), motion_vec)
        cv2.imshow("Camera Motion (Red = Direction)", vis)

        key = cv2.waitKey(1)
        if key == 27:
            break

        prev = frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to League replay clip (e.g., .mp4)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Downscale factor for resolution (e.g., 0.5 = 50%)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Only process every Nth frame (e.g., 2 = every other frame)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(args.video)

    main(args.video, args.scale, args.skip)
