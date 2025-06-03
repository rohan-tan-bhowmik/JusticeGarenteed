import io
import requests
import supervision as sv
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
from classes import CLASSES
import cv2
import time
from tqdm import tqdm  # <--- import tqdm

def predict(model: RFDETRBase, image: Image.Image, threshold=0.5) -> sv.Detections:
    """
    Predict bounding boxes and labels for the given image using the RF-DETR model.
    """
    img_array = np.array(image)
    detections = model.predict(img_array, threshold=threshold)
    labels = [
        f"{CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]
    return detections, labels

def draw_detections(image: Image.Image, detections: sv.Detections, labels) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    """
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    annotated_image = sv.BoxAnnotator().annotate(img_array, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)
    return annotated_image

def detect_for_dir(img_dir: str, model: RFDETRBase, threshold=0.5):
    """
    Detect objects in all images in the specified directory, with a tqdm bar.
    """
    import os
    from glob import glob

    img_paths = glob(os.path.join(img_dir, "*.jpg"))
    for img_path in tqdm(img_paths, desc="Processing images"):
        img = Image.open(img_path).convert("RGB")
        detections, labels = predict(model, img, threshold)
        print(f"Detected {len(detections)} objects in {img_path}")
        _ = draw_detections(img, detections, labels)  # we ignore display here

def detect_for_video(video_path: str, model: RFDETRBase, threshold=0.5, start_second=0):
    """
    Detect objects in a video using the RF-DETR model, starting from a specified second,
    and show a tqdm progress bar over frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = int(start_second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Compute how many frames remain from start_frame to end
    frames_to_process = max(0, total_frames - start_frame)
    pbar = tqdm(total=frames_to_process, desc="Video frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fuckgabe = time.time()
        detections, labels = predict(model, img, threshold)
        annotated_img = draw_detections(img, detections, labels)

        cv2.imshow("Detections", annotated_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pbar.update(1)

    pbar.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load the model
    weights_path = "champ_detection.pth"
    video_path = "replays/1-full.mp4"
    model = RFDETRBase(pretrain_weights=weights_path)
    model.device = "cuda"

    detect_for_video(video_path, model, threshold=0.5, start_second=300)
