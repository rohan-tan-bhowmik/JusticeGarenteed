import io
import requests
import supervision as sv
import numpy as np
from PIL import Image
from rfdetr import RFDETRBase
from classes import CLASSES
import cv2
import time

def predict(model: RFDETRBase, image: Image.Image, threshold = 0.5) -> sv.Detections:
    """
    Predict bounding boxes and labels for the given image using the RF-DETR model.
    
    :param model: RFDETRBase model instance
    :param image: PIL Image to predict on
    :return: Detections object containing bounding boxes and labels
    """
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Perform prediction
    detections = model.predict(img_array, threshold=threshold)
    
    labels = [
        f"{CLASSES[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    return detections, labels

def draw_detections(image: Image.Image, detections: sv.Detections, labels) -> Image.Image:
    """
    Draw bounding boxes and labels on the image.
    
    :param image: PIL Image to draw on
    :param detections: Detections object containing bounding boxes and labels
    :return: Image with drawn detections
    """
    # Convert PIL Image to numpy array for drawing
    img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Draw detections on the image
    annotated_image = sv.BoxAnnotator().annotate(img_array, detections)
    annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

    return annotated_image

def detect_for_dir(img_dir: str, model: RFDETRBase, threshold=0.5):
    """
    Detect objects in all images in the specified directory using the RF-DETR model.
    
    :param img_dir: Directory containing images to process
    :param model: RFDETRBase model instance
    :param threshold: Confidence threshold for detections
    """
    import os
    from glob import glob

    img_paths = glob(os.path.join(img_dir, "*.jpg"))
    
    for img_path in img_paths:
        img = Image.open(img_path)
        detections, labels = predict(model, img, threshold)
        print(f"Detected {len(detections)} objects in {img_path}")
        draw_detections(img, detections, labels)

def detect_for_video(video_path: str, model: RFDETRBase, threshold=0.5, start_second=0):
    """
    Detect objects in a video using the RF-DETR model, starting from a specified second.
    
    :param video_path: Path to the video file
    :param model: RFDETRBase model instance
    :param threshold: Confidence threshold for detections
    :param start_second: The second from which to start processing the video
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    
    # Get the video's FPS and calculate the starting frame
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set the starting frame
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detections, labels = predict(model, img, threshold)
        annotated_img = draw_detections(img, detections, labels)
        
        # Display the annotated frame
        cv2.imshow("Detections", np.array(annotated_img))
        
        if cv2.waitKey(15) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Load the model
    weights_path = "checkpoints/checkpoint_best_total.pth"
    img_dir_path = "test_images"
    video_path = "replays/garen_gangplank.mp4"
    model = RFDETRBase(pretrain_weights=weights_path)
    # model = model.optimize_for_inference()

    # detect_for_dir("test_images", model, threshold=0.5)
    detect_for_video(video_path, model, threshold=0.5, start_second = 90)
