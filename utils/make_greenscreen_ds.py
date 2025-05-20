import cv2
import numpy as np
import os
from tqdm import tqdm 
from roboflow import Roboflow
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed

def poisson_sample_indices(video_len, num_samples, offset= 0, lamb =None):
    if lamb is None:
        lamb = video_len / num_samples  # average distance between frames
        print(f"Using default lambda: {lamb}")
    
    indices = []
    current = offset
    while len(indices) < num_samples:
        gap = np.random.poisson(lamb)
        current += gap
        if current >= video_len:
            break
        indices.append(current)

    # If not enough samples, pad randomly
    while len(indices) < num_samples:
        indices.append(np.random.randint(offset, video_len))

    return sorted(indices[:num_samples])

def stratify_sample_indices(video_len, num_samples, offset = 0):
    chunk_size = (video_len - offset) // num_samples
    indices = []

    for i in range(num_samples):
        start = i * chunk_size + offset
        end = start + chunk_size if i < num_samples - 1 else video_len
        if start >= video_len:
            break
        idx = np.random.randint(start, end)
        indices.append(idx)

    return indices

def extract_frames(video_path, start_second, num_frames):
    """
    Extracts training frames from video file at specified start_second
    """
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    start_frame = int(start_second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    train_indices = poisson_sample_indices(frame_count, num_frames, start_frame)
    val_indices = stratify_sample_indices(frame_count, int(num_frames * 0.25), start_frame)
    #Extract frames from indices
    train_frames, val_frames = [], []
    for i in train_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            train_frames.append(frame)
        else:
            print(f"Failed to read frame at index {i}")

    for i in val_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            val_frames.append(frame)
        else:
            print(f"Failed to read frame at index {i}")

    cap.release()
    return train_frames, val_frames

def save_frames(frames, champion_name, output_dir):
    """
    Save frames to output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, frame in enumerate(frames):
        frame_path = os.path.join(output_dir, f"{champion_name}_{i:05d}.png")
        cv2.imwrite(frame_path, frame) 

def upload_to_roboflow(data_dir, api_key, project_name, split = "train", max_workers = 8):
    rf = Roboflow(api_key=api_key)
    project = rf.project(project_name)

    tasks = []    
    for fname in os.listdir(data_dir):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            fpath = os.path.join(data_dir, fname)
            tasks.append((fpath, split))

    def upload_one(fpath, split):
        try:
            project.upload(fpath, split=split)
            return (fpath, "ok")
        except Exception as e:
            return (fpath, str(e))

    # Run uploads in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(upload_one, fpath, split) for fpath, split in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Uploading"):
            results.append(future.result())

    failed = [r for r in results if r[1] != "ok"]
    if failed:
        print(f"\n Failed uploads ({len(failed)}):")
        for fpath, error in failed:
            print(f"{fpath} — {error}")
    else:
        print("\n All images uploaded successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility for extracting frames and uploading to Roboflow.")
    subparsers = parser.add_subparsers(dest="command", help="Subcommands: extract or upload")
    # Example usage: python make_greenscreen_ds.py extract --video_folder_path /path/to/videos --start_second 0 --num_frames 20 --output_dir /path/to/output --champion_name champion_name

    extract_parser = subparsers.add_parser("extract", help="Extract frames from a video file.")
    extract_parser.add_argument("--video_folder_path", type=str, required=True, help="Path to the folder with video file.")
    extract_parser.add_argument("--start_second", type=float, default=0, help="Start second for frame extraction.")
    extract_parser.add_argument("--num_frames", type=int, default=20, help="Number of frames to extract.")
    extract_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted frames.")
    extract_parser.add_argument("--champion_name", type=str, required=True, help="Champion name for naming frames.")

    upload_parser = subparsers.add_parser("upload", help="Upload frames to Roboflow.")
    upload_parser.add_argument("--data_dir", type=str, required=True, help="Directory containing frames to upload.")
    upload_parser.add_argument("--api_key", type=str, required=True, help="Roboflow API key.")
    upload_parser.add_argument("--project_name", type=str, required=True, help="Roboflow project name.")
    upload_parser.add_argument("--split", type=str, default="train", choices=["train", "valid"], help="Dataset split (train or val).")
    upload_parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent uploads.")
    args = parser.parse_args()

    if args.command == "extract":
        # Extract frames
        video_path = os.path.join(args.video_folder_path, args.champion_name + ".mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file {video_path} does not exist.")
        train_frames, val_frames = extract_frames(video_path, args.start_second, args.num_frames)

        # Save frames
        save_frames(train_frames, args.champion_name + "_train", os.path.join(args.output_dir, "train"))
        save_frames(val_frames, args.champion_name + "_val", os.path.join(args.output_dir, "val"))

    elif args.command == "upload":
        upload_to_roboflow(args.data_dir, args.api_key, args.project_name, args.split, args.max_workers)

    else:
        parser.print_help()