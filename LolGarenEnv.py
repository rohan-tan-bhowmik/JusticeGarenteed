import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import math
from collections import deque
import csv

# Lane‐bounds (normalized) for clipping movement
LANE_X_MIN, LANE_X_MAX = 0.15, 0.85
LANE_Y_MIN, LANE_Y_MAX = 0.18, 0.55

# Precompute unit vectors for 24 directions (every 15 degrees)
N_DIRS = 24
STEP_R = 0.05  # normalized step size
POLAR_UNIT_VECTORS = [
    (math.cos(2 * math.pi * i / N_DIRS), math.sin(2 * math.pi * i / N_DIRS))
    for i in range(N_DIRS)
]
from classes import CLASSES, HEALTHBAR_CLASSES

itochamp = CLASSES 
champtoi = {v: k for k, v in itochamp.items()}

def convert_row_dict(row_dict):
    """
    Parse the row dictionary from the CSV file into a numpy array.
    Detections are variable length - each champion (category) in top lane gets an embedding vector of size D
    Embeds all detections into fixed flattened vector (numpy array).
    Other continuous fields are included in the vector but kept as is
    """
    pass

def parse_csvs(ocr_csv, movement_csv, ocr_sampling_rate=9, movement_sampling_rate=3):
    """
    Returns a list of dicts, each mapping a movement-frame to a merged dict of
    movement info + OCR info (sampled at ocr_sampling_rate).
    """
    # 1) Read all OCR rows into a list of “{frame: {…fields…}}”
    ocr_data = []
    with open(ocr_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = row["frame"]
            # Copy everything except “frame” into a new dict
            info = {k: v for k, v in row.items() if k != "frame"}
            ocr_data.append({frame: info})

    all_data = []
    with open(movement_csv, newline="") as f:
        reader = csv.DictReader(f)

        # Pull off the very first movement-row to get start_frame
        try:
            first_row = next(reader)
        except StopIteration:
            return []  # empty file

        start_frame = int(first_row["frame"])
        offset = start_frame % ocr_sampling_rate

        # Prepare to merge OCR in lockstep
        ocr_idx = 0
        ocr_dict = ocr_data[ocr_idx]  # this is like {"123": {…ocr fields…}}

        # Process the first movement-row, then loop over the rest
        for row_dict in [first_row] + list(reader):
            frame = int(row_dict["frame"])
            print(frame)
            # Merge row_dict + ocr_ dict into a fresh dict
            # Note: ocr_dict is {"<ocr_frame>": {..fields..}}, 
            # so we need its inner dict, not the key.
            ocr_frame_str, ocr_fields = next(iter(ocr_dict.items()))

            merged = row_dict.copy()
            merged.update(ocr_fields)
            del merged["frame"]  # remove "frame" from merged dict
            all_data.append({frame: merged})

            # Update OCR index every time we cross an OCR sampling boundary
            if (ocr_idx % ocr_sampling_rate == offset
                    and ocr_idx < len(ocr_data) - 1):
                ocr_idx += 1
                ocr_dict = ocr_data[ocr_idx]

    return all_data

class LoLGarenEnv(gym.Env):
    pass

def main():
    pass

if __name__ == "__main__":
    main()