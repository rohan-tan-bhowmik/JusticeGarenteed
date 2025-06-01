import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw, ImageFont

from gen_healthbar import process_healthbar_template, load_font
import xml.etree.ElementTree as ET

import os
import sys
import random
import argparse
import math
from coco_util import create_coco_dict, add_to_coco

from tqdm import tqdm

from itertools import product
from joblib import Parallel, delayed

from gen_minimap import ImageDrawer
import traceback

base_dir = os.path.dirname(__file__) 
parent_dir = os.path.abspath(os.path.join(base_dir, ".."))
sys.path.insert(0, parent_dir)

from classes import CLASSES, HEALTHBAR_CLASSES

# Cannon and siege minions have a lower chance of appearing
CANNON_CHANCE = 0.15
SIEGE_CHANCE = 0.05
NUM_CHAMPS = 170
NUM_MINIONS = 8

# NOTE: 
    # Riot calls these champions: "Renata", "MonkeyKing", "Nilah", "JarvanIV", 
    # We call them: "renataglasc", "wukong", "nulah" (thanks for the typo Rohan), "jarvan"

WEIRD_NAMES = {
    "renataglasc": "renata",
    "wukong": "monkeyking",
    "nulah": "nilah",
    "jarvan": "jarvaniv"
}

champtoi = {}
itochamp = CLASSES.copy()

for i, champion in itochamp.items():
    champtoi[champion] = i

for healthbar_class in HEALTHBAR_CLASSES:
    champtoi[healthbar_class] = len(champtoi)
    itochamp[len(itochamp)] = healthbar_class

minimap_drawer = ImageDrawer(resize=(256,256))

def parse_pascal_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(champtoi[name])  

    return boxes, labels

def parse_coco_json(img_name, json_path):
    import json
    with open(json_path, 'r') as f:
        data = json.load(f)

    boxes = []
    labels = []

    for image in data['images']:
        if image['file_name'] == img_name:
            image_id = image['id']
            break
    
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            bbox = annotation['bbox']
            boxes.append(bbox)
            labels.append(annotation['category_id'])

    return boxes, labels

def plot_image_with_boxes(img, boxes, labels, output_dir, split, count):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10), dpi = 400)
    plt.imshow(img_rgb)
    
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=1))
        plt.text(xmin, ymin -25, str(itochamp[label]), color='red', fontsize=7)

    plt.savefig(f'{output_dir}/{split}/map_{count:04d}.jpg', dpi=400)

def generate_map_frames(video_path, dir_to_save, start_second, end_second, n = 5):
    """
    Generate frames from a video file and save them as images.
    
    :param video_path: Path to the input video file.
    :return: List of paths to the saved image frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every nth frame
        if start_frame + frame_count < end_frame:
            if frame_count % n == 0:
                frame_path = f"frame_{start_frame + frame_count}.jpg"
                cv2.imwrite(os.path.join(dir_to_save, frame_path), frame)
                saved_frames.append(frame_path)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
  
def chroma_key_with_glow_preservation(
    img_bgr: np.ndarray,
    hsv_lower: int = 50,
    hsv_upper: int = 70,
    aggressive: bool = False,
    champ_boxes=None,
    pet_boxes=None,
) -> np.ndarray:
    if champ_boxes is None: champ_boxes = []
    if pet_boxes is None: pet_boxes = []

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    shape = h.shape

    region = np.zeros(shape, dtype=np.uint8)  # 0: normal, 1: champ, 2: pet

    for x_min, y_min, x_max, y_max in champ_boxes:
        region[y_min:y_max, x_min:x_max] = 1
    for x_min, y_min, x_max, y_max in pet_boxes:
        region[y_min:y_max, x_min:x_max] = 2

    normal = region == 0
    champ = region == 1
    pet = region == 2

    # Strong lime removal
    strong_mask = np.zeros(shape, dtype=bool)
    if aggressive:
        strong_mask |= (h >= hsv_lower) & (h <= hsv_upper) & (s >= 100) & (v >= 100) & normal
    else:
        strong_mask |= (h >= hsv_lower) & (h <= hsv_upper) & (s >= 150) & (v >= 150) & normal
    strong_mask |= (h >= hsv_lower) & (h <= hsv_upper) & (s >= 150) & (v >= 150) & champ
    strong_mask |= (h >= hsv_lower) & (h <= hsv_upper) & (s >= 170) & (v >= 170) & pet

    out = img_bgr.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2BGRA)

    out[strong_mask] = (255, 255, 255, 0)

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    max_rgb = np.maximum(np.maximum(r, g), b)
    min_rgb = np.minimum(np.minimum(r, g), b)
    grayness = 1.0 - (max_rgb - min_rgb) / 255.0
    # Now normalize brightness
    brightness = v / 255.0
    # Whiteness should be large only when both grayness≈1 and brightness≈1
    whiteness = grayness * brightness**2 * 5

    # Compute per-pixel proximity to lime (centered in hsv_lower and hsv_upper)
    hue_center = (hsv_lower + hsv_upper) / 2
    hue_range = ((hsv_upper - hsv_lower) / 2) * 2
    distance_from_lime = ((np.abs(h - hue_center)) / hue_range)

    non_white = np.any(img_bgr != [255, 255, 255], axis=-1)
    avg_proximity = np.clip(float(distance_from_lime[non_white].mean()) * (3/hue_center / hue_range), 0, 1)

    # compute per-pixel darkening factor in [0..1]
    scale = np.clip(distance_from_lime * (255 - v * 0.25) * (1 - avg_proximity) * (1.0 + whiteness**2), 0, 255) / 255.0
    # scale *= (1.0 + whiteness)

    # mask of “foreground” (any pixel that isn’t pure white)
    fg = np.any(out[..., :3] != 255, axis=-1)

    # darken only those foreground pixels, channel by channel
    for c in range(3):
        ch = out[..., c].astype(np.float32)
        ch[fg] *= scale[fg]
        out[..., c] = np.clip(ch, 0, 255).astype(np.uint8)

    # leave the alpha as you originally intended
    out[..., 3] = np.clip((distance_from_lime + (255 - v*1) / 255)**2 * (255 - v * 0.4) * (1 - avg_proximity), 0, 255).astype(np.uint8)

    return out
  
def IoU(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes. Assumes coco format
    
    :param box1: First bounding box in coco format [x_min, y_min, w, h]
    :param box2: Second bounding box in coco format [x_min, y_min, x_max, y_max].
    :return: IoU value.
    """

    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

def check_box_in_box(box1, box2):
    """
    Check if box1 is inside box2.
    
    :param box1: First bounding box in PASCAL VOC format [x_min, y_min, x_max, y_max].
    :param box2: Second bounding box [x_min, y_min, x_max, y_max].
    :return: True if box1 is inside box2, False otherwise.
    """
    # get the coordinates of box1 in pascal VOC format
    return (box1[0] >= box2[0] and
            box1[1] >= box2[1] and
            box1[2] <= box2[2] and
            box1[3] <= box2[3])

def chroma_crop_out_white(img, box):
    """
   Crop the image using the bounding box, chroma key and then remove white areas.
    
    :param img: Input image.
    :return: Cropped image.
    """
    # shrink box to just hit edges (until there's no white)
    x_min, y_min, x_max, y_max = box

    cropped_img = img[y_min:int(y_max), x_min:int(x_max)]
    cropped_img = chroma_key_with_glow_preservation(cropped_img)

    mask = cv2.inRange(img, (0, 0, 0, 0), (255, 255, 255, 1))
    mask = cv2.bitwise_not(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Update the bounding box
        x_min = x + x_min
        y_min = y + y_min
        x_max = x + x_min + w
        y_max = y + y_min + h
        box = [x_min, y_min, x_max, y_max]
        
        return cropped_img, box

    else:
        # No contours found, return the cropped image and box
        return cropped_img, box

def expand_cutout(cutout: np.ndarray,
                  champion_box: list[int],
                  mask_box:    list[int],
                  pet_boxes: list[list[int]] = [],
                  scale_factor: float = 4/3):
    """
    Resize the cutout by `scale_factor` (both width and height), and
    scale champion_box & mask_box coordinates accordingly.

    Args:
      cutout:        H×W×C image.
      champion_box:  [x_min, y_min, x_max, y_max] in cutout coords.
      mask_box:      [x_min, y_min, x_max, y_max] in cutout coords.
      scale_factor:  how much to enlarge (e.g. 1.33 = 133%).

    Returns:
      resized_cutout:   the cutout resized to (W*scale, H*scale).
      champ_box_adj:    scaled champion_box in the new image.
      mask_box_adj:     scaled mask_box in the new image.
    """
    H, W = cutout.shape[:2]

    # 1) compute new dimensions
    new_W = int(round(W * scale_factor))
    new_H = int(round(H * scale_factor))

    # 2) resize the image
    resized = cv2.resize(cutout, (new_W, new_H), interpolation=cv2.INTER_LINEAR)

    # 3) scale the box coordinates
    def scale_box(box):
        x0, y0, x1, y1 = box
        return [
            int(round(x0 * scale_factor)),
            int(round(y0 * scale_factor)),
            int(round(x1 * scale_factor)),
            int(round(y1 * scale_factor)),
        ]

    champ_box_adj = scale_box(champion_box)
    mask_box_adj  = scale_box(mask_box)

    if not pet_boxes:
        pet_boxes_adj = []

    pet_boxes_adj = [scale_box(box) for box in pet_boxes]

    return resized, champ_box_adj, mask_box_adj, pet_boxes_adj

def reduce_transparency(cutout, min_alpha_factor = 0.2, max_alpha_factor=0.45):
    """
    Reduce the transparency of the cutout image by scaling the alpha channel.
    
    :param cutout: Cutout image with an alpha channel (HWC, BGRA).
    :return: Cutout image with reduced transparency.
    """
    if cutout.shape[2] != 4:
        raise ValueError("Cutout must have an alpha channel (BGRA format).")

    alpha_factor = random.uniform(min_alpha_factor, max_alpha_factor)
    cutout[..., 3] = (cutout[..., 3] * alpha_factor).astype(np.uint8)

    return cutout

def hv_jitter(cutout, hue_shift_limit=40, val_shift_limit=0.15):
    """
    Apply HV jitter to non-white pixels in the champion region of a BGRA cutout image.

    Parameters:
        cutout (np.ndarray): Input image in BGRA format.
        hue_shift_limit (int): Max hue shift in degrees (-h, h).
        sat_shift_limit (float): Max saturation shift percentage (-s, s).
        val_shift_limit (float): Max brightness shift percentage (-v, v).

    Returns:
        np.ndarray: Augmented image in BGRA format.
    """

    # x_min, y_min, x_max, y_max = champion_box
    # champion_region = cutout[y_min:y_max, x_min:x_max].copy()

    # Separate BGR and alpha
    bgr = cutout[:, :, :3]
    alpha = cutout[:, :, 3]

    # Mask for non-white pixels (white = [255,255,255])
    non_white_mask = np.any(bgr < 250, axis=-1)  # more robust than strict == 255

    # Convert to HSV
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Jitter values with a Gaussian
    hue_shift = np.random.normal(0, scale = hue_shift_limit)
    val_shift = np.random.normal(0, scale = val_shift_limit)

    # Apply jitter only where non-white
    hsv[..., 0][non_white_mask] = (hsv[..., 0][non_white_mask] + hue_shift) % 180
    hsv[..., 2][non_white_mask] = np.clip(hsv[..., 2][non_white_mask] * (1 + val_shift), 0, 255)

    # Back to BGR
    bgr_jittered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Reattach alpha
    champion_jittered = np.dstack((bgr_jittered, alpha))

    # Replace region in cutout
    # cutout[y_min:y_max, x_min:x_max] = champion_jittered

    return champion_jittered
 
def generate_cutout(img_path, annotation_path, minion=False, make_transparent_prob = 0.05):
    """
    Generate a cutout image from the original image and its corresponding XML file.
    Returns cutout and a dictionary with bounding boxes for champion, pet, and ability in PASCAL VOC format.
    
    :param img_path: Path to the input image file.
    :param xml_path: Path to the corresponding XML file.
    :param minion: Boolean indicating if the image is a minion image (default is False).
    :return: Cutout image with green screen removed.
    """
    img = cv2.imread(img_path)
    boxes, labels = parse_coco_json(os.path.basename(img_path), annotation_path)
    champion_box = None
    champion_label = None
    mask_box = None
    pet_boxes = []
    ability_boxes = set()
    label = 0

    box_dict = {'Ability' : [], 'Pet': [], 'Mask': []}

    for box, label in zip(boxes, labels):
        # Convert box to PASCAL VOC format [x_min, y_min, x_max, y_max]
        box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        if label != champtoi['Ability'] and label != champtoi['Pet']:
            champion_box = box
            champion_label = label
            box_dict[itochamp[champion_label]] = [champion_box]
        elif label == champtoi['Ability']:
            ability_boxes.add(tuple(box))
        else:
            pet_boxes.append(box)

    # Check if champion/pet box is inside ability box or significant overlap
    for box in ability_boxes:
        if check_box_in_box(champion_box, box) or IoU(champion_box, box) > 0.9:
            mask_box = box
        else:
            box_dict['Ability'].append(box)
    
    if mask_box is not None:
        ability_boxes.remove(mask_box)
        cutout, mask_box = chroma_crop_out_white(img, mask_box)  

    else: # If no mask box was found, we need to create one based on the champion box
        x_min, y_min, x_max, y_max = champion_box

        #Expand champion box by 85 percent
        x_min = max(0, int(x_min - (x_max - x_min) * 0.5))
        y_min = max(0, int(y_min - (y_max - y_min) * 0.5))
        x_max = min(img.shape[1], int(x_max + (x_max - x_min) * 0.5))
        y_max = min(img.shape[0], int(y_max + (y_max - y_min) * 0.5))
        mask_box = [x_min, y_min, x_max, y_max]
        box_dict['Mask'].append(mask_box)
        cutout, mask_box = chroma_crop_out_white(img, mask_box) 

    # Adjust champion box to be relative to the mask box            
    x_min, y_min, x_max, y_max = mask_box
    
    x_min_champ, y_min_champ, x_max_champ, y_max_champ = champion_box
    x_min_champ = max(0, int(x_min_champ - x_min))
    y_min_champ = max(0, int(y_min_champ - y_min))
    x_max_champ = min(cutout.shape[1], int(x_max_champ - x_min))
    y_max_champ = min(cutout.shape[0], int(y_max_champ - y_min))
    champion_box = [x_min_champ, y_min_champ, x_max_champ, y_max_champ]
    
    # Provide a random chance to expand the cutout
    if random.random() < 0.5:
        if not minion:  # skip if it's a red or blue minion 
            scale_factor = random.choices(
                population=[6/5, 4/3, 3/2],
                weights=[0.70, 0.25, 0.05],  # your custom probabilities
                k=1  # number of items to choose
            )[0]

            cutout, champion_box, mask_box, pet_boxes = expand_cutout(cutout, champion_box, mask_box, pet_boxes = pet_boxes, scale_factor=scale_factor)

        else:
            cutout, champion_box, mask_box, pet_boxes = expand_cutout(cutout, champion_box, mask_box, scale_factor = 6/5)

    # Add the mask box to the box_dict
    box_dict["Mask"].append(list(mask_box))

    for pet_box in pet_boxes:
        if check_box_in_box(pet_box, mask_box) or IoU(pet_box, mask_box) > 0.9:
            x_min, y_min, x_max, y_max = mask_box

            x_min_pet, y_min_pet, x_max_pet, y_max_pet = pet_box
            
            x_min_pet = max(0, int(x_min_pet - x_min))
            y_min_pet = max(0, int(y_min_pet - y_min))
            x_max_pet = min(cutout.shape[1], int(x_max_pet - x_min))
            y_max_pet = min(cutout.shape[0], int(y_max_pet - y_min))
            box_dict['Pet'].append([x_min_pet, y_min_pet, x_max_pet, y_max_pet])
            
    # box_dict[itochamp[champion_label]].append(champion_box)
    box_dict[itochamp[champion_label]] = [champion_box]

    if random.random() < 0.5:
        if not minion:  # skip if it's a red or blue minion
            # Apply HSV jitter to the cutout
            cutout = hv_jitter(cutout, hue_shift_limit=25, val_shift_limit=0.15)
        else:
            cutout = hv_jitter(cutout, hue_shift_limit=0, val_shift_limit=0.01)

    # Occasionally make cutout transparent to simulate hiding in bushes
    if random.random() < make_transparent_prob:
        cutout = reduce_transparency(cutout)

    return cutout, box_dict 
  
def count_non_white_pixels(img):
    """
    Count the number of non-white pixels in an image.
    
    :param img: Input image (HWC, RGB).
    :return: Number of non-white pixels.
    """
    # Define white color in RGB
    white = np.array([255, 255, 255, 255], dtype=np.uint8)
    
    # Create a mask for non-white pixels
    non_white_mask = np.any(img != white, axis=-1)
    
    # Count non-white pixels
    return np.sum(non_white_mask)

def generate_cutouts(champion_img_paths, minion_img_paths, annotation_path, map_boxes = None):
    """
    Generate cutouts for multiple images and their corresponding bounding boxes.
    
    :param img_paths: List of paths to the input image files.
    :param annotation_path: Path to the corresponding annotations file.
    :return: List of cutout images and their bounding boxes.
    """
    cutouts = []
    box_dicts = []
    minion_cutouts = []
    minion_box_dicts = []
    
    for img_path in champion_img_paths:
        cutout, box_dict = generate_cutout(img_path, annotation_path)
        cutouts.append(cutout)
        box_dicts.append(box_dict)
    
    for img_path in minion_img_paths:
        cutout, box_dict = generate_cutout(img_path, annotation_path, minion=True)
        minion_cutouts.append(cutout)
        minion_box_dicts.append(box_dict)

    cutouts_sorted, box_dicts_sorted = zip(*sorted(zip(cutouts, box_dicts), key=lambda x: count_non_white_pixels(x[0]), reverse=True))
    
    return cutouts_sorted, box_dicts_sorted, minion_cutouts, minion_box_dicts

def is_far_enough(new_pos, existing_pos, min_dist = 150):
    return all(np.linalg.norm(np.array(new_pos) - np.array(p)) > min_dist for p in existing_pos)

def generate_grid_offsets(spacing=60, rows=3, cols=3):
    """
    Generate a grid of (dx, dy) offsets centered around (0,0).
    """
    x_offsets = [spacing * (i - cols // 2) for i in range(cols)]
    y_offsets = [spacing * (j - rows // 2) for j in range(rows)]
    offsets = list(product(x_offsets, y_offsets))
    random.shuffle(offsets)  # Add randomness
    return offsets

def box_intersects_map_boxes(box, map_boxes, x_tolerance = 50, y_tolerance = 100):
    """
    Return if the box strays too much into a building (x or y coordinate is inside map box by x or y tolerance px).
    :param box: Bounding box in PASCAL VOC format [x_min, y_min, x_max, y_max].
    :param map_boxes: List of map bounding boxes in PASCAL VOC format.
    :param tolerance: Tolerance in pixels for intersection.
    """
    x_min, y_min, x_max, y_max = box

    for m in map_boxes:
        mx0, my0, mx1, my1 = m

        # compute overlap in each axis
        x_overlap = max(0, min(x_max, mx1) - max(x_min, mx0))
        y_overlap = max(0, min(y_max, my1) - max(y_min, my0))

        # if overlap exceeds tolerance in either direction, it's “too far in”
        if x_overlap > x_tolerance or y_overlap > y_tolerance:
            return True

    return False

def place_cutout(map_img, cutout, box_dict, x, y, map_boxes):
    """
    Place the cutout image on the map image at the specified coordinates,
    only pasting non-white pixels (to preserve transparency-like behavior).
    If the original mask box/champion box hugs a side of the image, the cutout can only be placed along that side.
    Abilities can be placed anywhere (even cut off the map) but champions and pets must be placed within the map bounds.

    :param map_img: The map image (HWC, RGB) where the cutout will be placed.
    :param cutout: The cutout image to be placed (HWC, RGBA).
    :param box_dict: Dictionary containing bounding boxes for champion, pet, and ability.
    :param x: X-coordinate for placement (top-left).
    :param y: Y-coordinate for placement (top-left).
    :param map_boxes: List of map bounding boxes in PASCAL VOC format to check for intersection.
    :return: Updated map_img and adjusted box_dict.
    """
    mask_x_min, mask_y_min, mask_x_max, mask_y_max = box_dict['Mask'][0]
    champ_name = next(k for k in box_dict if k not in ('Mask', 'Ability', 'Pet'))
    champ_x_min, champ_y_min, champ_x_max, champ_y_max = box_dict[champ_name][0]

    # Constrain placement if cutout hugs map edges
    if mask_x_min == 1 or champ_x_min == 1:
        x = 0
    if mask_y_min == 1 or champ_y_min == 1:
        y = 0
    if mask_x_max == map_img.shape[1] or champ_x_max == map_img.shape[1]:
        x = map_img.shape[1] - cutout.shape[1]
    if mask_y_max == map_img.shape[0] or champ_y_max == map_img.shape[0]:
        y = map_img.shape[0] - cutout.shape[0]

    # Clamp to image bounds
    x = max(0, min(x, map_img.shape[1] - 1))
    y = max(0, min(y, map_img.shape[0] - 1))

    orig_h, orig_w = cutout.shape[:2]
    h = max(0, min(orig_h, map_img.shape[0] - y))
    w = max(0, min(orig_w, map_img.shape[1] - x))
    if h == 0 or w == 0:
        return map_img, box_dict

    # Update bounding boxes relative to the map
    new_pet_boxes = []
    for key, blist in list(box_dict.items()):
        if not blist:
            continue
        updated_boxes = []
        for box in blist:
            x_min, y_min, x_max, y_max = box
            if key not in ('Mask', 'Ability'):
                # shift champion/pet boxes
                x0 = max(0, int(x_min + x))
                y0 = max(0, int(y_min + y))
                x1 = int(x_max + x)
                y1 = int(y_max + y)
                # bounds and map intersection checks
                if x1 > map_img.shape[1] or y1 > map_img.shape[0]:
                    raise ValueError(f"Box {key} goes out of bounds: {box}")
                if box_intersects_map_boxes([x0, y0, x1, y1], map_boxes):
                    raise ValueError(f"Box {key} intersects with map boxes: {[x0, y0, x1, y1]}")

                if key == 'Pet':
                    new_pet_boxes.append([x0, y0, x1, y1])
                else:
                    # champion
                    updated_boxes = [[x0, y0, x1, y1]]
                    box_dict[key] = updated_boxes
            else:
                # mask or ability: adjust relative to placement
                x_diff = box[0] - x
                y_diff = box[1] - y
                x0 = max(0, int(box[0] - x_diff))
                y0 = max(0, int(box[1] - y_diff))
                x1 = min(map_img.shape[1], int(box[2] - x_diff))
                y1 = min(map_img.shape[0], int(box[3] - y_diff))
                if x1 > map_img.shape[1] or y1 > map_img.shape[0]:
                    raise ValueError(f"Box {key} goes out of bounds: {box}")
                box_dict[key] = [[x0, y0, x1, y1]]

        if key == 'Pet' and new_pet_boxes:
            box_dict['Pet'] = new_pet_boxes

    # Composite cutout onto map
    roi = map_img[y:y+h, x:x+w].astype(np.float32)
    fg = cutout[:h, :w].astype(np.float32)
    alpha = fg[..., 3:4] / 255.0
    comp = fg[..., :4] * alpha + roi * (1 - alpha)
    map_img[y:y+h, x:x+w] = comp.astype(np.uint8)

    return map_img, box_dict

def place_cutouts_on_map(map_img, cutouts, box_dicts, map_boxes, num_sprites, minions = False, max_clusters = 5):
    """
    Place multiple cutouts on the map image. 
    
    :param map_img: The map image (HWC, RGB) where the cutouts will be placed.
    :param cutouts: List of cutout images to be placed (HWC, RGB).
    :param box_dicts: List of dictionaries containing bounding boxes for each cutout.
    :param map_boxes: List of map bounding boxes to check for intersection.
    :param num_sprites: Number of sprites to place on the map.
    :return: Map image with all cutouts placed.
    """        

    box_dict_all = {}
    H, W = map_img.shape[:2]
    # Poisson-disk parameters
    # Reduce the minimum separation between cutouts the more sprites there are
    MIN_SEP = 0.38 - num_sprites / 150  # minimum separation between cutouts (in px)
    SCALE   = 30     # stddev of cluster spread (px)
    MAX_TRIES = 800

    if minions:
        # 1) pick a few cluster anchors
        n_clusters = random.randint(2, max_clusters)
        anchors = [
            (random.randint(150, W - 150), random.randint(150, H - 150))
            for _ in range(n_clusters)
        ]

        placed_centers = []  # list of (x, y, radius)
        box_dict_all = {}

        for cutout, box_dict in zip(cutouts, box_dicts):
            h, w = cutout.shape[:2]
            radius = max(w, h) * 0.5

            attempts = 0
            while attempts < MAX_TRIES:
                # pick an anchor and jitter around it
                x0, y0 = random.choice(anchors)
                dx, dy = np.random.normal(scale=SCALE, size=2)
                x_cand = int(x0 + dx)
                y_cand = int(y0 + dy)

                # 2a) keep inside bounds
                if not (0 <= x_cand <= W - w and 0 <= y_cand <= H - h):
                    attempts += 1
                    continue

                # 2b) cheap circle‐based overlap check
                collision = False
                for x_o, y_o, r_o in placed_centers:
                    if np.hypot(x_cand - x_o, y_cand - y_o) < (radius + r_o) * MIN_SEP:
                        collision = True
                        break

                if collision:
                    attempts += 1
                    continue

                # 2c) attempt to paste—if it raises ValueError, count it and retry
                try:
                    map_img, placed = place_cutout(map_img, cutout, box_dict, x_cand, y_cand, map_boxes)
                    # success!
                    placed_centers.append((x_cand, y_cand, radius))
                    for k, v in placed.items():
                        if v:
                            box_dict_all.setdefault(k, []).extend(v)
                    break

                except ValueError:
                    attempts += 1
                    continue
                
            else:
                print(f"Failed to place one minion after {MAX_TRIES} attempts, skipping it.")
                continue
    else:                    
        positions = []  # to track where cutouts have been placed
        for cutout, box_dict in zip(cutouts, box_dicts):
            attempts = 0
            cutout_h, cutout_w = cutout.shape[:2]
            max_x = map_img.shape[1] - cutout_w
            max_y = map_img.shape[0] - cutout_h

            while attempts < MAX_TRIES:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                if not is_far_enough((x, y), positions):
                    attempts += 1
                    continue 

                try:
                    map_img, box_dict = place_cutout(map_img, cutout, box_dict, x, y, map_boxes)
                    positions.append((x, y))  # save successful placement
                    for key in box_dict:
                        if len(box_dict[key]):
                            if key not in box_dict_all:
                                box_dict_all[key] = []
                            box_dict_all[key].extend(box_dict[key])
                    break

                except ValueError:
                    attempts += 1
                    continue
            else:
                print(f"Failed to place champion cutout after {MAX_TRIES} attempts, skipping this cutout.")
                continue
    
    return map_img, box_dict_all
  
def get_boxes_from_box_dict(box_dict):
    """
    Convert the box_dict to a list of boxes for each category.
    
    :param box_dict: Dictionary containing bounding boxes for champion, pet, and ability.
    :return: List of boxes for each category.
    """
    boxes, labels = [], []
    for key, blist in box_dict.items():
        # we only want real objects (champions / minions), never Mask or Ability
        if key in ('Mask', 'Ability'):
            continue

        for b in blist:
            # b should now be [x0,y0,x1,y1]
            if not (isinstance(b, (list, tuple)) and len(b) == 4):
                # skip anything that isn't a proper box
                continue
            boxes.append(b)
            labels.append(champtoi[key])
    return boxes, labels

def add_healthbars(map_img, box_dict, mode, color, font, image_paths=None):
    """
    Add health bars above each object (champion, minion) on the map image.

    :param map_img: The map image (HWC, RGB).
    :param box_dict: Dictionary containing bounding boxes for champion, pet, and ability.
    :param mode: "champs" or "minions".
    :param image_paths: Required for mode="minions"; used to match team color.
    :return: Updated map image.
    """
    box_dict_healthbars = {}
    for healthbar_class in HEALTHBAR_CLASSES:
        box_dict_healthbars[healthbar_class] = []

    healthbar_dir = "../cropped_healthbars"
    healthbar_paths = [os.path.join(healthbar_dir, f) for f in os.listdir(healthbar_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    red_healthbar_dir = "../synthetic_healthbars/red_minions"
    blue_healthbar_dir = "../synthetic_healthbars/blue_minions"
    red_paths = [os.path.join(red_healthbar_dir, f) for f in os.listdir(red_healthbar_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    blue_paths = [os.path.join(blue_healthbar_dir, f) for f in os.listdir(blue_healthbar_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if mode == "champs":
        healthbar_type = ""
        for key, boxes in box_dict.items():
            if key in ['Mask', 'Ability']:
                continue
            if key == 'Pet':
                if color: 
                    hb_paths = blue_paths
                    healthbar_type = "BlueMinionHealthbar"
                else:
                    hb_paths = red_paths
                    healthbar_type = "RedMinionHealthbar"
                for pet_box in boxes:
                    map_img, bounding_box = _draw_healthbar(map_img, pet_box, hb_paths, color=color, font=None)
                    box_dict_healthbars[healthbar_type].append(bounding_box)
            else:
                healthbar_type = "BlueChampionHealthbar" if color else "RedChampionHealthbar"
                for box in boxes:
                    map_img, bounding_box = _draw_healthbar(map_img, box, healthbar_paths, color=color, font = font)
                    box_dict_healthbars[healthbar_type].append(bounding_box)

    elif mode == "minions":
        assert image_paths is not None, "image_paths is required for minion mode."

        for minion, boxes in box_dict.items():
            healthbar_type = ""

            if 'red' in minion.lower(): 
                hb_paths = red_paths
                healthbar_type = "RedMinionHealthbar"
            else:
                hb_paths = blue_paths
                healthbar_type = "BlueMinionHealthbar"

            if minion in ['Mask', 'Ability']:
                continue
            if minion == 'Pet':
                for pet_group in boxes:
                    for pet_box in pet_group:
                        map_img, bounding_box = _draw_healthbar(map_img, pet_box, hb_paths, font = None)
                        box_dict_healthbars[healthbar_type].append(bounding_box)
            else:
                for box in boxes:
                    map_img, bounding_box = _draw_healthbar(map_img, box, hb_paths, font = None)
                    box_dict_healthbars[healthbar_type].append(bounding_box)

    return map_img, box_dict_healthbars

def _draw_healthbar(map_img, box, healthbar_paths, damage_prob = 0.10, color=False, font = None):
    """
    Draw a healthbar above a single bounding box.

    :param map_img: Map image to draw on.
    :param box: Bounding box (x_min, y_min, x_max, y_max).
    :param healthbar_paths: List of available healthbar image paths.
    :return: Updated map image.
    """
    if len(box) != 4:
        return map_img

    x_min, y_min, x_max, y_max = box
    cutout_center_x = (x_min + x_max) // 2

    if font is not None:
        num_healthbar_indices = len(healthbar_paths)
        # Sample a gaussian distribution with mean 11, std 5
        sample_index = np.random.normal(loc=11, scale=5, size= 1)[0]
        sample_index = int(np.clip(sample_index, 0, num_healthbar_indices - 1))
        healthbar_path = sorted(healthbar_paths)[sample_index]

        # prob_blue = random.random()
        to_blue = color
        # if prob_blue > 0.5:
        #     to_blue = True
        healthbar = cv2.imread(healthbar_path, cv2.IMREAD_UNCHANGED)
        healthbar = process_healthbar_template(healthbar, damage_prob, to_blue, font = font)
    else:
        # Randomly select a healthbar image (minions)
        healthbar_path = random.choice(healthbar_paths)
        healthbar = cv2.imread(healthbar_path, cv2.IMREAD_UNCHANGED)
    
    if healthbar is None:
        return map_img

    healthbar_h, healthbar_w = healthbar.shape[:2]
    hb_x1 = int(cutout_center_x - healthbar_w // 2)
    hb_y1 = int(y_min - healthbar_h)

    # Clamp to image bounds
    hb_x1_clamped = int(max(0, hb_x1))
    hb_y1_clamped = int(max(0, hb_y1))
    hb_x2_clamped = int(min(hb_x1 + healthbar_w, map_img.shape[1]))
    hb_y2_clamped = int(min(hb_y1 + healthbar_h, map_img.shape[0]))

    x_offset = hb_x1_clamped - hb_x1
    y_offset = hb_y1_clamped - hb_y1

    cropped_h = hb_y2_clamped - hb_y1_clamped
    cropped_w = hb_x2_clamped - hb_x1_clamped

    if cropped_h <= 0 or cropped_w <= 0:
        return map_img

    roi = map_img[hb_y1_clamped:hb_y2_clamped, hb_x1_clamped:hb_x2_clamped]
    # also get bounding box of the healthbar
    bounding_box = [hb_x1_clamped, hb_y1_clamped, hb_x2_clamped, hb_y2_clamped]
    healthbar_cropped = healthbar[y_offset:y_offset + cropped_h, x_offset:x_offset + cropped_w]

    if healthbar_cropped.shape[2] == 4:  # RGBA
        roi = healthbar_cropped
    else:
        roi[..., :3] = healthbar_cropped
        roi[..., 3] = np.ones_like(healthbar_cropped[...,0])

    map_img[hb_y1_clamped:hb_y2_clamped, hb_x1_clamped:hb_x2_clamped] = roi

    return map_img, bounding_box

def remove_background(img: Image.Image, threshold: int = 60) -> Image.Image:
    img = img.convert("RGBA")
    data = np.array(img)
    # collect *all* edge pixels
    border = np.vstack([
      data[0,    :, :3],
      data[-1,   :, :3],
      data[:,  0, :3],
      data[:, -1, :3],
    ])
    key_color = Counter(map(tuple, border)).most_common(1)[0][0]
    key = np.array(key_color, dtype=np.float32)

    rgb  = data[..., :3].astype(np.float32)
    dist = np.linalg.norm(rgb - key, axis=2)
    mask = dist < threshold

    data[mask, 3]   = 0   # transparent
    data[~mask, 3]  = 255 # opaque
    return Image.fromarray(data, mode="RGBA")

def sample_repeat_count(p=0.5, max_repeats=10):
    """
    Return k with P[k >= n] = p^n, capped at max_repeats.
    """
    cnt = 0
    while random.random() < p**(cnt+1) and cnt < max_repeats:
        cnt += 1
    return cnt

def load_fx_images(folder):
    instances = []
    for fn in os.listdir(folder):
        if not fn.lower().endswith((".png","jpg","jpeg")):
            continue

        fx = Image.open(os.path.join(folder, fn))
        keyed = remove_background(fx, threshold=60)

        # sample how many copies to make
        repeats = sample_repeat_count(p=0.25, max_repeats=10)
        for _ in range(repeats):
            instances.append(keyed.copy())

    random.shuffle(instances)
    return instances


def distort_fx(fx: Image.Image, max_size = 100) -> Image.Image:
    # 1) work in RGBA from the get-go
    
    w, h = fx.size
    if max(w, h) > max_size:
        # Calculate the scaling factor
        scale_factor = max_size / max(w, h)
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        fx = fx.resize((new_w, new_h), Image.Resampling.LANCZOS)

    fx = fx.convert("RGBA")
    # 2) do your flips on the entire RGBA
    if random.random() < 0.5:
        fx = ImageOps.mirror(fx)
    if random.random() < 0.3:
        fx = ImageOps.flip(fx)

    # 3) affine‐warp the whole thing (this warps R,G,B *and* A)
    w, h = fx.size
    coeffs = (
        1, random.uniform(-0.3, 0.3), random.uniform(-0.2, 0.2) * w,
        random.uniform(-0.3, 0.3), 1,                  random.uniform(-0.2, 0.2) * h
    )
    fx = fx.transform((w, h),
                      Image.AFFINE,
                      coeffs,
                      resample=Image.Resampling.BICUBIC)

    # 4) now color-jitter *only* the RGB channels
    r, g, b, a = fx.split()
    rgb = Image.merge("RGB", (r, g, b))
    for Enh in (ImageEnhance.Color,
                ImageEnhance.Brightness,
                ImageEnhance.Contrast,
                ImageEnhance.Sharpness):
        rgb = Enh(rgb).enhance(random.uniform(0.7, 1.3))
    r2, g2, b2 = rgb.split()

    # 5) re-assemble with the **already-warped** alpha
    return Image.merge("RGBA", (r2, g2, b2, a))

from PIL import Image, ImageOps, ImageEnhance
from collections import Counter

def weave_and_compose(base_map: Image.Image,
                      cutout_layers: list[tuple[Image.Image,tuple[int,int]]],
                      fx_instances: list[Image.Image],
                      fx_prob: float=0.6) -> Image.Image:
    canvas = base_map.copy()
    # FX underneath
    for fx in fx_instances:
        if random.random() < fx_prob/2:
            fx2 = distort_fx(fx)  # RGBA

            # --- new: scale its alpha by a random factor in [0.2, 0.7] ---
            r, g, b, a = fx2.split()
            alpha_factor = random.uniform(0.2, 0.7)
            a = a.point(lambda p: int(p * alpha_factor))
            fx2.putalpha(a)
            # ----------------------------------------------------------

            x = random.randint(0, canvas.width - fx2.width)
            y = random.randint(0, canvas.height - fx2.height)
            canvas.paste(fx2, (x, y), fx2)

    # Champions & abilities (unchanged)
    for cut, (x, y) in cutout_layers:
        canvas.paste(cut, (x, y), cut)
    # FX on top
    for fx in fx_instances:
        if random.random() < fx_prob/2:
            fx2 = distort_fx(fx)

            # scale alpha again for the top-layer FX
            r, g, b, a = fx2.split()
            alpha_factor = random.uniform(0.2, 0.7)
            a = a.point(lambda p: int(p * alpha_factor))
            fx2.putalpha(a)

            x = random.randint(0, canvas.width - fx2.width)
            y = random.randint(0, canvas.height - fx2.height)
            canvas.paste(fx2, (x, y), fx2)

    return canvas

def convert_coco_to_pascal_voc(boxes):
    """
    Convert COCO format boxes to PASCAL VOC format.
    
    :param boxes: List of boxes in COCO format [x_min, y_min, width, height].
    :return: List of boxes in PASCAL VOC format [x_min, y_min, x_max, y_max].
    """
    pascal_boxes = []
    for box in boxes:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        pascal_boxes.append([x_min, y_min, x_max, y_max])
    return pascal_boxes

def compute_box_center(bbox: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

def fog_of_war(
    img: np.ndarray,
    box_dict_champ: dict[str, list[list[int]]],
    box_dict_minion: dict[str, list[list[int]]],
    pov_center: tuple[int, int] = None,
    r_vis_base: int = 550,
    r_fade_base: int = 1000,
    fog_threshold: float = 0.05,
) -> np.ndarray:
    
    """
    Modifies box_dict_champ and box_dict_minion in-place by removing fog-covered boxes.
    Returns the fogged image.
    """
    H, W = img.shape[:2]
    Y, X = np.ogrid[:H, :W]
    img_center = np.array([W / 2, H / 2])
    max_dist = np.linalg.norm([W / 2, H / 2])

    # Collect all units
    all_units = []
    for cls, boxes in box_dict_champ.items():
        for box in boxes:
            all_units.append({"bbox": box, "team": "champ", "class": cls})
    for cls, boxes in box_dict_minion.items():
        for box in boxes:
            team = "blue" if "blue" in cls else "red"
            all_units.append({"bbox": box, "team": team, "class": cls})

    # Determine POV center
    if pov_center is None:
        img_center = np.array([W / 2, H / 2])
        min_dist = float("inf")
        for u in all_units:
            if u["team"] != "champ":
                continue
            c = compute_box_center(u["bbox"])
            dist = np.linalg.norm(c - img_center)
            if dist < min_dist:
                min_dist = dist
                pov_center = c

    # Determine vision sources
    vision_sources = []
    for u in all_units:
        team = u["team"]
        cls = u["class"]
        if team == "champ":
            vision_sources.append(compute_box_center(u["bbox"]))
        elif "blue" in cls and pov_center[1] > H // 2:
            vision_sources.append(compute_box_center(u["bbox"]))
        elif "red" in cls and pov_center[1] <= H // 2:
            vision_sources.append(compute_box_center(u["bbox"]))

    # Build fog alpha mask
    alpha = np.zeros((H, W), dtype=np.float32)

    for cx, cy in vision_sources:
        pos = np.array([cx, cy])
        dist_to_center = np.linalg.norm(pos - img_center)
        scale_factor = 1 + 0.75 * (dist_to_center / max_dist)  # up to 1.75× larger

        r_vis = r_vis_base * scale_factor
        r_fade = r_fade_base * scale_factor

        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        local = np.ones_like(dist, dtype=np.float32)
        local[dist >= r_fade] = 0.0
        fade = (dist > r_vis) & (dist < r_fade)

        # Smoother falloff
        fade_dist = (dist[fade] - r_vis) / (r_fade - r_vis)
        local[fade] = np.exp(-4 * fade_dist**2)  # Gaussian falloff (gentler curve)

        alpha = np.maximum(alpha, local)

    # Apply fog
    # fog_color = np.array([0,0,0,], dtype=np.float32)  # BGR format
    # fogged_img = (img.astype(np.float32) * (alpha**1.2)[..., None] + fog_color * (1 - (alpha**1.2)[..., None])).astype(np.uint8)
    # Clamp minimum visibility to retain terrain features
    # Clamp minimum visibility to retain terrain features
    alpha = np.clip(alpha, 0.5, 1.0)  # never fully obscured

    # Define fog color (lighter than black to simulate in-game fog)
    fog_color = np.array([32,14,4,255], dtype=np.float32)  # BGR – adjust for tone

    # Blend between original image and fog color
    fogged_img = (
        img.astype(np.float32) * (alpha**1.2)[..., None] + fog_color * (1.0 - (alpha**1.2)[..., None])
    ).astype(np.uint8)

    # Filter boxes in-place
    for box_dict in [box_dict_champ, box_dict_minion]:
        for cls in list(box_dict.keys()):
            filtered = []
            for bbox in box_dict[cls]:
                x1, y1, x2, y2 = map(int, bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W - 1, x2)
                y2 = min(H - 1, y2)
                if x1 >= x2 or y1 >= y2:
                    continue
                patch_alpha = alpha[y1:y2+1, x1:x2+1]
                visible_ratio = np.mean(patch_alpha > 0.05)
                if visible_ratio >= (1.0 - fog_threshold):
                    filtered.append(bbox)
            box_dict[cls] = filtered

    return fogged_img

def overlay_random_fx(base_img: np.ndarray, fx_folder: str, num_samples: int = 5) -> np.ndarray:
    """
    Sample `num_samples` random images from `fx_folder` (with alpha channels)
    and alpha-blend each onto `base_img` at a random location.

    :param base_img: H×W×4 BGRA image to draw onto.
    :param fx_folder: path to a directory of .png/.jpg fx images (with transparency).
    :param num_samples: how many fx sprites to overlay.
    :return: the modified base_img (in-place).
    """
    # gather all image files
    fx_paths = [
        os.path.join(fx_folder, fn)
        for fn in os.listdir(fx_folder)
        if fn.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not fx_paths:
        return base_img

    H, W = base_img.shape[:2]

    for _ in range(num_samples):
        # pick one at random
        p = random.choice(fx_paths)
        fx = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if fx is None or fx.shape[2] != 4:
            continue

        h, w = fx.shape[:2]
        if h > H or w > W:
            # skip if too big
            continue

        # random top-left corner
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)

        # split fg/bg and alpha
        fg = fx[..., :3].astype(np.float32)
        alpha = fx[..., 3:4].astype(np.float32) / 255.0
        bg = base_img[y:y+h, x:x+w, :3].astype(np.float32)

        # blend and write back
        comp = (alpha * fg + (1 - alpha) * bg).astype(np.uint8)
        base_img[y:y+h, x:x+w, :3] = comp

    return base_img

def generate_single_image(
    i,
    split,
    champ_imgs,
    champ_names,
    minion_imgs_dict,
    map_imgs,
    fx_folder,
    hud_folder,
    icons_folder,
    font_path,
    output_dir,
    annotation_path):
    try:
        # Champions have mean 1 std 3 clamp to 1 to 10
        num_champions = int(np.clip(random.gauss(1, 3), 1, 10))
        # num_champions = 1
        # Minions have mean 6 std 7 clamp to 0 to 20
        num_minions = int(np.clip(random.gauss(8, 5), 0, 20))

        champs_chosen = 0
        attempts = 0
        test_imgs = []
        champion_names = set()

        # Ensure we don't choose more champions than available
        num_champions = min(num_champions, len(champ_names))

        while champs_chosen < num_champions:
            test_img = random.choice(champ_imgs)
            base_img_name = os.path.basename(test_img).split('/')[-1]
            champ_name = base_img_name[:base_img_name.find('_')]
            if champ_name not in champion_names:
                test_imgs.append(test_img)
                champion_names.add(champ_name)
                champs_chosen += 1
            attempts += 1

        minion_imgs = []
        for _ in range(num_minions):
            rand = random.random()
            if rand <= SIEGE_CHANCE:
                minion_type = 'siege'
            elif rand <= SIEGE_CHANCE + CANNON_CHANCE:
                minion_type = 'cannon'
            else:
                minion_type = random.choice(['melee', 'caster'])
            minion_imgs.append(random.choice(minion_imgs_dict[minion_type]))
        # Load map
        map_box_dict = {'RedNexus': [], 'BlueNexus': [], 'BlueTower': [], 'RedTower': [],
                        'BlueInhibitor': [], 'RedInhibitor': []}
        map_img_path = random.choice(map_imgs)
        map_img = cv2.imread(map_img_path)

        # Populate map_box_dict with the map image's bounding boxes (if there are any)
        map_img_name = os.path.basename(map_img_path)
        map_boxes, map_labels = parse_coco_json(map_img_name, annotation_path)
        map_boxes = convert_coco_to_pascal_voc(map_boxes)

        for box, label in zip(map_boxes, map_labels):
            if itochamp[label] != 'Pit': # Ignore Pit box
                map_box_dict[itochamp[label]].append(box)

        # Place champions & minions
        champ_cutouts, champ_box_dicts, minion_cutouts, minion_box_dicts = generate_cutouts(test_imgs, minion_imgs, annotation_path, map_boxes)

        # Ensure box_dicts are lists
        champ_box_dicts = list(champ_box_dicts)
        minion_box_dicts = list(minion_box_dicts)

        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2BGRA)
        # plt.imshow(map_img)
        # plt.show()

        map_img, box_dict_champ = place_cutouts_on_map(
            map_img, champ_cutouts, champ_box_dicts, map_boxes, num_champions
        )

        map_img, box_dict_minion = place_cutouts_on_map(
            map_img, minion_cutouts, minion_box_dicts,
            map_boxes, num_minions, minions=True
        )

        # ─── weave in FX ──────────────────────────────────────────────────────────
        # 1) convert to PIL RGBA
        # pil_map = Image.fromarray(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))\
        #             .convert("RGBA")
        rgba = cv2.cvtColor(map_img, cv2.COLOR_BGRA2RGBA)
        pil_map = Image.fromarray(rgba, mode="RGBA")

        # 2) build cutout layers [(PIL_cutout, (x,y)), …]
        cutout_layers = []
        for cut_bgr, bd in zip(champ_cutouts, champ_box_dicts):
            x0, y0 = bd['Mask'][0][:2]

            # 1) convert BGR→RGB
            rgb = cv2.cvtColor(cut_bgr, cv2.COLOR_BGR2RGB)

            # 2) build a binary mask (1 where there's real pixels, 0 where white)
            white = np.array([255, 255, 255], dtype=cut_bgr.dtype)
            mask = np.any(cut_bgr[:, :, :3] != white, axis=-1).astype(np.uint8)

            # → inject random translucency: 75% of the time pick a low-alpha, else high-alpha
            t = (
                np.clip(random.gauss(0.25, 0.2), 0, 1)
                if random.random() > 0.2
                else np.clip(random.gauss(0.75, 0.2), 0, 1)
            )
            alpha = (mask * t * 255).astype(np.uint8)

            # 3) stack into an H×W×4 RGBA image
            if alpha.ndim == 2:
                pass  # fine
            elif alpha.ndim == 1:
                alpha = alpha[:, np.newaxis]
            else:
                raise ValueError(f"Unexpected alpha shape: {alpha.shape}")

            if alpha.shape != rgb.shape[:2]:
                alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

            # 5) Stack RGBA
            rgba = np.dstack([rgb, alpha])

            # 4) convert to PIL with alpha channel
            pil_cut = Image.fromarray(rgba, mode="RGBA")

            cutout_layers.append((pil_cut, (x0, y0)))

        # 3) load & distort FX
        fx_instances = load_fx_images(fx_folder)

        # 4) weave under & over
        pil_composed = weave_and_compose(
            pil_map,
            cutout_layers,
            fx_instances,
            fx_prob=0.4
        )
        # 5) back to OpenCV BGR
        map_img = cv2.cvtColor(np.array(pil_composed), cv2.COLOR_RGBA2BGRA)
        pil_map = Image.fromarray(cv2.cvtColor(map_img, cv2.COLOR_BGRA2RGBA)).convert("RGBA")
        
        # Convert back to BGR OpenCV format
        map_img = cv2.cvtColor(np.array(pil_composed), cv2.COLOR_RGBA2BGRA)

        # Add health bars
        font = load_font(font_path, size = 10)
        map_img, champ_health_box_dict = add_healthbars(
            map_img,
            box_dict_champ,
            color=random.choice([True, False]),
            font = font,
            mode="champs"
        )

        map_img, minion_health_box_dict = add_healthbars(
            map_img,
            box_dict_minion,
            mode="minions",
            color=random.choice([True, False]),
            font=font,
            image_paths=minion_imgs
        )

        if random.random() > 0.5:
            map_img = overlay_random_fx(map_img, hud_folder, num_samples=1)

        # Clean up & plot
        box_dict_champ.pop('Mask', None)
        box_dict_minion.pop('Ability', None)
        box_dict_minion.pop('Mask', None)

        if random.random() > 0.5:
            map_img = fog_of_war(map_img, box_dict_champ, box_dict_minion)

        boxes_c, labels_c = get_boxes_from_box_dict(box_dict_champ)
        boxes_m, labels_m = get_boxes_from_box_dict(box_dict_minion)
        health_boxes_c, health_labels_c = get_boxes_from_box_dict(champ_health_box_dict)
        health_boxes_m, health_labels_m = get_boxes_from_box_dict(minion_health_box_dict)
        boxes_map, labels_map = get_boxes_from_box_dict(map_box_dict)

        
        boxes = boxes_c + boxes_m + boxes_map + health_boxes_c + health_boxes_m
        labels = labels_c + labels_m + labels_map + health_labels_c + health_labels_m

        blacklist = [
            ((952, 1080), (456, 1268)),   # (y0, y1), (x0, x1)
            ((760, 1080), (1664, 1920)),
            # add more if you like...
        ]

        filtered_boxes  = []
        filtered_labels = []

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            box_area = (xmax - xmin) * (ymax - ymin)

            # assume no intersection until proven otherwise
            intersects = False

            for (by0, by1), (bx0, bx1) in blacklist:
                # compute intersection coordinates
                inter_x0 = max(xmin, bx0)
                inter_x1 = min(xmax, bx1)
                inter_y0 = max(ymin, by0)
                inter_y1 = min(ymax, by1)

                # check that they actually overlap
                if inter_x1 > inter_x0 and inter_y1 > inter_y0:
                    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
                    # if more than 50% of the box is covered → drop
                    if inter_area > 0.2 * box_area:
                        intersects = True
                        break

            if not intersects:
                filtered_boxes.append(box)
                filtered_labels.append(label)

        boxes, labels = filtered_boxes, filtered_labels

        # minimap = generate_random_minimap()
        # plt.imshow(minimap)
        # plt.show()
        if random.random() > 0.5:
            minimap = cv2.cvtColor(minimap_drawer.get_minimap_np(), cv2.COLOR_BGR2RGB)
            minimap_uint8 = (minimap * 255).astype(np.uint8)
            map_img[-256:, -256:, :3] = minimap_uint8
            map_img[-256:, -256:,  3] = 255

        if random.random() > 0.5:
            folder = os.path.join(icons_folder, "champions")
            imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]

            for j in range(4):
                icon = cv2.cvtColor(cv2.resize(plt.imread(os.path.join(folder, random.choice(imgs))), dsize=(64,64)), cv2.COLOR_BGRA2RGBA)
                icon_uint8 = (icon * 255).astype(np.uint8)
                map_img[-320:-256, -256+64*j - 1:-256+64*j+64 - 1, :] = icon_uint8

        img_name = f'map_{i:04d}.jpg'
        img_path = os.path.join(output_dir, split, img_name)
        cv2.imwrite(img_path, map_img)

        # plot_image_with_boxes(
        #     map_img, health_boxes_c + health_boxes_m, health_labels_c + health_labels_m, output_dir=output_dir, split=split, count = i
        # )

        return {
            "filename": img_name,
            "width": map_img.shape[1],
            "height": map_img.shape[0],
            "boxes": boxes,
            "labels": labels
        }
    
    except Exception as e:
        print(f"[Warning] iteration {i} failed: {e!r}")
        traceback.print_exc()
        return None

def generate_synthetic_ds_parallel(img_dir: str, 
    split: str, 
    fx_folder: str, 
    hud_folder: str,
    icons_folder: str,
    annotation_path: str,
    font_path: str,
    output_dir: str, 
    champs_to_exclude: set = None, 
    rf_categories: list[dict] = None,
    images_per_unit: int = 500):

    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    rf_categories = annotations['categories']

    coco_dict = create_coco_dict(rf_categories)

    # Load image paths
    img_dir = os.path.join(img_dir, split)
    all_files = os.listdir(img_dir)
    all_imgs = [img for img in all_files if img.endswith('.jpg')]
    map_imgs = [os.path.join(img_dir, img) for img in all_imgs if img.startswith('map_frame_')]
    # Separate by role
    champ_imgs = []
    champ_names = set()
    for img in all_imgs:
        if not img.startswith(('red-','blue-', 'map_frame_')):
            if '-' in img:
                champ_name = img[:img.find('-')]
            else:
                champ_name = img[:img.find('_')]
            if champ_name in WEIRD_NAMES:
                champ_name = WEIRD_NAMES[champ_name]
            if champ_name not in champs_to_exclude:                
                champ_imgs.append(os.path.join(img_dir, img))
                champ_names.add(champ_name)

    minion_imgs = [os.path.join(img_dir, img)
                for img in all_imgs if img.startswith(('red-','blue-'))]
    
    minion_imgs_dict = {}
    for img in minion_imgs:
        if 'caster' in img:
            minion_imgs_dict.setdefault('caster', []).append(img)
        elif 'melee' in img:
            minion_imgs_dict.setdefault('melee', []).append(img)
        elif 'cannon' in img:
            minion_imgs_dict.setdefault('cannon', []).append(img)
        elif 'siege' in img:
            minion_imgs_dict.setdefault('siege', []).append(img)

    results = Parallel(n_jobs=-1, backend='loky', verbose=10)(
        delayed(generate_single_image)(
            i,
            split,
            champ_imgs,
            champ_names,
            minion_imgs_dict,
            map_imgs,
            fx_folder,
            hud_folder,
            icons_folder,
            font_path,
            output_dir,
            annotation_path,
        )
        for i in range(images_per_unit * (NUM_CHAMPS + NUM_MINIONS - len(champs_to_exclude)))
    )

    for result in results:
        if result is None:
            continue

        add_to_coco(
            coco_dict, rf_categories, 
            result["boxes"], result["labels"],
            result["width"], result["height"], 
            result["filename"]
        )

    with open(os.path.join(output_dir, split, 'annotations.json'), 'w') as f:
        json.dump(coco_dict, f, indent=4)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for League of Legends.")
    parser.add_argument("--dataset_dir", type=str, default="../greenscreends/")
    parser.add_argument("--split", type=str, choices=["train", "valid"], default="train",
                        help="Split to generate (train or val).")
    parser.add_argument("--fx_folder", type=str, default="../fx",
                        help="Folder containing FX images.")
    parser.add_argument("--hud_folder", type=str, default="../sample_hud",
                        help="Folder containing HUD images.")
    parser.add_argument("--icons_folder", type=str, default="../league_icons",
                    help="Folder containing league icons.")
    parser.add_argument("--font_path", type=str, default="../BeaufortForLoL-OTF/BeaufortforLOL-Bold.otf",
                        help="Path to the font file for health bars.")
    parser.add_argument("--output_dir", type=str, default="../synthetic_dataset/",
                        help="Directory to save the generated images.")
    parser.add_argument("--champs_to_exclude", type=str, default=None, 
                        help="Comma-separated list of champion names to exclude from the dataset.")
    parser.add_argument("--images_per_unit", type=int, default=500,
                        help="Number of images to generate per unit (champion or minion).")
    args = parser.parse_args()
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        print("Creating output directory:", args.output_dir)
        os.makedirs(args.output_dir)

    if not os.path.exists(os.path.join(args.output_dir, args.split)):
        print("Creating output directory for split:", args.split)
        os.makedirs(os.path.join(args.output_dir, args.split))

    # Convert comma-separated string to set
    if args.champs_to_exclude:
        excluded_champs = set()
        with open(args.champs_to_exclude, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].strip(): 
                    excluded_champs.add(row[0].strip().lower())

    # Generate the synthetic dataset
    generate_synthetic_ds_parallel(
        img_dir=args.dataset_dir,
        split=args.split,
        fx_folder=args.fx_folder,
        hud_folder=args.hud_folder,
        icons_folder=args.icons_folder,
        annotation_path = os.path.join(args.dataset_dir, args.split, "_annotations.coco.json"), 
        font_path=args.font_path,
        output_dir=args.output_dir,
        champs_to_exclude=excluded_champs, 
        images_per_unit=args.images_per_unit
    )
    
if __name__ == "__main__":
    main()

