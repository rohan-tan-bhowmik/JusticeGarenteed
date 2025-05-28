import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw, ImageFont

from test_healthbar import process_healthbar_template, load_font
import xml.etree.ElementTree as ET

import os
import random
import argparse
import math

from tqdm import tqdm

from itertools import product

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
itochamp = {}
annotation_path = '../greenscreends/train/_annotations.coco.json'
with open(annotation_path, 'r') as f:
    data = json.load(f)

for i, category in enumerate(data['categories']):
    champtoi[category['name']] = i
    itochamp[i] = category['name']

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

def plot_image_with_boxes(img, boxes, labels):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10), dpi = 100)
    plt.imshow(img_rgb)
    
    for box, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = box
        plt.gca().add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=1))
        plt.text(xmin, ymin -25, str(itochamp[label]), color='red', fontsize=7)

    plt.axis('off')
    plt.show()

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
    hsv_lower: int = 40,
    hsv_upper: int = 80,
    target_hue: int = 90,  # unused
    aggressive: bool = False,
    champ_boxes=None,
    pet_boxes=None,
    ability_boxes=None  # unused for now
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
    out[strong_mask] = (255, 255, 255)

    # Hue proximity to lime
    proximity = 1.0 - (np.abs(h - (hsv_lower + hsv_upper) / 2) / ((hsv_upper - hsv_lower) / 2))
    proximity = np.clip(proximity, 0, 1)

    lime_band = (h >= hsv_lower) & (h <= hsv_upper)
    darken_mask = lime_band & ~strong_mask

    # Base darkening
    strength = proximity * 0.8

    # Pets: always mild
    strength[pet] *= 0.3

    # Champions: dynamic darkening (how green they are on avg)
    for x_min, y_min, x_max, y_max in champ_boxes:
        box_mask = np.zeros_like(region, dtype=bool)
        box_mask[y_min:y_max, x_min:x_max] = True

        # Compute average proximity inside box
        avg_prox = np.mean(proximity[box_mask])

        # More lime (avg_prox near 1) ➝ reduce darkening
        box_strength = (1.0 - avg_prox)  # if fully lime, near 0
        strength[box_mask] *= box_strength  # scale whole box by this

    # Apply darkening
    v[darken_mask] *= (1 - strength[darken_mask])

    hsv[..., 2] = v
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    darkened = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    out[darken_mask] = darkened[darken_mask]

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

    mask = cv2.inRange(img, (0, 0, 0), (255, 255, 255))
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

def generate_cutout(img_path, annotation_path):
    """
    Generate a cutout image from the original image and its corresponding XML file.
    Returns cutout and a dictionary with bounding boxes for champion, pet, and ability in PASCAL VOC format.
    
    :param img_path: Path to the input image file.
    :param xml_path: Path to the corresponding XML file.
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
    print(img_path)
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

    # print(itochamp[champion_label] in box_dict)
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
            
    box_dict[itochamp[champion_label]].append(champion_box)
    # print(champion_box)
    return cutout, box_dict 

def place_cutout(map_img, cutout, box_dict, x, y):
    """
    Place the cutout image on the map image at the specified coordinates,
    only pasting non-white pixels (to preserve transparency-like behavior).
    If the original mask box/champion box hugs a side of the image, the cutout can only be placed along that side.
    Abilities can be placed anywhere (even cut off the map) but champions and pets must be placed within the map bounds

    :param map_img: The map image (HWC, RGB) where the cutout will be placed.
    :param cutout: The cutout image to be placed (HWC, RGB).
    :param box_dict: Dictionary containing bounding boxes for champion, pet, and ability.
    :param x: X-coordinate for placement (top-left).
    :param y: Y-coordinate for placement (top-left).
    :return: Map image with the cutout placed.
    """
    mask_x_min, mask_y_min, mask_x_max, mask_y_max = box_dict['Mask'][0]
    champ_name = ""
    for key in box_dict:
        if key != 'Mask' and key != 'Ability' and key != 'Pet':
            champ_name = key
            break

    champ_x_min, champ_y_min, champ_x_max, champ_y_max = box_dict[champ_name][0]
    # Check if cutout hugs side of image. If so, we can only place it along that side.
    if mask_x_min == 1 or champ_x_min == 1:
        x = 0
    if mask_y_min == 1 or champ_y_min == 1:
        y = 0
    if mask_x_max == map_img.shape[1] or champ_x_max == map_img.shape[1]:
        x = map_img.shape[1] - cutout.shape[1]
        # print(f"Cutout hugs right side of image, placing at x={x}")
    if mask_y_max == map_img.shape[0] or champ_y_max == map_img.shape[0]:
        y = map_img.shape[0] - cutout.shape[0]
        # print(f"Cutout hugs bottom side of image, placing at y={y}")

    x = max(0, min(x, map_img.shape[1] - 1))
    y = max(0, min(y, map_img.shape[0] - 1))

    # If placing the cutout makes the ability box go out of bounds, you can do so but up until the champion box hugging
    # the border of the image
    orig_h, orig_w = cutout.shape[:2]
    h = max(0, min(orig_h, map_img.shape[0] - y))
    w = max(0, min(orig_w, map_img.shape[1] - x))

    if h == 0 or w == 0:
        # print(f"Cutout at ({x}, {y}) with size ({h}, {w}) is out of bounds, skipping placement.")
        return map_img, box_dict

    # print(f"Placing cutout at ({x}, {y}) with size ({h}, {w})")
    
    # Update the bounding boxes in box_dict to be relative to the map image
    new_pet_boxes = []
    for key in box_dict:
        if len(box_dict[key]):
            for box in box_dict[key]:
                x_min, y_min, x_max, y_max = box
                if key != 'Mask' and key != 'Ability':
                    # Just shift the champion/pet box
                    x_min = max(0, int(x_min + x))
                    y_min = max(0, int(y_min + y))
                    x_max = int(x_max + x)
                    y_max = int(y_max + y)
                    if x_max > map_img.shape[1] or y_max > map_img.shape[0]:
                        raise ValueError(f"Box {key} goes out of bounds: {box}")
                    if key == 'Pet':
                        new_pet_boxes.append([x_min, y_min, x_max, y_max])
                    else:
                        box_dict[key] = [[ x_min, y_min, x_max, y_max ]]
                    
                else: 
                    x_diff = box[0] - x
                    y_diff = box[1] - y
                    x_min = max(0, int(x_min - x_diff))
                    y_min = max(0, int(y_min - y_diff))
                    x_max = min(map_img.shape[1], int(x_max - x_diff))
                    y_max = min(map_img.shape[0], int(y_max - y_diff))
                    box_dict[key] = [[ x_min, y_min, x_max, y_max ]]
    if len(new_pet_boxes):
        box_dict['Pet'] = new_pet_boxes

    # Crop region of interest (ROI) from the map image
    cutout = cutout[:h, :w]
    roi = map_img[y:y + h, x:x + w, :]

    # Create a mask for non-white pixels
    mask = ~(np.all(cutout == [255, 255, 255], axis=-1))

    # Broadcast mask to RGB
    mask_rgb = np.stack([mask]*3, axis=-1)    
    # Blend cutout into ROI using the mask
    roi[mask_rgb] = cutout[mask_rgb]

    map_img[y:y + h, x:x + w] = roi
    return map_img, box_dict
  
def count_non_white_pixels(img):
    """
    Count the number of non-white pixels in an image.
    
    :param img: Input image (HWC, RGB).
    :return: Number of non-white pixels.
    """
    # Define white color in RGB
    white = np.array([255, 255, 255], dtype=np.uint8)
    
    # Create a mask for non-white pixels
    non_white_mask = np.any(img != white, axis=-1)
    
    # Count non-white pixels
    return np.sum(non_white_mask)

def generate_cutouts(champion_img_paths, minion_img_paths, annotation_path):
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
        cutout, box_dict = generate_cutout(img_path, annotation_path)
        minion_cutouts.append(cutout)
        minion_box_dicts.append(box_dict)

    cutouts_sorted, box_dicts_sorted = zip(*sorted(zip(cutouts, box_dicts), key=lambda x: count_non_white_pixels(x[0]), reverse=True))
    
    return cutouts_sorted, box_dicts_sorted, minion_cutouts, minion_box_dicts

def is_far_enough(new_pos, existing_pos, min_dist = 100):
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

def place_cutouts_on_map(map_img, cutouts, box_dicts, num_sprites, minions = False, max_clusters = 5):
    """
    Place multiple cutouts on the map image. 
    
    :param map_img: The map image (HWC, RGB) where the cutouts will be placed.
    :param cutouts: List of cutout images to be placed (HWC, RGB).
    :param box_dicts: List of dictionaries containing bounding boxes for each cutout.
    :param num_sprites: Number of sprites to place on the map.
    :return: Map image with all cutouts placed.
    """        

    box_dict_all = {}
    if minions:
        # pick up to n random positions on the map and cluster the minions around them.
        clusters = random.randint(2, max_clusters)
        positions = [(random.randint(0, map_img.shape[1]), random.randint(0, map_img.shape[0])) for _ in range(clusters)]

        for cutout, box_dict in zip(cutouts, box_dicts):
            # Place the cutout around the center position
            # the larger the number of minions, the more spread out they are with linearly scaling offsets  
            attempts = 0
            while attempts < 200:    
                scale_factor = 1 + (num_sprites / 20)  
                offset = int(100 * scale_factor)  # increase offset based on number of sprites
                x_center, y_center = positions[random.randint(0, len(positions) - 1)]

                x_offset = random.randint(-offset, offset)
                y_offset = random.randint(-offset, offset)
                x = x_center + x_offset
                y = y_center + y_offset
                try: 
                    map_img, box_dict = place_cutout(map_img, cutout, box_dict, x, y)
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
                print(f"Failed to place minion cutout after 200 attempts, skipping this cutout.")
                continue
    else:                    
        positions = []  # to track where cutouts have been placed
        for cutout, box_dict in zip(cutouts, box_dicts):
            attempts = 0
            cutout_h, cutout_w = cutout.shape[:2]
            max_x = map_img.shape[1] - cutout_w
            max_y = map_img.shape[0] - cutout_h

            while attempts < 200:
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                
                if not is_far_enough((x, y), positions):
                    attempts += 1
                    continue

                try:
                    map_img, box_dict = place_cutout(map_img, cutout, box_dict, x, y)
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
                print(f"Failed to place champion cutout after 200 attempts, skipping this cutout.")
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

def add_healthbars(map_img, box_dict, mode, font, image_paths=None):
    """
    Add health bars above each object (champion, minion) on the map image.

    :param map_img: The map image (HWC, RGB).
    :param box_dict: Dictionary containing bounding boxes for champion, pet, and ability.
    :param mode: "champs" or "minions".
    :param image_paths: Required for mode="minions"; used to match team color.
    :return: Updated map image.
    """
    if mode == "champs":
        healthbar_dir = "../cropped_healthbars"
        healthbar_paths = [os.path.join(healthbar_dir, f) for f in os.listdir(healthbar_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for key, boxes in box_dict.items():
            if key in ['Mask', 'Ability']:
                continue
            if key == 'Pet':
                for pet_box in boxes:
                    map_img = _draw_healthbar(map_img, pet_box, healthbar_paths, font=font)
            else:
                for box in boxes:
                    map_img = _draw_healthbar(map_img, box, healthbar_paths, font = font)

    elif mode == "minions":
        assert image_paths is not None, "image_paths is required for minion mode."

        red_healthbar_dir = "../synthetic_healthbars/red_minions"
        blue_healthbar_dir = "../synthetic_healthbars/blue_minions"
        red_paths = [os.path.join(red_healthbar_dir, f) for f in os.listdir(red_healthbar_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        blue_paths = [os.path.join(blue_healthbar_dir, f) for f in os.listdir(blue_healthbar_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for minion, boxes in box_dict.items():
            if 'red' in minion.lower(): 
                hb_paths = red_paths
            else:
                hb_paths = blue_paths
            if minion in ['Mask', 'Ability']:
                continue
            if minion == 'Pet':
                for pet_group in boxes:
                    for pet_box in pet_group:
                        map_img = _draw_healthbar(map_img, pet_box, hb_paths, font = None)
            else:
                for box in boxes:
                    map_img = _draw_healthbar(map_img, box, hb_paths, font = None)

    return map_img

def _draw_healthbar(map_img, box, healthbar_paths, damage_prob = 0.10, font = None):
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
        sample_index = np.random.normal(loc=11, scale=5, size= 1)
        sample_index = int(np.clip(sample_index, 0, num_healthbar_indices - 1))
        healthbar_path = sorted(healthbar_paths)[sample_index]

        prob_blue = random.random()
        to_blue = False
        if prob_blue > 0.5:
            to_blue = True
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
    healthbar_cropped = healthbar[y_offset:y_offset + cropped_h, x_offset:x_offset + cropped_w]

    if healthbar_cropped.shape[2] == 4:  # RGBA
        alpha = healthbar_cropped[:, :, 3] / 255.0
        for c in range(3):
            roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * healthbar_cropped[:, :, c]
    else:
        roi[:, :, :] = healthbar_cropped[:, :, :3]

    map_img[hb_y1_clamped:hb_y2_clamped, hb_x1_clamped:hb_x2_clamped] = roi

    return map_img

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
    fog_color = np.array([32,14,4], dtype=np.float32)  # BGR – adjust for tone

    # Blend between original image and fog color
    fogged_img = (
        img.astype(np.float32) * (alpha**1.2)[..., None] + 
        fog_color * (1.0 - (alpha**1.2)[..., None])
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

def generate_synthetic_ds(img_dir: str, 
    split: str, 
    fx_folder: str, 
    font_path: str,
    output_dir: str, 
    champs_to_exclude: set = None, 
    images_per_unit: int = 500):
    """
    :param dataset_dir: Directory containing the dataset.
    :param split: split to generate 'train' and 'val'
    :param fx_folder: Directory containing the FX images.
    :param font_path: Path to the font file for health bars.
    :param output_dir: Directory to save the generated images.
    :param champs_to_exclude: set of champion names to exclude from the dataset.
    :param images_per_unit: Number of images to generate per unit (champion or minion).
    """

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

    for i in tqdm(range(images_per_unit * (NUM_CHAMPS + NUM_MINIONS - len(champs_to_exclude))), desc=f"Generating {split} dataset"):
        # Champions have mean 1 std 3 clamp to 1 to 10
        num_champions = int(np.clip(random.gauss(1, 3), 1, 10))
        # num_champions = 1
        # Minions have mean 5 std 7 clamp to 0 to 20
        num_minions = int(np.clip(random.gauss(5, 7), 0, 20))

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

        # Generate cutouts
        champ_cutouts, champ_box_dicts, minion_cutouts, minion_box_dicts = generate_cutouts(test_imgs, minion_imgs, annotation_path)

        # Ensure box_dicts are lists
        champ_box_dicts = list(champ_box_dicts)
        minion_box_dicts = list(minion_box_dicts)

        # Load map
        map_box_dict = {'RedNexus': [], 'BlueNexus': [], 'BlueTower': [], 'RedTower': [],
                        'BlueInhibitor': [], 'RedInhibitor': [], 'Pit': []}
        map_img_path = random.choice(map_imgs)
        map_img = cv2.imread(map_img_path)

        # Populate map_box_dict with the map image's bounding boxes (if there are any)
        map_img_name = os.path.basename(map_img_path)
        map_boxes, map_labels = parse_coco_json(map_img_name, annotation_path)
        map_boxes = convert_coco_to_pascal_voc(map_boxes)

        for box, label in zip(map_boxes, map_labels):
            map_box_dict[itochamp[label]].append(box)

        # Place champions & minions
        map_img, box_dict_champ = place_cutouts_on_map(
            map_img, champ_cutouts, champ_box_dicts,
            num_champions
        )
        map_img, box_dict_minion = place_cutouts_on_map(
            map_img, minion_cutouts, minion_box_dicts,
            num_minions, minions=True
        )

        # ─── weave in FX ──────────────────────────────────────────────────────────
        # 1) convert to PIL RGBA
        pil_map = Image.fromarray(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))\
                    .convert("RGBA")

        # 2) build cutout layers [(PIL_cutout, (x,y)), …]
        cutout_layers = []
        for cut_bgr, bd in zip(champ_cutouts, champ_box_dicts):
            x0, y0 = bd['Mask'][0][:2]

            # 1) convert BGR→RGB
            rgb = cv2.cvtColor(cut_bgr, cv2.COLOR_BGR2RGB)

            # 2) build a binary mask (1 where there's real pixels, 0 where white)
            mask = np.any(cut_bgr != [255, 255, 255], axis=-1).astype(np.uint8)

            # → inject random translucency: 75% of the time pick a low-alpha, else high-alpha
            t = (
                np.clip(random.gauss(0.25, 0.2), 0, 1)
                if random.random() > 0.2
                else np.clip(random.gauss(0.75, 0.2), 0, 1)
            )
            alpha = (mask * t * 255).astype(np.uint8)

            # 3) stack into an H×W×4 RGBA image
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
        map_img = cv2.cvtColor(np.array(pil_composed), cv2.COLOR_RGBA2BGR)
        pil_map = Image.fromarray(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        
        # Convert back to BGR OpenCV format
        map_img = cv2.cvtColor(np.array(pil_composed), cv2.COLOR_RGBA2BGR)

        # Add health bars
        font = load_font(font_path, size = 10)
        map_img = add_healthbars(
            map_img,
            box_dict_champ,
            font = font,
            mode="champs"
        )

        map_img = add_healthbars(
            map_img,
            box_dict_minion,
            mode="minions",
            font = font,
            image_paths=minion_imgs
        )

        # Clean up & plot
        box_dict_champ.pop('Mask', None)
        box_dict_minion.pop('Ability', None)
        box_dict_minion.pop('Mask', None)

        map_img = fog_of_war(map_img, box_dict_champ, box_dict_minion)

        boxes_c, labels_c = get_boxes_from_box_dict(box_dict_champ)
        boxes_m, labels_m = get_boxes_from_box_dict(box_dict_minion)
        boxes_map, labels_map = get_boxes_from_box_dict(map_box_dict)

        boxes = boxes_c + boxes_m + boxes_map
        labels = labels_c + labels_m + labels_map
        cv2.imwrite(f'{output_dir}/{split}/map_{i:04d}.jpg', map_img)
        # plot_image_with_boxes(map_img, boxes, labels)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for League of Legends.")
    parser.add_argument("--dataset_dir", type=str, default="../greenscreends/")
    parser.add_argument("--split", type=str, choices=["train", "val"], default="train",
                        help="Split to generate (train or val).")
    parser.add_argument("--fx_folder", type=str, default=FX_FOLDER,
                        help="Folder containing FX images.")
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
        os.makedirs(args.output_dir)
        if args.split:
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
    generate_synthetic_ds(
        img_dir=args.dataset_dir,
        split=args.split,
        fx_folder=args.fx_folder,
        font_path=args.font_path,
        output_dir=args.output_dir,
        champs_to_exclude=excluded_champs, 
        images_per_unit=args.images_per_unit
    )
    
if __name__ == "__main__":
    main()

