import cv2
import numpy as np
import os
import random
import argparse
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont

def load_font(font_path, size):
    base, ext = os.path.splitext(font_path)
    ext = ext.lower()
    if ext == '.ttf':
        return ImageFont.truetype(font_path, size=size)
    elif ext == '.otf':
        ttf_path = base + '.ttf'
        if not os.path.exists(ttf_path):
            TTFont(font_path).save(ttf_path)
        return ImageFont.truetype(ttf_path, size=size)
    else:
        return ImageFont.truetype(font_path, size=size)

def shift_red_to_blue(img: np.ndarray) -> np.ndarray:
    """Convert any red‐hue pixels to blue, preserving brightness."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask1 = cv2.inRange(hsv, np.array([0,30,30]), np.array([10,255,255]))
    mask2 = cv2.inRange(hsv, np.array([170,30,30]), np.array([180,255,255]))
    mask_red = (mask1 | mask2) > 0
    h[mask_red] = 102
    v[mask_red] = v[mask_red] * 1.2
    s_vals = s[mask_red]; s[mask_red] = np.maximum(s_vals, 150)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

def process_healthbar_template(img: np.ndarray,
                               damage_prob: float,
                               to_blue: bool,
                               font: ImageFont.FreeTypeFont) -> np.ndarray:
    # optionally shift reds to blue
    if to_blue:
        img = shift_red_to_blue(img)

    # crop to 27×132
    img = img[:27, :]
    if img.shape[1] != 132:
        d = img.shape[1] - 132
        img = img[:, d//2:d//2+132]
    orig = img.copy()

    # remove lime → black
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    maskg = cv2.inRange(hsv, np.array([40,80,80]), np.array([85,255,255]))
    img[maskg>0] = [0,0,0]

    # patch fill [8:20, 8:20]
    px1, px2, py1, py2 = 8,20,8,20
    top    = img[py1-1, px1:px2]
    bottom = img[py2,   px1:px2]
    left   = img[py1:py2, px1-1]
    right  = img[py1:py2, px2]
    neigh  = np.concatenate([top, bottom, left, right], axis=0)
    img[py1:py2, px1:px2] = np.mean(neigh, axis=0).astype(np.uint8)

    # simulate health segments
    x1, x2 = 25,130; y1, y2 = 5,17
    total_w = x2 - x1

    # decide on recent damage
    if random.random() < damage_prob:
        damage_w = random.randint(1, total_w)
    else:
        damage_w = 0
    health_w = random.randint(0, total_w - damage_w)
    h_end = x1 + health_w
    d_end = h_end + damage_w

    # draw sections
    img[y1:y2, x1:h_end]    = orig[y1:y2, x1:h_end]      # healthy (blue or red)
    img[y1:y2, h_end:d_end] = [0, 0, 255]                # recent damage red
    img[y1:y2, d_end:x2]    = [0, 0, 0]                  # missing black

    # draw number
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    num = str(random.randint(1,18))
    tb = draw.textbbox((0,0), num, font=font)
    tw, th = tb[2]-tb[0], tb[3]-tb[1]
    tx = px1 + ((px2-px1)-tw)//2 - tb[0] - 2
    ty = py1 + ((py2-py1)-th)//2 - tb[1]
    draw.text((tx, ty), num, font=font, fill=(255,255,255))

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def generate_dataset(input_dir, font_path, font_size, damage_prob, num_samples, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    font = load_font(font_path, font_size)
    templates = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    half = num_samples // 2
    for i in range(num_samples):
        fname = random.choice(templates)
        img = cv2.imread(os.path.join(input_dir, fname))

        # first half blue, second half red
        to_blue = (i < half)
        hb = process_healthbar_template(img, damage_prob, to_blue, font)

        color_tag = 'blue' if to_blue else 'red'
        out_fname = f"healthbar_{color_tag}_{i:05d}.png"
        cv2.imwrite(os.path.join(output_dir, out_fname), hb)

    print(f"Saved {num_samples} healthbars → {output_dir} (half blue, half red)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",   default="cropped_healthbars")
    parser.add_argument("--font_path",   required=True)
    parser.add_argument("--font_size",   type=int,   default=10)
    parser.add_argument("--damage_prob", type=float, default=0.15,
                        help="Chance recent damage appears")
    parser.add_argument("--num_samples", type=int,   default=5000)
    parser.add_argument("--output_dir",  default="generated_healthbars")
    args = parser.parse_args()

    generate_dataset(args.input_dir,
                     args.font_path,
                     args.font_size,
                     args.damage_prob,
                     args.num_samples,
                     args.output_dir)
