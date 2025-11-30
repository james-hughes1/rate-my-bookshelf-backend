import json
import os

import cv2
import numpy as np


# -------------------------------------------------------------
# Draw polygons on the image (for visual feedback)
# -------------------------------------------------------------
def draw_segments(image, polygons, color=(0, 255, 0), thickness=3):
    """
    Draw polygon segments on image.
    polygons: list of lists of [x,y] vertices
    """
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image.copy()

    for poly in polygons:
        poly_arr = np.array(poly, dtype=np.int32)
        cv2.polylines(img, [poly_arr], isClosed=True, color=color, thickness=thickness)

    return img


# -------------------------------------------------------------
# Save ground truth as polygons
# -------------------------------------------------------------
def save_ground_truth(image, polygons, output_dir="images/eval", index=None):
    """
    image: numpy array HxWx3 (RGB or BGR - converted correctly below)
    polygons: list of polygons, each polygon is a list of [x,y] points
    """

    os.makedirs(output_dir, exist_ok=True)

    # Auto-indexing
    if index is None:
        existing = [f for f in os.listdir(output_dir) if f.startswith("test_img_")]
        index = len(existing)

    img_filename = f"test_img_{index}.png"
    json_filename = "ground_truth.json"

    # Save image (convert RGB to BGR if necessary)
    img_path = os.path.join(output_dir, img_filename)
    if image.shape[2] == 3:
        cv2.imwrite(img_path, image[:, :, ::-1])  # assume RGB → convert to BGR
    else:
        cv2.imwrite(img_path, image)

    # Save polygons to JSON
    gt_path = os.path.join(output_dir, json_filename)
    if os.path.exists(gt_path):
        with open(gt_path, "r") as f:
            gt = json.load(f)
    else:
        gt = {}

    gt[img_filename] = polygons  # <--- polygons stored directly

    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2)

    return img_path, gt_path
