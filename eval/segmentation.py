import numpy as np
import json
import os
import cv2
from math import log2
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Variation of Information (unchanged)
# -------------------------------------------------------------
def variation_of_information(labels_true, labels_pred):
    """
    Computes VI = H(G|P) + H(P|G)
    Lower is better, 0 = perfect.
    """

    labels_true = np.ravel(labels_true)
    labels_pred = np.ravel(labels_pred)

    n = len(labels_true)
    true_ids = np.unique(labels_true)
    pred_ids = np.unique(labels_pred)

    # contingency table
    contingency = np.zeros((len(true_ids), len(pred_ids)), dtype=float)
    for i, t in enumerate(true_ids):
        for j, p in enumerate(pred_ids):
            contingency[i, j] = np.sum((labels_true == t) & (labels_pred == p))

    contingency /= n

    P_true = contingency.sum(axis=1)
    P_pred = contingency.sum(axis=0)

    H_true = -np.sum(P_true[P_true > 0] * np.log2(P_true[P_true > 0]))
    H_pred = -np.sum(P_pred[P_pred > 0] * np.log2(P_pred[P_pred > 0]))

    MI = mutual_information(contingency)

    H_true_given_pred = H_true - MI
    H_pred_given_true = H_pred - MI

    return H_true_given_pred + H_pred_given_true


def mutual_information(contingency):
    P = contingency
    P_true = P.sum(axis=1)
    P_pred = P.sum(axis=0)

    MI = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 0:
                MI += P[i, j] * log2(P[i, j] / (P_true[i] * P_pred[j]))
    return MI


# -------------------------------------------------------------
# NEW: accept binary masks directly, no bounding boxes
# -------------------------------------------------------------
def load_binary_mask(mask_path):
    """
    Load a binary mask image from disk.
    Assumes mask is stored as a PNG or similar.
    Non-zero pixels become 1.
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    return (m > 0).astype(np.int32)   # ensure 0/1 mask


def polygons_to_label_mask(img_h, img_w, polygons):
    """
    Converts multiple polygons into a single integer label mask.
    Each polygon gets a unique label starting from 1.
    Background = 0
    """
    mask = np.zeros((img_h, img_w), dtype=np.int32)
    for i, poly in enumerate(polygons, start=1):
        pts = np.array(poly, dtype=np.int32)
        cv2.fillPoly(mask, [pts], i)
    return mask


def binary_masks_to_label_mask(binary_masks):
    """
    binary_masks: numpy array of shape (N, H, W)
                  or list of (H,W) binary masks
    Returns a single label mask of shape (H,W) where:
        0 = background
        1..N = object labels
    """
    if isinstance(binary_masks, list):
        binary_masks = np.array(binary_masks)

    N, H, W = binary_masks.shape
    label_mask = np.zeros((H, W), dtype=np.int32)

    for i in range(N):
        label_mask[binary_masks[i] > 0] = i + 1

    return label_mask


# -------------------------------------------------------------
# NEW: evaluation using binary masks
# -------------------------------------------------------------
def evaluate_segmentation(eval_dir, segmentation_function):
    """
    eval_dir contains:
        test_img_*.png
        ground_truth.json   (maps image → list of polygons)

    segmentation_function(img) -> binary mask (0/1)
    """

    # Load polygon metadata
    gt_path = os.path.join(eval_dir, "ground_truth.json")
    with open(gt_path, "r") as f:
        gt_data = json.load(f)

    results = []

    for filename, gt_polygons in gt_data.items():

        img_path = os.path.join(eval_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"WARNING: could not load {filename}")
            continue

        h, w = img.shape[:2]

        # ---- Predicted mask ----
        pred_binary_masks = segmentation_function(img)  # shape (N,H,W) or list of (H,W)
        mask_pred = binary_masks_to_label_mask(pred_binary_masks)

        # ---- GT label mask ----
        mask_gt = polygons_to_label_mask(h, w, gt_polygons)

        # now shapes match
        assert mask_pred.shape == mask_gt.shape

        # ---- VI score ----
        vi = variation_of_information(mask_gt, mask_pred)

        print(f"{filename}: VI = {vi:.4f}")
        results.append((filename, vi))

    return results


def visualize_mean_colours(img, binary_masks):
    """
    img: original image (H,W,3) in RGB or BGR
    binary_masks: array of shape (N,H,W) or list of (H,W) masks
    Returns a visualisation image (H,W,3)
    """
    if isinstance(binary_masks, list):
        binary_masks = np.array(binary_masks)  # shape (N,H,W)

    H, W = binary_masks.shape[1], binary_masks.shape[2]
    vis = np.zeros_like(img, dtype=np.uint8)

    # Flatten image for faster indexing
    img_flat = img.reshape(-1, 3)

    for i in range(binary_masks.shape[0]):
        mask = binary_masks[i].astype(bool)
        if np.any(mask):
            # Compute mean color
            mean_color = img[mask].mean(axis=0)
            vis[mask] = mean_color.astype(np.uint8)

    # Background remains black (0,0,0)
    return vis


def compare_segmentations(eval_dir, segmentation_function1, segmentation_function2, image_index):
    """
    eval_dir: directory with images and ground_truth.json
    segmentation_functionX: function(img) -> list of binary masks (N,H,W)
    image_index: integer index of image (test_img_{index}.png)
    """
    img_filename = f"test_img_{image_index}.png"
    img_path = f"{eval_dir}/{img_filename}"
    gt_path = f"{eval_dir}/ground_truth.json"

    # Load image
    img = cv2.imread(img_path)[:,:,::-1]  # BGR -> RGB
    h, w = img.shape[:2]

    # Load GT polygons
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    gt_polygons = gt_data[img_filename]

    # Convert GT polygons -> binary masks
    mask_gt_label = polygons_to_label_mask(h, w, gt_polygons)

    # Run segmentation functions
    pred1_masks = segmentation_function1(img)  # list of binary masks
    pred2_masks = segmentation_function2(img)

    mask_pred1_label = binary_masks_to_label_mask(pred1_masks)
    mask_pred2_label = binary_masks_to_label_mask(pred2_masks)

    # Compute VI scores
    vi1 = variation_of_information(mask_gt_label, mask_pred1_label)
    vi2 = variation_of_information(mask_gt_label, mask_pred2_label)

    print(f"Segmentation 1 VI: {vi1:.4f}")
    print(f"Segmentation 2 VI: {vi2:.4f}")

    # Visualize using mean colours
    vis1 = visualize_mean_colours(img, pred1_masks)
    vis2 = visualize_mean_colours(img, pred2_masks)

    # Display side by side
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(vis1)
    plt.title(f"Segmentation 1 (VI={vi1:.2f})")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(vis2)
    plt.title(f"Segmentation 2 (VI={vi2:.2f})")
    plt.axis("off")
    plt.show()
