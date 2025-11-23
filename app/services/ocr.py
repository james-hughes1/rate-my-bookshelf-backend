import re
import os
from rapidocr_onnxruntime import RapidOCR
import numpy as np
from PIL import Image
import cv2

engine = RapidOCR()

def ocr_from_array(image_array):
    """
    image_array: numpy array of shape (H, W, 3) or (H, W)
    """
    result, elapse = engine(image_array)
    
    if result is None:
        return "No text detected"
    
    # Extract text from results
    # result format: list of [bbox, text, confidence]
    boxes = [line[0] for line in result]
    texts = [line[1] for line in result]
    confidences = [line[2] for line in result]
    
    return boxes, texts, confidences


def mask_to_bbox(mask):
    """Convert binary mask to bounding box (x1, y1, x2, y2)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def assign_text_to_segments(img, masks, ocr_data):
    """
    Assign OCR text to mask segments.

    Args:
        img: Original image (H,W,3)
        masks: list of binary masks (H,W)
        ocr_data: [boxes, texts, confidences]
                  boxes: list of list of 4 points [[ [x1,y1], ... ], ...]
                  texts: list of strings
                  confidences: list of floats

    Returns:
        segment_texts: list of tuples [(concatenated_string, mask_index), ...]
                       ordered by string length descending
    """
    ocr_boxes, ocr_texts, ocr_confs = ocr_data
    segment_texts = []

    for mask_idx, mask in enumerate(masks):
        # Get bounding box of mask for quick rejection
        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue
        
        x1, y1, x2, y2 = bbox
        mask_texts = []
        
        for box, text, conf in zip(ocr_boxes, ocr_texts, ocr_confs):
            if conf <= 0:
                continue
            
            # Convert OCR box to bounding rect
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            ox1, oy1, ox2, oy2 = min(xs), min(ys), max(xs), max(ys)
            
            # Quick rejection test - check if OCR box overlaps with mask bbox
            if ox2 < x1 or ox1 > x2 or oy2 < y1 or oy1 > y2:
                continue
            
            # Get center point of OCR box
            center_x = int((ox1 + ox2) / 2)
            center_y = int((oy1 + oy2) / 2)
            
            # Check if center point is inside the mask
            H, W = mask.shape
            if 0 <= center_y < H and 0 <= center_x < W:
                if mask[center_y, center_x] > 0:
                    mask_texts.append(text.strip())
        
        if mask_texts:
            combined_text = " ".join(mask_texts)
            segment_texts.append((combined_text, mask_idx))
    
    # Sort by string length descending
    segment_texts.sort(key=lambda x: len(x[0]), reverse=True)
    return segment_texts


def ocr_text_prompt(predictions):
    """
    Create a prompt for LLM based on OCR predictions.

    Args:
        predictions (List[Tuple[str, int]]): List of (text, mask_index) tuples.

    Returns:
        str: Formatted prompt for LLM.
    """
    prompt = ""
    for i, (text, mask_idx) in enumerate(predictions):
        prompt += f" | Spine {i}: {text} | "
    return prompt


# ============================================================================
# Visualization functions (moved from image_processing)
# ============================================================================

def create_mean_value_image(image, masks):
    """
    Create an image where each segment is filled with its mean color.
    
    Parameters:
    -----------
    image : ndarray
        Original image (H, W, 3)
    masks : list of ndarray
        List of binary masks
    
    Returns:
    --------
    ndarray : Image with each segment filled with mean color
    """
    result = np.zeros_like(image)
    
    for mask in masks:
        if mask.sum() > 0:
            # Calculate mean color for this segment
            segment_pixels = image[mask > 0]
            mean_color = segment_pixels.mean(axis=0).astype(np.uint8)
            
            # Fill segment with mean color
            result[mask > 0] = mean_color
    
    return result


def visualize_selected_segment(image, masks, selected_mask_idx, 
                               highlight_color=(255, 165, 0), thickness=4, 
                               dash_length=5):
    """
    Create mean-valued segmentation image and highlight selected segment.

    Args:
        image: Original image (H,W,3)
        masks: list of binary masks
        selected_mask_idx: index of mask to highlight
        highlight_color: RGB color for highlight
        thickness: line thickness
        dash_length: length of dashes in dashed line

    Returns:
        vis_img: mean-valued image with dashed box around selected segment
    """
    # Create mean value image
    mean_img = create_mean_value_image(image, masks)
    
    # Get selected mask and its bounding box
    if selected_mask_idx >= len(masks):
        return mean_img
    
    selected_mask = masks[selected_mask_idx]
    bbox = mask_to_bbox(selected_mask)
    
    if bbox is None:
        return mean_img
    
    x1, y1, x2, y2 = bbox
    
    # Draw dashed box around selected segment
    for i in range(x1, x2, dash_length * 2):
        cv2.line(mean_img, (i, y1), (min(i + dash_length, x2), y1), 
                highlight_color, thickness)
        cv2.line(mean_img, (i, y2), (min(i + dash_length, x2), y2), 
                highlight_color, thickness)
    
    for i in range(y1, y2, dash_length * 2):
        cv2.line(mean_img, (x1, i), (x1, min(i + dash_length, y2)), 
                highlight_color, thickness)
        cv2.line(mean_img, (x2, i), (x2, min(i + dash_length, y2)), 
                highlight_color, thickness)
    
    return mean_img


