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


def assign_text_to_segments(img, spines, ocr_data):
    """
    Assign OCR text to spine segments.

    Args:
        img: Original image (H,W,3)
        spines: list of 4-tuples [(x1,y1,x2,y2), ...]
        ocr_data: [boxes, texts, confidences]
                  boxes: list of list of 4 points [[ [x1,y1], ... ], ...]
                  texts: list of strings
                  confidences: list of floats

    Returns:
        segment_texts: list of tuples [(concatenated_string, [x1,y1,x2,y2]), ...]
                       ordered by string length descending
    """
    ocr_boxes, ocr_texts, ocr_confs = ocr_data
    segment_texts = []

    for x1, y1, x2, y2 in spines:
        spine_texts = []
        for box, text, conf in zip(ocr_boxes, ocr_texts, ocr_confs):
            if conf <= 0:
                continue
            # Convert OCR box to bounding rect
            xs = [pt[0] for pt in box]
            ys = [pt[1] for pt in box]
            ox1, oy1, ox2, oy2 = min(xs), min(ys), max(xs), max(ys)

            # Check how much of OCR box is inside the spine
            inter_x1 = max(x1, ox1)
            inter_y1 = max(y1, oy1)
            inter_x2 = min(x2, ox2)
            inter_y2 = min(y2, oy2)
            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            box_area = (ox2 - ox1) * (oy2 - oy1)

            if box_area > 0 and inter_area / box_area >= 0.5:
                # Consider this text part of the spine
                spine_texts.append(text.strip())

        if spine_texts:
            combined_text = " ".join(spine_texts)
            segment_texts.append((combined_text, [x1, y1, x2, y2]))

    # Sort by string length descending
    segment_texts.sort(key=lambda x: len(x[0]), reverse=True)
    return segment_texts


def ocr_text_prompt(predictions):
    """
    Create a prompt for LLM based on OCR predictions.

    Args:
        predictions (List[List[dict]]): List of OCR results for each image crop.

    Returns:
        str: Formatted prompt for LLM.
    """
    prompt = ""
    for i, prediction_segment in enumerate(predictions):
        prompt += f" | Spine {i}: " + prediction_segment[0] + " | "
    return prompt