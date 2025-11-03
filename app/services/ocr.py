import re
import easyocr
import numpy as np

def easyocr_predict(crops, word_confidence_threshold=0.2):
    reader = easyocr.Reader(['en'], gpu=True)
    predictions = []
    for cp in crops:
        if cp.dtype != np.uint8:
            cp = (cp * 255).astype(np.uint8)

        # Run model on all 4 rotations
        result = [reader.readtext(np.rot90(cp, k=idx)) for idx in range(2)]

        img_predictions = []

        # Extract titles
        for rotate_result in result:
            total_chars = 0
            total_conf = 0
            full_str = ""
            for text_result in rotate_result:
                conf = text_result[2]
                text = text_result[1]
                if conf > word_confidence_threshold:
                    total_chars += len(text)
                    total_conf += conf * len(text)
                    full_str += (text.strip() + " ")
            if (total_chars > 0) and re.search(r'[A-Za-z]', full_str):
                img_predictions.append({"string": full_str, "confidence": total_conf/total_chars})

        predictions.append(img_predictions)
    return predictions


def ocr_text_prompt(predictions):
    """
    Create a prompt for LLM based on OCR predictions.

    Args:
        predictions (List[List[dict]]): List of OCR results for each image crop.

    Returns:
        str: Formatted prompt for LLM.
    """
    prompt = "The following are the OCR results for the image crops:\n"
    for i, img_preds in enumerate(predictions):
        prompt += f" | Image Crop {i+1}:" + " ".join([pred['string'] for pred in img_preds]) + " | "
    return prompt