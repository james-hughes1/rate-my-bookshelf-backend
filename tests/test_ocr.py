import numpy as np

from app.services import ocr


def test_ocr_from_array_returns_structure():
    # We create a small white image - RapidOCR should return something
    # Since we don't actually test OCR, just check the function runs
    img = np.ones((10, 10, 3), dtype=np.uint8) * 255
    result = ocr.ocr_from_array(img)

    # Should return either "No text detected" or tuple
    assert isinstance(result, (str, tuple))
    if isinstance(result, tuple):
        boxes, texts, confs = result
        assert isinstance(boxes, list)
        assert isinstance(texts, list)
        assert isinstance(confs, list)


def test_assign_text_to_segments_basic():
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    masks = [np.ones((5, 5), dtype=np.uint8)]  # single mask

    # OCR-like data: [bbox, text, confidence]
    ocr_boxes = [[[0, 0], [4, 0], [4, 4], [0, 4]], [[5, 5], [6, 5], [6, 6], [5, 6]]]
    ocr_texts = ["Inside", "Outside"]
    ocr_confs = [0.9, 0.9]

    predictions = ocr.assign_text_to_segments(
        img, masks, (ocr_boxes, ocr_texts, ocr_confs)
    )

    # Only "Inside" should be assigned
    assert predictions == [("Inside", 0)]


def test_assign_text_to_segments_empty_mask():
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    masks = [np.zeros((5, 5), dtype=np.uint8)]  # empty mask

    predictions = ocr.assign_text_to_segments(img, masks, ([], [], []))
    assert predictions == []


def test_ocr_text_prompt_basic():
    predictions = [("Hello", 0), ("World", 1)]
    prompt = ocr.ocr_text_prompt(predictions)
    assert "Spine 0: Hello" in prompt
    assert "Spine 1: World" in prompt


def test_ocr_text_prompt_empty():
    prompt = ocr.ocr_text_prompt([])
    assert prompt == ""
