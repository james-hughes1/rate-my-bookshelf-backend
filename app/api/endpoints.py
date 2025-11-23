import time
import json
import cv2
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from ..services.image_processing import segment_rect, segment_hough, read_image, create_mean_value_image, visualize_selected_segment
from ..services.ocr import (ocr_from_array, ocr_text_prompt, 
                            assign_text_to_segments, mask_to_bbox)
from ..services.llm_client import (get_books_from_ocr, format_books_for_prompt, 
                                   analyse_bookshelf, analyse_library)

router = APIRouter()

@router.get("/ping")
async def ping():
    """
    Simple endpoint to test backend connectivity and CORS.
    """
    return {"status": "ok", "message": "Backend is reachable!"}


@router.post("/mybookshelf")
async def upload_bookshelf(file: UploadFile = File(...)):
    """
    Upload an image of a bookshelf, segment it, run OCR, prompt LLM.
    """
    # Save the image
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    img = read_image(image_path, max_dim=1024)

    # OCR
    print("Running OCR...")
    boxes, text, confidences = ocr_from_array(img)

    # Segment image (choose method - rect or hough)
    print("Segmenting image...")
    masks = segment_rect(
        img,
        min_size_factor=0.05,
        verbose=False
    )
    # Alternative: masks = segment_hough(img, max_depth=8, verbose=False)

    # Group text by segments
    print("Assigning text to segments...")
    segment_texts = assign_text_to_segments(
        img,
        masks,
        [boxes, text, confidences],
    )

    # Format text
    print("Formatting segmented text...")
    segment_texts_prompt = ocr_text_prompt(segment_texts)

    print(segment_texts_prompt)
    
    # Analyse the bookshelf
    print("Asking AI to analyse...")
    analysis = analyse_bookshelf(segment_texts_prompt, mode='analysis')
    age = analysis.age
    intensity = analysis.intensity
    mood = analysis.mood
    popularity = analysis.popularity
    focus = analysis.focus
    realism = analysis.realism
    word_one = analysis.word_one
    word_two = analysis.word_two
    word_three = analysis.word_three
    recommended_book = analysis.recommended_book
    explanation = analysis.explanation

    return JSONResponse(
        {
            "recommendation": {
                "recommended_book": recommended_book,
                "explanation": explanation
            },
            "three_words": {
                "word_one": word_one,
                "word_two": word_two,
                "word_three": word_three
            },
            "scores": {
                "age": age,
                "intensity": intensity,
                "mood": mood,
                "popularity": popularity,
                "focus": focus,
                "realism": realism
            }
        }
    )


@router.post("/library")
async def upload_library(
    file: UploadFile = File(...),
    description: str = Form(...)
):
    """
    Upload an image of a library shelf, segment it, run OCR, prompt LLM.
    """
    # Save the image
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    img = read_image(image_path, max_dim=1024)

    # OCR
    print("Running OCR...")
    boxes, text, confidences = ocr_from_array(img)

    # Segment image
    print("Segmenting image...")
    masks = segment_hough(
        img,
        verbose=True
    )

    # Group text by segments
    print("Assigning text to segments...")
    segment_texts = assign_text_to_segments(
        img,
        masks,
        [boxes, text, confidences],
    )

    # Format text
    print("Formatting segmented text...")
    segment_texts_prompt = ocr_text_prompt(segment_texts)
    print(segment_texts_prompt)

    # Ask AI for a recommendation
    print("Asking AI to analyse...")
    library_analysis = analyse_library(segment_texts_prompt, description)
    recommended_idx = library_analysis.recommended_idx
    
    # Get the mask index from segment_texts
    _, chosen_segment = segment_texts[recommended_idx]
    
    recommended_book = library_analysis.recommended_book
    explanation = library_analysis.explanation
    print(f"Recommended: {recommended_book}")
    print(f"Explanation: {explanation}")

    return JSONResponse(
        {
            "recommended_book": recommended_book,
            "explanation": explanation,
            "chosen_segment": int(chosen_segment),
            "num_segments": len(masks)
        }
    )


@router.post("/highlight")
async def highlight_segment(
    file: UploadFile = File(...),
    mask_idx: int = Form(...),
):
    """
    Accept an image + a selected mask index, return highlighted image.
    """
    print(f"Highlighting mask index: {mask_idx}")

    # Save uploaded image
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    img = read_image(image_path, max_dim=1024)

    # Recompute masks for correctness
    print("Re-segmenting image...")
    masks = segment_hough(
        img,
        verbose=True
    )

    # Create visualization with highlighted segment
    img_vis = visualize_selected_segment(
        img, 
        masks, 
        mask_idx,
        highlight_color=(255, 165, 0),  # Orange
        thickness=4,
        dash_length=5
    )

    # Convert RGB to BGR for OpenCV encoding
    img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # Encode as PNG
    success, encoded_image = cv2.imencode('.png', img_bgr)
    if not success:
        raise RuntimeError("Failed to encode image")

    return Response(content=encoded_image.tobytes(), media_type="image/png")