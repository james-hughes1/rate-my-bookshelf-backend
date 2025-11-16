import time
import json
import cv2
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from ..services.image_processing import SimpleSegmenter, read_image, mean_value_spine_image, visualize_selected_segments
from ..services.ocr import ocr_from_array, ocr_text_prompt, assign_text_to_segments
from ..services.llm_client import get_books_from_ocr, format_books_for_prompt, analyse_bookshelf, analyse_library

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

    # Initialize segmenter and segment image
    print("Segmenting image...")
    segmenter = SimpleSegmenter(image_path, min_size_factor=0.05)
    segments = segmenter.segment()

    # Group text by segments
    print("Assigning text to segments...")
    segment_texts = assign_text_to_segments(
        img,
        segments,
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

    # Initialize segmenter and segment image
    print("Segmenting image...")
    segmenter = SimpleSegmenter(image_path, min_size_factor=0.05)
    segments = segmenter.segment()

    # Group text by segments
    print("Assigning text to segments...")
    segment_texts = assign_text_to_segments(
        img,
        segments,
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
    chosen_segment = segment_texts[recommended_idx][1]
    recommended_book = library_analysis.recommended_book
    explanation = library_analysis.explanation
    print(f"Recommended: {recommended_book}")
    print(f"Explanation: {explanation}")

    return JSONResponse(
        {
            "recommended_book": recommended_book,
            "explanation": explanation,
            "chosen_segment": chosen_segment,
            "segments": segments
        }
    )

@router.post("/highlight")
async def highlight_segment(
    file: UploadFile = File(...),
    segment: str = Form(...),   # segment arrives as a JSON string "[x1, y1, x2, y2]"
):
    """
    Accept an image + a selected segment, return highlighted image.
    """
    # Parse segment JSON
    chosen_segment = json.loads(segment)  # -> [x1, y1, x2, y2]
    print(type(file))
    print(chosen_segment)

    # Save uploaded image
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    img = read_image(image_path, max_dim=1024)

    # Recompute spines for correctness
    segmenter = SimpleSegmenter(image_path, min_size_factor=0.05)
    segments = segmenter.segment()

    # Create flat spine image
    img_spines = mean_value_spine_image(img, segments)

    # Highlight segment
    img_vis = visualize_selected_segments(img_spines, [("", chosen_segment)])

    # Ensure colours correct
    img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

    # Encode as PNG
    success, encoded_image = cv2.imencode('.png', img_rgb)
    if not success:
        raise RuntimeError("Failed to encode image")

    return Response(content=encoded_image.tobytes(), media_type="image/png")