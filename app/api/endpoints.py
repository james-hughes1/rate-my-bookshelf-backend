import time
import json
import cv2
import io
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from ..services.image_processing import (segment_hough, read_image,
                                         SegmentationResult, create_segmentation_gif)
from ..services.ocr import (ocr_from_array, ocr_text_prompt, 
                            assign_text_to_segments, mask_to_bbox,
                            create_mean_value_image, visualize_selected_segment)
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

    # Segment image (returns SegmentationResult with tree data)
    print("Segmenting image...")
    seg_result = segment_hough(img, return_tree=True, verbose=True)
    
    # Save segmentation result for later GIF generation
    seg_result.save(f"/tmp/{file.filename}_segmentation.pkl")

    # Group text by segments
    print("Assigning text to segments...")
    segment_texts = assign_text_to_segments(
        img,
        seg_result.masks,  # Use masks from result
        [boxes, text, confidences]
    )

    # Format text
    print("Formatting segmented text...")
    segment_texts_prompt = ocr_text_prompt(segment_texts)
    print(segment_texts_prompt)
    
    # Analyse the bookshelf
    print("Asking AI to analyse...")
    analysis = analyse_bookshelf(segment_texts_prompt, mode='analysis')
    
    return JSONResponse(
        {
            "recommendation": {
                "recommended_book": analysis.recommended_book,
                "explanation": analysis.explanation
            },
            "three_words": {
                "word_one": analysis.word_one,
                "word_two": analysis.word_two,
                "word_three": analysis.word_three
            },
            "scores": {
                "age": analysis.age,
                "intensity": analysis.intensity,
                "mood": analysis.mood,
                "popularity": analysis.popularity,
                "focus": analysis.focus,
                "realism": analysis.realism
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

    # Segment image (returns SegmentationResult with tree data)
    print("Segmenting image...")
    seg_result = segment_hough(img, return_tree=True, verbose=True)
    
    # Save segmentation result for later GIF generation
    seg_result.save(f"/tmp/{file.filename}_segmentation.pkl")

    # Group text by segments
    print("Assigning text to segments...")
    segment_texts = assign_text_to_segments(
        img,
        seg_result.masks,
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
    _, chosen_mask_idx = segment_texts[recommended_idx]
    
    # Convert mask to bbox for JSON serialization
    chosen_bbox = mask_to_bbox(seg_result.masks[chosen_mask_idx])
    
    recommended_book = library_analysis.recommended_book
    explanation = library_analysis.explanation
    print(f"Recommended: {recommended_book}")
    print(f"Explanation: {explanation}")

    return JSONResponse(
        {
            "recommended_book": recommended_book,
            "explanation": explanation,
            "chosen_mask_idx": int(chosen_mask_idx)
        }
    )


@router.post("/highlight")
async def highlight_segment(
    file: UploadFile = File(...),
    mask_idx: int = Form(...),
):
    """
    Accept an image + a selected mask index, return highlighted image as PNG.
    """
    print(f"Highlighting mask index: {mask_idx}")

    # Load the image
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    img = read_image(image_path, max_dim=1024)

    # Load saved segmentation result (no re-segmentation!)
    try:
        seg_result = SegmentationResult.load(f"/tmp/{file.filename}_segmentation.pkl")
    except FileNotFoundError:
        # Fallback: re-segment if pkl not found
        print("Warning: Segmentation result not found, re-segmenting...")
        seg_result = segment_hough(img, return_tree=True, verbose=True)

    # Create visualization with highlighted segment
    img_vis = visualize_selected_segment(
        img, 
        seg_result.masks, 
        mask_idx,
        highlight_color=(255, 165, 0),
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


@router.post("/gif")
async def create_gif(
    file: UploadFile = File(...),
    mask_idx: int = Form(None),  # Optional: highlight specific segment
):
    """
    Create an animated GIF showing the segmentation process.
    Uses saved segmentation result - no re-segmentation needed!
    """
    print(f"Creating GIF, highlight mask: {mask_idx}")

    # Load the image
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())
    
    img = read_image(image_path, max_dim=1024)

    # Load saved segmentation result
    try:
        seg_result = SegmentationResult.load(f"/tmp/{file.filename}_segmentation.pkl")
    except FileNotFoundError:
        return JSONResponse(
            {"error": "Segmentation result not found. Please upload and segment first."},
            status_code=404
        )

    # Create GIF in memory
    gif_buffer = io.BytesIO()
    create_segmentation_gif(
        img,
        seg_result,
        output_path=None,
        io_buffer=gif_buffer,
        duration=300,
        final_duration=2000,
        boundary_color=(255, 165, 0),
        thickness=2,
        highlight_idx=mask_idx
    )
    
    gif_buffer.seek(0)
    
    return Response(content=gif_buffer.getvalue(), media_type="image/gif")