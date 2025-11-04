import time
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..services.image_processing import SimpleSegmenter
from ..services.ocr import easyocr_predict, ocr_text_prompt
from ..services.llm_client import get_books_from_ocr, format_books_for_prompt, analyse_bookshelf

router = APIRouter()

@router.get("/ping")
async def ping():
    """
    Simple endpoint to test backend connectivity and CORS.
    """
    return {"status": "ok", "message": "Backend is reachable!"}

@router.post("/process")
async def upload_bookshelf(file: UploadFile = File(...)):
    """
    Upload an image of a bookshelf, segment it, run OCR, prompt LLM.
    """
    # Save the image
    image_path = f"/tmp/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(await file.read())

    # Initialize segmenter and segment image
    segmenter = SimpleSegmenter(image_path)
    segments = segmenter.segment()

    # Return cropped images
    crops = segmenter.get_crops(segments)

    # Run OCR on crops
    predictions = easyocr_predict([cp[0] for cp in crops][:40])

    # Convert predictions to prompt
    prompt = ocr_text_prompt(predictions)

    # Send prompt to LLM
    books = get_books_from_ocr(prompt)
    print("BOOKS DISCOVERED: \n", books)
    time.sleep(5)

    # Format books for prompt
    formatted_books = format_books_for_prompt(books, confidence_threshold=0.5)
    print("PROMPT: \n", formatted_books)

    # Analyse the bookshelf
    analysis = analyse_bookshelf(formatted_books, mode='analysis')
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
            "books": [b.dict() for b in books],
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