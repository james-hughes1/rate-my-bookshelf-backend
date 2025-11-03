from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from ..services.image_processing import SimpleSegmenter
from ..services.ocr import easyocr_predict, ocr_text_prompt
from ..services.llm_client import get_books_from_ocr, format_books_for_prompt, analyse_bookshelf

router = APIRouter()

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
    predictions = easyocr_predict([cp[0] for cp in crops])

    # Convert predictions to prompt
    prompt = ocr_text_prompt(predictions)

    # # Send prompt to LLM
    # books = get_books_from_ocr(prompt)

    # # Format books for prompt
    # formatted_books = format_books_for_prompt(books)

    # # Give a recommendation
    # recommendation = analyse_bookshelf(formatted_books, mode='recommendation')
    # book_recommendation = recommendation.recommended_book
    # explanation = recommendation.explanation

    # # Describe in 3 words
    # three_words = analyse_bookshelf(formatted_books, mode='three_words')
    # words = ", ".join([three_words.word_one, three_words.word_two, three_words.word_three])

    # # Score the bookshelf
    # scores = analyse_bookshelf(formatted_books, mode='scores')
    # age = scores.age
    # intensity = scores.intensity
    # mood = scores.mood
    # popularity = scores.popularity
    # focus = scores.focus
    # realism = scores.realism

    return JSONResponse(
        {
            "prompt": prompt,
        }
    )