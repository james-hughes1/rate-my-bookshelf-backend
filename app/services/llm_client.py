import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

PROMPT_BOOKS = """
    You are an assistant that reads messy OCR text from books on a bookshelf.
    The OCR output for each shelf position is provided below. Each line starts with 'Scan i:'.

    Your task:
    1. For each scan, identify the books.
    2. Output a JSON array containing the each book you find, and include the index, title, author, and confidence score between 0 and 1.
    3. If a scan is empty or cannot be identified, you may skip it or leave title/author blank.
    4. There may be multiple books per scan.

    Here is the OCR data: 
"""

PROMPT_THREE_WORDS = """
    You are an insightful and creative literary assistant.
    You have already analyzed the list of books on this person's bookshelf, including each title, author, and confidence score (which reflects how certain the identification is).
    
    Your goal now is to distill the *essence* of this bookshelf into exactly three evocative, meaningful words.
    These words should feel personal—something that captures the reader's unique taste, mood, or identity through their books.
    
    Think like a poet or a curator, not a statistician.
    Avoid generic terms like "fiction" or "reading" unless they truly fit the collection.
    You may choose abstract or emotional words if they better reflect the spirit of the bookshelf.
    
    Output your result as valid JSON with the fields: word_one, word_two, and word_three.
    
    Here are the books you identified:
"""

PROMPT_RECOMMEND_BOOK = """
    You are an assistant that reads messy OCR text from books on a bookshelf.
    You have already helped me to identify the books on the shelf.
    Now, given your previous output, please provide a new book, not on the shelf, that the owner of this bookshelf might like to read.
    Output a JSON object and include the recommended book and explanation as two separate fields.
    Address your explanation to the owner of the bookshelf, using the second-person perspective.

    Here are the books you identified:
"""

PROMPT_BOOKSHELF_SCORES = """
    You are an assistant that reads messy OCR text from books on a bookshelf.
    You have already helped me to identify the books on the shelf.
    Now, given your previous output, please score the book collection on the following scales from -1.0 to 1.0:
    1. age. Classic (-1.0) to Modern (1.0)
    2. intensity. Beach-ready (-1.0) to Intense (1.0)
    3. mood. Dystopian (-1.0) to Light-hearted (1.0)
    4. popularity. Esoteric (-1.0) to Well-known (1.0)
    5. focus. Plot-driven (-1.0) to Character-driven (1.0)
    6. realism. Down-to-earth (-1.0) to Imaginary (1.0)
    Output a JSON object with the six scores as fields.

    Here are the books you identified:
"""

PROMPT_ANALYSE_SHELF = """
    You are an insightful and creative literary assistant.
    Below you are provided with an OCR scan of somebody's bookshelf - it is up to you to identify potential titles and authors amongst noisy text found.
    
    Your goal is now to analyse this person's bookshelf and help them understand their reading tastes and preferences, and provide a personalized recommendation.

    Tasks:
    1. Score the collection as a float between -1.0 to 1.0 on fields:
    age (Old -1.0 → Modern +1.0), intensity (Beach Read -1.0 → Intense Study +1.0), mood (Dystopian -1.0 → Inspiring +1.0), 
    popularity (Esoteric -1.0 → Well-known +1.0), focus (Plot -1.0 → Character +1.0), realism (Down-to-earth -1.0 → Imaginary +1.0).
    2. Distill the essence of the bookshelf into exactly three meaningful words that reflect the owner's taste and identity. Avoid generic words; try to make them unique to the person and their collection.
    3. Recommend a new book not on the shelf, and craft a short explanation addressed directly to the owner written in the second-person. Respond with fields recommended_book and explanation separately.
      
    Output your result as valid JSON.
    
    Here are the books you identified:
"""

class BookInfo(BaseModel):
    idx: int
    title: str
    author: str
    confidence: float

class ThreeWords(BaseModel):
    word_one: str
    word_two: str
    word_three: str

class Recommendation(BaseModel):
    recommended_book: str
    explanation: str

class BookshelfScores(BaseModel):
    age: float
    intensity: float
    mood: float
    popularity: float
    focus: float
    realism: float

class BookshelfAnalysis(BaseModel):
    age: float
    intensity: float
    mood: float
    popularity: float
    focus: float
    realism: float
    word_one: str
    word_two: str
    word_three: str
    recommended_book: str
    explanation: str

def get_books_from_ocr(ocr_data):
    """
    Sends OCR data to Gemini API and asks it to return a structured list of books.
    
    Args:
        ocr_data: string containing all OCR scan text (can be multiline)
        
    Returns:
        List[Dict] of book information, e.g.
        [
            {"idx": 0, "title": "The Bell Jar", "author": "Sylvia Plath", "confidence": 0.7},
            ...
        ]
    """
    # Craft the prompt
    

    try:
        # Call Gemini API
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=PROMPT_BOOKS+ocr_data,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[BookInfo],
            }
        )

        books: list[BookInfo] = response.parsed
        
        return books

    except Exception as e:
        print("Error calling Gemini API:", e)
        # fallback: return empty list
        return []


def format_books_for_prompt(books: list[BookInfo], confidence_threshold: float) -> str:
    """
    Convert a list of BookInfo objects into a clean, readable string
    suitable for use inside an LLM prompt.

    Example output:

    Bookshelf:
    - [0] The Bell Jar — Sylvia Plath (confidence 0.70)
    - [1] Dune — Frank Herbert (confidence 0.85)
    - [2] 1984 — George Orwell (confidence 0.90)
    """
    if not books:
        return "No books were identified."

    lines = ["Bookshelf: "]
    for book in books:
        if book.confidence > confidence_threshold:
            title = book.title.strip() or "Unknown Title"
            author = book.author.strip() or "Unknown Author"
            conf = f"{book.confidence:.2f}" if book.confidence is not None else "N/A"
            lines.append(f"[{book.idx}] {title} — {author} (confidence {conf})")

    return " // ".join(lines)


def analyse_bookshelf(books_string, mode):
    """
    Analyse the bookshelf based on identified books.
    
    Args:
        books: List of BookInfo objects
        mode: str, one of 'three_words', 'recommendation', 'scores'
        
    Returns:
        Depending on mode:
        - 'three_words': str of three words
        - 'recommendation': Recommendation object
        - 'scores': BookshelfScores object
    """
    if mode == 'three_words':
        prompt = PROMPT_THREE_WORDS + books_string
        response_schema = ThreeWords
    elif mode == 'recommendation':
        prompt = PROMPT_RECOMMEND_BOOK + books_string
        response_schema = Recommendation
    elif mode == 'scores':
        prompt = PROMPT_BOOKSHELF_SCORES + books_string
        response_schema = BookshelfScores
    elif mode == 'analysis':
        prompt = PROMPT_ANALYSE_SHELF + books_string
        response_schema = BookshelfAnalysis
    else:
        raise ValueError("Invalid mode for analyse_bookshelf")

    try:
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema,
            }
        )

        return response.parsed

    except Exception as e:
        print("Error calling Gemini API:", e)
        return None