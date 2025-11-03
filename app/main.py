from fastapi import FastAPI
from app.api import endpoints

app = FastAPI(title="Bookshelf OCR & Recommendation API")

app.include_router(endpoints.router)
