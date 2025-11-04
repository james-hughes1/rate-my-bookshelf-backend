import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import endpoints

app = FastAPI(title="Bookshelf OCR & Recommendation API")

frontend_origins = os.environ.get("FRONTEND_ORIGINS", "http://localhost:5173")

# Allow your frontend origins
origins = [origin.strip() for origin in frontend_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] for testing only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(endpoints.router)
