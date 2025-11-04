import easyocr
import os

CACHE_DIR = os.path.expanduser("~/.cache/easyocr")
os.makedirs(CACHE_DIR, exist_ok=True)

def init_easyocr():
    reader = easyocr.Reader(['en'], model_storage_directory=CACHE_DIR)
    return reader