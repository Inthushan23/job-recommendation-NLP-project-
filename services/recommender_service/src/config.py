import os
from dotenv import load_dotenv

load_dotenv()


ENCODER_MODEL = os.getenv("MODEL", "usc-isi/sbert-roberta-large-anli-mnli-snli")
URL = os.getenv("API_URL", "http://localhost:8000")