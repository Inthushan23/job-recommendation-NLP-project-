# Load environment variables for model and API URL
import os
from dotenv import load_dotenv

load_dotenv()

ENCODER_MODEL = os.getenv("MODEL", "usc-isi/sbert-roberta-large-anli-mnli-snli")
URL = os.getenv("API_URL", "http://recommender:8000")
