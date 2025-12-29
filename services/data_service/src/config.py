import os
from pathlib import Path 
from dotenv import load_dotenv

load_dotenv()


BUCKET_NAME = os.getenv("BUCKET_NAME", "s3-g3mg01")
DATA_KEY = os.getenv("DATA_KEY", "data/my_data.xlsx") 

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# Path to data
DATA_DIR = PROJECT_ROOT / "data"
EXCEL_PATH = DATA_DIR / "job_data.xlsx"

TASTES_EMBED_PATH = DATA_DIR / "tastes_embeddings.pkl"